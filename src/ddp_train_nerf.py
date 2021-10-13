import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim
import torch.distributed
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing

from collections import OrderedDict
from .ddp_model import NerfNetWithAutoExpo, NerfNetBoxWithAutoExpo, \
    NerfNetBoxOnlyWithAutoExpo, DepthOracle, DepthOracleBoxOnly, DepthOracleWithBox, NerfNetMoreBoxWithAutoExpo

from .nerf_network import WrapperModule

import time
from .data_loader_split import load_data_split
import numpy as np
from tensorboardX import SummaryWriter
from .utils import img2mse, mse2psnr, entropy_loss, crossEntropy, dep_l1l2loss, img_HWC2CHW, colorize, colorize_np,to8b, normalize_torch,TINY_NUMBER

from .helpers import calculate_metrics, log_plot_conf_mat, visualize_depth_label, loss_deviation, get_box_transmittance_weight, get_box_weight
import logging
import json

from .helpers import plot_ray_batch
time_program = False

logger = logging.getLogger(__package__)


def setup_logger():
    # create logger
    logger = logging.getLogger(__package__)
    # logger.setLevel(logging.DEBUG)
    logger.setLevel(logging.INFO)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')


    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)


def intersect_sphere(ray_o, ray_d):
    '''
    ray_o, ray_d: [..., 3]
    compute the depth of the intersection point between this ray and unit sphere
    '''
    # note: d1 becomes negative if this mid point is behind camera
    d1 = -torch.sum(ray_d * ray_o, dim=-1) / torch.sum(ray_d * ray_d, dim=-1)
    p = ray_o + d1.unsqueeze(-1) * ray_d
    # consider the case where the ray does not intersect the sphere
    ray_d_cos = 1. / torch.norm(ray_d, dim=-1)
    d2 = torch.sqrt(1. - torch.sum(p * p, dim=-1)) * ray_d_cos

    return d1 + d2


def perturb_samples(z_vals):
    # get intervals between samples
    mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
    lower = torch.cat([z_vals[..., 0:1], mids], dim=-1)
    # uniform samples in those intervals
    t_rand = torch.rand_like(z_vals)
    z_vals = lower + (upper - lower) * t_rand  # [N_rays, N_samples]

    return z_vals


def sample_pdf(bins, weights, N_samples, det=False):
    '''
    :param bins: tensor of shape [..., M+1], M is the number of bins
    :param weights: tensor of shape [..., M]
    :param N_samples: number of samples along each ray
    :param det: if True, will perform deterministic sampling
    :return: [..., N_samples]
    '''
    # Get pdf
    weights = weights + TINY_NUMBER      # prevent nans
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)    # [..., M]
    cdf = torch.cumsum(pdf, dim=-1)                             # [..., M]
    cdf = torch.cat([torch.zeros_like(cdf[..., 0:1]), cdf], dim=-1)     # [..., M+1]

    # Take uniform samples
    dots_sh = list(weights.shape[:-1])
    M = weights.shape[-1]

    min_cdf = 0.00
    max_cdf = 1.00       # prevent outlier samples

    if det:
        u = torch.linspace(min_cdf, max_cdf, N_samples, device=bins.device)
        u = u.view([1]*len(dots_sh) + [N_samples]).expand(dots_sh + [N_samples,])   # [..., N_samples]
    else:
        sh = dots_sh + [N_samples]
        u = torch.rand(*sh, device=bins.device) * (max_cdf - min_cdf) + min_cdf        # [..., N_samples]

    # Invert CDF
    # [..., N_samples, 1] >= [..., 1, M] ----> [..., N_samples, M] ----> [..., N_samples,]
    above_inds = torch.sum(u.unsqueeze(-1) >= cdf[..., :M].unsqueeze(-2), dim=-1).long()

    # random sample inside each bin
    below_inds = torch.clamp(above_inds-1, min=0)
    inds_g = torch.stack((below_inds, above_inds), dim=-1)     # [..., N_samples, 2]

    cdf = cdf.unsqueeze(-2).expand(dots_sh + [N_samples, M+1])   # [..., N_samples, M+1]
    cdf_g = torch.gather(input=cdf, dim=-1, index=inds_g)       # [..., N_samples, 2]

    bins = bins.unsqueeze(-2).expand(dots_sh + [N_samples, M+1])    # [..., N_samples, M+1]
    bins_g = torch.gather(input=bins, dim=-1, index=inds_g)  # [..., N_samples, 2]

    # fix numeric issue
    denom = cdf_g[..., 1] - cdf_g[..., 0]      # [..., N_samples]
    denom = torch.where(denom<TINY_NUMBER, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom

    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0] + TINY_NUMBER)

    return samples


def render_single_image(models, ray_sampler, chunk_size, box_props=None,
                        train_box_only=False, have_box=False,
                        donerf_pretrain=False, box_number=10,
                        front_sample=128, back_sample=128, 
                        fg_bg_net=True, use_zval=True,loss_type='bce',rank=0, DEBUG=True):
    """
    Render an image using the NERF.
    :param models: Dictionary of networks used for the render.
    :param ray_sampler: A sampler for the image.
    :param chunk_size: How many rays per single pass.
    :param train_box_only: Only trains the foreground.
    :param have_box: If there exists a box in the rays dictionary.
    :param donerf_pretrain: ??
    :param front_sample: Foreground samples.
    :param back_sample: Background samples.
    :param fg_bg_net: ??
    :param use_zval: ??
    :return: Result dictionary from the last model in the chain of models.
    """



    rays = ray_sampler.get_all_classifier(front_sample,
                                          back_sample,
                                          pretrain=donerf_pretrain, rank=rank, 
                                          train_box_only=train_box_only, 
                                          box_number=box_number)

    chunks = (len(rays['ray_d']) + chunk_size - 1) // chunk_size
    rays_split = [OrderedDict() for _ in range(chunks)]

    for k, v in rays.items():
        if torch.is_tensor(v):
            split = torch.split(v, chunk_size)
            for i in range(chunks):
                rays_split[i][k] = split[i]

    rgbs = []
    depths = []

    likelis_fg = []
    likelis_bg = []


    if DEBUG:
        depths_fg = []
        depths_bg = []
        rgbs_bg = []
        rgbs_fg = []

        depths_lamda = []



    for k,_rays in enumerate(rays_split):

        chunk_ret = render_rays(models, _rays, train_box_only, have_box,
                                donerf_pretrain, front_sample, back_sample,
                                fg_bg_net, use_zval, loss_type, box_number=box_number, box_props=box_props)


        if donerf_pretrain:
            likelis_fg.append(chunk_ret['likeli_fg'])
            likelis_bg.append(chunk_ret['likeli_bg'])


        else:

            likelis_fg.append(chunk_ret['likeli_fg'].cpu())
            likelis_bg.append(chunk_ret['likeli_bg'].cpu())
            rgbs.append(chunk_ret['rgb'].cpu())
            depths.append(chunk_ret['depth_fgbg'].cpu())

            if DEBUG:
                rgbs_fg.append(chunk_ret['fg_rgb'])
                depths_fg.append(chunk_ret['scene_weights'])

                rgbs_bg.append(chunk_ret['bg_rgb'])
                depths_bg.append(chunk_ret['bg_depth'])
                depths_lamda.append(chunk_ret['box_weights'])


    if donerf_pretrain:
        likeli_fg = torch.cat(likelis_fg).view(ray_sampler.H * ray_sampler.W, -1).squeeze().cpu()
        likeli_bg = torch.cat(likelis_bg).view(ray_sampler.H * ray_sampler.W, -1).squeeze().cpu()

        rgb = None
        d = None

    else:
        rgb = torch.cat(rgbs).view(ray_sampler.H, ray_sampler.W, -1).squeeze()
        d = torch.cat(depths).view(ray_sampler.H, ray_sampler.W, -1).squeeze()
        likeli_fg = torch.cat(likelis_fg).view(ray_sampler.H * ray_sampler.W, -1).squeeze().cpu()
        likeli_bg = torch.cat(likelis_bg).view(ray_sampler.H * ray_sampler.W, -1).squeeze().cpu()


        # likeli_fg_sig = torch.sigmoid(likeli_fg).numpy()
        # likeli_bg_sig = torch.sigmoid(likeli_bg).numpy()
        if DEBUG:
            others = {}
            rgb_fg = torch.cat(rgbs_fg).view(ray_sampler.H, ray_sampler.W, -1).squeeze()
            d_fg = torch.cat(depths_fg).view(ray_sampler.H, ray_sampler.W, -1).squeeze()
            others['rgb_fg'] = rgb_fg
            others['d_fg'] = d_fg


            rgb_bg = torch.cat(rgbs_bg).view(ray_sampler.H, ray_sampler.W, -1).squeeze()
            d_bg = torch.cat(depths_bg).view(ray_sampler.H, ray_sampler.W, -1).squeeze()
            d_lam = torch.cat(depths_lamda).view(ray_sampler.H, ray_sampler.W, -1).squeeze()
            others['rgb_bg'] = rgb_bg
            others['d_bg'] = d_bg
            others['d_lam'] = d_lam



            return rgb, d, likeli_fg, likeli_bg, rays['cls_label'], others

    return rgb, d, likeli_fg, likeli_bg, rays['cls_label'], None

def eval_oracle(rays, net_oracle, fg_bg_net, use_zval,  front_sample, back_sample, have_box):


    if have_box:

        # ret = net_oracle(rays['ray_o'], rays['ray_d'],
        #                  rays['fg_pts_flat'], rays['bg_pts_flat'],
        #                  rays['fg_far_depth'], rays['box_loc'][:,:2].float())
        ret = net_oracle(rays['ray_o'], rays['ray_d'],
                         rays['fg_pts_flat'], rays['bg_pts_flat'],
                         rays['fg_far_depth'])

    else:
        ret = net_oracle(rays['ray_o'], rays['ray_d'],
                 rays['fg_pts_flat'], rays['bg_pts_flat'],
                 rays['fg_far_depth'])

    return ret

def get_depths(data, front_sample, back_sample, fg_z_vals_centre,
               bg_z_vals_centre, samples, box_weights=None, train_box_only=False,
               loss_type='ce', fg_weights=None, bg_weights=None, use_or=True):


    if fg_weights is None:
        fg_weights = data['likeli_fg'].clone() # Avoid inplace ops
        bg_weights = data['likeli_bg'].clone()
    #
    # fg_z_vals_centre_np = fg_z_vals_centre.cpu().numpy()
    # bg_z_vals_centre_np = bg_z_vals_centre.cpu().numpy()
    #
    # fg_weights_np = fg_weights.cpu().numpy()
    # bg_weights_np = bg_weights.cpu().numpy()


    fg_weights = torch.sigmoid(fg_weights)

    fg_weights = fg_weights.clone()


    if box_weights is not None:
        # fg_weights = fg_weights + normalize_torch(box_weights[:, 1:])

        # box_norm_test = normalize_torch(box_weights).cpu().numpy()
        fg_weights = fg_weights + box_weights

    fg_weights[fg_z_vals_centre < 0.0002] = float(0.0)

    fg_depth_mid = 0.5 * (fg_z_vals_centre[:, 1:] + fg_z_vals_centre[:, :-1])
    bg_depth_mid = 0.5 * (bg_z_vals_centre[:, 1:] + bg_z_vals_centre[:, :-1])

    fg_depth,_ = torch.sort(sample_pdf(bins=fg_depth_mid, weights=fg_weights[:, 1:front_sample-1],
                          N_samples=samples, det=True))  # [..., N_samples]


    fg_depth = fg_depth.clone()
    fg_depth[fg_depth<0.0002] = float(0.0002)

    # fg_depth_np = fg_depth.cpu().numpy()


    bg_weights = torch.fliplr(bg_weights)

    bg_weights = torch.sigmoid(bg_weights)[:, 1:back_sample-1]

    bg_depth,_ = torch.sort(sample_pdf(bins=bg_depth_mid, weights=bg_weights,
                          N_samples=samples, det=True))  # [..., N_samples]

    # bg_depth_np = bg_depth.cpu().numpy()
    # fg_depth_np = fg_depth.cpu().numpy()
    if use_or:
        return fg_depth, bg_depth
    return fg_depth_mid, bg_depth_mid


def render_rays(models, rays, train_box_only, have_box, donerf_pretrain,
                front_sample, back_sample, fg_bg_net, use_zval, loss_type, box_number, box_props=None):

    """Render a set of rays using specific config."""

    # forward and backward

    ray_o = rays['ray_o']
    ray_d = rays['ray_d']
    box_loc = rays['box_loc'] if have_box else None

    box_props = box_props if box_props is not None else torch.ones([box_number,9]).type_as(box_loc)

    fg_z_vals_centre = rays['fg_z_vals_centre']

    fg_far_depth = rays['fg_far_depth']
    net_oracle = models['net_oracle']

    bg_z_vals_centre = rays['bg_z_vals_centre']
    use_label = False


    ret_or = eval_oracle(rays, net_oracle, fg_bg_net, use_zval, front_sample, back_sample, have_box)

    if use_label:
        ret_or['likeli_fg'] = rays['cls_label'][:,:front_sample]
        ret_or['likeli_bg'] = rays['cls_label'][:,front_sample:]

    if donerf_pretrain:
        return ret_or
    else:

        net = models['net_0']
        samples = models['cascade_samples'][0]

        box_weights = None

        if box_loc is not None:
            # box_weights = get_box_weight(box_loc=box_loc,
            #                              box_size=1. / 30.,
            #                              fg_z_vals=fg_z_vals_centre,
            #                              ray_d=ray_d, ray_o=ray_o, box_number=box_number)



            box_weights = get_box_transmittance_weight(box_loc=box_loc,
                                         fg_z_vals=fg_z_vals_centre,  ray_d=ray_d,
                                         ray_o=ray_o,
                                         fg_depth=fg_far_depth, box_number=box_number, box_props=box_props)

            ## resample from box_nerf
            # fg_weights = fg_weights + normalize_torch(box_weights)
            #
            # box_ret = net(ray_o, ray_d, fg_far_depth, fg_z_vals_centre,
            #               bg_z_vals=None, box_loc=box_loc,
            #               query_box_only=True)
            #
            # box_weights = box_ret['fg_box_sig']
            # box_test = box_weights.cpu().numpy()

        fg_depth, bg_depth = get_depths(ret_or, front_sample, back_sample,
                                        fg_z_vals_centre, bg_z_vals_centre,
                                        samples, box_weights, train_box_only, loss_type,use_or=True
                                       )


        if not have_box:
            ret = net(ray_o, ray_d, fg_far_depth, fg_depth, bg_depth)

        else:
            ret = net(ray_o.float(), ray_d.float(), fg_far_depth.float(), fg_depth.float(), bg_depth.float(), box_loc.float(), box_props.float())


        # fg_depth, bg_depth = get_depths(ret_or, front_sample, back_sample,
        #                                 fg_z_vals_centre, bg_z_vals_centre,
        #                                 samples, box_weights, train_box_only, loss_type,
        #                                 ret['fg_weights'], ret['bg_weights'])
        #
        # ret = net(ray_o, ray_d, fg_far_depth, fg_depth, bg_depth)
        #

        if box_weights is None:
            ret['likeli_fg'] = ret_or['likeli_fg']
        else:
            ret['likeli_fg'] = torch.sigmoid(ret_or['likeli_fg']) + (box_weights)

        ret['likeli_bg'] = ret_or['likeli_bg']
        #
        # ret['likeli_fg'] = ret['fg_weights']
        # ret['likeli_bg'] = ret['bg_weights']

        return ret

def log_view_to_tb(writer, global_step, rgb_predict, depth_predict, gt_img, mask, gt_depth=None, train_box_only= False, have_box=False, box_seg_mask=None, prefix='', DEBUG=False, others=None):
    rgb_im = img_HWC2CHW(torch.from_numpy(gt_img))
    writer.add_image(prefix + 'rgb_gt', rgb_im, global_step)

    loss = img2mse(rgb_predict.cpu().reshape(-1), torch.from_numpy(gt_img).reshape(-1)).item()
    writer.add_scalar(prefix + 'rgb_loss', loss, global_step)
    writer.add_scalar( prefix + 'psnr', mse2psnr(loss), global_step)

    depth_clip = 100.
    if gt_depth is not None:
        gt_depth[gt_depth > depth_clip] = depth_clip  ##### THIS IS THE DEPTH OUTPUT, HxW, value is meters away from camera centre
        depth_im = img_HWC2CHW(colorize(gt_depth, cmap_name='jet', append_cbar=True, mask=mask, is_np=True))
        writer.add_image(prefix + 'depth_gt', depth_im, global_step)

    if box_seg_mask is not None:

        box_seg_mask_im = img_HWC2CHW(colorize(box_seg_mask, cmap_name='jet', append_cbar=True, mask=mask, is_np=True))
        writer.add_image(prefix + 'level_{}/boxsegmask_gt'.format(0), box_seg_mask_im, global_step)


    rgb_im = img_HWC2CHW(rgb_predict)
    rgb_im = torch.clamp(rgb_im, min=0., max=1.)  # just in case diffuse+specular>1
    writer.add_image(prefix + 'level_0/rgb', rgb_im, global_step)

    depth_predict[depth_predict > depth_clip] = depth_clip
    depth_im = img_HWC2CHW(colorize(depth_predict, cmap_name='jet', append_cbar=True,
                                    mask=mask))
    writer.add_image(prefix + 'level_{0}/depth_fgbg', depth_im, global_step)
    #
    if DEBUG:

        rgb_im = img_HWC2CHW(others['rgb_fg'])
        rgb_im = torch.clamp(rgb_im, min=0., max=1.)  # just in case diffuse+specular>1
        writer.add_image(prefix + 'level_{}/fg_rgb'.format(0), rgb_im, global_step)

        depth = others['d_fg']
        depth_im = img_HWC2CHW(colorize(depth, cmap_name='jet', append_cbar=True,
                                         mask=mask))
        writer.add_image(prefix + 'level_{}/fg_depth'.format(0), depth_im, global_step)

        rgb_im = img_HWC2CHW(others['rgb_bg'])
        rgb_im = torch.clamp(rgb_im, min=0., max=1.)  # just in case diffuse+specular>1
        writer.add_image(prefix + 'level_{}/bg_rgb'.format(0), rgb_im, global_step)

        depth = others['d_bg']
        depth_im = img_HWC2CHW(colorize(depth, cmap_name='jet', append_cbar=True, mask=mask))
        writer.add_image(prefix + 'level_{}/bg_depth'.format(0), depth_im, global_step)

        bg_lambda = others['d_lam']
        bg_lambda_im = img_HWC2CHW(colorize(bg_lambda, cmap_name='jet', append_cbar=True,
                                            mask=mask))
        writer.add_image(prefix + 'level_{}/bg_lambda'.format(0), bg_lambda_im, global_step)


def create_nerf(rank, args):
    ###### create network and wrap in ddp; each process should do this
    # fix random seed just to make sure the network is initialized with same weights at different processes
    torch.manual_seed(777)

    models = OrderedDict()
    models['cascade_level'] = 1
    models['cascade_samples'] = [int(x.strip()) for x in args.cascade_samples.split(',')]

    img_names = None

    if args.train_box_only:
        ora_net = DepthOracleBoxOnly(args).to(rank)
        models['net_oracle'] = WrapperModule(ora_net)
        optim = torch.optim.Adam(ora_net.parameters(), lr=args.lrate)
        models['optim_oracle'] = optim
        net = NerfNetBoxOnlyWithAutoExpo(args, optim_autoexpo=args.optim_autoexpo, img_names=img_names).to(rank)
    elif args.have_box:

        # ora_net = DepthOracleWithBox(args).to(rank)

        ora_net = DepthOracle(args).to(rank)
        models['net_oracle'] = WrapperModule(ora_net)
        optim = torch.optim.Adam(ora_net.parameters(), lr=args.lrate)
        models['optim_oracle'] = optim
        net = NerfNetMoreBoxWithAutoExpo(args, optim_autoexpo=args.optim_autoexpo, img_names=img_names).to(rank)

        netscene = WrapperModule(NerfNetWithAutoExpo(args, optim_autoexpo=args.optim_autoexpo, img_names=img_names).to(rank))
    else:
        ora_net = DepthOracle(args).to(rank)
        models['net_oracle'] = WrapperModule(ora_net)
        optim = torch.optim.Adam(ora_net.parameters(), lr=args.lrate)
        models['optim_oracle'] = optim
        net = NerfNetWithAutoExpo(args, optim_autoexpo=args.optim_autoexpo, img_names=img_names).to(rank)


    optim = torch.optim.Adam(net.parameters(), lr=args.lrate)
    models['net_0'] = WrapperModule(net)
    # models['net_0'] = net
    models['optim_0'] = optim


    start = -1

    ###### load pretrained weights; each process should do this
    if (args.ckpt_path is not None) and (os.path.isfile(args.ckpt_path)):
        ckpts = [args.ckpt_path]
    else:
        ckpts = [os.path.join(args.basedir, args.expname, f)
                 for f in sorted(os.listdir(os.path.join(args.basedir, args.expname))) if f.endswith('.pth')]

    def path2iter(path):
        tmp = os.path.basename(path)[:-4]
        idx = tmp.rfind('_')
        return int(tmp[idx + 1:])

    ckpts = sorted(ckpts, key=path2iter)
    logger.info('Found ckpts: {}'.format(ckpts))



    if len(ckpts) > 0 and not args.no_reload:


        fpath = ckpts[-1]
        logger.info('Reloading from: {}'.format(fpath))
        start = path2iter(fpath)
        # start = 0





        # configure map_location properly for different processes
        map_location = 'cuda:%d' % rank

        to_load = torch.load(fpath, map_location=map_location)

        models['optim_oracle'].load_state_dict(to_load['optim_oracle'])
        models['net_oracle'].load_state_dict(to_load['net_oracle'])

        ###########################33 before donerf use random weights for nerf ##########
        for m in range(models['cascade_level']):
            for name in ['net_{}'.format(m), 'optim_{}'.format(m)]:
                 models[name].load_state_dict(to_load[name])
        ####################################################################################333
         #### tmp for reloading pretrained model`
        # fpath_sc = "/media/diskstation/sally/donerf/logs/pretrained/sceneonly/model_425000.pth"
        #
        # to_load_sc = torch.load(fpath_sc, map_location=map_location)
        #
        # for k in to_load_sc['net_0'].keys():
        #       to_load['net_0'][k] = to_load_sc['net_1'][k]
        #
        # models['net_0'].load_state_dict(to_load['net_0'])

        # fpath_sc = "/media/diskstation/sally/donerf/logs/pretrained/big_inters_norm15_comb_rgb_disp/model_775000.pth"
        # to_load_sc = torch.load(fpath_sc, map_location=map_location)
        #
        # for k in to_load['net_0'].keys():
        #     to_load['net_0'][k] = to_load_sc['net_1'][k]
        #
        # models['net_0'].load_state_dict(to_load['net_0'])

    elif not args.have_box:

        start = 0
        #
        map_location = 'cuda:%d' % rank
        #
        fpath_dep ="/home/sally/nerf_clone/nerfpp_depth/logs/box_only_train_norm3_K=9Z=9_bg127/model_340000.pth"
        #
        to_load_dep = torch.load(fpath_dep, map_location=map_location)

        models['optim_oracle'].load_state_dict(to_load_dep['optim_oracle'])
        models['net_oracle'].load_state_dict(to_load_dep['net_oracle'])

        ##########################33 before donerf use random weights for nerf ##########
        # for m in range(models['cascade_level']):
        #     for name in ['net_{}'.format(m), 'optim_{}'.format(m)]:
        #         models[name].load_state_dict(to_load[name])
        ###################################################################################333
        ### tmp for reloading pretrained model
        # fpath_sc = "/media/diskstation/sally/donerf/logs/pretrained/sceneonly/model_425000.pth"
        #
        # to_load_sc = torch.load(fpath_sc, map_location=map_location)
        #
        # for k in to_load_sc['net_0'].keys():
        #       to_load_dep['net_0'][k] = to_load_sc['net_1'][k]
        #
        # models['net_0'].load_state_dict(to_load_dep['net_0'])

        # fpath_sc = "/media/diskstation/sally/donerf/logs/pretrained/big_inters_norm15_comb_rgb_disp/model_775000.pth"
        # to_load_sc = torch.load(fpath_sc, map_location=map_location)
        #
        # for k in to_load_dep['net_0'].keys():
        #     to_load_dep['net_0'][k] = to_load_sc['net_1'][k]
        #
        # models['net_0'].load_state_dict(to_load_dep['net_0'])




    elif args.have_box:

        map_location = 'cuda:%d' % rank
        start = 0
        # fpath_box = '/home/sally/nerf/nerfplusplus/logs/box_300_2-1_fullview_sc0.5_mx100-140/model_485000.pth'
        # fpath_sc = '/home/sally/nerf/nerfplusplus/logs/newinter_10x10x18_npp/model_770000.pth'

        ############## small inters only #############333
        #fpath_box = '/home/sally/nerfpp/box_models/box_model_485000.pth'
        #fpath_sc = '/home/sally/nerfpp/box_models/bg_model_770000.pth'
        ############## small inters only #############333

        ############### big inters #############
        # fpath_box = '/media/diskstation/sally/pretrained/box_models/box_model_485000.pth'
        # fpath_sc = '/media/diskstation/sally/pretrained/big_inters_norm15_sceneonly/model_425000.pth'

        ## ---------------new--------------- ##
        fpath_comb = '/media/diskstation/sally/pretrained/big_inters_norm15_comb_correct/model_420000.pth'
        # fpath_depth_ora = "/home/sally/nerf_clone/nerfpp_depth/logs/seg_test_filt9_rgb/model_275000.pth"
        # fpath_depth_ora = "/home/sally/nerf_clone/nerfpp_depth/logs/box_sample192_rgb64_huri_fromtrained/model_490000.pth"
        fpath_depth_ora = "/home/sally/nerf_clone/nerfpp_depth/logs/box_sample192_rgb64_boxcorrect/model_725000.pth"
        # fpath_depth_ora = "/home/sally/nerf_clone/nerfpp_depth/logs/scene_nobox_set2_fr192_K9Z5//model_575000.pth"
        # fpath_depth_ora = "/home/sally/nerf_clone/nerfpp_depth/logs//box_oracle/model_995000.pth"
        fpath_depth_scene = "/home/sally/nerf_clone/nerfpp_depth/logs/scene_nobox_set2_fr192_K9Z5//model_970000.pth"


        # models['net_oracle'].load_state_dict(to_load_dep['net_oracle'])
        ############### big inters #############


        # to_load_box = torch.load(fpath_box, map_location=map_location)
        to_load_comb = torch.load(fpath_comb, map_location=map_location)
        to_load_dep = torch.load(fpath_depth_ora, map_location=map_location)
        to_load_sc_ora = torch.load(fpath_depth_scene, map_location=map_location)


        models['optim_oracle'].load_state_dict(to_load_sc_ora['optim_oracle'])
        models['net_oracle'].load_state_dict(to_load_sc_ora['net_oracle'])



        ## if the model being loaded has 2 cas level
        # for k in to_load_sc['net_1'].keys():
        #       to_load_sc['net_0'][k] = to_load_sc['net_1'][k]
        #

        # load scene & box from comb
        # for k in to_load_comb['net_1'].keys():
        #     to_load_comb['net_0'][k] = to_load_comb['net_1'][k]
        #
        # # load scene from donerf
        # for k in to_load_dep['net_0'].keys():
        #     to_load_comb['net_0'][k] = to_load_dep['net_0'][k]

        # models['net_0'].load_state_dict(to_load_comb['net_0'])

        # for load weights without box

        model_dict_old = netscene.state_dict()
        model_dict_new = models['net_0'].state_dict()

        for k in model_dict_old.keys():
            model_dict_new[k] = to_load_dep['net_0'][k]


        # model_dict = {k: v  for k,v in to_load_dep['net_0'].items() if k in model_dict_old  }
        #
        # print(model_dict_old)

        models['net_0'].load_state_dict(model_dict_new)

        ## ---------------new--------------- ##




        # for m in range(models['cascade_level']):
        #     # for name in ['net_{}'.format(m), 'optim_{}'.format(m)]:
        #     for name in ['net_{}'.format(m)]:
        #         for k in to_load_box[name].keys():
        #             to_load_sc[name][k] = to_load_box[name][k]
        #
        #         models[name].load_state_dict(to_load_sc[name])



    return start, models


def ddp_train_nerf(rank, args):

    ###### set up logger
    logger = logging.getLogger(__package__)
    setup_logger()


    ###### Create log dir and copy the config file

    os.makedirs(os.path.join(args.basedir, args.expname), exist_ok=True)
    f = os.path.join(args.basedir, args.expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in (vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(args.basedir, args.expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())


    # HERE SHOULD HAVE ONE HOT EMBEDDING
    ray_samplers = load_data_split(args.datadir, args.scene, split='train',
                                   try_load_min_depth=args.load_min_depth, have_box=args.have_box,
                                   train_depth=True, train_seg=args.train_seg, train_box_only=args.train_box_only,
                                   box_number=args.box_number)
    val_ray_samplers = load_data_split(args.datadir, args.scene, split='test',
                                       try_load_min_depth=args.load_min_depth, skip=args.testskip, have_box=args.have_box,
                                       train_depth=True , train_seg=args.train_seg, train_box_only=args.train_box_only,
                                       box_number=args.box_number)




    ###### create network and wrap in ddp; each process should do this
    start, models = create_nerf(rank, args)

    ##### important!!!
    # make sure different processes sample different rays
    np.random.seed((rank + 1) * 777)
    # make sure different processes have different perturbations in depth samples
    torch.manual_seed((rank + 1) * 777)

    writer = SummaryWriter(os.path.join(args.basedir, 'summaries', args.expname))

    # start training
    what_val_to_log = 0             # helper variable for parallel rendering of a image
    what_train_to_log = 0

    loss_bce = torch.nn.BCEWithLogitsLoss() # this has sigmoid before bce

    loss_type = 'bce'
    add_entropy = False

    # with torch.autograd.detect_anomaly():
    for global_step in range(start+1, start+1+args.N_iters):

        time0 = time.time()
        scalars_to_log = OrderedDict()
        i = np.random.randint(low=0, high=len(ray_samplers))

        ray_batch = ray_samplers[i].random_sample_classifier(args.N_rand,
                                                             args.front_sample, args.back_sample,
                                                             pretrain=args.donerf_pretrain, rank=rank)

        for key in ray_batch:
            if torch.is_tensor(ray_batch[key]):
                ray_batch[key] = ray_batch[key].to(rank)


        net_oracle = models['net_oracle']
        optim_oracle = models['optim_oracle']

        optim_oracle.zero_grad()



        if args.donerf_pretrain:

            if args.have_box:
                ret = net_oracle(ray_batch['ray_o'].float(), ray_batch['ray_d'].float(),
                                 ray_batch['fg_pts_flat'].float(), ray_batch['bg_pts_flat'].float(),
                                 ray_batch['fg_far_depth'].float(), box_loc=ray_batch['box_loc'][:,:2])
            else:
                ret = net_oracle(ray_batch['ray_o'].float(), ray_batch['ray_d'].float(),
                                 ray_batch['fg_pts_flat'].float(), ray_batch['bg_pts_flat'].float(),
                                 ray_batch['fg_far_depth'].float())



            label_fg = ray_batch['cls_label'][:, :args.front_sample]
            label_bg = ray_batch['cls_label'][:, args.front_sample:]

            loss_fg = loss_bce(ret['likeli_fg'], label_fg)
            scalars_to_log['bce_loss_fg'] = loss_fg.item()

            loss_bg = loss_bce(ret['likeli_bg'], label_bg)
            scalars_to_log['bce_loss_bg'] = loss_bg.item()




            loss_cls = loss_fg + loss_bg


            loss_cls.backward()
            optim_oracle.step()

            scalars_to_log['loss'] = loss_cls.item()



        else:

            with torch.no_grad():

                if args.have_box:
                    # ret = net_oracle(ray_batch['ray_o'].float(), ray_batch['ray_d'].float(),
                    #                  ray_batch['fg_pts_flat'].float(), ray_batch['bg_pts_flat'].float(),
                    #                  ray_batch['fg_far_depth'].float(), box_loc=ray_batch['box_loc'][:, :2])
                    ret = net_oracle(ray_batch['ray_o'].float(), ray_batch['ray_d'].float(),
                                     ray_batch['fg_pts_flat'].float(), ray_batch['bg_pts_flat'].float(),
                                     ray_batch['fg_far_depth'].float())

                else:
                    ret = net_oracle(ray_batch['ray_o'].float(), ray_batch['ray_d'].float(),
                                 ray_batch['fg_pts_flat'].float(), ray_batch['bg_pts_flat'].float(),
                                 ray_batch['fg_far_depth'].float())



            # # results on different cascade levels

            optim = models['optim_0']
            net = models['net_0']

            # sample depths
            N_samples = models['cascade_samples'][0]


            perturbed_seg_bound_fg = perturb_samples(ray_batch['fg_z_vals_centre']) #fg_z_vals_centre is actually the z_vals of seg bounds. pts are mid pints
            perturbed_seg_bound_bg = perturb_samples(ray_batch['bg_z_vals_centre']) #fg_z_vals_centre is actually the z_vals of seg bounds. pts are mid pints

            fg_mid = 0.5 * (perturbed_seg_bound_fg[:, 1:] + perturbed_seg_bound_fg[:, :-1])
            bg_mid = 0.5 * (perturbed_seg_bound_bg[:, 1:] + perturbed_seg_bound_bg[:, :-1])




            # fg_weights = ray_batch['cls_label'][:,:args.front_sample]
            fg_weights = ret['likeli_fg']


            fg_weights = torch.sigmoid(fg_weights).clone().detach()


            fg_weights[ray_batch['fg_z_vals_centre']<0.0002] = 0.

            if args.have_box:
                with torch.no_grad():


                    # option1

                    # ret_box = net(ray_batch['ray_o'], ray_batch['ray_d'], ray_batch['fg_far_depth'], ray_batch['fg_z_vals_centre'],
                    #         bg_z_vals=None, box_loc=ray_batch['box_loc'],
                    #         query_box_only=True, img_name=ray_batch['img_name'])
                    # box_weights = ret_box['fg_box_sig']
                    # fg_weights = fg_weights + normalize_torch(box_weights)
                    #
                    # option2
                    # lets test this --- we are gettin occupancy only, maybe we could get transmittance
                    # box_weights = get_box_weight(box_loc=ray_batch['box_loc'], box_size=1./30.,
                    #                              fg_z_vals=ray_batch['fg_z_vals_centre'],
                    #                              ray_d=ray_batch['ray_d'], ray_o=ray_batch['ray_o'], box_number=args.box_number )


                    box_weights = get_box_transmittance_weight(box_loc=ray_batch['box_loc'], box_size=args.box_size,
                                                               fg_z_vals=perturbed_seg_bound_fg, ray_d=ray_batch['ray_d'], ray_o=ray_batch['ray_o'],
                                                               fg_depth=ray_batch['fg_far_depth'], box_number=args.box_number)
                    box_weights[ray_batch['fg_z_vals_centre'] < 0.0002] = 0.

                    fg_weights = fg_weights + (box_weights)



            fg_depth,_ = torch.sort(sample_pdf(bins=fg_mid, weights=fg_weights[:, 1:args.front_sample-1],
                                          N_samples=N_samples, det=False))    # [..., N_samples]
            fg_depth[fg_depth<0.0002] = 0.0002

            bg_weights = torch.sigmoid(ret['likeli_bg'])[:, 1: args.back_sample-1].clone().detach()

            bg_weights = torch.fliplr(bg_weights)

            bg_depth,_ = torch.sort(sample_pdf(bins=bg_mid, weights=bg_weights,
                                          N_samples=N_samples, det=False))    # [..., N_samples]



            optim.zero_grad()
            if not args.have_box:
                ret = net(ray_batch['ray_o'], ray_batch['ray_d'], ray_batch['fg_far_depth'], fg_depth, bg_depth, img_name=ray_batch['img_name'])
            else:
                ret = net(ray_batch['ray_o'].float(), ray_batch['ray_d'].float(),
                          ray_batch['fg_far_depth'].float(), fg_depth.float(), bg_depth.float(),
                          ray_batch['box_loc'].float(), img_name=ray_batch['img_name'])


            rgb_gt = ray_batch['rgb'].to(rank)
            rgb_loss = img2mse(ret['rgb'], rgb_gt)
            loss = rgb_loss

            if args.depth_training:
                depth_gt = ray_batch['depth_gt'].to(rank)
                depth_pred = ret['depth_fgbg']

                inds = torch.where(depth_gt < 2001.)
                d_pred_map = depth_pred[inds]
                d_gt_map = depth_gt[inds]

                depth_loss = dep_l1l2loss(torch.div(1.,d_pred_map), torch.div(1.,d_gt_map), l1l2 = 'l1')
                #reg_loss = dep_l1l2loss(torch.div(1.,d_pred_map[:512])-torch.div(1.,d_pred_map[512:]), torch.div(1.,d_gt_map[:512])-torch.div(1.,d_gt_map[512:]), l1l2 = 'l1')
                loss += 0.1 * depth_loss
                #loss = rgb_loss * 0 +  depth_loss
                scalars_to_log['level_0/depth_loss'] = depth_loss.item()


            if args.seg_box_loss:
                seg_box_gt = ray_batch['seg_map_box']
                seg_prediction = ret['box_weights']
                seg_box_loss =  dep_l1l2loss(seg_prediction, seg_box_gt, l1l2 = 'l2')
                loss += seg_box_loss
                scalars_to_log['level_0/seg_box_loss'] = seg_box_loss.item()







            # print(global_step)
            scalars_to_log['level_0/rgb_loss'] = rgb_loss.item()
            scalars_to_log['level_0/pnsr'] = mse2psnr(rgb_loss.item())
            # print('before backward')
            loss.backward()
            # print('before step')
            optim.step()



        ### end of core optimization loop
        dt = time.time() - time0
        scalars_to_log['iter_time'] = dt

        ### only main process should do the logging
        if (global_step % args.i_print == 0 or global_step < 10):
            # print('=======likeli=====')
            # print(out_likeli_fg)
            # print(out_likeli_bg)

            logstr = '{} step: {} '.format(args.expname, global_step)
            for k in scalars_to_log:
                logstr += ' {}: {:.6f}'.format(k, scalars_to_log[k])
                writer.add_scalar(k, scalars_to_log[k], global_step)
            logger.info(logstr)

        # ## each process should do this; but only main process merges the results
        if global_step % args.i_img == 0 or global_step == start+1:

            # select_inds = np.random.choice(640*360, size=(200,), replace=False)

            select_inds = np.arange(640*280,640*281)
            select_inds2 = np.arange(640*240,640*241)

            #### critical: make sure each process is working on the same random image
            time0 = time.time()
            idx = what_val_to_log % len(val_ray_samplers)

            print('IMAGE_ID: {}'.format(idx))

            with torch.no_grad():
                rgb, d, pred_fg, pred_bg, label, others  = render_single_image(models, val_ray_samplers[idx], args.chunk_size,
                                               args.train_box_only, have_box=args.have_box, donerf_pretrain=args.donerf_pretrain,
                                               front_sample=args.front_sample, back_sample=args.back_sample, fg_bg_net=args.fg_bg_net,
                                               use_zval=args.use_zval,  loss_type=loss_type,  rank=rank, DEBUG=True, box_number=args.box_number, box_size=args.box_size)




            what_val_to_log += 1
            dt = time.time() - time0

            logger.info('Logged a random validation view in {} seconds'.format(dt))

            if args.donerf_pretrain:
                # if args.fg_bg_net:

                label_fg = label[:, :args.front_sample]

                label_bg = label[:, args.front_sample:]


                out_likeli_fg = np.array(torch.sigmoid(pred_fg).cpu().detach().numpy())

                out_likeli_bg = np.array(torch.sigmoid(pred_bg).cpu().detach().numpy())

                loss_fg = loss_bce(pred_fg[select_inds],  label_fg[select_inds].cpu())
                scalars_to_log['val/bce_loss_fg'] = loss_fg.item()

                loss_bg = loss_bce(pred_bg[select_inds], label_bg[select_inds].cpu())
                scalars_to_log['val/bce_loss_bg'] = loss_bg.item()

                loss_cls = (loss_fg + loss_bg)

                scalars_to_log['val/loss'] = loss_cls.item()



                del loss_fg
                del loss_bg

                metrics_fg = calculate_metrics(out_likeli_fg, np.array(label_fg.cpu().detach().numpy()),
                                               'micro')

                for k in metrics_fg:
                    scalars_to_log['val/'+ k + '_fg'] = metrics_fg[k]

                # log_plot_conf_mat(writer, metrics_fg['cm'], global_step, 'val/CM_fg')

                metrics_bg = calculate_metrics(out_likeli_bg, np.array(label_bg.cpu().detach().numpy()),
                                               'micro')

                for k in metrics_bg:
                    scalars_to_log['val/'+ k + '_bg'] = metrics_bg[k]

                # log_plot_conf_mat(writer, metrics_bg['cm'], global_step, 'val/CM_bg')
                visualize_depth_label(writer, np.concatenate([label_fg.cpu().detach().numpy()[select_inds], label_fg.cpu().detach().numpy()[select_inds2]], axis=1),
                                      np.concatenate([out_likeli_fg[select_inds], out_likeli_fg[select_inds2]], axis=1), global_step, 'val/dVis_fg')
                visualize_depth_label(writer,np.concatenate([label_bg.cpu().detach().numpy()[select_inds], label_bg.cpu().detach().numpy()[select_inds2]], axis=1),
                                      np.concatenate([out_likeli_bg[select_inds], out_likeli_bg[select_inds2]], axis=1), global_step, 'val/dVis_bg')


                logstr = '[=VALIDATION=] {} step: {} '.format(args.expname, global_step)
                for k in scalars_to_log:
                    logstr += ' {}: {:.6f}'.format(k, scalars_to_log[k])
                    writer.add_scalar(k, scalars_to_log[k], global_step)
                logger.info(logstr)

                del metrics_fg
                del metrics_bg
                del out_likeli_fg
                del out_likeli_bg



            else:
                # label_fg = label[:, :args.front_sample]
                # label_bg = label[:, args.front_sample:]
                #if args.depth_training:
                log_view_to_tb(writer, global_step, rgb,d , gt_depth=val_ray_samplers[idx].get_depth(), box_seg_mask=val_ray_samplers[idx].get_box_mask(),
                               gt_img=val_ray_samplers[idx].get_img(), mask=None, have_box=args.have_box,
                               train_box_only=args.train_box_only, prefix='val/',DEBUG=True, others=others)


                # visualize_depth_label(writer, np.concatenate(
                #     [label_fg.cpu().detach().numpy()[select_inds], label_fg.cpu().detach().numpy()[select_inds2]],axis=1),
                #                       np.concatenate([torch.sigmoid(pred_fg[select_inds]),
                #                                       torch.sigmoid(pred_fg[select_inds2])],
                #                                      axis=1), global_step, 'val/dVis_fg')
                # visualize_depth_label(writer, np.concatenate(
                #     [label_bg.cpu().detach().numpy()[select_inds], label_bg.cpu().detach().numpy()[select_inds2]],
                #     axis=1),
                #                       np.concatenate([torch.sigmoid(pred_bg[select_inds]),
                #                                       torch.sigmoid(pred_bg[select_inds2])],
                #                                      axis=1), global_step, 'val/dVis_fg')

                visualize_depth_label(writer, None,
                                          np.concatenate([torch.sigmoid(pred_fg[select_inds]),
                                                          torch.sigmoid(pred_fg[select_inds2])],
                                                         axis=1), global_step, 'val/dVis_fg')
                visualize_depth_label(writer, None,
                                          np.concatenate([torch.sigmoid(pred_bg[select_inds]),
                                                          torch.sigmoid(pred_bg[select_inds2])],
                                                         axis=1), global_step, 'val/dVis_fg')


                #else:
                 #   log_view_to_tb(writer, global_step, log_data, gt_img=val_ray_samplers[idx].get_img(), mask=None, have_box=args.have_box, train_box_only=args.train_box_only, prefix='val/')

            idx = what_train_to_log % len(ray_samplers)
            print('IMAGE_ID-train: {}'.format(idx))
            time0 = time.time()

            with torch.no_grad():
                rgb, d, pred_fg, pred_bg, label,others = render_single_image(models, ray_samplers[idx], args.chunk_size,
                                                      args.train_box_only, have_box=args.have_box,
                                                      donerf_pretrain=args.donerf_pretrain,
                                                      front_sample=args.front_sample, back_sample=args.back_sample,
                                                      fg_bg_net=args.fg_bg_net, use_zval=args.use_zval,  loss_type=loss_type,
                                                       rank=rank, DEBUG=True, box_number=args.box_number, box_size=args.box_size)

            what_train_to_log += 1
            dt = time.time() - time0

            logger.info('Logged a random validation view in {} seconds'.format(dt))

            if args.donerf_pretrain:
                # if args.fg_bg_net:

                label_fg = label[:, :args.front_sample]
                label_bg = label[:, args.front_sample:]


                loss_fg = loss_bce(pred_fg[select_inds], label_fg[select_inds].cpu())
                loss_bg = loss_bce(pred_bg[select_inds], label_bg[select_inds].cpu())
                scalars_to_log['train/bce_loss_fg'] = loss_fg.item()
                scalars_to_log['train/bce_loss_bg'] = loss_bg.item()


                loss_cls = (loss_fg + loss_bg)


                scalars_to_log['train/loss'] = loss_cls.item()

                del loss_fg
                del loss_bg


                out_likeli_fg = np.array(torch.sigmoid(pred_fg).cpu().detach().numpy())
                out_likeli_bg = np.array(torch.sigmoid(pred_bg).cpu().detach().numpy())

                metrics_fg = calculate_metrics(out_likeli_fg, np.array(label_fg.cpu().detach().numpy()),
                                               'micro')

                for k in metrics_fg:
                    scalars_to_log['train/' + k + '_fg'] = metrics_fg[k]


                metrics_bg = calculate_metrics(out_likeli_bg, np.array(label_bg.cpu().detach().numpy()),
                                               'micro')

                for k in metrics_bg:
                    scalars_to_log['train/' + k + '_bg'] = metrics_bg[k]

                # log_plot_conf_mat(writer, metrics_bg['cm'], global_step, 'val/CM_bg')

                visualize_depth_label(writer, np.concatenate(
                    [label_fg.cpu().detach().numpy()[select_inds], label_fg.cpu().detach().numpy()[select_inds2]], axis=1),
                                      np.concatenate([out_likeli_fg[select_inds], out_likeli_fg[select_inds2]], axis=1), global_step, 'train/dVis_fg')
                visualize_depth_label(writer, np.concatenate(
                    [label_bg.cpu().detach().numpy()[select_inds], label_bg.cpu().detach().numpy()[select_inds2]], axis=1),
                                      np.concatenate([out_likeli_bg[select_inds], out_likeli_bg[select_inds2]], axis=1), global_step, 'train/dVis_bg')


                # visualize_depth_label(writer, np.array(label_fg.cpu().detach().numpy())[select_inds],
                #                       out_likeli_fg[select_inds], global_step, 'train/dVis_fg')
                # visualize_depth_label(writer, np.array(label_bg.cpu().detach().numpy())[select_inds],
                #                       out_likeli_bg[select_inds], global_step, 'train/dVis_bg')
                #
                # loss_deviation(writer, np.array(label_fg.cpu().detach().numpy())[select_inds],
                #                       out_likeli_fg[select_inds], global_step, 'train/LossVis_fg')
                # loss_deviation(writer, np.array(label_bg.cpu().detach().numpy())[select_inds],
                #                       out_likeli_bg[select_inds], global_step, 'train/LossVis_bg')


                logstr = '[=VALID_TRAIN] {} step: {} '.format(args.expname, global_step)
                for k in scalars_to_log:
                    logstr += ' {}: {:.6f}'.format(k, scalars_to_log[k])
                    writer.add_scalar(k, scalars_to_log[k], global_step)
                logger.info(logstr)

                del metrics_fg
                del metrics_bg
                del out_likeli_fg
                del out_likeli_bg



            else:

                # label_fg = label[:, :args.front_sample]
                # label_bg = label[:, args.front_sample:]

                # if args.depth_training:
                log_view_to_tb(writer, global_step, rgb,d, gt_depth=ray_samplers[idx].get_depth(), box_seg_mask=ray_samplers[idx].get_box_mask(),
                               gt_img=ray_samplers[idx].get_img(), mask=None, have_box=args.have_box,
                               train_box_only=args.train_box_only, prefix='train/', DEBUG=True, others=others)



                # visualize_depth_label(writer, np.concatenate(
                #     [label_fg.cpu().detach().numpy()[select_inds], label_fg.cpu().detach().numpy()[select_inds2]],axis=1),
                #     np.concatenate([torch.sigmoid(pred_fg[select_inds]), torch.sigmoid(pred_fg[select_inds2])],
                #                                      axis=1), global_step, 'train/dVis_fg')
                # visualize_depth_label(writer, np.concatenate(
                #     [label_bg.cpu().detach().numpy()[select_inds], label_bg.cpu().detach().numpy()[select_inds2]],axis=1),
                #                       np.concatenate([torch.sigmoid(pred_bg[select_inds]),
                #                                       torch.sigmoid(pred_bg[select_inds2])],
                #                                      axis=1), global_step, 'train/dVis_fg')

                visualize_depth_label(writer, None,
                                      np.concatenate([torch.sigmoid(pred_fg[select_inds]),
                                                      torch.sigmoid(pred_fg[select_inds2])],
                                                     axis=1), global_step, 'train/dVis_fg')
                visualize_depth_label(writer, None,
                                      np.concatenate([torch.sigmoid(pred_bg[select_inds]),
                                                      torch.sigmoid(pred_bg[select_inds2])],
                                                     axis=1), global_step, 'train/dVis_fg')





            del rgb
            del d
            del pred_fg
            del pred_bg
            del others
            torch.cuda.empty_cache()

        if (global_step % args.i_weights == 0 and global_step > 0):

            # saving checkpoints and logging
            fpath = os.path.join(args.basedir, args.expname, 'model_{:06d}.pth'.format(global_step))
            to_save = OrderedDict()

            to_save['net_oracle'] = models['net_oracle'].state_dict()
            to_save['optim_oracle'] = models['optim_oracle'].state_dict()

            to_save['net_0'] = models['net_0'].state_dict()
            to_save['optim_0'] = models[ 'optim_0'].state_dict()
            torch.save(to_save, fpath)





def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument("--expname", type=str, help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', help='where to store ckpts and logs')
    # dataset options
    parser.add_argument("--datadir", type=str, default=None, help='input data directory')
    parser.add_argument("--scene", type=str, default=None, help='scene name')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
    # model size
    parser.add_argument("--netdepth", type=int, default=8, help='layers in coarse network')
    parser.add_argument("--netwidth", type=int, default=256, help='channels per layer in coarse network')
    parser.add_argument("--use_viewdirs", action='store_true', help='use full 5D input instead of 3D')
    # checkpoints
    parser.add_argument("--no_reload", action='store_true', help='do not reload weights from saved ckpt')
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
    # batch size
    parser.add_argument("--N_rand", type=int, default=32 * 32 * 2,
                        help='batch size (number of random rays per gradient step)')

    parser.add_argument("--back_sample", type=int, default=128,
                        help='num samples in the background')
    parser.add_argument("--front_sample", type=int, default=128,
                        help='num of samples in the front scene')

    parser.add_argument("--donerf_pretrain", action='store_true', help='donerf pretrain -- no rgb yet')

    parser.add_argument("--chunk_size", type=int, default=1024 * 8,
                        help='number of rays processed in parallel, decrease if running out of memory')
    # iterations
    parser.add_argument("--N_iters", type=int, default=250001,
                        help='number of iterations')
    # render only
    parser.add_argument("--render_splits", type=str, default='test',
                        help='splits to render')
    # cascade training
    parser.add_argument("--cascade_level", type=int, default=2,
                        help='number of cascade levels')
    parser.add_argument("--cascade_samples", type=str, default='64,64',
                        help='samples at each level')
    # multiprocess learning
    parser.add_argument("--world_size", type=int, default='-1',
                        help='number of processes')
    # optimize autoexposure
    parser.add_argument("--optim_autoexpo", action='store_true',
                        help='optimize autoexposure parameters')
    parser.add_argument("--lambda_autoexpo", type=float, default=1., help='regularization weight for autoexposure')

    # learning rate options
    parser.add_argument("--lrate", type=float, default=5e-4, help='learning rate')
    parser.add_argument("--lrate_decay_factor", type=float, default=0.1,
                        help='decay learning rate by a factor every specified number of steps')
    parser.add_argument("--lrate_decay_steps", type=int, default=5000,
                        help='decay learning rate by a factor every specified number of steps')
    # rendering options
    parser.add_argument("--det", action='store_true', help='deterministic sampling for coarse and fine samples')
    parser.add_argument("--max_freq_log2", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--max_freq_log2_viewdirs", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--load_min_depth", action='store_true', help='whether to load min depth')
    # logging/saving options
    parser.add_argument("--i_print", type=int, default=100, help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img", type=int, default=500, help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, help='frequency of weight ckpt saving')

    parser.add_argument("--have_box", action='store_true',
                        help='whether use box location in model')

    parser.add_argument("--train_box_only", action='store_true',
                        help='whether to train with the box only')

    parser.add_argument("--depth_training", action='store_true',
                        help='whether to train with depth')

    parser.add_argument("--fg_bg_net", action='store_true',
                        help='whether to train with depth')

    parser.add_argument("--pencode", action='store_true',
                        help='whether to use position encoding for input rayo rayd for depth oracle')

    parser.add_argument("--use_zval", action='store_true',
                        help='whether to input zval to network')

    parser.add_argument("--penc_pts", action='store_true',
                        help='penc segment points')

    parser.add_argument("--train_seg", action='store_true',
                        help='use segmentation mask to prioritize training')

    parser.add_argument("--seg_box_loss", action='store_true',
                        help='add segmentation box loss')

    parser.add_argument("--max_freq_log2_pts", type=int, default=3,
                        help='log2 of max freq for positional encoding (seg points)')

    parser.add_argument("--box_number", type=int, default=1,
                        help='number of box in the scene')



    parser.add_argument("--box_size", type=str, default='1,1,1',
                        help='size of box in the scene')


    return parser


def train():
    parser = config_parser()
    args = parser.parse_args()
    logger.info(parser.format_values())

    rank = torch.cuda.current_device()
    ddp_train_nerf(rank, args)

if __name__ == '__main__':
    setup_logger()
    train()
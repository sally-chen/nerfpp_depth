import torch
import torch.nn as nn
import torch.optim
import torch.distributed
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing
import os
from collections import OrderedDict
from ddp_model import NerfNetWithAutoExpo, NerfNetBoxWithAutoExpo, \
    NerfNetBoxOnlyWithAutoExpo, DepthOracleBig, DepthOracle

from nerf_network import WrapperModule

import time
from data_loader_split import load_data_split
import numpy as np
from tensorboardX import SummaryWriter
from utils import img2mse, mse2psnr, entropy_loss, dep_l1l2loss, img_HWC2CHW, colorize, colorize_np,to8b, TINY_NUMBER

from helpers import calculate_metrics, log_plot_conf_mat, visualize_depth_label, loss_deviation
import logging
import json

from helpers import plot_ray_batch


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





def render_single_image(models, ray_sampler, chunk_size,
                        train_box_only=False, have_box=False,
                        donerf_pretrain=False, front_sample=128, back_sample=128,
                        fg_bg_net=True, use_zval=True):
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
                                          pretrain=donerf_pretrain)

    chunks = (len(rays['ray_d']) + chunk_size - 1) // chunk_size
    rays_split = [OrderedDict() for _ in range(chunks)]

    for k, v in rays.items():
        if torch.is_tensor(v):
            split = torch.split(v, chunk_size)
            for i in range(chunks):
                rays_split[i][k] = split[i]

    rgbs = []
    depths = []

    for _rays in rays_split:
        chunk_ret = render_rays(models, _rays, train_box_only, have_box,
                                donerf_pretrain, front_sample, back_sample,
                                fg_bg_net, use_zval)
        rgbs.append(chunk_ret['rgb'])
        depths.append(chunk_ret['depth_fgbg'])

    rgb = torch.cat(rgbs).view(ray_sampler.H, ray_sampler.W, -1).squeeze()
    d = torch.cat(depths).view(ray_sampler.H, ray_sampler.W, -1).squeeze()
    return rgb, d

def eval_oracle(rays, net_oracle, fg_bg_net, use_zval):
    if fg_bg_net:

        if use_zval:

            ret = net_oracle(rays['ray_o'], rays['ray_d'],
                     rays['fg_z_vals_centre'], rays['bg_z_vals_centre'],
                     rays['fg_far_depth'])
        else:
            ret = net_oracle(rays['ray_o'], rays['ray_d'],
                         rays['fg_pts_flat'], rays['bg_pts_flat'],
                         rays['fg_far_depth'])
    else:
        if use_zval:
            pts = torch.cat([rays['fg_z_vals_centre'], rays['bg_z_vals_centre']], dim=-1)
        else:
            pts = torch.cat([rays['fg_pts_flat'], rays['bg_pts_flat']], dim=-1)
        ret = net_oracle(rays['ray_o'], rays['ray_d'], pts)


        ret['likeli_fg'] = ret['likeli'][:, :front_sample]
        ret['likeli_bg'] = ret['likeli'][:, front_sample:]
    return ret

def get_depths(data, front_sample, back_sample, fg_z_vals_centre, bg_z_vals_centre, samples, train_box_only=False):
    fg_depth_mid = fg_z_vals_centre
    fg_weights = data['likeli_fg'][:, 2:front_sample].clone() # Avoid inplace ops


    fg_weights[fg_depth_mid[:,:-1]<0.] =0.
    fg_depth,_ = torch.sort(sample_pdf(bins=fg_depth_mid, weights=fg_weights,
                          N_samples=samples, det=False))  # [..., N_samples]
    fg_depth[fg_depth<0.] = 0.

    if not train_box_only:
        # sample pdf and concat with earlier samples
        # bg_weights = ret['bg_weights'].clone().detach()
        # bg_depth_mid = .5 * (bg_depth[..., 1:] + bg_depth[..., :-1])
        # bg_weights = bg_weights[..., 1:-1]                              # [..., N_samples-2]

        bg_depth_mid = bg_z_vals_centre


        bg_weights = data['likeli_bg'][:, 1: back_sample-1].clone()
        bg_weights[bg_depth_mid[:,:-1]<0.] =0.
        bg_depth,_ = torch.sort(sample_pdf(bins=bg_depth_mid, weights=bg_weights,
                              N_samples=samples, det=False))  # [..., N_samples]
        bg_depth[bg_depth<0.] = 0.

    return fg_depth, bg_depth


def render_rays(models, rays, train_box_only, have_box, donerf_pretrain,
                front_sample, back_sample, fg_bg_net, use_zval):
    """Render a set of rays using specific config."""

    # forward and backward
    ret_merge_chunk = [OrderedDict() for _ in range(models['cascade_level'])]
    ray_o = rays['ray_o']
    ray_d = rays['ray_d']
    min_depth = rays['min_depth']
    box_loc = rays['box_loc'] if have_box else None
    fg_z_vals_centre = rays['fg_z_vals_centre']
    bg_z_vals_centre = rays['bg_z_vals_centre']
    fg_far_depth = rays['fg_far_depth']
    fg_pts_flat = rays['fg_pts_flat']
    bg_pts_flat = rays['bg_pts_flat']

    net_oracle = models['net_oracle']

    ret = eval_oracle(rays, net_oracle, fg_bg_net, use_zval)

    if not donerf_pretrain:
        for m in range(models['cascade_level']):
            net = models['net_{}'.format(m)]
            samples = models['cascade_samples'][m]
            fg_depth, bg_depth = get_depths(ret, front_sample, back_sample,
                                            fg_z_vals_centre, bg_z_vals_centre,
                                            samples, train_box_only)

            if not have_box:
                ret = net(ray_o, ray_d, fg_far_depth, fg_depth, bg_depth)
            elif not train_box_only:
                ret = net(ray_o, ray_d, fg_far_depth, fg_depth, bg_depth, box_loc)
            else:
                ret = net(ray_o, ray_d, fg_far_depth, fg_depth)

    return ret


def log_view_to_tb(writer, global_step, log_data, gt_img, mask, gt_depth=None, train_box_only= False, have_box=False, prefix=''):
    rgb_im = img_HWC2CHW(torch.from_numpy(gt_img))
    writer.add_image(prefix + 'rgb_gt', rgb_im, global_step)
    depth_clip = 100.
    # if gt_depth is not None:
    #
    #
    #     gt_depth[gt_depth > depth_clip] = depth_clip  ##### THIS IS THE DEPTH OUTPUT, HxW, value is meters away from camera centre
    #
    #
    #     depth_im = img_HWC2CHW(colorize(gt_depth, cmap_name='jet', append_cbar=True,
    #                                                         mask=mask, is_np=True))
    #     writer.add_image(prefix + 'depth_gt', depth_im, global_step)

    for m in range(len(log_data)):
        rgb_im = img_HWC2CHW(log_data[m]['rgb'])
        rgb_im = torch.clamp(rgb_im, min=0., max=1.)  # just in case diffuse+specular>1
        writer.add_image(prefix + 'level_{}/rgb'.format(m), rgb_im, global_step)
        #
        # #if have_box:
        # depth = log_data[m]['depth_fgbg']
        # depth[depth > depth_clip] = depth_clip
        # depth_im = img_HWC2CHW(colorize(depth, cmap_name='jet', append_cbar=True,
        #                                 mask=mask))
        # writer.add_image(prefix + 'level_{}/depth_fgbg'.format(m), depth_im, global_step)
        #


        rgb_im = img_HWC2CHW(log_data[m]['fg_rgb'])
        rgb_im = torch.clamp(rgb_im, min=0., max=1.)  # just in case diffuse+specular>1
        writer.add_image(prefix + 'level_{}/fg_rgb'.format(m), rgb_im, global_step)

        # depth = log_data[m]['fg_depth']
        # depth_im = img_HWC2CHW(colorize(depth, cmap_name='jet', append_cbar=True,
        #                                  mask=mask))
        # writer.add_image(prefix + 'level_{}/fg_depth'.format(m), depth_im, global_step)
        # #
        #
        if not train_box_only:
           rgb_im = img_HWC2CHW(log_data[m]['bg_rgb'])
           rgb_im = torch.clamp(rgb_im, min=0., max=1.)  # just in case diffuse+specular>1
           writer.add_image(prefix + 'level_{}/bg_rgb'.format(m), rgb_im, global_step)
           # depth = log_data[m]['bg_depth']
           # depth_im = img_HWC2CHW(colorize(depth, cmap_name='jet', append_cbar=True,
           #                                 mask=mask))
           # writer.add_image(prefix + 'level_{}/bg_depth'.format(m), depth_im, global_step)
           #bg_lambda = log_data[m]['bg_lambda']
           #bg_lambda_im = img_HWC2CHW(colorize(bg_lambda, cmap_name='hot', append_cbar=True,
           #                                      mask=mask))
           #writer.add_image(prefix + 'level_{}/bg_lambda'.format(m), bg_lambda_im, global_step)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    port = np.random.randint(12355, 12399)
    os.environ['MASTER_PORT'] = '{}'.format(port)
    #os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    torch.distributed.destroy_process_group()


def create_nerf(args, device='cuda:0'):
    models = OrderedDict()

    if args.fg_bg_net:
        ora_net = DepthOracle(args).to(device)
    else:
        ora_net = DepthOracleBig(args).to(device)

    net = ora_net
    models['net_oracle'] = net
    optim = torch.optim.Adam(net.parameters(), lr=args.lrate)
    models['optim_oracle'] = optim

    models['cascade_level'] = args.cascade_level
    models['cascade_samples'] = [int(x.strip())
                                    for x in args.cascade_samples.split(',')]

    for m in range(models['cascade_level']):
        img_names = None
        if args.optim_autoexpo:
            # load training image names for autoexposure
            f = os.path.join(args.basedir, args.expname, 'train_images.json')
            with open(f) as file:
                img_names = json.load(file)
        if args.train_box_only:
            net = NerfNetBoxOnlyWithAutoExpo(args,
                optim_autoexpo=args.optim_autoexpo, img_names=img_names).to(device)
        elif args.have_box:
            net = NerfNetBoxWithAutoExpo(args,
                optim_autoexpo=args.optim_autoexpo, img_names=img_names).to(device)
        else:
            net = NerfNetWithAutoExpo(args,
                optim_autoexpo=args.optim_autoexpo, img_names=img_names).to(device)

        optim = torch.optim.Adam(net.parameters(), lr=args.lrate)
        models['net_{}'.format(m)] = WrapperModule(net)
        models['optim_{}'.format(m)] = optim

    start = -1

    ###### load pretrained weights; each process should do this
    if (args.ckpt_path is not None) and (os.path.isfile(args.ckpt_path)):
        ckpts = [args.ckpt_path]
    else:
        files = sorted(os.listdir(os.path.join(args.basedir, args.expname)))
        ckpts = [os.path.join(args.basedir, args.expname, f)
                                        for f in files if f.endswith('.pth')]

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
        to_load = torch.load(fpath, map_location='cpu')

        models['optim_oracle'].load_state_dict(to_load['optim_oracle'])
        models['net_oracle'].load_state_dict(to_load['net_oracle'])

        # Before initializing donerf: use random weights for nerf.(?)
        for m in range(models['cascade_level']):
            optim_name = 'optim_{}'.format(m)
            net_name = 'net_{}'.format(m)
            models[optim_name].load_state_dict(to_load[optim_name])
            models[net_name].load_state_dict(to_load[net_name])
            #for param in models[net_name].parameters():
            #   param.requires_grad_(False)

    return start, models


def ddp_train_nerf(rank, args):
    ###### set up multi-processing
    setup(rank, args.world_size)
    ###### set up logger
    logger = logging.getLogger(__package__)
    setup_logger()

    ###### decide chunk size according to gpu memory
    logger.info('gpu_mem: {}'.format(torch.cuda.get_device_properties(rank).total_memory))
    if torch.cuda.get_device_properties(rank).total_memory / 1e9 > 14:
        logger.info('setting batch size according to 24G gpu')
        args.N_rand = 1024
        args.chunk_size = 10240
    else:
        logger.info('setting batch size according to 12G gpu')
        args.N_rand = 1024
        args.chunk_size = 10240

    ###### Create log dir and copy the config file
    if rank == 0:
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
    torch.distributed.barrier()


    # HERE SHOULD HAVE ONE HOT EMBEDDING
    ray_samplers = load_data_split(args.datadir, args.scene, split='train',
                                   try_load_min_depth=args.load_min_depth, have_box=args.have_box,
                                   train_depth=True)
    val_ray_samplers = load_data_split(args.datadir, args.scene, split='validation',
                                       try_load_min_depth=args.load_min_depth, skip=args.testskip, have_box=args.have_box,
                                       train_depth=True)



    ###### create network and wrap in ddp; each process should do this
    start, models = create_nerf(rank, args)

    ##### important!!!
    # make sure different processes sample different rays
    np.random.seed((rank + 1) * 777)
    # make sure different processes have different perturbations in depth samples
    torch.manual_seed((rank + 1) * 777)

    ##### only main process should do the logging
    if rank == 0:
        writer = SummaryWriter(os.path.join(args.basedir, 'summaries', args.expname))

    # start training
    what_val_to_log = 0             # helper variable for parallel rendering of a image
    what_train_to_log = 0

    loss_bce = torch.nn.BCELoss()


    for global_step in range(start+1, start+1+args.N_iters):
        time0 = time.time()
        scalars_to_log = OrderedDict()
        ### Start of core optimization loop
        scalars_to_log['resolution'] = ray_samplers[0].resolution_level
        # randomly sample rays and move to device
        i = np.random.randint(low=0, high=len(ray_samplers))

        if global_step == 120:
            print(126)

        # get rayo,rayd, seg centre and classlabel
        ray_batch = ray_samplers[i].random_sample_classifier(args.N_rand, \
                                                             args.front_sample, args.back_sample,\
                                                             center_crop=False, pretrain=args.donerf_pretrain)



        for key in ray_batch:
            if torch.is_tensor(ray_batch[key]):
                ray_batch[key] = ray_batch[key].to(rank)

        # r_o_np = ray_batch['ray_o'].cpu().numpy()
        # r_d_np = ray_batch['ray_d'].cpu().numpy()

        # forward and backward
        dots_sh = list(ray_batch['ray_d'].shape[:-1])  # number of rays
        all_rets = []

        net_oracle = models['net_oracle']
        optim_oracle = models['optim_oracle']


        if args.donerf_pretrain:

            if args.fg_bg_net:

                if not args.use_zval:
                    ret = net_oracle(ray_batch['ray_o'].float(), ray_batch['ray_d'].float(),
                                     ray_batch['fg_pts_flat'].float(), ray_batch['bg_pts_flat'].float(), \
                                     ray_batch['fg_far_depth'].float())
                else:
                    ret = net_oracle(ray_batch['ray_o'].float(), ray_batch['ray_d'].float(),
                                     ray_batch['fg_z_vals_centre'].float(), ray_batch['bg_z_vals_centre'].float(),\
                                     ray_batch['fg_far_depth'].float())

            else:

                if args.use_zval:
                    pts = torch.cat([ray_batch['fg_z_vals_centre'], ray_batch['bg_z_vals_centre']], dim=-1)
                else:
                    pts = torch.cat([ray_batch['fg_pts_flat'], ray_batch['bg_pts_flat']], dim=-1)
                ret = net_oracle(ray_batch['ray_o'].float(), ray_batch['ray_d'].float(), pts.float())

            # if args.fg_bg_net:

            label_fg = ray_batch['cls_label'][:, :args.front_sample]
            label_bg = ray_batch['cls_label'][:, args.front_sample:]

            if not args.fg_bg_net:
                ret['likeli_fg'] = ret['likeli'][:, :args.front_sample]
                ret['likeli_bg'] = ret['likeli'][:, args.front_sample:]
            #
            # if global_step > 100:
            #     print(global_step)
                # print('+++++++++++++++++++++++++++++++++fg+++++++++++++++++++++')
                # print(ret['likeli_fg'])
                # print('+++++++++++++++++++++++++++++++++brg+++++++++++++++++++++')
                # print(ret['likeli_bg'])


            loss_fg = loss_bce(ret['likeli_fg'], label_fg)
            loss_bg = loss_bce(ret['likeli_bg'], label_bg)

            loss_entr_fg = entropy_loss(ret['likeli_fg'])
            loss_entr_bg = entropy_loss(ret['likeli_bg'])




            scalars_to_log['bce_loss_fg'] = loss_fg.item()
            scalars_to_log['bce_loss_bg'] = loss_bg.item()

            scalars_to_log['entr_loss_fg'] = loss_entr_fg.item()
            scalars_to_log['entr_loss_bg'] = loss_entr_bg.item()

            out_likeli_fg = np.array(ret['likeli_fg'].cpu().detach().numpy())
            out_likeli_bg = np.array(ret['likeli_bg'].cpu().detach().numpy())

            metrics_fg = calculate_metrics(out_likeli_fg, np.array(label_fg.cpu().detach().numpy()),'micro')
            metrics_bg = calculate_metrics(out_likeli_bg, np.array(label_bg.cpu().detach().numpy()),'micro')
            for k in metrics_fg:
                scalars_to_log[k+'_fg'] = metrics_fg[k]

            for k in metrics_bg:
                scalars_to_log[k+'_bg'] = metrics_bg[k]

            # else:
            #     loss_cls = loss_bce(ret['likeli'], ray_batch['cls_label'])
            #
            #
            #     out_likeli = np.array(ret['likeli'].cpu().detach().numpy())
            #     cls_target = np.array(ray_batch['cls_label'].cpu().detach().numpy())
            #     metrics = calculate_metrics(out_likeli, cls_target ,
            #                                 'micro')
            #     for k in metrics:
            #         scalars_to_log[k] = metrics[k]

            if args.fg_bg_net:
                loss_cls = (loss_fg + loss_bg) + 100.0* (loss_entr_fg+loss_entr_bg)

            else:
                loss_cls = loss_bce(ret['likeli'], ray_batch['cls_label'])


            loss_cls.backward()
            optim_oracle.step()

            scalars_to_log['bce_loss'] = loss_cls.item()



        else:

            with torch.no_grad():

                if args.fg_bg_net:

                    if not args.use_zval:
                        ret = net_oracle(ray_batch['ray_o'].float(), ray_batch['ray_d'].float(),
                                         ray_batch['fg_pts_flat'].float(), ray_batch['bg_pts_flat'].float(), \
                                         ray_batch['fg_far_depth'].float())
                    else:
                        ret = net_oracle(ray_batch['ray_o'].float(), ray_batch['ray_d'].float(),
                                         ray_batch['fg_z_vals_centre'].float(), ray_batch['bg_z_vals_centre'].float(), \
                                         ray_batch['fg_far_depth'].float())

                else:

                    if args.use_zval:
                        pts = torch.cat([ray_batch['fg_z_vals_centre'], ray_batch['bg_z_vals_centre']], dim=-1)
                    else:
                        pts = torch.cat([ray_batch['fg_pts_flat'], ray_batch['bg_pts_flat']], dim=-1)
                    ret = net_oracle(ray_batch['ray_o'].float(), ray_batch['ray_d'].float(), pts.float())

            # results on different cascade levels
            for m in range(models['cascade_level']):
                optim = models['optim_{}'.format(m)]
                net = models['net_{}'.format(m)]

                # sample depths
                N_samples = models['cascade_samples'][m]


                ## plot ## ## plot ## ## plot ## ## plot ##
                # plot_ray_batch(ray_batch)
                ## plot ## ## plot ## ## plot ## ## plot ##



                    # print(m, global_step)
                    # sample pdf and concat with earlier samples
                    # fg_weights = ret['fg_weights'].clone().detach()

                    # fg_depth_mid = .5 * (fg_depth[..., 1:] + fg_depth[..., :-1])    # [..., N_samples-1]
                    # fg_weights = fg_weights[..., 1:-1]                              # [..., N_samples-2]

                fg_depth_mid = ray_batch['fg_z_vals_centre']



                # fg_depth_mid = torch.cat([torch.zeros_like(fg_depth_mid[..., 0:1]), fg_depth_mid], dim=-1)

                fg_weights = ret['likeli_fg'][:, 2:args.front_sample].clone().detach()

                fg_weights[fg_depth_mid[:,:-1]<0.] = 0.

                fg_depth,_ = torch.sort(sample_pdf(bins=fg_depth_mid, weights=fg_weights,
                                              N_samples=N_samples, det=False))    # [..., N_samples]
                fg_depth[fg_depth<0.] = 0.


                if not args.train_box_only:
                    # sample pdf and concat with earlier samples
                    # bg_weights = ret['bg_weights'].clone().detach()
                    # bg_depth_mid = .5 * (bg_depth[..., 1:] + bg_depth[..., :-1])
                    # bg_weights = bg_weights[..., 1:-1]                              # [..., N_samples-2]

                    bg_depth_mid = ray_batch['bg_z_vals_centre']
                    bg_weights = ret['likeli_bg'][:, 1: args.back_sample-1].clone().detach()

                    bg_weights[bg_depth_mid[:,:-1]<0.] =0.

                    bg_depth,_ = torch.sort(sample_pdf(bins=bg_depth_mid, weights=bg_weights,
                                                  N_samples=N_samples, det=False))    # [..., N_samples]

                    bg_depth[bg_depth<0.] = 0.



                optim.zero_grad()
                if not args.have_box:
                    ret = net(ray_batch['ray_o'], ray_batch['ray_d'], ray_batch['fg_far_depth'], fg_depth, bg_depth, img_name=ray_batch['img_name'])
                elif not args.train_box_only:
                    ret = net(ray_batch['ray_o'], ray_batch['ray_d'], ray_batch['fg_far_depth'], fg_depth, bg_depth, ray_batch['box_loc'], img_name=ray_batch['img_name'])
                else:
                    ret = net(ray_batch['ray_o'], ray_batch['ray_d'], ray_batch['fg_far_depth'], fg_depth,
                              img_name=ray_batch['img_name'])
                #writer.add_graph(net, [ray_batch['ray_o'], ray_batch['ray_d'], fg_far_depth, fg_depth, bg_depth, ray_batch['box_loc']])

                all_rets.append(ret)

                rgb_gt = ray_batch['rgb'].to(rank)
                rgb_loss = img2mse(ret['rgb'], rgb_gt)
                if args.depth_training:
                    depth_gt = ray_batch['depth_gt'].to(rank)
                    depth_pred = ret['depth_fgbg']
                    #mask =  torch.tensor(0. * np.ones(depth_pred.shape).astype(np.float32)).cuda()
                    inds = torch.where(depth_gt < 2001.)
                    d_pred_map = depth_pred[inds]
                    d_gt_map = depth_gt[inds]

                    depth_loss = dep_l1l2loss(torch.div(1.,d_pred_map), torch.div(1.,d_gt_map), l1l2 = 'l1')
                    #reg_loss = dep_l1l2loss(torch.div(1.,d_pred_map[:512])-torch.div(1.,d_pred_map[512:]), torch.div(1.,d_gt_map[:512])-torch.div(1.,d_gt_map[512:]), l1l2 = 'l1')
                    loss = rgb_loss * 0.  +  depth_loss
                    #scalars_to_log['level_{}/reg_loss'.format(m)] = reg_loss.item()

                    #loss = rgb_loss * 0 +  depth_loss
                    scalars_to_log['level_{}/depth_loss'.format(m)] = depth_loss.item()

                else:
                    loss = rgb_loss

                # print(global_step)
                scalars_to_log['level_{}/loss'.format(m)] = rgb_loss.item()
                scalars_to_log['level_{}/pnsr'.format(m)] = mse2psnr(rgb_loss.item())
                # print('before backward')
                loss.backward()
                # print('before step')
                optim.step()

                #writer.add_graph(net, [ray_batch['ray_o'], ray_batch['ray_d'], fg_far_depth, fg_depth, bg_depth, ray_batch['box_loc']])

                # # clean unused memory
                # torch.cuda.empty_cache()


        ### end of core optimization loop
        dt = time.time() - time0
        scalars_to_log['iter_time'] = dt

        ### only main process should do the logging
        if rank == 0 and (global_step % args.i_print == 0 or global_step < 10):
            # print('=======likeli=====')
            # print(out_likeli_fg)
            # print(out_likeli_bg)
            logstr = '{} step: {} '.format(args.expname, global_step)
            for k in scalars_to_log:
                logstr += ' {}: {:.6f}'.format(k, scalars_to_log[k])
                writer.add_scalar(k, scalars_to_log[k], global_step)
            logger.info(logstr)

        ## each process should do this; but only main process merges the results
        if global_step % args.i_img == 0 or global_step == start+1:

            select_inds = np.random.choice(640*360, size=(200,), replace=False)

            #### critical: make sure each process is working on the same random image
            time0 = time.time()
            idx = what_val_to_log % len(val_ray_samplers)
            log_data, label = render_single_image(rank, args.world_size, models, val_ray_samplers[idx], args.chunk_size, \
                                           args.train_box_only, have_box=args.have_box, donerf_pretrain=args.donerf_pretrain,\
                                           front_sample=args.front_sample, back_sample=args.back_sample, fg_bg_net=args.fg_bg_net, use_zval=args.use_zval)

            what_val_to_log += 1
            dt = time.time() - time0
            if rank == 0:    # only main process should do this
                logger.info('Logged a random validation view in {} seconds'.format(dt))

                if args.donerf_pretrain:
                    # if args.fg_bg_net:

                    label_fg = label[:, :args.front_sample]
                    label_bg = label[:, args.front_sample:]


                    pred_fg = torch.reshape(log_data[0]['likeli_fg'], (360*640,-1))
                    pred_bg = torch.reshape(log_data[0]['likeli_bg'], (360*640,-1))



                    loss_fg = loss_bce(pred_fg.detach().cpu(),  label_fg.detach().cpu())

                    loss_bg = loss_bce(pred_bg.detach().cpu(), label_bg.detach().cpu())

                    loss_entr_fg = entropy_loss(log_data[0]['likeli_fg'])
                    loss_entr_bg = entropy_loss(log_data[0]['likeli_bg'])


                    if args.fg_bg_net:
                        loss_cls = (loss_fg + loss_bg) + 100.0* ( loss_entr_fg + loss_entr_bg)
                    else:
                        pred = torch.reshape(log_data[0]['likeli'], (360 * 640, -1))
                        loss_cls = loss_bce(pred.detach().cpu(), label.detach().cpu())
                    scalars_to_log['val/bce_loss'] = loss_cls.item()
                    scalars_to_log['val/bce_loss_fg'] = loss_fg.item()
                    scalars_to_log['val/bce_loss_bg'] = loss_bg.item()
                    scalars_to_log['val/entr_loss_fg'] = loss_entr_fg.item()
                    scalars_to_log['val/entr_loss_bg'] = loss_entr_bg.item()

                    del loss_fg
                    del loss_bg

                    out_likeli_fg = np.array(pred_fg.cpu().detach().numpy())


                    metrics_fg = calculate_metrics(out_likeli_fg, np.array(label_fg.cpu().detach().numpy()),
                                                   'micro')

                    for k in metrics_fg:
                        scalars_to_log['val/'+ k + '_fg'] = metrics_fg[k]

                    # log_plot_conf_mat(writer, metrics_fg['cm'], global_step, 'val/CM_fg')

                    out_likeli_bg = np.array(pred_bg.cpu().detach().numpy())
                    metrics_bg = calculate_metrics(out_likeli_bg, np.array(label_bg.cpu().detach().numpy()),
                                                   'micro')

                    for k in metrics_bg:
                        scalars_to_log['val/'+ k + '_bg'] = metrics_bg[k]

                    # log_plot_conf_mat(writer, metrics_bg['cm'], global_step, 'val/CM_bg')
                    visualize_depth_label(writer, np.array(label_fg.cpu().detach().numpy())[select_inds], out_likeli_fg[select_inds], global_step, 'val/dVis_fg')
                    visualize_depth_label(writer, np.array(label_bg.cpu().detach().numpy())[select_inds], out_likeli_bg[select_inds], global_step, 'val/dVis_bg')

                    loss_deviation(writer, np.array(label_fg.cpu().detach().numpy())[select_inds],
                                   out_likeli_fg[select_inds], global_step, 'val/LossVis_fg')
                    loss_deviation(writer, np.array(label_bg.cpu().detach().numpy())[select_inds],
                                   out_likeli_bg[select_inds], global_step, 'val/LossVis_bg')


                    # else:
                    #     pred = torch.reshape(log_data[0]['likeli'], (360 * 640, -1))
                    #     loss_cls = loss_bce(pred.detach().cpu(), label.detach().cpu())
                    #     scalars_to_log['val/bce_loss'] = loss_cls.item()
                    #     metrics = calculate_metrics(np.array(pred.detach().cpu().numpy()),
                    #                                 np.array(label.detach().cpu().numpy()), 'micro')
                    #     for k in metrics:
                    #         scalars_to_log["val/" + k] = metrics[k]
                    #
                    #     visualize_depth_label(writer, np.array(label.cpu().detach().numpy())[select_inds], pred.cpu().detach().numpy()[select_inds],
                    #                           global_step, 'val/dVis')

                        # log_plot_conf_mat(writer, metrics['cm'], global_step, 'val/CM')



                    logstr = '[=VALIDATION=] {} step: {} '.format(args.expname, global_step)
                    for k in scalars_to_log:
                        logstr += ' {}: {:.6f}'.format(k, scalars_to_log[k])
                        writer.add_scalar(k, scalars_to_log[k], global_step)
                    logger.info(logstr)



                else:

                    #if args.depth_training:
                    log_view_to_tb(writer, global_step, log_data, gt_depth=ray_samplers[idx].get_depth(), gt_img=val_ray_samplers[idx].get_img(), mask=None, have_box=args.have_box, train_box_only=args.train_box_only, prefix='val/')


                    #else:
                     #   log_view_to_tb(writer, global_step, log_data, gt_img=val_ray_samplers[idx].get_img(), mask=None, have_box=args.have_box, train_box_only=args.train_box_only, prefix='val/')

            idx = what_val_to_log % len(val_ray_samplers)
            log_data, label = render_single_image(rank, args.world_size, models, ray_samplers[idx], args.chunk_size, \
                                                  args.train_box_only, have_box=args.have_box,
                                                  donerf_pretrain=args.donerf_pretrain, \
                                                  front_sample=args.front_sample, back_sample=args.back_sample,
                                                  fg_bg_net=args.fg_bg_net, use_zval=args.use_zval)

            what_val_to_log += 1
            dt = time.time() - time0
            if rank == 0:  # only main process should do this
                logger.info('Logged a random validation view in {} seconds'.format(dt))

                if args.donerf_pretrain:
                    # if args.fg_bg_net:

                    label_fg = label[:, :args.front_sample]
                    label_bg = label[:, args.front_sample:]

                    pred_fg = torch.reshape(log_data[0]['likeli_fg'], (360 * 640, -1))
                    pred_bg = torch.reshape(log_data[0]['likeli_bg'], (360 * 640, -1))

                    loss_fg = loss_bce(pred_fg.detach().cpu(), label_fg.detach().cpu())

                    loss_bg = loss_bce(pred_bg.detach().cpu(), label_bg.detach().cpu())

                    loss_entr_fg = entropy_loss(log_data[0]['likeli_fg'])
                    loss_entr_bg = entropy_loss(log_data[0]['likeli_bg'])

                    if args.fg_bg_net:
                        loss_cls = (loss_fg + loss_bg) + 100.0*(loss_entr_fg + loss_entr_bg)
                    else:
                        pred = torch.reshape(log_data[0]['likeli'], (360 * 640, -1))
                        loss_cls = loss_bce(pred.detach().cpu(), label.detach().cpu())

                    scalars_to_log['train/bce_loss'] = loss_cls.item()
                    scalars_to_log['train/bce_loss_fg'] = loss_fg.item()
                    scalars_to_log['train/bce_loss_bg'] = loss_bg.item()

                    scalars_to_log['train/entr_loss_fg'] = loss_entr_fg.item()
                    scalars_to_log['train/entr_loss_bg'] = loss_entr_bg.item()

                    del loss_fg
                    del loss_bg

                    out_likeli_fg = np.array(pred_fg.cpu().detach().numpy())

                    metrics_fg = calculate_metrics(out_likeli_fg, np.array(label_fg.cpu().detach().numpy()),
                                                   'micro')

                    for k in metrics_fg:
                        scalars_to_log['train/' + k + '_fg'] = metrics_fg[k]

                    # log_plot_conf_mat(writer, metrics_fg['cm'], global_step, 'val/CM_fg')

                    out_likeli_bg = np.array(pred_bg.cpu().detach().numpy())
                    metrics_bg = calculate_metrics(out_likeli_bg, np.array(label_bg.cpu().detach().numpy()),
                                                   'micro')

                    for k in metrics_bg:
                        scalars_to_log['train/' + k + '_bg'] = metrics_bg[k]

                    # log_plot_conf_mat(writer, metrics_bg['cm'], global_step, 'val/CM_bg')
                    visualize_depth_label(writer, np.array(label_fg.cpu().detach().numpy())[select_inds],
                                          out_likeli_fg[select_inds], global_step, 'train/dVis_fg')
                    visualize_depth_label(writer, np.array(label_bg.cpu().detach().numpy())[select_inds],
                                          out_likeli_bg[select_inds], global_step, 'train/dVis_bg')

                    loss_deviation(writer, np.array(label_fg.cpu().detach().numpy())[select_inds],
                                          out_likeli_fg[select_inds], global_step, 'train/LossVis_fg')
                    loss_deviation(writer, np.array(label_bg.cpu().detach().numpy())[select_inds],
                                          out_likeli_bg[select_inds], global_step, 'train/LossVis_bg')


                    # log_plot_conf_mat(writer, metrics_bg['cm'], global_step, 'train/CM_bg')

                    # else:
                    #     pred = torch.reshape(log_data[0]['likeli'], (360 * 640, -1))
                    #     loss_cls = loss_bce(pred.detach().cpu(), label.detach().cpu())
                    #     scalars_to_log['train/bce_loss'] = loss_cls.item()
                    #     metrics = calculate_metrics(np.array(pred.detach().cpu().numpy()),
                    #                                 np.array(label.detach().cpu().numpy()), 'micro')
                    #     for k in metrics:
                    #         scalars_to_log["train/" + k] = metrics[k]
                    #
                    #     # log_plot_conf_mat(writer, metrics['cm'], global_step, 'train/CM')
                    #     visualize_depth_label(writer, np.array(label.cpu().detach().numpy())[select_inds],
                    #                           pred.cpu().detach().numpy()[select_inds],
                    #                           global_step, 'train/dVis')

                    logstr = '[=VALID_TRAIN] {} step: {} '.format(args.expname, global_step)
                    for k in scalars_to_log:
                        logstr += ' {}: {:.6f}'.format(k, scalars_to_log[k])
                        writer.add_scalar(k, scalars_to_log[k], global_step)
                    logger.info(logstr)



                else:

                    # if args.depth_training:
                    log_view_to_tb(writer, global_step, log_data, gt_depth=ray_samplers[idx].get_depth(),
                                   gt_img=ray_samplers[idx].get_img(), mask=None, have_box=args.have_box,
                                   train_box_only=args.train_box_only, prefix='train/')

            # time0 = time.time()
            # idx = what_train_to_log % len(ray_samplers)
            # log_data, label = render_single_image(rank, args.world_size, models, ray_samplers[idx], \
            #                                args.chunk_size, args.train_box_only, have_box=args.have_box, \
            #                                donerf_pretrain=args.donerf_pretrain)
            # what_train_to_log += 1
            # dt = time.time() - time0
            # if rank == 0:   # only main process should do this
            #
            #     if args.donerf_pretrain:
            #         loss_cls = loss_bce(log_data[0]['likeli'].detach().cpu(), torch.reshape(label, (360, 640, 256)).detach().cpu())
            #         scalars_to_log['bce_loss'] = loss_cls.item()
            #         metrics = calculate_metrics(np.array(ret['likeli'].detach().cpu().numpy()),
            #                                     np.array(ray_batch['cls_label'].cpu().numpy()), 'micro')
            #         for k in metrics:
            #             scalars_to_log[k] = metrics[k]
            #         writer.add_scalar(k, scalars_to_log[k], global_step)
            #     else:
            #         logger.info('Logged a random training view in {} seconds'.format(dt))
            #         log_view_to_tb(writer, global_step, log_data, gt_img=ray_samplers[idx].get_img(), \
            #                        gt_depth=ray_samplers[idx].get_depth(), mask=None, have_box=args.have_box, \
            #                        train_box_only=args.train_box_only, prefix='train/')

            del log_data
            torch.cuda.empty_cache()

        if rank == 0 and (global_step % args.i_weights == 0 and global_step > 0):
            # saving checkpoints and logging
            fpath = os.path.join(args.basedir, args.expname, 'model_{:06d}.pth'.format(global_step))
            to_save = OrderedDict()

            to_save['net_oracle'] = models['net_oracle'].state_dict()
            to_save['optim_oracle'] = models['optim_oracle'].state_dict()

            for m in range(models['cascade_level']):
                name = 'net_{}'.format(m)
                to_save[name] = models[name].state_dict()

                name = 'optim_{}'.format(m)
                to_save[name] = models[name].state_dict()
            torch.save(to_save, fpath)

    # clean up for multi-processing
    cleanup()




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

    parser.add_argument("--max_freq_log2_pts", type=int, default=3,
                        help='log2 of max freq for positional encoding (seg points)')

    return parser


def train():
    parser = config_parser()
    args = parser.parse_args()
    logger.info(parser.format_values())

    if args.world_size == -1:
        args.world_size = torch.cuda.device_count()
        logger.info('Using # gpus: {}'.format(args.world_size))
    torch.multiprocessing.spawn(ddp_train_nerf,
                                args=(args,),
                                nprocs=args.world_size,
                                join=True)


if __name__ == '__main__':
    setup_logger()
    train()

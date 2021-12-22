import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim

from collections import OrderedDict
from ddp_model_mip import  DepthOracle, MipNerf
from nerf_network import WrapperModule

import time
from data_loader_split import load_data_split
import numpy as np
from tensorboardX import SummaryWriter
from utils import img2mse, mse2psnr, entropy_loss, crossEntropy, dep_l1l2loss, img_HWC2CHW, colorize, colorize_np,to8b, normalize_torch,TINY_NUMBER

from helpers import calculate_metrics, log_plot_conf_mat, visualize_depth_label, loss_deviation, get_box_transmittance_weight
import logging
import json

from helpers import plot_ray_batch
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
                        donerf_pretrain=False, box_number=10, box_size=1,
                        front_sample=128, back_sample=128,
                        fg_bg_net=True, use_zval=True,loss_type='bce',rank=0,
                        checkpoint=False, DEBUG=False):
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
        if checkpoint:
            chunk_ret = torch.utils.checkpoint.checkpoint(render_rays, 
                                          models, _rays,
                                        have_box, front_sample, back_sample,
                                       box_number, box_props)
        else:
            chunk_ret = render_rays(models, _rays,
                                    have_box, front_sample, back_sample,
                                    box_number, box_props)

    
        rgbs.append(chunk_ret['rgb'])
        depths.append(chunk_ret['depth'])

        if DEBUG:
            likelis_fg.append(chunk_ret['likeli_fg'])
            likelis_bg.append(chunk_ret['likeli_bg'])
            rgbs_fg.append(chunk_ret['fg_rgb'])
            depths_fg.append(chunk_ret['fg_depth'])

            rgbs_bg.append(chunk_ret['bg_rgb'])
            depths_bg.append(chunk_ret['bg_depth'])
            depths_lamda.append(chunk_ret['bg_lambda'])

    rgb = torch.cat(rgbs).view(ray_sampler.H, ray_sampler.W, -1).squeeze()
    d = torch.cat(depths).view(ray_sampler.H, ray_sampler.W, -1).squeeze()

    if DEBUG:
        others = {}
        rgb_fg = torch.cat(rgbs_fg).view(ray_sampler.H, ray_sampler.W, -1).squeeze()
        d_fg = torch.cat(depths_fg).view(ray_sampler.H, ray_sampler.W, -1).squeeze()
        others['rgb_fg'] = rgb_fg
        others['d_fg'] = d_fg


        rgb_bg = torch.cat(rgbs_bg).view(ray_sampler.H, ray_sampler.W, -1).squeeze()
        d_bg = torch.cat(depths_bg).view(ray_sampler.H, ray_sampler.W, -1).squeeze()
        d_lam = torch.cat(depths_lamda).view(ray_sampler.H, 
                                             ray_sampler.W, -1).squeeze()
        others['rgb_bg'] = rgb_bg
        others['d_bg'] = d_bg
        others['d_lam'] = d_lam

        likeli_fg = torch.cat(likelis_fg).view(ray_sampler.H
                                               * ray_sampler.W, -1).squeeze()
        likeli_bg = torch.cat(likelis_bg).view(ray_sampler.H
                                               * ray_sampler.W, -1).squeeze()

        return rgb, d, likeli_fg, likeli_bg, rays['cls_label'], others

    return rgb, d, None, None, rays['cls_label'], None

def eval_oracle(rays, net_oracle, front_sample, back_sample, have_box):


    if have_box:
        ret = net_oracle(rays['ray_o'], rays['ray_d'],
                         rays['fg_pts_flat'], rays['bg_pts_flat'],
                         rays['fg_far_depth'])

    else:
        ret = net_oracle(rays['ray_o'], rays['ray_d'],
                 rays['fg_pts_flat'], rays['bg_pts_flat'],
                 rays['fg_far_depth'])

    return ret

def get_depths(data, front_sample, back_sample,
                fg_z_vals_fence, bg_z_vals_fence,
                samples, box_weights):



    fg_weights = data['likeli_fg'].clone() # Avoid inplace ops
    bg_weights = data['likeli_bg'].clone()

    fg_weights = torch.sigmoid(fg_weights)
    fg_weights = fg_weights.clone()
    if box_weights is not None:
        fg_weights = fg_weights + box_weights

    fg_weights[fg_z_vals_fence < 0.0002] = float(0.0)
    fg_depth,_ = torch.sort(sample_pdf(bins=fg_z_vals_fence, 
                                    weights=fg_weights[:, :front_sample-1],
                                    N_samples=samples+1, det=True))  # [..., N_samples
    fg_depth = fg_depth.clone()
    fg_depth[fg_depth<0.0002] = float(0.0002)
    
    
#     ##
#     fg_weights = ret['likeli_fg']
#     fg_weights = torch.sigmoid(fg_weights).clone().detach()
#     fg_weights[ray_batch['fg_z_vals_fence']<0.0002] = 0.

#     # in mipnerf we are getting fencepoint or segbound from this
#     # [..., N_samples]  # these are all fencepoints, should get N_samples + 1 to get N_Samples intervals
#     fg_depth,_ = torch.sort(sample_pdf(bins=perturbed_seg_bound_fg, 
#                             weights=fg_weights[:, :args.front_sample-1],
#                                   N_samples=N_samples+1, det=False))  
    
#     ##
    

    bg_weights = torch.fliplr(bg_weights)
    bg_weights = torch.sigmoid(bg_weights)[:, :back_sample-1]
    bg_depth,_ = torch.sort(sample_pdf(bins=bg_z_vals_fence, weights=bg_weights,
                          N_samples=samples+1, det=True))  # [..., N_samples]

    return fg_depth, bg_depth


def render_rays(models, rays, have_box,
                front_sample, back_sample, box_number=None, box_props=None):

    """Render a set of rays using specific config."""

    # forward and backward

    ray_o = rays['ray_o']
    ray_d = rays['ray_d']
    box_loc = rays['box_loc'] if have_box else None
    radii = rays['radii']

#     box_props = box_props if box_props is not None \
#                     else torch.ones([box_number,9]).type_as(box_loc)

    fg_z_vals_fence = rays['fg_z_vals_fence']
    bg_z_vals_fence = rays['bg_z_vals_fence']

    fg_far_depth = rays['fg_far_depth']
    net_oracle = models['net_oracle']

    ret_or = eval_oracle(rays, net_oracle, front_sample, back_sample, have_box)
    

    net = models['net_0']
    samples = models['cascade_samples'][0]

    box_weights = None
  

    if box_loc is not None:
        box_weights = get_box_transmittance_weight(box_loc=box_loc.float(),
                                     fg_z_vals=fg_z_vals_fence,  ray_d=ray_d,
                                     ray_o=ray_o,
                                     fg_depth=fg_far_depth, box_number=box_number, 
                                     box_props=box_props)
        ret['likeli_fg'] = torch.sigmoid(ret_or['likeli_fg']) + (box_weights)

    fg_depth, bg_depth = get_depths(ret_or, front_sample, back_sample,
                                    fg_z_vals_fence, bg_z_vals_fence,
                                    samples, box_weights)

    if not have_box:
        ret = net(ray_o, ray_d, fg_far_depth, fg_depth, bg_depth, radii)
    else:
        ret = net(ray_o, ray_d, fg_far_depth, fg_depth, bg_depth, box_loc, box_props)

    ret['likeli_fg'] = ret_or['likeli_fg']
    ret['likeli_bg'] = ret_or['likeli_bg']
    
    return ret

def log_view_to_tb(writer, global_step, rgb_predict, depth_predict, 
                   gt_img, mask, gt_depth=None, 
                   have_box=False, box_seg_mask=None, prefix='', 
                   DEBUG=False, others=None):
    rgb_im = img_HWC2CHW(torch.from_numpy(gt_img))
    writer.add_image(prefix + 'rgb_gt', rgb_im, global_step)

    loss = img2mse(rgb_predict.cpu().reshape(-1),
                   torch.from_numpy(gt_img).reshape(-1)).item()
    writer.add_scalar(prefix + 'rgb_loss', loss, global_step)
    writer.add_scalar( prefix + 'psnr', mse2psnr(loss), global_step)

    depth_clip = 100.
    if gt_depth is not None:
        ##### THIS IS THE DEPTH OUTPUT, HxW, value is meters away from camera centre
        gt_depth[gt_depth > depth_clip] = depth_clip  
        depth_im = img_HWC2CHW(colorize(gt_depth, cmap_name='jet',
                                        append_cbar=True, mask=mask, is_np=True))
        writer.add_image(prefix + 'depth_gt', depth_im, global_step)

    if box_seg_mask is not None:
        box_seg_mask_im = img_HWC2CHW(colorize(box_seg_mask, cmap_name='jet', 
                                               append_cbar=True, mask=mask, is_np=True))
        writer.add_image(prefix + 'level_{}/boxsegmask_gt'.format(0),
                         box_seg_mask_im, global_step)


    rgb_im = img_HWC2CHW(rgb_predict)
    rgb_im = torch.clamp(rgb_im, min=0., max=1.)  # just in case diffuse+specular>1
    writer.add_image(prefix + 'level_0/rgb', rgb_im, global_step)

    depth_predict[depth_predict > depth_clip] = depth_clip
    depth_im = img_HWC2CHW(colorize(depth_predict, cmap_name='jet', 
                                    append_cbar=True,
                                    mask=mask))
    writer.add_image(prefix + 'level_{0}/depth_fgbg', depth_im, global_step)
    
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
        depth_im = img_HWC2CHW(colorize(depth, cmap_name='jet', 
                                        append_cbar=True, mask=mask))
        writer.add_image(prefix + 'level_{}/bg_depth'.format(0), depth_im, global_step)

        bg_lambda = others['d_lam']
        bg_lambda_im = img_HWC2CHW(colorize(bg_lambda, cmap_name='jet', append_cbar=True,
                                            mask=mask))
        writer.add_image(prefix + 'level_{}/bg_lambda'.format(0),
                         bg_lambda_im, global_step)


def create_nerf(rank, args):
    ###### create network and wrap in ddp; each process should do this
    # fix random seed just to make sure the network is initialized with same weights at different processes
    torch.manual_seed(777)

    models = OrderedDict()
    models['cascade_level'] = 1
    models['cascade_samples'] = [int(x.strip()) for x in args.cascade_samples.split(',')]

    img_names = None


    ora_net = DepthOracle(args).to(rank)
    models['net_oracle'] = WrapperModule(ora_net)
    
    net = MipNerf(args).to(rank)
    optim = torch.optim.Adam(net.parameters(), lr=args.lrate)
    models['net_0'] = net
    models['optim_0'] = optim


    start = -1

    ###### load pretrained weights; each process should do this
    if (args.ckpt_path is not None) and (os.path.isfile(args.ckpt_path)):
        ckpts = [args.ckpt_path]
    else:
        ckpts = [os.path.join(args.basedir, args.expname, f)
                 for f in sorted(os.listdir(os.path.join(args.basedir, args.expname)))\
                 if f.endswith('.pth')]

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
        to_load = torch.load(fpath)

      
        models['net_oracle'].load_state_dict(to_load['net_oracle'])

#         for m in range(models['cascade_level']):
#             for name in ['net_{}'.format(m), 'optim_{}'.format(m)]:
#                  models[name].load_state_dict(to_load[name])
                    

    elif not args.have_box:

        start = 0
        fpath_dep =\
        "/home/sally/nerf_clone/nerfpp_depth/logs/box_only_train_norm3_K=9Z=9_bg127/model_340000.pth"
        
        to_load_dep = torch.load(fpath_dep)

        models['net_oracle'].load_state_dict(to_load_dep['net_oracle'])

    start = 0
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

    ray_samplers = load_data_split(args.datadir, args.scene, split='train',
                                   have_box=args.have_box,
                                   train_depth=True, train_seg=args.train_seg, 
                                   box_number=args.box_number)
    val_ray_samplers = load_data_split(args.datadir, args.scene, split='test',
                                       skip=args.testskip, have_box=args.have_box,
                                       train_depth=True , train_seg=args.train_seg, 
                                       box_number=args.box_number)


    ###### create network and wrap in ddp; each process should do this
    start, models = create_nerf(rank, args)
    
    np.random.seed((rank + 1) * 777)
    torch.manual_seed((rank + 1) * 777)

    writer = SummaryWriter(os.path.join(args.basedir, 'summaries', args.expname))

    # start training
    what_val_to_log = 0             #
    what_train_to_log = 0

    for global_step in range(start+1, start+1+args.N_iters):

        time0 = time.time()
        scalars_to_log = OrderedDict()
        i = np.random.randint(low=0, high=len(ray_samplers))

        ray_batch = ray_samplers[i].random_sample_classifier(args.N_rand,
                                                             args.front_sample, 
                                                             args.back_sample,
                                                              rank=rank)
        for key in ray_batch:
            if torch.is_tensor(ray_batch[key]):
                ray_batch[key] = ray_batch[key].to(rank)
#                 print('key', ray_batch[key].type())
        
        net_oracle = models['net_oracle']
        with torch.no_grad():
            # pass midpoints to oracle s-1 ---> s
            ret = net_oracle(ray_batch['ray_o'], ray_batch['ray_d'],
                             ray_batch['fg_pts_flat'],
                             ray_batch['bg_pts_flat'], # these points are flipped 
                             ray_batch['fg_far_depth'])
            
     

        # sample depths
        N_samples = models['cascade_samples'][0]
         #fg/bg_z_vals_centre is actually the z_vals of seg bounds. pts are mid pints
        perturbed_seg_bound_fg = perturb_samples(ray_batch['fg_z_vals_fence'])
        perturbed_seg_bound_bg = perturb_samples(ray_batch['bg_z_vals_fence'])#0-->1


        fg_weights = ret['likeli_fg']
        fg_weights = torch.sigmoid(fg_weights).clone().detach()
        fg_weights[ray_batch['fg_z_vals_fence']<0.0002] = 0.

        # in mipnerf we are getting fencepoint or segbound from this
        # [..., N_samples]  # these are all fencepoints, should get N_samples + 1 to get N_Samples intervals
        fg_depth,_ = torch.sort(sample_pdf(bins=perturbed_seg_bound_fg, 
                                weights=fg_weights[:, :args.front_sample-1],
                                      N_samples=N_samples+1, det=False))   

        fg_depth[fg_depth<0.0002] = 0.0002
        bg_weights = torch.sigmoid(ret['likeli_bg'])[:, :args.back_sample-1].clone().detach()
        bg_weights = torch.fliplr(bg_weights)
        bg_depth,_ = torch.sort(sample_pdf(bins=perturbed_seg_bound_bg,
                                           weights=bg_weights,
                                      N_samples=N_samples+1, det=False))    

        net = models['net_0']
        optim = models['optim_0']
        optim.zero_grad()
        if not args.have_box:
            ret = net(ray_batch['ray_o'], ray_batch['ray_d'], 
                      ray_batch['fg_far_depth'], fg_depth, bg_depth, 
                      ray_batch['radii'])

        rgb_gt = ray_batch['rgb'].to(rank)
        rgb_loss = img2mse(ret['rgb'], rgb_gt)
        loss = rgb_loss
        
        # import ipdb; ipdb.set_trace()

        if args.depth_training:
            depth_gt = ray_batch['depth_gt'].to(rank)
            depth_pred = ret['depth']

            inds = torch.where(depth_gt < 2001.)
            d_pred_map = depth_pred[inds]
            d_gt_map = depth_gt[inds]

            depth_loss = dep_l1l2loss(torch.div(1.,d_pred_map+TINY_NUMBER),
                                      torch.div(1.,d_gt_map+TINY_NUMBER), l1l2 = 'l1')
            loss += 0.1 * depth_loss
            #loss = rgb_loss * 0 +  depth_loss
            scalars_to_log['level_0/depth_loss'] = depth_loss.item()


        # if args.seg_box_loss:
        #     seg_box_gt = ray_batch['seg_map_box']
        #     seg_prediction = ret['box_weights']
        #     seg_box_loss =  dep_l1l2loss(seg_prediction, seg_box_gt, l1l2 = 'l2')
        #     loss += seg_box_loss
        #     scalars_to_log['level_0/seg_box_loss'] = seg_box_loss.item()

        print(global_step)
        scalars_to_log['level_0/rgb_loss'] = rgb_loss.item()
        scalars_to_log['level_0/pnsr'] = mse2psnr(rgb_loss.item())
        # print('before backward')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.0001) 
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
            print(logstr)

#         # ## each process should do this; but only main process merges the results
#         if global_step % args.i_img == 0 or global_step == start+1:

#             # select_inds = np.random.choice(640*360, size=(200,), replace=False)

#             select_inds = np.arange(640*280,640*281)
#             select_inds2 = np.arange(640*240,640*241)

#             #### critical: make sure each process is working on the same random image
#             time0 = time.time()
#             idx = what_val_to_log % len(val_ray_samplers)

#             print('IMAGE_ID: {}'.format(idx))

#             with torch.no_grad():
#                 rgb, d, pred_fg, pred_bg, label, others  =\
#                         render_single_image(models, 
#                                             val_ray_samplers[idx], args.chunk_size,
#                                          have_box=args.have_box,                                         
#                                             front_sample=args.front_sample, 
#                                             back_sample=args.back_sample,                                          
#                                             rank=rank, DEBUG=True, 
#                                             box_number=args.box_number, 
#                                             box_size=args.box_size)




#             what_val_to_log += 1
#             dt = time.time() - time0

#             logger.info('Logged a random validation view in {} seconds'.format(dt))

      
              
#             log_view_to_tb(writer, global_step, rgb,d , gt_depth=val_ray_samplers[idx].get_depth(), box_seg_mask=val_ray_samplers[idx].get_box_mask(),
#                            gt_img=val_ray_samplers[idx].get_img(), mask=None, have_box=args.have_box,
#                            prefix='val/',DEBUG=True, others=others)


#             idx = what_train_to_log % len(ray_samplers)
#             print('IMAGE_ID-train: {}'.format(idx))
#             time0 = time.time()

#             with torch.no_grad():
#                 rgb, d, pred_fg, pred_bg, label,others = render_single_image(models, ray_samplers[idx], args.chunk_size,
#                                                       have_box=args.have_box,                                                     
#                                                       front_sample=args.front_sample, back_sample=args.back_sample,
#                                                        rank=rank, DEBUG=True, box_number=args.box_number, box_size=args.box_size)

#             what_train_to_log += 1
#             dt = time.time() - time0

#             logger.info('Logged a random validation view in {} seconds'.format(dt))



#             log_view_to_tb(writer, global_step, rgb,d, gt_depth=ray_samplers[idx].get_depth(), 
#                            box_seg_mask=ray_samplers[idx].get_box_mask(),
#                            gt_img=ray_samplers[idx].get_img(), mask=None, have_box=args.have_box,
#                             prefix='train/', DEBUG=True, others=others)





#             del rgb
#             del d
#             del pred_fg
#             del pred_bg
#             del others
#             torch.cuda.empty_cache()

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
   
    # logging/saving options
    parser.add_argument("--i_print", type=int, default=100, help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img", type=int, default=500, help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, help='frequency of weight ckpt saving')

    parser.add_argument("--have_box", action='store_true',
                        help='whether use box location in model')

    parser.add_argument("--train_seg", action='store_true',
                        help='use segmentation mask to prioritize training')
    
    parser.add_argument("--depth_training", action='store_true',
                        help='use depth image for training')

    parser.add_argument("--seg_box_loss", action='store_true',
                        help='add segmentation box loss')

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

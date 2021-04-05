import torch
import torch.nn as nn
import torch.optim
import torch.distributed
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing
import os
from collections import OrderedDict
from ddp_model import NerfNetWithAutoExpo, NerfNetBoxWithAutoExpo, NerfNetBoxOnlyWithAutoExpo
from nerf_network import WrapperModule
import time
from data_loader_split import load_data_split
import numpy as np
from tensorboardX import SummaryWriter
from utils import img2mse, mse2psnr, img_HWC2CHW, colorize, TINY_NUMBER
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

def checknan(*args, **kwargs):
    for i, arg in enumerate(args):
        if torch.is_tensor(arg) and torch.any(torch.isnan(arg)):
            raise ValueError("Found nan at index {:d} in arg".format(i))

    for k, v in kwargs.items():
        if torch.is_tensor(v) and torch.any(torch.isnan(v)):
            raise ValueError("Found nan in {}".format(k))


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

@torch.no_grad()
def eval_fg_pass(ray_o, ray_d, box_loc, fg_net, N_samples, fg_far_depth, fg_query_depths, bg_depth):
    ret = fg_net(ray_o, ray_d, fg_far_depth, fg_query_depths, bg_depth, box_loc)

    return ret

@torch.no_grad()
def eval_pass(ray_o, ray_d, box_loc, net, samples, fg_weights, bg_weights, fg_far_depth, fg_depth, bg_depth):
    fg_weights = fg_weights.detach()
    fg_depth_mid = .5 * (fg_depth[..., 1:] + fg_depth[..., :-1])    # [..., N_samples-1]
    checknan(fg_depth_mid)
    fg_weights = fg_weights[..., 1:-1]                              # [..., N_samples-2]
    fg_depth_samples = sample_pdf(bins=fg_depth_mid, weights=fg_weights,
                                  N_samples=samples, det=True)    # [..., N_samples]
    checknan(fg_depth_samples)
    fg_depth, _ = torch.sort(torch.cat((fg_depth, fg_depth_samples), dim=-1))

    # sample pdf and concat with earlier samples
    bg_weights = bg_weights.detach()
    bg_depth_mid = .5 * (bg_depth[..., 1:] + bg_depth[..., :-1])
    bg_weights = bg_weights[..., 1:-1]                              # [..., N_samples-2]
    bg_depth_samples = sample_pdf(bins=bg_depth_mid, weights=bg_weights,
                                  N_samples=samples, det=True)    # [..., N_samples]
    bg_depth, _ = torch.sort(torch.cat((bg_depth, bg_depth_samples), dim=-1))

    return net(ray_o, ray_d, fg_far_depth, fg_depth, bg_depth, box_loc), fg_depth, bg_depth,



#TODO: move to utils
def vlinspace(a, b, steps):
    return a.view(-1, 1) + (b - a).view(-1, 1) / (steps - 1) * torch.arange(steps, device=a.device)

def render_single_image(models, ray_sampler):
    ##### parallel rendering of a single image
    rays = ray_sampler.get_all()


    # forward and backward
    ret_merge_chunk = [OrderedDict() for _ in range(models['cascade_level'])]
    ray_o = rays['ray_o']
    ray_d = rays['ray_d']
    min_depth = rays['min_depth']
    box_loc = rays['box_loc']

    checknan(ray_o, ray_d, ray_d, min_depth, box_loc)


    fg_net = models['net_0']
    fg_samples = models['cascade_samples'][0]

    fg_far_depth = intersect_sphere(ray_o, ray_d)  # [...,]
    checknan(fg_far_depth)
    fg_near_depth = min_depth  # [..., ]
    fg_query_depths = vlinspace(fg_near_depth,
                                fg_far_depth, fg_samples).to(ray_o.device)

    dots_sh = list(ray_d.shape[:-1])
    bg_depth = torch.linspace(0., 1., fg_samples).view(
        [1, ] * len(dots_sh) + [fg_samples,]).expand(
                                        dots_sh + [fg_samples,]).to(ray_o.device)


    ret = eval_fg_pass(ray_o, ray_d, box_loc, fg_net, fg_samples,
    fg_far_depth, fg_query_depths, bg_depth)

    del fg_near_depth
    torch.cuda.empty_cache()

    checknan(**ret)

    for m in range(models['cascade_level']):
        if m == 0:
            continue

        net = models['net_{}'.format(m)]
        samples = models['cascade_samples'][m]

        ret, fg_query_depths, bg_depth = eval_pass(ray_o, ray_d, box_loc, net, samples,
                                            ret['fg_weights'], ret['bg_weights'],
                                            fg_far_depth, fg_query_depths, bg_depth)
        checknan(**ret)

    return ret


def log_view_to_tb(writer, global_step, log_data, gt_img, mask, train_box_only= False, prefix=''):
    rgb_im = img_HWC2CHW(torch.from_numpy(gt_img))
    writer.add_image(prefix + 'rgb_gt', rgb_im, global_step)

    for m in range(len(log_data)):
        rgb_im = img_HWC2CHW(log_data[m]['rgb'])
        rgb_im = torch.clamp(rgb_im, min=0., max=1.)  # just in case diffuse+specular>1
        writer.add_image(prefix + 'level_{}/rgb'.format(m), rgb_im, global_step)

        rgb_im = img_HWC2CHW(log_data[m]['fg_rgb'])
        rgb_im = torch.clamp(rgb_im, min=0., max=1.)  # just in case diffuse+specular>1
        writer.add_image(prefix + 'level_{}/fg_rgb'.format(m), rgb_im, global_step)
        depth = log_data[m]['fg_depth']
        depth_im = img_HWC2CHW(colorize(depth, cmap_name='jet', append_cbar=True,
                                        mask=mask))
        writer.add_image(prefix + 'level_{}/fg_depth'.format(m), depth_im, global_step)


        if not train_box_only:
            rgb_im = img_HWC2CHW(log_data[m]['bg_rgb'])
            rgb_im = torch.clamp(rgb_im, min=0., max=1.)  # just in case diffuse+specular>1
            writer.add_image(prefix + 'level_{}/bg_rgb'.format(m), rgb_im, global_step)
            depth = log_data[m]['bg_depth']
            depth_im = img_HWC2CHW(colorize(depth, cmap_name='jet', append_cbar=True,
                                            mask=mask))
            writer.add_image(prefix + 'level_{}/bg_depth'.format(m), depth_im, global_step)
            bg_lambda = log_data[m]['bg_lambda']
            bg_lambda_im = img_HWC2CHW(colorize(bg_lambda, cmap_name='hot', append_cbar=True,
                                                mask=mask))
            writer.add_image(prefix + 'level_{}/bg_lambda'.format(m), bg_lambda_im, global_step)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    # port = np.random.randint(12355, 12399)
    # os.environ['MASTER_PORT'] = '{}'.format(port)
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    torch.distributed.destroy_process_group()


def create_nerf(args, cuda_device='cuda:0'):
    torch.manual_seed(777)

    models = OrderedDict()
    models['cascade_level'] = args.cascade_level
    models['cascade_samples'] = [int(x.strip()) for x in args.cascade_samples.split(',')]
    for m in range(models['cascade_level']):
        img_names = None
        if args.optim_autoexpo:
            # load training image names for autoexposure
            f = os.path.join(args.basedir, args.expname, 'train_images.json')
            with open(f) as file:
                img_names = json.load(file)
        if args.train_box_only:
            net = NerfNetBoxOnlyWithAutoExpo(args, optim_autoexpo=args.optim_autoexpo, img_names=img_names).to(cuda_device)
        elif args.have_box:
            net = NerfNetBoxWithAutoExpo(args, optim_autoexpo=args.optim_autoexpo, img_names=img_names).to(cuda_device)
        else:
            net = NerfNetWithAutoExpo(args, optim_autoexpo=args.optim_autoexpo, img_names=img_names).to(cuda_device)
        optim = torch.optim.Adam(net.parameters(), lr=args.lrate)
        models['net_{}'.format(m)] = WrapperModule(net)
        models['optim_{}'.format(m)] = optim

    start = -1

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
        to_load = torch.load(fpath, map_location='cpu')

        for m in range(models['cascade_level']):
            optim_name = 'optim_{}'.format(m)
            net_name = 'net_{}'.format(m)
            models[optim_name].load_state_dict(to_load[optim_name])
            models[net_name].load_state_dict(to_load[net_name])

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
        args.chunk_size = 8192
    else:
        logger.info('setting batch size according to 12G gpu')
        args.N_rand = 512
        args.chunk_size = 4096

    ###### Create log dir and copy the config file
    if rank == 0:
        os.makedirs(os.path.join(args.basedir, args.expname), exist_ok=True)
        f = os.path.join(args.basedir, args.expname, 'args.txt')
        with open(f, 'w') as file:
            for arg in sorted(vars(args)):
                attr = getattr(args, arg)
                file.write('{} = {}\n'.format(arg, attr))
        if args.config is not None:
            f = os.path.join(args.basedir, args.expname, 'config.txt')
            with open(f, 'w') as file:
                file.write(open(args.config, 'r').read())
    torch.distributed.barrier()

    ray_samplers = load_data_split(args.datadir, args.scene, split='train',
                                   try_load_min_depth=args.load_min_depth, have_box=args.have_box)
    val_ray_samplers = load_data_split(args.datadir, args.scene, split='validation',
                                       try_load_min_depth=args.load_min_depth, skip=args.testskip, have_box=args.have_box)

    # write training image names for autoexposure
    if args.optim_autoexpo:
        f = os.path.join(args.basedir, args.expname, 'train_images.json')
        with open(f, 'w') as file:
            img_names = [ray_samplers[i].img_path for i in range(len(ray_samplers))]
            json.dump(img_names, file, indent=2)

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
    for global_step in range(start+1, start+1+args.N_iters):
        time0 = time.time()
        scalars_to_log = OrderedDict()
        ### Start of core optimization loop
        scalars_to_log['resolution'] = ray_samplers[0].resolution_level
        # randomly sample rays and move to device
        i = np.random.randint(low=0, high=len(ray_samplers))
        ray_batch = ray_samplers[i].random_sample(args.N_rand, center_crop=False)
        for key in ray_batch:
            if torch.is_tensor(ray_batch[key]):
                ray_batch[key] = ray_batch[key].to(rank)

        # r_o_np = ray_batch['ray_o'].cpu().numpy()
        # r_d_np = ray_batch['ray_d'].cpu().numpy()

        # forward and backward
        dots_sh = list(ray_batch['ray_d'].shape[:-1])  # number of rays
        all_rets = []                                  # results on different cascade levels
        for m in range(models['cascade_level']):
            optim = models['optim_{}'.format(m)]
            net = models['net_{}'.format(m)]

            # sample depths
            N_samples = models['cascade_samples'][m]


            ## plot ## ## plot ## ## plot ## ## plot ##
            # plot_ray_batch(ray_batch)
            ## plot ## ## plot ## ## plot ## ## plot ##

            if m == 0:
                # print(m, global_step)
                # foreground depth
                fg_far_depth = intersect_sphere(ray_batch['ray_o'], ray_batch['ray_d'])  # [...,]
                fg_near_depth = ray_batch['min_depth']  # [..., ]
                step = (fg_far_depth - fg_near_depth) / (N_samples - 1)
                fg_depth = torch.stack([fg_near_depth + i * step for i in range(N_samples)], dim=-1)  # [..., N_samples]
                fg_depth = perturb_samples(fg_depth)   # random perturbation during training

                if not args.train_box_only:
                    # background depth
                    bg_depth = torch.linspace(0., 1., N_samples).view(
                                [1, ] * len(dots_sh) + [N_samples,]).expand(dots_sh + [N_samples,]).to(rank)
                    bg_depth = perturb_samples(bg_depth)   # random perturbation during training
            else:
                # print(m, global_step)
                # sample pdf and concat with earlier samples
                fg_weights = ret['fg_weights'].clone().detach()
                fg_depth_mid = .5 * (fg_depth[..., 1:] + fg_depth[..., :-1])    # [..., N_samples-1]
                fg_weights = fg_weights[..., 1:-1]                              # [..., N_samples-2]
                fg_depth_samples = sample_pdf(bins=fg_depth_mid, weights=fg_weights,
                                              N_samples=N_samples, det=False)    # [..., N_samples]
                fg_depth, _ = torch.sort(torch.cat((fg_depth, fg_depth_samples), dim=-1))

                if not args.train_box_only:
                    # sample pdf and concat with earlier samples
                    bg_weights = ret['bg_weights'].clone().detach()
                    bg_depth_mid = .5 * (bg_depth[..., 1:] + bg_depth[..., :-1])
                    bg_weights = bg_weights[..., 1:-1]                              # [..., N_samples-2]
                    bg_depth_samples = sample_pdf(bins=bg_depth_mid, weights=bg_weights,
                                                  N_samples=N_samples, det=False)    # [..., N_samples]
                    bg_depth, _ = torch.sort(torch.cat((bg_depth, bg_depth_samples), dim=-1))

            optim.zero_grad()
            if not args.train_box_only:
                ret = net(ray_batch['ray_o'], ray_batch['ray_d'], fg_far_depth, fg_depth, bg_depth, ray_batch['box_loc'], img_name=ray_batch['img_name'])
            else:
                ret = net(ray_batch['ray_o'], ray_batch['ray_d'], fg_far_depth, fg_depth,
                          img_name=ray_batch['img_name'])

            all_rets.append(ret)

            rgb_gt = ray_batch['rgb'].to(rank)
            if 'autoexpo' in ret:
                scale, shift = ret['autoexpo']
                scalars_to_log['level_{}/autoexpo_scale'.format(m)] = scale.item()
                scalars_to_log['level_{}/autoexpo_shift'.format(m)] = shift.item()
                # rgb_gt = scale * rgb_gt + shift
                rgb_pred = (ret['rgb'] - shift) / scale
                rgb_loss = img2mse(rgb_pred, rgb_gt)
                loss = rgb_loss + args.lambda_autoexpo * (torch.abs(scale-1.)+torch.abs(shift))
            else:
                rgb_loss = img2mse(ret['rgb'], rgb_gt)
                loss = rgb_loss
            # print(global_step)
            scalars_to_log['level_{}/loss'.format(m)] = rgb_loss.item()
            scalars_to_log['level_{}/pnsr'.format(m)] = mse2psnr(rgb_loss.item())
            # print('before backward')
            loss.backward()
            # print('before step')
            optim.step()

            # # clean unused memory
            # torch.cuda.empty_cache()

        ### end of core optimization loop
        dt = time.time() - time0
        scalars_to_log['iter_time'] = dt

        ### only main process should do the logging
        if rank == 0 and (global_step % args.i_print == 0 or global_step < 10):
            logstr = '{} step: {} '.format(args.expname, global_step)
            for k in scalars_to_log:
                logstr += ' {}: {:.6f}'.format(k, scalars_to_log[k])
                writer.add_scalar(k, scalars_to_log[k], global_step)
            logger.info(logstr)

        ### each process should do this; but only main process merges the results
        if global_step % args.i_img == 0 or global_step == start+1:
            #### critical: make sure each process is working on the same random image
            time0 = time.time()
            idx = what_val_to_log % len(val_ray_samplers)
            log_data = render_single_image(rank, args.world_size, models, val_ray_samplers[idx], args.chunk_size, args.train_box_only)
            what_val_to_log += 1
            dt = time.time() - time0
            if rank == 0:    # only main process should do this
                logger.info('Logged a random validation view in {} seconds'.format(dt))
                log_view_to_tb(writer, global_step, log_data, gt_img=val_ray_samplers[idx].get_img(), mask=None, train_box_only=args.train_box_only, prefix='val/')

            time0 = time.time()
            idx = what_train_to_log % len(ray_samplers)
            log_data = render_single_image(rank, args.world_size, models, ray_samplers[idx], args.chunk_size, args.train_box_only)
            what_train_to_log += 1
            dt = time.time() - time0
            if rank == 0:   # only main process should do this
                logger.info('Logged a random training view in {} seconds'.format(dt))
                log_view_to_tb(writer, global_step, log_data, gt_img=ray_samplers[idx].get_img(), mask=None, train_box_only=args.train_box_only, prefix='train/')

            del log_data
            torch.cuda.empty_cache()

        if rank == 0 and (global_step % args.i_weights == 0 and global_step > 0):
            # saving checkpoints and logging
            fpath = os.path.join(args.basedir, args.expname, 'model_{:06d}.pth'.format(global_step))
            to_save = OrderedDict()
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



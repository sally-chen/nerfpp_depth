import torch
import torch.nn as nn
import torch.optim
import torch.distributed
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing
import os
from collections import OrderedDict
from ddp_model import NerfNetWithAutoExpo, NerfNetBoxWithAutoExpo, NerfNetBoxOnlyWithAutoExpo
import time
from data_loader_split import load_data_split
import numpy as np
from tensorboardX import SummaryWriter
from utils import img2mse, mse2psnr, dep_l1l2loss, img_HWC2CHW, colorize, colorize_np,to8b, TINY_NUMBER
import logging
import json
from nerf_network import WrapperModule

from helpers import plot_ray_batch
# from graphviz import Digraph
from torch.autograd import Variable


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

def plot_mult_pose(poses_list, name, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #
    colors = ['b', 'r', 'c', 'm', 'y', 'b', 'r', 'c', 'm']
    for i in range(len(poses_list)):
        plot_single_pose(poses_list[i],colors[i], ax, labels[i])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.legend()
    plt.title(name)
    plt.savefig("./train_poses.png")

def plot_single_pose(poses, color, ax, label):
    # poses shape N, 3, 4
    ax.scatter(poses[:, 0, 3], poses[:, 1, 3], poses[:, 2, 3], marker='o', color=color, label=label)
    #
    # for i in range(poses.shape[0]):
    #     ax.plot([poses[i, 0, 3], poses[i, 0, 3] + poses[i, 0, 2]],
    #             [poses[i, 1, 3], poses[i, 1, 3] + poses[i, 1, 2]],
    #             [poses[i, 2, 3], poses[i, 2, 3] + poses[i, 2, 2]], color=color)

def draw_sphere(p,d,o):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import numpy as np
    from itertools import product, combinations

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # ax.set_aspect(1)


    # draw sphere
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="r")

    ax.scatter(o[:, 0], o[:, 1], o[:, 2], marker='o', color='green', label='rayo')

    for i in range(d.shape[0]):
        ax.plot([o[i, 0], o[i, 0] + d[i, 0]],
                [o[i, 1], o[i, 1] + d[i, 1]],
                [o[i, 2], o[i, 2] + d[i, 2]], color='blue')

    zo = np.zeros((3))
    for i in range(d.shape[0]):
        ax.plot([zo[0], zo[0] + p[i, 0]],
                [zo[1], zo[1]+ p[i, 1]],
                [zo[2], zo[2] + p[i, 2]], color='magenta')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.legend()
    plt.show()



def intersect_sphere(ray_o, ray_d):
    '''
    ray_o, ray_d: [..., 3]
    compute the depth of the intersection point between this ray and unit sphere
    '''
    # note: d1 becomes negative if this mid point is behind camera

    ray_d_np = ray_d.cpu().detach().numpy()

    d_norm = torch.sum(ray_d * ray_d, dim=-1)
    d_norm_np = d_norm.cpu().detach().numpy()

    d1 = -torch.sum(ray_d * ray_o, dim=-1) / d_norm
    d1_np = d1.cpu().detach().numpy()

    p = ray_o + d1.unsqueeze(-1) * ray_d
    p_np = p.cpu().detach().numpy()

    # consider the case where the ray does not intersect the sphere
    ray_d_cos = 1. / torch.norm(ray_d, dim=-1)
    ray_d_cos_np =   ray_d_cos.cpu().detach().numpy()

    p_norm =  torch.sum(p * p, dim=-1)
    p_norm_np = p_norm.cpu().detach().numpy()

    d2 = torch.sqrt(1. - p_norm) * ray_d_cos

    d2_np = d2.cpu().detach().numpy()

    # draw_sphere(p_np[1600:1606], ray_d_np[1600:1606], ray_o.cpu().detach().numpy()[1600:1606])

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

def render_single_image_noddp(models, ray_sampler, chunk_size, \
                        train_box_only= False,have_box=False):
    ray_batch = ray_sampler.get_all()

    # split into chunks and render inside each process
    ray_batch_split = OrderedDict()
    for key in ray_batch:
        if torch.is_tensor(ray_batch[key]):
            ray_batch_split[key] = torch.split(ray_batch[key].to(torch.cuda.current_device()), chunk_size)

    # forward and backward
    ret_merge_chunk = [OrderedDict() for _ in range(models['cascade_level'])]
    for s in range(len(ray_batch_split['ray_d'])):
        ray_o = ray_batch_split['ray_o'][s]
        ray_d = ray_batch_split['ray_d'][s]
        min_depth = ray_batch_split['min_depth'][s]
        if have_box:
            box_loc = ray_batch_split['box_loc'][s]

        dots_sh = list(ray_d.shape[:-1])

        ## temp
        # models['cascade_level'] = 1
        for m in range(models['cascade_level']):

            net = models['net_{}'.format(m)]
            # sample depths
            N_samples = models['cascade_samples'][m]
            if m == 0:
                # foreground depth
                fg_far_depth = intersect_sphere(ray_o, ray_d)  # [...,]
                fg_near_depth = min_depth  # [..., ]
                step = (fg_far_depth - fg_near_depth) / (N_samples - 1)
                fg_depth = torch.stack([fg_near_depth + i * step for i in range(N_samples)], dim=-1)  # [..., N_samples]

                if not train_box_only:
                    # background depth
                    bg_depth = torch.linspace(0., 1., N_samples).view(
                        [1, ] * len(dots_sh) + [N_samples, ]).expand(dots_sh + [N_samples, ]).to(torch.cuda.current_device())

                # delete unused memory
                del fg_near_depth
                del step
                torch.cuda.empty_cache()
            else:
                # sample pdf and concat with earlier samples
                fg_weights_ = ret['fg_weights'].clone()
                fg_weights = fg_weights_.detach()
                fg_depth_mid = .5 * (fg_depth[..., 1:] + fg_depth[..., :-1])  # [..., N_samples-1]
                fg_weights = fg_weights[..., 1:-1]  # [..., N_samples-2]
                fg_depth_samples = sample_pdf(bins=fg_depth_mid, weights=fg_weights,
                                              N_samples=N_samples, det=True)  # [..., N_samples]
                fg_depth, _ = torch.sort(torch.cat((fg_depth, fg_depth_samples), dim=-1))

                if not train_box_only:
                    # sample pdf and concat with earlier samples
                    bg_weights = ret['bg_weights'].clone().detach()
                    bg_depth_mid = .5 * (bg_depth[..., 1:] + bg_depth[..., :-1])
                    bg_weights = bg_weights[..., 1:-1]  # [..., N_samples-2]
                    bg_depth_samples = sample_pdf(bins=bg_depth_mid, weights=bg_weights,
                                                  N_samples=N_samples, det=True)  # [..., N_samples]
                    bg_depth, _ = torch.sort(torch.cat((bg_depth, bg_depth_samples), dim=-1))

                # delete unused memory
                del fg_weights
                del fg_depth_mid
                del fg_depth_samples

                if not train_box_only:
                    del bg_weights
                    del bg_depth_mid
                    del bg_depth_samples
                torch.cuda.empty_cache()

            with torch.no_grad():
                if not have_box:
                    ret = net(ray_o, ray_d, fg_far_depth, fg_depth, bg_depth)
                elif not train_box_only:

                    # net  = net.double()
                    # ret = net(ray_o.double, ray_d.double, fg_far_depth.double, fg_depth.double, bg_depth.double, box_loc.double)
                    ret = net(ray_o, ray_d, fg_far_depth, fg_depth, bg_depth,
                              box_loc)
                else:
                    ret = net(ray_o, ray_d, fg_far_depth, fg_depth)

            # import torchvision.models as models
            # print(y)

            # g = make_dot(ret)
            # g.view()


            for key in ret:
                if key not in ['fg_weights', 'bg_weights']:
                    if torch.is_tensor(ret[key]):
                        if key not in ret_merge_chunk[m]:
                            ret_merge_chunk[m][key] = [ret[key].cpu(), ]
                        else:
                            ret_merge_chunk[m][key].append(ret[key].cpu())

                        ret[key] = None

            # clean unused memory
            torch.cuda.empty_cache()


    # merge results from different chunks
    for m in range(len(ret_merge_chunk)):
        for key in ret_merge_chunk[m]:
            ret_merge_chunk[m][key] = torch.cat(ret_merge_chunk[m][key], dim=0).reshape(
                (ray_sampler.H, ray_sampler.W, -1)).squeeze()

    # only rank 0 program returns
    return ret_merge_chunk

def log_view_to_tb(writer, global_step, log_data, gt_img, mask, gt_depth=None, train_box_only= False, have_box=False, prefix=''):
    rgb_im = img_HWC2CHW(torch.from_numpy(gt_img))
    writer.add_image(prefix + 'rgb_gt', rgb_im, global_step)
    depth_clip = 100.
    if gt_depth is not None:


        gt_depth[gt_depth > depth_clip] = depth_clip  ##### THIS IS THE DEPTH OUTPUT, HxW, value is meters away from camera centre

        
        depth_im = img_HWC2CHW(colorize(gt_depth, cmap_name='jet', append_cbar=True,
                                                            mask=mask, is_np=True))
        writer.add_image(prefix + 'depth_gt', depth_im, global_step)

    for m in range(len(log_data)):
        rgb_im = img_HWC2CHW(log_data[m]['rgb'])
        rgb_im = torch.clamp(rgb_im, min=0., max=1.)  # just in case diffuse+specular>1
        writer.add_image(prefix + 'level_{}/rgb'.format(m), rgb_im, global_step)
        
        #if have_box:
        depth = log_data[m]['depth_fgbg']
        depth[depth > depth_clip] = depth_clip
        depth_im = img_HWC2CHW(colorize(depth, cmap_name='jet', append_cbar=True,
                                        mask=mask))
        writer.add_image(prefix + 'level_{}/depth_fgbg'.format(m), depth_im, global_step)



        #rgb_im = img_HWC2CHW(log_data[m]['fg_rgb'])
        #rgb_im = torch.clamp(rgb_im, min=0., max=1.)  # just in case diffuse+specular>1
        #writer.add_image(prefix + 'level_{}/fg_rgb'.format(m), rgb_im, global_step)
        
        depth = log_data[m]['fg_depth']
        depth_im = img_HWC2CHW(colorize(depth, cmap_name='jet', append_cbar=True,
                                         mask=mask))
        writer.add_image(prefix + 'level_{}/fg_depth'.format(m), depth_im, global_step)
        #
        #
        if not train_box_only:
           #rgb_im = img_HWC2CHW(log_data[m]['bg_rgb'])
           #rgb_im = torch.clamp(rgb_im, min=0., max=1.)  # just in case diffuse+specular>1
           #writer.add_image(prefix + 'level_{}/bg_rgb'.format(m), rgb_im, global_step)
           depth = log_data[m]['bg_depth']
           depth_im = img_HWC2CHW(colorize(depth, cmap_name='jet', append_cbar=True,
                                           mask=mask))
           writer.add_image(prefix + 'level_{}/bg_depth'.format(m), depth_im, global_step)
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

def create_nerf_noddp(args):
    torch.manual_seed(777)
    models = OrderedDict()
    models['cascade_level'] = args.cascade_level
    models['cascade_samples'] = [int(x.strip()) for x in args.cascade_samples.split(',')]

    for m in range(models['cascade_level']):
        img_names = None

        if args.train_box_only:
            net = NerfNetBoxOnlyWithAutoExpo(args, optim_autoexpo=args.optim_autoexpo, img_names=img_names)
        elif args.have_box:
            net = NerfNetBoxWithAutoExpo(args, optim_autoexpo=args.optim_autoexpo, img_names=img_names)
            # net = NerfNetWithAutoExpo(args, optim_autoexpo=args.optim_autoexpo, img_names=img_names).to(rank)
        else:
            net = NerfNetWithAutoExpo(args, optim_autoexpo=args.optim_autoexpo, img_names=img_names)

        optim = torch.optim.Adam(net.parameters(), lr=args.lrate)
        models['net_{}'.format(m)] = WrapperModule(net)
        models['optim_{}'.format(m)] = optim

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

    # ckpts = ['/home/sally/nerfpp_depth_test/logs/big_inters_norm15_comb_disp_reg/noddp_model_700000.pth']
    logger.info('Found ckpts: {}'.format(ckpts))
    if len(ckpts) > 0 and not args.no_reload:

        fpath = ckpts[-1]
        logger.info('Reloading from: {}'.format(fpath))
        to_load = torch.load(fpath)

        # TODO: ====
        # Add extra name module.*
        # for m in range(models['cascade_level']):
        #     # name: net_0 net_1 optim_0 optim_1
        #     new_dict = dict()
        #     for k, v in to_load['net_{}'.format(m)].items():
        #         new_dict['module.' + k] = v
        #     to_load['net_{}'.format(m)] = new_dict
                
            
        # ========================

        for m in range(models['cascade_level']):
            # name: net_0 net_1 optim_0 optim_1
            for name in ['net_{}'.format(m), 'optim_{}'.format(m)]:
                models[name].load_state_dict(to_load[name])
                if name.startswith('net'):
                    models[name].to(torch.cuda.current_device())



    return start, models


def create_nerf(rank, args):
    print('Use ddp :{}'.format(args.use_ddp))
    ###### create network and wrap in ddp; each process should do this
    # fix random seed just to make sure the network is initialized with same weights at different processes
    torch.manual_seed(777)
    # very important!!! otherwise it might introduce extra memory in rank=0 gpu
    torch.cuda.set_device(rank)

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
            net = NerfNetBoxOnlyWithAutoExpo(args, optim_autoexpo=args.optim_autoexpo, img_names=img_names).to(rank)
        elif args.have_box:
            net = NerfNetBoxWithAutoExpo(args, optim_autoexpo=args.optim_autoexpo, img_names=img_names).to(rank)
            # net = NerfNetWithAutoExpo(args, optim_autoexpo=args.optim_autoexpo, img_names=img_names).to(rank)
        else:
            net = NerfNetWithAutoExpo(args, optim_autoexpo=args.optim_autoexpo, img_names=img_names).to(rank)

        if args.use_ddp:
            net = DDP(net, device_ids=[rank], output_device=rank, find_unused_parameters=True)
        # net = DDP(net, device_ids=[rank], output_device=rank)
        optim = torch.optim.Adam(net.parameters(), lr=args.lrate)
        models['net_{}'.format(m)] = net
        models['optim_{}'.format(m)] = optim

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
        # configure map_location properly for different processes
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        to_load = torch.load(fpath, map_location=map_location)




        for m in range(models['cascade_level']):
            for name in ['net_{}'.format(m), 'optim_{}'.format(m)]:

                if not args.use_ddp:
                    if name.startswith('net'):
                        ddp_tmp = DDP(models[name])
                        ddp_tmp.load_state_dict(to_load[name])

                        models[name] = ddp_tmp.module

                    else:
                        models[name].load_state_dict(to_load[name])


                else:
                    models[name].load_state_dict(to_load[name])

        if not args.use_ddp:
            ######## save model ########
            fpath_noddp = fpath[:-4] + 'noDDP' + '.pth'
            to_save = OrderedDict()
            for m in range(models['cascade_level']):
                name = 'net_{}'.format(m)
                to_save[name] = models[name].state_dict()

                name = 'optim_{}'.format(m)
                to_save[name] = models[name].state_dict()
            torch.save(to_save, fpath_noddp)





    elif args.have_box:

        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        # fpath_box = '/home/sally/nerf/nerfplusplus/logs/box_300_2-1_fullview_sc0.5_mx100-140/model_485000.pth'
        # fpath_sc = '/home/sally/nerf/nerfplusplus/logs/newinter_10x10x18_npp/model_770000.pth'
        
        ############## small inters only #############333
        #fpath_box = '/home/sally/nerfpp/box_models/box_model_485000.pth'
        #fpath_sc = '/home/sally/nerfpp/box_models/bg_model_770000.pth'
        ############## small inters only #############333 
        
        ############### big inters #############
        fpath_box = './pretrained/box_models/box_model_485000.pth'
        fpath_sc = './pretrained//big_inters_norm15_sceneonly/model_425000.pth'
        ############### big inters #############


        to_load_box = torch.load(fpath_box, map_location=map_location)
        to_load_sc = torch.load(fpath_sc, map_location=map_location)







        for m in range(models['cascade_level']):
            # for name in ['net_{}'.format(m), 'optim_{}'.format(m)]:
            for name in ['net_{}'.format(m)]:
                for k in to_load_box[name].keys():
                    to_load_sc[name][k] = to_load_box[name][k]

                # models[name].load_state_dict(to_load_box[name])
                models[name].load_state_dict(to_load_sc[name])



    return start, models


def ddp_train_nerf(rank, args):

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
        args.N_rand = 512
        args.chunk_size = 4096

    args.N_rand = 512
    args.chunk_size = 2048

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
                                   try_load_min_depth=args.load_min_depth, have_box=args.have_box,
                                   train_depth=True)
    val_ray_samplers = load_data_split(args.datadir, args.scene, split='validation',
                                       try_load_min_depth=args.load_min_depth, skip=args.testskip, have_box=args.have_box,
                                       train_depth=True)


    ###### create network and wrap in ddp; each process should do this
    start, models = create_nerf_noddp(args)

    ##### important!!!
    # make sure different processes sample different rays
    np.random.seed((rank + 1) * 777)
    # make sure different processes have different perturbations in depth samples
    torch.manual_seed((rank + 1) * 777)

    writer = SummaryWriter(os.path.join(args.basedir, 'summaries', args.expname))
        
    # start training
    what_val_to_log = 0             # helper variable for parallel rendering of a image
    what_train_to_log = 0
    for global_step in range(start+1, start+1+args.N_iters):
        time0 = time.time()
        scalars_to_log = OrderedDict()

        # randomly sample rays and move to device
        i = np.random.randint(low=0, high=len(ray_samplers))
        ray_batch = ray_samplers[i].random_sample(args.N_rand, center_crop=False)
        for key in ray_batch:
            if torch.is_tensor(ray_batch[key]):
                ray_batch[key] = ray_batch[key].to(rank)



        # forward and backward
        dots_sh = list(ray_batch['ray_d'].shape[:-1])  # number of rays
        all_rets = []                                  # results on different cascade levels
        for m in range(models['cascade_level']):
            optim = models['optim_{}'.format(m)]
            net = models['net_{}'.format(m)]

            # TODO: remove.
            # for parameter in net.parameters():
            #     parameter.requires_grad = False

            # for parameter in net.module.nerf_net.box_net.parameters():
            #     parameter.requires_grad = True

            # for parameter in net.module.nerf_net.fg_net.parameters():
            #     parameter.requires_grad - True
            # ====

            # sample depths
            N_samples = models['cascade_samples'][m]


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
            if not args.have_box:
                ret = net(ray_batch['ray_o'], ray_batch['ray_d'], fg_far_depth, fg_depth, bg_depth, img_name=ray_batch['img_name'])
            elif not args.train_box_only:
                ret = net(ray_batch['ray_o'], ray_batch['ray_d'], fg_far_depth, fg_depth, bg_depth, ray_batch['box_loc'], img_name=ray_batch['img_name'])
            else:
                ret = net(ray_batch['ray_o'], ray_batch['ray_d'], fg_far_depth, fg_depth,
                          img_name=ray_batch['img_name'])

            all_rets.append(ret)

            rgb_gt = ray_batch['rgb'].to(rank)

            rgb_loss = img2mse(ret['rgb'], rgb_gt)
            if args.depth_training:
                depth_gt = ray_batch['depth_gt'].to(rank)
                depth_pred = ret['depth_fgbg']

                inds = torch.where(depth_gt < 2001.)
                d_pred_map = depth_pred[inds]
                d_gt_map = depth_gt[inds]

                depth_loss = dep_l1l2loss(torch.div(1.,d_pred_map), torch.div(1.,d_gt_map), l1l2 = 'l1')
                #reg_loss = dep_l1l2loss(torch.div(1.,d_pred_map[:512])-torch.div(1.,d_pred_map[512:]), torch.div(1.,d_gt_map[:512])-torch.div(1.,d_gt_map[512:]), l1l2 = 'l1')
                loss = rgb_loss * 0.  +  depth_loss
                #scalars_to_log['level_{}/reg_loss'.format(m)] = reg_loss.item()

                scalars_to_log['level_{}/depth_loss'.format(m)] = depth_loss.item()

            else:
                loss = rgb_loss
                scalars_to_log['level_{}/loss'.format(m)] = rgb_loss.item()
                scalars_to_log['level_{}/pnsr'.format(m)] = mse2psnr(rgb_loss.item())


            loss.backward()
            optim.step()



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
            log_data = render_single_image_noddp(models, val_ray_samplers[idx], args.chunk_size, args.train_box_only, have_box=args.have_box)

            what_val_to_log += 1
            dt = time.time() - time0

            logger.info('Logged a random validation view in {} seconds'.format(dt))
            #if args.depth_training:
            log_view_to_tb(writer, global_step, log_data, gt_depth=val_ray_samplers[idx].get_depth(), gt_img=val_ray_samplers[idx].get_img(), mask=None, have_box=args.have_box, train_box_only=args.train_box_only, prefix='val/')
            #else:
                 #   log_view_to_tb(writer, global_step, log_data, gt_img=val_ray_samplers[idx].get_img(), mask=None, have_box=args.have_box, train_box_only=args.train_box_only, prefix='val/')

            time0 = time.time()
            idx = what_train_to_log % len(ray_samplers)
            log_data = render_single_image_noddp(models, ray_samplers[idx], args.chunk_size, args.train_box_only, have_box=args.have_box)
            what_train_to_log += 1
            dt = time.time() - time0

            logger.info('Logged a random training view in {} seconds'.format(dt))
            log_view_to_tb(writer, global_step, log_data, gt_img=ray_samplers[idx].get_img(), gt_depth=ray_samplers[idx].get_depth(), mask=None, have_box=args.have_box, train_box_only=args.train_box_only, prefix='train/')

            del log_data
            torch.cuda.empty_cache()

        if (global_step % args.i_weights == 0 and global_step > 0):
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
    # TODO
    # cleanup()


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

    parser.add_argument("--depth_training", action='store_true',
                        help='whether to train with depth')

    parser.add_argument("--use_ddp", action='store_true',
                        help='whether to use distributed training')

    return parser


def train():
    parser = config_parser()
    args = parser.parse_args()
    logger.info(parser.format_values())

    torch.cuda.set_device(0)
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
    ddp_train_nerf(rank=torch.cuda.current_device(), args=args)


if __name__ == '__main__':
    setup_logger()
    train()



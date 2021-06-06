import torch
# import torch.nn as nn
import torch.optim
import torch.distributed
# from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing
import numpy as np
import os
# from collections import OrderedDict
# from ddp_model import NerfNet
import time
from data_loader_split import load_data_split
from utils import mse2psnr, colorize_np, to8b
import imageio
from ddp_train_nerf import config_parser, setup_logger, setup, cleanup, create_nerf, render_single_image_noddp, create_nerf_noddp
import logging


logger = logging.getLogger(__package__)


def ddp_test_nerf_noddp(rank, args):
    logger = logging.getLogger(__package__)
    setup_logger()

    ###### decide chunk size according to gpu memory
    if torch.cuda.get_device_properties(rank).total_memory / 1e9 > 14:
        logger.info('setting batch size according to 24G gpu')
        args.N_rand = 1024
        args.chunk_size = 8192
    else:
        logger.info('setting batch size according to 12G gpu')
        args.N_rand = 512
        args.chunk_size = 4096

    ###### create network and wrap in ddp; each process should do this
    start, models = create_nerf_noddp(args)

    render_splits = [x.strip() for x in args.render_splits.strip().split(',')]
    # start testing
    for split in render_splits:
        out_dir = os.path.join(args.basedir, args.expname,
                               'render_{}_{:06d}'.format(split, start))

        os.makedirs(out_dir, exist_ok=True)

        ###### load data and create ray samplers; each process should do this
        ray_samplers = load_data_split(args.datadir, args.scene, split, try_load_min_depth=args.load_min_depth, \
                                       have_box=args.have_box, train_depth=True\
                                       )
        for idx in range(len(ray_samplers)):
            ### each process should do this; but only main process merges the results
            fname = '{:06d}.png'.format(idx)
            if ray_samplers[idx].img_path is not None:
                fname = os.path.basename(ray_samplers[idx].img_path)

            if os.path.isfile(os.path.join(out_dir, fname)):
                logger.info('Skipping {}'.format(fname))
                continue

            time0 = time.time()
            ret = render_single_image_noddp(models, ray_samplers[idx], args.chunk_size, \
                                      train_box_only=False, have_box=True)

            dt = time.time() - time0

            logger.info('Rendered {} in {} seconds'.format(fname, dt))

            # only save last level
            im = ret[-1]['rgb'].numpy()
            # compute psnr if ground-truth is available
            if ray_samplers[idx].img_path is not None:
                gt_im = ray_samplers[idx].get_img()
                psnr = mse2psnr(np.mean((gt_im - im) * (gt_im - im)))
                logger.info('{}: psnr={}'.format(fname, psnr))

            im = to8b(im)
            imageio.imwrite(os.path.join(out_dir, fname), im)

            im = ret[-1]['fg_rgb'].numpy()
            im = to8b(im)
            imageio.imwrite(os.path.join(out_dir, 'fg_' + fname), im)

            im = ret[-1]['bg_rgb'].numpy()
            im = to8b(im)
            imageio.imwrite(os.path.join(out_dir, 'bg_' + fname), im)

            im = ret[-1]['fg_depth'].numpy()
            im = colorize_np(im, cmap_name='jet', append_cbar=True)
            im = to8b(im)
            imageio.imwrite(os.path.join(out_dir, 'fg_depth_' + fname), im)

            im = ret[-1]['bg_depth'].numpy()
            im = colorize_np(im, cmap_name='jet', append_cbar=True)
            im = to8b(im)
            imageio.imwrite(os.path.join(out_dir, 'bg_depth_' + fname), im)

            depth_clip = 100.
            im = ret[-1]['depth_fgbg'].numpy()
            im[
                im > depth_clip] = depth_clip  ##### THIS IS THE DEPTH OUTPUT, HxW, value is meters away from camera centre

            im = colorize_np(im, cmap_name='jet', append_cbar=True)
            im = to8b(im)
            imageio.imwrite(os.path.join(out_dir, 'Depth_' + fname), im)

            im = ray_samplers[idx].get_depth()
            im[
                im > depth_clip] = depth_clip  ##### THIS IS THE DEPTH OUTPUT, HxW, value is meters away from camera centre

            im = colorize_np(im, cmap_name='jet', append_cbar=True)
            im = to8b(im)
            imageio.imwrite(os.path.join(out_dir, 'DepthGT_' + fname), im)

            torch.cuda.empty_cache()


def test():
    parser = config_parser()
    args = parser.parse_args()
    logger.info(parser.format_values())

    # if args.world_size == -1:
    #     args.world_size = torch.cuda.device_count()
    #     logger.info('Using # gpus: {}'.format(args.world_size))
    # torch.multiprocessing.spawn(ddp_test_nerf,
    #                             args=(args,),
    #                             nprocs=args.world_size,
    #                             join=True)

    ddp_test_nerf_noddp(torch.cuda.current_device(), args)


if __name__ == '__main__':
    setup_logger()
    test()

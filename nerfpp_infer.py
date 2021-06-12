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
from data_loader_split import load_data_split, load_data_array
from utils import mse2psnr, colorize_np, to8b
import imageio
from ddp_train_nerf import config_parser, setup_logger, setup, cleanup, render_single_image, create_nerf
import logging
import random
import pickle


logger = logging.getLogger(__package__)

def ddp_test_nerf_fromArr(rank, args, intrs, poses, locs, H, W, plot, normalize, depth_clip):
    ###### set up multi-processing
    setup(rank, args.world_size)
    ###### set up logger
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
    start, models = create_nerf(rank, args)

    out_dir = os.path.join(args.basedir, args.expname,
                           'render_{}_{:06d}'.format('test', start))
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    ###### load data and create ray samplers; each process should do this
    ###### test test read arr ####

    ray_samplers = load_data_array(intrs, poses, locs, H, W, plot, normalize)

    for idx in range(len(ray_samplers)):
        ### each process should do this; but only main process merges the results
        fname = '{:06d}.png'.format(idx)

        if os.path.isfile(os.path.join(out_dir, fname)):
            logger.info('Skipping {}'.format(fname))
            continue

        time0 = time.time()
        ret = render_single_image(rank, args.world_size, models, ray_samplers[idx], args.chunk_size,
                                  args.train_box_only)
        dt = time.time() - time0
        if rank == 0:  # only main process should do this
            logger.info('Rendered {} in {} seconds'.format(fname, dt))

            # only save last level
            im = ret[-1]['rgb'].numpy()
            # compute psnr if ground-truth is available
            if ray_samplers[idx].img_path is not None:
                gt_im = ray_samplers[idx].get_img()
                psnr = mse2psnr(np.mean((gt_im - im) * (gt_im - im)))
                logger.info('{}: psnr={}'.format(fname, psnr))

            im = to8b(im)  ##### THIS IS THE RGB OUTPUT, 3xHxW, value is int between 0 and 256
            imageio.imwrite(os.path.join(out_dir, fname), im)

            im = ret[-1]['depth_fgbg'].numpy()
            im[im > depth_clip] = depth_clip  ##### THIS IS THE DEPTH OUTPUT, HxW, value is meters away from camera centre


            im = colorize_np(im, cmap_name='jet', append_cbar=True)
            im = to8b(im)
            imageio.imwrite(os.path.join(out_dir, 'Depth_' + fname), im)

        torch.cuda.empty_cache()

    # clean up for multi-processing


    cleanup()


def test():
    parser = config_parser()
    args = parser.parse_args()
    logger.info(parser.format_values())

    if args.world_size == -1:
        args.world_size = torch.cuda.device_count()
        logger.info('Using # gpus: {}'.format(args.world_size))


    ## replace these 3 with your own in the format of list of np arrays
    ## format see data/inf_test/test
    # poses: 4x4 cam2world matrix
    # intrs: 4x4 intrinsics matrix
    # locs: [xloc yloc], height of camera is fixed in the code

    [poses, intrs, locs] = pickle.load(
        open('./data/inf_test/test/sample_arrs', 'rb'))

    H = 380 # high of image desired
    W = 640 # width of image desired
    depth_clip = 100.  # clip depth

    plot = True  # plot where the pose and box location is, can be disabled with False
    normalize = True # always true


    torch.multiprocessing.spawn(ddp_test_nerf_fromArr,
                                args=(args, intrs, poses, locs, H, W, plot, normalize, depth_clip),
                                nprocs=args.world_size,
                                join=True)


if __name__ == '__main__':
    setup_logger()
    test()


import torch
# import torch.nn as nn
import torch.optim
import torch.distributed
# from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing
from multiprocessing import Queue
import multiprocessing
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
class Sim:
    def __init__(self, args, frame_height, frame_width, max_depth):
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.max_depth = max_depth
        self.args = args
        self.data_queue = multiprocessing.Manager().Queue()


    def query(self, pose, cube_location):
        intrs = np.zeros((1, 4, 4))
        intrs[0, :, :] = np.eye(4)
        poses = pose.view(1, 4, 4)
        cube_locations = cube_location.view(1, -1)
        workers = torch.multiprocessing.spawn(ddp_test_nerf_fromArr,
                                    args=(self.args, intrs, poses, cube_locations,
                                        self.frame_height, self.frame_width,
                                        False, True, self.max_depth,
                                        self.data_queue),
                                    nprocs=self.args.world_size,
                                    join=False)


        workers.join()

        return self.data_queue.get()





def ddp_test_nerf_fromArr(rank, args, intrs, poses, locs, H, W, plot, normalize, depth_clip, data_out):
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
#    poses = poses.requires_grad_(True)
#    locs = locs.requires_grad_(True)

    ray_samplers = load_data_array(intrs, poses, locs, H, W, plot, normalize)

    for idx in range(len(ray_samplers)):
        ### each process should do this; but only main process merges the results
        fname = '{:06d}.png'.format(idx)

#        if os.path.isfile(os.path.join(out_dir, fname)):
#            logger.info('Skipping {}'.format(fname))
#            continue


        time0 = time.time()
        ret = render_single_image(rank, args.world_size, models, ray_samplers[idx], args.chunk_size,
                                  args.train_box_only)
        dt = time.time() - time0
        if rank == 0:  # only main process should do this
            logger.info('Rendered {} in {} seconds'.format(fname, dt))

            rgb = ret[-1]['rgb'].numpy()
            d = ret[-1]['depth_fgbg'].numpy()
            d[d > depth_clip] = depth_clip  ##### THIS IS THE DEPTH OUTPUT, HxW, value is meters away from camera centre


            data_out.put((rgb, d))
#            time.sleep(1)

        torch.cuda.empty_cache()

    # clean up for multi-processing


    cleanup()


def make_sim():
    parser = config_parser()
    args = parser.parse_args("--config ./configs/carla_box/carla_box.txt")
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

    print(locs)
    H = 32 # high of image desired
    W = 100 # width of image desired
    depth_clip = 60.  # clip depth

    plot = True  # plot where the pose and box location is, can be disabled with False
    normalize = True # always true

    sim = Sim(args, 32, 100, 60)

    pose = torch.tensor([[-1.0000e+00, -1.2246e-16,  0.0000e+00,  8.5000e+01],
        [ 1.2246e-16, -1.0000e+00,  0.0000e+00,  1.2486e+02],
        [ 0.0000e+00,  0.0000e+00,  1.0000e+00,  1.7000e+00],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]], dtype=torch.float)
    #pose = torch.from_numpy(poses[0])
    loc = torch.from_numpy(locs[0])
    print(pose, loc)
    print(sim.query(pose, loc))

    return sim


if __name__ == '__main__':
    setup_logger()
    make_sim()


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

class Sim:
    def __init__(self, args, frame_height, frame_width, max_depth):
        setup(0, args.world_size)
        self.args = args
        self.start, self.models = create_nerf(args)
        self.h = frame_height
        self.w = frame_width

    def __del__(self):
        cleanup()


    def run(self, i, x, c, max_depth=60):

        ray_samplers = load_data_array([i], [x], [c],
                                       self.h, self.w, False, True)

        time0 = time.time()
        ret = render_single_image(self.models, ray_samplers[0], 256)
        dt = time.time() - time0
        logger.info('Rendered {} in {} seconds'.format(0, dt))

        im = ret[0]

        d = ret[1]
        d[d > max_depth] = max_depth # clip invalid points

        return im, d



def test():
    parser = config_parser()
    args = parser.parse_args("--config configs/carla_box/donerf.txt")
    logger.info(parser.format_values())

    args.world_size = torch.cuda.device_count()
    logger.info('Using # gpus: {}'.format(args.world_size))


    ## replace these 3 with your own in the format of list of np arrays
    ## format see data/inf_test/test
    # poses: 4x4 cam2world matrix
    # intrs: 4x4 intrinsics matrix
    # locs: [xloc yloc], height of camera is fixed in the code

    [poses, intrs, locs] = pickle.load(
        open('./data/inf_test/test/sample_arrs', 'rb'))

    H = 32 # high of image desired
    W = 100 # width of image desired
    depth_clip = 60.  # clip depth

    intrs = [torch.from_numpy(intr).cuda() for intr in intrs]
    poses = [torch.from_numpy(pose).cuda().requires_grad_(True) for pose in poses]
    locs = [torch.from_numpy(loc).cuda().requires_grad_(True) for loc in locs]


    sim = Sim(args, H, W, depth_clip)
    i = 0

    rgb, d = sim.run(intrs[0], poses[0], locs[0])
    brgb = to8b(rgb.cpu().numpy())
    bd = colorize_np(d.cpu().numpy(), cmap_name='jet', append_cbar=True)
    bd = to8b(bd)
    imageio.imwrite("rgb_{}.png".format(i), brgb)
    imageio.imwrite("d_{}.png".format(i), bd)
    A = d.sum()
    A.backward()
    print(poses[0].grad)
    print(locs[0].grad)


def make_sim(config_path, channels, width, depth=60.):
    parser = config_parser()
    args = parser.parse_args("--config " + config_path)
    logger.info(parser.format_values())

    if args.world_size == -1:
        args.world_size = torch.cuda.device_count()
        logger.info('Using # gpus: {}'.format(args.world_size))

    sim = Sim(args, channels, width, depth)

    return sim

if __name__ == '__main__':
    setup_logger()
    test()


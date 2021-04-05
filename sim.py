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
    def __init__(self, args):
        setup(0, args.world_size)
        self.args = args
        self.start, self.models = create_nerf(args)

    def __del__(self):
        cleanup()


    def run(self, intrs, poses, locs, frame_height, frame_width, max_depth = 60):

        ray_samplers = load_data_array(intrs, poses, locs, frame_height, frame_width, False, True)

        images = []
        depths = []

        for idx in range(len(ray_samplers)):
            time0 = time.time()
            ret = render_single_image(self.models, ray_samplers[idx])
            dt = time.time() - time0
            logger.info('Rendered {} in {} seconds'.format(idx, dt))

            # only save last level
            im = ret['rgb']
#            images.append(im)

            d = ret['depth_fgbg']
            d[d > max_depth] = max_depth  ##### THIS IS THE DEPTH OUTPUT, HxW, value is meters away from camera centre
#            depths.append(im)

            return im, d

            torch.cuda.empty_cache()


def test():
    parser = config_parser()
    args = parser.parse_args("--config configs/carla_box/carla_box.txt")
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

    intrs = [torch.from_numpy(intr).to('cuda:0') for intr in intrs]
    poses = [torch.from_numpy(pose).to('cuda:0').requires_grad_(True) for pose in poses]
    locs = [torch.from_numpy(loc).to('cuda:0').requires_grad_(True) for loc in locs]


    sim = Sim(args)
    rgb, d = sim.run(intrs[:1], poses[:1], locs[:1], 32, 100)
    y = d.sum()
    y.backward()
    print(poses[0].grad)



if __name__ == '__main__':
    setup_logger()
    test()


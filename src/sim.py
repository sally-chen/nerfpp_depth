import torch
import torch.optim
import torch.distributed
import torch.multiprocessing
import numpy as np

import os
import time
from .data_loader_split import load_data_split, load_data_array
from .utils import mse2psnr, colorize_np, to8b
import imageio
from .ddp_train_nerf import config_parser, setup_logger, render_single_image, create_nerf
import logging
import random
import pickle

from scipy.spatial.transform import Rotation

logger = logging.getLogger(__package__)

class Sim:
    def __init__(self, args, frame_height, frame_width, max_depth, camera, checkpoint=False):
        #setup(0, args.world_size)
        self.args = args
        self.start, self.models = create_nerf(0, args)
        self.h = frame_height
        self.w = frame_width
        self.camera = camera
        self.checkpoint = checkpoint



#    @torch.no_grad()
    def run(self, x, cube_locs, cube_props, max_depth=60, rgb=False):

        intrs = torch.tensor([[380.,   0., 320.,   0.],
                              [  0., 380., 180.,   0.],
                              [  0.,   0.,   1.,   0.],
                              [  0.,   0.,   0.,   1.]]).type_as(x)

        ray_samplers = load_data_array([intrs], [x], [cube_locs],
                                       self.h, self.w, False,
                                       True, not self.camera)
        cube_size = None
        time0 = time.time()
        ret = render_single_image(self.models, ray_samplers[0], self.args.chunk_size,
                                  box_props = cube_props,
                                  train_box_only=self.args.train_box_only,
                                  have_box=self.args.have_box,
                                  donerf_pretrain=self.args.donerf_pretrain,
                                  box_number=self.args.box_number,
                                  front_sample=self.args.front_sample,
                                  back_sample=self.args.back_sample,
                                  fg_bg_net=self.args.fg_bg_net,
                                  use_zval=self.args.use_zval,
                                  checkpoint=self.checkpoint)
        dt = time.time() - time0
        logger.info('Rendered {} in {} seconds'.format(0, dt))

        im = ret[0]

        d = ret[1]
        d[d > max_depth] = max_depth # clip invalid points

        if rgb:
            return im, d

        return d



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
            open('sample_arrs', 'rb'))

    H = 320 # high of image desired
    W = 640 # width of image desired
    depth_clip = 60.  # clip depth


    intrs = [torch.from_numpy(intr).cuda() for intr in intrs]
    poses = [torch.from_numpy(pose).cuda().requires_grad_(True) for pose in poses]
    locs = [torch.from_numpy(loc).cuda().requires_grad_(True) for loc in locs]

    cube = torch.tensor([[90., 122., -1], [85, 122, -1], [80, 120, -1]]).cuda()
    pos = torch.tensor([92.4, 124.]).cuda()

    sim = Sim(args, H, W, depth_clip)
    i = 0
    torch.autograd.set_detect_anomaly(True)
    steps = 20
    for k in range(steps):
        yaw = -np.pi + k * 2 * np.pi / steps
        T = torch.eye(4).cuda()
        T[:2, 3] = pos

        P = yaw_to_mat(torch.tensor([yaw]).cuda()).squeeze().requires_grad_(True)
        T[:3, :3] = P

        rgb, d = sim.run(T, cube, intrs=intrs[0], rgb=True)
#        A = rgb.sum()
#        A.backward()
#        print(P.grad)
        brgb = to8b(rgb.detach().cpu().numpy())
        bd = colorize_np(d.detach().cpu().numpy(), cmap_name='jet', append_cbar=True)
        bd = to8b(bd)
        imageio.imwrite("rgb_{}.png".format(k), brgb)
        imageio.imwrite("d_{}.png".format(k), bd)

#    A = d.sum()
#    A.backward()
#    print(poses[0].grad)

def yaw_to_mat(yaws):

    Rp = torch.tensor([[0., 0., 1.,], [1., 0., 0.], [0., -1., 0.]]).type_as(yaws)
    yaws = yaws.view(-1, 1)

    cos = yaws.cos()
    sin = yaws.sin()

    K = torch.tensor([[0., -1., 0.],
                      [1., 0., 0.],
                      [0., 0., 0.]], device=yaws.device)

    K = torch.tensor([[0., 0., 1.], [0., 0., 0.], [-1., 0., 0.]]).type_as(yaws)
    KK = K.mm(K)
    KK = KK.expand(yaws.shape[0], -1, -1)
    K = K.expand(yaws.shape[0], -1, -1)

    I = torch.eye(3, device=yaws.device).expand(yaws.shape[0], -1,  -1)

    R = I +  sin.view(-1, 1, 1) * K + (1 - cos).view(-1, 1, 1) * KK
    Rf = torch.matmul(Rp, R)

    return Rf


def make_sim(config_path, frame_height, frame_width, depth=60., camera=False,
             box_num=6, chunk_size=-1, checkpoint=False):
    parser = config_parser()
    args = parser.parse_args("--config " + config_path)
    args.box_number = box_num
    if chunk_size > 0:
        args.chunk_size = chunk_size
    logger.info(parser.format_values())

    if args.world_size == -1:
        args.world_size = torch.cuda.device_count()
        logger.info('Using # gpus: {}'.format(args.world_size))

    sim = Sim(args, frame_height, frame_width, depth, camera, checkpoint)

    return sim

if __name__ == '__main__':
    setup_logger()
    test()


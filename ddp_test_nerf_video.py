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
from data_loader_video import load_data_split_video
from utils import mse2psnr, colorize_np, to8b
import imageio
from ddp_train_nerf import config_parser, setup_logger, render_single_image, create_nerf
import logging
from helpers import plot_mult_pose, write_depth_vis

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

logger = logging.getLogger(__package__)

def convert_back_loc(box):
    def convert_back_box(avg_pos, max_minus_min, min, v):
        v *= 2
        v += avg_pos[:2]
        v *= max_minus_min[:2]
        v += min[:2]
        return v
    avg_pos = np.array([0.5, 0.5, 0.])
    max_minus_min = np.array([15.0, 15.0, 0.])
    min = np.array([85.0, 125.0, 2.8])
    return convert_back_box(avg_pos, max_minus_min, min, box[0])

def watermark_text(input_image_path,
                   output_image_path,
                   texts):
    photo = Image.open(input_image_path)
    drawing = ImageDraw.Draw(photo)
    black = (3, 8, 12)
    font = ImageFont.truetype("./Marlboro.ttf", 20)
    drawing.text((0,0), "box_x: " + str(texts[0]), fill=black, font=font)
    drawing.text((0,20), "box_y: " + str(texts[1]), fill=black, font=font)
    drawing.text((0, 40), "cam_x: " + str(texts[2]), fill=black, font=font)
    drawing.text((0, 60), "cam_y: " + str(texts[3]), fill=black, font=font)
    drawing.text((0, 80), "cam_z: " + str(texts[4]), fill=black, font=font)
    photo.save(output_image_path)


def convert_pose_back(C2W):
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    flip = np.linalg.inv(flip_yz)
    C2W = np.matmul(C2W, flip)
    C2W = C2W[:-1, ...]
    avg_pos = np.array([0.5, 0.5, 0.])
    max_minus_min = np.array([15.0, 15.0, 0.])
    min_pose = np.array([85.0, 125.0, 2.8])

    C2W[:3, 3] *= 2.0
    C2W[:3, 3] += avg_pos
    C2W[:2, 3] *= max_minus_min[:2]
    C2W[:3, 3] += min_pose

    return C2W[:3, 3]


def ddp_test_nerf(rank, args):
    ###### set up multi-processing

    ###### set up logger
    logger = logging.getLogger(__package__)
    setup_logger()


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

    ###### create network and wrap in ddp; each process should do this
    start, models = create_nerf(rank, args)

    render_splits = [x.strip() for x in args.render_splits.strip().split(',')]
    # start testing
    for split in render_splits:
        out_dir = os.path.join(args.basedir, args.expname,
                               'render_{}_{:06d}'.format(split, start))

        os.makedirs(out_dir, exist_ok=True)

        ###### load data and create ray samplers; each process should do this
        # ray_samplers = load_data_split(args.datadir, args.scene, split, try_load_min_depth=args.load_min_depth)
        ray_samplers_video = load_data_split_video(args.datadir, args.scene, split, try_load_min_depth=args.load_min_depth)


        for idx in range(len(ray_samplers_video)):

            if idx < 8:
                continue

            print('rendering : {}'.format(idx))
            ### each process should do this; but only main process merges the results
            fname = '{:06d}.png'.format(idx)
            if ray_samplers_video[idx].img_path is not None:
                fname = os.path.basename(ray_samplers_video[idx].img_path)

            if os.path.isfile(os.path.join(out_dir, fname)):
                logger.info('Skipping {}'.format(fname))
                continue

            time0 = time.time()
            # ret = render_single_image(rank, args.world_size, models, ray_samplers_video[idx], args.chunk_size)
            with torch.no_grad():
                rgb, d, pred_fg, pred_bg, label, others = render_single_image(models, ray_samplers_video[idx],
                                                                          args.chunk_size,
                                                                          args.train_box_only, have_box=args.have_box,
                                                                          donerf_pretrain=args.donerf_pretrain,
                                                                          front_sample=args.front_sample,
                                                                          back_sample=args.back_sample,
                                                                          fg_bg_net=args.fg_bg_net,
                                                                          use_zval=args.use_zval, loss_type='bce',
                                                                          rank=rank, DEBUG=False)


            dt = time.time() - time0

            logger.info('Rendered {} in {} seconds'.format(fname, dt))

            # only save last level
            im = rgb.numpy()
            # compute psnr if ground-truth is available

            select_inds = np.arange(640 * 280, 640 * 281)
            select_inds2 = np.arange(640 * 180, 640 * 181)

            write_depth_vis(None,  np.concatenate([torch.sigmoid(pred_fg[select_inds]),
                                                  torch.sigmoid(pred_fg[select_inds2])],
                                                 axis=1), out_dir, 'depthpredfg_'+fname)
            write_depth_vis(None,  np.concatenate([torch.sigmoid(pred_bg[select_inds]),
                                                  torch.sigmoid(pred_bg[select_inds2])],
                                                 axis=1), out_dir, 'depthpredbg_'+fname)

            if ray_samplers_video[idx].img_path is not None:
                gt_im = ray_samplers_video[idx].get_img()
                psnr = mse2psnr(np.mean((gt_im - im) * (gt_im - im)))
                logger.info('{}: psnr={}'.format(fname, psnr))

                # box_loc = convert_back_loc(ray_samplers[idx].box_loc)
                # # print(ray_samplers[idx].c2w_mat.shape)
                # cam_loc = convert_pose_back(ray_samplers[idx].c2w_mat)
                # texts = [box_loc[0], box_loc[1], cam_loc[0], cam_loc[1], cam_loc[2]]
            im = to8b(im)
            imageio.imwrite(os.path.join(out_dir, fname), im)
                # watermark_text(os.path.join(out_dir, fname),
                #                os.path.join(out_dir, 'watermarked_' + fname),
                #                texts)

                # im = ret[-1]['fg_rgb'].numpy()
                # im = to8b(im)
                # imageio.imwrite(os.path.join(out_dir, 'fg_' + fname), im)
                #
                # im = ret[-1]['bg_rgb'].numpy()
                # im = to8b(im)
                # imageio.imwrite(os.path.join(out_dir, 'bg_' + fname), im)
                #

            im = d.numpy()
            im[im > 100.] = 100.
            im = colorize_np(im, cmap_name='jet', append_cbar=True)
            im = to8b(im)
            imageio.imwrite(os.path.join(out_dir, 'depth_' + fname), im)

                # im = ret[-1]['bg_depth'].numpy()
                # im = colorize_np(im, cmap_name='jet', append_cbar=True)
                # im = to8b(im)
                # imageio.imwrite(os.path.join(out_dir, 'bg_depth_' + fname), im)

            torch.cuda.empty_cache()




def test():
    parser = config_parser()
    args = parser.parse_args()
    logger.info(parser.format_values())

    ddp_test_nerf(0, args)


if __name__ == '__main__':
    setup_logger()
    test()


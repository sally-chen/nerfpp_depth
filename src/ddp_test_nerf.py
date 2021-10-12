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
from .data_loader_split import load_data_split
from .utils import mse2psnr, colorize_np, to8b
import imageio
from .ddp_train_nerf import config_parser, setup_logger, render_single_image, create_nerf
import logging


logger = logging.getLogger(__package__)


def ddp_test_nerf(rank, args):

    ###### set up logger
    logger = logging.getLogger(__package__)
    setup_logger()



    start, models = create_nerf(rank, args)

    render_splits = [x.strip() for x in args.render_splits.strip().split(',')]
    # start testing
    for split in render_splits:
        out_dir = os.path.join(args.basedir, args.expname,
                               'render_{}_{:06d}'.format(split, start))
        if rank == 0:
            os.makedirs(out_dir, exist_ok=True)

        ###### load data and create ray samplers; each process should do this
        # ray_samplers = load_data_split(args.datadir, args.scene, split, try_load_min_depth=args.load_min_depth,\
        #                                have_box=args.have_box, train_depth=True, custom_rays=True
        #                                )
        ray_samplers = load_data_split(args.datadir, args.scene, split='test',
                                           try_load_min_depth=args.load_min_depth, skip=args.testskip,
                                           have_box=args.have_box,
                                           train_depth=True)

        for idx in range(len(ray_samplers)):
            ### each process should do this; but only main process merges the results
            fname = '{:06d}.png'.format(idx)
            if ray_samplers[idx].img_path is not None:
                fname = os.path.basename(ray_samplers[idx].img_path)

            if os.path.isfile(os.path.join(out_dir, fname)):
                logger.info('Skipping {}'.format(fname))
                continue

            time0 = time.time()
            with torch.no_grad():
                rgb, d, pred_fg, pred_bg, label, others = render_single_image(models, ray_samplers[idx],
                                                                          args.chunk_size,
                                                                          args.train_box_only, have_box=args.have_box,
                                                                          donerf_pretrain=args.donerf_pretrain,
                                                                          front_sample=args.front_sample,
                                                                          back_sample=args.back_sample,
                                                                          fg_bg_net=args.fg_bg_net,
                                                                          use_zval=args.use_zval, loss_type='bce',
                                                                          rank=rank, DEBUG=True)
            dt = time.time() - time0
            if rank == 0:    # only main process should do this
                logger.info('Rendered {} in {} seconds'.format(fname, dt))


                # only save last level
                im = rgb.numpy()
                # compute psnr if ground-truth is available
                if ray_samplers[idx].img_path is not None:
                    gt_im = ray_samplers[idx].get_img()
                    psnr = mse2psnr(np.mean((gt_im - im) * (gt_im - im)))
                    logger.info('{}: psnr={}'.format(fname, psnr))

                im = to8b(im)
                imageio.imwrite(os.path.join(out_dir, fname), im)

                # im = ret[-1]['fg_rgb'].numpy()
                # im = to8b(im)
                # imageio.imwrite(os.path.join(out_dir, 'fg_' + fname), im)
                #
                # im = ret[-1]['bg_rgb'].numpy()
                # im = to8b(im)
                # imageio.imwrite(os.path.join(out_dir, 'bg_' + fname), im)

                # im = ret[-1]['fg_depth'].numpy()
                # im = colorize_np(im, cmap_name='jet', append_cbar=True)
                # im = to8b(im)
                # imageio.imwrite(os.path.join(out_dir, 'fg_depth_' + fname), im)
                #
                # im = ret[-1]['bg_depth'].numpy()
                # im = colorize_np(im, cmap_name='jet', append_cbar=True)
                # im = to8b(im)
                # imageio.imwrite(os.path.join(out_dir, 'bg_depth_' + fname), im)

                # depth_clip=100.
                # im = d['depth_fgbg'].numpy()
                # im[im > depth_clip] = depth_clip  ##### THIS IS THE DEPTH OUTPUT, HxW, value is meters away from camera centre
                #
                # im = colorize_np(im, cmap_name='jet', append_cbar=True)
                # im = to8b(im)
                # imageio.imwrite(os.path.join(out_dir, 'Depth_' + fname), im)
                #
                # im = ray_samplers[idx].get_depth()
                # im[im > depth_clip] = depth_clip  ##### THIS IS THE DEPTH OUTPUT, HxW, value is meters away from camera centre
                #
                # im = colorize_np(im, cmap_name='jet', append_cbar=True)
                # im = to8b(im)
                # imageio.imwrite(os.path.join(out_dir, 'DepthGT_' + fname), im)

            torch.cuda.empty_cache()




def test():
    parser = config_parser()
    args = parser.parse_args()
    logger.info(parser.format_values())

    ddp_test_nerf(0, args)


if __name__ == '__main__':
    setup_logger()
    test()

import numpy as np
import os, imageio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from load_blender import *
from helpers import plot_mult_pose

from scipy.spatial.transform import Rotation as R


def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return

    from shutil import copy
    from subprocess import check_output

    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir

    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100. / r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue

        print('Minifying', r, basedir)

        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)

        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)

        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')


def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True, stack_img=False, loc=True):
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    bds = poses_arr[:, -2:].transpose([1, 0])

    if loc:
        box_loc = np.load(os.path.join(basedir, 'loc_bounds.npy'))
        box_loc = np.concatenate([box_loc, 0.5 * np.ones((box_loc.shape[0], 1))], axis=1)[:1458]

    else:
        box_loc = None

    # poses = poses[:,:,:960]
    # bds = bds[:,:960]

    imgdir = os.path.join(basedir, 'images')
    if not os.path.exists(imgdir):
        print(imgdir, 'does not exist, returning')
        return

    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if
                f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print('Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]))
        return

    sh = imageio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1. / factor

    if not load_imgs:
        return poses, bds

    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)

    imgs = [imread(f)[..., :3] for f in imgfiles]

    if stack_img:
        imgs = np.stack(imgs, -1)

        print('Loaded image data', imgs[0].shape, poses[:, -1, 0])
    else:
        print('Loaded image data', len(imgs), poses[:, -1, 0])


    return poses, bds, imgs, box_loc


def find_files(dir, exts):
    if os.path.isdir(dir):
        # types should be ['*.png', '*.jpg']
        files_grabbed = []
        for ext in exts:
            files_grabbed.extend(glob.glob(os.path.join(dir, ext)))
        if len(files_grabbed) > 0:
            files_grabbed = sorted(files_grabbed)
        return files_grabbed
    else:
        return []

def load_data(basedir, loc=True):
    poses, bds, imgs, box_loc = _load_data(basedir, factor=1, loc=loc)  # factor=8 downsamples original imgs by 8x
    print('Loaded', basedir, bds.min(), bds.max())

    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    # imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    images = imgs
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)


    # filter here

    return poses, images, box_loc

def write_folder(path, images, poses_list, K, c2w_converted_ev=None, box_loc=None, val_split=180, test_split=120):

    i_test = np.arange((len(poses_list)))[::test_split]

    i_val = np.arange((len(poses_list)))[::val_split]
    i_train = np.array([i for i in np.arange(int(len(images))) if
                        (i not in i_test and i not in i_val)])


    choice_list = ['train', 'validation', 'test' ]
    indeces = [i_train, i_val, i_test]

    if not os.path.exists(path):
        os.mkdir(path)

    for choice_ind in range(3):
        top_path = path + '/' + choice_list[choice_ind]

        if not os.path.exists(top_path):
            os.mkdir(top_path)
            int_path = top_path + '/' + 'intrinsics'
            pose_path = top_path + '/' + 'pose'
            rgb_path = top_path + '/' + 'rgb'

            # box loc
            if box_loc is not None:
                fname = top_path + '/' + 'box_loc' + '.npy'
                np.save(fname, box_loc[indeces[choice_ind]])

            os.mkdir(int_path)
            os.mkdir(pose_path)
            os.mkdir(rgb_path)
            img_sels = [images[i] for i in indeces[choice_ind]]
            for count in range(len(indeces[choice_ind])):
                fname = int_path + '/' + str(count+1).zfill(5) + '.txt'
                np.savetxt(fname, np.reshape(K.flatten(), [1, -1]), fmt='%10.4f', delimiter=' ')

                fname = pose_path + '/' + str(count + 1).zfill(5) + '.txt'
                np.savetxt(fname, np.reshape(poses_list[indeces[choice_ind][count]].flatten(), [1, -1]), fmt='%10.4f', delimiter=' ')

                fname = rgb_path + '/' + str(count + 1).zfill(5) + '.jpg'
                imageio.imwrite(fname, img_sels[count])

    top_path = path + '/' + 'camera_path'

    if not os.path.exists(top_path):
        os.mkdir(top_path)
        int_path = top_path + '/' + 'intrinsics'
        pose_path = top_path + '/' + 'pose'
        os.mkdir(int_path)
        os.mkdir(pose_path)

    if c2w_converted_ev is not None:

        for i in range(len(c2w_converted_ev)):
            fname = int_path + '/' + str(i+ 1).zfill(5) + '.txt'
            np.savetxt(fname, np.reshape(K.flatten(), [1, -1]), fmt='%10.4f', delimiter=' ')

            fname = pose_path + '/' + str(i+ 1).zfill(5) + '.txt'
            np.savetxt(fname, np.reshape(c2w_converted_ev[i].flatten(), [1, -1]), fmt='%10.4f',
                       delimiter=' ')


def convert(poses, box_loc):

    ########### intrinsics #########
    h, w, f = poses[0, :3, -1:]
    # In this case Fx and Fy are the same since the pixel aspect
    # ratio is 1
    K = np.identity(4)
    K[0, 0] = K[1, 1] = f
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0



    ############### c2w  ##################
    c2w = poses[:, :3, :-1]

    # normalize


    max = np.amax(c2w[:, :3, 3], axis=0)
    max = np.array([100,140, 2.8]) # hard code for now
    min = np.amin(c2w[:, :3, 3], axis=0)

    #
    # new_li = []
    # ind = []
    # for i in range(c2w.shape[0]):
    #     x = c2w[i, 0, 3]
    #     y = c2w[i, 1, 3]
    #
    #     if x < max[0] and y < max[1]:
    #         ind.append(i)
    #
    # c2w = c2w[ind]
    #

    c2w[:, :3, 3] -= min
    c2w[:, :2, 3] /= (max-min)[:2]
    # avg_pos = np.mean(c2w[:, :3, 3], axis=0)
    avg_pos = np.array([0.5, 0.5, 0])

    c2w[:, :3, 3] -= avg_pos
    c2w[:, :3, 3] *= 0.5

    scale_factors = [max, min, avg_pos, 0.5]

    if box_loc is not None:
        box_loc -= min
        box_loc[:, :2] /= (max - min)[:2]
        box_loc -= avg_pos
        box_loc *= 0.5


    c2w_converted = []
    for i in range(c2w.shape[0]):
        c2w_C = convert_pose(c2w[i])
        c2w_converted.append(c2w_C)

    all_ev = np.load('/home/sally/nerf/nerfplusplus/data/newinter_10x10x18_ev.npy')
    all_ev[:, :3, 3] -= min
    all_ev[:, :2, 3] /= (max - min)[:2]

    all_ev[:, :3, 3] -= avg_pos
    all_ev[:, :3, 3] *= 0.5

    c2w_converted_ev = []
    for i in range(all_ev.shape[0]):
        po = all_ev[i]
        all_ev_ = convert_pose(po)
        c2w_converted_ev.append(all_ev_)

    return K, c2w_converted, c2w_converted_ev, box_loc, scale_factors

def convert_box(poses, h, w, f, scale_factors):
    ########### intrinsics #########

    # In this case Fx and Fy are the same since the pixel aspect
    # ratio is 1
    K = np.identity(4)
    K[0, 0] = K[1, 1] = f
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0

    ############### c2w  ##################

    c2w = poses

    max, min, avg_pos, scale = scale_factors

    # c2w[:, :3, 3] -= min
    c2w[:, :2, 3] /= (max - min)[:2]
    # avg_pos = np.mean(c2w[:, :3, 3], axis=0)
    # c2w[:, :3, 3] -= avg_pos
    c2w[:, :3, 3] *= scale
    # c2w[:, :3, 3] *= 0.01

    c2w_converted = []
    for i in range(c2w.shape[0]):
        c2w_C = convert_pose(c2w[i])
        c2w_converted.append(c2w_C[:4, :])


    return K, c2w_converted

def nerf2nerfpp(path_from, path_to, box_from=None, box_to=None, loc=True):

    poses_sc, images, box_loc = load_data(path_from, loc=loc)

    # train, test, validation/ intrinsics, pose, rgb
    K, c2w_converted, c2w_converted_ev, box_loc, scale_factors = convert(poses_sc, box_loc)

    write_folder(path_to, images, c2w_converted, K, c2w_converted_ev, box_loc)

    if box_from is not None:
        images, poses_box, render_poses, [H, W, focal], i_split = load_blender_data(box_from)
        K, c2w_converted_box = convert_box(poses_box, H, W, focal, scale_factors)
        # write_folder(box_to, images, c2w_converted_box, K, val_split=10, test_split=15)

    plot_mult_pose([np.stack(poses_sc, axis=0), np.stack(poses_box, axis=0)], 'input poses nerf ++', ['scene poses', 'box_poses'])


def convert_pose(C2W):
    C2W = np.concatenate([C2W, np.expand_dims(np.array([0,0,0,1]), axis=0)],axis=0)

    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    C2W = np.matmul(C2W, flip_yz)
    return C2W
#
# def makEv(pose_path):


if __name__ == '__main__':
    path_from = '/home/sally/nerf/nerfplusplus/data/carla_intnew/carla_box//'
    path_to = '/home/sally/nerf/nerfplusplus/data/carla_intnew/carla_box_9x9x18_npp_trunc/'
    # path_to = '/home/sally/nerf/nerfplusplus/data/carla_intnew/newinter_10x10x18_npp_/'

    # box_from =  '/home/sally/nerf/nerfplusplus/data/box/box_300_2-1_fullview/'
    # box_to =  '/home/sally/nerf/nerfplusplus/data/box/box_300_2-1_fullview_sc0.5_mx100-140/'
    box_from = '/home/sally/nerf/nerfplusplus/data/box/box_300_2-0_fullview/'
    box_to = '/home/sally/nerf/nerfplusplus/data/box/test_out/'

    nerf2nerfpp(path_from, path_to, box_from , box_to, loc = True)

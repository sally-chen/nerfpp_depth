import os
import numpy as np
import imageio
import logging
from nerf_sample_ray_split import RaySamplerSingleImage
import glob
from helpers import plot_mult_pose
import math

logger = logging.getLogger(__package__)

########################################################################################################################
# camera coordinate system: x-->right, y-->down, z-->scene (opencv/colmap convention)
# poses is camera-to-world
########################################################################################################################
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

def gen_box_locs(count=55, box_number=10):
    # loc = np.linspace((87,132),(90,135),35)

    # loc = np.concatenate((np.random.uniform(87., 90., (count,1)),
    #                       np.random.uniform(132., 135., (count,1))), axis=1)

    ct1 = 30
    ct2 = 20
    loc1 = np.concatenate((np.random.uniform(85., 88., (ct1, box_number, 1)),
                          np.random.uniform(110., 114., (ct1,box_number, 1))), axis=-1)

    loc2 = np.concatenate((np.random.uniform(95., 97., (ct2, box_number, 1)),
                          np.random.uniform(111., 114., (ct2, box_number,1))), axis=-1)

    loc = np.concatenate([loc1,loc2], axis=0)

    max = np.array([100., 140., 2.8])
    min = np.array([85., 125., 2.8])
    max_minus_min = max-min
    avg_pos = np.array([0.5, 0.5, 0.0])
    loc -= min[:2]
    loc /= (max_minus_min)[:2]
    # print(loc)
    loc -= avg_pos[:2]
    loc *= 0.5

    loc = np.concatenate((loc, -0.06 * np.ones((count,box_number,1))), axis=-1)

    return loc

def parse_cube_loc(input):
    nums_new = np.zeros([3])
    # nums_new[:2] = nums + random.randint(-20,20)/100.
    nums_new[:2] = input
    nums_new[2] = -1
    return nums_new.astype(np.float32)

def gen_poses():
    # xy1 = np.linspace((120,128),(92,128),80)
    xy1 = np.linspace((105,128),(92,128),5)
    dg1 = (np.ones((5, )) * 180.) * math.pi / 180.

    turn1 = np.linspace((92,128),(92,128),5)
    turn1_dg = (np.arange(180.0, 270.0, 18) + np.random.uniform(low=-1.0, high=1.0, size=(5,))) * math.pi / 180.

    xy2 = np.linspace((92,128),(92,108),10)
    dg2 = (np.ones((10,)) * 270.) * math.pi / 180.

    turn2 = np.linspace((92,108),(92,108),5)
    turn2_dg = (np.arange(270.0, 180.0, -18) + np.random.uniform(low=-1.0, high=1.0, size=(5,))) * math.pi / 180.

    xy3 = np.linspace((92,108),(91,108),5)
    dg3 = (np.ones((10,)) * 180.) * math.pi / 180.

    turn3 = np.linspace((91,108),(91,108),5)
    turn3_dg = (np.arange(180.0, 90, -18) + np.random.uniform(low=-1.0, high=1.0, size=(5,))) * math.pi / 180.


    xy4 = np.linspace((91,108),(91,135),15)
    dg4 = (np.ones((10,)) * 90.) * math.pi / 180.
    # turn4 = np.linspace((87,135),(87,135),10)
    # turn4_dg = (np.arange(90.0, 0.0, -9) + np.random.uniform(low=-1.0, high=1.0, size=(10,))) * math.pi / 180.


    # xy5 = np.linspace((87,135),(120,135),80)
    # dg5 = (np.ones((80,)) * 0.) * math.pi / 180.

    return np.concatenate((xy1, turn1, xy2, turn2, xy3, turn3, xy4), axis=0), np.concatenate((dg1, turn1_dg, dg2, turn2_dg, dg3, turn3_dg, dg4), axis=0)

def load_data_split(basedir, scene, split, skip=1, try_load_min_depth=True, only_img_files=False):

    def parse_txt(filename):
        assert os.path.isfile(filename)
        nums = open(filename).read().split()
        return np.array([float(x) for x in nums]).reshape([4, 4]).astype(np.float32)

    def parse_txt_loc(filename):
        assert os.path.isfile(filename)
        nums = open(filename).read().split()
        return np.array([float(x) for x in nums]).reshape([1, 2]).astype(np.float32)

    if basedir[-1] == '/':          # remove trailing '/'
        basedir = basedir[:-1]
     
    split_dir = '{}/{}/{}'.format(basedir, scene, split)

    if only_img_files:
        img_files = find_files('{}/rgb'.format(split_dir), exts=['*.png', '*.jpg'])
        return img_files

    # camera parameters files
    intrinsics_files = find_files('{}/intrinsics'.format(split_dir), exts=['*.txt'])
    pose_files = find_files('{}/pose'.format(split_dir), exts=['*.txt'])
    loc_files = find_files('{}/loc'.format(split_dir), exts=['*.txt'])
    logger.info('raw intrinsics_files: {}'.format(len(intrinsics_files)))
    logger.info('raw pose_files: {}'.format(len(pose_files)))

    intrinsics_files = intrinsics_files[::skip]
    pose_files = pose_files[::skip]
    loc_files = loc_files[::skip]
    cam_cnt = len(pose_files)

    # img files
    img_files = find_files('{}/rgb'.format(split_dir), exts=['*.png', '*.jpg'])
    if len(img_files) > 0:
        logger.info('raw img_files: {}'.format(len(img_files)))
        img_files = img_files[::skip]
        assert(len(img_files) == cam_cnt)
    else:
        img_files = [None, ] * cam_cnt

    # mask files
    mask_files = find_files('{}/mask'.format(split_dir), exts=['*.png', '*.jpg'])
    if len(mask_files) > 0:
        logger.info('raw mask_files: {}'.format(len(mask_files)))
        mask_files = mask_files[::skip]
        assert(len(mask_files) == cam_cnt)
    else:
        mask_files = [None, ] * cam_cnt

    # min depth files
    mindepth_files = find_files('{}/min_depth'.format(split_dir), exts=['*.png', '*.jpg'])
    if try_load_min_depth and len(mindepth_files) > 0:
        logger.info('raw mindepth_files: {}'.format(len(mindepth_files)))
        mindepth_files = mindepth_files[::skip]
        assert(len(mindepth_files) == cam_cnt)
    else:
        mindepth_files = [None, ] * cam_cnt

    # assume all images have the same size as training image
    train_imgfile = find_files('{}/{}/train/rgb'.format(basedir, scene), exts=['*.png', '*.jpg'])[0]
    train_im = imageio.imread(train_imgfile)
    H, W = train_im.shape[:2]

    # create ray samplers
    ray_samplers = []
    for i in range(cam_cnt):
        intrinsics = parse_txt(intrinsics_files[i])
        pose = parse_txt(pose_files[i])
        loc = parse_txt_loc(loc_files[i])
        # print(loc)

        # read max depth
        try:
            max_depth = float(open('{}/max_depth.txt'.format(split_dir)).readline().strip())
        except:
            max_depth = None
        # print("LOC IS")
        # print(loc)
        ray_samplers.append(RaySamplerSingleImage(H=H, W=W, intrinsics=intrinsics, c2w=pose, box_loc=loc,
                                                  img_path=img_files[i],
                                                  mask_path=mask_files[i],
                                                  min_depth_path=mindepth_files[i],
                                                  max_depth=max_depth))

    logger.info('Split {}, # views: {}'.format(split, cam_cnt))

    return ray_samplers

def load_pose_data(basedir):
    from scipy.spatial.transform import Rotation as R
    #
    # arr = np.load(os.path.join(basedir, 'expert_trajectory.npy'))
    # loc_arr = arr[:, :2]
    # yaw = arr[:, 2]

    loc_arr, yaw = gen_poses()

    z = np.ones((loc_arr.shape[0], 1)) * 2.8
    loc_arr = np.concatenate((loc_arr, z), axis=1)
    loc_arr = np.expand_dims(loc_arr, axis=2)
    # print(loc_arr[0])


    rotations = R.from_euler('z', yaw).as_matrix()
    rot_arr = rotations[:,:,[1,2,0]]
    rot_arr[:, :, 1] = -rot_arr[:, :, 1]
    # print(rot_arr[0])

    poses_arr = np.concatenate((rot_arr, loc_arr), axis=2)
    # print(poses_arr[0])

    bottom = np.tile(np.array([0, 0, 0, 1]), (poses_arr.shape[0], 1, 1))
    # print(bottom[0])
    # print(bottom)

    final_product = np.concatenate((poses_arr, bottom), axis=1)

    # plot_mult_pose([final_product], 'input poses nerf no normalized ++',
    #                ['scene poses'])


    max = np.array([100., 140., 2.8])
    min = np.array([85., 125., 2.8])
    max_minus_min = max-min

    final_product[:, :3, 3] -= min
    final_product[:, :2, 3] /= (max_minus_min)[:2]
    avg_pos = np.array([0.5,0.5,0.])
    final_product[:, :3, 3] -= avg_pos
    final_product[:, :3, 3] *= 0.5

    h, w, f = 360, 640, 380
    # In this case Fx and Fy are the same since the pixel aspect
    # ratio is 1
    K = np.identity(4)
    K[0, 0] = K[1, 1] = f
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0

    # for i in range(200):
    #     final_product[i, :2, 3] = (final_product[i, :2,3] / 0.5 + final_product) * (max - min) + min
        # locs[i][:2] = (locs[i][:2] / 0.5 + avg_pose) * (max - min) + min
    # if not have_box:
    # plot_mult_pose([final_product], 'input poses nerf ++',
    #                 ['scene poses'])

    return final_product.astype(np.double), K.astype(np.double)

def load_data_split_video(basedir, scene, split, skip=1, try_load_min_depth=True, only_img_files=False,
                          train_seg=False, train_depth=False, have_box=True, box_number=10):
    def parse_txt(filename):
        assert os.path.isfile(filename)
        nums = open(filename).read().split()
        return np.array([float(x) for x in nums]).reshape([4, 4]).astype(np.float32)

    def parse_txt_loc(filename):
        assert os.path.isfile(filename)
        nums = open(filename).read().split()
        return np.array([float(x) for x in nums]).reshape([1, 2]).astype(np.float32)

    if basedir[-1] == '/':  # remove trailing '/'
        basedir = basedir[:-1]

    split_dir = '{}/{}/{}'.format(basedir, scene, split)

    if only_img_files:
        img_files = find_files('{}/rgb'.format(split_dir), exts=['*.png', '*.jpg'])
        return img_files

    trajectory_pose, intrinsic_pose = load_pose_data(basedir)

    # camera parameters files
    intrinsics_files = find_files('{}/intrinsics'.format(split_dir), exts=['*.txt'])
    pose_files = find_files('{}/pose'.format(split_dir), exts=['*.txt'])
    loc_files = find_files('{}/loc'.format(split_dir), exts=['*.txt'])
    logger.info('raw intrinsics_files: {}'.format(len(intrinsics_files)))
    logger.info('raw pose_files: {}'.format(len(pose_files)))

    intrinsics_files = intrinsics_files[::skip]
    pose_files = pose_files[::skip]
    loc_files = loc_files[::skip]
    cam_cnt = len(pose_files)

    # img files
    img_files = find_files('{}/rgb'.format(split_dir), exts=['*.png', '*.jpg'])
    if len(img_files) > 0:
        logger.info('raw img_files: {}'.format(len(img_files)))
        img_files = img_files[::skip]
        assert (len(img_files) == cam_cnt)
    else:
        img_files = [None, ] * cam_cnt

    # mask files
    mask_files = find_files('{}/mask'.format(split_dir), exts=['*.png', '*.jpg'])
    if len(mask_files) > 0:
        logger.info('raw mask_files: {}'.format(len(mask_files)))
        mask_files = mask_files[::skip]
        assert (len(mask_files) == cam_cnt)
    else:
        mask_files = [None, ] * cam_cnt

    # min depth files
    mindepth_files = find_files('{}/min_depth'.format(split_dir), exts=['*.png', '*.jpg'])
    if try_load_min_depth and len(mindepth_files) > 0:
        logger.info('raw mindepth_files: {}'.format(len(mindepth_files)))
        mindepth_files = mindepth_files[::skip]
        assert (len(mindepth_files) == cam_cnt)
    else:
        mindepth_files = [None, ] * cam_cnt

    if train_seg:
        seg_files = find_files('{}/seg'.format(split_dir), exts=['*.png', '*.jpg'])
        seg_files = seg_files[::skip]
        assert (len(seg_files) == cam_cnt)

    else:
        seg_files = [None, ] * cam_cnt

    if train_depth:
        depth_files = find_files('{}/depth'.format(split_dir), exts=['*.png', '*.jpg'])
        logger.info('raw depth_files: {}'.format(len(depth_files)))
        depth_files = depth_files[::skip]
        assert (len(depth_files) == cam_cnt)
    else:
        depth_files = [None, ] * cam_cnt

    cube_loc = gen_box_locs(trajectory_pose.shape[0], box_number)

    # assume all images have the same size as training image
    train_imgfile = find_files('{}/{}/train/rgb'.format(basedir, scene), exts=['*.png', '*.jpg'])[0]
    train_im = imageio.imread(train_imgfile)
    H, W = train_im.shape[:2]

    # create ray samplers
    ray_samplers = []
    # for i in range(cam_cnt):
    for i in range(trajectory_pose.shape[0]):
        intrinsics = parse_txt(intrinsics_files[0])
        pose = parse_txt(pose_files[0])
        print("________________________")
        print(pose.dtype)
        posex = trajectory_pose[i].astype(np.float32)
        print(posex.dtype)
        print("++++++++++++++++++++++++")

        # if len(loc_files) is not 0:
        #
        #     loc = parse_txt_loc(loc_files[0])
        # else:
        #     loc = None
        loc = cube_loc[i]
        # print(loc)

        # read max depth
        try:
            max_depth = float(open('{}/max_depth.txt'.format(split_dir)).readline().strip())
        except:
            max_depth = None
        # print("LOC IS")
        # print(loc)
        # ray_samplers.append(RaySamplerSingleImage(H=H, W=W, intrinsics=intrinsics, c2w=posex, box_loc=loc,
        #                                           img_path=img_files[i],
        #                                           mask_path=mask_files[i],
        #                                           min_depth_path=mindepth_files[i],
        #                                           max_depth=max_depth))

        ray_samplers.append(RaySamplerSingleImage(H=H, W=W, intrinsics=intrinsics, c2w=posex,
                                                  img_path=None,
                                                  mask_path=None,
                                                  min_depth_path=None,
                                                  max_depth=max_depth, box_loc=loc,
                                                  depth_path=None,
                                                  seg_path=None,
                                                  make_class_label=False))

    # max = np.array([100., 140.])
    # min = np.array([85., 125.])
    # avg_pose = np.array([0.5, 0.5])
    #
    # for i in range(trajectory_pose.shape[0]):
    #     trajectory_pose[i,:2, 3] = (trajectory_pose[i,:2, 3]/ 0.5 + avg_pose) * (max - min) + min
    #
    #     if have_box:
    #         cube_loc[i,:2] = (cube_loc[i,:2] / 0.5 + avg_pose) * (max - min) + min
    # if not have_box:
    #     plot_mult_pose([trajectory_pose], 'input poses nerf ++',
    #                 ['scene poses'])
    # else:
    #     dummy_pose_loc = np.zeros((trajectory_pose.shape))
    #     # locs = np.stack(locs, axis=0)
    #     dummy_pose_loc[:,:3, 3] = cube_loc
    #     plot_mult_pose([trajectory_pose, dummy_pose_loc], 'input poses {} nerf ++'.format(split),
    #                    ['scene poses','box'])

    logger.info('Split {}, # views: {}'.format(split, cam_cnt))

    return ray_samplers

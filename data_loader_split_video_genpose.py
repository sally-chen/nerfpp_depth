import os
import numpy as np
import imageio
import logging
from nerf_sample_ray_split import RaySamplerSingleImage
import glob
from helpers import plot_mult_pose
import random
import pickle
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

def load_data_array(intrs, poses, locs, H, W, plot, normalize=True):

    if plot:
        dummy_pose_loc = np.zeros((np.stack(poses, axis=0).shape))
        locs = np.stack(locs, axis=0)
        dummy_pose_loc[:, :3, 3] = locs
        plot_mult_pose([np.stack(poses, axis=0), dummy_pose_loc], 'input poses nerf ++',
                       ['scene poses', 'box'])

    ray_samplers = []

    for i in range(len(poses)):
        intrinsics = intrs[i]
        pose = poses[i]
        loc = locs[i]

        if normalize:
            max = np.array([100., 140.])
            min = np.array([85., 125.])
            avg_pose = np.array([0.5, 0.5])

            pose[:2, 3] = ((pose[:2, 3] - min ) / (max - min) - avg_pose ) * 0.5
            loc[:2] = ((loc[:2] - min ) / (max - min) - avg_pose ) * 0.5


        ray_samplers.append(RaySamplerSingleImage(H=H, W=W, intrinsics=intrinsics, c2w=pose, box_loc=loc))



    return ray_samplers
    # max_depth=max_depth, box_loc=locs[i]))

def load_data_split(basedir, scene, split, skip=1, try_load_min_depth=True, only_img_files=False, have_box = False, train_depth=False):

    def parse_txt(filename):
        assert os.path.isfile(filename)
        nums = open(filename).read().split()
        return np.array([float(x) for x in nums]).reshape([4, 4]).astype(np.float32)

    split_dir = '{}/{}/{}'.format(basedir, scene, split)

    if only_img_files:
        img_files = find_files('{}/rgb'.format(split_dir), exts=['*.png', '*.jpg'])
        return img_files




    # camera parameters files
    intrinsics_files = find_files('{}/intrinsics'.format(split_dir), exts=['*.txt'])
    pose_files = find_files('{}/pose'.format(split_dir), exts=['*.txt'])
    logger.info('raw intrinsics_files: {}'.format(len(intrinsics_files)))
    logger.info('raw pose_files: {}'.format(len(pose_files)))




    intrinsics_files = intrinsics_files[::skip]
    pose_files = pose_files[::skip]
    cam_cnt = len(pose_files)

    def parse_txt_loc(filename, norm=False):
        nums = open(filename).read().split()
        nums = np.array([float(x) for x in nums])

        if norm:
            max = np.array([100,140])
            min = np.array([85, 125])
            avg_pos = np.array([0.5, 0.5])

            nums -= min
            nums /= (max-min)
            nums -= avg_pos
            nums *= 0.5

        nums_new = np.zeros([3])
        # nums_new[:2] = nums + random.randint(-20,20)/100.
        nums_new[:2] = nums
        nums_new[2] = -1
        return nums_new.astype(np.float32)

    if have_box:
        # locs = np.load('{}/box_loc.npy'.format(split_dir))
        # locs = locs[::skip]

        loc_files = find_files('{}/loc'.format(split_dir), exts=['*.txt'])
        loc_files = loc_files[::skip]






    # img files
    img_files = find_files('{}/rgb'.format(split_dir), exts=['*.png', '*.jpg'])

    if train_depth:
        depth_files = find_files('{}/depth'.format(split_dir), exts=['*.png', '*.jpg'])
        logger.info('raw depth_files: {}'.format(len(depth_files)))
        depth_files = depth_files[::skip]
        assert (len(depth_files) == cam_cnt)
    else:
        depth_files = [None, ] * cam_cnt

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
    poses = []
    intrins = []
    locs = []
    for i in range(cam_cnt):
        intrinsics = parse_txt(intrinsics_files[i])
        pose = parse_txt(pose_files[i])

        ################## rand ##############33
        #pose[:2,3] = pose[:2,3] * 0.5
        ################## rand ##############33
        # pose[:2,3] = pose[:2,3]

        poses.append(pose)
        intrins.append(intrinsics)
        if have_box:
            loc = parse_txt_loc(loc_files[i])
            #loc[:2] += 0.1
            locs.append(loc)





        # read max depth
        try:
            max_depth = float(open('{}/max_depth.txt'.format(split_dir)).readline().strip())
        except:
            max_depth = None


        if have_box:
            ray_samplers.append(RaySamplerSingleImage(H=H, W=W, intrinsics=intrinsics, c2w=pose,
                                                      img_path=img_files[i],
                                                      mask_path=mask_files[i],
                                                      min_depth_path=mindepth_files[i],
                                                      max_depth=max_depth, box_loc=loc,
                                                      depth_path=depth_files[i]))
                                                      # max_depth=max_depth, box_loc=locs[i]))
        else:
            ray_samplers.append(RaySamplerSingleImage(H=H, W=W, intrinsics=intrinsics, c2w=pose,
                                                      img_path=img_files[i],
                                                      mask_path=mask_files[i],
                                                      min_depth_path=mindepth_files[i],
                                                      max_depth=max_depth, depth_path=depth_files[i]
                                                      ))

    logger.info('Split {}, # views: {}'.format(split, cam_cnt))

    #
    # if not have_box:
    #     plot_mult_pose([np.stack(poses, axis=0)], 'input poses nerf ++',
    #                    ['scene poses'])
    #
    # else:
    #     dummy_pose_loc = np.zeros((np.stack(poses, axis=0).shape))
    #     locs = np.stack(locs, axis=0)
    #     dummy_pose_loc[:,:3, 3] = locs
    #     plot_mult_pose([np.stack(poses, axis=0), dummy_pose_loc], 'input poses nerf ++',
    #                    ['scene poses','box'])
    #
    #
    # ## denorm first and save for test ##
    #
    # max = np.array([100., 140.])
    # min = np.array([85., 125.])
    # avg_pose = np.array([0.5, 0.5])
    #
    # for i, p in enumerate(poses):
    #     poses[i][:2, 3] = (p[:2,3] / 0.5 + avg_pose) * (max - min) + min
    #     locs[i][:2] = (locs[i][:2] / 0.5 + avg_pose) * (max - min) + min
    #
    #
    # filename = split_dir + '/sample_arrs'
    # outfile = open(filename, 'wb')
    # pickle.dump([poses, intrins, locs], outfile)
    # outfile.close()
    return ray_samplers

def gen_box_locs():
    loc = np.linspace((87,132),(90,135),35)
    max = np.array([100., 140., 2.8])
    min = np.array([85., 125., 2.8])
    max_minus_min = max-min
    avg_pos = np.array([0.5, 0.5, 0.0])
    loc -= min[:2]
    loc /= (max_minus_min)[:2]
    # print(loc)
    loc -= avg_pos[:2]
    loc *= 0.5

    return loc



def gen_poses():
    xy1 = np.linspace((120,128),(92,128),80)
    dg1 = (np.ones((80, )) * 180.) * math.pi / 180.
    turn1 = np.linspace((92,128),(92,128),10)
    turn1_dg = (np.arange(180.0, 270.0, 9) + np.random.uniform(low=-1.0, high=1.0, size=(10,))) * math.pi / 180.

    xy2 = np.linspace((92,128),(92,105),80)
    dg2 = (np.ones((80,)) * 270.) * math.pi / 180.
    turn2 = np.linspace((92,105),(92,105),10)
    turn2_dg = (np.arange(270.0, 180.0, -9) + np.random.uniform(low=-1.0, high=1.0, size=(10,))) * math.pi / 180.

    xy3 = np.linspace((92,105),(87,105),80)
    dg3 = (np.ones((80,)) * 180.) * math.pi / 180.
    turn3 = np.linspace((87,105),(87,105),10)
    turn3_dg = (np.arange(180.0, 90, -9) + np.random.uniform(low=-1.0, high=1.0, size=(10,))) * math.pi / 180.


    xy4 = np.linspace((87,105),(87,135),80)
    dg4 = (np.ones((80,)) * 90.) * math.pi / 180.
    # turn4 = np.linspace((87,135),(87,135),10)
    # turn4_dg = (np.arange(90.0, 0.0, -9) + np.random.uniform(low=-1.0, high=1.0, size=(10,))) * math.pi / 180.


    # xy5 = np.linspace((87,135),(120,135),80)
    # dg5 = (np.ones((80,)) * 0.) * math.pi / 180.

    return np.concatenate((xy1, turn1, xy2, turn2, xy3, turn3, xy4), axis=0), np.concatenate((dg1, turn1_dg, dg2, turn2_dg, dg3, turn3_dg, dg4), axis=0)

def load_pose_data(basedir):
    # from scipy.spatial.transform import Rotation as R

    # arr = np.load(os.path.join(basedir, 'expert_trajectory.npy'))
    # loc_arr = arr[:, :2]
    # z = np.ones((loc_arr.shape[0], 1)) * 2.8
    # loc_arr = np.concatenate((loc_arr, z), axis=1)
    # loc_arr = np.expand_dims(loc_arr, axis=2)
    # # print(loc_arr[0])

    # yaw = arr[:, 2]
    # rotations = R.from_euler('z', yaw).as_matrix()
    # rot_arr = rotations[:,:,[1,2,0]]
    # rot_arr[:, :, 1] = -rot_arr[:, :, 1]
    # # print(rot_arr[0])

    # poses_arr = np.concatenate((rot_arr, loc_arr), axis=2)
    # # print(poses_arr[0])

    # bottom = np.tile(np.array([0, 0, 0, 1]), (200, 1, 1))
    # # print(bottom[0])
    # # print(bottom)

    # final_product = np.concatenate((poses_arr, bottom), axis=1)
    # max = np.array([100., 140., 2.8])
    # min = np.array([85., 125., 2.8])
    # max_minus_min = max-min

    # final_product[:, :3, 3] -= min
    # final_product[:, :2, 3] /= (max_minus_min)[:2]
    # avg_pos = np.array([0.5, 0.5, 0.])
    # final_product[:, :3, 3] -= avg_pos
    # final_product[:, :3, 3] *= 0.5

    # h, w, f = final_product[0, :3, -1:]
    # # In this case Fx and Fy are the same since the pixel aspect
    # # ratio is 1
    # K = np.identity(4)
    # K[0, 0] = K[1, 1] = f
    # K[0, 2] = w / 2.0
    # K[1, 2] = h / 2.0

    # return final_product.astype(np.double), K.astype(np.double)

    from scipy.spatial.transform import Rotation as R

    loc_arr, yaw = gen_poses()

    # arr = np.load(os.path.join(basedir, 'expert_trajectory.npy'))
    # loc_arr = arr[:, :2]
    # print(loc_arr)
    z = np.ones((loc_arr.shape[0], 1)) * 2.8
    loc_arr = np.concatenate((loc_arr, z), axis=1)
    loc_arr = np.expand_dims(loc_arr, axis=2)

    # yaw = arr[:, 2]
    rotations = R.from_euler('z', yaw).as_matrix()
    rot_arr = rotations[:,:,[1,2,0]]
    rot_arr[:, :, 1] = -rot_arr[:, :, 1]
    # print(rot_arr)

    poses_arr = np.concatenate((rot_arr, loc_arr), axis=2)
    # print(poses_arr[0])

    exp = np.array([[0, 0, 0, 1]])
    bottom = np.tile(np.array([0, 0, 0, 1]), (350, 1, 1))
    # print(bottom[0])
    # print(bottom)

    final_product = np.concatenate((poses_arr, bottom), axis=1)
    max = np.array([100., 140., 2.8])
    min = np.array([85., 125., 2.8])
    max_minus_min = max-min

    final_product[:, :3, 3] -= min
    final_product[:, :2, 3] /= (max_minus_min)[:2]
    avg_pos = np.array([0.5, 0.5, 0.0])
    final_product[:, :3, 3] -= avg_pos
    final_product[:, :3, 3] *= 0.5

    h, w, f = final_product[0, :3, -1:]
    # In this case Fx and Fy are the same since the pixel aspect
    # ratio is 1
    K = np.identity(4)
    K[0, 0] = K[1, 1] = f
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0


    return final_product.astype(np.double), K.astype(np.double)


def load_data_split_video(basedir, scene, split, skip=1, try_load_min_depth=True, only_img_files=False):
    def parse_txt(filename):
        assert os.path.isfile(filename)
        nums = open(filename).read().split()
        return np.array([float(x) for x in nums]).reshape([4, 4]).astype(np.float32)

    # def parse_txt_loc(filename):
    #     assert os.path.isfile(filename)
    #     nums = open(filename).read().split()
    #     return np.array([float(x) for x in nums]).reshape([1, 2]).astype(np.float32)

    def parse_txt_loc(filename, norm=False):
        nums = open(filename).read().split()
        nums = np.array([float(x) for x in nums])

        if norm:
            max = np.array([100,140])
            min = np.array([85, 125])
            avg_pos = np.array([0.5, 0.5])

            nums -= min
            nums /= (max-min)
            nums -= avg_pos
            nums *= 0.5

        nums_new = np.zeros([3])
        # nums_new[:2] = nums + random.randint(-20,20)/100.
        nums_new[:2] = nums
        nums_new[2] = -1
        return nums_new.astype(np.float32)

    def parse_cube_loc(input):
        nums_new = np.zeros([3])
        # nums_new[:2] = nums + random.randint(-20,20)/100.
        nums_new[:2] = input
        nums_new[2] = -1
        return nums_new.astype(np.float32)



    if basedir[-1] == '/':  # remove trailing '/'
        basedir = basedir[:-1]

    split_dir = '{}/{}/{}'.format(basedir, scene, split)

    loc_files = find_files('{}/loc'.format(split_dir), exts=['*.txt'])
    loc_files = loc_files[::skip]


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
    cam_cnt = len(loc_files)

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

    cube_loc = gen_box_locs()

    # assume all images have the same size as training image
    train_imgfile = find_files('{}/{}/train/rgb'.format(basedir, scene), exts=['*.png', '*.jpg'])[0]
    train_im = imageio.imread(train_imgfile)
    H, W = train_im.shape[:2]

    # create ray samplers
    ray_samplers = []
    # for i in range(cam_cnt):
    for i in range(350):
        intrinsics = parse_txt(intrinsics_files[0])
        posex = trajectory_pose[i].astype(np.float32)
        # loc = parse_txt_loc(loc_files[0])
        # # print(loc)
        # loc = parse_txt_loc(loc_files[0])
        loc = parse_cube_loc(cube_loc[i//10])
        print("LOC")
        print(loc)

        # read max depth
        try:
            max_depth = float(open('{}/max_depth.txt'.format(split_dir)).readline().strip())
        except:
            max_depth = None
        # print("LOC IS")
        # print(loc)
        ray_samplers.append(RaySamplerSingleImage(H=H, W=W, intrinsics=intrinsics, c2w=posex, box_loc=loc,
                                                  img_path=img_files[0],
                                                  mask_path=mask_files[0],
                                                  min_depth_path=mindepth_files[0],
                                                  max_depth=max_depth))

    logger.info('Split {}, # views: {}'.format(split, cam_cnt))

    return ray_samplers
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


def box_loc_from_txt(count=50, box_number=10):

    box_props = np.array([ 98.863   ,125.34992 ,  1.786512,  0.124561,  0.09506 ,  0.674434,
   1.608457,  1.38573 ,  1.159217, -0.367203,-19.261133,-22.64998 ,
  98.611984,120.35097 ,  0.978648,  0.719407,  0.964313,  0.016076,
   1.24886 ,  1.331308,  1.186513, 15.027966, 22.126894, -3.96351 ,
  97.05769 ,118.040504,  0.769688,  0.36452 ,  0.205555,  0.309259,
   1.725814,  1.899079,  1.26358 , 19.352983,-24.408157,  8.496014,
  98.77501 ,127.32707 ,  1.734848,  0.817463,  0.635946,  0.996193,
   1.221624,  1.678531,  1.458348, 12.843408, -4.735205,  1.275918,
  95.58389 ,117.241104,  3.840275,  0.193399,  0.765102,  0.537566,
   1.083747,  1.065293,  1.087662, -4.203038, -0.571052, -0.551875,
  95.00536 ,116.738266,  4.399919,  0.526893,  0.237533,  0.266745,
   1.0818  ,  1.064159,  1.049099,  4.072644, -3.405717,  0.020287,
  93.59534 ,119.71914 ,  3.515487,  0.277713,  0.040806,  0.807433,
   1.079355,  1.051508,  1.058991,  1.367564, -0.676404, -1.146593,
  93.673836,116.66906 ,  4.207787,  0.985084,  0.992383,  0.368961,
   1.027816,  1.055402,  1.031426,  3.248581,  4.195982,  2.043893,
  94.189674,117.064445,  3.570103,  0.84365 ,  0.545266,  0.621931,
   1.065653,  1.056373,  1.000977, -2.784878, -3.016634, -0.614177,
  93.93175 ,119.01718 ,  4.366326,  0.422892,  0.477283,  0.398576,
   1.007378,  1.027783,  1.037599, -2.991591,  1.11616 ,  3.839408,
  94.66836 ,117.84271 ,  4.229572,  0.489019,  0.488361,  0.643265,
   1.07982 ,  1.084895,  1.082835, -1.061113,  2.840016, -0.463086,
  87.796974,128.67583 ,  0.620225,  0.425344,  0.945928,  0.720886,
   1.449482,  1.276958,  1.000375,  5.72198 ,-20.893444,-11.427114,
  87.818924,116.68828 ,  1.553515,  0.68468 ,  0.492555,  0.546187,
   1.865345,  1.74898 ,  1.248352,-14.865788, -0.147108, -6.17542 ,
  87.53005 ,117.04619 ,  0.828899,  0.390882,  0.267236,  0.998679,
   1.023151,  1.51043 ,  1.467531, -6.659768,-21.50057 ,-27.167158,
  88.979614,125.16991 ,  1.159713,  0.671381,  0.297712,  0.436971,
   1.850786,  1.575423,  1.508775, 21.700033, 19.290792,-27.110655]).reshape([15,12])
    loc = box_props[:,:3]
    
 
    props = box_props[:,3:]
    props[:,3:6] /= 2
    loc[:,2] -=2.5
    loc[:,0] -=5.

    max = np.array([100., 140., 2.8])
    min = np.array([85., 125., 2.8])
    max_minus_min = max-min
    avg_pos = np.array([0.5, 0.5, 0.0])
    loc[...,:2] -= min[:2]
    loc[...,:2] /= (max_minus_min)[:2]
    # print(loc)![](logs/box_sample192_rgb64_6box_transm25-m10/render_test_525000/000008.png)
    loc[...,:2] -= avg_pos[:2]
    loc[...,:2] *= 0.5
    z_box_scene = 1.
    z_cam_scene = 2.8



    loc[:, 2:] = (loc[:, 2:] - z_cam_scene) / 30.
    
    return loc,props

def gen_box_locs(count=55, box_number=10):
    # loc = np.linspace((87,132),(90,135),35)

    # loc = np.concatenate((np.random.uniform(87., 90., (count,1)),
    #                       np.random.uniform(132., 135., (count,1))), axis=1)

    ct = 50
    np.random.seed(123)
    loc1 = np.concatenate((np.random.uniform(80.0, 81.0, (ct, 1, 1)),
                            np.random.uniform(112., 113., (ct, 1, 1)),
                          np.random.uniform(2.45, 2.5, (ct,1, 1))), axis=-1)

    loc2 = np.concatenate((np.random.uniform(92.0, 93.0, (ct, 1, 1)),
                            np.random.uniform(119., 120., (ct, 1, 1)),
                          np.random.uniform(5.5, 5.55, (ct,1, 1))), axis=-1)
    
    loc3 = np.concatenate((np.random.uniform(100.0, 101.0, (ct, 1, 1)),
                            np.random.uniform(112., 113., (ct, 1, 1)),
                          np.random.uniform(2.45, 2.5, (ct,1, 1))), axis=-1)

    loc = np.concatenate([loc1,loc2, loc3], axis=1)

    max = np.array([100., 140., 2.8])
    min = np.array([85., 125., 2.8])
    max_minus_min = max-min
    avg_pos = np.array([0.5, 0.5, 0.0])
    loc[...,:2] -= min[:2]
    loc[...,:2] /= (max_minus_min)[:2]
    # print(loc)![](logs/box_sample192_rgb64_6box_transm25-m10/render_test_525000/000008.png)
    loc[...,:2] -= avg_pos[:2]
    loc[...,:2] *= 0.5
    
#     z_box_scene = 1.
    z_cam_scene = 2.8

#     z_box_normed = (z_box_scene - z_cam_scene) / 30.

    loc[...,2]  = (loc[...,2] - z_cam_scene) / 30.
    # loc = np.concatenate((loc, -0.06 * np.ones((count,box_number,1))), axis=-1)
    # loc = np.concatenate((loc, -1. * np.ones((count,box_number,1))), axis=-1)

    return loc

def parse_cube_loc(input):
    nums_new = np.zeros([3])
    # nums_new[:2] = nums + random.randint(-20,20)/100.
    nums_new[:2] = input
    nums_new[2] = -1
    return nums_new.astype(np.float32)

def gen_poses():


    # xy1 = np.linspace((120,128),(92,128),80)
    xy1 = np.linspace((105,128),(92,128),3)
    dg1 = (np.ones((3, )) * 180.) * math.pi / 180.

    turn1 = np.linspace((92,128),(92,128),5)
    turn1_dg = (np.arange(180.0, 270.0, 18) + np.random.uniform(low=-1.0, high=1.0, size=(5,))) * math.pi / 180.

    xy2 = np.linspace((92.4,128),(90,111),10)
    #xy2 = np.linspace((92,122.5),(92,119.5),16)
    dg2 = (np.ones((10,)) * 270. +  np.random.uniform(-5., 5., (10,  ))) * math.pi  / 180.

    # turn2 = np.linspace((92,108),(92,108),2)
    # turn2_dg = (np.arange(270.0, 180.0, -18) + np.random.uniform(low=-1.0, high=1.0, size=(5,))) * math.pi / 180.

    # xy3 = np.linspace((92,108),(91,108),5)
    # dg3 = (np.ones((10,)) * 180.) * math.pi / 180.

    turn3 = np.linspace((92.4,112),(92.4,112),5)
    turn3_dg = (np.arange(180.0, 90, -18) + np.random.uniform(low=-1.0, high=1.0, size=(5,))) * math.pi / 180.


    xy4 = np.linspace((92.4,112),(92.4,128),27)
    dg4 = (np.ones((27,)) * 90. + np.random.uniform(-5., 5., (27,  )) ) * math.pi / 180.
    # turn4 = np.linspace((87,135),(87,135),10)
    # turn4_dg = (np.arange(90.0, 0.0, -9) + np.random.uniform(low=-1.0, high=1.0, size=(10,))) * math.pi / 180.


    # xy5 = np.linspace((87,135),(120,135),80)
    # dg5 = (np.ones((80,)) * 0.) * math.pi / 180.

    return np.concatenate((xy1, turn1, xy2, turn3, xy4), axis=0), np.concatenate((dg1, turn1_dg, dg2, turn3_dg, dg4), axis=0)


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
                          train_seg=False, train_depth=False, have_box=True, box_number=10,  box_props_path='' ):
  
    if basedir[-1] == '/':  # remove trailing '/'
        basedir = basedir[:-1]

    split_dir = '{}/{}/{}'.format(basedir, scene, split)

    trajectory_pose, intrinsic_pose = load_pose_data(basedir)

    cube_loc, props = box_loc_from_txt()
    cube_loc = gen_box_locs(trajectory_pose.shape[0], box_number)

    H, W = 360, 640

    ray_samplers = []

    for i in range(trajectory_pose.shape[0]):
        intrinsics = intrinsic_pose.astype(np.float32)
        posex = trajectory_pose[i].astype(np.float32)
      

        loc = cube_loc[i]
        # print(loc)
        
        if not have_box:
            loc = None

        ray_samplers.append(RaySamplerSingleImage(H=H, W=W, intrinsics=intrinsics, c2w=posex,
                                                  img_path=None,
                                                  mask_path=None,
                                                  min_depth_path=None,
                                                  max_depth=None, box_loc=loc,
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


    return ray_samplers, props

import os
import torch
import numpy as np
import imageio
import logging
from nerf_sample_ray_split import RaySamplerSingleImage
import glob
from helpers import plot_mult_pose
import random
import pickle

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

def load_data_array(intrs, poses, locs, H, W, plot, normalize=True, lidar_image=True):

    if plot:
        dummy_pose_loc = np.zeros((np.stack(poses, axis=0).shape))
        locs = np.stack(locs, axis=0)
        dummy_pose_loc[:, :3, 3] = locs
        plot_mult_pose([np.stack(poses, axis=0), dummy_pose_loc], 'input poses nerf ++',
                       ['scene poses', 'box'])

    ray_samplers = []

    for i in range(len(poses)):
        intrinsics = intrs[i]
        pose = poses[i].clone()
        loc = locs[i].clone()

        if normalize:
            max = torch.tensor([100., 140.]).type_as(pose)
            min = torch.tensor([85., 125.]).type_as(pose)
            avg_pose = torch.tensor([0.5, 0.5]).type_as(pose)
            print(pose.shape, loc.shape)

            pose[:2, 3] = ((pose[:2, 3] - min ) / (max - min) - avg_pose ) * 0.5
            loc[:2] = ((loc[:2] - min ) / (max - min) - avg_pose ) * 0.5


        ray_samplers.append(RaySamplerSingleImage(H=H, W=W, intrinsics=intrinsics, c2w=pose, box_loc=loc, lidar_scan=lidar_image))



    return ray_samplers
    # max_depth=max_depth, box_loc=locs[i]))

def load_data_split(basedir, scene, split, skip=1, try_load_min_depth=True, only_img_files=False, have_box = False,
                    train_depth=False, train_seg = False,  train_box_only=False, custom_rays=None):

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

        # !!! CHANGE from -1 to z-normlized value
        z_box_scene = 1.
        z_cam_scene = 2.8

        z_box_normed = (z_box_scene - z_cam_scene) / 30.

        nums_new[2] = z_box_normed
        return nums_new.astype(np.float32)

    if have_box:
        # locs = np.load('{}/box_loc.npy'.format(split_dir))
        # locs = locs[::skip]

        loc_files = find_files('{}/loc'.format(split_dir), exts=['*.txt'])
        loc_files = loc_files[::skip]






    # img files
    img_files = find_files('{}/rgb'.format(split_dir), exts=['*.png', '*.jpg'])

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
    #H, W = 90, 160

    if custom_rays:
        import pickle
        ray_samplers = []
        rays_li = pickle.load(open('./ray_list', 'rb'))

        for ray in rays_li:
            loc = np.array([ -0.2477,-0.0833, -1.])
            ray_samplers.append(RaySamplerSingleImage(H=100, W=32, rays=ray, box_loc = loc))

        return ray_samplers



    # create ray samplers
    ray_samplers = []
    poses = []
    intrins = []
    locs = []

    # tmp ##
    # if cam_cnt > 5:
    #     cam_cnt = 5

    ## tmp ##

    max = np.array([100., 140.])
    min = np.array([85., 125.])
    avg_pose = np.array([0.5, 0.5])

    count = 0
    for i in range(cam_cnt):

        print(i)
        intrinsics = parse_txt(intrinsics_files[i])
        pose = parse_txt(pose_files[i])

        # p = (pose[:2, 3] / 0.5 + avg_pose) * (max - min) + min
        # if (p[0] > 95. and p[1] > 136) or (p[0] > 95. and p[1] < 128) or (p[0] > 105) or (p[0] < 86):
        #     continue
        count += 1
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



        if train_box_only:

            ray_samplers.append(RaySamplerSingleImage(H=H, W=W, intrinsics=intrinsics, c2w=pose,
                                                      img_path=img_files[i],
                                                      mask_path=mask_files[i],
                                                      min_depth_path=mindepth_files[i],
                                                      max_depth=max_depth, box_loc=None,
                                                      depth_path=depth_files[i],
                                                      seg_path=seg_files[i], train_box_only=True,
                                                      make_class_label=True))


        elif have_box:
            ray_samplers.append(RaySamplerSingleImage(H=H, W=W, intrinsics=intrinsics, c2w=pose,
                                                      img_path=img_files[i],
                                                      mask_path=mask_files[i],
                                                      min_depth_path=mindepth_files[i],
                                                      max_depth=max_depth, box_loc=loc,
                                                      depth_path=depth_files[i],
                                                      seg_path = None,#seg_files[i],
                                                    make_class_label=False))

        else:
            ray_samplers.append(RaySamplerSingleImage(H=360, W=640, intrinsics=intrinsics, c2w=pose,
                                                      img_path=img_files[i],
                                                      mask_path=mask_files[i],
                                                      min_depth_path=mindepth_files[i],
                                                      seg_path=seg_files[i],
                                                      max_depth=max_depth, depth_path=depth_files[i],
                                                      make_class_label=False))


    # for i, p in enumerate(poses):
    #     poses[i][:2, 3] = (p[:2,3] / 0.5 + avg_pose) * (max - min) + min
    #
    #     if have_box:
    #         locs[i][:2] = (locs[i][:2] / 0.5 + avg_pose) * (max - min) + min
    # if not have_box:
    #     plot_mult_pose([np.stack(poses, axis=0)], 'input poses nerf ++',
    #                 ['scene poses'])
    # #
    # else:
    #     dummy_pose_loc = np.zeros((np.stack(poses, axis=0).shape))
    #     locs = np.stack(locs, axis=0)
    #     dummy_pose_loc[:,:3, 3] = locs
    #     plot_mult_pose([np.stack(poses, axis=0), dummy_pose_loc], 'input poses {} nerf ++'.format(split),
    #                    ['scene poses','box'])


    #
    # filename = split_dir + '/sample_arrs'
    # outfile = open(filename, 'wb')
    # pickle.dump([poses, intrins, locs], outfile)
    # outfile.close()

    print("finished getting rays")
    print("[Number of image: {}]".format(count))
    return ray_samplers

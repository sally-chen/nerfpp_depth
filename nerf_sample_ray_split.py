import numpy as np
from collections import OrderedDict
import torch
import cv2
import imageio
from PIL import Image
import math
from ddp_model import depth2pts_outside, depth2pts_outside_np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
import pickle

visualize_depth = True
import os
import zarr


########################################################################################################################
# ray batch sampling
########################################################################################################################

def intersect_sphere(ray_o, ray_d):
    '''
    ray_o, ray_d: [..., 3]
    compute the depth of the intersection point between this ray and unit sphere

    NOTICE: ray_d here is world coordinate, not normalized;
    result is t = d1 + d1, ||o + td|| = 1 (this means norm of this is 1 , and is with respect to the un-normalized ray_d;
            but also in camera coordinate z is 1, so t here measures length )
            Technically this is the distance to camera as well, as when transformation happens, euclidea distance between 2 points do not change
            and the z value in camera coordinate should always be 1 thus this value should indeed be the distance on z axis to the camera centre (depth map)

    Issue is:  i know the far plane at 4 / norm=3 ~ 1.33 should be within the distance to unit sphere;
                and if t really is the distance from camera to unit sphere, why is it in the range of 0.5~0.8?
                In our problem setting at least 1 ray would intersect with world origin making t > 1
    '''
    # note: d1 becomes negative if this mid point is behind camera
    d1 = -np.sum(ray_d * ray_o, axis=-1) / np.sum(ray_d * ray_d, axis=-1)  # project ray_o on ray_d
    p = ray_o + d1[..., None] * ray_d
    # consider the case where the ray does not intersect the sphere
    ray_d_cos = 1. / np.linalg.norm(ray_d, axis=-1)

    p_sum = np.sum(p * p, axis=-1)

    d2 = np.sqrt(1. - np.sum(p * p, axis=-1)) * ray_d_cos

    # inters = ray_o + (d1 + d2)[:,None] * ray_d
    # inters_norm = np.linalg.norm(inters, axis=1)
    return d1 + d2


def get_rays_single_image(H, W, intrinsics, c2w):
    '''
    :param H: image height
    :param W: image width
    :param intrinsics: 4 by 4 intrinsic matrix
    :param c2w: 4 by 4 camera to world extrinsic matrix
    :return:
    '''
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    u = u.reshape(-1).astype(dtype=np.float32) + 0.5  # add half pixel
    v = v.reshape(-1).astype(dtype=np.float32) + 0.5
    pixels = np.stack((u, v, np.ones_like(u)), axis=0)  # (3, H*W)

    rays_d = np.dot(np.linalg.inv(intrinsics[:3, :3]), pixels) # sensor's location to camera
    #norm_0 = np.linalg.norm(rays_d.transpose((1, 0)), axis=-1)

    rays_d = np.dot(c2w[:3, :3], rays_d)  # (3, H*W)
    rays_d = rays_d.transpose((1, 0))  # (H*W, 3)

    #norm_1 = np.linalg.norm(rays_d, axis=-1)

    rays_o = c2w[:3, 3].reshape((1, 3))
    rays_o = np.tile(rays_o, (rays_d.shape[0], 1))  # (H*W, 3)

    ray_d_new = rays_d - rays_o
    #norm_2 = np.linalg.norm(ray_d_new, axis=-1)

    depth = np.linalg.inv(c2w)[2, 3]
    depth = depth * np.ones((rays_o.shape[0],), dtype=np.float32)  # (H*W,)

    return rays_o, rays_d, depth


class RaySamplerSingleImage(object):
    def __init__(self, H, W, intrinsics=None, c2w=None,
                 img_path=None,
                 resolution_level=1,
                 mask_path=None,
                 min_depth_path=None,
                 max_depth=None,
                 box_loc=None,
                 depth_path=None,
                 rays=None,
                 seg_path=None,
                 make_class_label=None,
                 train_box_only=False,
                 ):
        super().__init__()

        self.W_orig = W
        self.H_orig = H
        self.intrinsics_orig = intrinsics
        self.c2w_mat = c2w

        self.img_path = img_path
        self.depth_path = depth_path
        self.mask_path = mask_path
        self.min_depth_path = min_depth_path
        self.max_depth = max_depth

        self.seg_path = seg_path
        self.train_box_only = train_box_only

        self.resolution_level = -1
        self.set_resolution_level(resolution_level, rays=rays)

        self.box_loc = box_loc

        if img_path is not None:
            number = self.img_path[-9:-4]
            self.label_path = self.img_path[:-13] + '/cube_scene_filt9_z5/' + number
        else:
            self.label_path = None
        #
        # p = os.path.normpath(self.img_path[:-13])
        # fol = p.split(os.sep)[-1]
        # self.label_path = '/home/sally/data/' +fol + '/class_label_sameseg_rayd128_filt7_seg/' + number

        if make_class_label:

            # os.makedirs('/home/sally/data/' +fol+ '/class_label_sameseg_rayd128_filt7_seg/', exist_ok=True)
            os.makedirs(self.img_path[:-13] + '/cube_scene_filt9_z5/', exist_ok=True)
            if self.train_box_only:
                self.get_classifier_label_torch_boxonly(N_front_sample=128, pretrain=True, save=True,
                                                        train_box_only=self.train_box_only)

            else:
                self.get_classifier_label_torch(N_front_sample=128, N_back_sample=128, pretrain=True, save=True)

    def set_resolution_level(self, resolution_level, rays=None) -> object:

        if resolution_level != self.resolution_level:
            self.resolution_level = resolution_level
            self.W = self.W_orig // resolution_level
            self.H = self.H_orig // resolution_level

            if rays is None:
                self.intrinsics = np.copy(self.intrinsics_orig)
                self.intrinsics[:2, :3] /= resolution_level
            # only load image at this time
            if self.img_path is not None:

                if self.train_box_only:
                    self.img = np.array(imageio.imread(self.img_path).astype(np.float32)[:, :, :3])
                    self.img[self.img < 8.] = 255.
                    # self.img[self.img > 252.] = 0.

                    self.img /= 255.
                else:
                    self.img = imageio.imread(self.img_path).astype(np.float32) / 255.
                self.img = cv2.resize(self.img, (self.W, self.H), interpolation=cv2.INTER_AREA)
                self.img = self.img.reshape((-1, 3))
            else:
                self.img = None

            if self.mask_path is not None:
                self.mask = imageio.imread(self.mask_path).astype(np.float32) / 255.
                self.mask = cv2.resize(self.mask, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
                self.mask = self.mask.reshape((-1))
            else:
                self.mask = None

            if self.min_depth_path is not None:
                self.min_depth = imageio.imread(self.min_depth_path).astype(np.float32) / 255. * self.max_depth + 1e-4
                self.min_depth = cv2.resize(self.min_depth, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
                self.min_depth = self.min_depth.reshape((-1))
            else:
                self.min_depth = None

            if rays is not None:
                self.rays_o, self.rays_d, self.depth = rays[0].cpu().detach().numpy(), \
                                                       rays[1].cpu().detach().numpy(), \
                                                       rays[2].cpu().detach().numpy()
            else:
                self.rays_o, self.rays_d, self.depth = get_rays_single_image(self.H, self.W,
                                                                             self.intrinsics, self.c2w_mat)

                self.depth_sphere = intersect_sphere(self.rays_o, self.rays_d)

            self.pole_inds = None
            if self.seg_path is not None:
                seg_map = np.array(imageio.imread(self.seg_path)[..., 0]).reshape(-1)
                self.pole_inds = (seg_map == 5).nonzero()[0]

                print(self.pole_inds.shape[0])

                if self.pole_inds.shape[0] == 0:
                    self.pole_inds = None

            if self.depth_path is not None:

                if self.train_box_only:
                    self.depth_map = imageio.imread(self.depth_path)[..., 0]

                    near = 1.1
                    far = 2.7

                    # self.box_inds = None

                    # self.box_inds = np.nonzero(np.tile(self.depth_map[None, ...], (128, 1, 1)) == 255.)  # None
                    # self.box_inds = np.nonzero(self.depth_map.reshape(-1) != 255.)  # None
                    self.box_inds = np.nonzero(self.depth_map.reshape(-1) == 255.)  # None
                    self.depth_map =np.array(near + self.depth_map / 255. * (far - near))


                    self.depth_map_nonorm = self.depth_map.reshape(-1)
                    self.depth_map = self.depth_normalize(self.depth_map.reshape(-1))



                else:

                    # h*w*3
                    self.depth_map = imageio.imread(self.depth_path)[..., :3]
                    imgs = cv2.resize(self.depth_map, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
                    r = imgs[..., 0]
                    g = imgs[..., 1]
                    b = imgs[..., 2]
                    far = 1000.
                    tmp = r + g * 256. + b * 256. * 256.
                    tmp = tmp / (256. * 256. * 256. - 1.)
                    tmp = tmp * far

                    # if visualize_depth:
                    # img = Image.fromarray(np.uint8(tmp/1000. * 255), 'L')
                    # img.save('depth_sample.png')
                    # img.show()
                    # visualize_depth = False

                    # tmp = cv2.GaussianBlur(tmp, (,5), 0 )
                    depth_map = tmp.reshape((-1))

                    # to create depth segments for donerf need to normalze depthmap, so that when we so search sort everything is in normalied coordinates

                    self.depth_map_nonorm = None #depth_map
                    self.depth_map = self.depth_normalize(depth_map)
            else:
                self.depth_map_nonorm = None
                self.depth_map = None

    def depth_normalize(self, depth_map):

        # downscal = np.sqrt(3 * np.power(30., 2))

        depth_map_norm = np.array(depth_map) / 30.#3.  # 30. in normal scene

        # ray_d_cos = 1. / np.linalg.norm(self.rays_d, axis=-1, keepdims=False)
        #
        # re = depth_map_norm * ray_d_cos

        # return re
        return depth_map_norm

    def get_img(self):
        if self.img is not None:
            return self.img.reshape((self.H, self.W, 3))
        else:
            return None

    def get_depth(self):
        if self.depth_map_nonorm is not None:
            return self.depth_map_nonorm.reshape((self.H, self.W))
        else:
            return None

    def get_all(self):
        if self.min_depth is not None:
            min_depth = self.min_depth
        else:
            min_depth = 1e-4 * np.ones_like(self.rays_d[..., 0])

        if self.box_loc is None:
            box_loc = None
        else:
            box_loc = np.tile(self.box_loc, (self.rays_d.shape[0], 1))

        ret = OrderedDict([
            ('ray_o', self.rays_o),
            ('ray_d', self.rays_d),
            ('depth', self.depth),
            ('depth_gt', self.depth_map),
            ('rgb', self.img),
            ('mask', self.mask),
            ('min_depth', min_depth),
            ('box_loc', box_loc)
        ])
        # return torch tensors
        for k in ret:
            if ret[k] is not None:
                if isinstance(ret[k], np.ndarray):
                    ret[k] = torch.from_numpy(ret[k])
        return ret

    def get_all_classifier(self, N_front_sample, N_back_sample, pretrain, rank, train_box_only=False):

        if self.train_box_only:
            axis_filtered_depth_flat, fg_pts_flat, fg_z_vals_centre \
                = self.get_classifier_label_torch_boxonly(N_front_sample, pretrain, select_inds=None, rank=rank,
                                                          train_box_only=self.train_box_only)

            bg_pts_flat = None
            bg_z_vals_centre = None

        else:
            axis_filtered_depth_flat, fg_pts_flat, bg_pts_flat, bg_z_vals_centre, fg_z_vals_centre = \
                self.get_classifier_label_torch(N_front_sample, N_back_sample, pretrain=pretrain, rank=rank)

        if pretrain:
            cls_label_filtered = axis_filtered_depth_flat
        else:
            cls_label_filtered = None

        if self.min_depth is not None:
            min_depth = self.min_depth
        else:
            min_depth = 1e-4 * np.ones_like(self.rays_d[..., 0])

        if self.box_loc is None:
            box_loc = None
        else:
            box_loc = np.tile(self.box_loc, (self.rays_d.shape[0], 1))

        ret = OrderedDict([
            ('ray_o', self.rays_o),
            ('ray_d', self.rays_d),
            ('cls_label', cls_label_filtered),
            ('fg_pts_flat', fg_pts_flat),
            ('bg_pts_flat', bg_pts_flat),
            ('bg_z_vals_centre', bg_z_vals_centre),
            ('fg_z_vals_centre', fg_z_vals_centre),
            ('fg_far_depth', self.depth_sphere),

            ('depth_gt', self.depth_map_nonorm),
            ('rgb', self.img),
            ('mask', self.mask),
            ('min_depth', min_depth),
            ('img_name', self.img_path),
            ('box_loc', box_loc)

        ])
        # return torch tensors
        for k in ret:

            if ret[k] is not None and isinstance(ret[k], np.ndarray):
                ret[k] = torch.from_numpy(ret[k]).to(rank)

        return ret

    def get_classifier_label_torch_boxonly(self, N_front_sample, pretrain, select_inds=None, save=False, rank=0,
                                           train_box_only=False):
        import time
        time_program = False
        if time_program:
            cur_time = time.time()

        N_rays = self.H * self.W

        depth_sphere = torch.from_numpy(self.depth_sphere).to(rank)
        rays_o = torch.from_numpy(self.rays_o).to(rank)
        rays_d = torch.from_numpy(self.rays_d).to(rank)

        fg_far_depth = depth_sphere.to(rank)  # how far is the sphere to rayo [ H*W,]

        same_seg = True

        if same_seg:
            # fg_near_depth = fg_far_depth - 2.  # [H*W,]
            ray_d_cos = 1. / torch.norm(rays_d, dim=-1, keepdim=False)
            fg_near_depth = fg_far_depth - 2. * ray_d_cos
        else:

            fg_near_depth = torch.from_numpy(1e-4 * np.ones_like(self.rays_d[..., 0])).to(rank)

        step = (fg_far_depth - fg_near_depth) / (
                N_front_sample - 1)  # fg step size  --> will make this constant eventually [H*W]

        fg_z_vals = torch.stack([fg_near_depth + i * step for i in range(N_front_sample)],
                                dim=-1)  # [..., N_samples] distance to camera till unit sphere

        fg_z_vals_centre = step.unsqueeze(-1) / 2. + fg_z_vals
        fg_z_vals_centre = fg_z_vals_centre[:, :-1]
        fg_pts = rays_o.unsqueeze(-2) + fg_z_vals_centre.unsqueeze(-1) * rays_d.unsqueeze(-2)  # [H*W, N_samples, 3]

        if time_program:
            cur_time_sp = time.time() - cur_time
            print('[TIME] get segs {}'.format(cur_time_sp))
            cur_time = time.time()

        fg_pts_flat = fg_pts.view(N_rays, -1)

        if pretrain and save:

            # if True:

            depth_map = torch.from_numpy(self.depth_map).to(rank)

            # they are both distance to camera
            depth_segs = fg_z_vals

            seg_ind = torch.searchsorted(depth_segs, depth_map.unsqueeze(-1))


            if self.box_inds is not None:
                seg_ind[self.box_inds] = 127

                del self.box_inds




            if time_program:
                cur_time_sp = time.time() - cur_time
                print('[TIME] searchsort {}'.format(cur_time_sp))
                cur_time = time.time()

            x_range = torch.arange(self.W).to(rank)
            y_range = torch.arange(self.H).to(rank)

            y, x = torch.meshgrid(y_range, x_range)

            x = x.reshape(-1, 1)
            y = y.reshape(-1, 1)

            # if self.box_inds is not None:
            #     x = torch.take(x, torch.Tensor(self.box_inds[0]).long().to(rank)).reshape(-1, 1)
            #     y = torch.take(y, torch.Tensor(self.box_inds[0]).long().to(rank)).reshape(-1, 1)
            #     seg_ind = torch.take(seg_ind, torch.Tensor(self.box_inds[0]).long().to(rank)).reshape(-1, 1)
            #     # maxs[self.box_inds[1], self.box_inds[2], self.box_inds[0]] = 0.
            #
            #     del self.box_inds


            # ones = torch.cat([x.unsqueeze(-1),y.unsqueeze(-1),seg_ind.unsqueeze(-1)],dim=-1)

            ## you can technically loop through ones and compute that function
            K_spat = 5
            K_half_fl = int(np.floor(K_spat / 2))

            target_values = np.zeros((K_spat, K_spat))
            sort_ind = np.zeros((K_spat, K_spat))
            dis_li = {}
            ct = 0
            for i_ind, i in enumerate(range(K_half_fl * -1, K_half_fl + 1)):
                for j_ind, j in enumerate(range(K_half_fl * -1, K_half_fl + 1)):
                    target_values[i_ind, j_ind] = max(1 - math.sqrt(i ** 2 + j ** 2) / (math.sqrt(2) * K_half_fl), 0)
                    dis = math.sqrt(i ** 2 + j ** 2)
                    if dis not in dis_li.keys():
                        dis_li[dis] = ct
                        ct += 1
                    sort_ind[i_ind, j_ind] = int(dis_li[dis])
            target_values = torch.from_numpy(target_values).to(rank).float()
            sort_ind = torch.from_numpy(sort_ind).to(rank).long()

            # target_values = torch.from_numpy(np.array([[0., 0.20943058, 0.29289322, 0.20943058, 0.        ],\
            #                    [0.20943058, 0.5       , 0.64644661, 0.5       , 0.20943058],\
            #                    [0.29289322, 0.64644661, 1.        , 0.64644661, 0.29289322],\
            #                    [0.20943058, 0.5       , 0.64644661, 0.5       , 0.20943058],\
            #                    [0.        , 0.20943058, 0.29289322, 0.20943058, 0.        ]])).to(rank).float()
            #
            # sort_ind = torch.from_numpy(np.array([[5,4,3,4,5],\
            #                           [4,2,1,2,4],\
            #                           [3,1,0,1,3],\
            #                           [4,2,1,2,4],\
            #                           [5,4,3,4,5]])).to(rank)
            ind_list = []

            for i in range(K_half_fl * -1, K_half_fl + 1):
                tmp = []
                for j in range(K_half_fl * -1, K_half_fl + 1):
                    cat = torch.cat([x + j, y + i, seg_ind], dim=-1)
                    cat = cat[(cat[:, 0] >= 0) & (cat[:, 1] >= 0) & (cat[:, 0] < self.W) & (cat[:, 1] < self.H)]
                    tmp.append(cat)
                ind_list.append(tmp)

            # ind out of bound -1,-1 ?
            cls_label = torch.zeros((self.H, self.W, depth_segs.shape[1], torch.max(sort_ind) + 1)).to(rank)

            ct = 0
            for i in range(K_spat):
                for j in range(K_spat):
                    inds = ind_list[i][j]
                    cls_label[inds[:, 1], inds[:, 0], inds[:, 2], sort_ind[i, j]] = target_values[i, j]
                    ct += 1

            maxs, _ = torch.max(cls_label, dim=-1, keepdim=False)

            # if self.box_inds is not None:
            #     maxs[self.box_inds[1], self.box_inds[2], self.box_inds[0]] = 0.
            #
            #     del self.box_inds

            if time_program:
                cur_time_sp = time.time() - cur_time
                print('[TIME] spatial filter {}'.format(cur_time_sp))
                cur_time = time.time()

            # test = maxs.detach().cpu().numpy()
            # test1 = seg_ind.view(self.H, self.W).detach().cpu().numpy()
            # input – input tensor of shape minibatch,in_channels,iW)

            # weight – filters of shape (out_channels,groups/in_channels, KW)
            # tri_filter = torch.Tensor([[0.33333, 0.666666, 1, 0.666666, 0.33333]]).unsqueeze(0).to(rank)
            cls_label_flat = maxs.view(self.H * self.W, depth_segs.shape[1])  # [H* W, numseg]

            Z = 5  # triangle filter
            tri_filter = []

            Z_half_floor = np.floor(Z / 2)
            for i in range(int(-Z_half_floor), int(Z_half_floor) + 1):
                tri_filter.append((Z_half_floor + 1 - np.absolute(i)) / (Z_half_floor + 1))
            tri_filter = torch.Tensor(tri_filter).unsqueeze(0).unsqueeze(0).to(rank) ## need to be [1, 1, Z]
                # test15 = cls_label_flat.detach().cpu().numpy()



            cls_label_flat_filtered = torch.squeeze(
                torch.nn.functional.conv1d(cls_label_flat.unsqueeze(-2), tri_filter, padding=int(Z_half_floor)))


            cls_label_flat_filtered[cls_label_flat_filtered > 1.] = 1.
            cls_label_flat_filtered[cls_label_flat_filtered < 0.] = 0.

            # test15 = cls_label_flat_filtered.detach().cpu().numpy()

            if time_program:
                cur_time_sp = time.time() - cur_time
                print('[TIME] depth filter {}'.format(cur_time_sp))

            zarr.save(self.label_path + '.zarr', cls_label_flat_filtered.detach().cpu().numpy())

            return
            # cls_label_flat_filtered_ = cls_label_flat_filtered
            # if select_inds is not None:
            #     cls_label_flat_filtered_ = cls_label_flat_filtered[select_inds]
            # else:
            #     cls_label_flat_filtered_ = cls_label_flat_filtered


        elif pretrain is False:
            cls_label_flat_filtered_ = None

        else:

            # select_inds = np.random.choice(self.H * self.W, size=(1024,), replace=False)
            if time_program:
                cur_time = time.time()
            if select_inds is not None:
                cls_label_flat_filtered = zarr.load(self.label_path + '.zarr')
            else:
                cls_label_flat_filtered = zarr.load(self.label_path + '.zarr')

            # loader = np.load(self.label_path+'.npz', mmap_mode='r')
            if time_program:
                cur_time_sp = time.time() - cur_time
                print('[TIME] loading takes {}'.format(cur_time_sp))
                cur_time = time.time()
            # cls_label_flat_filtered = loader['arr_0'][select_inds]
            # cls_label_flat_filtered = loader
            if select_inds is not None:
                cls_label_flat_filtered_ = cls_label_flat_filtered[select_inds]
            else:
                cls_label_flat_filtered_ = cls_label_flat_filtered
            if time_program:
                cur_time_sp = time.time() - cur_time
                print('[TIME] get value takes {}'.format(cur_time_sp))
                cur_time = time.time()
            del cls_label_flat_filtered

        return cls_label_flat_filtered_, fg_pts_flat, fg_z_vals

    def get_classifier_label_torch(self, N_front_sample, N_back_sample, pretrain, select_inds=None, save=False, rank=0,
                                   train_box_only=False):
        import time
        time_program = False
        if time_program:
            cur_time = time.time()

        N_rays = self.H * self.W

        depth_sphere = torch.from_numpy(self.depth_sphere).to(rank)
        rays_o = torch.from_numpy(self.rays_o).to(rank)
        rays_d = torch.from_numpy(self.rays_d).to(rank)

        fg_far_depth = depth_sphere.to(rank)  # how far is the sphere to rayo [ H*W,]

        same_seg = True

        if same_seg:
            # fg_near_depth = fg_far_depth - 2.  # [H*W,]
            ray_d_cos = 1. / torch.norm(rays_d, dim=-1, keepdim=False)
            fg_near_depth = fg_far_depth - 2. * ray_d_cos
        else:

            fg_near_depth = torch.from_numpy(1e-4 * np.ones_like(self.rays_d[..., 0])).to(rank)

        step = (fg_far_depth - fg_near_depth) / (
                N_front_sample - 1)  # fg step size  --> will make this constant eventually [H*W]

        fg_z_vals = torch.stack([fg_near_depth + i * step for i in range(N_front_sample)],
                                dim=-1)  # [..., N_samples] distance to camera till unit sphere

        fg_z_vals_centre = step.unsqueeze(-1) / 2. + fg_z_vals
        fg_z_vals_centre = fg_z_vals_centre[:, :-1]
        fg_pts = rays_o.unsqueeze(-2) + fg_z_vals_centre.unsqueeze(-1) * rays_d.unsqueeze(-2)  # [H*W, N_samples, 3]

        bg_z_vals = torch.linspace(0., 1., N_back_sample).to(rank)
        step = bg_z_vals[1] - bg_z_vals[0]
        bg_z_vals_centre = bg_z_vals[:-1] + step / 2.

        bg_z_vals = bg_z_vals.view(
            [1, ] + [N_back_sample, ]).expand([N_rays] + [N_back_sample, ])  # [H*W, N_samples]
        bg_z_vals_centre = bg_z_vals_centre.view(
            [1, ] + [N_back_sample - 1, ]).expand([N_rays] + [N_back_sample - 1, ])  # [H*W, N_samples]

        bg_pts, _ = depth2pts_outside(
            rays_o.unsqueeze(-2).expand([N_rays] + [N_back_sample - 1, 3]),
            rays_d.unsqueeze(-2).expand([N_rays] + [N_back_sample - 1, 3]),
            bg_z_vals_centre)  # [H*W, N_samples, 4],  # [H*W, N_samples]

        if time_program:
            cur_time_sp = time.time() - cur_time
            print('[TIME] get segs {}'.format(cur_time_sp))
            cur_time = time.time()

        fg_pts_flat = fg_pts.view(N_rays, -1)
        bg_pts_flat = torch.flip(bg_pts, dims=[1, ]).view(N_rays, -1)

        if pretrain and save:

            # if True:

            depth_map = torch.from_numpy(self.depth_map).to(rank)

            _, bg_depth_real = depth2pts_outside(
                rays_o.unsqueeze(-2).expand([N_rays] + [N_back_sample, 3]),
                rays_d.unsqueeze(-2).expand([N_rays] + [N_back_sample, 3]),
                bg_z_vals)  # [H*W, N_samples, 4],  # [H*W, N_samples]

            # flip left and right
            # bg_pts, bg_depth_real = torch.flip(bg_pts, dims=[1,]), torch.fliplr(bg_depth_real)
            bg_depth_real = torch.fliplr(bg_depth_real)

            # bg_depth_real_np = bg_depth_real.cpu().numpy()
            # fg_z_vals_np = fg_z_vals.cpu().numpy()

            # if pretrain:
            bg_depth_real[:, 0] = fg_z_vals[:, -1]  # there can be potential mismatches

            # they are both distance to camera
            depth_segs = torch.cat([fg_z_vals, bg_depth_real], dim=-1)  # [H*W, numseg]

            seg_ind = torch.searchsorted(depth_segs, depth_map.unsqueeze(-1))

            if time_program:
                cur_time_sp = time.time() - cur_time
                print('[TIME] searchsort {}'.format(cur_time_sp))
                cur_time = time.time()

            x_range = torch.arange(self.W).to(rank)
            y_range = torch.arange(self.H).to(rank)

            y, x = torch.meshgrid(y_range, x_range)

            x = x.reshape(-1, 1)
            y = y.reshape(-1, 1)
            # ones = torch.cat([x.unsqueeze(-1),y.unsqueeze(-1),seg_ind.unsqueeze(-1)],dim=-1)

            ## you can technically loop through ones and compute that function
            K_spat = 5
            K_half_fl = int(np.floor(K_spat / 2))

            target_values = np.zeros((K_spat, K_spat))
            sort_ind = np.zeros((K_spat, K_spat))
            dis_li = {}
            ct = 0
            for i_ind, i in enumerate(range(K_half_fl * -1, K_half_fl + 1)):
                for j_ind, j in enumerate(range(K_half_fl * -1, K_half_fl + 1)):
                    target_values[i_ind, j_ind] = max(1 - math.sqrt(i ** 2 + j ** 2) / (math.sqrt(2) * K_half_fl), 0)
                    dis = math.sqrt(i ** 2 + j ** 2)
                    if dis not in dis_li.keys():
                        dis_li[dis] = ct
                        ct += 1
                    sort_ind[i_ind, j_ind] = int(dis_li[dis])
            target_values = torch.from_numpy(target_values).to(rank).float()
            sort_ind = torch.from_numpy(sort_ind).to(rank).long()

            # target_values = torch.from_numpy(np.array([[0., 0.20943058, 0.29289322, 0.20943058, 0.        ],\
            #                    [0.20943058, 0.5       , 0.64644661, 0.5       , 0.20943058],\
            #                    [0.29289322, 0.64644661, 1.        , 0.64644661, 0.29289322],\
            #                    [0.20943058, 0.5       , 0.64644661, 0.5       , 0.20943058],\
            #                    [0.        , 0.20943058, 0.29289322, 0.20943058, 0.        ]])).to(rank).float()
            #
            # sort_ind = torch.from_numpy(np.array([[5,4,3,4,5],\
            #                           [4,2,1,2,4],\
            #                           [3,1,0,1,3],\
            #                           [4,2,1,2,4],\
            #                           [5,4,3,4,5]])).to(rank)
            ind_list = []

            for i in range(K_half_fl * -1, K_half_fl + 1):
                tmp = []
                for j in range(K_half_fl * -1, K_half_fl + 1):
                    cat = torch.cat([x + j, y + i, seg_ind], dim=-1)
                    cat = cat[(cat[:, 0] >= 0) & (cat[:, 1] >= 0) & (cat[:, 0] < self.W) & (cat[:, 1] < self.H)]
                    tmp.append(cat)
                ind_list.append(tmp)

            # ind out of bound -1,-1 ?
            cls_label = torch.zeros((self.H, self.W, depth_segs.shape[1], torch.max(sort_ind) + 1)).to(rank)

            ct = 0
            for i in range(K_spat):
                for j in range(K_spat):
                    inds = ind_list[i][j]
                    cls_label[inds[:, 1], inds[:, 0], inds[:, 2], sort_ind[i, j]] = target_values[i, j]
                    ct += 1

            maxs, _ = torch.max(cls_label, dim=-1, keepdim=False)

            if time_program:
                cur_time_sp = time.time() - cur_time
                print('[TIME] spatial filter {}'.format(cur_time_sp))
                cur_time = time.time()

            # test = maxs.detach().cpu().numpy()
            # test1 = seg_ind.view(self.H, self.W).detach().cpu().numpy()
            # input – input tensor of shape minibatch,in_channels,iW)

            # weight – filters of shape (out_channels,groups/in_channels, KW)
            # tri_filter = torch.Tensor([[0.33333, 0.666666, 1, 0.666666, 0.33333]]).unsqueeze(0).to(rank)
            cls_label_flat = maxs.view(self.H * self.W, depth_segs.shape[1])  # [H* W, numseg]



            Z = 5  # triangle filter
            tri_filter = []

            Z_half_floor = np.floor(Z / 2)
            for i in range(int(-Z_half_floor), int(Z_half_floor) + 1):
                tri_filter.append((Z_half_floor + 1 - np.absolute(i)) / (Z_half_floor + 1))
            tri_filter = torch.Tensor(tri_filter).unsqueeze(0).unsqueeze(0).to(rank)  ## need to be [1, 1, Z]
            # test15 = cls_label_flat.detach().cpu().numpy()

            cls_label_flat_filtered = torch.squeeze(
                torch.nn.functional.conv1d(cls_label_flat.unsqueeze(-2), tri_filter, padding=int(Z_half_floor)))

            cls_label_flat_filtered[cls_label_flat_filtered > 1.] = 1.
            cls_label_flat_filtered[cls_label_flat_filtered < 0.] = 0.

            # test15 = cls_label_flat.detach().cpu().numpy()

            if time_program:
                cur_time_sp = time.time() - cur_time
                print('[TIME] depth filter {}'.format(cur_time_sp))

            zarr.save(self.label_path + '.zarr', cls_label_flat_filtered.detach().cpu().numpy())

            return
            # cls_label_flat_filtered_ = cls_label_flat_filtered
            # if select_inds is not None:
            #     cls_label_flat_filtered_ = cls_label_flat_filtered[select_inds]
            # else:
            #     cls_label_flat_filtered_ = cls_label_flat_filtered


        elif pretrain is False:
            cls_label_flat_filtered_ = None

        else:

            # select_inds = np.random.choice(self.H * self.W, size=(1024,), replace=False)
            if time_program:
                cur_time = time.time()
            if select_inds is not None:
                cls_label_flat_filtered = zarr.load(self.label_path + '.zarr')
            else:
                cls_label_flat_filtered = zarr.load(self.label_path + '.zarr')

            # loader = np.load(self.label_path+'.npz', mmap_mode='r')
            if time_program:
                cur_time_sp = time.time() - cur_time
                print('[TIME] loading takes {}'.format(cur_time_sp))
                cur_time = time.time()
            # cls_label_flat_filtered = loader['arr_0'][select_inds]
            # cls_label_flat_filtered = loader
            if select_inds is not None:
                cls_label_flat_filtered_ = cls_label_flat_filtered[select_inds]
            else:
                cls_label_flat_filtered_ = cls_label_flat_filtered
            if time_program:
                cur_time_sp = time.time() - cur_time
                print('[TIME] get value takes {}'.format(cur_time_sp))
                cur_time = time.time()
            del cls_label_flat_filtered

        return cls_label_flat_filtered_, fg_pts_flat, bg_pts_flat, bg_z_vals, fg_z_vals

    def random_sample_classifier(self, N_rand, N_front_sample, N_back_sample, pretrain, rank):

        if self.pole_inds is not None:
            all_ind = np.delete(np.arange(self.H * self.W), self.pole_inds)
            num_pix_pole = self.pole_inds.shape[0]

            if num_pix_pole > N_rand * 0.5:
                select_ind_pole = np.random.choice(self.pole_inds, size=(int(N_rand * 0.5),), replace=False)
                select_inds = np.random.choice(all_ind, size=(int(N_rand * 0.5),), replace=False)
                select_inds = np.concatenate([select_inds, select_ind_pole])

            else:

                num_sel = N_rand - self.pole_inds.shape[0]
                select_inds = np.concatenate(
                    [np.random.choice(all_ind, size=(num_sel,), replace=False), self.pole_inds])
        else:
            select_inds = np.random.choice(self.H * self.W, size=(N_rand,), replace=False)

        with torch.no_grad():

            if self.train_box_only:
                axis_filtered_depth_flat, fg_pts_flat, fg_z_vals_centre \
                    = self.get_classifier_label_torch_boxonly(N_front_sample, pretrain, select_inds=select_inds,
                                                              rank=rank, train_box_only=self.train_box_only)

            else:
                axis_filtered_depth_flat, fg_pts_flat, bg_pts_flat, bg_z_vals_centre, fg_z_vals_centre = \
                    self.get_classifier_label_torch(N_front_sample, N_back_sample, pretrain, select_inds=select_inds,
                                                    rank=rank)

        # if center_crop:
        #     half_H = self.H // 2
        #     half_W = self.W // 2
        #     quad_H = half_H // 2
        #     quad_W = half_W // 2
        #
        #     # pixel coordinates
        #     u, v = np.meshgrid(np.arange(half_W - quad_W, half_W + quad_W),
        #                        np.arange(half_H - quad_H, half_H + quad_H))
        #     u = u.reshape(-1)
        #     v = v.reshape(-1)
        #
        #     select_inds = np.random.choice(u.shape[0], size=(N_rand,), replace=False)
        #
        #     # Convert back to original image
        #     select_inds = v[select_inds] * self.W + u[select_inds]

        # select_inds = np.random.choice(self.box_inds[0].shape[0], size=(N_rand,), replace=False)
        # select_inds = self.box_inds[0][select_inds]

        rays_o = self.rays_o[select_inds, :]  # [N_rand, 3]
        rays_d = self.rays_d[select_inds, :]  # [N_rand, 3]

        fg_pts_flat = fg_pts_flat[select_inds]
        fg_z_vals_centre = fg_z_vals_centre[select_inds]

        if self.train_box_only:
            bg_pts_flat = None
            bg_z_vals_centre = None
        else:
            bg_pts_flat = bg_pts_flat[select_inds]
            bg_z_vals_centre = bg_z_vals_centre[select_inds]

        depth_sph = self.depth_sphere[select_inds]

        if pretrain:
            cls_label_filtered = axis_filtered_depth_flat
        else:
            cls_label_filtered = axis_filtered_depth_flat

        if self.box_loc is not None:
            box_loc = np.tile(self.box_loc, (self.rays_d.shape[0], 1))[select_inds, :]
        else:
            box_loc = None

        if self.img is not None:
            rgb = self.img[select_inds, :]  # [N_rand, 3]
        else:
            rgb = None

        if self.depth_map_nonorm is not None:
            depth_map = self.depth_map_nonorm[select_inds]  # [N_rand, 3]
        else:
            depth_map = None

        if self.mask is not None:
            mask = self.mask[select_inds]
        else:
            mask = None

        if self.min_depth is not None:
            min_depth = self.min_depth[select_inds]
        else:
            min_depth = 1e-4 * np.ones_like(rays_d[..., 0])

        ret = OrderedDict([
            ('ray_o', rays_o),
            ('ray_d', rays_d),
            ('cls_label', cls_label_filtered),
            ('fg_pts_flat', fg_pts_flat),
            ('bg_pts_flat', bg_pts_flat),
            ('bg_z_vals_centre', bg_z_vals_centre),
            ('fg_z_vals_centre', fg_z_vals_centre),
            ('fg_far_depth', depth_sph),

            ('depth_gt', depth_map),
            ('rgb', rgb),
            ('mask', mask),
            ('min_depth', min_depth),
            ('img_name', self.img_path),
            ('box_loc', box_loc)

        ])
        # return torch tensors
        for k in ret:

            if ret[k] is not None and isinstance(ret[k], np.ndarray):
                ret[k] = torch.from_numpy(ret[k])
        del axis_filtered_depth_flat

        return ret

    def random_sample(self, N_rand, center_crop=False):
        '''
        :param N_rand: number of rays to be casted
        :return:
        '''
        if center_crop:
            half_H = self.H // 2
            half_W = self.W // 2
            quad_H = half_H // 2
            quad_W = half_W // 2

            # pixel coordinates
            u, v = np.meshgrid(np.arange(half_W - quad_W, half_W + quad_W),
                               np.arange(half_H - quad_H, half_H + quad_H))
            u = u.reshape(-1)
            v = v.reshape(-1)

            select_inds = np.random.choice(u.shape[0], size=(N_rand,), replace=False)

            # Convert back to original image
            select_inds = v[select_inds] * self.W + u[select_inds]
        else:
            # Random from one image
            select_inds = np.random.choice(self.H * self.W, size=(N_rand,), replace=False)

        rays_o = self.rays_o[select_inds, :]  # [N_rand, 3]
        rays_d = self.rays_d[select_inds, :]  # [N_rand, 3]
        depth = self.depth[select_inds]  # [N_rand, ]
        if self.box_loc is not None:
            box_loc = np.tile(self.box_loc, (self.rays_d.shape[0], 1))[select_inds, :]
        else:
            box_loc = None

        if self.img is not None:
            rgb = self.img[select_inds, :]  # [N_rand, 3]
        else:
            rgb = None

        if self.depth_map is not None:
            depth_map = self.depth_map[select_inds]  # [N_rand, 3]
        else:
            depth_map = None

        if self.mask is not None:
            mask = self.mask[select_inds]
        else:
            mask = None

        if self.min_depth is not None:
            min_depth = self.min_depth[select_inds]
        else:
            min_depth = 1e-4 * np.ones_like(rays_d[..., 0])

        ret = OrderedDict([
            ('ray_o', rays_o),
            ('ray_d', rays_d),
            ('depth', depth),
            ('depth_gt', depth_map),
            ('rgb', rgb),
            ('mask', mask),
            ('min_depth', min_depth),
            ('img_name', self.img_path),
            ('box_loc', box_loc)
        ])
        # return torch tensors
        for k in ret:

            if ret[k] is not None and isinstance(ret[k], np.ndarray):
                ret[k] = torch.from_numpy(ret[k])

        return ret

import numpy as np
from collections import OrderedDict
import torch
import cv2
import imageio
from PIL import Image
import math
from ddp_model_mip import depth2pts_outside, depth2pts_outside_np
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


def intersect_sphere_t(ray_o, ray_d):
    '''
    ray_o, ray_d: [..., 3]
    compute the depth of the intersection point between this ray and unit sphere
    '''
    # note: d1 becomes negative if this mid point is behind camera
    d1 = -torch.sum(ray_d * ray_o, dim=-1) / torch.sum(ray_d * ray_d, dim=-1)
    p = ray_o + d1.unsqueeze(-1) * ray_d
    # consider the case where the ray does not intersect the sphere
    ray_d_cos = 1. / torch.norm(ray_d, dim=-1)
    d2 = torch.sqrt(1. - torch.sum(p * p, dim=-1)) * ray_d_cos

    return d1 + d2

def get_rays_mipnerf(H, W, intrinsics, c2w):
    """Generating rays for all images."""
    x, y = np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
        np.arange(W, dtype=np.float32),  # X-Axis (columns)
        np.arange(H, dtype=np.float32),  # Y-Axis (rows)
        indexing='xy')
    
    F = intrinsics[0,0]
    
    camera_dirs = np.stack(
        [(x - W * 0.5 + 0.5) / F,
         -(y - H * 0.5 + 0.5) / F,
         -np.ones_like(x)], axis=-1)
    
    print(camera_dirs.shape, c2w.shape)
    directions = ((camera_dirs[..., None, :] * c2w[None, None, :3, :3]).sum(axis=-1)) # [B, W,H, 3]
    print('directions',directions.shape)
     
    origins = np.broadcast_to(c2w[None, None, :3, -1], directions.shape) # [B, W,H, 3]
  
    # Distance from each unit-norm direction vector to its x-axis neighbor.
    dx = np.sqrt(
        np.sum((directions[:, :-1, :] - directions[:, 1:, :])**2, -1, keepdims=True))
    dx = np.concatenate([dx, dx[:, -2:-1, :]], 1) # concat with the last dx (this is repeated)
    # Cut the distance in half, and then round it out so that it's
    # halfway between inscribed by / circumscribed about the pixel. 
    print('dx',dx.shape)
    radii = dx * 2. / np.sqrt(12.) #[W,H, 1]
    print('radii',radii.shape)
    
    return origins.astype(np.float32), directions.astype(np.float32), radii.astype(np.float32)



def get_rays_single_image_t(H, W, intrinsics, c2w):
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
    pixels = torch.from_numpy(pixels).type_as(c2w)

    rays_d = torch.matmul(torch.inverse(intrinsics[:3, :3]), pixels) # sensor's location to camera
    #norm_0 = np.linalg.norm(rays_d.transpose((1, 0)), axis=-1)

    rays_d = torch.matmul(c2w[:3, :3], rays_d)  # (3, H*W)
    rays_d = rays_d.transpose(1, 0)  # (H*W, 3)

    #norm_1 = np.linalg.norm(rays_d, axis=-1)

    rays_o = c2w[:3, 3].reshape(1, 3)
    rays_o = rays_o.expand(rays_d.shape[0], -1)  # (H*W, 3)

    ray_d_new = rays_d - rays_o
    #norm_2 = np.linalg.norm(ray_d_new, axis=-1)

    depth = torch.inverse(c2w)[2, 3]
    depth = depth * torch.ones(rays_o.shape[0]).type_as(c2w)  # (H*W,)

    return rays_o, rays_d, depth

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

def get_rays_scan(H, W, c2w):
    phi = np.linspace(-np.pi, np.pi, W).astype(dtype=np.float32)
    theta = np.linspace(-np.pi/4, np.pi/4, H,).astype(dtype=np.float32)

    pv, tv = np.meshgrid(phi, theta)
    x = np.sin(pv) * np.cos(tv)
    y = np.sin(tv)
    z = np.cos(pv) * np.cos(tv)

    rays_d = torch.from_numpy(np.stack([x, y, z]).reshape(3, -1)).to(c2w.device)
    rays_d = c2w[:3, :3].matmul(rays_d).T

    rays_o = c2w[:3, 3].reshape((1, 3))
    rays_o = rays_o.expand(rays_d.shape[0], -1)

    depth = torch.inverse(c2w)[2, 3]
    depth = depth * torch.ones((rays_o.shape[0],), dtype=torch.float32).to(c2w.device)  # (H*W,)
    
#     print('rays_o',rays_o.type())
#     print('rays_d',rays_d.type())
#     print('depth',depth.type())

    return rays_o, rays_d, depth


class RaySamplerSingleImageMip(object):
    def __init__(self, H, W, intrinsics=None, c2w=None,
                 img_path=None,
                 box_loc=None,
                 depth_path=None,
                 rays=None,
                 seg_path=None,
                 make_class_label=None,
                 lidar_scan=False,
                 train_seg=False
                 ):
        super().__init__()

        self.W = W
        self.H = H
        self.intrinsics_orig = intrinsics
        self.c2w_mat = c2w
        
        self.box_seg = None
        self.img_path = img_path
        self.depth_path = depth_path
    
       
        self.seg_path = seg_path
        self.lidar_scan = lidar_scan

        self.box_loc = box_loc
        self.train_seg = train_seg
        self.get_all_rays(rays=rays)

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
            self.get_classifier_label_torch(N_front_sample=128, 
                                            N_back_sample=128, pretrain=True, save=True)

    def get_all_rays(self, rays=None) -> object:
        self.intrinsics = self.intrinsics_orig
        self.img = None
        if self.img_path is not None:
            self.img = imageio.imread(self.img_path).astype(np.float32) / 255.
            self.img = cv2.resize(self.img, (self.W, self.H), interpolation=cv2.INTER_AREA)
            self.img = self.img.reshape((-1, 3))
            

        self.rays_o, self.rays_d, self.radii = get_rays_mipnerf(self.H, self.W, self.intrinsics, self.c2w_mat)
        self.rays_o, self.rays_d, self.radii = self.rays_o.reshape([-1,3]), \
                                self.rays_d.reshape([-1,3]), self.radii.reshape([-1,1])
        self.depth_sphere = intersect_sphere(self.rays_o, self.rays_d)

        self.pole_inds = None
        if self.seg_path is not None:
            seg_map = np.array(imageio.imread(self.seg_path)[..., 0]).reshape(-1)

            if self.box_loc is not None:
                self.box_seg = np.zeros(seg_map.shape)
                self.box_seg[seg_map==3] = 1

            if self.train_seg:
                self.pole_inds = (seg_map == 3).nonzero()[0] #5 is pole 3 is box
                print('seg inds length {}'.format(self.pole_inds.shape[0]))
                if self.pole_inds.shape[0] == 0:
                    self.pole_inds = None

        self.depth_map_nonorm = None
        self.depth_map = None
        if self.depth_path is not None:
            # h*w*3
            self.depth_map = imageio.imread(self.depth_path)[..., :3]
            imgs = cv2.resize(self.depth_map, (self.W, self.H),
                              interpolation=cv2.INTER_LINEAR)
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

            depth_map = tmp.reshape((-1))
            
            

            self.depth_map_nonorm = depth_map
            self.depth_map = None#self.depth_normalize(depth_map)
            
        print('depth',self.depth_map_nonorm)

              

    def depth_normalize(self, depth_map):
        return np.array(depth_map) / 30.#3.  # 30. in normal scene


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

    def get_box_mask(self):
        if self.box_seg is not None:
            return self.box_seg.reshape((self.H, self.W))
        else:
            return None

    def get_all_classifier(self, N_front_sample, N_back_sample, pretrain, rank, train_box_only=False, box_number=10):

        axis_filtered_depth_flat, fg_pts_flat, \
        bg_pts_flat, bg_z_vals, fg_z_vals = \
                self.get_classifier_label_torch(N_front_sample, 
                                                N_back_sample, 
                                                rank=rank)

        cls_label_filtered = None

        if self.box_loc is None:
            box_loc = None
        else:
            box_loc = self.box_loc.unsqueeze(0).to(rank).expand(
                self.rays_d.shape[0], box_number, 3)


        ret = OrderedDict([
            ('ray_o', self.rays_o),
            ('ray_d', self.rays_d),
             ('radii', self.radii),
            ('cls_label', cls_label_filtered),
            ('fg_pts_flat', fg_pts_flat),
            ('bg_pts_flat', bg_pts_flat),
            ('bg_z_vals_fence', bg_z_vals),
            ('fg_z_vals_fence', fg_z_vals),
            ('fg_far_depth', self.depth_sphere),
            ('depth_gt', self.depth_map_nonorm),
            ('rgb', self.img),
            ('box_loc', box_loc)


        ])
        for k in ret:

            if ret[k] is not None and isinstance(ret[k], np.ndarray):
                ret[k] = torch.from_numpy(ret[k]).to(rank)

        return ret

    def random_sample_classifier(self, N_rand, N_front_sample, N_back_sample, rank, box_number=10):

        if self.pole_inds is not None:
            all_ind = np.delete(np.arange(self.H * self.W), self.pole_inds)
            num_pix_pole = self.pole_inds.shape[0]

            if num_pix_pole > N_rand * 0.5:
                select_ind_pole = np.random.choice(self.pole_inds, 
                                                size=(int(N_rand * 0.5),), replace=False)
                select_inds = np.random.choice(all_ind, 
                                              size=(int(N_rand * 0.5),), replace=False)
                select_inds = np.concatenate([select_inds, select_ind_pole])
            else:
                num_sel = N_rand - self.pole_inds.shape[0]
                select_inds = np.concatenate(
                    [np.random.choice(all_ind, size=(num_sel,), replace=False), 
                     self.pole_inds])
        else:
            select_inds = np.random.choice(self.H * self.W, size=(N_rand,), replace=False)

        with torch.no_grad():
            axis_filtered_depth_flat, fg_pts_flat, bg_pts_flat, \
            bg_z_vals, fg_z_vals = \
                self.get_classifier_label_torch(N_front_sample, N_back_sample, 
                                                select_inds=select_inds,
                                                rank=rank)
        # select_inds = np.random.choice(self.box_inds[0].shape[0], size=(N_rand,), replace=False)
        # select_inds = self.box_inds[0][select_inds]

        rays_o = self.rays_o[select_inds, :]  # [N_rand, 3]
        rays_d = self.rays_d[select_inds, :]  # [N_rand, 3]
        radii = self.radii[select_inds, :] 
        
        if self.box_seg is not None:
            seg_map_box = self.box_seg[select_inds]
        else:
            seg_map_box = None

        fg_pts_flat = fg_pts_flat[select_inds]
        fg_z_vals = fg_z_vals[select_inds]

        bg_pts_flat = bg_pts_flat[select_inds]
        bg_z_vals = bg_z_vals[select_inds]

        depth_sph = self.depth_sphere[select_inds]

        cls_label_filtered = axis_filtered_depth_flat

        if self.box_loc is not None:
            box_loc = np.tile(self.box_loc, (self.rays_d.shape[0], 1,1))[select_inds, :]
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

        ret = OrderedDict([
            ('ray_o', rays_o),
            ('ray_d', rays_d),
            ('radii', radii),
            ('cls_label', cls_label_filtered),
            ('fg_pts_flat', fg_pts_flat),
            ('bg_pts_flat', bg_pts_flat),
            ('bg_z_vals_fence', bg_z_vals),
            ('fg_z_vals_fence', fg_z_vals),
            ('fg_far_depth', depth_sph),
            ('depth_gt', depth_map),
            ('rgb', rgb),
            ('seg_map_box', seg_map_box),
            ('box_loc', box_loc)

        ])
        # return torch tensors
        for k in ret:

            if ret[k] is not None and isinstance(ret[k], np.ndarray):
                ret[k] = torch.from_numpy(ret[k])
        del axis_filtered_depth_flat

        return ret
    
    def get_classifier_label_torch(self, N_front_sample, 
                                      N_back_sample, select_inds=None, 
                                      save=False, rank=0,
                                   train_box_only=False):

        N_rays = self.H * self.W

        rays_o = torch.tensor(self.rays_o).cuda().to(rank)
        rays_d = torch.tensor(self.rays_d).cuda().to(rank)

        fg_far_depth = torch.tensor(self.depth_sphere).cuda().to(rank)  # how far is the sphere to rayo [ H*W,]

        ray_d_cos = 1. / torch.norm(rays_d, dim=-1, keepdim=False)
        fg_near_depth = fg_far_depth - 2. * ray_d_cos

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

        fg_pts_flat = fg_pts.view(N_rays, -1)
        bg_pts_flat = torch.flip(bg_pts, dims=[1, ]).view(N_rays, -1)
 
        cls_label_flat_filtered_ = None
        return cls_label_flat_filtered_, fg_pts_flat, bg_pts_flat, bg_z_vals, fg_z_vals


 
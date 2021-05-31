import numpy as np
from collections import OrderedDict
import torch
import cv2
import imageio
from PIL import Image
from ddp_model import depth2pts_outside, depth2pts_outside_np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
import pickle

visualize_depth = True
########################################################################################################################
# ray batch sampling
########################################################################################################################

def intersect_sphere(ray_o, ray_d):
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

def get_rays_single_image(H, W, intrinsics, c2w):
    '''
    :param H: image height
    :param W: image width
    :param intrinsics: 4 by 4 intrinsic matrix
    :param c2w: 4 by 4 camera to world extrinsic matrix
    :return:
    '''
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    u = u.reshape(-1).astype(dtype=np.float32) + 0.5    # add half pixel
    v = v.reshape(-1).astype(dtype=np.float32) + 0.5
    pixels = np.stack((u, v, np.ones_like(u)), axis=0)  # (3, H*W)
    pixels = torch.from_numpy(pixels).to(c2w.device)

    rays_d = torch.matmul(torch.inverse(intrinsics[:3, :3]), pixels)
    rays_d = torch.matmul(c2w[:3, :3], rays_d)  # (3, H*W)
    rays_d = rays_d.transpose(1, 0)  # (H*W, 3)

    rays_o = c2w[:3, 3].view(1, 3)
    rays_o = rays_o.expand(rays_d.shape[0], -1)  # (H*W, 3)

    depth = torch.inverse(c2w)[2, 3]
    depth = depth * torch.ones(rays_o.shape[0], device=c2w.device)  # (H*W,)

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
    depth = depth * torch.ones((rays_o.shape[0],)).to(c2w.device)  # (H*W,)

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
            rays = None,
            make_class_label=None,
            lidar_scan=False
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

        self.resolution_level = -1
        self.lidar_scan = lidar_scan
        self.set_resolution_level(resolution_level, rays=rays)


        self.box_loc = box_loc


        if make_class_label:
            self.get_classifier_label_torch(N_front_sample=128, N_back_sample=128, save=True)

        #print("H:{}, W:{}".format(self.H,self.W))

    # # this happens 1) when we query a ray, 2) when we preprcess and generate rays (we'll see)
    # def image_filter(depth_map):
    #     ## nvm the sampling of the rays have to be here then
    #
    #     ## in terms of warping
    #     ## 1. gt depth -- > which segment? (you need to know where the far-front depth is )
    #
    #     if depth < far-front depth:
    #         # use code beflow
    #     else:
    #         # first somehow assign to segments
    #
    #     ## 2. smoothing..?
    #
    #
    #     from scipy.ndimage import gaussian_filter, gaussian_filter1d
    #     # stacked_depth = np.tile(depth_map, (4, 1, 1))
    #     depths = [0, 10, 50, 200, 500, 1001]
    #     stacked = []
    #     for i in range(len(depths) - 1):
    #         temp = np.where((depth_map < depths[i]) | (depth_map >= depths[i + 1]), 0.0, depth_map)
    #         temp = np.where(temp != 0, 1.0, temp)
    #         temp = gaussian_filter(temp, sigma=0.5)
    #         # stacked_depth[i, ...][stacked_depth[i, ...] >= depths[i]] = 0
    #         # stacked_depth[i, ...][stacked_depth[i, ...] < depths[i+1]] = 0
    #         stacked.append(temp)
    #     stacked_depth = np.stack(stacked, axis=0)
    #     stacked_depth = np.transpose(stacked_depth, (1, 2, 0))
    #
    #     axis_filtered_depth = gaussian_filter1d(stacked_depth, 0.5)
    #     print("----------------------------------------------")
    #     print(axis_filtered_depth)
    #
    #     return 0
    #

    def set_resolution_level(self, resolution_level, rays=None):

        if resolution_level != self.resolution_level:
            self.resolution_level = resolution_level
            self.W = self.W_orig // resolution_level
            self.H = self.H_orig // resolution_level

            if rays is None:
                self.intrinsics = self.intrinsics_orig.clone()
                self.intrinsics[:2, :3] /= resolution_level
            # only load image at this time
            if self.img_path is not None:
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
                self.rays_o, self.rays_d, self.depth = (rays[0].cpu().detach().numpy(),
                                                       rays[1].cpu().detach().numpy(),
                                                       rays[2].cpu().detach().numpy())
            elif self.lidar_scan:

                self.rays_o, self.rays_d, self.depth = get_rays_scan(self.H, self.W,
                        self.c2w_mat)

                self.depth_sphere = intersect_sphere(self.rays_o, self.rays_d)
            else:
                self.rays_o, self.rays_d, self.depth = get_rays_single_image(self.H, self.W,
                        self.intrinsics, self.c2w_mat)

                self.depth_sphere = intersect_sphere(self.rays_o, self.rays_d)




            if self.depth_path is not None:
                # h*w*3
                self.depth_map = imageio.imread(self.depth_path)[...,:3]
                imgs = cv2.resize(self.depth_map, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
                r = imgs[..., 0]
                g = imgs[..., 1]
                b = imgs[..., 2]
                far = 1000.
                tmp = r + g * 256. + b * 256. * 256.
                tmp = tmp/(256.*256.*256.-1.)
                tmp = tmp * far

                # if visualize_depth:
                # img = Image.fromarray(np.uint8(tmp/1000. * 255), 'L')
                # img.save('depth_sample.png')
                # img.show()
                    # visualize_depth = False


                depth_map = tmp.reshape((-1))



                self.depth_map = self.depth_normalize(depth_map)

                # self.seg_ind = self.get_classifier_label_tmp(
                #     N_front_sample=128, N_back_sample=128)

            else:
                self.depth_map = None



    def depth_normalize(self,depth_map):
        max = np.array([100., 140.])
        min = np.array([85., 125.])
        avg_pose = np.array([0.5, 0.5])

        ro_denorm = ((self.rays_o[:, :2]) / 0.5 + avg_pose) * (max - min) + min # ray o to real
        depth_real = ro_denorm + depth_map[:, None] * self.rays_d[:, :2] # get obj depth in real
        depth_real_norm = ((depth_real - min) / (max - min) - avg_pose) * 0.5 # get obj depth in norm
        depth_map_norm = np.linalg.norm(depth_real_norm[:, :2] - self.rays_o[:, :2], axis=-1, keepdims=False) # get depth map in norm


        return depth_map_norm





    def get_img(self):
        if self.img is not None:
            return self.img.reshape((self.H, self.W, 3))
        else:
            return None

    def get_depth(self):
        if self.depth_map is not None:
            return self.depth_map.reshape((self.H, self.W))
        else:
            return None

    def get_all(self):
        if self.min_depth is not None:
            min_depth = self.min_depth
        else:
            min_depth = 1e-4 * torch.ones_like(self.rays_d[..., 0])

        if self.box_loc is None:
            box_loc = None
        else:
            box_loc = self.box_loc.expand(self.rays_d.shape[0], -1)

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

    def get_all_classifier(self, N_front_sample, N_back_sample, pretrain, center_crop=False, ):

        axis_filtered_depth_flat, fg_pts_flat, bg_pts_flat, bg_z_vals_centre, fg_z_vals_centre = \
                self.get_classifier_label_torch(N_front_sample, N_back_sample, pretrain=pretrain)

        if pretrain:
            cls_label_filtered = axis_filtered_depth_flat
        else:
            cls_label_filtered = None

        if self.min_depth is not None:
            min_depth = self.min_depth
        else:
            min_depth = 1e-4 * torch.ones_like(self.rays_d[..., 0])

        if self.box_loc is None:
            box_loc = None
        else:
            box_loc = self.box_loc.expand(self.rays_d.shape[0], -1)

        ret = OrderedDict([
            ('ray_o', self.rays_o),
            ('ray_d', self.rays_d),
            ('cls_label', cls_label_filtered),
            ('fg_pts_flat', fg_pts_flat),
            ('bg_pts_flat', bg_pts_flat),
            ('bg_z_vals_centre', bg_z_vals_centre),
            ('fg_z_vals_centre', fg_z_vals_centre),
            ('fg_far_depth', self.depth_sphere),

            ('depth_gt', self.depth_map),
            ('rgb', self.img),
            ('mask', self.mask),
            ('min_depth', min_depth),
            ('img_name', self.img_path),
            ('box_loc', box_loc)

            ])
        # return torch tensors
        for k in ret:

            if ret[k] is not None and isinstance(ret[k], np.ndarray):
                ret[k] = torch.from_numpy(ret[k])

        return ret

    def get_classifier_label(self, N_front_sample, N_back_sample):
        N_rays = self.H * self.W

        if self.min_depth is None:
            min_depth = 1e-4 * np.ones_like(self.rays_d[..., 0])


        fg_far_depth = self.depth_sphere  # how far is the sphere to rayo [ H*W,]

        same_seg = True

        if same_seg:
            fg_near_depth = self.depth_sphere - 2.  # [H*W,]
        else:
            fg_near_depth = min_depth  # [H*W,]
        step = (fg_far_depth - fg_near_depth) / (
                N_front_sample - 1)  # fg step size  --> will make this constant eventually [H*W]

        fg_z_vals = np.stack([fg_near_depth + i * step for i in range(N_front_sample)],
                axis=-1)  # [..., N_samples] distance to camera till unit sphere , depth value

        fg_z_vals_centre = step[..., None] / 2. + fg_z_vals

        # slice till last sample as wel are taking the mid points
        fg_pts = self.rays_o[..., None, :] + fg_z_vals_centre[:, :-1][..., None] * self.rays_d[..., None,
                :]  # [H*W, N_samples, 3]

        bg_z_val = np.linspace(0., 1., N_back_sample)
        step = bg_z_val[1] - bg_z_val[0]

        bg_z_vals_centre = bg_z_val[:-1] + step/2.

        bg_z_vals = np.tile(bg_z_val, (N_rays, 1))  # [H*W, N_samples]
        bg_z_vals_centre = np.tile(bg_z_vals_centre, (N_rays, 1))  # [H*W, N_samples]

        _, bg_depth_real = depth2pts_outside_np(self.rays_o, self.rays_d,
                bg_z_vals)  # [H*W, N_samples, 4],  # [H*W, N_samples]
        bg_pts, _ = depth2pts_outside_np(self.rays_o, self.rays_d,
                bg_z_vals_centre)  # [H*W, N_samples, 4],  # [H*W, N_samples]

        # flip left and right
        bg_pts, bg_depth_real = np.fliplr(bg_pts), np.fliplr(bg_depth_real)

        bg_depth_real[:, 0] = fg_z_vals[:, -1]  # there can be potential mismatches

        # they are both distance to camera
        depth_segs = np.concatenate([fg_z_vals, bg_depth_real], axis=-1)  # [H*W, numseg]
        points = np.concatenate([np.reshape(fg_pts, (N_rays, -1)), np.reshape(bg_pts, (N_rays, -1))], axis=-1)

        seg_ind = []  # index of bin the depth belongs to, [H*W]
        for i in range(self.depth_map.shape[0]):
            seg_ind.append(np.digitize(self.depth_map[i], depth_segs[i]))

        seg_ind = np.array(seg_ind)

        cls_label = np.zeros((self.H * self.W, depth_segs.shape[1]))

        np.put_along_axis(cls_label, seg_ind[:, None], 1., axis=-1)  # [H*W, numseg]

        cls_label = np.reshape(cls_label, (self.H, self.W, depth_segs.shape[1]))  # [H, W, numseg]

        # kernel = np.arra([0.3, 0.65, 1.0, 0.65, 0.3])
        spat_blur = cv2.GaussianBlur(cls_label, (5, 5), 1.2)
        axis_filtered_depth = gaussian_filter1d(spat_blur, 0.5)  # [H, W, numseg]
        axis_filtered_depth_flat = np.reshape(axis_filtered_depth, (self.H * self.W, depth_segs.shape[1]))

        return axis_filtered_depth_flat, points, bg_z_vals_centre, fg_z_vals_centre

    def get_classifier_label_torch(self, N_front_sample, N_back_sample, pretrain):

        # if not save:
        #     file = open(self.label_path, 'rb')
        #     [cls_label_flat, points, bg_z_vals_centre, fg_z_vals_centre] = pickle.load(file)
        #
        #
        #     return cls_label_flat, points, bg_z_vals_centre, fg_z_vals_centre
        #

        N_rays = self.H * self.W

        if self.min_depth is None:
            min_depth = 1e-4 * torch.ones_like(self.rays_d[..., 0])


        device = 'cuda:0'

        depth_sphere = self.depth_sphere.to(device)
        rays_o = self.rays_o.to(device)
        rays_d = self.rays_d.to(device)
        depth_map = torch.tensor(self.depth_map).to(device) if self.depth_map else None


        fg_far_depth = depth_sphere.to(device)  # how far is the sphere to rayo [ H*W,]

        same_seg = True

        if same_seg:
            fg_near_depth = fg_far_depth - 2.  # [H*W,]
        else:
            fg_near_depth = min_depth.to(device)  # [H*W,]



        step = (fg_far_depth - fg_near_depth) / (
                N_front_sample - 1)  # fg step size  --> will make this constant eventually [H*W]

        fg_z_vals = torch.stack([fg_near_depth + i * step for i in range(N_front_sample)],
                dim=-1)  # [..., N_samples] distance to camera till unit sphere

        fg_z_vals_centre = step.unsqueeze(-1) / 2. + fg_z_vals
        fg_z_vals_centre = fg_z_vals_centre[:, :-1]

        # slice till last sample as wel are taking the mid points
        fg_pts = rays_o.unsqueeze(-2) + fg_z_vals_centre.unsqueeze(-1) * rays_d.unsqueeze(-2)  # [H*W, N_samples, 3]

        bg_z_vals = torch.linspace(0., 1., N_back_sample).to(device)
        step = bg_z_vals[1] - bg_z_vals[0]

        bg_z_vals_centre = bg_z_vals[:-1] + step / 2.

        bg_z_vals = bg_z_vals.view(
                [1, ]  + [N_back_sample, ]).expand([N_rays] + [N_back_sample, ])  # [H*W, N_samples]
        bg_z_vals_centre = bg_z_vals_centre.view(
                [1, ]  + [N_back_sample - 1, ]).expand([N_rays] + [N_back_sample - 1, ])  # [H*W, N_samples]




        _, bg_depth_real = depth2pts_outside(
                rays_o.unsqueeze(-2).expand([N_rays] + [N_back_sample, 3]), rays_d.unsqueeze(-2).expand([N_rays] + [N_back_sample, 3]),bg_z_vals)  # [H*W, N_samples, 4],  # [H*W, N_samples]
        bg_pts, _ = depth2pts_outside(
                rays_o.unsqueeze(-2).expand([N_rays] + [N_back_sample-1, 3]),
                rays_d.unsqueeze(-2).expand([N_rays] + [N_back_sample-1, 3]),
                bg_z_vals_centre)  # [H*W, N_samples, 4],  # [H*W, N_samples]

        # flip left and right
        bg_pts, bg_depth_real = torch.flip(bg_pts, dims=[1,]), torch.fliplr(bg_depth_real)



        fg_pts_flat = fg_pts.view(N_rays, -1)
        bg_pts_flat = bg_pts.view(N_rays, -1)

        if pretrain:

            bg_depth_real[:, 0] = fg_z_vals[:, -1]  # there can be potential mismatches

            # they are both distance to camera
            depth_segs = torch.cat([fg_z_vals, bg_depth_real], dim=-1)  # [H*W, numseg]

            seg_ind = torch.searchsorted(depth_segs,depth_map.unsqueeze(-1))

            x_range = torch.arange(self.W).to(device)
            y_range = torch.arange(self.H).to(device)

            y,x = torch.meshgrid(y_range,x_range)

            x = x.reshape(-1,1)
            y = y.reshape(-1,1)
            # ones = torch.cat([x.unsqueeze(-1),y.unsqueeze(-1),seg_ind.unsqueeze(-1)],dim=-1)

            ## you can technically loop through ones and compute that function

            target_values = np.array([[0., 0.20943058, 0.29289322, 0.20943058, 0.        ],\
                    [0.20943058, 0.5       , 0.64644661, 0.5       , 0.20943058],\
                    [0.29289322, 0.64644661, 1.        , 0.64644661, 0.29289322],\
                    [0.20943058, 0.5       , 0.64644661, 0.5       , 0.20943058],\
                    [0.        , 0.20943058, 0.29289322, 0.20943058, 0.        ]])

            sort_ind = np.array([[5,4,3,4,5],\
                    [4,2,1,2,4],\
                    [3,1,0,1,3],\
                    [4,2,1,2,4],\
                    [5,4,3,4,5]])
            ind_list = []

            for i in range(-2,3):
                tmp=[]
                for j in range(-2,3):
                    cat = torch.cat([x+j,y+i,seg_ind],dim=-1)
                    cat = cat[(cat[:,0]>=0) & (cat[:,1]>=0) & (cat[:,0]<self.W) & (cat[:,1]<self.H)]
                    tmp.append(cat)
                ind_list.append(tmp)

        # ind out of bound -1,-1 ?
            cls_label = torch.zeros((self.H , self.W, depth_segs.shape[1],6)).to(device)


            ct = 0
            for i in range(5):
                for j in range(5):


                    inds = ind_list[i][j]
                    cls_label[inds[:,1],inds[:,0],inds[:,2], sort_ind[i,j]] = target_values[i,j]
                    ct +=1

            maxs, _ = torch.max(cls_label, dim=-1, keepdim=False)


            # test = maxs.detach().cpu().numpy()
            # test1 = seg_ind.view(self.H, self.W).detach().cpu().numpy()
            # input – input tensor of shape minibatch,in_channels,iW)



            # weight – filters of shape (out_channels,groups/in_channels, KW)
            tri_filter = torch.Tensor([[0.33333, 0.666666, 1, 0.666666, 0.33333]]).unsqueeze(0).to(device)
            cls_label_flat = maxs.view(self.H * self.W, depth_segs.shape[1])  # [H* W, numseg]

            # test15 = cls_label_flat.detach().cpu().numpy()


            cls_label_flat_filtered = torch.squeeze(torch.nn.functional.conv1d(cls_label_flat.unsqueeze(-2), tri_filter, padding=2))

            cls_label_flat_filtered[cls_label_flat_filtered > 1.] = 1.
            cls_label_flat_filtered[cls_label_flat_filtered < 0.] = 0.


            # test2 = cls_label_flat_filtered.detach().cpu().numpy()

            # file = open(self.label_path, 'wb+')
            # pickle.dump([cls_label_flat_filtered.detach().cpu().numpy(), points.detach().cpu().numpy(), \
                    #              bg_z_vals_centre.detach().cpu().numpy(), fg_z_vals_centre.detach().cpu().numpy()], file)
        else:
            cls_label_flat_filtered = None

        return cls_label_flat_filtered, fg_pts_flat,bg_pts_flat, bg_z_vals_centre, fg_z_vals_centre





    def random_sample_classifier(self,N_rand, N_front_sample, N_back_sample,pretrain,center_crop=False):
        ## can precompute before where each ray intersect sphere


        axis_filtered_depth_flat, fg_pts_flat, bg_pts_flat, bg_z_vals_centre, fg_z_vals_centre = \
                self.get_classifier_label_torch( N_front_sample, N_back_sample, pretrain)

        select_inds = np.random.choice(self.H * self.W, size=(N_rand,), replace=False)

        rays_o = self.rays_o[select_inds, :]  # [N_rand, 3]
        rays_d = self.rays_d[select_inds, :]  # [N_rand, 3]



        fg_pts_flat = fg_pts_flat[select_inds]
        bg_pts_flat = bg_pts_flat[select_inds]

        bg_z_vals_centre = bg_z_vals_centre[select_inds]
        fg_z_vals_centre = fg_z_vals_centre[select_inds]

        depth_sph = self.depth_sphere[select_inds]

        if pretrain:
            cls_label_filtered = axis_filtered_depth_flat[select_inds]
        else:
            cls_label_filtered = None


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
            u, v = np.meshgrid(np.arange(half_W-quad_W, half_W+quad_W),
                    np.arange(half_H-quad_H, half_H+quad_H))
            u = u.reshape(-1)
            v = v.reshape(-1)

            select_inds = np.random.choice(u.shape[0], size=(N_rand,), replace=False)

            # Convert back to original image
            select_inds = v[select_inds] * self.W + u[select_inds]
        else:
            # Random from one image
            select_inds = np.random.choice(self.H*self.W, size=(N_rand,), replace=False)

        rays_o = self.rays_o[select_inds, :]    # [N_rand, 3]
        rays_d = self.rays_d[select_inds, :]    # [N_rand, 3]
        depth = self.depth[select_inds]         # [N_rand, ]
        if self.box_loc is not None:
            box_loc = np.tile(self.box_loc, (self.rays_d.shape[0],1))[select_inds, :]
        else:
            box_loc = None

        if self.img is not None:
            rgb = self.img[select_inds, :]          # [N_rand, 3]
        else:
            rgb = None

        if self.depth_map is not None:
            depth_map = self.depth_map[select_inds]          # [N_rand, 3]
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

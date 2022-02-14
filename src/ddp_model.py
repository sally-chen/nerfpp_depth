import torch
import torch.nn as nn
import numpy as np
# import torch.nn.functional as F
# import numpy as np
from .utils import TINY_NUMBER, HUGE_NUMBER
from collections import OrderedDict
from .nerf_network import Embedder, MLPNet, MLPNetClassier
import os
import logging
from pytorch3d.transforms import euler_angles_to_matrix
from  .helpers import check_shadow, check_shadow_aabb_inters

logger = logging.getLogger(__package__)


######################################################################################
# wrapper to simplify the use of nerfnet
######################################################################################
def depth2pts_outside(ray_o, ray_d, depth):
    '''
    ray_o, ray_d: [..., 3]
    depth: [...]; inverse of distance to sphere origin
    '''
    # note: d1 becomes negative if this mid point is behind camera
    d1 = -torch.sum(ray_d * ray_o, dim=-1) / torch.sum(ray_d * ray_d, dim=-1)  # tb
    p_mid = ray_o + d1.unsqueeze(-1) * ray_d  # b
    p_mid_norm = torch.norm(p_mid, dim=-1)
    ray_d_cos = 1. / (torch.norm(ray_d, dim=-1) + 0.0001)
    d2 = torch.sqrt(1. - p_mid_norm * p_mid_norm) * ray_d_cos
    p_sphere = ray_o + (d1 + d2).unsqueeze(-1) * ray_d

    rot_axis = torch.cross(ray_o, p_sphere, dim=-1)
    rot_axis = rot_axis / (torch.norm(rot_axis, dim=-1, keepdim=True) + 0.0001)
    phi = torch.asin(p_mid_norm)
    theta = torch.asin(p_mid_norm * depth)  # depth is inside [0, 1]
    rot_angle = (phi - theta).unsqueeze(-1)  # [..., 1]

    # now rotate p_sphere
    # Rodrigues formula: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    p_sphere_new = p_sphere * torch.cos(rot_angle) + \
                   torch.cross(rot_axis, p_sphere, dim=-1) * torch.sin(rot_angle) + \
                   rot_axis * torch.sum(rot_axis * p_sphere, dim=-1, keepdim=True) * (1. - torch.cos(rot_angle))
    p_sphere_new = p_sphere_new / torch.norm(p_sphere_new, dim=-1, keepdim=True)
    pts = torch.cat((p_sphere_new, depth.unsqueeze(-1)), dim=-1)

    # now calculate conventional depth
    # torch.cos(theta) = pmid --> edge of sphere
    # depth_real = 1. / (depth + TINY_NUMBER) * torch.cos(theta) * ray_d_cos + d1
    depth_real = 1. / (depth) * torch.cos(theta) * ray_d_cos + d1
    return pts, depth_real


def depth2pts_outside_np(ray_o, ray_d, depth):
    '''
    ray_o, ray_d: [..., 3]
    depth: [...]; inverse of distance to sphere origin
    '''
    # note: d1 becomes negative if this mid point is behind camera
    d1 = -np.sum(ray_d * ray_o, axis=-1) / np.sum(ray_d * ray_d, axis=-1)  # tb
    p_mid = ray_o + d1[..., None] * ray_d  # b
    p_mid_norm = np.linalg.norm(p_mid, axis=-1)
    ray_d_cos = 1. / (np.linalg.norm(ray_d, axis=-1) + 0.0001)
    d2 = np.sqrt(1. - p_mid_norm * p_mid_norm) * ray_d_cos
    p_sphere = ray_o + (d1 + d2)[..., None] * ray_d

    rot_axis = np.cross(ray_o, p_sphere, axis=-1)
    rot_axis = rot_axis / (np.linalg.norm(rot_axis, axis=-1, keepdims=True) + 0.0001)
    phi = np.arcsin(p_mid_norm)
    theta = np.arcsin(p_mid_norm[..., None] * depth)  # depth is inside [0, 1]
    rot_angle = (phi[..., None] - theta)[..., None]  # [..., 1]

    # now rotate p_sphere
    # Rodrigues formula: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula

    # print("p_sphere:{} rot_angle {} rot_axis {}".format(p_sphere.shape,rot_angle.shape,rot_axis.shape))

    term1 = p_sphere[..., None, :] * np.cos(rot_angle)
    term2 = np.cross(rot_axis, p_sphere, axis=-1)[..., None, :] * np.sin(rot_angle)
    term3 = (rot_axis * np.sum(rot_axis * p_sphere, axis=-1, keepdims=True))[..., None, :] * (1. - np.cos(rot_angle))
    p_sphere_new = term1 + term2 + term3
    p_sphere_new = p_sphere_new / np.linalg.norm(p_sphere_new, axis=-1, keepdims=True)
    pts = np.concatenate((p_sphere_new, depth[..., None]), axis=-1)
    # now calculate conventional depth
    depth_real = 1. / (depth + TINY_NUMBER) * np.cos(theta) * ray_d_cos[..., None] + d1[..., None]
    return pts, depth_real




class DepthOracle(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.pencode = args.pencode
        self.penc_pts = args.penc_pts

        self.use_zval = args.use_zval

        if self.use_zval:
            pos_ch_fg = args.front_sample - 1
            pos_ch_bg = args.back_sample - 1
        else:
            pos_ch_fg = (args.front_sample - 1) * 3
            pos_ch_bg = (args.back_sample - 1) * 4

        if self.penc_pts:
            self.embedder_pts = Embedder(input_dim=pos_ch_fg,
                                         max_freq_log2=args.max_freq_log2_pts - 1,
                                         N_freqs=args.max_freq_log2_pts)

            self.embedder_pts_bg = Embedder(input_dim=pos_ch_bg,
                                            max_freq_log2=args.max_freq_log2_pts - 1,
                                            N_freqs=args.max_freq_log2_pts)
            pos_ch_fg = self.embedder_pts.out_dim
            pos_ch_bg = self.embedder_pts_bg.out_dim

        if self.pencode:
            self.embedder_position = Embedder(input_dim=3,
                                              max_freq_log2=args.max_freq_log2 - 1,
                                              N_freqs=args.max_freq_log2)
            self.embedder_viewdir = Embedder(input_dim=3,
                                             max_freq_log2=args.max_freq_log2_viewdirs - 1,
                                             N_freqs=args.max_freq_log2_viewdirs)

            self.ora_net_fg = MLPNetClassier(input_ch=self.embedder_position.out_dim,
                                             input_ch_viewdirs=self.embedder_viewdir.out_dim, D=args.netdepth,
                                             W=args.netwidth,
                                             pos_ch=pos_ch_fg,
                                             out_dim=args.front_sample)

            self.embedder_position_bg = Embedder(input_dim=3,
                                                 max_freq_log2=args.max_freq_log2 - 1,
                                                 N_freqs=args.max_freq_log2)

            self.ora_net_bg = MLPNetClassier(input_ch=self.embedder_position_bg.out_dim,
                                             input_ch_viewdirs=self.embedder_viewdir.out_dim, D=args.netdepth,
                                             W=args.netwidth,
                                             pos_ch=pos_ch_bg,
                                             out_dim=args.back_sample)
        else:

            self.ora_net_fg = MLPNetClassier(D=args.netdepth, W=args.netwidth,
                                             pos_ch=pos_ch_fg,
                                             out_dim=args.front_sample)

            self.ora_net_bg = MLPNetClassier(D=args.netdepth, W=args.netwidth,
                                             pos_ch=pos_ch_bg,
                                             out_dim=args.back_sample)

    def forward(self, ray_o, ray_d, points_fg, points_bg, fg_far_depth):
        '''
       :param ray_o, ray_d: [N_rays, 3]
       :param cls_label: [N_rays, N_samples]
       :param points: [N_rays, 3*N_front_sample +4 * N_back_samples]
       :return
       '''

        # print(ray_o.shape, ray_d.shape, fg_z_max.shape, fg_z_vals.shape, bg_z_vals.shape)
        ray_d_norm = torch.norm(ray_d, dim=-1, keepdim=True)  # [..., 1]
        viewdirs = ray_d / (ray_d_norm + TINY_NUMBER)  # [..., 3]

        ray_o_bg = ray_o + ray_d * fg_far_depth.unsqueeze(-1)

        if self.pencode:
            viewdirs = self.embedder_viewdir(viewdirs)
            ray_o = self.embedder_position(ray_o)
            ray_o_bg = self.embedder_position_bg(ray_o_bg)

        if self.penc_pts:
            points_fg = self.embedder_pts(points_fg)
            points_bg = self.embedder_pts_bg(points_bg)

        depth_likeli_fg = self.ora_net_fg(ray_o, viewdirs, points_fg)
        depth_likeli_bg = self.ora_net_bg(ray_o_bg, viewdirs, points_bg)

        ret = OrderedDict([('likeli_fg', depth_likeli_fg), ('likeli_bg', depth_likeli_bg)])

        return ret


class NerfNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        # foreground
        self.fg_embedder_position = Embedder(input_dim=3,
                                             max_freq_log2=args.max_freq_log2 - 1,
                                             N_freqs=args.max_freq_log2)
        self.fg_embedder_viewdir = Embedder(input_dim=3,
                                            max_freq_log2=args.max_freq_log2_viewdirs - 1,
                                            N_freqs=args.max_freq_log2_viewdirs)
        self.fg_net = MLPNet(D=args.netdepth, W=args.netwidth,
                             input_ch=self.fg_embedder_position.out_dim,
                             input_ch_viewdirs=self.fg_embedder_viewdir.out_dim,
                             use_viewdirs=args.use_viewdirs)
        # background; bg_pt is (x, y, z, 1/r)
        self.bg_embedder_position = Embedder(input_dim=4,
                                             max_freq_log2=args.max_freq_log2 - 1,
                                             N_freqs=args.max_freq_log2)
        self.bg_embedder_viewdir = Embedder(input_dim=3,
                                            max_freq_log2=args.max_freq_log2_viewdirs - 1,
                                            N_freqs=args.max_freq_log2_viewdirs)
        self.bg_net = MLPNet(D=args.netdepth, W=args.netwidth,
                             input_ch=self.bg_embedder_position.out_dim,
                             input_ch_viewdirs=self.bg_embedder_viewdir.out_dim,
                             use_viewdirs=args.use_viewdirs)

    def forward(self, ray_o, ray_d, fg_z_max, fg_z_vals, bg_z_vals):
        '''
        :param ray_o, ray_d: [..., 3]
        :param fg_z_max: [...,]
        :param fg_z_vals, bg_z_vals: [..., N_samples]
        :return
        '''

        # print(ray_o.shape, ray_d.shape, fg_z_max.shape, fg_z_vals.shape, bg_z_vals.shape)
        ray_d_norm = torch.norm(ray_d, dim=-1, keepdim=True)  # [..., 1]
        viewdirs = ray_d / ray_d_norm  # [..., 3]
        dots_sh = list(ray_d.shape[:-1])

        ######### render foreground
        N_samples = fg_z_vals.shape[-1]
        fg_ray_o = ray_o.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        fg_ray_d = ray_d.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        fg_viewdirs = viewdirs.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        fg_pts = fg_ray_o + fg_z_vals.unsqueeze(-1) * fg_ray_d
        input = torch.cat((self.fg_embedder_position(fg_pts),
                           self.fg_embedder_viewdir(fg_viewdirs)), dim=-1)
        fg_raw = self.fg_net(input)
        # alpha blending
        fg_dists = fg_z_vals[..., 1:] - fg_z_vals[..., :-1]
        # account for view directions
        fg_dists = ray_d_norm * torch.cat((fg_dists, fg_z_max.unsqueeze(-1) - fg_z_vals[..., -1:]),
                                          dim=-1)  # [..., N_samples]
        fg_alpha = 1. - torch.exp(-fg_raw['sigma'] * fg_dists)  # [..., N_samples]
        T = torch.cumprod(1. - fg_alpha + TINY_NUMBER, dim=-1)  # [..., N_samples]
        bg_lambda = T[..., -1]
        T = torch.cat((torch.ones_like(T[..., 0:1]), T[..., :-1]), dim=-1)  # [..., N_samples]
        fg_weights = fg_alpha * T  # [..., N_samples]
        fg_rgb_map = torch.sum(fg_weights.unsqueeze(-1) * fg_raw['rgb'], dim=-2)  # [..., 3]
        fg_depth_map = torch.sum(fg_weights * fg_z_vals, dim=-1)  # [...,]

        # render background
        N_samples = bg_z_vals.shape[-1]
        bg_ray_o = ray_o.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        bg_ray_d = ray_d.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        bg_viewdirs = viewdirs.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        bg_pts, _ = depth2pts_outside(bg_ray_o, bg_ray_d, bg_z_vals)  # [..., N_samples, 4]
        input = torch.cat((self.bg_embedder_position(bg_pts),
                           self.bg_embedder_viewdir(bg_viewdirs)), dim=-1)
        # near_depth: physical far; far_depth: physical near
        input = torch.flip(input, dims=[-2, ])
        bg_z_vals = torch.flip(bg_z_vals, dims=[-1, ])  # 1--->0
        bg_dists = bg_z_vals[..., :-1] - bg_z_vals[..., 1:]
        bg_dists = torch.cat((bg_dists, HUGE_NUMBER * torch.ones_like(bg_dists[..., 0:1])), dim=-1)  # [..., N_samples]
        bg_raw = self.bg_net(input)
        bg_alpha = 1. - torch.exp(-bg_raw['sigma'] * bg_dists)  # [..., N_samples]
        # Eq. (3): T
        # maths show weights, and summation of weights along a ray, are always inside [0, 1]
        T = torch.cumprod(1. - bg_alpha + TINY_NUMBER, dim=-1)[..., :-1]  # [..., N_samples-1]
        T = torch.cat((torch.ones_like(T[..., 0:1]), T), dim=-1)  # [..., N_samples]
        bg_weights = bg_alpha * T  # [..., N_samples]
        bg_rgb_map = torch.sum(bg_weights.unsqueeze(-1) * bg_raw['rgb'], dim=-2)  # [..., 3]
        bg_depth_map = torch.sum(bg_weights * bg_z_vals, dim=-1)  # [...,]

        # composite foreground and background
        bg_rgb_map = bg_lambda.unsqueeze(-1) * bg_rgb_map
        # bg_depth_map = bg_lambda * bg_depth_map
        rgb_map = fg_rgb_map + bg_rgb_map

        _, bg_depth_map = depth2pts_outside(ray_o, ray_d, bg_depth_map)
        depth_map = fg_depth_map + bg_lambda * bg_depth_map

        # device_num = torch.cuda.current_device()
        # max = torch.tensor([100., 140.]).to(device_num)
        # min = torch.tensor([85., 125.]).to(device_num)
        # avg_pose = torch.tensor([0.5,  0.5]).to(device_num)
        #
        # obj_pt_norm = ray_o + depth_map.unsqueeze(-1) * ray_d
        #
        # obj_pt_denorm = obj_pt_norm.clone()
        # obj_pt_denorm[:,:2] = (obj_pt_norm[:,:2]/ 0.5 + avg_pose) * 15.
        # ro_denorm = ray_o.clone()
        # ro_denorm[:,:2]   = ((ray_o[:,:2]) / 0.5 + avg_pose) * 15.
        # depth_map = torch.norm(obj_pt_denorm - ro_denorm, dim=1, keepdim=False)


        ret = OrderedDict([('rgb', rgb_map),  # loss
                           ('fg_weights', fg_weights),  # importance sampling
                           ('bg_weights', bg_weights),  # importance sampling
                           ('fg_rgb', fg_rgb_map),  # below are for logging
                           ('fg_depth', fg_depth_map * 30.),
                           ('bg_rgb', bg_rgb_map),
                           ('bg_depth', bg_depth_map * 30.),
                           ('bg_lambda', bg_lambda),
                           ('depth_fgbg', depth_map * 30. )])
        return ret





class NerfNetBox(nn.Module):
    def __init__(self, args):
        super().__init__()
        # foreground
        self.fg_embedder_position = Embedder(input_dim=3,
                                             max_freq_log2=args.max_freq_log2 - 1,
                                             N_freqs=args.max_freq_log2)
        self.fg_embedder_viewdir = Embedder(input_dim=3,
                                            max_freq_log2=args.max_freq_log2_viewdirs - 1,
                                            N_freqs=args.max_freq_log2_viewdirs)
        self.fg_net = MLPNet(D=args.netdepth, W=args.netwidth,
                             input_ch=self.fg_embedder_position.out_dim,
                             input_ch_viewdirs=self.fg_embedder_viewdir.out_dim,
                             use_viewdirs=args.use_viewdirs)
        # background; bg_pt is (x, y, z, 1/r)
        self.bg_embedder_position = Embedder(input_dim=4,
                                             max_freq_log2=args.max_freq_log2 - 1,
                                             N_freqs=args.max_freq_log2)
        self.bg_embedder_viewdir = Embedder(input_dim=3,
                                            max_freq_log2=args.max_freq_log2_viewdirs - 1,
                                            N_freqs=args.max_freq_log2_viewdirs)
        self.bg_net = MLPNet(D=args.netdepth, W=args.netwidth,
                             input_ch=self.bg_embedder_position.out_dim,
                             input_ch_viewdirs=self.bg_embedder_viewdir.out_dim,
                             use_viewdirs=args.use_viewdirs)
        # this is the net for the box

        # input to this should be box position and sampled position
        self.box_net = MLPNet(D=args.netdepth, W=args.netwidth,
                              input_ch=self.fg_embedder_position.out_dim,
                              input_ch_viewdirs=self.fg_embedder_viewdir.out_dim,
                              use_viewdirs=args.use_viewdirs)

    def forward(self, ray_o, ray_d, fg_z_max, fg_z_vals, bg_z_vals, box_loc, query_box_only=False):
        '''orch.c
        :param ray_o, ray_d: [..., 3]
        :param fg_z_max: [...,]
        :param fg_z_vals, bg_z_vals: [..., N_samples]
        :param box_locs: [..., 3]  (N. [x,y, z])
        :return
        '''

        # print(ray_o.shape, ray_d.shape, fg_z_max.shape, fg_z_vals.shape, bg_z_vals.shape)
        ray_d_norm = torch.norm(ray_d, dim=-1, keepdim=True)  # [..., 1]
        viewdirs = ray_d / ray_d_norm  # [..., 3]
        dots_sh = list(ray_d.shape[:-1])

        ######### render foreground
        N_samples = fg_z_vals.shape[-1]
        fg_ray_o = ray_o.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        fg_ray_d = ray_d.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        fg_viewdirs = viewdirs.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        fg_pts = fg_ray_o + fg_z_vals.unsqueeze(-1) * fg_ray_d

        ######
        # get output from boxnet
        # convert to the box coordinate
        # box_offset = (fg_pts - box_loc.unsqueeze(
        #     -2)) * 0.5

        box_offset = (fg_pts - box_loc.unsqueeze(-2))
        input_box = torch.cat((self.fg_embedder_position(box_offset),
                               self.fg_embedder_viewdir(fg_viewdirs)), dim=-1)
        fg_box_raw = self.box_net(input_box.float())

        ######

        if query_box_only:
            ret = OrderedDict([('fg_box_sig', fg_box_raw['sigma'])])
            return ret

        input = torch.cat((self.fg_embedder_position(fg_pts),
                           self.fg_embedder_viewdir(fg_viewdirs)), dim=-1)
        fg_raw = self.fg_net(input)

        # alpha blending
        fg_dists = fg_z_vals[..., 1:] - fg_z_vals[..., :-1]
        # account for view directions
        fg_dists = ray_d_norm * torch.cat((fg_dists, fg_z_max.unsqueeze(-1) - fg_z_vals[..., -1:]),
                                          dim=-1)  # [..., N_samples]
        fg_alpha = 1. - torch.exp(-(fg_raw['sigma'] + fg_box_raw['sigma']) * fg_dists)  # [..., N_samples]
        T = torch.cumprod(1. - fg_alpha + TINY_NUMBER, dim=-1)  # [..., N_samples]
        bg_lambda = T[..., -1]
        T = torch.cat((torch.ones_like(T[..., 0:1]), T[..., :-1]), dim=-1)  # [..., N_samples]
        fg_weights = fg_alpha * T  # [..., N_samples]

        fg_rgb = torch.div(
            fg_raw['sigma'].unsqueeze(-1) * fg_raw['rgb'] + fg_box_raw['sigma'].unsqueeze(-1) * fg_box_raw['rgb'],
            fg_raw['sigma'].unsqueeze(-1) + fg_box_raw['sigma'].unsqueeze(-1))

        fg_rgb_map = torch.sum(fg_weights.unsqueeze(-1) * fg_rgb, dim=-2)  # [..., 3]
        fg_depth_map = torch.sum(fg_weights * fg_z_vals,
                                 dim=-1)  # [...,]     ############################bg_depth_map################################

        # render background
        N_samples = bg_z_vals.shape[-1]
        bg_ray_o = ray_o.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        bg_ray_d = ray_d.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        bg_viewdirs = viewdirs.unsqueeze(-2).expand(dots_sh + [N_samples, 3])

        bg_pts, bg_depth_real = depth2pts_outside(bg_ray_o, bg_ray_d, bg_z_vals)  # [..., N_samples, 4]
        input = torch.cat((self.bg_embedder_position(bg_pts),
                           self.bg_embedder_viewdir(bg_viewdirs)), dim=-1)
        # near_depth: physical far; far_depth: physical near
        input = torch.flip(input, dims=[-2, ])
        bg_z_vals = torch.flip(bg_z_vals, dims=[-1, ])  # 1--->0
        bg_dists = bg_z_vals[..., :-1] - bg_z_vals[..., 1:]
        bg_dists = torch.cat((bg_dists, HUGE_NUMBER * torch.ones_like(bg_dists[..., 0:1])), dim=-1)  # [..., N_samples]
        bg_raw = self.bg_net(input)

        ######
        # get output from boxnet
        ######

        bg_alpha = 1. - torch.exp(-bg_raw['sigma'] * bg_dists)  # [..., N_samples]
        # Eq. (3): T
        # maths show weights, and summation of weights along a ray, are always inside [0, 1]
        T = torch.cumprod(1. - bg_alpha + TINY_NUMBER, dim=-1)[..., :-1]  # [..., N_samples-1]
        T = torch.cat((torch.ones_like(T[..., 0:1]), T), dim=-1)  # [..., N_samples]
        bg_weights = bg_alpha * T  # [..., N_samples]
        bg_rgb_map = torch.sum(bg_weights.unsqueeze(-1) * bg_raw['rgb'], dim=-2)  # [..., 3]
        bg_depth_map = torch.sum(bg_weights * bg_z_vals,
                                 dim=-1)  # [...,]  ############################bg_depth_map################################
        # bg_depth_map = torch.sum(bg_weights * bg_depth_real, dim=-1)

        # composite foreground and background
        bg_rgb_map = bg_lambda.unsqueeze(-1) * bg_rgb_map

        _, bg_depth_map = depth2pts_outside(ray_o, ray_d, bg_depth_map)
        bg_depth_map = bg_lambda * bg_depth_map

        depth_map = fg_depth_map + bg_depth_map

        rgb_map = fg_rgb_map + bg_rgb_map
        ## combine foregroung and background in the right depth unit s well

        ## need inverse normalization
        # device_num = torch.cuda.current_device()
        #
        # max = torch.tensor([100., 140.]).to(device_num)
        # min = torch.tensor([85., 125.]).to(device_num)
        # avg_pose = torch.tensor([0.5, 0.5]).to(device_num)
        #
        # depth_pt_denorm = ((ray_o[:, :2] + depth_map.unsqueeze(-1) * viewdirs[:, :2]) / 0.5 + avg_pose) * ( max - min) + min
        # ro_denorm = ((ray_o[:, :2]) / 0.5 + avg_pose) * (max - min) + min
        # depth_map = torch.norm(depth_pt_denorm[:, :2] - ro_denorm, dim=1, keepdim=False)
        #
        # ##
        # ret = [rgb_map,fg_weights,bg_weights,fg_rgb_map,fg_depth_map,bg_rgb_map,bg_depth_map,bg_lambda,depth_map]
        ## somehow denoise pls
        ret = OrderedDict([('rgb', rgb_map),  # loss
                           ('fg_weights', fg_weights),  # importance sampling
                           ('bg_weights', bg_weights),  # importance sampling
                           ('fg_rgb', fg_rgb_map),  # below are for logging
                           ('fg_depth', fg_depth_map*30.),
                           ('bg_rgb', bg_rgb_map),
                           ('bg_depth', bg_depth_map*30.),
                           ('bg_lambda', bg_lambda),
                           ('depth_fgbg', depth_map*30.)])
        return ret


class NerfNetMoreBoxIndep(nn.Module):
    def __init__(self, args):
        super().__init__()
        # foreground
        self.fg_embedder_position = Embedder(input_dim=3,
                                             max_freq_log2=args.max_freq_log2 - 1,
                                             N_freqs=args.max_freq_log2)
        self.fg_embedder_viewdir = Embedder(input_dim=3,
                                            max_freq_log2=args.max_freq_log2_viewdirs - 1,
                                            N_freqs=args.max_freq_log2_viewdirs)

        self.fg_embedder_position_box = Embedder(input_dim=3,
                                             max_freq_log2=args.max_freq_log2 - 1,
                                             N_freqs=args.max_freq_log2)

        self.fg_embedder_viewdir_box = Embedder(input_dim=3,
                                            max_freq_log2=args.max_freq_log2_viewdirs - 1,
                                            N_freqs=args.max_freq_log2_viewdirs)


        self.fg_net = MLPNet(D=args.netdepth, W=args.netwidth,
                             input_ch=self.fg_embedder_position.out_dim,
                             input_ch_viewdirs=self.fg_embedder_viewdir.out_dim,
                             use_viewdirs=args.use_viewdirs)
        # background; bg_pt is (x, y, z, 1/r)
        self.bg_embedder_position = Embedder(input_dim=4,
                                             max_freq_log2=args.max_freq_log2 - 1,
                                             N_freqs=args.max_freq_log2)
        self.bg_embedder_viewdir = Embedder(input_dim=3,
                                            max_freq_log2=args.max_freq_log2_viewdirs - 1,
                                            N_freqs=args.max_freq_log2_viewdirs)
        self.bg_net = MLPNet(D=args.netdepth, W=args.netwidth,
                             input_ch=self.bg_embedder_position.out_dim,
                             input_ch_viewdirs=self.bg_embedder_viewdir.out_dim,
                             use_viewdirs=args.use_viewdirs)
        # this is the net for the box
        self.box_number = args.box_number
        
        self.box_nets = torch.nn.ModuleList( [ MLPNet(D=4, W=args.netwidth,
                              input_ch=self.fg_embedder_position_box.out_dim,
                              input_ch_viewdirs=self.fg_embedder_viewdir_box.out_dim,
                              use_viewdirs=args.use_viewdirs) for _  in range(self.box_number)])

        # self.box_size = [float(size) for size in args.box_size.split(',')]
        #
        # # device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else 'cpu')
        # self.box_size = torch.tensor(self.box_size).cuda().to(torch.cuda.current_device())

    def forward(self, ray_o, ray_d, fg_z_max, fg_z_vals, bg_z_vals, box_loc, box_props, query_box_only=False):
        '''orch.c
        :param ray_o, ray_d: [..., 3]
        :param fg_z_max: [...,]
        :param fg_z_vals, bg_z_vals: [..., N_samples]
        :param box_locs: [..., 3]  (N. [x,y, z])
        :return
        '''
        print('indep')
        # print(ray_o.shape, ray_d.shape, fg_z_max.shape, fg_z_vals.shape, bg_z_vals.shape)
        ray_d_norm = torch.norm(ray_d, dim=-1, keepdim=True)  # [..., 1]
        viewdirs = ray_d / (ray_d_norm  + TINY_NUMBER) # [..., 3]
        dots_sh = list(ray_d.shape[:-1])

        ######### render foreground
        N_samples = fg_z_vals.shape[-1]
        fg_ray_o = ray_o.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        fg_ray_d = ray_d.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        fg_viewdirs = viewdirs.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        fg_pts = fg_ray_o + fg_z_vals.unsqueeze(-1) * fg_ray_d

        box_loc = box_loc.view(-1, self.box_number, 3)
        colors, box_sizes, box_rot =  box_props[:, 0:3], box_props[:, 3:6], box_props[:, 6:]

        assert box_sizes.shape == (self.box_number, 3), 'box_sizes shape is wrong'
        assert box_rot.shape == (self.box_number, 3), 'box_rot shape is wrong'

        fg_box_raw_lst = []
        self.box_nets = self.box_nets.to(torch.cuda.current_device())

        r = euler_angles_to_matrix(torch.deg2rad(torch.cat([box_rot[:,2:],-1*box_rot[:,1:2], -1*box_rot[:,0:1] ], 
                                                           dim=-1)), convention='ZYX')
        r_mat_expand = r.unsqueeze(0).unsqueeze(1).expand(dots_sh + [N_samples, self.box_number, 3, 3])

        box_offset = ((torch.matmul(torch.inverse(r_mat_expand) ,
                        (fg_pts.unsqueeze(-2).expand(dots_sh + [N_samples, self.box_number, 3])
                        - box_loc.unsqueeze(1).expand(dots_sh + [N_samples, self.box_number, 3])).unsqueeze(-1)).squeeze(-1))/
                      (box_sizes*30.+TINY_NUMBER).unsqueeze(0).unsqueeze(0))\
            .permute(0,2,1,3).reshape(dots_sh[0],self.box_number, N_samples, 3)
        
        expanded_viewdir = fg_viewdirs.unsqueeze(-2).expand(dots_sh + [N_samples, self.box_number, 3]).permute(0,2,1,3)
        expanded_viewdir_reshape = expanded_viewdir.reshape(dots_sh[0],self.box_number, N_samples, 3)


#         assert input_box.shape == (dots_sh[0]*self.box_number, N_samples, self.fg_embedder_position_box.out_dim + self.fg_embedder_viewdir_box.out_dim)
        
        box_sigmas = []
        box_rgbs = []

        for j,box_net in enumerate(self.box_nets):
            input_box = torch.cat((self.fg_embedder_position_box(box_offset[:,j, :,:]),
                               self.fg_embedder_viewdir_box(expanded_viewdir_reshape[:,j, :,:])), dim=-1)
            fg_box_raw = box_net(input_box)  # (N, *)
            box_sigmas.append(fg_box_raw['sigma'].view(dots_sh[0], 1,N_samples))
            box_rgbs.append(fg_box_raw['rgb'].view(dots_sh[0], 1, N_samples, 3))
            
        fg_box_raw_sigma = torch.cat(box_sigmas, dim=1) 
        fg_box_raw_rgb = torch.cat(box_rgbs, dim=1) 
#         print(fg_box_raw_sigma.view(32,100, self.box_number, N_samples)[2,50,...])

        colors = colors.unsqueeze(0).unsqueeze(-2).expand(dots_sh[0], -1, N_samples, 3)
        fg_box_raw_rgb = fg_box_raw_rgb.view(dots_sh[0], self.box_number, N_samples, 3) * colors

        # use sigmoid to filter sigma in empty space
        abs_dist = torch.abs(box_offset.reshape(dots_sh[0], self.box_number, N_samples, 3))
        inside_box = 0.5 / 28. - abs_dist
        weights = torch.prod(torch.sigmoid(inside_box * 50.), dim=-1)        
        fg_box_raw_sigma *= weights


        input = torch.cat((self.fg_embedder_position(fg_pts),
                           self.fg_embedder_viewdir(fg_viewdirs)), dim=-1)
        fg_raw = self.fg_net(input)
        
        
        fg_raw['rgb'] = fg_raw['rgb'] * check_shadow_aabb_inters(
            fg_pts, box_loc.unsqueeze(1).expand(dots_sh + [N_samples, self.box_number, 3]), box_sizes, r, self.box_number)

        # alpha blending
        fg_dists = fg_z_vals[..., 1:] - fg_z_vals[..., :-1]
        # account for view directions
        fg_dists = ray_d_norm * torch.cat((fg_dists, fg_z_max.unsqueeze(-1) - fg_z_vals[..., -1:]),
                                          dim=-1)  # [..., N_samples]

        fg_alpha = 1. - torch.exp(-(fg_raw['sigma'] + torch.sum(fg_box_raw_sigma, dim=1)) * fg_dists)  # [..., N_samples]

        T = torch.cumprod(1. - fg_alpha + TINY_NUMBER, dim=-1)  # [..., N_samples]
        bg_lambda = T[..., -1]
        T = torch.cat((torch.ones_like(T[..., 0:1]), T[..., :-1]), dim=-1)  # [..., N_samples]
        fg_weights = fg_alpha * T  # [..., N_samples]


        fg_rgb = torch.div(torch.sum(fg_box_raw_sigma.unsqueeze(-1) * fg_box_raw_rgb, dim=1)
                  + fg_raw['sigma'].unsqueeze(-1) * fg_raw['rgb'],
                  fg_raw['sigma'].unsqueeze(-1) + torch.sum(fg_box_raw_sigma, dim=1).unsqueeze(-1) + 0.0001)

        fg_rgb_map = torch.sum(fg_weights.unsqueeze(-1) * fg_rgb, dim=-2)  # [..., 3]
        fg_depth_map = torch.sum(fg_weights * fg_z_vals, dim=-1)  # [...,]
   
        # render background
        N_samples = bg_z_vals.shape[-1]
        bg_ray_o = ray_o.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        bg_ray_d = ray_d.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        bg_viewdirs = viewdirs.unsqueeze(-2).expand(dots_sh + [N_samples, 3])

        bg_pts, bg_depth_real = depth2pts_outside(bg_ray_o, bg_ray_d, bg_z_vals)  # [..., N_samples, 4]
        input = torch.cat((self.bg_embedder_position(bg_pts),
                           self.bg_embedder_viewdir(bg_viewdirs)), dim=-1)
        # near_depth: physical far; far_depth: physical near
        input = torch.flip(input, dims=[-2, ])
        bg_z_vals = torch.flip(bg_z_vals, dims=[-1, ])  # 1--->0
        bg_dists = bg_z_vals[..., :-1] - bg_z_vals[..., 1:]
        bg_dists = torch.cat((bg_dists, HUGE_NUMBER * torch.ones_like(bg_dists[..., 0:1])), dim=-1)  # [..., N_samples]
        bg_raw = self.bg_net(input)

        bg_alpha = 1. - torch.exp(-bg_raw['sigma'] * bg_dists)  # [..., N_samples]
        # Eq. (3): T
        # maths show weights, and summation of weights along a ray, are always inside [0, 1]
        T = torch.cumprod(1. - bg_alpha + TINY_NUMBER, dim=-1)[..., :-1]  # [..., N_samples-1]
        T = torch.cat((torch.ones_like(T[..., 0:1]), T), dim=-1)  # [..., N_samples]
        bg_weights = bg_alpha * T  # [..., N_samples]
        bg_rgb_map = torch.sum(bg_weights.unsqueeze(-1) * bg_raw['rgb'], dim=-2)  # [..., 3]
        bg_depth_map = torch.sum(bg_weights * bg_z_vals, dim=-1)  

        # composite foreground and background
        bg_rgb_map = bg_lambda.unsqueeze(-1) * bg_rgb_map

        _, bg_depth_map = depth2pts_outside(ray_o, ray_d, bg_depth_map)
        bg_depth_map = bg_lambda * bg_depth_map

        depth_map = fg_depth_map + bg_depth_map
        rgb_map = fg_rgb_map + bg_rgb_map
        
         
        
#         print('depth_map', depth_map.type())
#         print('rgb_map', rgb_map.type())
        ret = OrderedDict([('rgb', rgb_map),  # loss
                           ('fg_weights', fg_weights),  # importance sampling
                           ('bg_weights', bg_weights),  # importance sampling
                           ('fg_rgb', fg_rgb_map),  # below are for logging
                           ('fg_depth', fg_depth_map*30.),
                           ('bg_rgb', bg_rgb_map),
                           ('bg_depth', bg_depth_map*30.),
                           ('bg_lambda', bg_lambda),
                           ('depth_fgbg', depth_map*30.)])
#                            ('box_weights', torch.sum(fg_weights_boxes, dim=-1)),
#                            ('scene_weights', torch.sum(fg_weights_scene, dim=-1)),
                           
        return ret


class NerfNetMoreBox(nn.Module):
    def __init__(self, args):
        super().__init__()
        # foreground
        self.fg_embedder_position = Embedder(input_dim=3,
                                             max_freq_log2=args.max_freq_log2 - 1,
                                             N_freqs=args.max_freq_log2)
        self.fg_embedder_viewdir = Embedder(input_dim=3,
                                            max_freq_log2=args.max_freq_log2_viewdirs - 1,
                                            N_freqs=args.max_freq_log2_viewdirs)

        self.fg_embedder_position_box = Embedder(input_dim=3,
                                             max_freq_log2=args.max_freq_log2 - 1,
                                             N_freqs=args.max_freq_log2)

        self.fg_embedder_viewdir_box = Embedder(input_dim=3,
                                            max_freq_log2=args.max_freq_log2_viewdirs - 1,
                                            N_freqs=args.max_freq_log2_viewdirs)


        self.fg_net = MLPNet(D=args.netdepth, W=args.netwidth,
                             input_ch=self.fg_embedder_position.out_dim,
                             input_ch_viewdirs=self.fg_embedder_viewdir.out_dim,
                             use_viewdirs=args.use_viewdirs)
        # background; bg_pt is (x, y, z, 1/r)
        self.bg_embedder_position = Embedder(input_dim=4,
                                             max_freq_log2=args.max_freq_log2 - 1,
                                             N_freqs=args.max_freq_log2)
        self.bg_embedder_viewdir = Embedder(input_dim=3,
                                            max_freq_log2=args.max_freq_log2_viewdirs - 1,
                                            N_freqs=args.max_freq_log2_viewdirs)
        self.bg_net = MLPNet(D=args.netdepth, W=args.netwidth,
                             input_ch=self.bg_embedder_position.out_dim,
                             input_ch_viewdirs=self.bg_embedder_viewdir.out_dim,
                             use_viewdirs=args.use_viewdirs)
        # this is the net for the box

        # input to this should be box position and sampled position
        self.box_net = MLPNet(D=4, W=args.netwidth,
                              input_ch=self.fg_embedder_position_box.out_dim,
                              input_ch_viewdirs=self.fg_embedder_viewdir_box.out_dim,
                              use_viewdirs=args.use_viewdirs)

        self.box_number = args.box_number

        # self.box_size = [float(size) for size in args.box_size.split(',')]
        #
        # # device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else 'cpu')
        # self.box_size = torch.tensor(self.box_size).cuda().to(torch.cuda.current_device())

    def forward(self, ray_o, ray_d, fg_z_max, fg_z_vals, bg_z_vals, box_loc, box_props, query_box_only=False):
        '''orch.c
        :param ray_o, ray_d: [..., 3]
        :param fg_z_max: [...,]
        :param fg_z_vals, bg_z_vals: [..., N_samples]
        :param box_locs: [..., 3]  (N. [x,y, z])
        :return
        '''

#         print(ray_o.shape, ray_d.shape, fg_z_max.shape, fg_z_vals.shape, bg_z_vals.shape)
      
        ray_d_norm = torch.norm(ray_d, dim=-1, keepdim=True)  # [..., 1]
        viewdirs = ray_d / ray_d_norm  # [..., 3]
        dots_sh = list(ray_d.shape[:-1])
        
#         fg_z_vals.register_hook(lambda grad: print("[fg_z_vals]" +" Gradients bigger than 1e6:", bool((grad > 1e6).any())))  
        

        ######### render foreground
        N_samples = fg_z_vals.shape[-1]
        fg_ray_o = ray_o.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        fg_ray_d = ray_d.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        fg_viewdirs = viewdirs.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        fg_pts = fg_ray_o + fg_z_vals.unsqueeze(-1) * fg_ray_d

        
        box_loc = box_loc.view(-1, self.box_number, 3)
        colors, box_sizes, box_rot =  box_props[:, 0:3], box_props[:, 3:6], box_props[:, 6:]

        assert box_sizes.shape == (self.box_number, 3), 'box_sizes shape is wrong'
        assert box_rot.shape == (self.box_number, 3), 'box_rot shape is wrong'

        fg_box_raw_lst = []
        self.box_net = self.box_net.to(torch.cuda.current_device())

        r = euler_angles_to_matrix(torch.deg2rad(torch.cat([box_rot[:,2:],-1*box_rot[:,1:2], -1*box_rot[:,0:1] ], 
                                                           dim=-1)), convention='ZYX')
        r_mat_expand = r.unsqueeze(0).unsqueeze(1).expand(dots_sh + [N_samples, self.box_number, 3, 3])

        box_offset = ((torch.matmul(torch.inverse(r_mat_expand) ,
                        (fg_pts.unsqueeze(-2).expand(dots_sh + [N_samples, self.box_number, 3])
                        - box_loc.unsqueeze(1).expand(dots_sh + [N_samples, self.box_number, 3])).unsqueeze(-1)).squeeze(-1))/
                      (box_sizes*30.+ TINY_NUMBER).unsqueeze(0).unsqueeze(0))\
            .permute(0,2,1,3).reshape(dots_sh[0]*self.box_number, N_samples, 3)
        
#         box_offset.register_hook(lambda grad: print("[box_offset]" +" Gradients bigger than 1e6:", bool((grad > 1e6).any())))  
        
        
        expanded_viewdir = fg_viewdirs.unsqueeze(-2).expand(dots_sh + [N_samples, self.box_number, 3]).permute(0,2,1,3)
        expanded_viewdir_reshape = expanded_viewdir.reshape(dots_sh[0]*self.box_number, N_samples, 3)
        
#         expanded_viewdir.register_hook(lambda grad: print("[expanded_viewdir]" +" Gradients bigger than 1e6:", bool((grad > 1e6).any())))  
        
        
        input_box = torch.cat((self.fg_embedder_position_box(box_offset),
                               self.fg_embedder_viewdir_box(expanded_viewdir_reshape)), dim=-1)
        
#         input_box.register_hook(lambda grad: print("[input_box]" +" Gradients bigger than 1e6:", bool((grad > 1e6).any())))  
        

        assert input_box.shape == (dots_sh[0]*self.box_number, N_samples, self.fg_embedder_position_box.out_dim + self.fg_embedder_viewdir_box.out_dim)

        fg_box_raw = self.box_net(input_box)  # (N, *)
        
#         fg_box_raw.register_hook(lambda grad: print("[fg_box_raw]" +" Gradients bigger than 1e6:", bool((grad > 1e6).any()))) 
#         
        fg_box_raw_sigma = fg_box_raw['sigma'].view(dots_sh[0], self.box_number, N_samples)
        
#         print(fg_box_raw_sigma.view(32,100, self.box_number, N_samples)[2,50,...])

        colors = colors.unsqueeze(0).unsqueeze(-2).expand(dots_sh[0], -1, N_samples, 3)
        fg_box_raw_rgb = fg_box_raw['rgb'].view(dots_sh[0], self.box_number, N_samples, 3) * colors

        # use sigmoid to filter sigma in empty space
        abs_dist = torch.abs(box_offset.reshape(dots_sh[0], self.box_number, N_samples, 3))
        inside_box = 0.5 / 28. - abs_dist
<<<<<<< HEAD
        weights = torch.prod(torch.sigmoid(inside_box * 20.), dim=-1)        
=======
        weights = torch.prod(torch.sigmoid(inside_box * 50.), dim=-1)        
>>>>>>> 6dfd4e02438b009a09e047057ae6200f7ffb4f84
        fg_box_raw_sigma *= weights
        
#         fg_box_raw_sigma.register_hook(lambda grad: print("[fg_box_raw_sigma]" +" Gradients bigger than 1e6:", bool((grad > 1e6).any()))) 


        input = torch.cat((self.fg_embedder_position(fg_pts),
                           self.fg_embedder_viewdir(fg_viewdirs)), dim=-1)
        fg_raw = self.fg_net(input)
        
#         fg_raw.register_hook(lambda grad: print("[fg_raw]" +" Gradients bigger than 1e6:", bool((grad > 1e6).any()))) 
        
        
        fg_raw['rgb'] = fg_raw['rgb'] * check_shadow_aabb_inters(
            fg_pts, box_loc.unsqueeze(1).expand(dots_sh + [N_samples, self.box_number, 3]), box_sizes, r, self.box_number)
        
#         fg_raw['rgb'].register_hook(lambda grad: print("[fg_raw['rgb']]" +" Gradients bigger than 1e6:", bool((grad > 1e6).any()))) 

        # alpha blending
        fg_dists = fg_z_vals[..., 1:] - fg_z_vals[..., :-1]
        # account for view directions
        fg_dists = ray_d_norm * torch.cat((fg_dists, fg_z_max.unsqueeze(-1) - fg_z_vals[..., -1:]),
                                          dim=-1)  # [..., N_samples]

        fg_alpha = 1. - torch.exp(-(fg_raw['sigma'] + torch.sum(fg_box_raw_sigma, dim=1)) * fg_dists)  # [..., N_samples]
        
#         fg_alpha.register_hook(lambda grad: print("[fg_alpha]" +" Gradients bigger than 1e6:", bool((grad > 1e6).any()))) 

        T = torch.cumprod(1. - fg_alpha + TINY_NUMBER, dim=-1)  # [..., N_samples]
        bg_lambda = T[..., -1]
        T = torch.cat((torch.ones_like(T[..., 0:1]), T[..., :-1]), dim=-1)  # [..., N_samples]
        fg_weights = fg_alpha * T  # [..., N_samples]
        
#         fg_weights.register_hook(lambda grad: print("[fg_weights]" +" Gradients bigger than 1e6:", bool((grad > 1e6).any()))) 


        fg_rgb = torch.div(torch.sum(fg_box_raw_sigma.unsqueeze(-1) * fg_box_raw_rgb, dim=1)
                  + fg_raw['sigma'].unsqueeze(-1) * fg_raw['rgb'],
                  fg_raw['sigma'].unsqueeze(-1) + torch.sum(fg_box_raw_sigma, dim=1).unsqueeze(-1) + 0.0001)
        
#         fg_rgb.register_hook(lambda grad: print("[fg_rgb]" +" Gradients bigger than 1e6:", bool((grad > 1e6).any()))) 

        fg_rgb_map = torch.sum(fg_weights.unsqueeze(-1) * fg_rgb, dim=-2)  # [..., 3]
        fg_depth_map = torch.sum(fg_weights * fg_z_vals, dim=-1)  # [...,]
   
        # render background
        N_samples = bg_z_vals.shape[-1]
        bg_ray_o = ray_o.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        bg_ray_d = ray_d.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        bg_viewdirs = viewdirs.unsqueeze(-2).expand(dots_sh + [N_samples, 3])

        bg_pts, bg_depth_real = depth2pts_outside(bg_ray_o, bg_ray_d, bg_z_vals)  # [..., N_samples, 4]
        input = torch.cat((self.bg_embedder_position(bg_pts),
                           self.bg_embedder_viewdir(bg_viewdirs)), dim=-1)
        # near_depth: physical far; far_depth: physical near
        input = torch.flip(input, dims=[-2, ])
        bg_z_vals = torch.flip(bg_z_vals, dims=[-1, ])  # 1--->0
        
        
        bg_dists = bg_z_vals[..., :-1] - bg_z_vals[..., 1:]
        bg_dists = torch.cat((bg_dists, HUGE_NUMBER * torch.ones_like(bg_dists[..., 0:1])), dim=-1)  # [..., N_samples]
        bg_raw = self.bg_net(input)

#         bg_raw.register_hook(lambda grad: print("[bg_raw]" +" Gradients bigger than 1e6:", bool((grad > 1e6).any()))) 

        bg_alpha = 1. - torch.exp(-bg_raw['sigma'] * bg_dists)  # [..., N_samples]
        # Eq. (3): T
        # maths show weights, and summation of weights along a ray, are always inside [0, 1]
        T = torch.cumprod(1. - bg_alpha + TINY_NUMBER, dim=-1)[..., :-1]  # [..., N_samples-1]
        T = torch.cat((torch.ones_like(T[..., 0:1]), T), dim=-1)  # [..., N_samples]
        bg_weights = bg_alpha * T  # [..., N_samples]
        bg_rgb_map = torch.sum(bg_weights.unsqueeze(-1) * bg_raw['rgb'], dim=-2)  # [..., 3]
        bg_depth_map = torch.sum(bg_weights * bg_z_vals, dim=-1)  

        # composite foreground and background
        bg_rgb_map = bg_lambda.unsqueeze(-1) * bg_rgb_map

        _, bg_depth_map = depth2pts_outside(ray_o, ray_d, bg_depth_map)
        bg_depth_map = bg_lambda * bg_depth_map

        depth_map = fg_depth_map + bg_depth_map
#         fg_depth_map.register_hook(lambda grad: print("[fg_depth_map]" +" Gradients bigger than 1e6:", bool((grad > 1e6).any()))) 
#         bg_depth_map.register_hook(lambda grad: print("[bg_depth_map]" +" Gradients bigger than 1e6:", bool((grad > 1e6).any()))) 
        
        rgb_map = fg_rgb_map + bg_rgb_map
        
#         print('here')
#         fg_rgb_map.register_hook(lambda grad: print(grad) )
#         bg_rgb_map.register_hook(lambda grad: print("[bg_rgb_map]" +" Gradients bigger than 1e6:", bool((grad > 1e6).any()))) 
        
#         print('depth_map', depth_map.type())
#         print('rgb_map', rgb_map.type())
        ret = OrderedDict([('rgb', rgb_map),  # loss
                           ('fg_weights', fg_weights),  # importance sampling
                           ('bg_weights', bg_weights),  # importance sampling
                           ('fg_rgb', fg_rgb_map),  # below are for logging
                           ('fg_depth', fg_depth_map*30.),
                           ('bg_rgb', bg_rgb_map),
                           ('bg_depth', bg_depth_map*30.),
                           ('bg_lambda', bg_lambda),
                           ('depth_fgbg', depth_map*30.)])
#                            ('box_weights', torch.sum(fg_weights_boxes, dim=-1)),
#                            ('scene_weights', torch.sum(fg_weights_scene, dim=-1)),
                           
        return ret


def remap_name(name):
    name = name.replace('.', '-')  # dot is not allowed by pytorch
    if name[-1] == '/':
        name = name[:-1]
    idx = name.rfind('/')
    for i in range(2):
        if idx >= 0:
            idx = name[:idx].rfind('/')
    return name[idx + 1:]


class NerfNetWithAutoExpo(nn.Module):
    def __init__(self, args, optim_autoexpo=False, img_names=None):
        super().__init__()
        self.nerf_net = NerfNet(args)

        self.optim_autoexpo = optim_autoexpo
        if self.optim_autoexpo:
            assert (img_names is not None)
            logger.info('Optimizing autoexposure!')

            self.img_names = [remap_name(x) for x in img_names]
            logger.info('\n'.join(self.img_names))
            self.autoexpo_params = nn.ParameterDict(
                OrderedDict([(x, nn.Parameter(torch.Tensor([0.5, 0.]))) for x in self.img_names]))

    def forward(self, ray_o, ray_d, fg_z_max, fg_z_vals, bg_z_vals, img_name=None):
        '''
        :param ray_o, ray_d: [..., 3]
        :param fg_z_max: [...,]
        :param fg_z_vals, bg_z_vals: [..., N_samples]
        :return
        '''
        ret = self.nerf_net(ray_o, ray_d, fg_z_max, fg_z_vals, bg_z_vals )

        if img_name is not None:
            img_name = remap_name(img_name)
        if self.optim_autoexpo and (img_name in self.autoexpo_params):
            autoexpo = self.autoexpo_params[img_name]
            scale = torch.abs(autoexpo[0]) + 0.5  # make sure scale is always positive
            shift = autoexpo[1]
            ret['autoexpo'] = (scale, shift)

        return ret


class NerfNetWithAutoExpo(nn.Module):
    def __init__(self, args, optim_autoexpo=False, img_names=None):
        super().__init__()
        self.nerf_net = NerfNet(args)

        self.optim_autoexpo = optim_autoexpo
        if self.optim_autoexpo:
            assert (img_names is not None)
            logger.info('Optimizing autoexposure!')

            self.img_names = [remap_name(x) for x in img_names]
            logger.info('\n'.join(self.img_names))
            self.autoexpo_params = nn.ParameterDict(
                OrderedDict([(x, nn.Parameter(torch.Tensor([0.5, 0.]))) for x in self.img_names]))

    def forward(self, ray_o, ray_d, fg_z_max, fg_z_vals, bg_z_vals, img_name=None):
        '''
        :param ray_o, ray_d: [..., 3]
        :param fg_z_max: [...,]
        :param fg_z_vals, bg_z_vals: [..., N_samples]
        :return
        '''
        ret = self.nerf_net(ray_o, ray_d, fg_z_max, fg_z_vals, bg_z_vals)

        if img_name is not None:
            img_name = remap_name(img_name)
        if self.optim_autoexpo and (img_name in self.autoexpo_params):
            autoexpo = self.autoexpo_params[img_name]
            scale = torch.abs(autoexpo[0]) + 0.5  # make sure scale is always positive
            shift = autoexpo[1]
            ret['autoexpo'] = (scale, shift)

        return ret


class NerfNetBoxOnlyWithAutoExpo(nn.Module):
    def __init__(self, args, optim_autoexpo=False, img_names=None):
        super().__init__()
        self.nerf_net = NerfNetBoxOnly(args)

        self.optim_autoexpo = optim_autoexpo
        if self.optim_autoexpo:
            assert (img_names is not None)
            logger.info('Optimizing autoexposure!')

            self.img_names = [remap_name(x) for x in img_names]
            logger.info('\n'.join(self.img_names))
            self.autoexpo_params = nn.ParameterDict(
                OrderedDict([(x, nn.Parameter(torch.Tensor([0.5, 0.]))) for x in self.img_names]))

    def forward(self, ray_o, ray_d, fg_z_max, fg_z_vals, img_name=None):
        '''
        :param ray_o, ray_d: [..., 3]
        :param fg_z_max: [...,]
        :param fg_z_vals, bg_z_vals: [..., N_samples]
        :return
        '''
        # print('forward, img_name:{}'.format(img_name))
        ret = self.nerf_net(ray_o, ray_d, fg_z_max, fg_z_vals)

        if img_name is not None:
            img_name = remap_name(img_name)
        if self.optim_autoexpo and (img_name in self.autoexpo_params):
            autoexpo = self.autoexpo_params[img_name]
            scale = torch.abs(autoexpo[0]) + 0.5  # make sure scale is always positive
            shift = autoexpo[1]
            ret['autoexpo'] = (scale, shift)

        return ret
class NerfNetMoreBoxIndepWithAutoExpo(nn.Module):
    def __init__(self, args, optim_autoexpo=False, img_names=None):
        super().__init__()
        self.nerf_net = NerfNetMoreBoxIndep(args)

        self.optim_autoexpo = optim_autoexpo
        if self.optim_autoexpo:
            assert (img_names is not None)
            logger.info('Optimizing autoexposure!')

            self.img_names = [remap_name(x) for x in img_names]
            logger.info('\n'.join(self.img_names))
            self.autoexpo_params = nn.ParameterDict(
                OrderedDict([(x, nn.Parameter(torch.Tensor([0.5, 0.]))) for x in self.img_names]))

    def forward(self, ray_o, ray_d, fg_z_max, fg_z_vals, bg_z_vals, box_loc, box_props, query_box_only=False, img_name=None):
        '''
        :param ray_o, ray_d: [..., 3]
        :param fg_z_max: [...,]
        :param fg_z_vals, bg_z_vals: [..., N_samples]
        :param box_loc: [..., 3]
        :return
        '''
        ret = self.nerf_net(ray_o, ray_d, fg_z_max, fg_z_vals, bg_z_vals, box_loc, box_props, query_box_only)

        if img_name is not None:
            img_name = remap_name(img_name)
        if self.optim_autoexpo and (img_name in self.autoexpo_params):
            autoexpo = self.autoexpo_params[img_name]
            scale = torch.abs(autoexpo[0]) + 0.5  # make sure scale is always positive
            shift = autoexpo[1]
            ret['autoexpo'] = (scale, shift)

        return ret
class NerfNetMoreBoxWithAutoExpo(nn.Module):
    def __init__(self, args, optim_autoexpo=False, img_names=None):
        super().__init__()
        self.nerf_net = NerfNetMoreBox(args)

        self.optim_autoexpo = optim_autoexpo
        if self.optim_autoexpo:
            assert (img_names is not None)
            logger.info('Optimizing autoexposure!')

            self.img_names = [remap_name(x) for x in img_names]
            logger.info('\n'.join(self.img_names))
            self.autoexpo_params = nn.ParameterDict(
                OrderedDict([(x, nn.Parameter(torch.Tensor([0.5, 0.]))) for x in self.img_names]))

    def forward(self, ray_o, ray_d, fg_z_max, fg_z_vals, bg_z_vals, box_loc, box_props, query_box_only=False, img_name=None):
        '''
        :param ray_o, ray_d: [..., 3]
        :param fg_z_max: [...,]
        :param fg_z_vals, bg_z_vals: [..., N_samples]
        :param box_loc: [..., 3]
        :return
        '''
        ret = self.nerf_net(ray_o, ray_d, fg_z_max, fg_z_vals, bg_z_vals, box_loc, box_props, query_box_only)

        if img_name is not None:
            img_name = remap_name(img_name)
        if self.optim_autoexpo and (img_name in self.autoexpo_params):
            autoexpo = self.autoexpo_params[img_name]
            scale = torch.abs(autoexpo[0]) + 0.5  # make sure scale is always positive
            shift = autoexpo[1]
            ret['autoexpo'] = (scale, shift)

        return ret

class NerfNetBoxWithAutoExpo(nn.Module):
    def __init__(self, args, optim_autoexpo=False, img_names=None):
        super().__init__()
        self.nerf_net = NerfNetBox(args)

        self.optim_autoexpo = optim_autoexpo
        if self.optim_autoexpo:
            assert (img_names is not None)
            logger.info('Optimizing autoexposure!')

            self.img_names = [remap_name(x) for x in img_names]
            logger.info('\n'.join(self.img_names))
            self.autoexpo_params = nn.ParameterDict(
                OrderedDict([(x, nn.Parameter(torch.Tensor([0.5, 0.]))) for x in self.img_names]))

    def forward(self, ray_o, ray_d, fg_z_max, fg_z_vals, bg_z_vals, box_loc, query_box_only=False, img_name=None):
        '''
        :param ray_o, ray_d: [..., 3]
        :param fg_z_max: [...,]
        :param fg_z_vals, bg_z_vals: [..., N_samples]
        :param box_loc: [..., 3]
        :return
        '''
        ret = self.nerf_net(ray_o, ray_d, fg_z_max, fg_z_vals, bg_z_vals, box_loc, query_box_only)

        if img_name is not None:
            img_name = remap_name(img_name)
        if self.optim_autoexpo and (img_name in self.autoexpo_params):
            autoexpo = self.autoexpo_params[img_name]
            scale = torch.abs(autoexpo[0]) + 0.5  # make sure scale is always positive
            shift = autoexpo[1]
            ret['autoexpo'] = (scale, shift)

        return ret

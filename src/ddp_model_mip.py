import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# import numpy as np
from utils import TINY_NUMBER, HUGE_NUMBER
from collections import OrderedDict
from nerf_network import Embedder, MLPNet, MLPNetClassier, MLPNetMip
import os
import logging
from pytorch3d.transforms import euler_angles_to_matrix
from  helpers import check_shadow, check_shadow_aabb_inters


logger = logging.getLogger(__package__)


######################################################################################
# wrapper to simplify the use of nerfnet
######################################################################################
def bgzval2pts_outside(ray_o, ray_d, depth):
    '''
    ray_o, ray_d: [..., 3]
    depth: [...]; inverse of distance to sphere origin
    '''
    # note: d1 becomes negative if this mid point is behind camera
    d1 = -torch.sum(ray_d * ray_o, dim=-1) / torch.sum(ray_d * ray_d, dim=-1)  # tb
    p_mid = ray_o + d1.unsqueeze(-1) * ray_d  # b
    p_mid_norm = torch.norm(p_mid, dim=-1)
    ray_d_cos = 1. / (torch.norm(ray_d, dim=-1) + TINY_NUMBER)
    
#     print('p_mid_norm', p_mid_norm.shape)
#     print('depth', depth.shape)
    theta = torch.asin(p_mid_norm.unsqueeze(-1) * depth)  # depth is inside [0, 1]
    depth_real = 1. / (depth) * torch.cos(theta) * ray_d_cos.unsqueeze(-1) + d1.unsqueeze(-1) ## ray_d_cos brings absolute distance to z distance
    return depth_real
    
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
    depth_real = 1. / (depth + TINY_NUMBER) * np.cos(theta) * ray_d_cos[..., None] + d1[..., None] # this gives depth on z axis (camera)
    return pts, depth_real





class DepthOracle(nn.Module):

    def __init__(self, args):
        super().__init__()

        pos_ch_fg = (args.front_sample - 1) * 3
        pos_ch_bg = (args.back_sample - 1) * 4

        
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

        depth_likeli_fg = self.ora_net_fg(ray_o, viewdirs, points_fg)
        depth_likeli_bg = self.ora_net_bg(ray_o_bg, viewdirs, points_bg)

        ret = OrderedDict([('likeli_fg', depth_likeli_fg),
                           ('likeli_bg', depth_likeli_bg)])

        return ret





    
def contract_gaussian(mean, cov_full):
    '''
    mean and cov in world coord
    *before: Mean, cov from each segment ---> x ddT lift --> expected sin/cos with formula ---> IPE
    *mip360: Mean, cov from each segment ---> x ddT lift --> contracted (f(mean), J(mean) * cov * J(mean)^T) --> 
    multiply with basis P -->  expected sin/cos with formula
    
    param: mean [N, Ns, 3]
    param: cov_full [N, Ns, 3, 3]
    
    
    '''
    # contract 
    f = lambda x: ( 2. - 1. / (torch.norm(x, dim=-1, keepdim=True)+TINY_NUMBER)
                ) * (x / (torch.norm(x, dim=-1, keepdim=True)+TINY_NUMBER))
    
    N_rays, N_Samples = mean.shape[0], mean.shape[1]
    J_mean = torch.autograd.functional.jacobian(f, mean.reshape([N_rays* N_Samples, -1]))
    J_mean = torch.diagonal(J_mean, dim1=0, dim2=2).reshape([N_rays, N_Samples, 3,3])
    mean_contract = f(mean)
    
    
#     print('mean', mean.shape)
#     print('mean_contract', mean_contract.shape)
#     print('J_mean', J_mean.shape)
    cov_contract = torch.matmul(torch.matmul( J_mean, cov_full), torch.transpose(J_mean, -2,-1))
    return mean_contract, cov_contract
    
    
   

    
    
def lift_gaussian(d, t_mean, t_var, r_var, diag, bg=False):
    """Lift a Gaussian defined along a ray to 3D coordinates."""
    
#     print('d', d.shape)
#     print('t mean', t_mean.shape)
    d = d.unsqueeze(-2)
    mean = d * t_mean.unsqueeze(-1) #  [N, Nsample, 3]
    
    
    d_mag_sq = torch.sum(d**2, dim=-1, keepdims=True)
    d_mag_sq = torch.max(1e-8 * torch.ones_like(d_mag_sq), d_mag_sq) # vector norm

#     print('lift gaus')
#     print('t_var', t_var.shape)
#     print('r_var', r_var.shape)
        
    if diag:
        d_outer_diag = d**2
        null_outer_diag = 1. - d_outer_diag / d_mag_sq
        t_cov_diag = t_var.unsqueeze(-1) * d_outer_diag
        xy_cov_diag = r_var.unsqueeze(-1) * null_outer_diag
        cov_diag = t_cov_diag + xy_cov_diag # [N, Nsample, 3]    
#         print('mean', mean.shape)
#         print('cov', cov_diag.shape)
        return mean, cov_diag
    else:
#         print('d',d.shape)
        d_outer = torch.matmul(d.unsqueeze(-1), d.unsqueeze(-2))
        eye = torch.eye(d.shape[-1]).cuda()
        null_outer = eye - torch.matmul(d.unsqueeze(-1), (d / d_mag_sq).unsqueeze(-2))
#         print('d_outer', d_outer.shape)
#         print('null_outer', null_outer.shape)
#         print('t_var', t_var.shape)
#         print('r_var', r_var.shape)
        
        
        t_cov = t_var.unsqueeze(-1).unsqueeze(-1) * d_outer
        xy_cov = r_var.unsqueeze(-1).unsqueeze(-1) * null_outer
        cov = t_cov + xy_cov
#         print('mean', mean.shape)
#         print('cov', cov.shape)
        return mean, cov # [N, Nsample, 3,3]  
        
def checknan(t):
    print(torch.isnan(t).any())

def get_nan_index(t):
    print(torch.isnan(t).nonzero(as_tuple=True))
    
    
def cast_rays(t_vals, origins, directions, radii, diag=True):
    """Cast rays (cone- or cylinder-shaped) and featurize sections of it.
    Args:
    t_vals: float array, the "fencepost" distances along the ray. [N, N samples]
    origins: float array, the ray origin coordinates. [N, N samples, 3]
    directions: float array, the ray direction vectors.[N, N samples, 3] !!!!!!!!!!!!!!check whether this is normalized 
    radii: float array, the radii (base radii for cones) of the rays. [N, 1]
    diag: boolean, whether or not the covariance matrices should be diagonal.
    Returns:
    a tuple of arrays of means and covariances.
    """

    t0 = t_vals[..., :-1]
    t1 = t_vals[..., 1:]
    means, covs = conical_frustum_to_gaussian(directions, t0, t1, radii, diag)
    
#     print('cast rays')
#     print('origins', origins.shape)
#     print('means', means.shape)
    means = means + origins.unsqueeze(-2)
    return means, covs
    
def conical_frustum_to_gaussian(d, t0, t1, base_radius, diag, stable=True):
    """Approximate a conical frustum as a Gaussian distribution (mean+cov).
    Assumes the ray is originating from the origin, and base_radius is the
    radius at dist=1. Doesn't assume `d` is normalized.
    Args:
    d: jnp.float32 3-vector, the axis of the cone
    t0: float, the starting distance of the frustum.
    t1: float, the ending distance of the frustum.
    base_radius: float, the scale of the radius as a function of distance.
    diag: boolean, whether or the Gaussian will be diagonal or full-covariance.
    stable: boolean, whether or not to use the stable computation described in
      the paper (setting this to False will cause catastrophic failure).
    Returns:
    a Gaussian (mean and covariance).
    """
#     print('--conical_frustum_to_gaussian--')
#     print('t0', t0.shape)
#     print('t1', t1.shape)
    mu = (t0 + t1) / 2. # [N, Nsample-1]
    hw = (t1 - t0) / 2. # [N, Nsample-1]
    t_mean = mu + (2. * mu * hw**2) / (3. * mu**2 + hw**2) # local std on the ray
    t_var = (hw**2) / 3. - (4. / 15.) * ((hw**4 * (12. * mu**2 - hw**2)) /
                                      (3. * mu**2 + hw**2)**2) # [N, Nsample-1]
    r_var = base_radius**2 * ((mu**2) / 4. + (5. / 12.) * hw**2 - 4. / 15. *
                              (hw**4) / (3. * mu**2 + hw**2)) #  [N, Nsample-1]

    return lift_gaussian(d, t_mean, t_var, r_var, diag)
    

    # multiply with P
P = torch.tensor([
        0.8506508, 0., 0.5257311,
        0.809017, 0.5, 0.309017,
        0.5257311, 0.8506508, 0.,
        1., 0., 0.,
        0.809017, 0.5, -0.309017,
        0.8506508, 0., -0.5257311,
        0.309017, 0.809017, -0.5,
        0., 0.5257311, -0.8506508,
        0.5, 0.309017, -0.809017,
        0., 1., 0.,
        -0.5257311, 0.8506508, 0.,
        -0.309017, 0.809017, -0.5,
        0., 0.5257311, 0.8506508,
        -0.309017, 0.809017, 0.5,
        0.309017, 0.809017, 0.5,
        0.5, 0.309017, 0.809017,
        0.5, -0.309017, 0.809017,
        0., 0., 1.,
        -0.5, 0.309017, 0.809017,
        -0.809017, 0.5, 0.309017,
        -0.809017, 0.5, -0.309017]).cuda().float().reshape((3,21))

    
def integrated_pos_enc(x_coord, min_deg, max_deg, diag=True):
    """Encode `x` with sinusoids scaled by 2^[min_deg:max_deg-1].

    Args:
    x_coord: a tuple containing:
    x, variables to be encoded. Should
      be in [-pi, pi]. [N, Nsample, 3]
    x_cov, jnp.ndarray, covariance matrices for `x`. [N, Nsample, 3] or [N, Nsample, 3]
    min_deg: int, the min degree of the encoding.
    max_deg: int, the max degree of the encoding.
    diag: bool, if true, expects input covariances to be diagonal (full
      otherwise).

    Returns:
    encoded: jnp.ndarray, encoded variables. [N, Nsample, 3*2*16]
    """
    
    if not diag:
        x, x_cov = x_coord
        basis = P # shape [21, 3]
         #jnp.concatenate(
#             [2**i * jnp.eye(num_dims) for i in range(min_deg, max_deg)], 1) 
        y = torch.matmul(x, basis)
        # Get the diagonal of a covariance matrix (ie, variance). This is equivalent
        # to jax.vmap(jnp.diag)((basis.T @ covs) @ basis).
        y_var = torch.sum(torch.matmul(x_cov, basis.unsqueeze(0).unsqueeze(0)) * basis, dim=-2) # [N, Ns, 21 ]
        shape = list(y.shape[:-1]) + [-1]
        scales = torch.tensor([2**i for i in range(min_deg, max_deg)]).type_as(x)
        y = torch.reshape(y.unsqueeze(-2) * scales.unsqueeze(-1), shape) # [N, Ns, 21*16 ]
        y_var = torch.reshape(y_var.unsqueeze(-2) * scales.unsqueeze(-1)**2, shape) # [N, Ns, 21*16 ]

    else:
        x, x_cov_diag = x_coord # both [N, Ns, 3]
        scales = torch.tensor([2**i for i in range(min_deg, max_deg)]).type_as(x)
        shape = list(x.shape[:-1]) + [-1]
        y = torch.reshape(x.unsqueeze(-2) * scales.unsqueeze(-1), shape)  # [N, Ns, 3*16]
        y_var = torch.reshape(x_cov_diag.unsqueeze(-2) * scales.unsqueeze(-1)**2, shape) # [N, Ns, 3*16]

    return expected_sin(
      torch.cat([y, y + 0.5 * np.pi], axis=-1),
      torch.cat([y_var] * 2, axis=-1))[0]  # shape ipe (1024, 128, 96)  or (1024, 128, 21*16*2)
    
def expected_sin(x, x_var):
    """Estimates mean and variance of sin(z), z ~ N(x, var)."""
    # When the variance is wide, shrink sin towards zero.
    
#     print('--expected sin--')
#     print('x', x.shape)
#     print('x_var', x.shape)
    
    y = torch.exp(-0.5 * x_var) * torch.sin(x)
    y_var = torch.max(
      torch.zeros_like(x_var), 0.5 * (1. - torch.exp(-2. * x_var) * torch.cos(2. * x)) - y**2)
    
#     print('y', y.shape)
    return y, y_var

class MipNerf(nn.Module):
    def __init__(self, args):
        super().__init__()
        # foreground

        self.embedder_viewdir = Embedder(input_dim=3,
                    max_freq_log2=args.max_freq_log2_viewdirs - 1,include_input=False,
                    N_freqs=args.max_freq_log2_viewdirs)


        self.fg_net = MLPNetMip(num_samples=int(args.cascade_samples), feature_dim=16*2*3, 
                                direction_dim=args.max_freq_log2_viewdirs*3*2)
        self.bg_net = MLPNetMip(num_samples=int(args.cascade_samples), feature_dim=16*2*21,
                               direction_dim=args.max_freq_log2_viewdirs*3*2)

        self.box_number = args.box_number
        self.density_bias = -1.
        self.rgb_padding = 0.001
        self.min_deg_point = 0
        self.max_deg_point = 16
        
    def volume_rendering(self, density, rgb,  dists, z_vals):
        
        print('---volume_rendering--')
#         print('density', density.shape)
#         print('dists', dists.shape)
        
        alpha = 1. - torch.exp(-density.squeeze(-1) * dists)  # [..., N_samples]
        T = torch.cumprod(1. - alpha + TINY_NUMBER, dim=-1)  # [..., N_samples]
        T = torch.cat((torch.ones_like(T[..., 0:1]), T[..., :-1]), dim=-1) # [..., N_samples]
        weights = alpha * T  # [..., N_samples]
        rgb_map = torch.sum(weights.unsqueeze(-1) * rgb, dim=-2)  # [..., 3]
        acc = torch.sum(weights, dim=-1)    
        # use mid points for depth
        t_mids = 0.5 * (z_vals[..., :-1] + z_vals[..., 1:])
        depth_map = torch.sum(weights * t_mids, dim=-1) / (acc+ TINY_NUMBER) # [...,] uses midpoint instead

        checknan(alpha)
        checknan(T)
        checknan(weights)
        checknan(rgb_map)
        checknan(acc)
        checknan(t_mids)
        checknan(depth_map)
        return rgb_map, depth_map, T
        
    def process_raw(self, raw_rgb, raw_density):
        rgb = torch.sigmoid(raw_rgb)
        rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding
        density = F.softplus(raw_density + self.density_bias)
        return rgb, density

    def forward(self, ray_o, ray_d, fg_z_max, fg_z_vals, bg_z_vals, radii):
        '''orch.c
        :param ray_o, ray_d: [..., 3]
        :param radii: [...,1]
        :param fg_z_max: [...,]
        :param fg_z_vals, bg_z_vals: [..., N_samples] fencepoints
        :param box_locs: [..., 3]  (N. [x,y, z])
        :return
        '''

        N_rays, N_samples = fg_z_vals.shape[0], fg_z_vals.shape[1] -1               
#         ray_o = ray_o.unsqueeze(-2).expand([N_rays, N_samples, 3])        
        
        # print(ray_o.shape, ray_d.shape, fg_z_max.shape, fg_z_vals.shape, bg_z_vals.shape)
        ray_d_norm = torch.norm(ray_d, dim=-1, keepdim=True)
        viewdirs = ray_d / (ray_d_norm+TINY_NUMBER)   # [..., 3]
        
#         ray_d = ray_d.unsqueeze(-2).expand([N_rays,N_samples, 3])
        viewdirs = viewdirs.unsqueeze(-2).expand( [N_rays,N_samples, 3])
        
        fg_mean, fg_cov = cast_rays(fg_z_vals, ray_o, ray_d, radii, diag=True)
        samples_enc_fg = integrated_pos_enc(
          (fg_mean, fg_cov),
          self.min_deg_point,
          self.max_deg_point,
          )

        checknan(fg_mean)
        checknan(fg_cov)
        checknan(samples_enc_fg)
        
        direction_enc = self.embedder_viewdir(viewdirs)
#         print('direction_enc', direction_enc.shape)
        fg_raw_rgb, fg_raw_density = self.fg_net(samples_enc_fg, direction_enc )

        checknan(direction_enc)
        checknan(fg_raw_rgb)
        checknan(fg_raw_density)
    
        # import ipdb; ipdb.set_trace()

        
        # Add noise to regularize the density predictions if needed.
#           if randomized and (self.density_noise > 0):
#             key, rng = random.split(rng)
#             raw_density += self.density_noise * random.normal(
#                 key, raw_density.shape, dtype=raw_density.dtype)

        # Volumetric rendering
        fg_rgb, fg_density = self.process_raw(fg_raw_rgb, fg_raw_density)
        
             
        checknan(fg_rgb)
        checknan(fg_density)
        
        
        fg_dists = fg_z_vals[..., 1:] - fg_z_vals[..., :-1]
        # account for view directions
        fg_dists = ray_d_norm * torch.cat((fg_dists[..., :-1],
                                           fg_z_max.unsqueeze(-1) - fg_z_vals[..., -1:]),
                                          dim=-1)  # [..., N_samples]
        
        fg_rgb_map, fg_depth_map, T = self.volume_rendering(
          fg_density, fg_rgb,  fg_dists, fg_z_vals
        )


        checknan(fg_dists)
        checknan(fg_rgb_map)
        checknan(fg_depth_map)

        print('-----bg-----')
        
#         ret.append((comp_rgb, distance, acc))
        bg_lambda = T[..., -1]
        bg_z_vals_1_0 = torch.flip(bg_z_vals, dims=[-1, ]) 
        bg_z_vals_real = bgzval2pts_outside(ray_o, ray_d, bg_z_vals_1_0)       

        
        checknan(bg_lambda)
        checknan(bg_z_vals_real)

#         print('bg_z_vals_real',bg_z_vals_real.shape)
        
        bg_mean, bg_cov_full = cast_rays(bg_z_vals_real, ray_o, 
                                         ray_d, radii, diag=False)
        bg_meanP, bg_cov_fullP = contract_gaussian(bg_mean, bg_cov_full)

        checknan(bg_mean)
        checknan(bg_cov_full)
        checknan(bg_meanP)
        checknan(bg_cov_fullP)
        
#         print('bg_meanP',bg_meanP.shape)
#         print('bg_cov_fullP',bg_cov_fullP.shape)
        
        samples_enc_bg = integrated_pos_enc(
          (bg_meanP, bg_cov_fullP),
          self.min_deg_point,
          self.max_deg_point,
            diag=False
          )
        print('--bgnet inference--')
        print(samples_enc_bg[396, 13, :])
        bg_raw_rgb, bg_raw_density = self.bg_net(samples_enc_bg, direction_enc )
        get_nan_index(bg_raw_rgb)
        get_nan_index(bg_raw_density)
        bg_rgb, bg_density = self.process_raw(bg_raw_rgb, bg_raw_density)
        checknan(samples_enc_bg)


        
        
   
        
        # import ipdb; ipdb.set_trace()
        bg_dists = bg_z_vals_1_0[..., :-1] - bg_z_vals_1_0[..., 1:]
        bg_dists = torch.cat((bg_dists[..., :-1], HUGE_NUMBER *
                              torch.ones_like(bg_dists[..., 0:1])), dim=-1)#[..., N_samples]
        
        bg_rgb_map, bg_depth_map, _ = self.volume_rendering(
          bg_density, bg_rgb,  bg_dists, bg_z_vals_1_0
        )

        bg_depth_map = bgzval2pts_outside(ray_o, ray_d, bg_depth_map) 
        
        checknan(bg_dists)
        checknan(bg_rgb_map)
        checknan(bg_depth_map)

         # composite foreground and background
        bg_rgb_map = bg_lambda.unsqueeze(-1) * bg_rgb_map
        rgb_map = fg_rgb_map +  bg_rgb_map
        
        bg_depth_map = bg_lambda * bg_depth_map
        depth_map = fg_depth_map + bg_depth_map
        
        
        

        return {'fg_depth': fg_depth_map*30.,
               'bg_depth': bg_depth_map*30.,
                'depth': depth_map*30.,
               'fg_rgb': fg_rgb_map,
               'bg_rgb': bg_rgb_map,
                'rgb': rgb_map,
               'bg_lambda': bg_lambda}


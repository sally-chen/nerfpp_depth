import torch
import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
from collections import OrderedDict

import logging
logger = logging.getLogger(__package__)


class Embedder(nn.Module):
    def __init__(self, input_dim, max_freq_log2, N_freqs,
                       log_sampling=True, include_input=True,
                       periodic_fns=(torch.sin, torch.cos)):
        '''
        :param input_dim: dimension of input to be embedded
        :param max_freq_log2: log2 of max freq; min freq is 1 by default
        :param N_freqs: number of frequency bands
        :param log_sampling: if True, frequency bands are linerly sampled in log-space
        :param include_input: if True, raw input is included in the embedding
        :param periodic_fns: periodic functions used to embed input
        '''
        super().__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.out_dim = 0
        if self.include_input:
            self.out_dim += self.input_dim

        self.out_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2. ** torch.linspace(0., max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(2. ** 0., 2. ** max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self, input):
        '''
        :param input: tensor of shape [..., self.input_dim]
        :return: tensor of shape [..., self.out_dim]
        '''
        assert (input.shape[-1] == self.input_dim)

        out = []
        if self.include_input:
            out.append(input)

        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq))
        out = torch.cat(out, dim=-1)

        assert (out.shape[-1] == self.out_dim)
        return out

# default tensorflow initialization of linear layers
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)
            
class MLPNetMip(nn.Module):
    def __init__(self, num_samples, feature_dim, direction_dim):
        super().__init__()
        """A simple MLP."""
        self.net_depth  = 8  # The depth of the first part of MLP.
        self.net_width = 256  # The width of the first part of MLP.
        self.net_depth_condition = 1  # The depth of the second part of MLP.
        self.net_width_condition = 128  # The width of the second part of MLP.
        self.skip_layer = 4  # Add a skip connection to the output of every N layers.
        self.num_rgb_channels = 3  # The number of RGB channels.
        self.num_density_channels = 1  # The number of density channels.
        self.feature_dim = feature_dim
        self.num_samples = num_samples
        self.direction_dim = direction_dim
        self.density_dim = 1
        self.rgb_dim = 3
        
        # above from mip
        
        self.base_layers = []
        dim = self.feature_dim
        for i in range(self.net_depth):
            self.base_layers.append(
                nn.Sequential(nn.Linear(dim, self.net_width), nn.ReLU())
            )
            dim = self.net_width
            if i == self.skip_layer  and i != (self.net_depth - 1):  # skip connection after i^th layer
                dim += self.feature_dim
                
        self.base_layers = nn.ModuleList(self.base_layers)
        # self.base_layers.apply(weights_init)        # xavier init
 
        remap_density_layers = [nn.Linear(dim, self.density_dim), ]
        self.remap_density_layers = nn.Sequential(*remap_density_layers)
        # after this get raw density
        
        
        ## add direction to get rgb
        self.bottleneck = nn.Linear(dim,self.net_width)
        
        dim_combined = self.net_width + self.direction_dim
        
        self.dir_layers = []
        for i in range(self.net_depth_condition):
            self.dir_layers.append(
                nn.Sequential(nn.Linear(dim_combined, 
                                        self.net_width_condition), nn.ReLU())
            )
            dim_condition = self.net_width_condition
            
        self.dir_layers = nn.ModuleList(self.dir_layers)
        # self.dir_layers.apply(weights_init)        # xavier init
        
        remap_rgb_layers = [nn.Linear(dim_condition, self.rgb_dim), ]
        self.remap_rgb_layers = nn.Sequential(*remap_rgb_layers) 
        # after this get raw density
  
    def forward(self, x, d):
        """Evaluate the MLP.

        Args:
          x: jnp.ndarray(float32), [batch, num_samples * feature], points.
          d: jnp.ndarray(float32), [batch, num_samples * feature], if not None, this
            variable will be part of the input to the second part of the MLP
            concatenated with the output vector of the first part of the MLP. If
            None, only the first part of the MLP will be used with input x. In the
            original paper, this variable is the view direction.

        Returns:
          raw_rgb: jnp.ndarray(float32), with a shape of
               [batch, num_samples, num_rgb_channels].
          raw_density: jnp.ndarray(float32), with a shape of
               [batch, num_samples, num_density_channels].
        """

        x = x.reshape((-1, self.feature_dim ))
        d = d.reshape((-1, self.direction_dim ))
        
#         print('in mlp')
#         print('x', x.shape)
#         print('d', d.shape)

        base = self.base_layers[0](x)
        for i in range(len(self.base_layers) - 1):
            if i == self.skip_layer:
                base = torch.cat((base,x), dim=-1)
            base = self.base_layers[i + 1](base)

        raw_density = self.remap_density_layers(base)

        base_dir = torch.cat((self.bottleneck(base), d), dim=-1)
        for i in range(len(self.dir_layers) ):
            base_dir = self.dir_layers[i](base_dir)

        raw_rgb = self.remap_rgb_layers(base_dir)

        return raw_rgb.reshape((-1, self.num_samples, self.rgb_dim )),\
            raw_density.reshape((-1, self.num_samples, self.density_dim ))
    

class MLPNetClassier(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_viewdirs=3, pos_ch=128*3, out_dim=128,
                 skips=[4], use_viewdirs=False):

        '''
               :param D: network depth
               :param W: network width
               :param input_ch: input channels for encodings of (x, y, z)
               :param input_ch_viewdirs: input channels for encodings of view directions
               :param skips: skip connection in network
               :param use_viewdirs: if True, will use the view directions as input
               '''
        super().__init__()

        self.input_ch = input_ch +input_ch_viewdirs
        self.input_ch_viewdirs = input_ch_viewdirs


        self.pos_ch = pos_ch
        self.out_dim = out_dim

        self.use_viewdirs = use_viewdirs
        self.skips = skips

        self.base_layers = []
        dim = self.input_ch  + self.pos_ch
        for i in range(D):
            self.base_layers.append(
                nn.Sequential(nn.Linear(dim, W), nn.ReLU())
            )
            dim = W
            if i in self.skips and i != (D - 1):  # skip connection after i^th layer
                dim += self.input_ch

        self.base_layers = nn.ModuleList(self.base_layers)
        # self.base_layers.apply(weights_init)        # xavier init

        base_remap_layers = [nn.Linear(dim, self.out_dim), ]
        self.base_remap_layers = nn.Sequential(*base_remap_layers)
        # self.base_remap_layers.apply(weights_init)

    def forward(self, ray_o, ray_d, positions):
        '''
        :param ray_o: [..., 3]
        :param ray_d: [..., 3]
        :param positions: [..., 128 *  3]
        :return [..., 128] occupancy likelihood
        '''
        input_pts = torch.cat((ray_o, ray_d, positions), dim=-1)
        input_pts_skip = torch.cat((ray_o, ray_d), dim=-1)

        base = self.base_layers[0](input_pts)
        for i in range(len(self.base_layers) - 1):
            if i in self.skips:
                base = torch.cat((input_pts_skip, base), dim=-1)
            base = self.base_layers[i + 1](base)

        occ_likeli = self.base_remap_layers(base)

        return occ_likeli


class MLPNet(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_viewdirs=3,
                 skips=[4], use_viewdirs=False):
        '''
        :param D: network depth
        :param W: network width
        :param input_ch: input channels for encodings of (x, y, z)
        :param input_ch_viewdirs: input channels for encodings of view directions
        :param skips: skip connection in network
        :param use_viewdirs: if True, will use the view directions as input
        '''
        super().__init__()

        self.input_ch = input_ch
        self.input_ch_viewdirs = input_ch_viewdirs
        self.use_viewdirs = use_viewdirs
        self.skips = skips

        self.base_layers = []
        dim = self.input_ch
        for i in range(D):
            self.base_layers.append(
                nn.Sequential(nn.Linear(dim, W), nn.ReLU())
            )
            dim = W
            if i in self.skips and i != (D-1):      # skip connection after i^th layer
                dim += input_ch

        self.base_layers = nn.ModuleList(self.base_layers)
        # self.base_layers.apply(weights_init)        # xavier init

        sigma_layers = [nn.Linear(dim, 1), ]       # sigma must be positive
        self.sigma_layers = nn.Sequential(*sigma_layers)
        # self.sigma_layers.apply(weights_init)      # xavier init

        # rgb color
        rgb_layers = []
        base_remap_layers = [nn.Linear(dim, 256), ]
        self.base_remap_layers = nn.Sequential(*base_remap_layers)
        # self.base_remap_layers.apply(weights_init)

        dim = 256 + self.input_ch_viewdirs
        for i in range(1):
            rgb_layers.append(nn.Linear(dim, W // 2))
            rgb_layers.append(nn.ReLU())
            dim = W // 2
        rgb_layers.append(nn.Linear(dim, 3))
        rgb_layers.append(nn.Sigmoid())     # rgb values are normalized to [0, 1]
        self.rgb_layers = nn.Sequential(*rgb_layers)
        # self.rgb_layers.apply(weights_init)

    def forward(self, input):
        '''
        :param input: [..., input_ch+input_ch_viewdirs]
        :return [..., 4]
        '''
        input_pts = input[..., :self.input_ch]

        base = self.base_layers[0](input_pts)
        for i in range(len(self.base_layers)-1):
            if i in self.skips:
                base = torch.cat((input_pts, base), dim=-1)
            base = self.base_layers[i+1](base)

        sigma = self.sigma_layers(base)
        sigma = torch.abs(sigma)

        base_remap = self.base_remap_layers(base)
        input_viewdirs = input[..., -self.input_ch_viewdirs:]
        rgb = self.rgb_layers(torch.cat((base_remap, input_viewdirs), dim=-1))

        ret = OrderedDict([('rgb', rgb),
                           ('sigma', sigma.squeeze(-1))])
        return ret


class WrapperModule(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module.forward(*args, **kwargs)

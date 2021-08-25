import torch
# import torch.nn as nn
# import torch.nn.functional as F
import numpy as np
from icecream import ic


HUGE_NUMBER = 1e10
TINY_NUMBER = float(1e-6)      # float32 only has 7 decimal digits precision


# misc utils
def entropy_loss(p):


    sm = torch.nn.Softmax(dim=-1)
    lsm = torch.nn.LogSoftmax(dim=-1)

    sm_res = sm(p)
    lsm_res = lsm(p)

    # if p.shape[0] >1024:
    #     p_ = np.reshape(p.detach().cpu().numpy(),(360,640, 128))
    #     sm_res_ = np.reshape(sm_res.detach().cpu().numpy(),(360,640, 128))
    #     lsm_res_ = np.reshape(lsm_res.detach().cpu().numpy(),(360,640, 128))


    # print('[!!]P: {} LOG: {}'.format(sm_res, lsm_res))

    return torch.mean(torch.sum(-1.0 * sm_res * lsm_res, dim=-1))

    ### gives nans
    # p_plus = torch.sigmoid(p)
    # p_normalize = p_plus / torch.sum(p_plus, dim=1, keepdim=True)
    # log_term = torch.clamp(torch.log2(p_normalize), min=-100., max=0.)
    # return torch.mean(torch.sum(-1.0 * p_normalize * log_term, dim=1))
# def entropy_loss(p):
#     ic(p)
#     p_normalize = torch.tanh(p)+1.
#
#     ic(p_normalize)
#
#     log_term = torch.clamp(torch.log2(p_normalize), min=-100., max=0.)
#     return torch.mean(torch.sum(-1.0 * p_normalize * log_term, dim=-1))

def crossEntropy(label, pred):

    sm = torch.nn.Softmax(dim=-1)
    lsm = torch.nn.LogSoftmax(dim=-1)

    sm_res = sm(label)
    lsm_res = lsm(pred)

    # print('[!!]P: {} LOG: {}'.format(sm_res, lsm_res))

    return torch.mean(torch.sum(-1.0 * sm_res * lsm_res, dim=-1))


def img2mse(x, y, mask=None):
    if mask is None:
        return torch.mean((x - y) * (x - y))
    else:
        return torch.sum((x - y) * (x - y) * mask.unsqueeze(-1)) / (torch.sum(mask) * x.shape[-1] + TINY_NUMBER)

def dep_l1l2loss(x,y,l1l2='l1'):
    if l1l2 == 'l1':
        return torch.mean(torch.abs(x-y))
    else:
        return torch.mean((x - y) * (x - y))

img_HWC2CHW = lambda x: x.permute(2, 0, 1)
gray2rgb = lambda x: x.unsqueeze(2).repeat(1, 1, 3)


def normalize(x):
    min = x.min()
    max = x.max()

    return (x - min) / ((max - min) + TINY_NUMBER)

def normalize_torch(x):
    #min,_ = torch.min(x,dim=-1,keepdim=True)
    #max,_ = torch.max(x,dim=-1,keepdim=True)

    #return (x - min) / ((max - min) + TINY_NUMBER)
    x_minus = x - x.min(dim=-1, keepdim=True)[0]
    x_norm = x_minus / (x_minus.max(dim=-1, keepdim=True)[0] + TINY_NUMBER)
    return x_norm


to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
# gray2rgb = lambda x: np.tile(x[:,:,np.newaxis], (1, 1, 3))
mse2psnr = lambda x: -10. * np.log(x+TINY_NUMBER) / np.log(10.)


########################################################################################################################
#
########################################################################################################################
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib as mpl
from matplotlib import cm
import cv2


def get_vertical_colorbar(h, vmin, vmax, cmap_name='jet', label=None):
    fig = Figure(figsize=(1.2, 8), dpi=100)
    # fig = Figure(figsize=(2.2, 8), dpi=100)
    fig.subplots_adjust(right=1.5)
    canvas = FigureCanvasAgg(fig)

    # Do some plotting.
    ax = fig.add_subplot(111)
    cmap = cm.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    tick_cnt = 6
    tick_loc = np.linspace(vmin, vmax, tick_cnt)
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm,
                                    ticks=tick_loc,
                                    orientation='vertical')

    tick_label = ['{:6.2f}'.format(x) for x in tick_loc]
    cb1.set_ticklabels(tick_label)

    cb1.ax.tick_params(labelsize=10, rotation=0)

    if label is not None:
        cb1.set_label(label)

    fig.tight_layout()

    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()

    im = np.frombuffer(s, np.uint8).reshape((height, width, 4))

    im = im[:, :, :3].astype(np.float32) / 255.
    if h != im.shape[0]:
        w = int(im.shape[1] / im.shape[0] * h)
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)

    return im


def colorize_np(x, cmap_name='jet', mask=None, append_cbar=False):
    if mask is not None:
        # vmin, vmax = np.percentile(x[mask], (1, 99))
        vmin = np.min(x[mask])
        vmax = np.max(x[mask])
        vmin = vmin - np.abs(vmin) * 0.01
        x[np.logical_not(mask)] = vmin
        x = np.clip(x, vmin, vmax)
        # print(vmin, vmax)
    else:
        vmin = x.min()
        vmax = x.max() + TINY_NUMBER

    x = (x - vmin) / (vmax - vmin)
    # x = np.clip(x, 0., 1.)

    cmap = cm.get_cmap(cmap_name)
    x_new = cmap(x)[:, :, :3]

    if mask is not None:
        mask = np.float32(mask[:, :, np.newaxis])
        x_new = x_new * mask + np.zeros_like(x_new) * (1. - mask)

    # so this is indeed min and max of the depth map
    cbar = get_vertical_colorbar(h=x.shape[0], vmin=vmin, vmax=vmax, cmap_name=cmap_name)

    if append_cbar:
        x_new = np.concatenate((x_new, np.zeros_like(x_new[:, :5, :]), cbar), axis=1)
        return x_new
    else:
        return x_new, cbar


# tensor
def colorize(x, cmap_name='jet', append_cbar=False, mask=None,is_np=False):
    if not is_np:
        x = x.cpu().numpy()
    if mask is not None:
        mask = mask.numpy().astype(dtype=np.bool)
    x, cbar = colorize_np(x, cmap_name, mask)

    if append_cbar:
        x = np.concatenate((x, np.zeros_like(x[:, :5, :]), cbar), axis=1)

    x = torch.from_numpy(x)
    return x

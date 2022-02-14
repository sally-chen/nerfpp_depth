import os.path

import imageio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import PIL.Image
from torchvision.transforms import ToTensor
import io
import torch
from scipy.spatial.transform import Rotation as R

from .utils import normalize_torch
from pytorch3d.transforms import euler_angles_to_matrix

TINY_NUMBER = float(1e-6)

def plot_mult_pose(poses_list, name, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #
    colors = ['b', 'r', 'c', 'm', 'y', 'b', 'r', 'c', 'm']
    for i in range(len(poses_list)):
        plot_single_pose(poses_list[i],colors[i], ax, labels[i])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.legend()
    plt.title(name)
    plt.show()
    # plt.savefig("./{}.png".format(name))

def plot_single_pose(poses, color, ax, label):
    # poses shape N, 3, 4
    ax.scatter(poses[:, 0, 3], poses[:, 1, 3], poses[:, 2, 3], marker='o', color='red', label=label)
    #
    # for i in range(poses.shape[0]):
    #     ax.plot([poses[i, 0, 3], poses[i, 0, 3] + poses[i, 0, 2]],
    #             [poses[i, 1, 3], poses[i, 1, 3] + poses[i, 1, 2]],
    #             [poses[i, 2, 3], poses[i, 2, 3] + poses[i, 2, 2]], color='blue')


def plot_ray_batch(batch):

    rays_o = batch['ray_o'].cpu().detach().numpy()
    rays_d = batch['ray_d'].cpu().detach().numpy()
    plot_size = 500

    rays_o_plot = np.reshape(rays_o, [-1, rays_o.shape[-1]])
    rays_d_plot = np.reshape(rays_d, [-1, rays_d.shape[-1]])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(plot_size):
        ax.plot([rays_o_plot[i, 0], rays_o_plot[i, 0] + rays_d_plot[i, 0]],
                [rays_o_plot[i, 1], rays_o_plot[i, 1] + rays_d_plot[i, 1]],
                [rays_o_plot[i, 2], rays_o_plot[i, 2] + rays_d_plot[i, 2]], color='blue')

    ax.scatter(rays_o_plot[:plot_size, 0], rays_o_plot[:plot_size, 1], rays_o_plot[:plot_size, 2], color='red',
               label='ray_o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    # ax.set_xlim3d(-10, 10)
    # ax.set_ylim3d(-10, 10)
    # # ti = r'ndc - before moving to near plane \n u[%0.1f,%0.1f,%0.1f]\n ' \
    # #      r'v[%0.1f,%0.1f,%0.1f]\n' \
    # #      r' w[%0.1f,%0.1f,%0.1f]\n' \
    # #      r' t[%0.1f,%0.1f%0.1f]  '%(c2w[0,0],c2w[1,0],c2w[2,0],
    # #                                 c2w[0,1],c2w[1,1],c2w[2,1],
    # #                                 c2w[0,2],c2w[1,2],c2w[2,2],
    # #                                 c2w[0,3],c2w[1,3],c2w[2,3])

    plt.legend()
    name = r'sampled_poses'
    plt.title(name)
    plt.show()


def calculate_metrics(pred, target, avg_method='micro', threshold=0.5, zero_division=0):

    pred = np.array(pred > threshold, dtype=float)
    target = np.array(target > threshold, dtype=float)


    return {'precision': precision_score(y_true=target, y_pred=pred, average=avg_method,zero_division=0),
            'recall': recall_score(y_true=target, y_pred=pred, average=avg_method,zero_division=0),
            'f1': f1_score(y_true=target, y_pred=pred, average=avg_method,zero_division=0)}
            # 'cm':confusion_matrix(target, pred)}

def log_plot_conf_mat(writer, cm, step, name):
    labels = [str(i) for i in range(cm.shape[0])]
    print(cm)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)

    image = PIL.Image.open(buf)
    image = ToTensor()(image).unsqueeze(0)

    writer.add_image(name, image, step)

def write_depth_vis(label, pred, out_dir, frname):
    fig = plt.figure(figsize=(15, 8))

    # pos = [str(i) for i in range(pred.shape[1])]

    if label is not None:
        ax_label = fig.add_subplot(121)
        pcm_label = ax_label.matshow(label)
        ax_label.set_title("Depth Label")
        # ax_label.set_xticklabels([''] + pos)

        ax_pred = fig.add_subplot(122)

    else:
        ax_pred = fig.add_subplot(111)
    pcm_pred = ax_pred.matshow(pred)
    ax_pred.set_title("Depth Pred")
    # ax_pred.set_xticklabels([''] + pos)

    # plt.title('Prob Distribution of Depth Label vs Prediction')
    # cax=plt.imshow()

    if label is not None:
        plt.colorbar(pcm_label, ax=ax_label)

    plt.colorbar(pcm_pred, ax=ax_pred)
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')

    plt.clf()
    plt.close(fig)
    buf.seek(0)

    image = PIL.Image.open(buf)
    imageio.imwrite(os.path.join(out_dir, frname), image)




def visualize_depth_label(writer, label, pred, step, name):
    fig = plt.figure(figsize=(15,8))

    # pos = [str(i) for i in range(pred.shape[1])]


    if label is not None:
        ax_label = fig.add_subplot(121)
        pcm_label = ax_label.matshow(label)
        ax_label.set_title("Depth Label")
        # ax_label.set_xticklabels([''] + pos)

        ax_pred = fig.add_subplot(122)

    else:
        ax_pred = fig.add_subplot(111)
    pcm_pred = ax_pred.matshow(pred)
    ax_pred.set_title("Depth Pred")
    # ax_pred.set_xticklabels([''] + pos)



    # plt.title('Prob Distribution of Depth Label vs Prediction')
    # cax=plt.imshow()

    if label is not None:
        plt.colorbar(pcm_label, ax=ax_label)

    plt.colorbar(pcm_pred, ax=ax_pred)


    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')

    plt.clf()
    plt.close(fig)
    buf.seek(0)

    image = PIL.Image.open(buf)
    image = ToTensor()(image)

    writer.add_image(name, image, step)

def loss_deviation(writer, label, pred, step, name):
    fig = plt.figure(figsize=(15, 8))

    label[label==0] = 0.00001
    label[label==1] = 0.99999

    pred[pred==0] = 0.00001
    pred[pred==1] = 0.99999

    yn = label
    xn = label
    bce_gt = -yn*np.log(xn) - (1-yn )* np.log(1-xn)

    yn = label
    xn = pred
    bce_true = -yn * np.log(xn) - (1 - yn) * np.log(1 - xn)
    bce_dev =  bce_true- bce_gt



    pos = [str(i) for i in range(label.shape[1])]

    ax_label = fig.add_subplot(121)
    pcm_label = ax_label.matshow(bce_gt)
    ax_label.set_title("Min loss")
    # ax_label.set_xticklabels([''] + pos)

    ax_pred = fig.add_subplot(122)
    pcm_pred = ax_pred.matshow(bce_dev)
    ax_pred.set_title("Deviation from Min ")
    # ax_pred.set_xticklabels([''] + pos)

    # plt.title('Prob Distribution of Depth Label vs Prediction')
    # cax=plt.imshow()
    plt.colorbar(pcm_label, ax=ax_label)

    plt.colorbar(pcm_pred, ax=ax_pred)

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')

    plt.clf()
    plt.close(fig)
    buf.seek(0)

    image = PIL.Image.open(buf)
    image = ToTensor()(image)

    writer.add_image(name, image, step)

def triangle_filter( occupancy, Z=5):

    '''
    occupancy: [N_rays, N_samples  ] value: 000000NNNN00000000

    '''

    tri_filter = []

    Z_half_floor = np.floor(Z / 2)
    for i in range(int(-Z_half_floor), int(Z_half_floor) + 1):
        tri_filter.append((Z_half_floor + 1 - np.absolute(i)) / (Z_half_floor + 1))
    tri_filter = torch.Tensor(tri_filter).unsqueeze(0).unsqueeze(0).type_as(occupancy)  ## need to be [1, 1, Z]
    # test15 = cls_label_flat.detach().cpu().numpy()

    occupancy_filtered = torch.squeeze(
        torch.nn.functional.conv1d(occupancy.unsqueeze(-2), tri_filter, padding=int(Z_half_floor)))

    occupancy_filtered[occupancy_filtered > 1.] = 1.
    occupancy_filtered[occupancy_filtered < 0.] = 0.

    return occupancy_filtered

def get_box_transmittance_weight(box_loc, fg_z_vals, ray_d, ray_o, fg_depth,box_number=10, box_props=None):


    # pts: N x 128 x 3
    # assume axis aligned box

    multiplier = 75.
    box_loc = box_loc.clone()

    assert box_loc.shape == (ray_o.shape[0], box_number, 3)

    N_rays = list(ray_d.shape[:1])
    N_samples = fg_z_vals.shape[-1]

    box_sizes, box_rot = box_props[:,3:6], box_props[:,6:]

    assert box_sizes.shape == (box_number, 3), 'box_sizes shape is wrong'
    assert box_rot.shape == (box_number, 3), 'box_rot shape is wrong'

    r = euler_angles_to_matrix(torch.deg2rad(torch.cat([box_rot[:,2:],-1*box_rot[:,1:2], -1*box_rot[:,0:1] ], dim=-1)), convention='ZYX')
    r_mat = r.unsqueeze(0).unsqueeze(1).expand(N_rays + [N_samples, box_number, 3,3]).float()

    fg_ray_o = ray_o.unsqueeze(-2).expand(N_rays + [N_samples, 3])
    fg_ray_d = ray_d.unsqueeze(-2).expand(N_rays + [N_samples, 3])
    fg_pts = fg_ray_o + fg_z_vals.unsqueeze(-1) * fg_ray_d

    fg_pts = fg_pts.unsqueeze(-2).expand(N_rays + [N_samples, box_number, 3])


    #box_size = (box_sizes/26.).type_as(box_loc).unsqueeze(0).expand(N_rays + [box_number, 3])#bf moving normalization out

    box_size = (box_sizes*30./26.).type_as(box_loc).unsqueeze(0).expand(N_rays + [box_number, 3])
    assert box_size.shape == (N_rays[0], box_number, 3)

    offset = (fg_pts - box_loc.unsqueeze(1).expand(-1,N_samples,-1,-1)).unsqueeze(-1).float()
    offset_rot = torch.abs(torch.matmul(torch.inverse(r_mat), offset)).squeeze(-1)
    abs_dist  = offset_rot / (box_size.unsqueeze(1).expand(-1,N_samples,-1,-1) +TINY_NUMBER) #box_offset.reshape(dots_sh[0], self.box_number, N_samples, 3))
    inside_box = 0.5  - abs_dist
    weights = torch.prod(torch.sigmoid(inside_box * 20.), dim=-1) # N_rays + [N_samples, box_number]


    box_occupancy = (torch.sigmoid(torch.sum(weights, dim=-1)*20.) - 0.5 ) * 2

    # in_boxes = weights > 0.95  # torch.sum(in_boxes_compare, dim=-1) == 3  # N, N_samples, N_b,
    #
    # # test = in_boxes.cpu().numpy()
    # in_any_box = torch.sum(in_boxes, dim=-1) > 0.0  # N, N_samples,

    # box_occupancy = torch.zeros(fg_z_vals.shape).type_as(box_loc)  # N, 127
    # # box_occupancy = torch.where(torch.sum(torch.gt(fg_pts, mins) & torch.lt(fg_pts, maxs), dim=-1)==3, torch.tensor(1.).type_as(box_loc), box_occupancy)
    # box_occupancy = torch.where(in_any_box, torch.tensor(1.).type_as(box_loc), box_occupancy)

    assert box_occupancy.shape == (N_rays[0], N_samples)

    box_occupancy_filtered = triangle_filter(box_occupancy, Z=3) * multiplier

    assert box_occupancy_filtered.shape == (N_rays[0], N_samples)


    # alpha blending
    ray_d_norm = torch.norm(ray_d, dim=-1, keepdim=True)  # [..., 1]

    fg_dists = fg_z_vals[..., 1:] - fg_z_vals[..., :-1]
    # account for view directions
    fg_dists = ray_d_norm * torch.cat((fg_dists, fg_depth.unsqueeze(-1) - fg_z_vals[..., -1:]),
                                      dim=-1)  # [..., N_samples]
    fg_alpha = 1. - torch.exp(-box_occupancy_filtered * fg_dists)  # [..., N_samples]
    T = torch.cumprod(1. - fg_alpha + TINY_NUMBER, dim=-1)  # [..., N_samples]
    T = torch.cat((torch.ones_like(T[..., 0:1]), T[..., :-1]), dim=-1)  # [..., N_samples]
    fg_weights = fg_alpha * T  # [..., N_samples]

    fg_weights_normed = normalize_torch(fg_weights)
    # if torch.sum(torch.sum(box_occupancy_filtered, dim=-1)) > 0:
    #     print("STOP HERE")
    #     test_box_occ = box_occupancy.cpu().numpy()
    #     test_box_occ_fil = box_occupancy_filtered.cpu().numpy()
    #     test_fg_alpha = fg_alpha.cpu().numpy()
    #     test_T = T.cpu().numpy()
    #     test_fg_weights = fg_weights_normed.cpu().numpy()
    return fg_weights_normed * 3.

def check_shadow_aabb_inters(fg_pts, box_loc, box_sizes, box_rot, box_number):
    # box_rot in matrix form already, box_size original
    from .utils import HUGE_NUMBER

    input_shpe = list(fg_pts.shape[:2])
    fg_pts = fg_pts.clone().reshape(-1, 3)
    box_loc = box_loc.clone().reshape(-1, box_number, 3)
    N_rays = fg_pts.shape[0]
    sun_location = torch.Tensor([0., 0., 99.999]).type_as(fg_pts).unsqueeze(0).expand(N_rays, -1)

    assert box_loc.shape == (N_rays, box_number, 3)
    assert box_sizes.shape == (box_number, 3), 'box_sizes shape is wrong'
    assert box_rot.shape == (box_number, 3,3), 'box_rot shape is wrong'

    # convert ray origin and ray end to box coordinate
    sun_location = sun_location.unsqueeze(-2).expand([N_rays, box_number, 3])  - box_loc
    fg_pts = fg_pts.unsqueeze(-2).expand([N_rays, box_number, 3]) - box_loc

    # rotate to axis alighed box
    sun_location_rot = torch.matmul(torch.inverse(box_rot), sun_location.unsqueeze(-1)).squeeze(-1)
    fg_pts_rot = torch.matmul(torch.inverse(box_rot), fg_pts.unsqueeze(-1)).squeeze(-1)

    # rayd
    ray_d_norm_rot = (fg_pts_rot - sun_location_rot) / (torch.norm(fg_pts_rot - sun_location_rot, dim =-1, keepdim=True)+TINY_NUMBER) # normalize
    ray_d_frac = torch.zeros_like(ray_d_norm_rot)
    ray_d_frac[ray_d_norm_rot == 0.] = HUGE_NUMBER
    ray_d_frac[ray_d_norm_rot != 0.] = torch.div(1., ray_d_norm_rot[ray_d_norm_rot != 0.])

    # get AABB bounds
    box_size = (box_sizes*30. / 28.).type_as(box_loc).unsqueeze(0).expand(N_rays, box_number, 3)
    box_mins = torch.zeros_like(box_size) - box_size / 2  # N, N_b, 3
    box_maxs = torch.zeros_like(box_size) + box_size / 2  # N, N_b, 3

    box_mins_maxs = torch.stack([box_mins, box_maxs], dim=-1)

    ts = (box_mins_maxs - sun_location_rot.unsqueeze(-1)) * ray_d_frac.unsqueeze(-1)
    t_mins = torch.min(ts, dim=-1)[0]
    t_maxs = torch.max(ts, dim=-1)[0]

    t_maxof_mins = torch.max(t_mins, dim=-1)[0]
    t_minof_maxes = torch.min(t_maxs, dim=-1)[0]


    # there is no intersection if tmins > tmaxs for all axis, or tmax < 0 (box behind ray) for each box
    # use sigmoid to make it differentiable
    not_behind_cam = torch.sigmoid(t_minof_maxes*50.)
    inters_box_each = torch.sigmoid((t_minof_maxes-t_maxof_mins)*50.) * not_behind_cam
    rad_show = -1 * torch.sigmoid(torch.sum(inters_box_each, dim=-1)*50.) + 1.5

    # inters_box_each = torch.ones([N_rays, box_number]).type_as(box_loc)  # N, 127
    # inters_box_each = torch.where(torch.logical_or(torch.gt(t_maxof_mins, t_minof_maxes), torch.lt(t_minof_maxes, 0.)) ,
    #                                 torch.tensor(0.0).type_as(box_loc), inters_box_each)
    # inters_box = torch.sum(inters_box_each, dim=-1) > 0.98
    # rad_show = torch.ones([N_rays]).type_as(box_loc)
    # rad_show[inters_box] = 0.5


    return rad_show.reshape(input_shpe).unsqueeze(-1)  # [N_rays, n_samples,1]


def check_shadow(fg_pts, box_loc, box_sizes, box_rot, box_number):

    # box_rot in matrix form already, box_size original

    input_shpe = list(fg_pts.shape[:2])

    fg_pts = fg_pts.clone().reshape(-1,3)
    box_loc = box_loc.clone().reshape(-1,box_number,3)
    N_rays = fg_pts.shape[0]
    N_samples = 40
    multiplier = 10
    sun_location = torch.Tensor([0., 0., 0.999]).type_as(fg_pts).unsqueeze(0).expand(N_rays, -1)

    assert box_loc.shape == (N_rays, box_number, 3)
    assert box_sizes.shape == (box_number, 3), 'box_sizes shape is wrong'
    assert box_rot.shape == (box_number, 3,3), 'box_rot shape is wrong'

    ray_d_norm = (fg_pts - sun_location) / torch.norm(fg_pts - sun_location, dim =-1, keepdim=True) # normalize

    # linear
    # step = torch.norm(fg_pts - sun_location, dim =-1) / (N_samples)  # fg step size  --> will make this constant eventually [H*W]
    # fg_z_vals = torch.stack([i * step for i in range(N_samples+1)],  dim=-1)  # [..., N_samples] distance to camera till unit sphere
    # fg_z_vals_centre = step.unsqueeze(-1) / 2. + fg_z_vals
    # fg_z_vals_centre = fg_z_vals_centre[:, :-1]


    near, far = 0.001*torch.ones([N_rays]),  torch.norm(fg_pts - sun_location, dim =-1)-0.001
    near, far = near.unsqueeze(-1).expand(-1, N_samples+1).type_as(box_loc), \
                far.unsqueeze(-1).expand(-1, N_samples+1).type_as(box_loc)

    # inverse sample
    # t_vals = torch.linspace(0.,1.,N_samples+1).unsqueeze(0).expand(N_rays, N_samples+1).type_as(box_loc)
    # fg_z_vals = 1. / (1. / near * ( 1 - t_vals) + 1 / far * (t_vals))

    t_vals = (torch.logspace(0., 1., N_samples + 1, base=10).unsqueeze(0).expand(N_rays, N_samples + 1).type_as(box_loc) - 1.)/9.
    fg_z_vals = torch.flip(near * (t_vals) + far * (1. - t_vals), dims=[-1])

    # steps_ = (fg_z_vals[:, 1:] - fg_z_vals[:, :1]).cpu().numpy()
    # fg_z_vals_test = fg_z_vals.cpu().numpy()

    fg_z_vals_centre = (fg_z_vals[:, 1:] - fg_z_vals[:, :1])/2 + fg_z_vals[:,:-1]

    sun_ray_pts = sun_location.unsqueeze(-2).expand(N_rays,N_samples,-1) + fg_z_vals_centre.unsqueeze(-1) * ray_d_norm.unsqueeze(-2)  # [H*W, N_samples, 3]

    sun_ray_pts = sun_ray_pts.unsqueeze(-2).expand([N_rays, N_samples, box_number, 3])

    box_size = (box_sizes / 28.).type_as(box_loc).unsqueeze(0).expand(N_rays , box_number, 3)

    offset = (sun_ray_pts - box_loc.unsqueeze(1).expand(-1, N_samples, -1, -1)).unsqueeze(-1).float()

    offset_rot = torch.abs(torch.matmul(torch.inverse(box_rot), offset)).squeeze(-1)
    abs_dist = offset_rot / box_size.unsqueeze(1).expand(-1, N_samples, -1,
                                                         -1)  # box_offset.reshape(dots_sh[0], self.box_number, N_samples, 3))
    inside_box = 0.5 - abs_dist
    weights = torch.prod(torch.sigmoid(inside_box * 1000.), dim=-1)  # N_rays + [N_samples, box_number]
    # print(inside_box[0])

    # in_boxes_compare = torch.gt(fg_pts, mins) & torch.lt(fg_pts, maxs)  # N, N_samples, N_b,
    in_boxes = weights > 0.99  # torch.sum(in_boxes_compare, dim=-1) == 3  # N, N_samples, N_b,

    # test = in_boxes.cpu().numpy()
    in_any_box = torch.sum(in_boxes, dim=-1) > 0.0  # N, N_samples,

    box_occupancy = torch.zeros(fg_z_vals_centre.shape).type_as(box_loc)  # N, 127
    # box_occupancy = torch.where(torch.sum(torch.gt(fg_pts, mins) & torch.lt(fg_pts, maxs), dim=-1)==3, torch.tensor(1.).type_as(box_loc), box_occupancy)
    box_occupancy = torch.where(in_any_box, torch.tensor(1.).type_as(box_loc), box_occupancy)

    assert box_occupancy.shape == (N_rays, N_samples)

    box_occupancy_filtered = box_occupancy * multiplier

    assert box_occupancy_filtered.shape == (N_rays, N_samples)

    # alpha blending

    fg_dists = fg_z_vals[..., 1:] - fg_z_vals[..., :-1]
    fg_alpha = 1. - torch.exp(-box_occupancy_filtered * fg_dists)  # [..., N_samples]
    T_light = torch.cumprod(1. - fg_alpha + TINY_NUMBER, dim=-1)[:,-1].reshape(input_shpe)  # [..., N_samples]


    # if torch.sum(torch.sum(box_occupancy_filtered, dim=-1)) > 0:
    #     print("STOP HERE")
    #     test_box_occ = box_occupancy.cpu().numpy()
    #     test_box_occ_fil = box_occupancy_filtered.cpu().numpy()
    #     test_fg_alpha = fg_alpha.cpu().numpy()
    #     test_T = T.cpu().numpy()
    #     test_fg_weights = fg_weights_normed.cpu().numpy()
    # return T_light.unsqueeze(-1) # [N_rays, n_samples,1]

    # directly use occupancy -- check if the ray intersect anybox
    in_any_box_ray = torch.sum(in_any_box, dim=-1) > 0.0  # N, N_samples,

    box_occupancy_ray = torch.ones([N_rays]).type_as(box_loc)  # N, 127
    # box_occupancy = torch.where(torch.sum(torch.gt(fg_pts, mins) & torch.lt(fg_pts, maxs), dim=-1)==3, torch.tensor(1.).type_as(box_loc), box_occupancy)
    box_occupancy_ray = torch.where(in_any_box_ray, torch.tensor(0.7).type_as(box_loc), box_occupancy_ray)

    return box_occupancy_ray.reshape(input_shpe).unsqueeze(-1) # [N_rays, n_samples,1]


def get_box_weight(box_loc, box_size, fg_z_vals, ray_d, ray_o, box_number=10):
    # pts: N x 128 x 3
    # assume axis aligned box
    box_loc = box_loc.clone()

    assert box_loc.shape == (ray_o.shape[0],box_number,3)

    N_rays = list(ray_d.shape[:1])

    N_samples = fg_z_vals.shape[-1]
    fg_ray_o = ray_o.unsqueeze(-2).expand(N_rays + [N_samples, 3])
    fg_ray_d = ray_d.unsqueeze(-2).expand(N_rays + [N_samples, 3])
    fg_pts = fg_ray_o + fg_z_vals.unsqueeze(-1) * fg_ray_d

    fg_pts = fg_pts.unsqueeze(-2).expand(N_rays + [N_samples, box_number, 3])

    # box_loc[:,2] = -1.#-1.8/60.

    # box_size = torch.Tensor([[1/20.,1/20.,3.]]).to(torch.cuda.current_device())
    box_size = torch.Tensor([[1/26.,1/26.,1/26.]]).type_as(box_loc).unsqueeze(0).expand(N_rays + [box_number,3])

    assert box_size.shape == (N_rays[0], box_number, 3)


    mins = box_loc - box_size / 2  # N, N_b, 3
    maxs = box_loc + box_size / 2  # N, N_b, 3


    mins = mins.unsqueeze(1).expand(N_rays + [N_samples, box_number, 3]) # N, N_sample, N_b, 3
    maxs = maxs.unsqueeze(1).expand(N_rays + [N_samples, box_number, 3])  # N, N_sample, N_b, 3

    # we give it 1 when a point falls in any of the boxes

    in_boxes_compare = torch.gt(fg_pts, mins) & torch.lt(fg_pts, maxs)# N, N_samples, N_b,
    in_boxes = torch.sum(in_boxes_compare, dim=-1) ==3 # N, N_samples, N_b,

    # test = in_boxes.cpu().numpy()
    in_any_box = torch.sum(in_boxes, dim=-1) > 0# N, N_samples,


    box_occupancy = torch.zeros(fg_z_vals.shape).type_as(box_loc) # N, 127
    # box_occupancy = torch.where(torch.sum(torch.gt(fg_pts, mins) & torch.lt(fg_pts, maxs), dim=-1)==3, torch.tensor(1.).type_as(box_loc), box_occupancy)
    box_occupancy = torch.where(in_any_box, torch.tensor(1.).type_as(box_loc), box_occupancy)

    assert box_occupancy.shape == (N_rays[0], N_samples)

    return box_occupancy * 1000.



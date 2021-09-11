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

from utils import normalize_torch

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

def get_box_transmittance_weight(box_loc, box_size, fg_z_vals, ray_d, ray_o, fg_depth,box_number=10):



    sizes = [float(size) for size in box_size.split(',')]
    assert len(sizes) == 3


    # pts: N x 128 x 3
    # assume axis aligned box

    multiplier = 75
    box_loc = box_loc.clone()

    assert box_loc.shape == (ray_o.shape[0], box_number, 3)

    N_rays = list(ray_d.shape[:1])

    N_samples = fg_z_vals.shape[-1]
    fg_ray_o = ray_o.unsqueeze(-2).expand(N_rays + [N_samples, 3])
    fg_ray_d = ray_d.unsqueeze(-2).expand(N_rays + [N_samples, 3])
    fg_pts = fg_ray_o + fg_z_vals.unsqueeze(-1) * fg_ray_d

    fg_pts = fg_pts.unsqueeze(-2).expand(N_rays + [N_samples, box_number, 3])

    # box_loc[:,2] = -1.#-1.8/60.

    # box_size = torch.Tensor([[1/20.,1/20.,3.]]).to(torch.cuda.current_device

    # box_size = torch.Tensor([[1/20.,1/20.,3.]]).to(torch.cuda.current_device())

    box_size = torch.Tensor([[sizes[0] / 26., sizes[1] / 26., sizes[2]/26.]]).type_as(box_loc).unsqueeze(0).expand(
        N_rays + [box_number, 3])

    assert box_size.shape == (N_rays[0], box_number, 3)

    mins = box_loc - box_size / 2  # N, N_b, 3
    maxs = box_loc + box_size / 2  # N, N_b, 3

    mins = mins.unsqueeze(1).expand(N_rays + [N_samples, box_number, 3])  # N, N_sample, N_b, 3
    maxs = maxs.unsqueeze(1).expand(N_rays + [N_samples, box_number, 3])  # N, N_sample, N_b, 3

    # we give it 1 when a point falls in any of the boxes

    in_boxes_compare = torch.gt(fg_pts, mins) & torch.lt(fg_pts, maxs)  # N, N_samples, N_b,
    in_boxes = torch.sum(in_boxes_compare, dim=-1) == 3  # N, N_samples, N_b,

    # test = in_boxes.cpu().numpy()
    in_any_box = torch.sum(in_boxes, dim=-1) > 0.0  # N, N_samples,

    box_occupancy = torch.zeros(fg_z_vals.shape).type_as(box_loc)  # N, 127
    # box_occupancy = torch.where(torch.sum(torch.gt(fg_pts, mins) & torch.lt(fg_pts, maxs), dim=-1)==3, torch.tensor(1.).type_as(box_loc), box_occupancy)
    box_occupancy = torch.where(in_any_box, torch.tensor(1.).type_as(box_loc), box_occupancy)

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
    return fg_weights_normed * 100.


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



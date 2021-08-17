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
    for i in range(poses.shape[0]):
        ax.plot([poses[i, 0, 3], poses[i, 0, 3] + poses[i, 0, 2]],
                [poses[i, 1, 3], poses[i, 1, 3] + poses[i, 1, 2]],
                [poses[i, 2, 3], poses[i, 2, 3] + poses[i, 2, 2]], color='blue')


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


def get_box_weight(box_loc, box_size, fg_z_vals, ray_d, ray_o):
    # pts: N x 128 x 3
    # assume axis aligned box


    dots_sh = list(ray_d.shape[:-1])

    N_samples = fg_z_vals.shape[-1]
    fg_ray_o = ray_o.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
    fg_ray_d = ray_d.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
    fg_pts = fg_ray_o + fg_z_vals.unsqueeze(-1) * fg_ray_d


    box_loc[:,2] = -1.#-1.8/60.

    box_size = torch.Tensor([[1/15.,1/15.,4.]]).to(torch.cuda.current_device())

    mins = box_loc - box_size / 2  # N, 3
    maxs = box_loc + box_size / 2  # N, 3


    mins = mins.unsqueeze(1).expand(dots_sh + [N_samples, 3])
    maxs = maxs.unsqueeze(1).expand(dots_sh + [N_samples, 3])


    box_occupancy = torch.zeros(fg_z_vals.shape).to(torch.cuda.current_device()) # N, 127
    box_occupancy = torch.where(torch.sum(torch.gt(fg_pts.double(), mins) & torch.lt(fg_pts.double(), maxs), dim=-1)==3, 1., box_occupancy.double())

    return box_occupancy.double()



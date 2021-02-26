import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


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
    plt.savefig("./train_poses.png")

def plot_single_pose(poses, color, ax, label):
    # poses shape N, 3, 4
    ax.scatter(poses[:, 0, 3], poses[:, 1, 3], poses[:, 2, 3], marker='o', color=color, label=label)
    #
    # for i in range(poses.shape[0]):
    #     ax.plot([poses[i, 0, 3], poses[i, 0, 3] + poses[i, 0, 2]],
    #             [poses[i, 1, 3], poses[i, 1, 3] + poses[i, 1, 2]],
    #             [poses[i, 2, 3], poses[i, 2, 3] + poses[i, 2, 2]], color=color)


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

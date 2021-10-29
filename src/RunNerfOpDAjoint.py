import sys
#sys.path.append("..")

import torch
import numpy as np
import matplotlib.pyplot as plt
from cubeadv.opt.discrete_adjoint import discrete_adjoint as dadjoint
from cubeadv.sim.sensors.depth_lidar import FreeLidarSensor
from cubeadv.sim.simple_sim.mlp import MLP
from cubeadv.sim.dynamics import Dynamics
from cubeadv.nerf import make_sim
from cubeadv.sim.utils import PathMapCost
from cubeadv.policies.depth_policy import RGBNet, Policy
from torch.autograd import grad
from torch.autograd import Variable

nerf_builder = lambda _, x, y: make_sim("../configs/nerf/donerf.txt", x, y, box_num=1,chunk_size=2000)
sensor = FreeLidarSensor(nerf_builder)


net = RGBNet()
net.load_state_dict(torch.load("../experiments/policy-rgb/model_rgb.pt"))
net.double().cuda()
policy = Policy(net, rgb=True)

dynamics = Dynamics(0.001)


def show_img(o):
    plt.imshow(o.transpose(1, 0).cpu().detach().numpy(), cmap='gray')
    plt.show()
    
def F(t2, t1, xn, x, p): 
    o = sensor(x, p)    
    u = policy(o.double())
    xd = dynamics.f(x, u)
    xn = x + xd*(t2 - t1)
    print('end of F,o shape={} u= {} x={}'.format( o.shape, u.cpu().detach().numpy(),x.cpu().detach().numpy()))
    return xn

def test():
    pm = PathMapCost.get_carla_town()
    C = lambda x: -pm.cost(x, None)

    T = torch.linspace(0, 2, 200).double().cuda()
    x0 = torch.tensor([92.4, 124.0, -np.pi/2]).double().cuda()
    c0 = torch.Tensor([92.4,124.0, 1.0, 1.,1.,1.,1.,1.,1.,1.,1.,1.]).double().cuda().requires_grad_(True)
    # c0 = torch.ones((12)).double().cuda().requires_grad_(True)

    loss_fn = lambda c_ : dadjoint(F, C, x0, T, c_)

    steps = 50
    params = Variable(c0,  requires_grad=True)
    optimizer = torch.optim.Adam([params], lr = 1e-1)
    for _ in range(steps):   
        optimizer.zero_grad()
        loss = loss_fn(params)
        loss.backward()
        optimizer.step()
        print('[* LOSS:{},params: {}]'.format(loss,params))




if __name__ == '__main__':
    setup_logger()
    test()



 
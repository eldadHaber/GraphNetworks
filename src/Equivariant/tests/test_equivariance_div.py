import math
from typing import Dict, Union
import numpy as np
import torch
from torch_geometric.data import Data
from torch_cluster import radius_graph
from torch_scatter import scatter


from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import FullyConnectedNet, Gate, ExtractIr, Activation
from e3nn.o3 import TensorProduct, FullyConnectedTensorProduct
from e3nn.util.jit import compile_mode
from torch.autograd import grad
import torch.nn.functional as F
from src.Equivariant.EQ_operations import *
from e3nn.util.test import equivariance_error


class test_net(torch.nn.Module):
    def __init__(self,irreps) -> None:
        super().__init__()
        self.tp = o3.ElementwiseTensorProduct(irreps, irreps)
        self.irreps_out = self.tp.irreps_out
        self.W1 = torch.randn((10))
        self.W2 = torch.zeros_like(self.W1)
        tmp = torch.randn((4))
        self.W2[0:3] = tmp[:3]
        self.W2[3] = tmp[2]
        self.W2[4] = tmp[2]
        self.W2[5] = tmp[3]
        self.W2[6] = tmp[3]
        self.W2[7] = tmp[3]
        self.W2[8] = tmp[3]
        self.W2[9] = tmp[3]
        # self.W1 = torch.ones((10),dtype=torch.float32)
        # self.W2 = torch.ones((10),dtype=torch.float32)
        return

    def nodeGrad(self, x, esrc,edst,W):
        # W = torch.randn((1,x.shape[-1]))
        g = W * (x[edst, :] - x[esrc, :])
        return g



    def forward(self, xn,xe):
        y = self.tp(xn,xe)
        z1 = self.W1 * (y[1,:] - y[0,:])
        z2 = self.W2 * (y[1,:] - y[0,:])
        return z1, z2

if __name__ == '__main__':

    irreps = o3.Irreps("1x0e+1x1o")
    xn = irreps.randn(2,-1)
    xe = irreps.randn(2,-1)
    model = test_net(irreps)
    irreps_out = model.irreps_out

    rot = o3.rand_matrix(1)
    D_in = irreps.D_from_matrix(rot)
    D_out = irreps_out.D_from_matrix(rot)

    xn_rot = (xn @ D_in).squeeze()
    xe_rot = (xe @ D_in).squeeze()

    z1, z2 = model(xn,xe)
    zr1, zr2 = model(xn_rot,xe_rot)

    z1r = z1 @ D_out
    z2r = z2 @ D_out




    print("Done")




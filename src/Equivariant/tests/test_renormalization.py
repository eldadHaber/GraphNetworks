import math
from typing import Dict, Union
import numpy as np
import torch
from torch_geometric.data import Data
from torch_cluster import radius_graph
from torch_scatter import scatter

import e3nn
from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import FullyConnectedNet, Gate, ExtractIr, Activation
from e3nn.o3 import TensorProduct, FullyConnectedTensorProduct
from e3nn.util.jit import compile_mode
from torch.autograd import grad
import torch.nn.functional as F
from src.Equivariant.EQ_operations import *
from e3nn.util.test import equivariance_error


class Norm(torch.nn.Module):
    def __init__(self, irreps):
        super().__init__()
        self.irreps_in = irreps
        self.irreps_out = irreps

        nd = irreps.dim
        nr = irreps.num_irreps
        degen = torch.empty(nr, dtype=torch.int64)
        m_degen = torch.empty(nr, dtype=torch.bool)
        idx = 0
        for mul, ir in irreps:
            li = 2 * ir.l + 1
            for i in range(mul):
                degen[idx + i] = li
                m_degen[idx + i] = ir.l == 0
            idx += i + 1
        M = m_degen.repeat_interleave(degen)
        self.register_buffer("m_scalar", M)
        return

    def forward(self, x, eps=1e-6):
        nb, _ = x.shape
        ms = self.m_scalar
        # x[:,ms] = x[:,ms] - torch.mean(x[:,ms], dim=1, keepdim=True)
        x[:, ms] /= torch.sqrt(torch.sum(x[:, ms] ** 2, dim=1, keepdim=True) + eps)

        mv = ~ms
        # We need to decide how to handle the vectors, eps is the tricky part
        xx = x[:, mv].view(nb, -1, 3)
        norm = xx.norm(dim=-1)
        xx /= norm[:, :, None]
        x[:, mv] = xx.view(nb, -1)
        return x


class test_net(torch.nn.Module):
    def __init__(self,irreps_in,irreps_hidden,irreps_out,nlayers) -> None:
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.nlayers = nlayers
        self.tp_open = FullyConnectedTensorProduct(irreps_in, irreps_in, irreps_hidden)
        self.norm = NormActivation_v2(irreps_hidden,torch.sigmoid) #Norm(irreps_hidden)
        # self.norm = Norm(irreps_hidden)
        self.layers = torch.nn.ModuleList()
        for i in range(nlayers):
            tp = SelfInteraction(irreps_in=irreps_hidden,irreps_out=irreps_hidden)
            self.layers.append(tp)

        self.tp_close = FullyConnectedTensorProduct(irreps_in,irreps_hidden,irreps_out)
        return



    def forward(self, xn,xe):
        x = self.tp_open(xn,xe)
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = self.norm(x)
        y = self.tp_close(xn,x)
        return y

if __name__ == '__main__':

    irreps_in = o3.Irreps("2x0e+2x1o")
    irreps_hidden = o3.Irreps("2x0e+2x1o")
    irreps_out = o3.Irreps("2x0e+2x1o")
    xn = irreps_in.randn(2,-1)
    xe = irreps_in.randn(2,-1)
    nlayers = 5
    model = test_net(irreps_in,irreps_hidden,irreps_out,nlayers)
    irreps_out = model.irreps_out

    rot = o3.rand_matrix(1)
    D_in = irreps_in.D_from_matrix(rot)
    D_out = irreps_out.D_from_matrix(rot)

    xn_rot = (xn @ D_in).squeeze()
    xe_rot = (xe @ D_in).squeeze()

    z = model(xn,xe)
    zr = model(xn_rot,xe_rot)

    zra = z @ D_out




    print("Done")




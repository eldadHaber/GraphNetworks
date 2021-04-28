"""model with self-interactions and gates
Exact equivariance to :math:`E(3)`
version of february 2021
"""
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



def smooth_cutoff(x):
    u = 2 * (x - 1)
    y = (math.pi * u).cos().neg().add(1).div(2)
    y[u > 0] = 0
    y[u < -1] = 1
    return y


def tp_path_exists(irreps_in1, irreps_in2, ir_out):
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False



@compile_mode('script')
class Convolution(torch.nn.Module):
    r"""equivariant convolution
    Parameters
    ----------
    irreps_in : `Irreps`
        representation of the input node features
    irreps_node_attr : `Irreps`
        representation of the node attributes
    irreps_edge_attr : `Irreps`
        representation of the edge attributes
    irreps_out : `Irreps` or None
        representation of the output node features
    number_of_edge_features : int
        number of scalar (0e) features of the edge used to feed the FC network
    radial_layers : int
        number of hidden layers in the radial fully connected network
    radial_neurons : int
        number of neurons in the hidden layers of the radial fully connected network
    num_neighbors : float
        typical number of nodes convolved over
    """
    def __init__(
        self,
        irreps_in,
        irreps_node_attr,
        irreps_edge_attr,
        irreps_out,
        number_of_edge_features,
        radial_layers,
        radial_neurons,
        num_neighbors
    ) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_out = o3.Irreps(irreps_out)
        self.num_neighbors = num_neighbors

        self.sc = FullyConnectedTensorProduct(self.irreps_in, self.irreps_node_attr, self.irreps_out)

        self.lin1 = FullyConnectedTensorProduct(self.irreps_in, self.irreps_node_attr, self.irreps_in)

        irreps_mid = []
        instructions = []
        for i, (mul, ir_in) in enumerate(self.irreps_in):
            for j, (_, ir_edge) in enumerate(self.irreps_edge_attr):
                for ir_out in ir_in * ir_edge:
                    if ir_out in self.irreps_out:
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, 'uvu', True))
        irreps_mid = o3.Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()

        instructions = [
            (i_1, i_2, p[i_out], mode, train)
            for i_1, i_2, i_out, mode, train in instructions
        ]

        tp = TensorProduct(
            self.irreps_in,
            self.irreps_edge_attr,
            irreps_mid,
            instructions,
            internal_weights=False,
            shared_weights=False,
        )
        self.fc = FullyConnectedNet(
            [number_of_edge_features] + radial_layers * [radial_neurons] + [tp.weight_numel],
            torch.nn.functional.silu
        )
        self.tp = tp

        self.lin2 = FullyConnectedTensorProduct(irreps_mid, self.irreps_node_attr, self.irreps_out)

    def forward(self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_features) -> torch.Tensor:
        weight = self.fc(edge_features)

        x = node_input

        s = self.sc(x, node_attr)
        x = self.lin1(x, node_attr)

        edge_features = self.tp(x[edge_src], edge_attr, weight)
        x = scatter(edge_features, edge_dst, dim=0, dim_size=x.shape[0]).div(self.num_neighbors**0.5)

        x = self.lin2(x, node_attr)

        c_s, c_x = math.sin(math.pi / 8), math.cos(math.pi / 8)
        m = self.sc.output_mask
        c_x = (1 - m) + c_x * m
        return c_s * s + c_x * x



class SelfExpandingGate(torch.nn.Module):
    def __init__(self, irreps):
        super().__init__()
        self.irreps_in = irreps
        self.irreps_out = irreps

        act = {
            1: torch.nn.functional.silu,
            -1: torch.tanh,
        }
        act_gates = {
            1: torch.sigmoid,
            -1: torch.tanh,
        }

        irreps_scalars = o3.Irreps([(mul, ir) for mul, ir in irreps if ir.l == 0])
        irreps_gated = o3.Irreps([(mul, ir) for mul, ir in irreps if ir.l > 0 ])
        ir = "0e"
        irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated])
        if o3.Irreps(irreps_gated).dim == 0:
            self.si = Identity()
            activation_fnc = []
            for mul,ir in o3.Irreps(irreps_scalars):
                if ir.p == 1:
                    activation_fnc.append(torch.nn.functional.silu)
                else:
                    activation_fnc.append(torch.tanh)
            self.gate = Activation(irreps_scalars, activation_fnc)
        else:
            self.gate = Gate(
                irreps_scalars, [act[ir.p] for _, ir in irreps_scalars],  # scalar
                irreps_gates, [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated)  # gated tensors
            self.si = SelfInteraction(irreps, self.gate.irreps_in)
        return

    def forward(self,x):
        x = self.si(x)
        x = self.gate(x)
        return x



class Compose(torch.nn.Module):
    def __init__(self, first, second):
        super().__init__()
        self.first = first
        self.second = second
        self.irreps_in = self.first.irreps_in
        self.irreps_out = self.second.irreps_out

    def forward(self, *input):
        x = self.first(*input)
        return self.second(x)

class Identity(torch.nn.Module):
        def __init__(self):
            super().__init__()
            return
        def forward(self, input):
            return input

class Filter(torch.nn.Module):
    def __init__(self, number_of_basis, radial_layers, radial_neurons, irrep):
        super().__init__()
        self.irrep = irrep
        nd = irrep.dim
        nr = irrep.num_irreps
        self.net = FullyConnectedNet([number_of_basis] + radial_layers * [radial_neurons] + [nr], torch.nn.functional.silu)
        S = torch.empty(nr,dtype=torch.int64)
        idx = 0
        for mul,ir in irrep:
            li = 2*ir.l+1
            for i in range(mul):
                S[idx+i] = li
            idx += i+1
        self.register_buffer("degen", S)
        return

    def forward(self, x):
        x = self.net(x)
        y = x.repeat_interleave(self.degen,dim=1)
        return y


class SelfInteraction(torch.nn.Module):
    def __init__(self,irreps_in,irreps_out):
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.tp = o3.FullyConnectedTensorProduct(irreps_in, irreps_in, irreps_out)
        return

    def forward(self, x):
        x = self.tp(x,x)
        return x



class TvNorm(torch.nn.Module):
        def __init__(self,irreps):
            super().__init__()
            self.irreps_in = irreps
            self.irreps_out = irreps

            nd = irreps.dim
            nr = irreps.num_irreps
            degen = torch.empty(nr, dtype=torch.int64)
            m_degen = torch.empty(nr,dtype=torch.bool)
            idx = 0
            for mul, ir in irreps:
                li = 2 * ir.l + 1
                for i in range(mul):
                    degen[idx + i] = li
                    m_degen[idx+i] = ir.l == 0
                idx += i + 1
            M = m_degen.repeat_interleave(degen)
            self.register_buffer("m_scalar", M)
            return

        def forward(self, x, eps=1e-6):
            nb,_ = x.shape
            ms = self.m_scalar
            # x[:,ms] = x[:,ms] - torch.mean(x[:,ms], dim=1, keepdim=True)
            x[:,ms] /= torch.sqrt(torch.sum(x[:,ms] ** 2, dim=1, keepdim=True) + eps)

            mv = ~ms
            #We need to decide how to handle the vectors, eps is the tricky part
            xx = x[:,mv].view(nb,-1,3)
            norm = xx.norm(dim=-1)
            xx /= norm[:,:,None]
            x[:,mv] = xx.view(nb,-1)
            return x


def tv_norm(X, eps=1e-3):
    X = X - torch.mean(X, dim=1, keepdim=True)
    X = X / torch.sqrt(torch.sum(X ** 2, dim=1, keepdim=True) + eps)
    return X


class DoubleLayer(torch.nn.Module):
    def __init__(self, irreps_in,irreps_hidden,irreps_out):
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_hidden = irreps_hidden
        self.irreps_out_intended = irreps_out
        self.non_linear_act1 = e3nn.nn.NormActivation(self.irreps_in,torch.sigmoid,normalize=False,bias=True)
        # self.g1 = SelfExpandingGate(irreps_in)


        irreps = o3.Irreps([(mul, ir) for mul, ir in irreps_hidden if
                   tp_path_exists(irreps_in, irreps_in, ir)])
        self.si1 = SelfInteraction(self.irreps_in,irreps)
        self.normalize_and_non_linear_act2 = e3nn.nn.NormActivation(irreps,torch.sigmoid,normalize=True,bias=False)

        # self.tv = TvNorm(irreps)
        # self.g2 = SelfExpandingGate(irreps)
        irreps2 = o3.Irreps([(mul, ir) for mul, ir in irreps_out if
                   tp_path_exists(irreps, irreps, ir)])
        self.si2 = SelfInteraction(irreps,irreps2)
        # self.g3 = SelfExpandingGate(irreps2)
        self.non_linear_act3 = e3nn.nn.NormActivation(irreps2, torch.sigmoid, normalize=False, bias=True)

        self.irreps_out = irreps2
        return

    def forward(self, x):
        x = self.non_linear_act1(x)
        x = self.si1(x)
        x = self.normalize_and_non_linear_act2(x)
        x = self.si2(x)
        x = self.non_linear_act3(x)
        return x

def zero_small_numbers(x,eps=1e-6):
    M = torch.abs(x) < eps
    x[M] = 0
    return x


class Concatenate(torch.nn.Module):
    def __init__(self,irreps_in):
        super().__init__()
        self.irreps_in = irreps_in
        irreps_sorted, J, I = irreps_in.sort()
        I = torch.tensor(I)
        J = torch.tensor(J)
        self.irreps_out = irreps_sorted.simplify()
        idx_conversion = torch.empty(irreps_in.dim,dtype=torch.int64)
        S = torch.empty(len(I),dtype=torch.int64)
        for i, (mul,ir) in enumerate(irreps_in):
            li = 2*ir.l+1
            S[i] = li*mul
        idx_cum = torch.zeros((len(I)+1),dtype=torch.int64)
        idx_cum[1:] = torch.cumsum(S,dim=0)
        ii0 = 0
        for i,Ii in enumerate(I):
            idx_conversion[ii0:ii0+S[Ii]] = torch.arange(idx_cum[Ii],idx_cum[Ii]+S[Ii])
            ii0 += S[Ii]
        idx_conversion_rev = torch.argsort(idx_conversion)
        self.register_buffer("idx_conversion", idx_conversion)
        self.register_buffer("idx_conversion_rev", idx_conversion_rev)
        return

    def forward(self, x,dim):
        x = torch.cat(x,dim)
        x = torch.index_select(x, dim, self.idx_conversion)
        return x

    def reverse_idx(self,x,dim):
        x = torch.index_select(x, dim, self.idx_conversion_rev)
        return x





if __name__ == '__main__':
    #
    # irreps = o3.Irreps("2x0e+1x1e")
    # catfnc = Concatenate(3*irreps)
    #
    # x = irreps.randn(5, -1, normalization='norm')
    # y = catfnc([x,x,x],dim=1)
    # print('done')
    a = torch.tensor([0,2,4,1,3])
    b = torch.argsort(a)
    # b = torch.arange(5)
    # c = b[a]
    # d = a[b]
    print('done')
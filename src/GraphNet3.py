import os, sys

import e3nn.o3
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import torch.optim as optim
from torch_scatter import scatter


class Mixing(nn.Module):
    def __init__(self, dim_in1,dim_in2,dim_out,use_bilinear=True,use_e3nn=True):
        super(Mixing, self).__init__()
        self.use_bilinear = use_bilinear
        if use_bilinear:
            if use_e3nn:
                irreps1 = e3nn.o3.Irreps("{:}x0e".format(dim_in1))
                irreps2 = e3nn.o3.Irreps("{:}x0e".format(dim_in2))
                self.bilinear = e3nn.o3.FullyConnectedTensorProduct(irreps1, irreps2, irreps1)
            else:
                self.bilinear = nn.Bilinear(dim_in1, dim_in2, dim_out, bias=False)
        self.lin = nn.Linear(2*dim_in1+dim_in2,dim_out)

    def forward(self, x1,x2):
        x = torch.cat([x1,x2],dim=-1)
        if self.use_bilinear:
            x_bi = self.bilinear(x1,x2)
            x = torch.cat([x,x_bi],dim=-1)
        x = self.lin(x)
        return x


class NodesToEdges(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.mix = Mixing(dim_in,dim_in,dim_out)

    def forward(self,xn,xe_src,xe_dst, W):
        xe_grad = W * (xn[xe_src] - xn[xe_dst])
        xe_ave = W * (xn[xe_src] + xn[xe_dst]) / 2
        xe = self.mix(xe_grad,xe_ave)
        return xe


class EdgesToNodes(nn.Module):
    def __init__(self, dim_in, dim_out, num_neighbours=20,use_e3nn=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.norm = 1 / math.sqrt(num_neighbours)
        self.mix = Mixing(dim_in, dim_in, dim_out)

    def forward(self, xe, xe_src, xe_dst, W):
        xn_1 = scatter(W * xe,xe_dst,dim=0) * self.norm
        xn_2 = scatter(W * xe,xe_src,dim=0) * self.norm
        xn_div = xn_1 - xn_2
        xn_ave = xn_1 + xn_2
        xn = self.mix(xn_div, xn_ave)
        return xn

class FullyConnectedNet(nn.Module):
    """
    A simple fully connected network, with an activation function at the end
    """
    def __init__(self,dimensions,activation_fnc):
        super(FullyConnectedNet, self).__init__()
        self.layers = nn.ModuleList()
        self.nlayers = len(dimensions)-1
        for i in range(self.nlayers):
            ll = nn.Linear(dimensions[i],dimensions[i+1])
            self.layers.append(ll)
        self.activation = activation_fnc
        return

    def forward(self,x):
        for i in range(self.nlayers):
            x = self.layers[i](x)
        x = self.activation(x)
        return x

class PropagationBlock(nn.Module):
    def __init__(self, xn_dim, xn_attr_dim, xe_dim, xe_attr_dim, use_e3nn=True):
        super().__init__()

        self.fc1 = FullyConnectedNet([1, xe_dim],activation_fnc=torch.nn.functional.silu)
        self.fc2 = FullyConnectedNet([1, xn_dim],activation_fnc=torch.nn.functional.silu)

        self.mix_xn = Mixing(xn_dim,xn_attr_dim,xn_dim)
        self.mix_xe = Mixing(xe_dim,xe_attr_dim,xe_dim)

        self.nodes_to_edges = NodesToEdges(xn_dim,xn_dim)
        self.edges_to_nodes = EdgesToNodes(xe_dim,xe_dim)

        self.activation = torch.nn.functional.silu

        return

    def forward(self, xn, xn_attr, xe_attr, xe_src, xe_dst):
        eps = 1e-9


        xn = self.mix_xn(xn, xn_attr)
        xn = xn / (xn.std(dim=1)[:,None] + eps)

        weight = self.fc1(xe_attr)
        xe = self.nodes_to_edges(xn, xe_src, xe_dst, weight)

        xe = self.mix_xe(xe, xe_attr)
        xe = xe / (xe.std(dim=1)[:,None] + eps)

        weight = self.fc2(xe_attr)
        xn = self.edges_to_nodes(xe, xe_src, xe_dst, weight)

        xn = self.activation(xn)
        xn = xn / (xn.std(dim=1)[:,None] + eps)

        return xn


class GraphNet3(nn.Module):
    """
    This network is designed to predict the 3D coordinates of a set of particles.
    """
    def __init__(self, xn_attr_dim, xn_dim, xe_dim, xe_attr_dim, nlayers, xn_dim_in=None):
        super().__init__()

        self.nlayers = nlayers

        w = torch.empty((3, xn_dim))
        nn.init.xavier_normal_(w,gain=1/math.sqrt(xn_dim)) # Filled according to "Semi-Orthogonal Low-Rank Matrix Factorization for Deep Neural Networks"
        self.projection_matrix = nn.Parameter(w)
        if xn_dim_in is not None:
            self.xn_open = nn.Linear(xn_dim_in,xn_dim,bias=False) #nn.Conv1d(xn_dim_in,xn_dim_hidden,kernel_size=1)
        self.PropagationBlocks = nn.ModuleList()
        self.angles = nn.ModuleList()
        for i in range(nlayers):
            block = PropagationBlock(xn_dim=xn_dim, xn_attr_dim=xn_attr_dim, xe_dim=xe_dim, xe_attr_dim=xe_attr_dim)
            self.PropagationBlocks.append(block)
            angle = nn.Linear(xn_dim,1)
            self.angles.append(angle)
        return



    def make_matrix_semi_unitary(self, M,debug=False):
        I = torch.eye(M.shape[-2])
        if debug:
            M_org = M.clone()
        for i in range(10):
            M = M - 0.5 * (M @ M.t() - I) @ M

        if debug:
            pre_error = torch.norm(I - M_org @ M_org.t())
            post_error = torch.norm(I - M @ M.t())
            print(f"Deviation from unitary before: {pre_error:2.2e}, deviation from unitary after: {post_error:2.2e}")
        return M

    def project(self,x, std=None):
        M = self.projection_matrix
        M = self.make_matrix_semi_unitary(M)
        R = x @ M.t()
        return R

    def uplift(self,R, eps=1e-9):
        M = self.projection_matrix
        M = self.make_matrix_semi_unitary(M)
        x = R @ M
        return x

    def apply_constraints(self, x, n=1, d=3.8):
        for j in range(n):
            x3 = self.project(x)
            c = constraint(x3.t(), d)
            lam = dConstraintT(c, x3.t())
            lam = self.uplift(lam.t())

            with torch.no_grad():
                if j == 0:
                    alpha = 1.0 / lam.norm()
                lsiter = 0
                while True:
                    xtry = x - alpha * lam
                    x3 = self.project(xtry)
                    ctry = constraint(x3.t(), d)
                    if torch.norm(ctry) < torch.norm(c):
                        break
                    alpha = alpha / 2
                    lsiter = lsiter + 1
                    if lsiter > 10:
                        break
                if lsiter == 0:
                    alpha = alpha * 1.5
            x = x - alpha * lam
        return x

    def forward(self, input, xn_attr, xe_src, xe_dst):

        if input.shape[-1] == 3:
            R = input
            xn = self.uplift(R) # We take R from 3D to nhidden dimension with a reversible learnable operation where the reverse is the projection. Maybe a unitary matrix would be the answer if this is possible to learn in pytorch?

        for i in range(self.nlayers):
            # xe_vec, xe_src, xe_dst = self.compute_graph(R)
            xe_vec = R[xe_dst] - R[xe_src]
            xe_attr = 1 / (xe_vec.norm(dim=1)[:,None] + 1e-9)
            xn_org = xn.clone() # Make sure this actually becomes a clone and not just a pointer

            #We mix node information by using a node_to_edge_propagation block.
            xn = self.PropagationBlocks[i](xn, xn_attr, xe_attr,xe_src,xe_dst)

            #We let the network compute the mixing angle/strength
            node_angle = 0.1 * self.angles[i](xn)

            #which we use in a way that preserves variance
            w_self_conn, w_layer_conn = node_angle.cos(), node_angle.sin()

            xn = w_self_conn * xn_org + w_layer_conn * xn

            xn = self.apply_constraints(xn, n=100)
            #Finally we prepare to go to the next layer
            R = self.project(xn)

        return R # * R_norm


def diffX(X):
    X = X.squeeze()
    return X[:,1:] - X[:,:-1]

def diffXT(X):
    X  = X.squeeze()
    D  = X[:,:-1] - X[:,1:]
    d0 = -X[:,0].unsqueeze(1)
    d1 = X[:,-1].unsqueeze(1)
    D  = torch.cat([d0,D,d1],dim=1)
    return D


def constraint(X,d=3.8):
    X = X.squeeze()
    c = torch.ones(1,3,device=X.device)@(diffX(X)**2) - d**2

    return c

def dConstraint(S,X):
    dX = diffX(X)
    dS = diffX(S)
    e  = torch.ones(1,3,device=X.device)
    dc = 2*e@(dX*dS)
    return dc

def dConstraintT(c,X):
    dX = diffX(X)
    e = torch.ones(3, 1, device=X.device)
    C = (e@c)*dX
    C = diffXT(C)
    return 2*C


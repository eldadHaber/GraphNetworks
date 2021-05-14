import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import torch.optim as optim
## r=1
from torch_geometric.nn import global_max_pool

from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, LeakyReLU as LRU

try:
    from src import graphOps as GO
    from src.batchGraphOps import getConnectivity
    from mpl_toolkits.mplot3d import Axes3D
    from src.utils import saveMesh, h_swish
    from src.inits import glorot, identityInit
except:
    import graphOps as GO
    from batchGraphOps import getConnectivity
    from mpl_toolkits.mplot3d import Axes3D
    from utils import saveMesh, h_swish
    from inits import glorot, identityInit

from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import GCN2Conv
from torch_scatter import scatter_add


def conv2(X, Kernel):
    return F.conv2d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))


def conv1(X, Kernel):
    return F.conv1d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))


def conv1T(X, Kernel):
    return F.conv_transpose1d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))


def conv2T(X, Kernel):
    return F.conv_transpose2d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def tv_norm(X, eps=1e-6):
    # X = X - torch.mean(X, dim=1, keepdim=True)
    X = X / torch.sqrt(torch.sum(X ** 2, dim=1, keepdim=True) + eps)
    return X


def doubleLayer(x, K1, K2):
    x = F.conv1d(x, K1.unsqueeze(-1))
    x = F.layer_norm(x, x.shape)
    x = torch.relu(x)
    x = F.conv1d(x, K2.unsqueeze(-1))
    return x


class graphNetwork(nn.Module):

    def __init__(self, nNin, nEin, nopen, nhid, nNclose, nlayer, h=0.1, dense=False, varlet=False):
        super(graphNetwork, self).__init__()

        self.h = h
        self.varlet = varlet
        self.dense = dense
        stdv = 1e-2
        stdvp = 1e-3
        self.K1Nopen = nn.Parameter(torch.randn(nopen, nNin) * stdv)
        self.K2Nopen = nn.Parameter(torch.randn(nopen, nopen) * stdv)
        if dense:
            self.K1Eopen = nn.Parameter(torch.randn(nopen, nEin, 9, 9) * stdv)
            self.K2Eopen = nn.Parameter(torch.randn(nopen, nopen, 9, 9) * stdv)
        else:
            self.K1Eopen = nn.Parameter(torch.randn(nopen, nEin) * stdv)
            self.K2Eopen = nn.Parameter(torch.randn(nopen, nopen) * stdv)

        self.KNclose = nn.Parameter(torch.randn(nNclose, nopen) * stdv)

        if varlet:
            Nfeatures = 2 * nopen
        else:
            Nfeatures = 3 * nopen
        if dense:
            self.KE1 = nn.Parameter(torch.rand(nlayer, nhid, Nfeatures, 9, 9) * stdvp)
            self.KE2 = nn.Parameter(torch.rand(nlayer, nopen, nhid, 9, 9) * stdvp)
        else:
            self.KE1 = nn.Parameter(torch.rand(nlayer, nhid, Nfeatures) * stdvp)
            self.KE2 = nn.Parameter(torch.rand(nlayer, nopen, nhid) * stdvp)

        self.KN1 = nn.Parameter(torch.rand(nlayer, nhid, Nfeatures) * stdvp)
        self.KN2 = nn.Parameter(torch.rand(nlayer, nopen, nhid) * stdvp)

    def edgeConv(self, xe, K):
        if xe.dim() == 4:
            if K.dim() == 2:
                xe = F.conv2d(xe, K.unsqueeze(-1).unsqueeze(-1))
            else:
                xe = conv2(xe, K)
        elif xe.dim() == 3:
            if K.dim() == 2:
                xe = F.conv1d(xe, K.unsqueeze(-1))
            else:
                xe = conv1(xe, K)
        return xe

    def doubleLayer(self, x, K1, K2):
        x = self.edgeConv(x, K1)
        x = F.layer_norm(x, x.shape)
        x = torch.relu(x)
        x = self.edgeConv(x, K2)
        return x

    def forward(self, xn, xe, Graph):

        # Opening layer
        # xn = [B, C, N]
        # xe = [B, C, N, N] or [B, C, E]
        # Opening layer
        xn = self.doubleLayer(xn, self.K1Nopen, self.K2Nopen)
        xe = self.doubleLayer(xe, self.K1Eopen, self.K2Eopen)
        N = Graph.nnodes
        nlayers = self.KE1.shape[0]
        xn_old = xn.clone()
        xe_old = xe.clone()
        for i in range(nlayers):

            tmp_node = xn.clone()
            tmp_edge = xe.clone()
            # gradX = torch.exp(-torch.abs(Graph.nodeGrad(xn)))
            gradX = Graph.nodeGrad(xn)
            intX = Graph.nodeAve(xn)
            if self.varlet:
                dxe = torch.cat([intX, gradX], dim=1)
            else:
                dxe = torch.cat([intX, xe, gradX], dim=1)

            dxe = self.doubleLayer(dxe, self.KE1[i], self.KE2[i])

            dxe = F.layer_norm(dxe, dxe.shape)
            # dxe = torch.relu(dxe)
            # xe = (xe + self.h * dxe)

            divE = Graph.edgeDiv(xe)
            aveE = Graph.edgeAve(xe, method='ave')
            # divE = Graph.edgeDiv(dxe)
            # aveE = Graph.edgeAve(dxe, method='ave')

            if self.varlet:
                dxn = torch.cat([aveE, divE], dim=1)
            else:
                dxn = torch.cat([aveE, divE, xn], dim=1)

            dxn = self.doubleLayer(dxn, self.KN1[i], self.KN2[i])

            xe = xe + self.h * dxe
            xn = (xn + self.h * dxn)
            # xn = 2*xn - xn_old + self.h**2 * dxn
            # xe = 2*xe - xe_old + self.h**2 * dxe

            debug = True
            if debug:
                print("xn shape:", xn.shape)
                print("xe shape:", xe.shape)
                xn_norm = xn.detach().squeeze().norm(dim=0).cpu().numpy()
                xe_norm = xe.detach().squeeze().norm(dim=0).cpu().numpy()

                plt.figure()
                plt.plot(xn_norm)
                plt.show()
                plt.savefig('plots/xn_norm_layer_verlet' + str(i) + '.jpg')
                plt.close()

                plt.figure()
                plt.plot(xe_norm)
                plt.show()
                plt.savefig('plots/xe_norm_layer_verlet' + str(i) + '.jpg')
                plt.close()

        xn = F.conv1d(xn, self.KNclose.unsqueeze(-1))

        return xn, xe


class graphNetwork_try(nn.Module):

    def __init__(self, nNin, nEin, nopen, nhid, nNclose, nlayer, h=0.1, dense=False, varlet=False, wave=True,
                 diffOrder=1, num_output=1024, dropOut=False):
        super(graphNetwork_try, self).__init__()
        self.wave = wave
        if not wave:
            self.heat = True
        else:
            self.heat = False
        self.h = h
        self.varlet = varlet
        self.dense = dense
        self.diffOrder = diffOrder
        self.num_output = num_output
        self.dropout = dropOut
        stdv = 1e-1
        stdvp = 1e-1
        self.K1Nopen = nn.Parameter(torch.randn(nopen, nNin) * stdv)
        self.K2Nopen = nn.Parameter(torch.randn(nopen, nopen) * stdv)
        if dense:
            self.K1Eopen = nn.Parameter(torch.randn(nopen, nEin, 9, 9) * stdv)
            self.K2Eopen = nn.Parameter(torch.randn(nopen, nopen, 9, 9) * stdv)
        else:
            self.K1Eopen = nn.Parameter(torch.randn(nopen, nEin) * stdv)
            self.K2Eopen = nn.Parameter(torch.randn(nopen, nopen) * stdv)

        self.KNclose = nn.Parameter(torch.randn(nNclose, nopen) * stdv)

        if varlet:
            Nfeatures = 2 * nopen
        else:
            Nfeatures = 3 * nopen
        if dense:
            self.KE1 = nn.Parameter(torch.rand(nlayer, nhid, Nfeatures, 9, 9) * stdvp)
            self.KE2 = nn.Parameter(torch.rand(nlayer, nopen, nhid, 9, 9) * stdvp)
        else:
            self.KE1 = nn.Parameter(torch.rand(nlayer, nhid, Nfeatures) * stdvp)
            self.KE2 = nn.Parameter(torch.rand(nlayer, nopen, nhid) * stdvp)

        self.KN1 = nn.Parameter(torch.rand(nlayer, nhid, Nfeatures) * stdvp)
        self.KN2 = nn.Parameter(torch.rand(nlayer, nopen, nhid) * stdvp)

        self.lin1 = torch.nn.Linear(nopen, 256)
        # self.lin2 = torch.nn.Linear(256, self.num_nodes)

        self.lin2 = torch.nn.Linear(256, num_output)

    def edgeConv(self, xe, K):
        if xe.dim() == 4:
            if K.dim() == 2:
                xe = F.conv2d(xe, K.unsqueeze(-1).unsqueeze(-1))
            else:
                xe = conv2(xe, K)
        elif xe.dim() == 3:
            if K.dim() == 2:
                xe = F.conv1d(xe, K.unsqueeze(-1))
            else:
                xe = conv1(xe, K)
        return xe

    def doubleLayer(self, x, K1, K2):
        x = self.edgeConv(x, K1)
        if self.dropout:
            x = F.dropout(x, p=0.6, training=self.training)
        x = F.layer_norm(x, x.shape)
        x = torch.tanh(x)
        x = self.edgeConv(x, K2)
        if self.dropout:
            x = F.dropout(x, p=0.6, training=self.training)
        return x

    def nodeDeriv(self, features, Graph, order=1, edgeSpace=True):
        ## if edgeSpace==True, return features in edge space.
        x = features
        operators = []
        for i in torch.arange(0, order):
            x = Graph.nodeGrad(x)
            if edgeSpace:
                operators.append(x)

            # if i == order - 1:
            #    break

            x = Graph.edgeDiv(x)
            if not edgeSpace:
                operators.append(x)

        if edgeSpace:
            out = x
        else:
            out = Graph.edgeAve(x, method='ave')
        operators.append(out)
        return operators

    def saveOperatorImages(self, operators):
        for i in torch.arange(0, len(operators)):
            op = operators[i]
            op = op.detach().squeeze().cpu()  # .numpy()

            plt.figure()
            img = op.reshape(32, 32)
            img = img / img.max()
            plt.imshow(img)
            plt.colorbar()
            plt.show()
            plt.savefig('plots/operator' + str(i) + '.jpg')
            plt.close()

    def savePropagationImage(self, xn, xe, dxe, Graph, i=0):
        plt.figure()
        img = xn.clone().detach().squeeze().reshape(32, 32).cpu().numpy()
        # img = img / img.max()
        plt.imshow(img)
        plt.colorbar()
        plt.show()
        plt.savefig('plots/img_xn_norm_layer_heat' + str(i) + 'order_nodeDeriv' + str(self.diffOrder) + '.jpg')
        plt.close()

        divE = Graph.edgeDiv(dxe)
        plt.figure()
        img = divE.clone().detach().squeeze().reshape(32, 32).cpu().numpy()
        # img = img / img.max()
        plt.imshow(img)
        plt.colorbar()
        plt.show()
        plt.savefig('plots/img_xe_div_norm_layer_heat' + str(i) + 'order_nodeDeriv' + str(self.diffOrder) + '.jpg')
        plt.close()

    def forward(self, xn, xe, Graph):

        # Opening layer
        # xn = [B, C, N]
        # xe = [B, C, N, N] or [B, C, E]
        # Opening layer
        if self.dropout:
            xn = F.dropout(xn, p=0.6, training=self.training)
        xn = self.doubleLayer(xn, self.K1Nopen, self.K2Nopen)
        xe = self.doubleLayer(xe, self.K1Eopen, self.K2Eopen)
        if self.dropout:
            xn = F.dropout(xn, p=0.6, training=self.training)
        debug = False
        if debug:
            image = False
            if image:
                plt.figure()
                img = xn.clone().detach().squeeze().reshape(32, 32).cpu().numpy()
                img = img / img.max()
                plt.imshow(img)
                plt.colorbar()
                plt.show()
                plt.savefig('plots/img_xn_norm_layer_verlet' + str(0) + 'order_nodeDeriv' + str(0) + '.jpg')
                plt.close()
            else:
                saveMesh(xn.squeeze().t(), Graph.faces, Graph.pos, 0)

        N = Graph.nnodes
        nlayers = self.KE1.shape[0]
        xn_old = xn.clone()
        xe_old = xe.clone()
        for i in range(nlayers):
            if i % 200 == 199:  # update graph
                I, J = getConnectivity(xn.squeeze(0))
                Graph = GO.graph(I, J, N)
            tmp_node = xn.clone()
            tmp_edge = xe.clone()
            if 1 == 0:
                features = xn.squeeze()
                D = torch.relu(torch.sum(features ** 2, dim=0, keepdim=True) + \
                               torch.sum(features ** 2, dim=0, keepdim=True).t() - \
                               2 * features.t() @ features)

                D = D / D.std()
                D = torch.exp(-2 * D)
                I = Graph.iInd
                J = Graph.jInd
                w = D[I, J]
                Graph = GO.graph(I, J, N, W=w, pos=None, faces=None)

            gradX = Graph.nodeGrad(xn)
            intX = Graph.nodeAve(xn)

            # operators = self.nodeDeriv(xn, Graph, order=self.diffOrder, edgeSpace=True)
            # if debug and image:
            #    self.saveOperatorImages(operators)

            if self.varlet:
                dxe = torch.cat([intX, gradX], dim=1)
            else:
                dxe = torch.cat([intX, xe, gradX], dim=1)

            dxe = self.doubleLayer(dxe, self.KE1[i], self.KE2[i])

            dxe = F.layer_norm(dxe, dxe.shape)

            if self.wave:
                xe = xe + self.h * dxe
                divE = Graph.edgeDiv(xe)
                aveE = Graph.edgeAve(xe, method='ave')

            if self.heat:
                dxe = torch.tanh(dxe)
                divE = Graph.edgeDiv(dxe)
                aveE = Graph.edgeAve(dxe, method='ave')

            if self.varlet:
                dxn = torch.cat([aveE, divE], dim=1)
            else:
                dxn = torch.cat([aveE, divE, xn], dim=1)

            dxn = self.doubleLayer(dxn, self.KN1[i], self.KN2[i])

            if self.wave:
                xn = xn + self.h * dxn
            else:
                xn = xn - self.h * dxn

            if debug:
                if image:
                    self.savePropagationImage(xn, xe, dxe, Graph, i + 1)
                else:
                    saveMesh(xn.squeeze().t(), Graph.faces, Graph.pos, i + 1)

        xn = F.conv1d(xn, self.KNclose.unsqueeze(-1))
        xn = xn.squeeze().t()

        if self.dropout:
            # for cora
            x = F.dropout(xn, p=0.6, training=self.training)
            x = F.relu(self.lin1(xn))
        else:
            # for faust
            x = F.elu(self.lin1(xn))
        if self.dropout:
            x = F.dropout(x, p=0.6, training=self.training)
        x = self.lin2(x)

        return F.log_softmax(x, dim=1)

        # return xn, xe


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), BN(channels[i]), ReLU())
        for i in range(1, len(channels))
    ])


# , BN(channels[i])

class graphNetwork_nodesOnly(nn.Module):

    def __init__(self, nNin, nopen, nhid, nNclose, nlayer, h=0.1, dense=False, varlet=False, wave=True,
                 diffOrder=1, num_output=1024, dropOut=False, modelnet=False, faust=False, GCNII=False,
                 graphUpdate=None, PPI=False, gated=False, realVarlet=False, mixDyamics=False, doubleConv=False,
                 tripleConv=False):
        super(graphNetwork_nodesOnly, self).__init__()
        self.wave = wave
        self.realVarlet = realVarlet
        if not wave:
            self.heat = True
        else:
            self.heat = False
        self.mixDynamics = mixDyamics
        self.h = h
        self.varlet = varlet
        self.dense = dense
        self.diffOrder = diffOrder
        self.num_output = num_output
        self.graphUpdate = graphUpdate
        self.doubleConv = doubleConv
        self.tripleConv = tripleConv
        self.gated = gated
        self.faust = faust
        self.PPI = PPI
        if dropOut > 0.0:
            self.dropout = dropOut
        else:
            self.dropout = False
        self.nlayers = nlayer
        stdv = 1e-2
        stdvp = 1e-2
        if self.faust or self.PPI:
            stdv = 1e-1
            stdvp = 1e-1
            stdv = 1e-2
            stdvp = 1e-2
        self.K1Nopen = nn.Parameter(torch.randn(nopen, nNin) * stdv)
        self.K2Nopen = nn.Parameter(torch.randn(nopen, nopen) * stdv)
        self.convs1x1 = nn.Parameter(torch.randn(nlayer, nopen, nopen) * stdv)

        if not self.faust:
            self.KNclose = nn.Parameter(torch.randn(num_output, nopen) * stdv)  # num_output on left size
            # self.KNclose2 = nn.Parameter(torch.randn(num_output, int(round(nopen / 2))) * stdv)

        else:
            self.KNclose = nn.Parameter(torch.randn(nopen, nopen) * stdv)

        if varlet:
            Nfeatures = 1 * nopen
        else:
            Nfeatures = 1 * nopen

        self.KN1 = nn.Parameter(torch.rand(nlayer, Nfeatures, nhid) * stdvp)
        rrnd = torch.rand(nlayer, Nfeatures, nhid) * (1e-3)

        self.KN1 = nn.Parameter(identityInit(self.KN1) + rrnd)

        # if Nfeatures != nhid:
        #    self.interClosing = nn.Parameter(torch.rand(nlayer, Nfeatures, nhid) * stdvp)

        if self.realVarlet:
            self.KN1 = nn.Parameter(torch.rand(nlayer, nhid, 2 * Nfeatures) * stdvp)
            self.KE1 = nn.Parameter(torch.rand(nlayer, nhid, 2 * Nfeatures) * stdvp)

        if self.mixDynamics:
            # self.alpha = nn.Parameter(torch.rand(nlayer, 1) * stdvp)
            self.alpha = nn.Parameter(-0 * torch.ones(1, 1))

        self.KN2 = nn.Parameter(torch.rand(nlayer, nhid, 1 * nhid) * stdvp)
        rrnd2 = torch.rand(nlayer, nhid, nhid) * (1e-3)
        self.KN2 = nn.Parameter(identityInit(self.KN2))

        if self.tripleConv:
            self.KN3 = nn.Parameter(torch.rand(nlayer, nopen, 1 * nhid) * stdvp)
            self.KN3 = nn.Parameter(identityInit(self.KN3))
        self.GCNII = GCNII
        if self.GCNII:
            self.convs = torch.nn.ModuleList()
            alpha = 0.1
            theta = 0.5
            shared_weights = True
            for layer in range(self.nlayers):
                self.convs.append(
                    GCN2Conv(nhid, alpha, theta, layer + 1,
                             shared_weights, normalize=False))

            self.lin1 = torch.nn.Linear(nopen, nopen)
            self.lin2 = torch.nn.Linear(nopen, num_output)

        if self.faust:
            self.lin1 = torch.nn.Linear(nopen, nopen)
            self.lin2 = torch.nn.Linear(nopen, num_output)

        self.modelnet = modelnet

        self.PPI = PPI
        if self.modelnet:
            self.mlp = Seq(
                MLP([nopen, nopen]), MLP([nopen, nopen]),
                Lin(nopen, 10))

    def reset_parameters(self):
        # glorot(self.KN1)
        # glorot(self.KN2)

        glorot(self.K1Nopen)
        glorot(self.K2Nopen)
        glorot(self.KNclose)
        if self.realVarlet:
            glorot(self.KE1)
        if self.modelnet:
            glorot(self.mlp)

    def edgeConv(self, xe, K, groups=1):
        if xe.dim() == 4:
            if K.dim() == 2:
                xe = F.conv2d(xe, K.unsqueeze(-1).unsqueeze(-1), groups=groups)
            else:
                xe = conv2(xe, K, groups=groups)
        elif xe.dim() == 3:
            if K.dim() == 2:
                xe = F.conv1d(xe, K.unsqueeze(-1), groups=groups)
            else:
                xe = conv1(xe, K, groups=groups)
        return xe

    def singleLayer(self, x, K, relu=True, norm=False, groups=1, openclose=False):
        if openclose:  # if K.shape[0] != K.shape[1]:
            x = self.edgeConv(x, K, groups=groups)
            if norm:
                x = F.instance_norm(x)
            if relu:
                x = F.relu(x)
            else:
                x = F.tanh(x)
        if not openclose:  # if K.shape[0] == K.shape[1]:
            # x = F.relu(x)
            x = self.edgeConv(x, K, groups=groups)
            if not relu:
                x = F.tanh(x)
            else:
                x = F.relu(x)
            if norm:
                # x = F.layer_norm(x, x.shape)
                beta = torch.norm(x)
                x = beta * tv_norm(x)
            x = self.edgeConv(x, K.t(), groups=groups)
            # F.relu(x)
        return x

    def newDoubleLayer(self, x, K1, K2):
        x = K1(x)
        # x = F.layer_norm(x, x.shape)
        x = torch.tanh(x)
        if self.dropout:
            # print("self.training:", self.training)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = K2(x)

        return x

    def doubleLayer(self, x, K1, K2):
        x = self.edgeConv(x, K1)
        if not self.dropout:
            x = F.layer_norm(x, x.shape)
            x = torch.tanh(x)
        else:
            x = F.dropout(x, p=self.dropout, training=self.training)
            # x = F.layer_norm(x, x.shape)
            x = torch.relu(x)
        x = self.edgeConv(x, K2)
        x = F.relu(x)

        return x

    def finalDoubleLayer(self, x, K1, K2):
        x = F.tanh(x)
        x = self.edgeConv(x, K1)
        x = F.tanh(x)
        x = self.edgeConv(x, K2)
        x = F.tanh(x)
        x = self.edgeConv(x, K2.t())
        x = F.tanh(x)
        x = self.edgeConv(x, K1.t())
        x = F.tanh(x)
        return x

    def nodeDeriv(self, features, Graph, order=1, edgeSpace=True):
        ## if edgeSpace==True, return features in edge space.
        x = features
        operators = []
        for i in torch.arange(0, order):
            x = Graph.nodeGrad(x)
            if edgeSpace:
                operators.append(x)

            if i == order - 1:
                break

            x = Graph.edgeDiv(x)
            if not edgeSpace:
                operators.append(x)

        if edgeSpace:
            out = x
        else:
            out = Graph.edgeDiv(x)
        operators.append(out)
        return operators

    def saveOperatorImages(self, operators):
        for i in torch.arange(0, len(operators)):
            op = operators[i]
            op = op.detach().squeeze().cpu()  # .numpy()

            plt.figure()
            img = op.reshape(32, 32)
            img = img / img.max()
            plt.imshow(img)
            plt.colorbar()
            plt.show()
            plt.savefig('plots/operator' + str(i) + '.jpg')
            plt.close()

    def savePropagationImage(self, xn, Graph, i=0, minv=None, maxv=None):
        plt.figure()
        print("xn shape:", xn.shape)
        img = xn.clone().detach().squeeze().reshape(32, 32).cpu().numpy()
        # img = img / img.max()
        if (maxv is not None) and (minv is not None):
            plt.imshow(img, vmax=maxv, vmin=minv)
        else:
            plt.imshow(img)

        plt.colorbar()
        plt.show()
        # plt.savefig('plots/img_xn_norm_layer_heat_order_nodeDeriv' + str(self.diffOrder) + '_layer'+ str(i)  + '.jpg')
        plt.savefig('plots/layer' + str(i) + '.jpg')

        plt.close()

    def updateGraph(self, Graph, features=None):
        # If features are given - update graph according to feaure space l2 distance
        N = Graph.nnodes
        I = Graph.iInd
        J = Graph.jInd
        edge_index = torch.cat([I.unsqueeze(0), J.unsqueeze(0)], dim=0)
        if features is not None:
            features = features.squeeze()
            D = torch.relu(torch.sum(features ** 2, dim=0, keepdim=True) + \
                           torch.sum(features ** 2, dim=0, keepdim=True).t() - \
                           2 * features.t() @ features)
            D = D / D.std()
            D = torch.exp(-2 * D)
            w = D[I, J]
            Graph = GO.graph(I, J, N, W=w, pos=None, faces=None)

        else:
            [edge_index, edge_weights] = gcn_norm(edge_index)  # Pre-process GCN normalization.
            I = edge_index[0, :]
            J = edge_index[1, :]
            # deg = self.getDegreeMat(Graph)
            Graph = GO.graph(I, J, N, W=edge_weights, pos=None, faces=None)

        return Graph, edge_index

    def getDegreeMat(self, Graph):
        N = Graph.nnodes
        I = Graph.iInd
        J = Graph.jInd
        edge_index = torch.cat([I.unsqueeze(0), J.unsqueeze(0)], dim=0)
        edge_weight = torch.ones((edge_index.size(1),), dtype=torch.float,
                                 device=edge_index.device)
        deg = scatter_add(edge_weight, J, dim=0, dim_size=N)

        return deg

    def forward(self, xn, Graph, data=None, xe=None):
        # Opening layer
        # xn = [B, C, N]
        # xe = [B, C, N, N] or [B, C, E]
        # Opening layer

        if not self.faust:
            [Graph, edge_index] = self.updateGraph(Graph)
        # if self.faust:
        #    xn = torch.cat([xn, Graph.edgeDiv(xe)], dim=1)

        debug = False

        if debug:
            xnnorm = torch.norm(xn, dim=1)
            vmin = xnnorm.min().detach().numpy()
            vmax = xnnorm.max().detach().numpy()
            xnnorm = torch.norm(xn, dim=1)
            # vmin = xnnorm.min().detach().numpy()
            # vmax = xnnorm.max().detach().numpy()

            saveMesh(xn.squeeze().t(), Graph.faces, Graph.pos, -1, vmax=vmax, vmin=vmin)

        if self.realVarlet:
            xe = Graph.nodeGrad(xn)
            if self.dropout:
                xe = F.dropout(xe, p=self.dropout, training=self.training)
            xe = self.singleLayer(xe, self.K2Nopen, relu=True)

        if self.dropout:
            xn = F.dropout(xn, p=self.dropout, training=self.training)

        if 1 == 0:
            plt.figure()
            print("xn shape:", xn.shape)
            img = xn.clone().detach().squeeze()[0, :].cpu().numpy().reshape(32, 32, order='C')
            print("img shape:", img.shape)
            print("frasures:", img.squeeze()[0, :].squeeze())

            # img = img / img.max()
            plt.imshow(img)
            plt.colorbar()
            plt.show()
            plt.savefig('plots/img_xn_norm_layer_verlet' + str(0) + 'order_nodeDeriv' + str(0) + '.jpg')
            plt.close()

        xn = self.singleLayer(xn, self.K1Nopen, relu=True, openclose=True, norm=False)
        # xn = F.normalize(xn)
        # xn = self.singleLayer(xn, self.K2Nopen, relu=True, openclose=True)

        x0 = xn.clone()
        debug = False
        if debug:
            xnnorm = torch.norm(xn, dim=1)
            # vmin = xnnorm.min().detach().numpy()
            # vmax = xnnorm.max().detach().numpy()
            image = False
            if image:
                plt.figure()
                print("xn shape:", xn.shape)
                img = xn.clone().detach().squeeze().cpu().numpy().reshape(32, 32)
                minv = img.min()
                maxv = img.max()
                # img = img / img.max()
                plt.imshow(img, vmax=maxv, vmin=minv)
                plt.colorbar()
                plt.show()
                plt.savefig('plots/img_xn_norm_layer_verlet' + str(1) + 'order_nodeDeriv' + str(0) + '.jpg')
                plt.close()
            else:
                saveMesh(xn.squeeze().t(), Graph.faces, Graph.pos, 0, vmax=vmax, vmin=vmin)

        if 1 == 0:
            deg = self.getDegreeMat(Graph)
            print("deg:", deg)
            print("deg shape:", deg.shape)
            print("mean deg:", deg.mean())
            print("std deg:", deg.std())
            exit()

        xn_old = x0
        nlayers = self.nlayers
        for i in range(nlayers):

            if self.graphUpdate is not None:
                if i % self.graphUpdate == self.graphUpdate - 1:  # update graph
                    # I, J = getConnectivity(xn.squeeze(0))
                    # Graph = GO.graph(I, J, N)
                    Graph, edge_index = self.updateGraph(Graph, features=xn)
                    dxe = Graph.nodeAve(xn)
            # lapX = Graph.nodeLap(xn)

            # operators = self.nodeDeriv(xn, Graph, order=2, edgeSpace=False)
            # if debug and image:
            #    self.saveOperatorImages(operators)
            if self.realVarlet:
                gradX = Graph.nodeGrad(xn)
                intX = Graph.nodeAve(xn)
                dxe = torch.cat([intX, gradX], dim=1)
                if self.dropout:
                    dxe = F.dropout(dxe, p=self.dropout, training=self.training)
                dxe = (self.singleLayer(dxe, self.KE1[i], relu=False))
                xe = (xe + self.h * dxe)

                divE = Graph.edgeDiv(xe)
                aveE = Graph.edgeAve(xe, method='ave')
                dxn = torch.cat([aveE, divE], dim=1)
                if self.dropout:
                    dxn = F.dropout(dxn, p=self.dropout, training=self.training)
                dxn = F.tanh(self.singleLayer(dxn, self.KN1[i], relu=False))
                xn = (xn + self.h * dxn)
            if not self.realVarlet:
                if self.varlet:
                    # Define operators:
                    # gradX = Graph.nodeGrad(xn)
                    # nodalGradX = Graph.edgeAve(gradX, method='ave')
                    # dxn = torch.cat([xn, nodalGradX], dim=1)
                    # dxn = nodalGradX

                    gradX = Graph.nodeGrad(xn)
                    # gradXtmp = F.tanh(Graph.edgeDiv(gradX))
                    # gradXtmp = Graph.nodeGrad(gradXtmp)
                    # gradX = gradX
                    # dxe = gradX

                # else:
                #    dxn = torch.cat([xn, intX, gradX], dim=1)

                if self.dropout:
                    if self.varlet:
                        gradX = F.dropout(gradX, p=self.dropout, training=self.training)
                        # intX = F.dropout(intX, p=self.dropout, training=self.training)

                        # lapX = F.dropout(lapX, p=self.dropout, training=self.training)
                # dxn = self.doubleLayer(dxn, self.KN1[i], self.KN2[i])
                # dxe = F.tanh(self.singleLayer(dxe, self.KN2[i], relu=False))
                # dxe = Graph.edgeDiv(dxe)

                # that's the best for cora etc
                if self.varlet and not self.gated:
                    efficient = True
                    if efficient:
                        if not self.doubleConv:
                            # gradSq = gradX * gradX #Graph.nodeAve(xn)
                            # gradX = torch.cat([gradX, gradSq], dim=1)
                            dxn = (self.singleLayer(gradX, self.KN1[i], norm=False, relu=True, groups=1))  # KN2
                            # dxn = (self.singleLayer(dxn, self.interClosing[i], norm=False, relu=False, groups=1))
                        else:
                            dxn = self.finalDoubleLayer(gradX, self.KN1[i], self.KN2[i])
                        dxn = Graph.edgeDiv(dxn)  # + Graph.edgeAve(dxe2, method='ave')

                        if self.tripleConv:
                            dxn = self.singleLayer(dxn, self.KN3[i], norm=False, relu=False)
                    else:
                        if not self.doubleConv:
                            dxe = (self.singleLayer(gradX, self.KN1[i], norm=False, relu=False, groups=1))
                        else:
                            dxe = self.finalDoubleLayer(gradX, self.KN1[i], self.KN2[i])
                        dxn = Graph.edgeDiv(dxe)  # + Graph.edgeAve(dxe2, method='ave')
                        if self.tripleConv:
                            dxn = self.singleLayer(dxn, self.KN3[i], norm=False, relu=False)

                    # dxe2 = (self.singleLayer(gradX, self.KN1[i], norm=False, relu=False))
                    # gradX = self.singleLayer(gradX, self.KN1[i], norm=False, relu=False)
                    # dxn = F.tanh(F.tanh(lapX) + F.tanh(Graph.edgeDiv(gradX)) + F.tanh(Graph.edgeAve(dxe, method='max')))
                    # dxn = (lapX)

                elif self.varlet and self.gated:
                    W = F.tanh(Graph.nodeGrad(self.singleLayer(xn, self.KN2[i], relu=False)))
                    lapX = Graph.nodeLap(xn)
                    dxn = F.tanh(lapX + Graph.edgeDiv(W * Graph.nodeGrad(xn)))
                else:
                    dxn = (self.singleLayer(lapX, self.KN1[i], relu=False))
                    dxn = F.tanh(dxn)

                # dxe = F.relu(self.singleLayer(dxe, self.KN2[i], relu=False))
                # dxn = F.tanh(lapX + Graph.edgeDiv(dxe))
                # dxn = lapX + Graph.edgeAve(dxe, method='max')

                # dxn = F.tanh(Graph.edgeAve(dxe, method='ave') + dxn)
                # dxn = F.tanh(dxn)
                # dxn = F.tanh(Graph.edgeDiv(F.tanh(dxe)) + F.tanh(dxn))
                # dxn = F.tanh(F.tanh(dxn) + Graph.edgeDiv(F.tanh(dxe)) + Graph.edgeAve(F.tanh(dxe), method='max'))
                # dxn = F.tanh(Graph.edgeDiv(F.tanh(dxe)))
                if self.mixDynamics:
                    tmp_xn = xn.clone()
                    # xn_wave = 2 * xn - xn_old - (self.h ** 2) * dxn
                    # xn_heat = (xn - self.h * dxn)

                    beta = F.sigmoid(self.alpha)
                    alpha = 1 - beta
                    # print("heat portion:", alpha)
                    # print("wave portion:", beta)
                    if 1 == 1:
                        alpha = alpha / self.h
                        beta = beta / (self.h ** 2)

                        xn = (2 * beta * xn - beta * xn_old + alpha * xn - dxn) / (beta + alpha)
                    else:
                        alpha = 0.5 * alpha / self.h
                        beta = beta / (self.h ** 2)

                        xn = (2 * beta * xn - beta * xn_old + alpha * xn_old - dxn) / (beta + alpha)
                    xn_old = tmp_xn
                    ##########  FE
                    # (beta)dudtt + alpha*dudt = Lu
                    # beta*((xnn - 2xn + xno)/h**2) + alpha*((xnn - xn)/h) = dxn
                    # betah*((xnn - 2xn + xno)) + alphah(((xnn - xn)) = dxn # alphah= alpha/h, betah =beta/h**2
                    # (betah + alphah)xnn = 2*betah*xn - betah*xno + alphah*xn + dxn
                    # xnn = (2*betah*xn - betah*xno + alphah*xn + dxn) / (betah + alphah)

                    ######## DFF
                    # (beta)dudtt + alpha*dudt = Lu
                    # beta*((xnn - 2xn + xno)/h**2) + 0.5*alpha*((xnn - xno)/h) = dxn
                    # betah*((xnn - 2xn + xno)) + alphah(((xnn - xno)) = dxn # alphah= 0.5*alpha/h, betah =(1-alpha)/h**2
                    # (betah + alphah)xnn = 2*betah*xn - betah*xno + alphah*xno + dxn
                    # xnn = (2*betah*xn - betah*xno + alphah*xno + dxn) / (betah + alphah)

                    # softmax
                    # xn = (1 - F.sigmoid(self.alpha)) * xn_heat + F.sigmoid(self.alpha) * xn_wave
                elif self.wave:
                    tmp_xn = xn.clone()
                    xn = 2 * xn - xn_old - (self.h ** 2) * dxn
                    xn_old = tmp_xn
                else:
                    tmp = xn.clone()
                    # print("xn shape:", xn.shape)
                    # print("dxn shape:", dxn.shape)
                    xn = (xn - self.h * dxn)  # +
                    # xn = (xn_old - self.h * dxn)
                    xn_old = tmp
            # xn = F.conv1d(xn, self.convs1x1[i].unsqueeze(-1))

            if debug:
                if image:
                    self.savePropagationImage(xn, Graph, i + 1, minv=minv, maxv=maxv)
                else:
                    saveMesh(xn.squeeze().t(), Graph.faces, Graph.pos, i + 1, vmax=vmax, vmin=vmin)

        xn = F.dropout(xn, p=self.dropout, training=self.training)
        xn = F.conv1d(xn, self.KNclose.unsqueeze(-1))
        # xn = F.conv1d(xn, self.KNclose2.unsqueeze(-1))
        xn = xn.squeeze().t()
        if self.modelnet:
            out = global_max_pool(xn, data.batch)
            out = self.mlp(out)
            return F.log_softmax(out, dim=-1)

        if self.faust:
            x = F.elu(self.lin1(xn))
            if self.dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lin2(x)
            return F.log_softmax(x, dim=1), F.sigmoid(self.alpha)

        if self.PPI:
            return xn, Graph

        ## Otherwise its citation graph node classification:
        return F.log_softmax(xn, dim=1), Graph  # , F.sigmoid(self.alpha)

        # if self.dropout:
        #     # for cora
        #     x = F.dropout(xn, p=self.dropout, training=self.training)
        #     x = F.relu(self.lin1(xn))
        # else:
        #     # for faust
        #     x = F.elu(self.lin1(xn))
        # # if self.dropout:
        # # x = F.dropout(x, p=0.6, training=self.training)
        # x = self.lin2(x)
        #
        # return F.log_softmax(x, dim=1)

        # return xn, xe


# ----------------------------------------------------------------------------

class graphNetwork_proteins(nn.Module):

    def __init__(self, nNin, nEin, nopen, nhid, nNclose, nlayer, h=0.1, dense=False, varlet=False):
        super(graphNetwork_proteins, self).__init__()

        self.h = h
        self.varlet = varlet
        self.dense = dense
        stdv = 1e-2
        stdvp = 1e-3
        self.K1Nopen = nn.Parameter(torch.randn(nopen, nNin) * stdv)
        self.K2Nopen = nn.Parameter(torch.randn(nopen, nopen) * stdv)
        if dense:
            self.K1Eopen = nn.Parameter(torch.randn(nopen, nEin, 9, 9) * stdv)
            self.K2Eopen = nn.Parameter(torch.randn(nopen, nopen, 9, 9) * stdv)
        else:
            self.K1Eopen = nn.Parameter(torch.randn(nopen, nEin) * stdv)
            self.K2Eopen = nn.Parameter(torch.randn(nopen, nopen) * stdv)

        self.KNclose = nn.Parameter(torch.randn(nNclose, nopen) * stdv)

        if varlet:
            Nfeatures = 2 * nopen
        else:
            Nfeatures = 3 * nopen
        if dense:
            self.KE1 = nn.Parameter(torch.rand(nlayer, nhid, Nfeatures, 9, 9) * stdvp)
            self.KE2 = nn.Parameter(torch.rand(nlayer, nopen, nhid, 9, 9) * stdvp)
        else:
            Id = torch.eye(nhid, Nfeatures).unsqueeze(0)
            Idt = torch.eye(nopen, nhid).unsqueeze(0)

            IdTensor = torch.repeat_interleave(Id, nlayer, dim=0)
            IdTensort = torch.repeat_interleave(Idt, nlayer, dim=0)

            # self.KE1 = nn.Parameter(torch.rand(nlayer, nhid, Nfeatures) * stdvp)
            # self.KE2 = nn.Parameter(torch.rand(nlayer, nopen, nhid) * stdvp)
            self.KE1 = nn.Parameter(IdTensor * stdvp)
            self.KE2 = nn.Parameter(IdTensort * stdvp)

        Id = torch.eye(nhid, Nfeatures).unsqueeze(0)
        Idt = torch.eye(nopen, nhid).unsqueeze(0)
        IdTensor = torch.repeat_interleave(Id, nlayer, dim=0)
        IdTensort = torch.repeat_interleave(Idt, nlayer, dim=0)

        self.KN1 = nn.Parameter(IdTensor * stdvp)
        self.KN2 = nn.Parameter(IdTensort * stdvp)

        # self.KN1 = nn.Parameter(torch.rand(nlayer, nhid, Nfeatures) * stdvp)
        # self.KN2 = nn.Parameter(torch.rand(nlayer, nopen, nhid) * stdvp)

    def edgeConv(self, xe, K):
        if xe.dim() == 4:
            if K.dim() == 2:
                xe = F.conv2d(xe, K.unsqueeze(-1).unsqueeze(-1))
            else:
                xe = conv2(xe, K)
        elif xe.dim() == 3:
            if K.dim() == 2:
                xe = F.conv1d(xe, K.unsqueeze(-1))
            else:
                xe = conv1(xe, K)
        return xe

    def doubleLayer(self, x, K1, K2):
        x = self.edgeConv(x, K1)
        # x = F.layer_norm(x, x.shape)
        # x = tv_norm(x)
        # x = torch.relu(x)
        x = torch.tanh(x)
        x = self.edgeConv(x, K2)

        return x

    def forward(self, xn, xe, Graph):

        # Opening layer
        # xn = [B, C, N]
        # xe = [B, C, N, N] or [B, C, E]
        # Opening layer
        xn = self.doubleLayer(xn, self.K1Nopen, self.K2Nopen)
        xe = self.doubleLayer(xe, self.K1Eopen, self.K2Eopen)

        nlayers = self.KE1.shape[0]

        for i in range(nlayers):
            # gradX = torch.exp(-torch.abs(Graph.nodeGrad(xn)))
            gradX = Graph.nodeGrad(xn)
            intX = Graph.nodeAve(xn)
            if self.varlet:
                dxe = torch.cat([intX, gradX], dim=1)
            else:
                dxe = torch.cat([intX, xe, gradX], dim=1)

            dxe = self.doubleLayer(dxe, self.KE1[i], self.KE2[i])
            # dxe = F.layer_norm(dxe, dxe.shape)
            # dxe = tv_norm(dxe)

            # dxe = torch.relu(dxe)
            if self.varlet:
                xe = xe + self.h * dxe
                flux = xe
            else:
                flux = dxe
            divE = Graph.edgeDiv(flux)
            aveE = Graph.edgeAve(flux, method='ave')

            if self.varlet:
                dxn = torch.cat([aveE, divE], dim=1)
            else:
                dxn = torch.cat([aveE, divE, xn], dim=1)

            dxn = self.doubleLayer(dxn, self.KN1[i], self.KN2[i])

            xn = xn - self.h * dxn
            if self.varlet == False:
                xe = xe - self.h * dxe

        xn = F.conv1d(xn, self.KNclose.unsqueeze(-1))

        return xn, xe


# ------------------------------------------------------------


class graphLayer(nn.Module):
    def __init__(self, nNin, nopen, nhid, h=0.1, dense=False, varlet=False, wave=True,
                 dropOut=False,
                 graphUpdate=None, gated=False, mixDyamics=False, doubleConv=False):
        super(graphLayer, self).__init__()

        self.wave = wave
        if not wave:
            self.heat = True
        else:
            self.heat = False
        self.mixDynamics = mixDyamics
        self.h = h
        self.varlet = varlet
        self.dense = dense
        self.doubleConv = doubleConv
        self.graphUpdate = graphUpdate
        self.gated = gated
        if dropOut > 0.0:
            self.dropout = dropOut
        else:
            self.dropout = False
        stdv = 1e-2
        stdvp = 1e-2
        if varlet:
            Nfeatures = 1 * nopen
        else:
            Nfeatures = 1 * nopen

        if self.mixDynamics:
            self.alpha = nn.Parameter(torch.rand(1, 1) * stdvp)

        self.KN2 = nn.Parameter(torch.rand(1, nopen, 1 * nhid) * stdvp)
        self.KN2 = nn.Parameter(identityInit(self.KN2))
        if self.doubleConv:
            self.KN1 = nn.Parameter(torch.rand(1, nopen, 1 * nhid) * stdvp)
            self.KN1 = nn.Parameter(identityInit(self.KN1))

    def reset_parameters(self):
        glorot(self.K1Nopen)
        glorot(self.K2Nopen)
        glorot(self.KNclose)
        if self.realVarlet:
            glorot(self.KE1)
        if self.modelnet:
            glorot(self.mlp)

    def edgeConv(self, xe, K):
        if xe.dim() == 4:
            if K.dim() == 2:
                xe = F.conv2d(xe, K.unsqueeze(-1).unsqueeze(-1))
            else:
                xe = conv2(xe, K)
        elif xe.dim() == 3:
            if K.dim() == 2:
                xe = F.conv1d(xe, K.unsqueeze(-1))
            else:
                xe = conv1(xe, K)
        return xe

    def singleLayer(self, x, K, relu=True, norm=False):
        if K.shape[0] != K.shape[1]:
            x = self.edgeConv(x, K)

        if K.shape[0] == K.shape[1]:
            x = self.edgeConv(x, K)

            x = F.tanh(x)
            if norm:
                # x = F.layer_norm(x, x.shape)
                beta = torch.norm(x)
                x = beta * tv_norm(x)
            x = self.edgeConv(x, K.t())

        if not relu:
            return x
        x = F.relu(x)
        return x

    def finalDoubleLayer(self, x, K1, K2):
        x = F.tanh(x)
        x = self.edgeConv(x, K1)
        x = F.tanh(x)
        x = self.edgeConv(x, K2)
        x = F.tanh(x)
        x = self.edgeConv(x, K2.t())
        x = F.tanh(x)
        x = self.edgeConv(x, K1.t())
        x = F.tanh(x)
        return x

    def forward(self, xn, xn_old, I, J, N, W, data=None):
        Graph = GO.graph(I, J, N, W=W, pos=None, faces=None)

        if self.dropout:
            xn = F.dropout(xn, p=self.dropout, training=self.training)

        if self.varlet:
            gradX = Graph.nodeGrad(xn)

        if self.dropout:
            if self.varlet:
                gradX = F.dropout(gradX, p=self.dropout, training=self.training)

        if self.varlet and not self.gated:
            efficient = True
            if efficient:
                if not self.doubleConv:
                    dxn = (self.singleLayer(gradX, self.KN2[0], norm=False, relu=False))
                else:
                    dxn = self.finalDoubleLayer(gradX, self.KN1[0], self.KN2[0])
                dxn = Graph.edgeDiv(dxn)  # + Graph.edgeAve(dxe2, method='ave')
            else:
                dxe = (self.singleLayer(gradX, self.KN2[0], norm=False, relu=False))
                dxn = Graph.edgeDiv(dxe)  # + Graph.edgeAve(dxe2, method='ave')

        elif self.varlet and self.gated:
            W = F.tanh(Graph.nodeGrad(self.singleLayer(xn, self.KN2[0], relu=False)))
            lapX = Graph.nodeLap(xn)
            dxn = F.tanh(lapX + Graph.edgeDiv(W * Graph.nodeGrad(xn)))
        if self.mixDynamics:
            tmp_xn = xn.clone()
            xn_wave = 2 * xn - xn_old - (self.h ** 2) * dxn
            xn_heat = (xn - self.h * dxn)
            xn_old = tmp_xn

            xn = (1 - F.sigmoid(self.alpha[0])) * xn_wave + F.sigmoid(self.alpha[0]) * xn_heat
        elif self.wave:
            tmp_xn = xn.clone()
            xn = 2 * xn - xn_old - (self.h ** 2) * dxn
            xn_old = tmp_xn
        else:
            xn = (xn - self.h * dxn)

        return xn, xn_old


import torch.utils.checkpoint as checkpoint
from torch.autograd import Variable


class graphNetwork_seq(nn.Module):

    def __init__(self, nNin, nopen, nhid, nNclose, nlayer, h=0.1, dense=False, varlet=False, wave=True,
                 diffOrder=1, num_output=1024, dropOut=False, modelnet=False, faust=False, GCNII=False,
                 graphUpdate=None, PPI=False, gated=False, realVarlet=False, mixDyamics=False, doubleConv=False):
        super(graphNetwork_seq, self).__init__()
        self.wave = wave
        self.realVarlet = realVarlet
        if not wave:
            self.heat = True
        else:
            self.heat = False
        self.mixDynamics = mixDyamics
        self.h = h
        self.varlet = varlet
        self.dense = dense
        self.diffOrder = diffOrder
        self.num_output = num_output
        self.graphUpdate = graphUpdate
        self.gated = gated
        if dropOut > 0.0:
            self.dropout = dropOut
        else:
            self.dropout = False
        self.nlayers = nlayer
        self.graph_convs = nn.ModuleList()

        for i in range(0, nlayer):
            self.graph_convs.append(graphLayer(nNin, nopen, nhid, h=h, dense=False, varlet=varlet, wave=wave,
                                               dropOut=dropOut,
                                               graphUpdate=None, gated=gated, mixDyamics=mixDyamics,
                                               doubleConv=doubleConv))

        stdv = 1e-2
        stdvp = 1e-2
        self.K1Nopen = nn.Parameter(torch.randn(nopen, nNin) * stdv)
        self.KNclose = nn.Parameter(torch.randn(num_output, nopen) * stdv)
        self.modelnet = modelnet
        self.faust = faust
        self.PPI = PPI
        if self.modelnet:
            self.mlp = Seq(
                MLP([nopen, nopen]), MLP([nopen, nopen]),
                Lin(nopen, 10))

    def reset_parameters(self):
        # glorot(self.KN1)
        # glorot(self.KN2)

        glorot(self.K1Nopen)
        # glorot(self.K2Nopen)
        glorot(self.KNclose)
        if self.realVarlet:
            glorot(self.KE1)
        if self.modelnet:
            glorot(self.mlp)

    def edgeConv(self, xe, K):
        if xe.dim() == 4:
            if K.dim() == 2:
                xe = F.conv2d(xe, K.unsqueeze(-1).unsqueeze(-1))
            else:
                xe = conv2(xe, K)
        elif xe.dim() == 3:
            if K.dim() == 2:
                xe = F.conv1d(xe, K.unsqueeze(-1))
            else:
                xe = conv1(xe, K)
        return xe

    def singleLayer(self, x, K, relu=True, norm=False):
        if K.shape[0] != K.shape[1]:
            x = self.edgeConv(x, K)

        if K.shape[0] == K.shape[1]:
            x = self.edgeConv(x, K)

            x = F.tanh(x)
            if norm:
                # x = F.layer_norm(x, x.shape)
                beta = torch.norm(x)
                x = beta * tv_norm(x)
            x = self.edgeConv(x, K.t())

        if not relu:
            return x
        x = F.relu(x)
        return x

    def updateGraph(self, Graph, features=None):
        # If features are given - update graph according to feaure space l2 distance
        N = Graph.nnodes
        I = Graph.iInd
        J = Graph.jInd
        edge_index = torch.cat([I.unsqueeze(0), J.unsqueeze(0)], dim=0)
        if features is not None:
            features = features.squeeze()
            D = torch.relu(torch.sum(features ** 2, dim=0, keepdim=True) + \
                           torch.sum(features ** 2, dim=0, keepdim=True).t() - \
                           2 * features.t() @ features)
            D = D / D.std()
            D = torch.exp(-2 * D)
            w = D[I, J]
            Graph = GO.graph(I, J, N, W=w, pos=None, faces=None)

        else:
            [edge_index, edge_weights] = gcn_norm(edge_index)  # Pre-process GCN normalization.
            I = edge_index[0, :]
            J = edge_index[1, :]
            # deg = self.getDegreeMat(Graph)
            Graph = GO.graph(I, J, N, W=edge_weights, pos=None, faces=None)

        return Graph, edge_index

    def run_function(self, start, end):
        def custom_forward(*inputs):
            xn = inputs[0]
            xn_old = inputs[1]
            I = inputs[2]
            J = inputs[3]
            N = inputs[4]
            W = inputs[5]
            for i in range(start, end):
                xn, xn_old = self.graph_convs[i](xn, xn_old, I, J, N, W)
            return xn, xn_old

        return custom_forward

    def forward(self, xn, Graph, data=None, segments=4):
        [Graph, edge_index] = self.updateGraph(Graph)
        I = Graph.iInd
        J = Graph.jInd
        N = torch.tensor(Graph.nnodes, dtype=torch.int)
        W = Graph.W
        if self.realVarlet:
            xe = Graph.nodeGrad(xn)
            if self.dropout:
                xe = F.dropout(xe, p=self.dropout, training=self.training)
            xe = self.singleLayer(xe, self.K2Nopen, relu=True)

        if self.dropout:
            xn = F.dropout(xn, p=self.dropout, training=self.training)
        xn = self.singleLayer(xn, self.K1Nopen, relu=True)
        x0 = xn.clone()

        xn_old = x0
        nlayers = self.nlayers

        segment_size = len(self.graph_convs) // segments

        for start in range(0, segment_size * (segments), segment_size):
            end = start + segment_size
            # Note that if there are multiple inputs, we pass them as as is without
            # wrapping in a tuple etc.

            [xn, xn_old] = checkpoint.checkpoint(
                self.run_function(start, end), xn, xn_old, I, J, N, W)

        xn = F.dropout(xn, p=self.dropout, training=self.training)
        xn = F.conv1d(xn, self.KNclose.unsqueeze(-1))

        xn = xn.squeeze().t()
        if self.modelnet:
            out = global_max_pool(xn, data.batch)
            out = self.mlp(out)
            return F.log_softmax(out, dim=-1)

        if self.faust:
            x = F.elu(self.lin1(xn))
            if self.dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lin2(x)
            return F.log_softmax(x, dim=1)

        if self.PPI:
            return xn, Graph

        ## Otherwise its citation graph node classification:
        return F.log_softmax(xn, dim=1), Graph

    # ------------faust ##############


class graphNetwork_faust(nn.Module):

    def __init__(self, nNin, nEin, nopen, nhid, nNclose, nlayer, h=0.1, dense=False, varlet=False, wave=True,
                 diffOrder=1, num_nodes=1024, mixDynamics=False):
        super(graphNetwork_faust, self).__init__()
        self.wave = wave
        if not wave:
            self.heat = True
        else:
            self.heat = False
        self.h = h
        self.mixDynamics = mixDynamics
        self.varlet = varlet
        self.dense = dense
        self.diffOrder = diffOrder
        self.num_nodes = num_nodes
        stdv = 1e-1
        stdvp = 1e-1
        self.K1Nopen = nn.Parameter(torch.randn(nopen, nNin) * stdv)
        self.K2Nopen = nn.Parameter(torch.randn(nopen, nopen) * stdv)
        if dense:
            self.K1Eopen = nn.Parameter(torch.randn(nopen, nEin, 9, 9) * stdv)
            self.K2Eopen = nn.Parameter(torch.randn(nopen, nopen, 9, 9) * stdv)
        else:
            self.K1Eopen = nn.Parameter(torch.randn(nopen, nEin) * stdv)
            self.K2Eopen = nn.Parameter(torch.randn(nopen, nopen) * stdv)

        self.KNclose = nn.Parameter(torch.randn(nNclose, nopen) * stdv)
        if self.mixDynamics:
            # self.alpha = nn.Parameter(torch.rand(nlayer, 1) * stdvp)
            self.alpha = nn.Parameter(-0 * torch.ones(1, 1))
        else:
            self.alpha = nn.Parameter(-0 * torch.ones(1, 1))
        if varlet:
            Nfeatures = 2 * nopen
        else:
            Nfeatures = 3 * nopen
        if dense:
            self.KE1 = nn.Parameter(torch.rand(nlayer, nhid, Nfeatures, 9, 9) * stdvp)
            self.KE2 = nn.Parameter(torch.rand(nlayer, nopen, nhid, 9, 9) * stdvp)
        else:
            self.KE1 = nn.Parameter(torch.rand(nlayer, nhid, Nfeatures) * stdvp)
            self.KE2 = nn.Parameter(torch.rand(nlayer, nopen, nhid) * stdvp)

        self.KN1 = nn.Parameter(torch.rand(nlayer, nhid, Nfeatures) * stdvp)
        self.KN2 = nn.Parameter(torch.rand(nlayer, nopen, nhid) * stdvp)

        self.lin1 = torch.nn.Linear(nopen, 256)
        self.lin2 = torch.nn.Linear(256, self.num_nodes)

    def edgeConv(self, xe, K):
        if xe.dim() == 4:
            if K.dim() == 2:
                xe = F.conv2d(xe, K.unsqueeze(-1).unsqueeze(-1))
            else:
                xe = conv2(xe, K)
        elif xe.dim() == 3:
            if K.dim() == 2:
                xe = F.conv1d(xe, K.unsqueeze(-1))
            else:
                xe = conv1(xe, K)
        return xe

    def doubleLayer(self, x, K1, K2):
        x = self.edgeConv(x, K1)
        x = F.layer_norm(x, x.shape)
        x = torch.tanh(x)
        x = self.edgeConv(x, K2)
        # x = torch.tanh(x)
        return x

    def nodeDeriv(self, features, Graph, order=1, edgeSpace=True):
        ## if edgeSpace==True, return features in edge space.
        x = features
        operators = []
        for i in torch.arange(0, order):
            x = Graph.nodeGrad(x)
            if edgeSpace:
                operators.append(x)

            # if i == order - 1:
            #    break

            x = Graph.edgeDiv(x)
            if not edgeSpace:
                operators.append(x)

        if edgeSpace:
            out = x
        else:
            out = Graph.edgeAve(x, method='ave')
        operators.append(out)
        return operators

    def saveOperatorImages(self, operators):
        for i in torch.arange(0, len(operators)):
            op = operators[i]
            op = op.detach().squeeze().cpu()  # .numpy()

            plt.figure()
            img = op.reshape(32, 32)
            img = img / img.max()
            plt.imshow(img)
            plt.colorbar()
            plt.show()
            plt.savefig('plots/operator' + str(i) + '.jpg')
            plt.close()

    def savePropagationImage(self, xn, xe, dxe, Graph, i=0):
        plt.figure()
        img = xn.clone().detach().squeeze().reshape(32, 32).cpu().numpy()
        # img = img / img.max()
        plt.imshow(img)
        plt.colorbar()
        plt.show()
        plt.savefig('plots/img_xn_norm_layer_heat' + str(i) + 'order_nodeDeriv' + str(self.diffOrder) + '.jpg')
        plt.close()

        divE = Graph.edgeDiv(dxe)
        plt.figure()
        img = divE.clone().detach().squeeze().reshape(32, 32).cpu().numpy()
        # img = img / img.max()
        plt.imshow(img)
        plt.colorbar()
        plt.show()
        plt.savefig('plots/img_xe_div_norm_layer_heat' + str(i) + 'order_nodeDeriv' + str(self.diffOrder) + '.jpg')
        plt.close()

    def forward(self, xn, xe, Graph):

        # Opening layer
        # xn = [B, C, N]
        # xe = [B, C, N, N] or [B, C, E]
        # Opening layer
        saveMesh(xn.squeeze().t(), Graph.faces, Graph.pos, -1)
        if not self.wave:
            xn = Graph.edgeAve(xe, method="ave")
        xn = self.doubleLayer(xn, self.K1Nopen, self.K2Nopen)
        xe = self.doubleLayer(xe, self.K1Eopen, self.K2Eopen)

        debug = True
        if debug:
            image = False
            if image:
                plt.figure()
                img = xn.clone().detach().squeeze().reshape(32, 32).cpu().numpy()
                img = img / img.max()
                plt.imshow(img)
                plt.colorbar()
                plt.show()
                plt.savefig('plots/img_xn_norm_layer_verlet' + str(0) + 'order_nodeDeriv' + str(0) + '.jpg')
                plt.close()
            else:
                saveMesh(xn.squeeze().t(), Graph.faces, Graph.pos, 0)

        N = Graph.nnodes
        nlayers = self.KE1.shape[0]
        xn_old = xn.clone()
        xe_old = xe.clone()
        for i in range(nlayers):
            if i % 200 == 199:  # update graph
                I, J = getConnectivity(xn.squeeze(0))
                Graph = GO.graph(I, J, N)
            tmp_node = xn.clone()
            tmp_edge = xe.clone()

            gradX = Graph.nodeGrad(xn)
            intX = Graph.nodeAve(xn)

            # operators = self.nodeDeriv(xn, Graph, order=self.diffOrder, edgeSpace=True)
            # if debug and image:
            #    self.saveOperatorImages(operators)

            if self.varlet:
                dxe = torch.cat([intX, gradX], dim=1)
            else:
                dxe = torch.cat([intX, xe, gradX], dim=1)

            dxe = self.doubleLayer(dxe, self.KE1[i], self.KE2[i])

            # dxe = F.layer_norm(dxe, dxe.shape)
            if self.mixDynamics:
                xe = xe + self.h * dxe

                beta = F.sigmoid(self.alpha)
                alpha = 1 - beta
                # print("heat portion:", alpha)
                # print("wave portion:", beta)
                alpha = alpha / self.h
                beta = beta / (self.h ** 2)

                xe = beta * xe + alpha * dxe
                divE = Graph.edgeDiv(xe)
                aveE = Graph.edgeAve(xe, method='ave')
            elif self.wave:
                xe = xe + self.h * dxe
                print("xe shape:", xe.shape)
                divE = Graph.edgeDiv(xe)
                aveE = Graph.edgeAve(xe, method='ave')

            elif self.heat:
                dxe = torch.tanh(dxe)
                divE = Graph.edgeDiv(dxe)
                aveE = Graph.edgeAve(dxe, method='ave')

            if self.varlet:
                dxn = torch.cat([aveE, divE], dim=1)
            else:
                dxn = torch.cat([aveE, divE, xn], dim=1)

            dxn = self.doubleLayer(dxn, self.KN1[i], self.KN2[i])

            if self.wave:
                xn = xn + self.h * dxn
            else:
                xn = xn - self.h * dxn

            if debug:
                if image:
                    self.savePropagationImage(xn, xe, dxe, Graph, i + 1)
                else:
                    saveMesh(xn.squeeze().t(), Graph.faces, Graph.pos, i + 1)

        xn = F.conv1d(xn, self.KNclose.unsqueeze(-1))
        xn = xn.squeeze().t()
        x = F.elu(self.lin1(xn))
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1), F.sigmoid(self.alpha)

        # return xn, xe


Test = False
if Test:
    nNin = 20
    nEin = 3
    nNopen = 32
    nEopen = 16
    nEhid = 128
    nNclose = 3
    nEclose = 2
    nlayer = 18
    model = graphNetwork(nNin, nEin, nNopen, nEopen, nEhid, nNclose, nEclose, nlayer)

    L = 55
    xn = torch.zeros(1, nNin, L)
    xn[0, :, 23] = 1
    xe = torch.ones(1, nEin, L, L)

    G = GO.dense_graph(L)

    xnout, xeout = model(xn, xe, G)

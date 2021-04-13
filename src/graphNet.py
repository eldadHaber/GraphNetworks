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
    from src.inits import glorot
except:
    import graphOps as GO
    from batchGraphOps import getConnectivity
    from mpl_toolkits.mplot3d import Axes3D
    from utils import saveMesh, h_swish
    from inits import glorot

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


def tv_norm(X, eps=1e-3):
    X = X - torch.mean(X, dim=1, keepdim=True)
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
        stdv = 1e-3
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

            operators = self.nodeDeriv(xn, Graph, order=self.diffOrder, edgeSpace=True)
            if debug and image:
                self.saveOperatorImages(operators)

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




class graphNetwork_nodesOnly(nn.Module):

    def __init__(self, nNin, nopen, nhid, nNclose, nlayer, h=0.1, dense=False, varlet=False, wave=True,
                 diffOrder=1, num_output=1024, dropOut=False, modelnet=False, faust=False, GCNII=False,
                 graphUpdate=None, PPI=False, gated=False, realVarlet=False):
        super(graphNetwork_nodesOnly, self).__init__()
        self.wave = wave
        self.realVarlet = realVarlet
        if not wave:
            self.heat = True
        else:
            self.heat = False
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
        stdv = 1e-3
        stdvp = 1e-3
        self.K1Nopen = nn.Parameter(torch.randn(nopen, nNin) * stdv)
        self.K2Nopen = nn.Parameter(torch.randn(nopen, nopen) * stdv)
        self.KNclose = nn.Parameter(torch.randn(num_output, nopen) * stdv)
        if varlet:
            Nfeatures = 1 * nopen
        else:
            Nfeatures = 1 * nopen

        self.KN1 = nn.Parameter(torch.rand(nlayer, nhid, Nfeatures) * stdvp)
        self.KN2 = nn.Parameter(torch.rand(nlayer, nopen, 1 * nhid) * stdvp)

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

        self.modelnet = modelnet
        self.faust = faust
        self.PPI = PPI
        if self.modelnet:
            self.mlp = Seq(
                MLP([nopen, nopen]), MLP([nopen, nopen]),
                Lin(nopen, 10))

    def reset_parameters(self):
        glorot(self.KN1)
        glorot(self.KN2)
        glorot(self.K1Nopen)
        glorot(self.K2Nopen)
        glorot(self.KNclose)

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
        x = self.edgeConv(x, K)
        if norm:
            x = F.layer_norm(x, x.shape)
        if not relu:
            return x
        x = F.relu(x)
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

    def savePropagationImage(self, xn, Graph, i=0):
        plt.figure()
        print("xn shape:", xn.shape)
        img = xn.clone().detach().squeeze().reshape(32, 32).cpu().numpy()
        # img = img / img.max()
        plt.imshow(img)
        plt.colorbar()
        plt.show()
        #plt.savefig('plots/img_xn_norm_layer_heat_order_nodeDeriv' + str(self.diffOrder) + '_layer'+ str(i)  + '.jpg')
        plt.savefig('plots/layer'+ str(i)  + '.jpg')

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

    def forward(self, xn, Graph, data=None):
        # Opening layer
        # xn = [B, C, N]
        # xe = [B, C, N, N] or [B, C, E]
        # Opening layer

        if self.dropout:
            xn = F.dropout(xn, p=self.dropout, training=self.training)
        if self.varlet:
            xe = Graph.nodeAve(xn)
            xe = self.singleLayer(xe, self.K2Nopen, relu=True)

        xn = self.singleLayer(xn, self.K1Nopen, relu=True)
        x0 = xn.clone()
        debug = False
        if debug:
            image = True
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

        if 1 == 0:
            deg = self.getDegreeMat(Graph)
            print("deg:", deg)
            print("deg shape:", deg.shape)
            print("mean deg:", deg.mean())
            print("std deg:", deg.std())
            exit()

        xn_old = x0
        [Graph, edge_index] = self.updateGraph(Graph)
        nlayers = self.nlayers
        for i in range(nlayers):
            if 1 == 1:
                if self.graphUpdate is not None:
                    if i % self.graphUpdate == self.graphUpdate - 1:  # update graph
                        # I, J = getConnectivity(xn.squeeze(0))
                        # Graph = GO.graph(I, J, N)
                        Graph, edge_index = self.updateGraph(Graph, features=xn)
                        dxe = Graph.nodeAve(xn)
                tmp_xn = xn.clone()

                lapX = Graph.nodeLap(xn)

                # operators = self.nodeDeriv(xn, Graph, order=2, edgeSpace=False)
                # if debug and image:
                #    self.saveOperatorImages(operators)
                if self.realVarlet:
                    gradX = Graph.nodeGrad(xn)
                    intX = Graph.nodeAve(xn)
                    dxe = torch.cat([intX, gradX], dim=1)
                    dxe = F.tanh(self.singleLayer(dxe, self.KE1[i], relu=False))
                    xe = (xe + self.h * dxe)


                    divE = Graph.edgeDiv(xe)
                    aveE = Graph.edgeAve(xe, method='ave')
                    dxn = torch.cat([aveE, divE], dim=1)
                    dxn = F.tanh(self.singleLayer(dxn, self.KN1[i], relu=False))
                    xe = xe + self.h * dxe
                    xn = (xn + self.h * dxn)
                if not self.realVarlet:
                    if self.varlet:
                        # Define operators:
                        # gradX = Graph.nodeGrad(xn)
                        # nodalGradX = Graph.edgeAve(gradX, method='ave')
                        # dxn = torch.cat([xn, nodalGradX], dim=1)
                        # dxn = nodalGradX
                        intX = Graph.nodeAve(xn)
                        dxe = intX  # torch.cat([intX], dim=1)
                    # else:
                    #    dxn = torch.cat([xn, intX, gradX], dim=1)

                    if self.dropout:
                        if self.varlet:
                            # dxn = F.dropout(dxn, p=self.dropout, training=self.training)
                            dxe = F.dropout(dxe, p=self.dropout, training=self.training)
                        else:
                            lapX = F.dropout(lapX, p=self.dropout, training=self.training)
                    # dxn = self.doubleLayer(dxn, self.KN1[i], self.KN2[i])
                    # dxe = F.tanh(self.singleLayer(dxe, self.KN2[i], relu=False))
                    # dxe = Graph.edgeDiv(dxe)

                    # that's the best for cora etc
                    if self.varlet and not self.gated:
                        dxe = F.tanh(self.singleLayer(dxe, self.KN2[i], relu=False)) # + Graph.nodeGrad(lapX)
                        dxn = F.tanh(lapX + Graph.edgeDiv(dxe))
                    elif self.varlet and self.gated:
                        W = F.tanh(Graph.nodeGrad(self.singleLayer(xn, self.KN2[i], relu=False)))
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

                    if self.wave:
                        xn = 2 * xn - xn_old - (self.h ** 2) * dxn
                        xn_old = tmp_xn
                    else:
                        xn = (xn - self.h * dxn)
            if debug:
                if image:
                    self.savePropagationImage(xn, Graph, i + 1)
                else:
                    saveMesh(xn.squeeze().t(), Graph.faces, Graph.pos, i + 1)

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

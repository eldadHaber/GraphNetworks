import os
import random
import networkx as nx

from torch_poly_lr_decay import PolynomialLRDecay
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
import sys
import pickle
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import datetime
import os
from PIL import Image
import warnings
from scipy.io import loadmat

from torch_geometric.data import DataLoader, Data
from torch_geometric.nn import global_max_pool
from torch.nn import Sequential as Seq, Dropout, Linear as Lin
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, LeakyReLU as LRU
from torch_cluster import knn
from torch_geometric.typing import PairTensor

from utils_seg import AverageMeter, inter_and_union, evaluate, preprocess, id2label, preprocess_test
import sys
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
from torch_geometric.utils.convert import to_networkx
from torch_geometric.nn import DynamicEdgeConv, global_max_pool, EdgeConv, GCNConv
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, LeakyReLU as LRU


nImg = 1024
nClasses = 19
batch_size = 1
if not "s" in sys.argv:
    dataroot = "/home/cluster/users/erant_group/cityscapes_data/cityscapes_new"
else:
    dataroot = "/home/erant_group/cityscapes_data_jona/cityscapes"

trainset = torchvision.datasets.Cityscapes(root=dataroot, split='train', mode='fine', target_type='semantic',
                                           transforms=preprocess)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=6, drop_last=True)

testset = torchvision.datasets.Cityscapes(root=dataroot, split='val', mode='fine', target_type='semantic',
                                          transforms=preprocess_test)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,
                                         num_workers=6, drop_last=True)


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), BN(channels[i]), ReLU())
        for i in range(1, len(channels))
    ])

class Net(torch.nn.Module):
    def __init__(self, out_channels, k=30, aggr='max'):
        super(Net, self).__init__()

        self.conv1 = DynamicEdgeConv(MLP([2 * 6, 64, 64]), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 64, 64]), k, aggr)
        self.conv3 = DynamicEdgeConv(MLP([2 * 64, 64, 64]), k, aggr)
        self.lin1 = MLP([3 * 64, 1024])

        self.mlp = Seq(MLP([1024, 256]), Dropout(0.5), MLP([256, 128]),
                       Dropout(0.5), Lin(128, out_channels))

    def forward(self, data):
        x, pos, batch = data.x, data.pos, data.batch
        x0 = torch.cat([x, pos], dim=-1)
        x1 = self.conv1(x0, batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)
        out = self.lin1(torch.cat([x1, x2, x3], dim=1))
        out = self.mlp(out)
        return F.log_softmax(out, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(nClasses, k=10).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
numFeatures = 3 + 1 #RGB + SEG

imgsize = nImg*nImg/2
batch_vec = torch.zeros(imgsize, dtype=torch.int)
if batch_size>1:
    for  i in torch.arange(1, batch_size):
        batch_vec = torch.cat([batch_vec, i*torch.ones(imgsize, dtype=torch.int)])

def train():
    model.train()

    total_loss = correct_nodes = total_nodes = 0
    for i, data in enumerate(trainloader):

        data = data.to(device)
        img, seg = data
        features = torch.cat([img, seg], dim=1)
        features = features.view(batch_size, numFeatures, -1)
        gt =
        data = Data(x=features, y = )
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct_nodes += out.argmax(dim=1).eq(data.y).sum().item()
        total_nodes += data.num_nodes

        if (i + 1) % 10 == 0:
            print(f'[{i+1}/{len(trainloader)}] Loss: {total_loss / 10:.4f} '
                  f'Train Acc: {correct_nodes / total_nodes:.4f}')
            total_loss = correct_nodes = total_nodes = 0


img, seg = trainset[0]

cmap = loadmat('./pascal_seg_colormap.mat')['colormap']
cmap = (cmap * 255).astype(np.uint8).flatten().tolist()
cmap_orig = loadmat('./pascal_seg_colormap.mat')['colormap']

import matplotlib

plt.figure()
plot_seg = seg.clone().cpu().numpy().squeeze().astype(np.uint8)
plot_seg = Image.fromarray(plot_seg)
plot_seg.putpalette(cmap)
plt.imshow(plot_seg)
plt.show()

img, seg = trainset[0]
seg = seg.float()
for i in torch.arange(0, 1):
    seg = F.max_pool2d(seg.unsqueeze(0), 3, stride=2).squeeze()
    plot_seg = seg.clone().cpu().numpy().squeeze().astype(np.uint8)
    plot_seg = Image.fromarray(plot_seg)
    plot_seg.putpalette(cmap)
    plt.imshow(plot_seg)
    plt.show()

batch_vec = torch.zeros(torch.numel(seg), dtype=torch.int64)
xtmp: PairTensor = (seg.flatten().float(), seg.flatten().float())
b = (batch_vec, batch_vec)
k = 20
edge_index = knn(xtmp[0], xtmp[1], k, b[0], b[1],
                 num_workers=6)
I = edge_index[0, :].squeeze()
J = edge_index[1, :].squeeze()

xs = torch.arange(0, seg.shape[0])
ys = torch.arange(0, seg.shape[1])

X, Y = torch.meshgrid(xs, ys)

pos = torch.stack([X, Y])
pos = pos.view(2, -1)
img_Graph = Data(x=seg.flatten(), edge_index=edge_index)
coragraph = to_networkx(img_Graph)
node_labels = seg.flatten().numpy()
top2, _ = torch.topk(torch.from_numpy(node_labels).unique(), k=2)
node_labels[node_labels == top2[0].float().cpu().numpy()] = top2[1] + 1

import matplotlib.pyplot as plt

plt.figure(1, figsize=(20, 20))

nx.draw(coragraph, node_color=node_labels, node_size=30, linewidths=6, vmin=0, vmax=nClasses + 1)
plt.show()

print("edge index:", edge_index)

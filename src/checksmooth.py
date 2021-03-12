import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Dropout, Linear as Lin
from torch_geometric.datasets import ModelNet, FAUST
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import DynamicEdgeConv, global_max_pool, EdgeConv, GCNConv
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, LeakyReLU as LRU

import trimesh
import matplotlib.pyplot as plt



def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), BN(channels[i]), ReLU())
        for i in range(1, len(channels))
    ])

def saveMesh(xn, faces, pos, i=0):
    print("xn shape:", xn.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(pos[:, 0].clone().detach().cpu().numpy(), pos[:, 1].clone().detach().cpu().numpy(),
                   pos[:, 2].clone().detach().cpu().numpy(),
                   c=xn.squeeze(0).norm(dim=1).clone().detach().cpu().numpy())
    fig.colorbar(p)
    plt.savefig(
        "/users/others/eliasof/GraphNetworks/plots/xn_norm_verlet_layer_" + str(i))
    plt.close()

    mesh = trimesh.Trimesh(vertices=pos, faces=faces.t(), process=False)
    colors = xn.squeeze(0).norm(dim=1).clone().detach().cpu().numpy()
    vect_col_map = trimesh.visual.color.interpolate(colors,
                                                    color_map='jet')

    if xn.shape[0] == mesh.vertices.shape[0]:
        print("case 1")
        mesh.visual.vertex_colors = vect_col_map
    elif xn.shape[0] == mesh.faces.shape[0]:
        print("case 2")
        mesh.visual.face_colors = vect_col_map
        smooth = False

    trimesh.exchange.export.export_mesh(mesh,
                                        "/users/others/eliasof/GraphNetworks/plots/xn_norm_verlet_layer_" + str(
                                            i) + ".ply", "ply")


path = '/home/cluster/users/erant_group/faust'
transform = T.FaceToEdge(remove_faces=False)
train_dataset = FAUST(path, True, transform)
train_loader = DataLoader(
    train_dataset, batch_size=1, shuffle=False, num_workers=6)


class Net(torch.nn.Module):
    def __init__(self, out_channels, k=10, aggr='max'):
        super().__init__()
        self.numlayers = 10
        #self.conv1 = EdgeConv(MLP([2 * 3, 3]), aggr)
        #self.conv2 = EdgeConv(MLP([2 * 3, 3]), aggr)
        self.Layers = torch.nn.ModuleList()
        for i in torch.arange(0, self.numlayers):
            self.Layers.append(GCNConv(in_channels=3, out_channels=3))

        self.lin1 = MLP([64 + 64, 64])

        self.mlp = Seq(
            MLP([64, 64]), MLP([64, 64]),
            Lin(64, out_channels))

    def forward(self, data):
        pos, batch = data.pos, data.batch
        xn = torch.zeros(3, pos.shape[0]).float()
        xn[:, 100:300] = 1.0
        print("data.edgeindex:", data.edge_index)
        print("data.pos shape:", data.pos.shape)
        print("xn shape:", xn.shape)
        #xn = data.pos
        saveMesh(xn, data.face, data.pos, 0)
        for i,layer in enumerate(self.Layers):
            xn = layer(xn, data.edge_index)
            saveMesh(xn, data.face, data.pos, i+1)

        exit()
        out = self.lin1(torch.cat([xn, xn], dim=1))
        out = global_max_pool(out, batch)
        out = self.mlp(out)
        return F.log_softmax(out, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(train_dataset.num_classes, k=10).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
import numpy as np


def train():
    model.eval()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(train_dataset)


def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


for epoch in range(1, 201):
    loss = train()
    test_acc = test(test_loader)
    print('Epoch {:03d}, Loss: {:.4f}, Test: {:.4f}'.format(
        epoch, loss, test_acc))
    scheduler.step()

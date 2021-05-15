import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Dropout, Linear as Lin
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import DynamicEdgeConv, global_max_pool, EdgeConv
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, LeakyReLU as LRU

def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), BN(channels[i]), ReLU())
        for i in range(1, len(channels))
    ])

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/ModelNet10_fixed')
#path = '/home/cluster/users/erant_group/moshe/ModelNet10_fixed'
#pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)
pre_transform = T.NormalizeScale()
transform = T.FaceToEdge(remove_faces=False)
train_dataset = ModelNet(path, '10', True, transform, pre_transform)
test_dataset = ModelNet(path, '10', False, transform, pre_transform)
train_loader = DataLoader(
    train_dataset, batch_size=16, shuffle=True, num_workers=6)
test_loader = DataLoader(
    test_dataset, batch_size=16, shuffle=False, num_workers=6)


class Net(torch.nn.Module):
    def __init__(self, out_channels, k=10, aggr='max'):
        super().__init__()

        self.conv1 = EdgeConv(MLP([2 * 3, 64, 64, 64]), aggr)
        self.conv2 = EdgeConv(MLP([2 * 64, 64]), aggr)
        self.lin1 = MLP([64 + 64, 128])

        self.mlp = Seq(
            MLP([128, 128]), Dropout(0.5), MLP([128, 128]), Dropout(0.5),
            Lin(128, out_channels))

    def forward(self, data):
        pos, batch = data.pos, data.batch
        edge_index = data.edge_index
        #x1 = self.conv1(pos, batch)
        #x2 = self.conv2(x1, batch)
        x1 = self.conv1(pos, edge_index)
        x2 = self.conv2(x1, edge_index)
        out = self.lin1(torch.cat([x1, x2], dim=1))
        out = global_max_pool(out, batch)
        out = self.mlp(out)
        return F.log_softmax(out, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(train_dataset.num_classes, k=10).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)


def train():
    model.train()

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
        epoch, loss, test_acc), flush=True)
    scheduler.step()
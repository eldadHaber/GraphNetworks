import os.path as osp

import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_properties('cuda:0'))

from torch.nn import Linear
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch_geometric.datasets import PPI
import torch_geometric.transforms as T
from torch_geometric.nn import GCN2Conv
from torch_geometric.data import DataLoader

dataset = "ppi"
import sys

if "s" in sys.argv:
    path = '/home/eliasof/GraphNetworks/data/' + dataset
else:
    path = '/home/cluster/users/erant_group/moshe/' + dataset
pre_transform = T.Compose([T.GCNNorm(), T.ToSparseTensor()])
pre_transform = T.Compose([T.GCNNorm()])

train_dataset = PPI(path, split='train', pre_transform=None)
val_dataset = PPI(path, split='val', pre_transform=None)
test_dataset = PPI(path, split='test', pre_transform=None)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


class Net(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, alpha, theta,
                 shared_weights=True, dropout=0.0):
        super(Net, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(train_dataset.num_features, hidden_channels))
        self.lins.append(Linear(hidden_channels, train_dataset.num_classes))

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(hidden_channels, alpha, theta, layer + 1,
                         shared_weights, normalize=False))

        self.dropout = dropout

    def forward(self, x, adj_t):
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()

        for conv in self.convs:
            h = F.dropout(x, self.dropout, training=self.training)
            h = conv(h, x_0, adj_t)
            x = h + x
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)

        return x


import sys

if "s" in sys.argv:
    base_path = '/home/eliasof/pdeGraphs/data/'
    import graphOps as GO
    import processContacts as prc
    import utils
    import graphNet as GN
    import pnetArch as PNA


elif "e" in sys.argv:
    base_path = '/home/cluster/users/erant_group/pfold/'
    from src import graphOps as GO
    from src import processContacts as prc
    from src import utils
    from src import graphNet as GN
    from src import pnetArch as PNA


else:
    base_path = '../../../data/'
    from src import graphOps as GO
    from src import processContacts as prc
    from src import utils
    from src import graphNet as GN
    from src import pnetArch as PNA

# Setup the network and its parameters
dataset = 'ppi'

nNin = train_dataset.num_features
nEin = 1
nopen = 2048
nhid = 2048
nNclose = 2048
nlayer = 2
h = 1 / nlayer
dropout = 0.2
# h = 20 / nlayer
print("dataset:", dataset)
print("n channels:", nopen)
print("n layers:", nlayer)
print("h step:", h)
print("dropout:", dropout)
print("wave eq. !")
wave = False
print("Wave = ", wave)
print_files = False
if print_files:
    file2Open = "src/GN_ppi.py"
    print("------------------------------------ Driver file: ------------------------------------")

    f = open(file2Open, "r")
    for line in f:
        print(line, end='', flush=True)

    print("------------------------------------ Graph Networks file: ------------------------------------")
    file2Open = "src/graphNet.py"
    f = open(file2Open, "r")
    for line in f:
        print(line, end='', flush=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GN.graphNetwork_nodesOnly(nNin, nopen, nhid, nNclose, nlayer, h=h, dense=False, varlet=True, wave=wave,
                                  diffOrder=1, num_output=train_dataset.num_classes, dropOut=dropout, PPI=True,
                                  realVarlet=False)
model.reset_parameters()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam([
    dict(params=model.KN1, lr=0.001, weight_decay=0),
    dict(params=model.KN2, lr=0.001, weight_decay=0),
    dict(params=model.K1Nopen, weight_decay=5e-4),
    dict(params=model.KNclose, weight_decay=5e-4)
], lr=0.001)


def train():
    model.train()

    total_loss = total_examples = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        I = data.edge_index[0, :]
        J = data.edge_index[1, :]
        N = data.y.shape[0]

        G = GO.graph(I, J, N, pos=None, faces=None)
        G = G.to(device)
        xn = data.x.t().unsqueeze(0).to(device)
        xe = torch.ones(1, 1, I.shape[0]).to(device)
        [out, G] = model(xn, G)

        loss = criterion(out, data.y)
        loss.backward()

        gKN2 = model.KN2.grad.norm().item()
        #gKN1 = model.KN1.grad.norm().item()
        gKN1 = 0
        gKo = model.K1Nopen.grad.norm().item()
        gKc = model.KNclose.grad.norm().item()
        print("gKo gKN1  gKN2    gKc")
        print("%10.3E   %10.3E   %10.3E   %10.3E" %
              (gKo, gKN1, gKN2, gKc), flush=True)

        optimizer.step()
        total_loss += loss.item() * data.num_nodes
        total_examples += data.num_nodes
    return total_loss / total_examples


@torch.no_grad()
def test(loader):
    model.eval()

    ys, preds = [], []
    for data in loader:
        ys.append(data.y)
        data.to(device)
        I = data.edge_index[0, :]
        J = data.edge_index[1, :]
        N = data.y.shape[0]

        G = GO.graph(I, J, N, pos=None, faces=None)
        G = G.to(device)
        xn = data.x.t().unsqueeze(0)
        xe = torch.ones(1, 1, I.shape[0]).to(device)
        [out, G] = model(xn, G)

        preds.append((out > 0).float().cpu())

    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0


for epoch in range(1, 2001):
    loss = train()
    train_f1 = test(train_dataset)
    val_f1 = test(val_loader)
    test_f1 = test(test_loader)
    print('Epoch: {:02d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'.format(
        epoch, loss, val_f1, test_f1), flush=True)
    print("Train F1:", train_f1, flush=True)

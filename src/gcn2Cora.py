import os.path as osp

import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
#from torch_geometric.nn import GCN2Conv
from src.gcn2conv import GCN2Conv

from torch_geometric.nn.conv.gcn_conv import gcn_norm



dataset = 'Cora'
path = '/home/cluster/users/erant_group/moshe/cora'
transform = T.Compose([T.NormalizeFeatures(), T.ToSparseTensor()])
transform = T.Compose([T.NormalizeFeatures()])

dataset = Planetoid(path, dataset, transform=transform)
data = dataset[0]
#data.adj_t = gcn_norm(data.adj_t)  # Pre-process GCN normalization.
[data.edge_index, data.edge_weights] = gcn_norm(data.edge_index)  # Pre-process GCN normalization.


class Net(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, alpha, theta,
                 shared_weights=True, dropout=0.0):
        super(Net, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(dataset.num_features, hidden_channels))
        self.lins.append(Linear(hidden_channels, dataset.num_classes))

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(hidden_channels, alpha, theta, layer + 1,
                         shared_weights, normalize=False))

        self.dropout = dropout

    def forward(self, x, data, adj_t=None):
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()

        for conv in self.convs:
            x = F.dropout(x, self.dropout, training=self.training)
            #x = conv(x, x_0, adj_t)
            x = conv(x, x_0, data.edge_index, data.edge_weights)

            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)

        return x.log_softmax(dim=-1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(hidden_channels=64, num_layers=64, alpha=0.1, theta=0.5,
            shared_weights=True, dropout=0.6).to(device)
data = data.to(device)
optimizer = torch.optim.Adam([
    dict(params=model.convs.parameters(), weight_decay=0.01),
    dict(params=model.lins.parameters(), weight_decay=5e-4)
], lr=0.01)


def train():
    model.train()
    optimizer.zero_grad()
    #out = model(data.x, data.adj_t)
    out = model(data.x, data)

    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    #pred, accs = model(data.x, data.adj_t).argmax(dim=-1), []
    pred, accs = model(data.x, data).argmax(dim=-1), []

    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs


best_val_acc = test_acc = 0
for epoch in range(1, 1001):
    loss = train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    print(f'Epoch: {epoch:04d}, Loss: {loss:.4f} Train: {train_acc:.4f}, '
          f'Val: {val_acc:.4f}, Test: {tmp_test_acc:.4f}, '
          f'Final Test: {test_acc:.4f}')
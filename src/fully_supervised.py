import os.path as osp

import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCN2Conv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import optuna
import numpy as np
import sys
import process
import utils_gcnii
from torch_geometric.utils import sparse as sparseConvert

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
dataset = 'Cora'

if dataset == 'Cora':
    nNin = 1433
elif dataset == 'CiteSeer':
    nNin = 3703
elif dataset == 'PubMed':
    nNin = 500
nEin = 1
nopen = 64
nhid = 64
nNclose = 64
nlayer = 64
h = 1  # 16 / nlayer

dropout = 0.6
# h = 20 / nlayer
print("dataset:", dataset)
print("n channels:", nopen)
print("n layers:", nlayer)
print("h step:", h)
print("dropout:", dropout)
batchSize = 32

if "s" in sys.argv:
    path = '/home/eliasof/GraphNetworks/data/' + dataset
else:
    path = '/home/cluster/users/erant_group/moshe/' + dataset
transform = T.Compose([T.NormalizeFeatures()])
dataset = Planetoid(path, dataset, transform=transform)
data = dataset[0]
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
data = data.to(device)
realVarlet = False


def train_step(model, optimizer, features, labels, adj, idx_train):
    model.train()
    optimizer.zero_grad()
    I = adj[0, :]
    J = adj[1, :]
    N = labels.shape[0]
    w = torch.ones(adj.shape[1]).to(device)
    G = GO.graph(I, J, N, W=w, pos=None, faces=None)
    G = G.to(device)
    xn = features  # data.x.t().unsqueeze(0)
    xe = torch.ones(1, 1, I.shape[0]).to(device)

    # out = model(xn, xe, G)
    [out, G] = model(xn, G)

    g = G.nodeGrad(out.t().unsqueeze(0))
    eps = 1e-4
    absg = torch.sum(g ** 2, dim=1)
    tvreg = absg.mean()
    # tvreg = torch.norm(G.nodeGrad(out.t().unsqueeze(0)), p=1) / I.shape[0]
    # out = out.squeeze()
    acc_train = utils_gcnii.accuracy(out[idx_train], labels[idx_train].to(device))
    loss_train = F.nll_loss(out[idx_train], labels[idx_train].to(device))
    loss_train.backward()
    optimizer.step()
    return loss_train.item(), acc_train.item()


def test_step(model, features, labels, adj, idx_test):
    model.eval()
    with torch.no_grad():
        I = adj[0, :]
        J = adj[1, :]
        N = labels.shape[0]
        w = torch.ones(adj.shape[1]).to(device)

        G = GO.graph(I, J, N, W=w, pos=None, faces=None)
        G = G.to(device)
        xn = features  # data.x.t().unsqueeze(0)
        xe = torch.ones(1, 1, I.shape[0]).to(device)

        # out = model(xn, xe, G)
        [out, G] = model(xn, G)

        loss_test = F.nll_loss(out[idx_test], labels[idx_test].to(device))
        acc_test = utils_gcnii.accuracy(out[idx_test], labels[idx_test].to(device))
        return loss_test.item(), acc_test.item()


def train(datastr, splitstr):
    adj, features, labels, idx_train, idx_val, idx_test, num_features, num_labels = process.full_load_data(datastr,
                                                                                                           splitstr)
    adj = adj.to_dense()
    print("adj shape:", adj.shape)

    [edge_index, edge_weight] = sparseConvert.dense_to_sparse(adj)
    del adj
    print("edge index shape:", edge_index.shape)

    print("features shape:", features.shape)
    print("labels shhape:", labels.shape)
    print("idx shape:", idx_train.shape)
    edge_index = edge_index.to(device)
    features = features.to(device).t().unsqueeze(0)
    idx_train = idx_train.to(device)
    idx_test = idx_test.to(device)
    labels = labels.to(device)
    #

    model = GN.graphNetwork_nodesOnly(num_features, nopen, nhid, nNclose, nlayer, h=h, dense=False, varlet=True,
                                      wave=False,
                                      diffOrder=1, num_output=dataset.num_classes, dropOut=dropout, gated=False,
                                      realVarlet=realVarlet, mixDyamics=True)
    model = model.to(device)

    optimizer = torch.optim.Adam([
        dict(params=model.KN1, lr=0.00001, weight_decay=0),
        dict(params=model.KN2, lr=0.00001, weight_decay=0),
        dict(params=model.K1Nopen, weight_decay=5e-4),
        dict(params=model.KNclose, weight_decay=5e-4),
        dict(params=model.alpha, lr=0.001, weight_decay=0)
    ], lr=0.01)

    bad_counter = 0
    best = 0
    for epoch in range(1000):
        loss_tra, acc_tra = train_step(model, optimizer, features, labels, edge_index, idx_train)
        loss_val, acc_val = test_step(model, features, labels, edge_index, idx_test)
        if (epoch + 1) % 1 == 0:
            print('Epoch:{:04d}'.format(epoch + 1),
                  'train',
                  'loss:{:.3f}'.format(loss_tra),
                  'acc:{:.2f}'.format(acc_tra * 100),
                  '| test',
                  'loss:{:.3f}'.format(loss_val),
                  'acc:{:.2f}'.format(acc_val * 100))
        if acc_val > best:
            best = acc_val
            # torch.save(model.state_dict(), checkpt_file)
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == 100:
            break
    acc = best

    return acc * 100


acc_list = []
for i in range(10):
    datastr = "texas"
    splitstr = '../splits/' + datastr + '_split_0.6_0.2_' + str(i) + '.npz'
    acc_list.append(train(datastr, splitstr))
    print(i, ": {:.2f}".format(acc_list[-1]))
print("Test acc.:{:.2f}".format(np.mean(acc_list)))

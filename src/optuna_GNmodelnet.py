import os.path as osp

import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, ModelNet
import torch_geometric.transforms as T
from torch_geometric.nn import GCN2Conv
# from src.gcn2conv import GCN2Conv
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_max_pool
from torch.nn import Sequential as Seq, Dropout, Linear as Lin
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, LeakyReLU as LRU
from torch_cluster import knn
from torch_geometric.typing import PairTensor
import optuna
import sys

path = '/home/cluster/users/erant_group/ModelNet10'

if "s" in sys.argv:
    path = '/home/eliasof/GraphNetworks/data/'
    import processContacts as prc
    import utils
    import graphOps as GO
    import graphNet as GN
    import pnetArch as PNA
if "j" in sys.argv:
    path = '/home/ephrathj/GraphNetworks/data/'
    import processContacts as prc
    import utils
    import graphNet as GN
    import pnetArch as PNA
    import graphOps as GO
else:
    from src import graphOps as GO
    from src import processContacts as prc
    from src import utils
    from src import graphNet as GN
    from src import pnetArch as PNA

pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)
train_dataset = ModelNet(path, '10', True, transform, pre_transform)
test_dataset = ModelNet(path, '10', False, transform, pre_transform)
train_loader = DataLoader(
    train_dataset, batch_size=64, shuffle=True, num_workers=6)
test_loader = DataLoader(
    test_dataset, batch_size=64, shuffle=False, num_workers=6)


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), BN(channels[i]), ReLU())
        for i in range(1, len(channels))
    ])


printFiles = False
if "s" in sys.argv or "j" in sys.argv:
    printFiles = True
num_layers = [4, 8]
if printFiles:
    print("**********************************************************************************")
    file2Open = "src/optuna_GNmodelnet.py"
    print("DRIVER CODE:")
    f = open(file2Open, "r")
    for line in f:
        print(line, end='', flush=True)

    print("NETWORKS CODE:")
    file2Open = "src/graphNet.py"
    f = open(file2Open, "r")
    for line in f:
        print(line, end='', flush=True)

    print("**********************************************************************************")
for nlayers in num_layers:
    torch.cuda.synchronize()
    print("Without symmetric conv - not using TransposedConv !!!")
    print("Doing experiment for ", nlayers, " layers!", flush=True)
    torch.cuda.synchronize()


    def objective(trial):
        nEin = 1
        nopen = trial.suggest_categorical('n_channels', [64, 128, 256])
        nNin = 3
        nhid = nopen
        nNclose = nopen
        nlayer = nlayers
        h = 1  # / nlayer
        dropout = 0.0
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        lrGCN = trial.suggest_float("lrGCN", 1e-5, 1e-2, log=True)
        wd = trial.suggest_float("wd", 5e-12, 1e-3, log=True)
        h = trial.suggest_discrete_uniform('h', 1 / (nlayer), 3, q=1 / (nlayer))
        wave = False
        import datetime
        now = datetime.datetime.now()
        filename = 'nopen_' + str(nopen) + 'nhid_' + str(nhid) + 'nlayer_' + str(nlayer) + 'h_' + str(
            h) + 'dropout_' + str(
            dropout) + 'wave_' + str(wave) + "_" + str(now.month) + "_" + str(now.day) + "_" + str(
            now.hour) + "_" + str(
            now.minute) + "_" + str(
            now.second) + '.pth'
        model = GN.graphNetwork_nodesOnly(nNin, nopen, nhid, nNclose, nlayer, h=h, dense=False, varlet=True, wave=wave,
                                          diffOrder=1, num_output=nopen, dropOut=dropout, modelnet=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)

        optimizer = torch.optim.Adam([
            dict(params=model.KN1, lr=lrGCN, weight_decay=0),
            dict(params=model.KN2, lr=lrGCN, weight_decay=0),
            dict(params=model.K1Nopen, weight_decay=wd),
            dict(params=model.KNclose, weight_decay=wd),
            dict(params=model.mlp.parameters(), weight_decay=0)
            # dict(params=model.alpha, lr=0.1, weight_decay=0),
        ], lr=lr)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)

        def train():
            model.train()

            total_loss = 0
            for data in train_loader:
                xtmp: PairTensor = (data.pos, data.pos)
                b = (data.batch, data.batch)
                k = 10
                edge_index = knn(xtmp[0], xtmp[1], k, b[0], b[1],
                                 num_workers=6)
                edge_index = edge_index.to(device)
                I = edge_index[0, :]
                J = edge_index[1, :]
                N = data.pos.shape[0]
                W = torch.ones(N).to(device)
                G = GO.graph(I, J, N, W=W ,pos=None, faces=None)
                G = G.to(device)
                data = data.to(device)
                optimizer.zero_grad()
                xn = data.pos.t().unsqueeze(0).cuda()
                out = model(xn, G, data=data)
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
                xtmp: PairTensor = (data.pos, data.pos)
                b = (data.batch, data.batch)
                k = 10
                edge_index = knn(xtmp[0], xtmp[1], k, b[0], b[1],
                                 num_workers=3)
                edge_index = edge_index.to(device)
                I = edge_index[0, :]
                J = edge_index[1, :]
                N = data.pos.shape[0]
                W = torch.ones(N).to(device)
                G = GO.graph(I, J, N, W=W,pos=None, faces=None)
                G = G.to(device)
                data = data.to(device)
                optimizer.zero_grad()
                xn = data.pos.t().unsqueeze(0).cuda()
                with torch.no_grad():
                    pred = model(xn, G, data=data).max(dim=1)[1]
                correct += pred.eq(data.y).sum().item()
            return correct / len(loader.dataset)

        save_path = '/home/cluster/users/erant_group/moshe/pdegcnCheckpoints/' + filename

        best_test_acc = 0
        patience = 40
        counter = 0
        for epoch in range(1, 201):
            loss = train()
            test_acc = test(test_loader)
            if test_acc >= best_test_acc:
                best_test_acc = test_acc
                counter = 0
                print('Epoch {:03d}, Loss: {:.4f}, Test: {:.4f}'.format(
                    epoch, loss, test_acc))
                # torch.save(model.state_dict(), save_path)

            else:
                counter = counter + 1

            if counter >= patience:
                return best_test_acc

            scheduler.step()
        print("best acc:", best_test_acc, flush=True)
        return best_test_acc


    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

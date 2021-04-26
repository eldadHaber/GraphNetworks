import os.path as osp

import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCN2Conv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import optuna

import sys

if "s" in sys.argv:
    base_path = '/home/eliasof/pFold/data/'
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

num_layers = [2, 4, 8, 16, 32, 64]
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_properties('cuda:0'))

print("**********************************************************************************")
file2Open = "src/optuna_GNcora.py"
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

# Setup the network and its parameters
for nlayers in num_layers:
    torch.cuda.synchronize()
    print("Without symmetric conv - not using TransposedConv !!!")
    print("Doing experiment for ", nlayers, " layers!", flush=True)
    torch.cuda.synchronize()


    def objective(trial):
        dataset = 'CiteSeer'
        if dataset == 'Cora':
            nNin = 1433
        elif dataset == 'CiteSeer':
            nNin = 3703
        elif dataset == 'PubMed':
            nNin = 500
        nEin = 1
        n_channels = 256  # trial.suggest_categorical('n_channels', [64, 128, 256])
        nopen = n_channels
        nhid = n_channels
        nNclose = n_channels
        n_layers = nlayers
        print("DATA SET IS:", dataset)
        # h = 1 / n_layers
        h = trial.suggest_discrete_uniform('h', 1 / (n_layers), 3, q=1 / (n_layers))
        batchSize = 32

        if "s" in sys.argv:
            path = '/home/eliasof/GraphNetworks/data/' + dataset
        else:
            path = '/home/cluster/users/erant_group/moshe/' + dataset
        transform = T.Compose([T.NormalizeFeatures()])
        dataset = Planetoid(path, dataset, transform=transform)
        data = dataset[0]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = data.to(device)
        dropout = trial.suggest_discrete_uniform('dropout', 0.5, 0.8, q=0.1)
        lr = trial.suggest_float("lr", 1e-3, 1e-1, log=True)
        lrGCN = trial.suggest_float("lrGCN", 1e-6, 1e-3, log=True)
        wd = trial.suggest_float("wd", 5e-8, 1e-4, log=True)
        # wdGCN = trial.suggest_float("wdGCN", 1e-10, 1e-2, log=True)
        lr_alpha = trial.suggest_float("lr_alpha", 1e-5, 1e-2, log=True)
        model = GN.graphNetwork_nodesOnly(nNin, nopen, nhid, nNclose, n_layers, h=h, dense=False, varlet=True,
                                          wave=False,
                                          diffOrder=1, num_output=dataset.num_classes, dropOut=dropout, gated=False,
                                          realVarlet=False, mixDyamics=False, doubleConv=True)
        model.reset_parameters()
        model.to(device)
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        optimizer = torch.optim.Adam([
            dict(params=model.KN1, lr=lrGCN, weight_decay=0),
            dict(params=model.KN2, lr=lrGCN, weight_decay=0),
            dict(params=model.K1Nopen, weight_decay=wd),
            dict(params=model.KNclose, weight_decay=wd),
            # dict(params=model.alpha, lr=lr_alpha, weight_decay=0),

        ], lr=lr)

        # optimizer = torch.optim.Adam([
        #     dict(params=model.convs.parameters(), weight_decay=0.01),
        #     dict(params=model.K1Nopen, weight_decay=5e-4),
        #     dict(params=model.KNclose, weight_decay=5e-4)
        # ], lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)

        def train():
            model.train()
            optimizer.zero_grad()

            I = data.edge_index[0, :]
            J = data.edge_index[1, :]
            N = data.y.shape[0]

            features = data.x.squeeze().t()
            D = torch.relu(torch.sum(features ** 2, dim=0, keepdim=True) + \
                           torch.sum(features ** 2, dim=0, keepdim=True).t() - \
                           2 * features.t() @ features)

            D = D / D.std()
            D = torch.exp(-2 * D)

            w = D[I, J]
            G = GO.graph(I, J, N, W=w, pos=None, faces=None)
            G = G.to(device)
            xn = data.x.t().unsqueeze(0)
            xe = torch.ones(1, 1, I.shape[0]).to(device)

            # out = model(xn, xe, G)
            [out, G] = model(xn, G)
            [valmax, argmax] = torch.max(out, dim=1)
            g = G.nodeGrad(out.t().unsqueeze(0))
            eps = 1e-4
            absg = torch.sum(g ** 2, dim=1)
            tvreg = absg.mean()
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])  # 0.1 * tvreg +
            loss.backward()
            optimizer.step()
            # scheduler.step()
            return float(loss)

        @torch.no_grad()
        def test():
            model.eval()
            I = data.edge_index[0, :]
            J = data.edge_index[1, :]
            N = data.y.shape[0]
            features = data.x.squeeze().t()
            D = torch.relu(torch.sum(features ** 2, dim=0, keepdim=True) + \
                           torch.sum(features ** 2, dim=0, keepdim=True).t() - \
                           2 * features.t() @ features)

            D = D / D.std()
            D = torch.exp(-2 * D)
            w = D[I, J]
            G = GO.graph(I, J, N, W=w, pos=None, faces=None)
            G = G.to(device)
            xn = data.x.t().unsqueeze(0)
            xe = torch.ones(1, 1, I.shape[0]).to(device)
            [out, G] = model(xn, G)
            pred, accs = out.argmax(dim=-1), []
            for _, mask in data('train_mask', 'val_mask', 'test_mask'):
                accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
            return accs

        best_val_acc = test_acc = 0
        for epoch in range(1, 2001):
            loss = train()
            train_acc, val_acc, tmp_test_acc = test()
            if tmp_test_acc > best_val_acc:
                best_val_acc = tmp_test_acc
                test_acc = tmp_test_acc
                print(f'Epoch: {epoch:04d}, Loss: {loss:.4f} Train: {train_acc:.4f}, '
                      f'Val: {val_acc:.4f}, Test: {tmp_test_acc:.4f}, '
                      f'Final Test: {test_acc:.4f}')

        return test_acc


    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

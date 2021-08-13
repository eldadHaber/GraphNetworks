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
import optuna
print(torch.cuda.get_device_properties('cuda:0'))

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

print("**********************************************************************************")
file2Open = "src/optuna_fully_supervised.py"
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

num_layers = [2, 4, 8, 16, 32, 64]
quant_bits = [32, 8, 4, 2, 1]
for nlayers in num_layers:
    for bit in quant_bits:
        torch.cuda.synchronize()
        print("Without symmetric conv - not using TransposedConv !!!")
        print("Doing experiment for ", nlayers, " layers!", flush=True)
        print("Doing experiment for ", bit, " bit!", flush=True)

        torch.cuda.synchronize()


        def objective(trial):
            nEin = 1
            n_channels = 64  # trial.suggest_categorical('n_channels', [64, 128, 256])
            nopen = n_channels
            nhid = n_channels
            nNclose = n_channels
            nlayer = nlayers
            datastr = "cora"
            print("DATA SET IS:", datastr)
            # h = 1 / n_layers
            # h = trial.suggest_discrete_uniform('h', 0.1 / nlayer, 3, q=0.1 / (nlayer))
            h = trial.suggest_discrete_uniform('h', 0.1, 3, q=0.1)
            dropout = trial.suggest_discrete_uniform('dropout', 0.5, 0.7, q=0.1)
            # dropout = 0.6
            # h = 20 / nlayer
            print("n channels:", nopen)
            print("n layers:", nlayer)
            print("h step:", h)
            print("dropout:", dropout)
            print("bit:", bit)

            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            realVarlet = False

            lr = trial.suggest_float("lr", 1e-3, 1e-1, log=True)
            lr_alpha = trial.suggest_float("lr_alpha", 1e-6, 1e-3, log=True)
            lrBit = trial.suggest_float("lrBit", 1e-5, 1e-2, log=True)
            lrGCN = trial.suggest_float("lrGCN", 1e-5, 1e-2, log=True)
            wd = trial.suggest_float("wd", 5e-8, 1e-3, log=True)

            # wdGCN = trial.suggest_float("wdGCN", 1e-10, 1e-2, log=True)
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

            def train(datastr, splitstr, num_output):
                slurm = ("s" in sys.argv) or ("e" in sys.argv)
                adj, features, labels, idx_train, idx_val, idx_test, num_features, num_labels = process.full_load_data(
                    datastr,
                    splitstr, slurm=slurm)
                adj = adj.to_dense()
                # print("adj shape:", adj.shape)
                [edge_index, edge_weight] = sparseConvert.dense_to_sparse(adj)
                del adj
                # print("edge index shape:", edge_index.shape)
                # print("features shape:", features.shape)
                # print("labels shhape:", labels.shape)
                # print("idx shape:", idx_train.shape)
                edge_index = edge_index.to(device)
                features = features.to(device).t().unsqueeze(0)
                idx_train = idx_train.to(device)
                idx_test = idx_test.to(device)
                labels = labels.to(device)
                #
                model = GN.graphNetwork_nodesOnly_quant(num_features, nopen, nhid, nNclose, nlayer, h=h, dense=False,
                                                        varlet=True,
                                                        wave=False,
                                                        diffOrder=1, num_output=num_output, dropOut=dropout,
                                                        gated=False,
                                                        realVarlet=realVarlet, mixDyamics=True, perLayerDynamics=True,
                                                        act_bit=bit)
                model = model.to(device)

                optimizer = torch.optim.Adam([
                    dict(params=model.KN1, lr=lrGCN, weight_decay=0),
                    dict(params=model.KN2, lr=lrGCN, weight_decay=0),
                    dict(params=model.K1Nopen, weight_decay=wd),
                    dict(params=model.KNclose, weight_decay=wd),
                    dict(params=model.alpha, lr=lr_alpha, weight_decay=0),
                    dict(params=model.final_activation_alpha, lr=lrBit, weight_decay=0)
                ], lr=lr)

                bad_counter = 0
                best = 0
                for epoch in range(200):
                    loss_tra, acc_tra = train_step(model, optimizer, features, labels, edge_index, idx_train)
                    loss_val, acc_test = test_step(model, features, labels, edge_index, idx_test)
                    if (epoch + 1) % 10000000000 == 0:
                        print('Epoch:{:04d}'.format(epoch + 1),
                              'train',
                              'loss:{:.3f}'.format(loss_tra),
                              'acc:{:.2f}'.format(acc_tra * 100),
                              '| test',
                              'loss:{:.3f}'.format(loss_val),
                              'acc:{:.2f}'.format(acc_test * 100))
                    if acc_test > best:
                        best = acc_test
                        # torch.save(model.state_dict(), checkpt_file)
                        bad_counter = 0
                    else:
                        bad_counter += 1

                    if bad_counter == 200:
                        break
                acc = best

                return acc * 100

            acc_list = []
            for i in range(10):
                # datastr = "citeseer"
                if datastr == "cora":
                    num_output = 7
                elif datastr == "citeseer":
                    num_output = 6
                elif datastr == "pubmed":
                    num_output = 3
                elif datastr == "chameleon":
                    num_output = 5
                else:
                    num_output = 5
                if ("s" in sys.argv) or ("e" in sys.argv):
                    splitstr = 'splits/' + datastr + '_split_0.6_0.2_' + str(i) + '.npz'
                else:
                    splitstr = '../splits/' + datastr + '_split_0.6_0.2_' + str(i) + '.npz'

                acc_list.append(train(datastr, splitstr, num_output))
                print(i, ": {:.2f}".format(acc_list[-1]))

            mean_test_acc = np.mean(acc_list)
            print("Test acc.:{:.2f}".format(mean_test_acc))
            return mean_test_acc


        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)

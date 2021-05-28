import os, sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
import torch.autograd.profiler as profiler


from src import graphOps as GO
from src import processContacts as prc
from src import utils
from src import graphNet as GN
from torch.autograd import grad
from torch.utils.data import DataLoader

from src.Equivariant.NequIP_network import NequIP
from src.Equivariant.eq_utils import use_model_eq, use_proteinmodel_eq
# from src.Equivariant.networks import Network, GraphNet_EQ

from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import FullyConnectedNet, Gate, ExtractIr
from e3nn.o3 import TensorProduct, FullyConnectedTensorProduct
from e3nn.util.jit import compile_mode

from src.Equivariant.protein_network import Protein_network
from src.MD17_utils import getIterData_MD17, print_distogram, print_3d_structure, Dataset_MD17, use_model, \
    calculate_mean_coordinates
from src.protein_utils import Dataset_protein

if __name__ == '__main__':
    print(os.getcwd())
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # device='cpu'
    base_path = '../../../data/'
    caspver = 'casp11'

    # load training data
    # Aind = torch.load(base_path + caspver + '/AminoAcidIdx.pt')
    # Yobs = torch.load(base_path + caspver + '/RCalpha.pt')
    # MSK = torch.load(base_path + caspver + '/Masks.pt')
    # S = torch.load(base_path + caspver + '/PSSM.pt')
    # # load validation data
    # AindVal = torch.load(base_path + caspver + '/AminoAcidIdxVal.pt')
    # YobsVal = torch.load(base_path + caspver + '/RCalphaVal.pt')
    # MSKVal = torch.load(base_path + caspver + '/MasksVal.pt')
    # SVal = torch.load(base_path + caspver + '/PSSMVal.pt')

    # load Testing data
    AindTest = torch.load(base_path + caspver + '/AminoAcidIdxTesting.pt')
    YobsTest = torch.load(base_path + caspver + '/RCalphaTesting.pt')
    MSKTest = torch.load(base_path + caspver + '/MasksTesting.pt')
    STest = torch.load(base_path + caspver + '/PSSMTesting.pt')

    Aind = AindTest
    Yobs = YobsTest
    MSK = MSKTest
    S = STest

    ndata = len(Aind)
    n_train = 81
    epochs_for_lr_adjustment = 50
    batch_size = 1

    print('Number of data: {:}'.format(ndata))

    # Following Equivariant paper, we select 1000 configurations from these as our training set, 1000 as our validation set, and the rest are used as test data.
    dataset_train = Dataset_protein(Aind[:n_train],Yobs[:n_train],MSK[:n_train],S[:n_train])
    dataset_val = Dataset_protein(Aind[:n_train],Yobs[:n_train],MSK[:n_train],S[:n_train])
    # dataset_test = Dataset_MD17(R_test, F_test, E_test, z)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=False)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, drop_last=False)


    # Setup the network and its parameters


    irreps_in = o3.Irreps("40x0e")
    irreps_hidden = o3.Irreps("100x0e+1x0o+5x1e+5x1o")
    irreps_out = o3.Irreps("3x0e")
    irreps_node_attr = o3.Irreps("40x0e")
    irreps_edge_attr = o3.Irreps("1x0e+1x1o")
    layers = 6
    max_radius = 5
    number_of_basis = 8
    radial_layers = 1
    radial_neurons = 8
    num_neighbors = 15
    num_nodes = 0
    model = Protein_network(irreps_in=irreps_in, irreps_hidden=irreps_hidden, irreps_out=irreps_out, irreps_node_attr=irreps_node_attr, irreps_edge_attr=irreps_edge_attr, layers=layers, max_radius=max_radius,
                    number_of_basis=number_of_basis, radial_layers=radial_layers, radial_neurons=radial_neurons, num_neighbors=num_neighbors, num_nodes=num_nodes)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print('Number of parameters ', total_params)


    #### Start Training ####
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    alossBest = 1e6
    epochs = 100000

    bestModel = model
    hist = torch.zeros(epochs)
    eps = 1e-10
    nprnt = 1
    nprnt2 = min(nprnt, n_train)
    t0 = time.time()
    MAE_best = 1e6
    epochs_since_best = 0
    fig = plt.figure(num=1,figsize=[10,10])
    for epoch in range(epochs):
        t1 = time.time()
        aloss_t,MAE_t,Fps_t,Fts_t, t_dataload_t, t_prepare_t, t_model_t, t_backprop_t = use_proteinmodel_eq(model, dataloader_train, train=True, max_samples=1e6, optimizer=optimizer, batch_size=batch_size)
        t2 = time.time()
        aloss_v,MAE_v,Fps_v,Fts_v,t_dataload_v, t_prepare_v, t_model_v, t_backprop_v = use_proteinmodel_eq(model, dataloader_val, train=False, max_samples=10, optimizer=optimizer, batch_size=batch_size)
        t3 = time.time()

        if MAE_v < MAE_best:
            MAE_best = MAE_v
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= epochs_for_lr_adjustment:
                for g in optimizer.param_groups:
                    g['lr'] *= 0.8
                    lr = g['lr']
                epochs_since_best = 0

        # print(f' t_dataloader(train): {t_dataload_t:.3f}s  t_dataloader(val): {t_dataload_v:.3f}s  t_prepare(train): {t_prepare_t:.3f}s  t_prepare(val): {t_prepare_v:.3f}s  t_model(train): {t_model_t:.3f}s  t_model(val): {t_model_v:.3f}s  t_backprop(train): {t_backprop_t:.3f}s  t_backprop(val): {t_backprop_v:.3f}s')
        print(f'{epoch:2d}  Loss(train): {aloss_t:.2e}  Loss(val): {aloss_v:.2f}  MAE(train): {MAE_t:.2f}  MAE(val): {MAE_v:.2f}  |F_pred|(train): {Fps_t:.2f}  |F_pred|(val): {Fps_v:.2f}  |F_true|(train): {Fts_t:.2f}  |F_true|(val): {Fts_v:.2f}  MAE(best): {MAE_best:.2f}  Time(train): {t2-t1:.1f}s  Time(val): {t3-t2:.1f}s  Lr: {lr:2.2e} ')


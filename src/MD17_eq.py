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

from src.Equivariant.eq_utils import use_model_eq
from src.Equivariant.networks import Network

from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import FullyConnectedNet, Gate, ExtractIr
from e3nn.o3 import TensorProduct, FullyConnectedTensorProduct
from e3nn.util.jit import compile_mode

from src.MD17_utils import getIterData_MD17, print_distogram, print_3d_structure, Dataset_MD17, use_model, \
    calculate_mean_coordinates


if __name__ == '__main__':

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device='cpu'
    print_distograms = False
    print_3d_structures = False
    use_mean_map = False
    # load training data
    data = np.load('../../data/MD/MD17/aspirin_dft.npz')
    E = data['E']
    Force = data['F']
    R = data['R']
    epochs_for_lr_adjustment = 50
    z = torch.from_numpy(data['z']).to(dtype=torch.float32, device=device)
    # All geometries in Å, energy labels in kcal mol-1 and force labels in kcal mol-1 Å-1.
    # According to http://quantum-machine.org/gdml/#datasets, we need to reach a test error of 0.35 kcal mol-1 Å-1 on force prediction to be compatible with the best



    ndata = E.shape[0]
    natoms = z.shape[0]
    R_mean = None

    print('Number of data: {:}, Number of atoms {:}'.format(ndata, natoms))

    # Following Equivariant paper, we select 1000 configurations from these as our training set, 1000 as our validation set, and the rest are used as test data.
    n_train = 1000
    n_val = 1000
    batch_size = 5

    ndata_rand = 0 + np.arange(ndata)
    np.random.shuffle(ndata_rand)


    E_train = torch.from_numpy(E[ndata_rand[:n_train]]).to(dtype=torch.float32, device=device)
    F_train = torch.from_numpy(Force[ndata_rand[:n_train]]).to(dtype=torch.float32, device=device)
    R_train = torch.from_numpy(R[ndata_rand[:n_train]]).to(dtype=torch.float32, device=device)

    E_val = torch.from_numpy(E[ndata_rand[n_train:n_train+n_val]]).to(dtype=torch.float32, device=device)
    F_val = torch.from_numpy(Force[ndata_rand[n_train:n_train+n_val]]).to(dtype=torch.float32, device=device)
    R_val = torch.from_numpy(R[ndata_rand[n_train:n_train+n_val]]).to(dtype=torch.float32, device=device)

    E_test = torch.from_numpy(E[ndata_rand[n_train+n_val:]]).to(dtype=torch.float32, device=device)
    F_test = torch.from_numpy(Force[ndata_rand[n_train+n_val:]]).to(dtype=torch.float32, device=device)
    R_test = torch.from_numpy(R[ndata_rand[n_train+n_val:]]).to(dtype=torch.float32, device=device)


    dataset_train = Dataset_MD17(R_train, F_train, E_train, z)
    dataset_val = Dataset_MD17(R_val, F_val, E_val, z)
    dataset_test = Dataset_MD17(R_test, F_test, E_test, z)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=False)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, drop_last=False)

    # Setup the network and its parameters


    irreps_in = o3.Irreps("1x0e")
    irreps_hidden = o3.Irreps("50x0e+50x0o+10x1e+10x1o")
    irreps_out = o3.Irreps("1x0e")
    irreps_node_attr = o3.Irreps("1x0e")
    irreps_edge_attr = o3.Irreps("1x0e+1x1o")
    layers = 6
    max_radius = 5
    number_of_basis = 10
    radial_layers = 2
    radial_neurons = 20
    num_neighbors = 15
    num_nodes = natoms
    model = Network(irreps_in=irreps_in, irreps_hidden=irreps_hidden, irreps_out=irreps_out, irreps_node_attr=irreps_node_attr, irreps_edge_attr=irreps_edge_attr, layers=layers, max_radius=max_radius,
                    number_of_basis=number_of_basis, radial_layers=radial_layers, radial_neurons=radial_neurons, num_neighbors=num_neighbors, num_nodes=num_nodes)

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
        aloss_t,MAE_t,Fps_t,Fts_t, t_dataload_t, t_prepare_t, t_model_t, t_backprop_t = use_model_eq(model, dataloader_train, train=True, max_samples=1e6, optimizer=optimizer, batch_size=batch_size)
        t2 = time.time()
        aloss_v,MAE_v,Fps_v,Fts_v,t_dataload_v, t_prepare_v, t_model_v, t_backprop_v = use_model_eq(model, dataloader_val, train=False, max_samples=1, optimizer=optimizer, batch_size=batch_size)
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
        print(f'{epoch:2d}  Loss(train): {aloss_t:.2f}  Loss(val): {aloss_v:.2f}  MAE(train): {MAE_t:.2f}  MAE(val): {MAE_v:.2f}  |F_pred|(train): {Fps_t:.2f}  |F_pred|(val): {Fps_v:.2f}  |F_true|(train): {Fts_t:.2f}  |F_true|(val): {Fts_v:.2f}  MAE(best): {MAE_best:.2f}  Time(train): {t2-t1:.1f}s  Time(val): {t3-t2:.1f}s  Lr: {lr:2.2e} ')


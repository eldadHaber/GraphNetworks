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
from src import utils
from src import graphNet as GN
from torch.autograd import grad
from torch.utils.data import DataLoader
from e3nn import o3

from src.Equivariant.NequIP_network import NequIP
from src.Equivariant.eq_utils import use_model_eq
from src.MD17_utils import getIterData_MD17, print_distogram, print_3d_structure, Dataset_MD17, getBatchData_MD17
from src.Poincare.NequIP_network_pc import NequIP_pc
from src.Poincare.pc_utils import use_model_eq_pc, Dataset_MD17_pc


def generate_poincare_datasets(nhist,nskips,R):
    nR, natoms, ndim = R.shape
    ndata = nR - nhist - nskips

    Rin = torch.empty((ndata,natoms,nhist,ndim),dtype=torch.float32,device=R.device)

    R_target = R[nhist:]
    for i in range(nhist):
        Rin[:,:,i,:] = R[i:ndata+i]

    return Rin,R_target



if __name__ == '__main__':
    n_train = 10000
    n_val = 1000
    batch_size = 40
    nhist = 2
    nskips = 0
    epochs_for_lr_adjustment = 50
    use_validation = False
    lr = 1e-2

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # device='cpu'
    # load training data
    data = np.load('../../../../data/MD/MD17/aspirin_dft.npz')
    # E = data['E']
    # Force = data['F']
    R = torch.from_numpy(data['R']).to(dtype=torch.float32, device=device)
    Rin, Rout = generate_poincare_datasets(nhist, nskips, R)

    z = torch.from_numpy(data['z']).to(dtype=torch.float32, device=device)


    ndata = R.shape[0]
    natoms = z.shape[0]

    print('Number of data: {:}, Number of atoms {:}'.format(ndata, natoms))

    # Following Equivariant paper, we select 1000 configurations from these as our training set, 1000 as our validation set, and the rest are used as test data.

    ndata_rand = 0 + np.arange(ndata)
    # np.random.shuffle(ndata_rand)


    Rin_train = Rin[ndata_rand[:n_train]]
    Rout_train = Rout[ndata_rand[:n_train]]

    Rin_val = Rin[ndata_rand[n_train:n_train+n_val]]
    Rout_val = Rout[ndata_rand[n_train:n_train+n_val]]

    dataset_train = Dataset_MD17_pc(Rin_train, Rout_train, z)
    dataset_val = Dataset_MD17_pc(Rin_val, Rout_val, z)
    # dataset_test = Dataset_MD17(R_test, F_test, E_test, z)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=False)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, drop_last=False)

    irreps_in = None  # o3.Irreps("0x0e")
    irreps_hidden = o3.Irreps("100x0e+100x0o+50x1e+50x1o")
    irreps_out = o3.Irreps("1x1o")
    irreps_node_attr = o3.Irreps("1x0e")
    irreps_edge_attr = o3.Irreps("{:}x0e+{:}x1o".format(nhist,nhist))
    layers = 6
    max_radius = 5
    number_of_basis = 8
    radial_neurons = [16, 16]
    num_neighbors = 15
    num_nodes = natoms
    model = NequIP_pc(irreps_in=irreps_in, irreps_hidden=irreps_hidden, irreps_out=irreps_out,
                   irreps_node_attr=irreps_node_attr, irreps_edge_attr=irreps_edge_attr, layers=layers,
                   max_radius=max_radius,
                   number_of_basis=number_of_basis, radial_neurons=radial_neurons, num_neighbors=num_neighbors,
                   num_nodes=num_nodes,reduce_output=False)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print('Number of parameters ', total_params)

    #### Start Training ####
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    alossBest = 1e6
    epochs = 100000

    bestModel = model
    hist = torch.zeros(epochs)
    eps = 1e-10
    nprnt = 1
    nprnt2 = min(nprnt, n_train)
    t0 = time.time()
    loss_best = 1e6
    epochs_since_best = 0
    fig = plt.figure(num=1, figsize=[10, 10])
    for epoch in range(epochs):
        t1 = time.time()
        aloss_t, aloss_ref_t, t_dataload_t, t_prepare_t, t_model_t, t_backprop_t = use_model_eq_pc(model, dataloader_train, train=True, max_samples=1e6, optimizer=optimizer, batch_size=batch_size)
        t2 = time.time()
        if use_validation:
            aloss_v, t_dataload_v, t_prepare_v, t_model_v, t_backprop_v = use_model_eq_pc(model, dataloader_val, train=False, max_samples=10, optimizer=optimizer, batch_size=batch_size)
        else:
            aloss_v, t_dataload_v, t_prepare_v, t_model_v, t_backprop_v = 0,0,0,0,0
        t3 = time.time()

        if aloss_t < alossBest:
            alossBest = aloss_t
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= epochs_for_lr_adjustment:
                for g in optimizer.param_groups:
                    g['lr'] *= 0.8
                    lr = g['lr']
                epochs_since_best = 0

        # print(f' t_dataloader(train): {t_dataload_t:.3f}s  t_dataloader(val): {t_dataload_v:.3f}s  t_prepare(train): {t_prepare_t:.3f}s  t_prepare(val): {t_prepare_v:.3f}s  t_model(train): {t_model_t:.3f}s  t_model(val): {t_model_v:.3f}s  t_backprop(train): {t_backprop_t:.3f}s  t_backprop(val): {t_backprop_v:.3f}s')
        print(f'{epoch:2d}  Loss(train): {aloss_t:.2e}  Loss_ref(train): {aloss_ref_t:.2e}  Loss(val): {aloss_v:.2f}  Time(train): {t2 - t1:.1f}s  Time(val): {t3 - t2:.1f}s  Lr: {lr:2.2e} ')



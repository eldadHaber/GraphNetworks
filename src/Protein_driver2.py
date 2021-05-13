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
from src.protein_utils import Dataset_protein, GraphCollate


def move_list_to_device(a, device):
    a = [x.to(device=device) for x in a]
    return a

if __name__ == '__main__':
    print(os.getcwd())
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # device='cpu'
    base_path = '../../../data/'
    caspver = 'casp11'

    # load training data
    Aind = torch.load(base_path + caspver + '/AminoAcidIdx.pt')
    Yobs = torch.load(base_path + caspver + '/RCalpha.pt')
    MSK = torch.load(base_path + caspver + '/Masks.pt')
    S = torch.load(base_path + caspver + '/PSSM.pt')
    # load validation data
    AindVal = torch.load(base_path + caspver + '/AminoAcidIdxVal.pt')
    YobsVal = torch.load(base_path + caspver + '/RCalphaVal.pt')
    MSKVal = torch.load(base_path + caspver + '/MasksVal.pt')
    SVal = torch.load(base_path + caspver + '/PSSMVal.pt')

    # load Testing data
    AindTest = torch.load(base_path + caspver + '/AminoAcidIdxTesting.pt')
    YobsTest = torch.load(base_path + caspver + '/RCalphaTesting.pt')
    MSKTest = torch.load(base_path + caspver + '/MasksTesting.pt')
    STest = torch.load(base_path + caspver + '/PSSMTesting.pt')

    Aind = AindTest
    Yobs = YobsTest
    MSK = MSKTest
    S = STest

    Aind = move_list_to_device(Aind, device)
    Yobs = move_list_to_device(Yobs, device)
    MSK = move_list_to_device(MSK, device)
    S = move_list_to_device(S, device)

    AindVal = move_list_to_device(AindVal, device)
    YobsVal = move_list_to_device(YobsVal, device)
    MSKVal = move_list_to_device(MSKVal, device)
    SVal = move_list_to_device(SVal, device)

    ndata = len(Aind)
    ndata = 1
    epochs_for_lr_adjustment = 50
    batch_size = 1
    use_validation = False
    print(f'Number of training data: {len(Aind):}, and validation data: {len(AindVal):}')


    # Following Equivariant paper, we select 1000 configurations from these as our training set, 1000 as our validation set, and the rest are used as test data.
    dataset_train = Dataset_protein(Aind[:ndata],Yobs[:ndata],MSK[:ndata],S[:ndata],device=device)
    dataset_val = Dataset_protein(AindVal,YobsVal,MSKVal,SVal,device=device)
    # dataset_test = Dataset_MD17(R_test, F_test, E_test, z)

    collator = GraphCollate()
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=False, collate_fn=collator)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, drop_last=False, collate_fn=collator)


    # Setup the network and its parameters


    irreps_in = None #o3.Irreps("0x0e")
    irreps_hidden = o3.Irreps("64x0e+32x1o")
    irreps_out = o3.Irreps("1x1o")
    irreps_node_attr = o3.Irreps("40x0e")
    irreps_edge_attr = o3.Irreps("1x0e+1x1o")
    layers = 6
    max_radius = 5
    number_of_basis = 5
    # radial_layers = 1
    radial_neurons = [5]
    num_neighbors = 17
    # num_nodes = 0
    model = Protein_network(irreps_in=irreps_in, irreps_hidden=irreps_hidden, irreps_out=irreps_out, irreps_node_attr=irreps_node_attr, irreps_edge_attr=irreps_edge_attr, layers=layers, max_radius=max_radius,
                    number_of_basis=number_of_basis, radial_neurons=radial_neurons, num_neighbors=num_neighbors, reduce_output=False)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print('Number of parameters ', total_params)
    # print("network parameters")


    #### Start Training ####
    lr = 1e-2
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    alossBest = 1e6
    epochs = 10000

    bestModel = model
    hist = torch.zeros(epochs)
    eps = 1e-10
    t0 = time.time()
    epochs_since_best = 0
    for epoch in range(epochs):
        w = epoch/epochs * 0.0
        t1 = time.time()
        aloss_t,aloss_rel, aloss_coords= use_proteinmodel_eq(model, dataloader_train, train=True, max_samples=1e6, optimizer=optimizer, batch_size=batch_size,w=w)
        t2 = time.time()
        if use_validation:
            aloss_v,aloss_rel_v, aloss_coords_v= use_proteinmodel_eq(model, dataloader_val, train=False, max_samples=1, optimizer=optimizer, batch_size=batch_size,w=w)
        else:
            aloss_v, aloss_rel_v, aloss_coords_v = 0,0,0
        # aloss_v= use_proteinmodel_eq(model, dataloader_val, train=False, max_samples=10, optimizer=optimizer, batch_size=batch_size)
        t3 = time.time()

        if (epoch +1) % 50 == 0:
            for g in optimizer.param_groups:
                g['lr'] *= 0.8
                lr = g['lr']
        # print(f' t_dataloader(train): {t_dataload_t:.3f}s  t_dataloader(val): {t_dataload_v:.3f}s  t_prepare(train): {t_prepare_t:.3f}s  t_prepare(val): {t_prepare_v:.3f}s  t_model(train): {t_model_t:.3f}s  t_model(val): {t_model_v:.3f}s  t_backprop(train): {t_backprop_t:.3f}s  t_backprop(val): {t_backprop_v:.3f}s')
        print(f'{epoch:2d}  Loss(train): {aloss_t:.2e}  Loss_distogram(train): {aloss_rel:.2e}    Loss_coordinates(train): {aloss_coords:.2e} Loss(val): {aloss_v:.2e}  Loss_distogram(val): {aloss_rel_v:.2e}    Time(train): {t2-t1:.1f}s  Time(val): {t3-t2:.1f}s  Lr: {lr:2.2e}  w: {w:2.7f}')



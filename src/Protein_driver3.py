import os, sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from torch.utils.data import DataLoader

from src.GraphNet2 import GraphNet2
from src.protein_utils import Dataset_protein, GraphCollate, use_proteinmodel


def move_list_to_device(a, device):
    a = [x.to(device=device) for x in a]
    return a

if __name__ == '__main__':
    # torch.autograd.set_detect_anomaly(True)
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

    Aind = move_list_to_device(Aind, device)
    Yobs = move_list_to_device(Yobs, device)
    MSK = move_list_to_device(MSK, device)
    S = move_list_to_device(S, device)

    ndata = len(Aind)
    n_train = 1
    epochs_for_lr_adjustment = 50
    batch_size = 1

    print('Number of data: {:}'.format(ndata))


    # Following Equivariant paper, we select 1000 configurations from these as our training set, 1000 as our validation set, and the rest are used as test data.
    dataset_train = Dataset_protein(Aind[:n_train],Yobs[:n_train],MSK[:n_train],S[:n_train],device=device)
    dataset_val = Dataset_protein(Aind,Yobs,MSK,S,device=device)
    # dataset_test = Dataset_MD17(R_test, F_test, E_test, z)

    collator = GraphCollate()
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=False, collate_fn=collator)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, drop_last=False, collate_fn=collator)


    # Setup the network and its parameters

    # xn_dim_in = 40
    xn_dim = 64
    xn_attr_dim = 40
    xe_dim = 64
    xe_attr_dim = 1
    nlayers = 6



    model = GraphNet2(xn_attr_dim, xn_dim, xe_dim, xe_attr_dim, nlayers)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print('Number of parameters ', total_params)


    #### Start Training ####
    lr = 1e-2
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
        w = epoch/epochs * 0.05
        if (epoch +1) % 50 == 0:
            viz = True
        else:
            viz = False
        t1 = time.time()
        aloss_t, aloss_disto, aloss_disto_rel, aloss_coords= use_proteinmodel(model, dataloader_train, train=True, max_samples=1e6, optimizer=optimizer, w=w, batch_size=batch_size, epoch=epoch, viz=viz)
        t2 = time.time()
        # aloss_v= use_proteinmodel_eq(model, dataloader_val, train=False, max_samples=10, optimizer=optimizer, batch_size=batch_size)
        t3 = time.time()

        if (epoch +1) % 50 == 0:
            for g in optimizer.param_groups:
                g['lr'] *= 0.8
                lr = g['lr']
        # print(f' t_dataloader(train): {t_dataload_t:.3f}s  t_dataloader(val): {t_dataload_v:.3f}s  t_prepare(train): {t_prepare_t:.3f}s  t_prepare(val): {t_prepare_v:.3f}s  t_model(train): {t_model_t:.3f}s  t_model(val): {t_model_v:.3f}s  t_backprop(train): {t_backprop_t:.3f}s  t_backprop(val): {t_backprop_v:.3f}s')
        print(f'{epoch:2d}  Loss(train): {aloss_t:.2e}  Loss_disto(train): {aloss_disto:.2e}  Loss_disto_rel(train): {aloss_disto_rel:.2e}   Loss_coordinates: {aloss_coords:.2e}   Time(train): {t2-t1:.1f}s  Time(val): {t3-t2:.1f}s  Lr: {lr:2.2e}  w: {w:2.7f}')
        # print(f'{epoch:2d}  Loss(train): {aloss_t:.2e}  Loss_disto(train): {aloss_rel:.2e}    Loss_coordinates: {aloss_coords:.2e}   Time(train): {t2-t1:.1f}s  Time(val): {t3-t2:.1f}s  Lr: {lr:2.2e}  w: {w:2.7f}')



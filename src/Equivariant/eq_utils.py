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
from torch_cluster import radius_graph

from src import graphOps as GO
from src import processContacts as prc
from src import utils
from src import graphNet as GN
from torch.autograd import grad
from torch.utils.data import DataLoader
import torch.utils.data as data
from torch_geometric.data import Data, DataLoader


def use_model_eq(model,dataloader,train,max_samples,optimizer,batch_size=1):
    aloss = 0.0
    aloss_E = 0.0
    aloss_F = 0.0
    Fps = 0.0
    Fts = 0.0
    MAE = 0.0
    t_dataload = 0.0
    t_prepare = 0.0
    t_model = 0.0
    t_backprop = 0.0
    if train:
        model.train()
    else:
        model.eval()
    t3 = time.time()
    for i, (Ri, Fi, Ei, zi) in enumerate(dataloader):
        t0 = time.time()
        Ri.requires_grad_(True)
        Ri_vec = Ri.reshape(-1,Ri.shape[-1])
        zi_vec = zi.reshape(-1,zi.shape[-1])
        batch = torch.arange(Ri.shape[0]).repeat_interleave(Ri.shape[1]).to(device=Ri.device)

        data = {
                'batch': batch,
                'pos': Ri_vec,
                'z': zi_vec
                }

        optimizer.zero_grad()
        t1 = time.time()
        E_pred = model(data)
        E_pred_tot = torch.sum(E_pred)
        t2 = time.time()

        if train:
            F_pred = -grad(E_pred_tot, Ri, create_graph=True)[0].requires_grad_(True)
        else:
            F_pred = -grad(E_pred_tot, Ri, create_graph=False)[0]
        loss = F.mse_loss(F_pred, Fi)
        Fps += torch.mean(torch.sqrt(torch.sum(F_pred.detach() ** 2, dim=1)))
        Fts += torch.mean(torch.sqrt(torch.sum(Fi ** 2, dim=1)))
        MAEi = torch.mean(torch.abs(F_pred - Fi)).detach()
        MAE += MAEi
        if train:
            loss.backward()
            optimizer.step()
        aloss += loss.detach()
        t_dataload += t0 - t3
        t3 = time.time()
        t_prepare += t1 - t0
        t_model += t2 - t1
        t_backprop += t3 - t2
        if (i+1)*batch_size >= max_samples:
            break
    aloss /= (i+1)
    aloss_E /= (i+1)
    aloss_F /= (i+1)
    MAE /= (i+1)
    Fps /= (i+1)
    Fts /= (i+1)
    t_dataload /= (i+1)
    t_prepare /= (i+1)
    t_model /= (i+1)
    t_backprop /= (i+1)

    return aloss,MAE,Fps,Fts, t_dataload, t_prepare, t_model, t_backprop


def use_proteinmodel_eq(model,dataloader,train,max_samples,optimizer,batch_size=1):
    aloss = 0.0
    max_radius = 5
    if train:
        model.train()
    else:
        model.eval()
    t3 = time.time()
    for i, (seq, pssm, coords, mask, D, I, J, V) in enumerate(dataloader):
        nb,n,_ = coords.shape
        batch = torch.zeros(n,dtype=torch.int64)

        t0 = time.time()
        data = {
                'batch': batch,
                'pos': coords.squeeze().to(dtype=torch.float32),
                'edge_src': J.squeeze(),
                'edge_dst': I.squeeze(),
                'edge_vec': V.squeeze(),
                'seq': seq.squeeze(),
                'pssm': pssm.squeeze().to(dtype=torch.float32)
                }

        optimizer.zero_grad()
        t1 = time.time()
        node_vec = model(data)

        M = torch.ger(mask.squeeze(), mask.squeeze())
        Dout = utils.getDistMat(node_vec.transpose(0,1))
        Dtrue = utils.getDistMat(coords.squeeze().transpose(0,1))
        loss = F.mse_loss(M * Dout, M * Dtrue)
        if train:
            loss.backward()
            optimizer.step()
        aloss += loss.detach()
        if (i+1)*batch_size >= max_samples:
            break
    aloss /= (i+1)
    return aloss
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

def maskMat(T,M):
    M = M.squeeze()
    MT = (M*(M*T).t()).t()
    return MT


def use_proteinmodel_eq(model,dataloader,train,max_samples,optimizer,batch_size=1):
    aloss = 0.0
    aloss_rel = 0.0
    if train:
        model.train()
    else:
        model.eval()
    t3 = time.time()
    for i, (batch, seq, pssm, coords, mask, D, I, J, V, I_all, J_all) in enumerate(dataloader):
        t0 = time.time()
        data = {
                'batch': batch,
                'pos': coords,
                'edge_src': J,
                'edge_dst': I,
                'edge_vec': V,
                'seq': seq,
                'pssm': pssm
                }

        optimizer.zero_grad()
        t1 = time.time()
        node_vec = model(data)

        Dout = torch.norm(node_vec[I_all] - node_vec[J_all],p=2,dim=-1).view(-1,1)
        Dtrue = torch.norm(coords[I_all]-coords[J_all],p=2,dim=-1).view(-1,1)

        If, _ = torch.nonzero(Dtrue < 7*3.8, as_tuple=True)
        Dtrue = Dtrue[If]
        Dout = Dout[If]

        loss = F.mse_loss(Dout, Dtrue)
        loss_rel = loss / F.mse_loss(Dtrue * 0, Dtrue)

        if train:
            loss.backward()
            optimizer.step()
        aloss += loss.detach()
        aloss_rel += loss_rel.detach()
        if (i + 1) * batch_size >= max_samples:
            break
    aloss /= (i + 1)
    aloss_rel /= (i + 1)

    return aloss, aloss_rel






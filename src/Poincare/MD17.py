import argparse
from datetime import datetime
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
from src.Poincare import log
from src.Poincare.NequIP_network_pc import NequIP_pc
from src.Poincare.log import log_all_parameters
from src.Poincare.pc_utils import use_model_eq_pc, Dataset_MD17_pc, fix_seed


def generate_poincare_datasets(nhist,nskips,R):
    nR, natoms, ndim = R.shape
    ndata = nR - nhist - nskips

    Rin = torch.empty((ndata,natoms,nhist,ndim),dtype=torch.float32,device=R.device)

    R_target = R[nhist+nskips:]
    for i in range(nhist):
        Rin[:,:,i,:] = R[i:ndata+i]

    return Rin,R_target



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MD17 poincare')
    args = parser.parse_args()

    args.mode ='standard'
    args.n_train = 100000
    args.n_val = 1000
    args.batch_size = 50
    args.nhist = 2
    args.nskips = 99
    args.epochs_for_lr_adjustment = 50
    args.use_validation = True
    args.lr = 1e-2
    args.seed = 123545
    args.epochs = 100000

    args.network = {
        'irreps_in': None,  # o3.Irreps("0x0e")
        'irreps_hidden': o3.Irreps("100x0e+100x0o+50x1e+50x1o"),
        'irreps_out': o3.Irreps("1x1o"),
        'irreps_node_attr': o3.Irreps("1x0e"),
        'irreps_edge_attr': o3.Irreps("{:}x0e+{:}x1o".format(args.nhist,args.nhist)),
        'layers': 6,
        'max_radius': 5,
        'number_of_basis': 8,
        'radial_neurons': [16, 16],
        'num_neighbors': 15,
    }
    args.basefolder = os.path.basename(__file__).split(".")[0]
    c = vars(args)
    cn = c['network']

    fix_seed(c['seed']) #Set a seed, so we make reproducible results.


    c['result_dir'] = "../../{root}/{runner_name}/{date:%Y-%m-%d_%H_%M_%S}".format(
        root='results',
        runner_name=c['basefolder'],
        date=datetime.now(),
    )

    os.makedirs(c['result_dir'])
    logfile_loc = "{}/{}.log".format(c['result_dir'], 'output')
    LOG = log.setup_custom_logger('runner',logfile_loc,c['mode'])
    log_all_parameters(LOG, c)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # device='cpu'
    # load training data
    data = np.load('../../../../data/MD/MD17/aspirin_dft.npz')
    R = torch.from_numpy(data['R']).to(dtype=torch.float32, device=device)
    Rin, Rout = generate_poincare_datasets(c['nhist'], c['nskips'], R)

    z = torch.from_numpy(data['z']).to(dtype=torch.float32, device=device)

    ndata = Rout.shape[0]
    natoms = z.shape[0]

    print('Number of data: {:}, Number of atoms {:}'.format(ndata, natoms))

    # Following Equivariant paper, we select 1000 configurations from these as our training set, 1000 as our validation set, and the rest are used as test data.

    ndata_rand = 0 + np.arange(ndata)
    np.random.shuffle(ndata_rand)


    Rin_train = Rin[ndata_rand[:c['n_train']]]
    Rout_train = Rout[ndata_rand[:c['n_train']]]

    Rin_val = Rin[ndata_rand[c['n_train']:c['n_train']+c['n_val']]]
    Rout_val = Rout[ndata_rand[c['n_train']:c['n_train']+c['n_val']]]

    dataset_train = Dataset_MD17_pc(Rin_train, Rout_train, z)
    dataset_val = Dataset_MD17_pc(Rin_val, Rout_val, z)
    # dataset_test = Dataset_MD17(R_test, F_test, E_test, z)

    dataloader_train = DataLoader(dataset_train, batch_size=c['batch_size'], shuffle=True, drop_last=True)
    dataloader_val = DataLoader(dataset_val, batch_size=c['batch_size'], shuffle=True, drop_last=False)

    model = NequIP_pc(irreps_in=cn['irreps_in'], irreps_hidden=cn['irreps_hidden'], irreps_out=cn['irreps_out'],
                   irreps_node_attr=cn['irreps_node_attr'], irreps_edge_attr=cn['irreps_edge_attr'], layers=cn['layers'],
                   max_radius=cn['max_radius'],
                   number_of_basis=cn['number_of_basis'], radial_neurons=cn['radial_neurons'], num_neighbors=cn['num_neighbors'],
                   num_nodes=natoms,reduce_output=False)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    LOG.info('Number of parameters {:}'.format(total_params))

    #### Start Training ####
    optimizer = torch.optim.Adam(model.parameters(), lr=c['lr'])

    alossBest = 1e6
    lr = c['lr']
    t0 = time.time()
    epochs_since_best = 0
    for epoch in range(c['epochs']):
        t1 = time.time()
        aloss_t, aloss_ref_t, MAE_t, t_dataload_t, t_prepare_t, t_model_t, t_backprop_t = use_model_eq_pc(model, dataloader_train, train=True, max_samples=1e6, optimizer=optimizer, batch_size=c['batch_size'])
        t2 = time.time()
        if c['use_validation']:
            aloss_v, aloss_ref_v, MAE_v, t_dataload_v, t_prepare_v, t_model_v, t_backprop_v = use_model_eq_pc(model, dataloader_val, train=False, max_samples=50, optimizer=optimizer, batch_size=c['batch_size'])
        else:
            aloss_v, aloss_ref_v, MAE_v, t_dataload_v, t_prepare_v, t_model_v, t_backprop_v = 0,0,0,0,0,0,0
        t3 = time.time()

        if aloss_v < alossBest:
            alossBest = aloss_v
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= c['epochs_for_lr_adjustment']:
                for g in optimizer.param_groups:
                    g['lr'] *= 0.8
                    lr = g['lr']
                epochs_since_best = 0

        # print(f' t_dataloader(train): {t_dataload_t:.3f}s  t_dataloader(val): {t_dataload_v:.3f}s  t_prepare(train): {t_prepare_t:.3f}s  t_prepare(val): {t_prepare_v:.3f}s  t_model(train): {t_model_t:.3f}s  t_model(val): {t_model_v:.3f}s  t_backprop(train): {t_backprop_t:.3f}s  t_backprop(val): {t_backprop_v:.3f}s')
        LOG.info(f'{epoch:2d}  Loss(train): {aloss_t:.2e}  Loss_ref(train): {aloss_ref_t:.2e} MAE(train) {MAE_t:.2e}  Loss(val): {aloss_v:.2e}   Loss_ref(val): {aloss_ref_v:.2e} MAE(val) {MAE_v:.2e}  Time(train): {t2 - t1:.1f}s  Time(val): {t3 - t2:.1f}s  Lr: {lr:2.2e} ')


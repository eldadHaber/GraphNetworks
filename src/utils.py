import numpy as np
import scipy
# import scipy.spatial
from scipy import interpolate
import string
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import trimesh
import matplotlib.pyplot as plt


class list2np(object):
    def __init__(self):
        pass

    def __call__(self, *args):
        args_array = ()
        for arg in args:
            args_array += (np.asarray(arg),)
        return args_array

    def __repr__(self):
        return self.__class__.__name__ + '()'


def getPointDistance(output, targets, alpha=0.5):
    # outRot, tarRot, R = rotatePoints(output.squeeze(0), targets.squeeze(0))
    # doutRot = outRot[:, 1:] - outRot[:, :-1]
    # dtarRot = tarRot[:, 1:] - tarRot[:, :-1]
    misfitDis = 0.0
    for i in range(output.shape[0]):
        outi = output[i, :, :]
        tari = targets[i, :, :]
        Dc = torch.sum(outi ** 2, dim=0, keepdim=True) + torch.sum(outi ** 2, dim=0,
                                                                   keepdim=True).t() - 2 * outi.t() @ outi
        Dc = torch.sqrt(torch.relu(Dc))
        Do = torch.sum(tari ** 2, dim=0, keepdim=True) + torch.sum(tari ** 2, dim=0,
                                                                   keepdim=True).t() - 2 * tari.t() @ tari
        Do = torch.sqrt(torch.relu(Do))
        misfitDis += F.mse_loss(Dc, Do) / F.mse_loss(Do, 0 * Do)
    misfitDis = misfitDis / output.shape[0]

    # misfitCoo = F.mse_loss(doutRot, dtarRot) / F.mse_loss(dtarRot, dtarRot * 0)
    misfitCoo = F.mse_loss(output, targets) / F.mse_loss(targets, targets * 0)

    misfit = alpha * misfitDis + (1 - alpha) * misfitCoo
    return misfit, misfitDis, misfitCoo


def coord_loss(r1s, r2s, mask):
    ind = mask.squeeze() > 0
    r1 = r1s[0, :, ind]
    r2 = r2s[0, :, ind]

    # First we translate the two sets, by setting both their centroids to origin
    r1c = r1 - torch.sum(r1, dim=1, keepdim=True) / r1.shape[1]
    r2c = r2 - torch.sum(r2, dim=1, keepdim=True) / r2.shape[1]

    H = r1c @ r2c.t()
    U, S, V = torch.svd(H)

    d = F.softsign(torch.det(V @ U.t()))

    ones = torch.ones_like(d, device=r1s.device)
    a = torch.stack((ones, ones, d), dim=-1)
    tmp = torch.diag_embed(a)

    R = V @ tmp @ U.t()

    r1cr = torch.zeros(1, 3, r1s.shape[2], device=r1.device)
    r1cr[0, :, ind] = R @ r1c
    r2cr = torch.zeros(1, 3, r2s.shape[2], device=r1.device)
    r2cr[0, :, ind] = r2c
    loss_tr = torch.norm(r1cr - r2cr) ** 2 / torch.norm(r2cr) ** 2
    return loss_tr, r1cr, r2cr


def getDistMat(X, msk=torch.tensor([1.0])):
    X = X.squeeze(0)
    D = torch.sum(torch.pow(X, 2), dim=0, keepdim=True) + \
        torch.sum(torch.pow(X, 2), dim=0, keepdim=True).t() - \
        2 * X.t() @ X

    dev = X.device
    msk = msk.to(dev)

    mm = torch.ger(msk, msk)
    return mm * torch.sqrt(torch.relu(D))


def getNormMat(N, msk=torch.tensor([1.0])):
    N = N / torch.sqrt(torch.sum(N ** 2, dim=0, keepdim=True) + 1e-9)
    D = N.t() @ N
    mm = torch.ger(msk, msk)
    return mm * D


def orgProtData(x, normals, s, msk, sigma=1.0):
    n = s.shape[1]
    D = getDistMat(x, msk)
    D = torch.exp(-sigma * D)
    N = getNormMat(normals, msk)
    XX = torch.zeros(20, 20, n, n)
    NN = torch.zeros(20, 20, n, n)
    mm = torch.ger(msk, msk)
    mm = mm.view(-1)

    for i in range(20):
        for j in range(20):
            sij = 0.5 * (torch.ger(s[i, :], s[j, :]) + torch.ger(s[j, :], s[i, :]))
            XX[i, j, :, :] = sij * D
            NN[i, j, :, :] = sij * N

    XX = XX.reshape((400, -1))
    NN = NN.reshape((400, -1))
    # XX = XX[:, mm > 0]
    # NN = NN[:, mm > 0]
    return XX, NN


def linearInterp1D(X, M):
    n = X.shape[1]
    ti = np.arange(0, n)
    t = ti[M != 0]
    f = interpolate.interp1d(t, X[:, M != 0], kind='slinear', axis=-1, copy=True, bounds_error=None,
                             fill_value='extrapolate')
    Xnew = f(ti)

    return Xnew


def distPenality(D, dc=0.379, M=torch.ones(1)):
    U = torch.triu(D, 2)
    p2 = torch.norm(M * torch.relu(2 * dc - U)) ** 2 / torch.sum(M > 0)

    return p2


def distConstraint(X, dc=0.379, M=torch.tensor([1])):
    X = X.squeeze()
    M = M.squeeze()
    n = X.shape[1]
    dX = X[:, 1:] - X[:, :-1]
    d = torch.sum(dX ** 2, dim=0)

    if torch.numel(M) > 1:
        avM = (M[1:] + M[:-1]) / 2 < 0.5
        dc = (avM == 0) * dc
    else:
        avM = 1e-3
    dX = (dX / torch.sqrt(d + avM)) * dc

    Xh = torch.zeros(X.shape[0], n, device=X.device)
    Xh[:, 0] = X[:, 0]
    Xh[:, 1:] = X[:, 0].unsqueeze(1) + torch.cumsum(dX, dim=1)
    Xh = M * Xh
    return Xh


def kl_div(p, q, weight=False):
    n = p.shape[1]
    p = torch.log_softmax(p, dim=0)
    KLD = F.kl_div(p.unsqueeze(0), q.unsqueeze(0), reduction='none').squeeze(0)
    if weight:
        r = torch.sum(q, dim=1)
    else:
        r = torch.ones(q.shape[0], device=p.device)

    r = r / r.sum()
    KLD = torch.diag(1 - r) @ KLD
    return KLD.sum() / KLD.shape[1]


def dMat(X):
    XX = X.t() @ X
    d = torch.diag(XX).unsqueeze(1)
    D = d + d.t() - 2 * XX
    # print("min D:", D.min())
    # print("max D:", D.max())
    eps = 1e-6
    D = torch.sqrt(torch.relu(D) + eps)  #### HERE
    # D = torch.relu(D)
    if torch.isnan(D).float().sum() > 0:
        print("its nan !")
    return D


def dRMSD(X, Xobs, M):
    Morig = M.clone()
    X = torch.squeeze(X)
    Xobs = torch.squeeze(Xobs)
    M = torch.squeeze(M)

    # Compute distance matrices
    D = dMat(X)
    Dobs = dMat(Xobs)

    # Filter non-physical ones
    n = X.shape[-1]
    Xl = torch.zeros(3, n, device=X.device)
    Xl[0, :] = 3.8 * torch.arange(0, n)
    Dl = dMat(Xl)

    ML = (M * Dl - M * Dobs) > 0

    MS = Dobs < 8 * (3.8)
    M = M > 0
    M = (M & ML & MS) * 1.0
    R = torch.triu(D - Dobs, 2)
    M = torch.triu(M, 2)
    if torch.sum(M) < 1:
        print("Problem in M:,", torch.sum(M), flush=True)
        print("M original (not triu) nnz:", torch.sum(Morig))
        print("Nonzero indices:", torch.nonzero(Morig))
        print("Morig shape:", Morig.shape)

    if torch.isnan(M).float().sum() > 0:
        print("Problem, NaNs in M", flush=True)

    if torch.isnan(R).float().sum() > 0:
        print("Problem, NaNs in R", flush=True)
    loss = torch.norm(M * R) ** 2 / torch.sum(M)  #

    return loss


def saveMesh(xn, faces, pos, i=0, vmax=None, vmin=None):
    # xn of shape [points, features]
    # if with our net dim = 2 else 1
    print("xn shape:", xn.shape)
    print("pos shape:", pos.shape)
    print("faces.shape:", faces.shape)
    print("colors:", xn.squeeze(0).norm(dim=1).clone().detach().cpu().numpy())
    if 1==1:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        p = ax.scatter(pos[:, 0].clone().detach().cpu().numpy(), pos[:, 1].clone().detach().cpu().numpy(),
                       pos[:, 2].clone().detach().cpu().numpy(),
                       c=xn.squeeze(0).norm(dim=1).clone().detach().cpu().numpy(), vmin=0.0, vmax=1.0)
        fig.colorbar(p)
        plt.savefig(
            "/users/others/eliasof/GraphNetworks/plots_wave/xn_norm_wave_layer_" + str(i))
        plt.close()

    mesh = trimesh.Trimesh(vertices=pos, faces=faces, process=False)
    colors = xn.squeeze(0).norm(dim=1).clone().detach().cpu().numpy() # xn.squeeze(0).clone().detach().cpu().numpy()[:, 0]
    if vmax is not None:
        colors[colors < vmin] = vmin
        colors[colors > vmax] = vmax
        add = np.array([[vmax], [vmin]], dtype=np.float).squeeze()
    else:
        colors[colors < 0.0] = 0.0
        colors[colors > 1.0] = 1.0
        add = np.array([[1.0], [0.0]], dtype=np.float).squeeze()
    vect_col_map2 = trimesh.visual.color.interpolate(colors,
                                                     color_map='jet')

    colors = np.concatenate((add, colors), axis=0)
    colors = xn.squeeze(0).norm(dim=1).clone().detach().cpu().numpy()
    vect_col_map = trimesh.visual.color.interpolate(colors,
                                                    color_map='jet')
    #vect_col_map = vect_col_map[2:, :]
    if xn.shape[0] == mesh.vertices.shape[0]:
        mesh.visual.vertex_colors = vect_col_map
    elif xn.shape[0] == mesh.faces.shape[0]:
        mesh.visual.face_colors = vect_col_map
        smooth = False

    trimesh.exchange.export.export_mesh(mesh,
                                        "/users/others/eliasof/GraphNetworks/plots_wave/xn_norm_wave_layer_" + str(
                                            i) + ".ply", "ply")


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

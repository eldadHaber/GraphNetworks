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

    #outRot, tarRot, R = rotatePoints(output.squeeze(0), targets.squeeze(0))
    #doutRot = outRot[:, 1:] - outRot[:, :-1]
    #dtarRot = tarRot[:, 1:] - tarRot[:, :-1]
    misfitDis = 0.0
    for i in range(output.shape[0]):
        outi  = output[i,:,:]
        tari = targets[i,:,:]
        Dc = torch.sum(outi**2, dim=0, keepdim=True) + torch.sum(outi**2, dim=0,
                                                                 keepdim=True).t() - 2*outi.t()@outi
        Dc = torch.sqrt(torch.relu(Dc))
        Do = torch.sum(tari**2, dim=0, keepdim=True) + torch.sum(tari**2, dim=0,
                                                                  keepdim=True).t() - 2*tari.t()@tari
        Do = torch.sqrt(torch.relu(Do))
        misfitDis += F.mse_loss(Dc, Do) / F.mse_loss(Do, 0*Do)
    misfitDis = misfitDis/output.shape[0]

    #misfitCoo = F.mse_loss(doutRot, dtarRot) / F.mse_loss(dtarRot, dtarRot * 0)
    misfitCoo = F.mse_loss(output, targets) / F.mse_loss(targets, targets * 0)

    misfit = alpha*misfitDis + (1-alpha)*misfitCoo
    return misfit, misfitDis, misfitCoo

def getRandomMask(n,m):
    mask = torch.ones(m)
    i = torch.randint(0,m-n,(1,))
    mask[i:i+n] = 0
    return mask


def getRandomCrop(X,M, winsize=64, batchSize=[]):

    # Find potential windows
    k = X.shape[-1]
    ind = []
    T = []
    for i in range(k-winsize):
        t = torch.sum(M[i:i+winsize])
        if t == winsize:
            ind.append(i)
            T.append(X[:,:,i:i+winsize])

    # Choose on in random
    n = len(ind)
    Xout = []
    Tout = []
    if n > 0:
        ii = torch.randint(0,n,(1, ))
        Xout = X[:,:,ind[ii]:ind[ii]+winsize]
        Tout = torch.Tensor(len(T),X.shape[1],winsize)
        for i in range(n):
            Tout[i,:,:] = T[i]
    if len(batchSize)>0:
        if n>batchSize[0]:
            jj = torch.randint(0,n-batchSize[0],(1,))
            Tout = Tout[jj:jj+batchSize[0],:,:]
    return Xout, Tout


def getRotMat(t):
    A1 = torch.tensor([[torch.cos(t[0]), -torch.sin(t[0]), 0],
                       [torch.sin(t[0]), torch.cos(t[0]), 0],
                         [0, 0, 1.0]])
    A2 = torch.tensor([[torch.cos(t[1]),  0, -torch.sin(t[1])],
                       [0, 1.0, 0],
                       [torch.sin(t[1]), 0 , torch.cos(t[1])]])
    A3 = torch.tensor([[1,0,0],
                       [0, torch.cos(t[2]), -torch.sin(t[2])],
                       [0, torch.sin(t[2]), torch.cos(t[2])]])
    return A1@A2@A3

def tr2Dist(Y):

    k = Y.shape[2]
    Z = Y[0,:,:]
    Z = Z - torch.mean(Z, dim=0, keepdim=True)
    D = torch.sum(Z**2, dim=1).unsqueeze(0) + torch.sum(Z**2, dim=1).unsqueeze(1) - 2*Z@Z.t()
    D = 3*D/k
    return torch.sqrt(torch.relu(D))

def Seq2Dist(Y):

    D = torch.sum(Y**2,dim=1) + torch.sum(Y**2,dim=1).t() -  Y[0,:,:].t()@Y[0,:,:]
    return D

def rotatePoints(X, Xo):
    # Find a matrix R such that Xo@R = X
    # Translate X to fit Xo
    # (X+c)R = V*S*V' R + c*R = Xo
    # X = Uo*So*Vo'*R' - C

    # R = V*inv(S)*U'
    X = X.squeeze(0).t()
    Xo = Xo.t()
    if X.shape != Xo.shape:
        U, S, V =  torch.svd(X)
        X = U[:,:3]@torch.diag(S[:3])@V[:3,:3].t()

    n, dim = X.shape

    Xc  = X - X.mean(dim=0)
    Xco = Xo - Xo.mean(dim=0)

    C = (Xc.t()@Xco) / n
    U, S, V = torch.svd(C)
    d = torch.sign((torch.det(U) * torch.det(V)))

    R  = V@torch.diag(torch.tensor([1.0,1,d],dtype=U.dtype))@U.t()

    Xr = Xc@R.t()
    #print(torch.norm(Xco - Xc @ R.t()))

    return Xr.t(), Xco.t(), R

def getRotDist(Xc, Xo, alpha = 1.0):

    Xr, Xco, R = rotatePoints(Xc, Xo)
    Do = torch.sum(Xo**2,dim=1,keepdim=True) + torch.sum(Xo**2,dim=1,keepdim=True).t() - 2*Xo@Xo.t()
    Do = torch.sqrt(torch.relu(Do))
    Dc = torch.sum(Xc**2,dim=1,keepdim=True) + torch.sum(Xc**2,dim=1,keepdim=True).t() - 2*Xc@Xc.t()
    Dc = torch.sqrt(torch.relu(Dc))

    dcoord = torch.norm(Xr-Xco)**2/torch.norm(Xco)**2
    ddmat  = torch.norm(Dc-Do)**2/torch.norm(Do)**2
    return dcoord + alpha*ddmat
# Define some functions

def coord_loss(r1s, r2s, mask):
    ind = mask.squeeze()>0
    r1 = r1s[0, :, ind]
    r2 = r2s[0, :, ind]

    # First we translate the two sets, by setting both their centroids to origin
    r1c = r1 - torch.sum(r1, dim=1, keepdim=True) / r1.shape[1]
    r2c = r2 - torch.sum(r2, dim=1, keepdim=True) / r2.shape[1]

    H = r1c@r2c.t()
    U, S, V = torch.svd(H)

    d = F.softsign(torch.det(V @ U.t()))

    ones = torch.ones_like(d,device=r1s.device)
    a = torch.stack((ones, ones, d), dim=-1)
    tmp = torch.diag_embed(a)

    R = V @ tmp @ U.t()

    r1cr = torch.zeros(1,3,r1s.shape[2],device=r1.device)
    r1cr[0,:,ind] = R @ r1c
    r2cr = torch.zeros(1,3, r2s.shape[2], device=r1.device)
    r2cr[0,:, ind] = r2c
    loss_tr = torch.norm(r1cr - r2cr) ** 2 / torch.norm(r2cr) ** 2
    return loss_tr, r1cr, r2cr

def move_tuple_to(args,device,non_blocking=True):
    new_args = ()
    for arg in args:
        new_args += (arg.to(device,non_blocking=non_blocking),)
    return new_args


def fix_seed(seed, include_cuda=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # if you are using GPU
    if include_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def determine_network_param(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)



def getDistMat(X,msk=torch.tensor([1.0])):
    D = torch.sum(torch.pow(X,2), dim=0, keepdim=True) + torch.sum(torch.pow(X,2), dim=0, keepdim=True).t() - 2*X.t()@X
    
    dev = X.device
    msk = msk.to(dev)

    mm = torch.ger(msk,msk)
    return mm*torch.sqrt(torch.relu(D))

def getNormMat(N,msk=torch.tensor([1.0])):
    N = N/torch.sqrt(torch.sum(N**2,dim=0,keepdim=True)+1e-9)
    D = N.t()@N
    mm = torch.ger(msk, msk)
    return mm*D

def orgProtData(x,normals,s, msk, sigma=1.0):
    n = s.shape[1]
    D = getDistMat(x,msk)
    D = torch.exp(-sigma*D)
    N = getNormMat(normals,msk)
    XX = torch.zeros(20, 20, n, n)
    NN = torch.zeros(20, 20, n, n)
    mm = torch.ger(msk, msk)
    mm = mm.view(-1)

    for i in range(20):
        for j in range(20):
            sij = 0.5*(torch.ger(s[i, :], s[j, :]) + torch.ger(s[j, :], s[i, :]))
            XX[i, j, :, :] = sij * D
            NN[i, j, :, :] = sij * N

    XX = XX.reshape((400, -1))
    NN = NN.reshape((400, -1))
    # XX = XX[:, mm > 0]
    # NN = NN[:, mm > 0]
    return XX, NN



def getGraphLap(X,sig=10):

    # normalize the data
    X = X - torch.mean(X, dim=1, keepdim=True)
    X = X / (torch.std(X, dim=1, keepdim=True) + 1e-3)
    #X = X/torch.sqrt(torch.sum(X**2,dim=1,keepdim=True)/X.shape[1] + 1e-4)
    # add  position vector
    pos = torch.linspace(-0.5, 0.5, X.shape[1]).unsqueeze(0)
    dev = X.device
    pos = pos.to(dev)
    Xa = torch.cat((X, 5e1*pos), dim=0)
    W = getDistMat(Xa)
    We = torch.exp(-W/sig)
    D = torch.diag(torch.sum(We, dim=0))
    L = D - We
    Dh = torch.diag(1/torch.sqrt(torch.diag(D)));
    L = Dh @ L @ Dh

    L = 0.5 * (L + L.t())

    return L, W


def randsvd(A,k):

    n = A.shape
    Omega = torch.randn(n[1],k,dtype=A.dtype)
    Y     = A@Omega
    Q,R   = torch.qr(Y)
    B     = Q.t()@A
    U, S, V = torch.svd(B)
    U = Q@U
    return U, S, V

def cheby(K,L,Z):
# apply T_k(x) = 2x*T_{k-1}(x) - T_{k-2}(x)
# KZ = \sum K[i] T_i(Z)
    n  = len(K)
    T0 = torch.ones(Z.shape) #torch.eye(L.shape[0])
    A  = K[0]*T0 #K[0]*Z@T0
    T1 = Z@L #L
    A  = A + K[1]*T1 #A + K[1]*Z@T1
    for i in range(2,n):
        T2 = 2*T1@L - T0
        A  = A + K[i]*T2
        T0 = T1.clone()
        T1 = T2.clone()
    return A

def linearInterp1D(X,M):
    n  = X.shape[1]
    ti = np.arange(0,n)
    t  = ti[M!=0]
    f = interpolate.interp1d(t, X[:,M!=0], kind='slinear', axis=-1, copy=True, bounds_error=None,
                             fill_value='extrapolate')
    Xnew = f(ti)

    return Xnew

def distPenality(D,dc=0.379,M=torch.ones(1)):
    U = torch.triu(D,2)
    p2 = torch.norm(M*torch.relu(2*dc - U))**2/torch.sum(M>0)

    return p2

def distConstraint(X,dc=0.379, M=torch.tensor([1])):
    X = X.squeeze()
    M = M.squeeze()
    n = X.shape[1]
    dX = X[:,1:] - X[:,:-1]
    d  = torch.sum(dX**2,dim=0)

    if torch.numel(M)>1:
        avM = (M[1:]+M[:-1])/2 < 0.5
        dc = (avM==0)*dc
    else:
        avM = 1e-3
    dX = (dX / torch.sqrt(d+avM)) * dc

    Xh = torch.zeros(X.shape[0],n, device=X.device)
    Xh[:, 0]  = X[:, 0]
    Xh[:, 1:] = X[:, 0].unsqueeze(1) + torch.cumsum(dX, dim=1)
    Xh = M*Xh
    return Xh

def kl_div(p, q, weight=False):
    n = p.shape[1]
    p   = torch.log_softmax(p, dim=0)
    KLD = F.kl_div(p.unsqueeze(0), q.unsqueeze(0), reduction='none').squeeze(0)
    if weight:
        r = torch.sum(q,dim=1)
    else:
        r = torch.ones(q.shape[0],device=p.device)

    r = r/r.sum()
    KLD = torch.diag(1-r)@KLD
    return KLD.sum()/KLD.shape[1]

def graphGrad(b, D):
    #D = torch.relu(torch.sum(X**2,dim=0,keepdim=True) + torch.sum(X**2, dim=0, keepdim=True).t() - 2*X.t()@X)
    n = D.shape[1]
    m = b.shape[0]
    idx = torch.triu(torch.ones(n, n,device=b.device), 1) > 0
    h = D[idx]

    B = b.unsqueeze(2) - b.unsqueeze(2).transpose(1,2)
    c = B[:,idx]/h
    return c

def graphDiv(c, D):

    #D = torch.relu(torch.sum(X **2,dim=0,keepdim=True) + torch.sum(X**2,dim=0,keepdim=True).t() - 2*X.t()@X)
    n = D.shape[1]
    m = c.shape[0]
    idx = torch.triu(torch.ones(n, n,device=D.device), 1) > 0
    h = D[idx]

    C = torch.zeros(m,n,n,device=D.device)
    C[:,idx] = c/h

    b = torch.sum(C,2) - torch.sum(C,1)

    return b


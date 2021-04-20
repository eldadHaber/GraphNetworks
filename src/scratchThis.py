import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import torch.optim as optim
## r=1
from src import graphOps as GO

def diffX(X):
    X = X.squeeze()
    return X[:,1:] - X[:,:-1]

def diffXT(X):
    X  = X.squeeze()
    D  = X[:,:-1] - X[:,1:]
    d0 = -X[:,0].unsqueeze(1)
    d1 = X[:,-1].unsqueeze(1)
    D  = torch.cat([d0,D,d1],dim=1)
    return D


def constraint(X,d):
    X = X.squeeze()
    c = torch.ones(1,3,device=X.device)@(diffX(X)**2) - d**2

    return c

def dConstraint(S,X):
    dX = diffX(X)
    dS = diffX(S)
    e  = torch.ones(1,3,device=X.device)
    dc = 2*e@(dX*dS)
    return dc

def dConstraintT(c,X):
    dX = diffX(X)
    e = torch.ones(3, 1, device=X.device)
    C = (e@c)*dX
    C = diffXT(C)
    return 2*C

def pcg(X, c, n=5):
    b = constraint(X, c)
    r = dConstraintT(b, X)
    d = b
    p = r
    t = dConstraint(p, X)
    x = torch.zeros(X.shape, device=X.device)

    for i in range(n):
        alpha = (torch.norm(r)**2)/(torch.norm(t)**2)
        x     = x + alpha*p
        d     = d - alpha*t
        rold  = r.clone()
        r     = dConstraintT(d, X)
        beta  = (torch.norm(r)**2)/(torch.norm(rold)**2)
        p     = r + beta*p
        t     = dConstraint(p, X)

    return x

def conNet(x,K,Ko, Kc,h):

    d = 1.0
    n = K.shape[0]
    x = torch.tanh(Ko@x)
    for j in range(100):
        x3 = Kc@x
        c = constraint(x3, d)
        print(j, torch.mean(torch.abs(c)))
        lam = Kc.t()@dConstraintT(c, x3)
        alpha = 0.1 / torch.norm(lam)
        x = x - alpha * lam

    for i in range(n):
        Ki = K[i,:,:]
        a = torch.tanh(Ki @ x)
        a = a/torch.norm(a)
        x  = x - h*Ki.t()@a

        for j in range(1):
            x3 = Kc@x
            c  = constraint(x3,d)
            print(j, torch.mean(torch.abs(c)))
            lam = Kc.t()@dConstraintT(c,x3)
            alpha = 0.1 / torch.norm(lam)
            x   = x - alpha*lam

        print(' ')
        print(' ')

    return x

Ko = torch.randn(16,5)
Kc = torch.eye(3,16)
K = torch.randn(18,16,16)
x = torch.randn(5,100)
#x = torch.zeros(3, 100)
#x[0, :] = 1.0 * torch.arange(0, 100)
z = conNet(x,K,Ko, Kc,0.1)

#c = constraint(z,1.0)
#print(torch.mean(torch.abs(c)))


if 0:
    X = torch.randn(1,3,100)
    S = torch.randn(1,3,100)*1e-2

    dc = dConstraint(S,X)
    c0 = constraint(X,1.0)
    c1 = constraint(X + S,1.0)
    print(torch.norm(c1-c0))
    print(torch.norm(c1-c0-dc))

    v = torch.randn(1,99)
    a1 = torch.dot(v.squeeze(),dc.squeeze())
    print(a1)
    t = dConstraintT(v,X)

    S = S.squeeze()

    print(torch.sum(t*S))
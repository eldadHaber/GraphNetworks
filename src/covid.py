import os, sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
from MDAnalysis.lib.formats.libdcd import DCDFile


if __name__ == '__main__':
    filename = '../../../data/MD/covid_19/spike_WE.dcd'
    with DCDFile(filename) as f:
        for i,frame in enumerate(f):
            # print(frame.x)
            print(i)
            if (i+1) % 100==0:
                print(frame.xyz)
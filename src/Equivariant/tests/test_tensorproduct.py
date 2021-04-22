import torch
from torch_cluster import radius_graph
from torch_geometric.data import Data, DataLoader
from torch_scatter import scatter
import numpy as np
from e3nn import o3
from e3nn.nn import FullyConnectedNet, Gate
from e3nn.o3 import TensorProduct
from e3nn.math import soft_one_hot_linspace

#Simple normalized cross product of two odd parity vectors, which gives an even parity vector (if you set the output to odd, it fails)
module = TensorProduct(
    "1x1o", "1x1o", "1x1e",
    [
        (0, 0, 0, "uuu", False)
    ]
)

x = torch.tensor([[1.0,0.0,0.0],[2.0,0,0]])
y = torch.tensor([[0,1.0,0],[0,2.0,0]])
z = module(x,y)



#Now the input vectors are 6D instead of 3D, and can essentially just be treated as 2 catenated 3D vectors as far as I can see.
module = TensorProduct(
    "2x1o", "2x1o", "2x1e",
    [
        (0, 0, 0, "uuu", False)
    ]
)
x = torch.tensor([[1.0,0.0,0.0,2.0,0,0]])
y = torch.tensor([[0,1.0,0,0,2.0,0]])
z = module(x,y)



# We have added a scalar to the first input. If this even scalar is first the parity changes, while if it is second it does not. No idea why.
module = TensorProduct(
    "1x1o+1x0e", "1x1o+1x1o", "1x1e+1x1o", #Here we explain the type of irreps going into input1,input2,and output.
    [
        (0, 0, 0, "uuu", False)            #Here we explain how we populate the output, so in this case we take the first irrep in input1 product with first irrep in input2, and put it in first irrep in output.
    ]                                      #'uuu' means that the connection is elementwise, which I'm not sure on yet
)

x = torch.tensor([[1.0,0,1.0,1.0]])
y = torch.tensor([[1,1.0,1,1,1,1]])
z = module(x,y)
#Hence this will only give something in the first part of z on output, since we never specified anything going into the second part.



module = TensorProduct(
    "1x1o+1x0e", "1x1o+1x1o", "1x1o+1x1e", #Here we explain the type of irreps going into input1,input2,and output.
    [
        (0, 0, 1, "uuu", False)            #Here we explain how we populate the output, so in this case we take the first irrep in input1 product with first irrep in input2, and put it in first irrep in output.
    ]                                      #'uuu' means that the connection is elementwise, which I'm not sure on yet
)

x = torch.tensor([[1.0,0,1.0,1.0]])
y = torch.tensor([[1,1.0,1,1,1,1]])
z = module(x,y)

module = TensorProduct(
    "1x0e+1x1o", "1x1o+1x1o", "1x1o+1x1e",
    [
        (0, 0, 0, "uuu", False),
        (1, 1, 1, 'uuu', False)
    ]
)

x = torch.tensor([[2.0,1,0.0,0.0]])
y = torch.tensor([[1,1.0,1,0,1,0]])
z = module(x,y)




#Now lets figure out how "uuu", "vvv", "uvw" and all these combinations works. I think they only matter when we have more than one of each irrep, since they designate mixing of them somehow

module = TensorProduct(
    "2x0e", "2x0e", "2x0e",
    [
        (0, 0, 0, "uuu", False)
    ]
)
x = torch.tensor([[1,2]],dtype=torch.float32)
y = torch.tensor([[3,4]],dtype=torch.float32)
z1 = module(x,y)


module = TensorProduct(
    "2x0e", "2x0e", "2x0e",
    [
        (0, 0, 0, "uvv", False)
    ]
)
x = torch.tensor([[5,0]],dtype=torch.float32)
y = torch.tensor([[0,5]],dtype=torch.float32)
z2 = module(x,y)
module.visualize()



print('done')

irreps = o3.Irreps("3x0e + 4x0o + 1e + 2o + 3o")
module2 = TensorProduct(irreps, irreps, "0e", [
    (i, i, 0, 'uuw', False)
    for i, (mul, ir) in enumerate(irreps)
    ])

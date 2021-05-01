import torch
from torch_cluster import radius_graph
from torch_geometric.data import Data, DataLoader
from torch_scatter import scatter
import numpy as np
from e3nn import o3
from e3nn.nn import FullyConnectedNet, Gate
from e3nn.o3 import TensorProduct, FullyConnectedTensorProduct, ElementwiseTensorProduct, FullTensorProduct
from e3nn.math import soft_one_hot_linspace
import matplotlib.pyplot as plt
# irreps = o3.Irreps("1x0e+1x1o")
from src.Equivariant.EQ_operations import SelfInteraction
import math


def stat(x,name):
    print(f"{name}: var={x.var():2.5f}, std={x.std():2.5f}, mean={x.mean():2.5f}, component={(x.norm(dim=1)/math.sqrt(x.shape[1])).mean():2.5f}, norm={x.norm(dim=1).mean():2.5f}")


plt.figure()
# plt.show()
n = 100
nd = 100
x = 10*torch.randn((n,nd))

# x = torch.tensor([[2.0, -0.0, -0.0, 0.0]])
stat(x,'x')


y = 1e7*torch.randn((n,nd))

in1_var = [100.0]
tp = ElementwiseTensorProduct("{:}x0e".format(nd),"{:}x0e".format(nd),in1_var=[x.var()],in2_var=[x.var()],out_var=[1.0])
tp0 = ElementwiseTensorProduct("{:}x0e".format(nd),"{:}x0e".format(nd))

xx = tp(x,x)
xx0 = tp0(x,x)
stat(xx,'xx')
stat(xx0,'xx0')




irreps = o3.Irreps("{:}x0e".format(nd))
si = SelfInteraction(irreps_in=irreps,irreps_out=irreps)

xsi = si(x, normalize_variance=False)
ysi = si(y)

stat(y,'y')
stat(ysi,'ysi')
stat(x,'x')
stat(xsi,'xsi')













tp = ElementwiseTensorProduct("1x0e","1x0e")

xy = tp(x,y)
xx = tp(x,x)
yy = tp(y,y)
stat(x,'x')
stat(y,'y')
stat(xx,'xx')
stat(xy,'xy')
stat(yy,'yy')
xx_xx = tp(xx,xx)
stat(xx_xx,'xx_xx')


xx_normal = xx/xx.std()
stat(xx_normal,'xx_normal')



print("__________________________")
irrep = "1x0e+1x1o"
tp = FullTensorProduct(irrep,irrep)
ftp = FullyConnectedTensorProduct(irrep,irrep,tp.irreps_out) # So a fullyconnected tensorproduct also mixes the outputs if they are of similar kinds, which is what the last number of paths are. However it seems like this last mixing is done differently than normal
#The last mixing is done purely on a level of elements
# ftp.weight.requires_grad_(False)
# ftp.weight[:] = 1
print(tp)
print(ftp)
s = torch.tensor([[1]])
v = torch.tensor([[1.0,1.0,1]])
print(v.norm(dim=1))
x = torch.cat((s,v),dim=1)
otp = tp(x,x)
oftp = ftp(x,x)


filter = iter([o3.Irrep("0e"),o3.Irrep("1e"),o3.Irrep("2e")])
tp = ElementwiseTensorProduct("1e + 1e", "1e + 1e", filter)
tp2 = ElementwiseTensorProduct("1e + 1e", "1e + 1e", "0e+1e+2e")


o3.Irrep("1x0o+1x0e")
filter = iter([o3.Irrep("0e"),o3.Irrep("1e"),o3.Irrep("1o"),o3.Irrep("1o")])
tp3 = FullTensorProduct("1x1o","1x1o",filter)
tp2 = FullTensorProduct("1x1o","1x1o")

x = torch.tensor([[2,0.0,0.0]])
y = torch.tensor([[0.0,2,0.0]])
out = tp3(x,y)
out2 = x.norm()*y.norm()/math.sqrt(2)
# tp2 = FullyConnectedTensorProduct("1x0e+1x1o","1x0e+1x1o",tp.irreps_out)

# normalization: str = 'component',
# internal_weights: Optional[bool] = None,
# shared_weights: Optional[bool] = None,
print(tp)
print(tp2)
s = torch.tensor([[10]])
v = torch.tensor([[11.123,1.31234,0.764]])
u = v[0]
print(v.norm(dim=1))
x = torch.cat((s,v),dim=1)
y = tp(x,x)
y2 = tp2(x,x)
y3 = tp3(v,v)


yy = u.dot(u)/math.sqrt(3)
plt.subplot(1,2,1)
tp.visualize()
plt.subplot(1,2,2)
tp2.visualize()
plt.show()
tp = FullyConnectedTensorProduct("1x0e+1x1o","1x0e+1x1o","1x0e+1x1o")

x = torch.tensor([[1,0.5,0.5,0.5]],dtype=torch.float32)
x = torch.randn(1000000,4)
y = tp(x,x)



tp = FullyConnectedTensorProduct("1x0e+1x1o","1x0e+1x1o","1x0e+1x1e+1x1o")

x = torch.tensor([[1,0.5,0.5,0.5]],dtype=torch.float32)
y = tp(x,x)





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

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
plt.figure()
# plt.show()
import math














def make_matrix_semi_unitary(M, debug=True):
    I = torch.eye(M.shape[-2])
    if debug:
        M_org = M.clone()
    for i in range(100):
        M = M - 0.5 * (M @ M.t() - I) @ M

    if debug:
        pre_error = torch.norm(I - M_org @ M_org.t())
        post_error = torch.norm(I - M @ M.t())
        print(f"Deviation from unitary before: {pre_error:2.2e}, deviation from unitary after: {post_error:2.2e}")
    return M

def uplift(x,M):
    x2 = x.reshape(x.shape[0], -1, 3).transpose(1, 2)
    y2 = x2 @ M
    y_vec = y2.transpose(1, 2).reshape(x.shape[0], -1)
    return y_vec


n = 5
n_vec_in = 3
n_vec_out = 10
rot = o3.rand_matrix(1)

M = torch.randn(n_vec_in,n_vec_out)
M = torch.empty((n_vec_in,n_vec_out))
torch.nn.init.xavier_normal_(M, gain=1 / math.sqrt(n_vec_in))  # Filled according to "Semi-Orthogonal Low-Rank Matrix Factorization for Deep Neural Networks"
M = make_matrix_semi_unitary(M)

x = torch.randn((n*n_vec_in,3))
xrot = (x @ rot)[0]

x_vec = x.reshape(n,-1)
xrot_vec = xrot.reshape(n,-1)

y = uplift(x_vec,M)
y_rot_before = uplift(xrot_vec,M)

y_rot_after = (y.reshape(-1,3) @ rot)[0]
y_rot_before2 = y_rot_before.reshape(-1,3)





vu = torch.tensor([[1.0,2.0,3.0],[4.0,5.0,6.0]])
v = torch.tensor([[1.0,2.0,3.0]])
u = torch.tensor([[4.0,5.0,6.0]])

vs = o3.spherical_harmonics(o3.Irreps("1x1o"), v, False, normalization='component')
us = o3.spherical_harmonics(o3.Irreps("1x1o"), u, False, normalization='component')
vus = o3.spherical_harmonics(o3.Irreps("1x1o"), vu, False, normalization='component')

vs
# edge_sh2 = o3.spherical_harmonics(o3.Irreps("1x0e+2x1o"), v, True, normalization='component')

irreps_x = o3.Irreps('2x1o')
irreps_z = o3.Irreps('4x1o')


M = torch.randn((2,4))



# irreps_z = o3.Irreps('100x0e+100x0o+50x1e+50x1o')

tp = o3.FullTensorProduct(irreps_x, irreps_z, ['0e'])
print(tp)

irreps_y = tp.irreps_out

z = irreps_z.randn(-1)

M = tp.right(z)
print(M.shape)

# forward
x = irreps_x.randn(-1)
v = torch.tensor([1.0,2.0,3.0,6.0,7.0,8.0])
y = x @ M

# inverse
invM = torch.pinverse(M)
x2 = y @ invM

assert torch.allclose(x, x2)


for weight in tp.weight_views():
    print(weight.shape)
print("done")





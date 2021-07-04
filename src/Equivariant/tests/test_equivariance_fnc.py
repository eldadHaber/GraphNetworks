import torch
from torch_geometric.data import Data
from e3nn.nn.models.v2103.gate_points_networks import SimpleNetwork
from e3nn.util.test import assert_equivariant


import e3nn.o3
from e3nn.util.test import equivariance_error

tp = e3nn.o3.FullyConnectedTensorProduct("2x0e + 3x1o", "2x0e + 3x1o", "2x1o")

equivariance_error(
    tp,
    args_in=[tp.irreps_in1.randn(1, -1), tp.irreps_in2.randn(1, -1)],
    irreps_in=[tp.irreps_in1, tp.irreps_in2],
    irreps_out=[tp.irreps_out]
)

assert_equivariant(tp)

f = SimpleNetwork(
    "3x0e + 2x1o",
    "4x0e + 1x1o",
    max_radius=2.0,
    num_neighbors=3.0,
    num_nodes=5.0
)

def wrapper(pos, x):
    data = dict(pos=pos, x=x)
    return f(data)

print(assert_equivariant(
    wrapper,
    irreps_in=['cartesian_points', f.irreps_in],
    irreps_out=[f.irreps_out],
))
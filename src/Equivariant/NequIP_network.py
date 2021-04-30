"""model with self-interactions and gates
Exact equivariance to :math:`E(3)`
version of february 2021
"""
import math
from typing import Dict, Union
import numpy as np
import torch
from torch_geometric.data import Data
from torch_cluster import radius_graph
from torch_scatter import scatter

from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import FullyConnectedNet, Gate, ExtractIr, Activation
from e3nn.o3 import TensorProduct, FullyConnectedTensorProduct
from e3nn.util.jit import compile_mode
from torch.autograd import grad
import torch.nn.functional as F

from src.Equivariant.EQ_operations import Convolution, TvNorm


def smooth_cutoff(x):
    u = 2 * (x - 1)
    y = (math.pi * u).cos().neg().add(1).div(2)
    y[u > 0] = 0
    y[u < -1] = 1
    return y


def tp_path_exists(irreps_in1, irreps_in2, ir_out):
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False

class NequIP(torch.nn.Module):
    r"""equivariant neural network
    Parameters
    ----------
    irreps_in : `Irreps` or None
        representation of the input features
        can be set to ``None`` if nodes don't have input features
    irreps_hidden : `Irreps`
        representation of the hidden features
    irreps_out : `Irreps`
        representation of the output features
    irreps_node_attr : `Irreps` or None
        representation of the nodes attributes
        can be set to ``None`` if nodes don't have attributes
    irreps_edge_attr : `Irreps`
        representation of the edge attributes
        the edge attributes are :math:`h(r) Y(\vec r / r)`
        where :math:`h` is a smooth function that goes to zero at ``max_radius``
        and :math:`Y` are the spherical harmonics polynomials
    layers : int
        number of gates (non linearities)
    max_radius : float
        maximum radius for the convolution
    number_of_basis : int
        number of basis on which the edge length are projected
    radial_layers : int
        number of hidden layers in the radial fully connected network
    radial_neurons : int
        number of neurons in the hidden layers of the radial fully connected network
    num_neighbors : float
        typical number of nodes at a distance ``max_radius``
    num_nodes : float
        typical number of nodes in a graph
    """
    def __init__(
        self,
        irreps_in,
        irreps_hidden,
        irreps_out,
        irreps_node_attr,
        irreps_edge_attr,
        layers,
        max_radius,
        number_of_basis,
        radial_neurons,
        num_neighbors,
        num_nodes,
        reduce_output=True,
    ) -> None:
        super().__init__()
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.num_neighbors = num_neighbors
        self.num_nodes = num_nodes
        self.reduce_output = reduce_output

        self.irreps_in = o3.Irreps(irreps_in) if irreps_in is not None else None
        self.irreps_hidden = o3.Irreps(irreps_hidden)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr) if irreps_node_attr is not None else o3.Irreps("0e")
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)

        self.input_has_node_in = (irreps_in is not None)
        self.input_has_node_attr = (irreps_node_attr is not None)

        self.ext_z = ExtractIr(self.irreps_node_attr, '0e')

        act = {
            1: torch.nn.functional.silu,
            -1: torch.tanh,
        }
        act_gates = {
            1: torch.sigmoid,
            -1: torch.tanh,
        }
        nmax_atoms = 20
        embed_dim = 8
        self.node_embedder = torch.nn.Embedding(nmax_atoms,embed_dim)
        irreps = o3.Irreps("{:}x0e".format(embed_dim))

        self.tp_self = torch.nn.ModuleList()
        self.tp_self.append(o3.FullyConnectedTensorProduct(irreps,irreps,irreps))
        for _ in range(1,layers):
            tp = o3.FullyConnectedTensorProduct(self.irreps_hidden,self.irreps_hidden,self.irreps_hidden)
            self.tp_self.append(tp)
        # n_0e = o3.Irreps(self.irreps_hidden).count('0e')
        second_to_last_irrep = "16x0e"
        last_irrep = "1x0e"
        self.tp_self.append(o3.FullyConnectedTensorProduct(self.irreps_hidden,self.irreps_hidden,second_to_last_irrep))
        self.tp_self.append(o3.FullyConnectedTensorProduct(second_to_last_irrep,second_to_last_irrep,last_irrep))
        self.activation = Activation("16x0e", [torch.nn.functional.silu])
        # n_1e = o3.Irreps(self.irreps_hidden).count('0e')
        # n_1o = o3.Irreps(self.irreps_hidden).count('1o')


        self.convolutions = torch.nn.ModuleList()
        self.gates = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for _ in range(layers):
            irreps_scalars = o3.Irreps([(mul, ir) for mul, ir in self.irreps_hidden if ir.l == 0 and tp_path_exists(irreps, self.irreps_edge_attr, ir)])
            irreps_gated = o3.Irreps([(mul, ir) for mul, ir in self.irreps_hidden if ir.l > 0 and tp_path_exists(irreps, self.irreps_edge_attr, ir)])
            ir = "0e" if tp_path_exists(irreps, self.irreps_edge_attr, "0e") else "0o"
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated])

            gate = Gate(
                irreps_scalars, [act[ir.p] for _, ir in irreps_scalars],  # scalar
                irreps_gates, [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated  # gated tensors
            )
            conv = Convolution(
                irreps,
                self.irreps_node_attr,
                self.irreps_edge_attr,
                gate.irreps_in,
                radial_neurons,
                num_neighbors
            )
            self.norms.append(TvNorm(gate.irreps_in))
            irreps = gate.irreps_out
            self.convolutions.append(conv)
            self.gates.append(gate)

    def atomwise_interaction(self,x,index):
        '''
        This operation computes a self-interaction in an equivariant way. The self-interaction essentially mixes the channels for each particle, like a 1x1 filter convolution would do.
        We could also have these layers change the dimensions of the channels (number of features)
        Should we allow bias?

        So in our case I guess we have scalar and vector features, and we then need to do elementwise products to get the interaction? (no that will not work since elementwise operations have no learnable weights)
        :return:

        nnodes,nf = x.shape
        W = torch.randn((nf,nf))
        xmix = x @ W
        Will not work since it is not equivariant
        '''
        xmix = self.tp_self[index](x,x)

        return xmix


    def forward(self, data: Union[Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """evaluate the network
        Parameters
        ----------
        data : `torch_geometric.data.Data` or dict
            data object containing
            - ``pos`` the position of the nodes (atoms)
            - ``x`` the input features of the nodes, optional
            - ``z`` the attributes of the nodes, for instance the atom type, optional
            - ``batch`` the graph to which the node belong, optional
        """
        if 'batch' in data:
            batch = data['batch']
        else:
            batch = data['pos'].new_zeros(data['pos'].shape[0], dtype=torch.long)

        h = 0.1
        edge_index = radius_graph(data['pos'], self.max_radius, batch)
        edge_src = edge_index[0]
        edge_dst = edge_index[1]
        edge_vec = data['pos'][edge_src] - data['pos'][edge_dst]
        edge_sh = o3.spherical_harmonics(self.irreps_edge_attr, edge_vec, True, normalization='component')
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedded = soft_one_hot_linspace(
            x=edge_length,
            start=0.0,
            end=self.max_radius,
            number=self.number_of_basis,
            basis='bessel',
            cutoff=False
        ).mul(self.number_of_basis**0.5)
        edge_attr = smooth_cutoff(edge_length / self.max_radius)[:, None] * edge_sh

        if self.input_has_node_in and 'x' in data:
            assert self.irreps_in is not None
            x = data['x']
        else:
            assert self.irreps_in is None
            x = data['pos'].new_ones((data['pos'].shape[0], 1))

        if self.input_has_node_attr and 'z' in data:
            z = data['z']
        else:
            assert self.irreps_node_attr == o3.Irreps("0e")
            z = data['pos'].new_ones((data['pos'].shape[0], 1))

        # scalar_z = self.ext_z(z)
        edge_features = edge_length_embedded

        x = self.node_embedder(x.to(dtype=torch.int64)).squeeze()
        x = self.atomwise_interaction(x,0)

        for i,(conv,norm,gate) in enumerate(zip(self.convolutions,self.norms,self.gates)):
            y = conv(x, z, edge_src, edge_dst, edge_attr, edge_features)
            y = norm(y)
            y = gate(y)
            if y.shape == x.shape:
                y = self.atomwise_interaction(y, i)
                x = x + h*y
            else:
                x = y
            # print(f'mean(abs(x))={torch.abs(x).mean():2.2f},norm={x.norm():2.2f}')
        x = self.atomwise_interaction(x,-2)
        x = self.activation(x)
        x = self.atomwise_interaction(x,-1)

        if self.reduce_output:
            return scatter(x, batch, dim=0).div(self.num_nodes**0.5)
        else:
            return x


def test_equivariance(model,data=None):
    if data is None:
        na = 7 #Number of atoms
        nb = 1 #Number of batches
        nf = 1 #Number of features
        # irreps_out = "{:}x1e".format(na)
        R = torch.randn((nb, na, 3), dtype=torch.float32)
        F = torch.randn((1,4,3),dtype=torch.float32)
        node_attr = torch.randint(0,10,(nb,na,nf))

        total_params = sum(p.numel() for p in model.parameters())
        print('Number of parameters ', total_params)


        data = {'pos': R.squeeze(),
                'x': node_attr.squeeze(0)
                }

    #Test equivariance
    rot = o3.rand_matrix(1)
    D_in = irreps_in.D_from_matrix(rot)
    D_out = irreps_out.D_from_matrix(rot)

    # rotate before
    data2 = {'pos': (R @ rot.transpose(1,2)).squeeze(0),
            'x': node_attr.squeeze(0)
            }
    f_before = model(data2)
    # f_before = model(node_features @ D_in.T, R @ rot.transpose(1,2))

    # rotate after
    f_after = model(data) @ D_out.transpose(1,2)

    is_equivariant = torch.allclose(f_before, f_after, rtol=1e-4, atol=1e-4)
    assert is_equivariant,"Network is not equivariant"
    print("network is equivariant")
    return



if __name__ == '__main__':
    # na = 7 #Number of atoms
    # nb = 1 #Number of batches
    # nf = 1 #Number of features
    # # irreps_out = "{:}x1e".format(na)
    # R = torch.randn((nb, na, 3), dtype=torch.float32)
    # F = torch.randn((1,4,3),dtype=torch.float32)
    # node_attr = torch.randint(0,10,(nb,na,nf))

    data = np.load('../../../../data/MD/MD17/aspirin_dft.npz')
    E = torch.from_numpy(data['E']).to(dtype=torch.float32)
    Force = torch.from_numpy(data['F']).to(dtype=torch.float32)
    R = torch.from_numpy(data['R']).to(dtype=torch.float32)
    z = torch.from_numpy(data['z']).to(dtype=torch.float32)

    nd,na,_ = R.shape

    irreps_in = o3.Irreps("1x0e")
    irreps_hidden = o3.Irreps("64x0e+64x0o+64x1e+64x1o")
    irreps_out = o3.Irreps("1x0e")
    irreps_node_attr = o3.Irreps("1x0e")
    irreps_edge_attr = o3.Irreps("1x0e+1x1o")
    layers = 6
    max_radius = 5
    number_of_basis = 8
    radial_layers = 1
    radial_neurons = 8
    num_neighbors = 15
    num_nodes = na
    model = NequIP(irreps_in=irreps_in, irreps_hidden=irreps_hidden, irreps_out=irreps_out, irreps_node_attr=irreps_node_attr, irreps_edge_attr=irreps_edge_attr, layers=layers, max_radius=max_radius,
                    number_of_basis=number_of_basis, radial_layers=radial_layers, radial_neurons=radial_neurons, num_neighbors=num_neighbors, num_nodes=num_nodes)

    total_params = sum(p.numel() for p in model.parameters())
    print('Number of parameters ', total_params)

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)


    Ri = R[0,:,:]
    Ri.requires_grad_(True)
    Fi = Force[0,:,:]


    data = {'pos': Ri,
            'x': z[:,None]
            }

    test_equivariance(model)


    niter = 100000
    for i in range(niter):
        optim.zero_grad()
        E_pred = model(data)
        F_pred = -grad(E_pred, Ri, create_graph=True)[0].requires_grad_(True)
        print(f'mean(abs(F_pred))={torch.abs(F_pred).mean():2.2f}')
        loss = F.mse_loss(F_pred, Fi)
        MAE = torch.mean(torch.abs(F_pred - Fi)).detach()
        loss.backward()
        optim.step()
        print(f'{i:}, loss:{loss.detach():2.2e}, MAE:{MAE:2.2f}')
    print('done')

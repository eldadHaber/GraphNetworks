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
from e3nn.nn import FullyConnectedNet, Gate, ExtractIr
from e3nn.o3 import TensorProduct, FullyConnectedTensorProduct
from e3nn.util.jit import compile_mode
from torch.autograd import grad
import torch.nn.functional as F


@compile_mode('script')
class Convolution(torch.nn.Module):
    r"""equivariant convolution
    Parameters
    ----------
    irreps_in : `Irreps`
        representation of the input node features
    irreps_node_attr : `Irreps`
        representation of the node attributes
    irreps_edge_attr : `Irreps`
        representation of the edge attributes
    irreps_out : `Irreps` or None
        representation of the output node features
    number_of_edge_features : int
        number of scalar (0e) features of the edge used to feed the FC network
    radial_layers : int
        number of hidden layers in the radial fully connected network
    radial_neurons : int
        number of neurons in the hidden layers of the radial fully connected network
    num_neighbors : float
        typical number of nodes convolved over
    """
    def __init__(
        self,
        irreps_in,
        irreps_node_attr,
        irreps_edge_attr,
        irreps_out,
        number_of_edge_features,
        radial_layers,
        radial_neurons,
        num_neighbors
    ) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_out = o3.Irreps(irreps_out)
        self.num_neighbors = num_neighbors

        self.sc = FullyConnectedTensorProduct(self.irreps_in, self.irreps_node_attr, self.irreps_out)

        self.lin1 = FullyConnectedTensorProduct(self.irreps_in, self.irreps_node_attr, self.irreps_in)

        irreps_mid = []
        instructions = []
        for i, (mul, ir_in) in enumerate(self.irreps_in):
            for j, (_, ir_edge) in enumerate(self.irreps_edge_attr):
                for ir_out in ir_in * ir_edge:
                    if ir_out in self.irreps_out:
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, 'uvu', True))
        irreps_mid = o3.Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()

        instructions = [
            (i_1, i_2, p[i_out], mode, train)
            for i_1, i_2, i_out, mode, train in instructions
        ]

        tp = TensorProduct(
            self.irreps_in,
            self.irreps_edge_attr,
            irreps_mid,
            instructions,
            internal_weights=False,
            shared_weights=False,
        )
        self.fc = FullyConnectedNet(
            [number_of_edge_features] + radial_layers * [radial_neurons] + [tp.weight_numel],
            torch.nn.functional.silu
        )
        self.tp = tp

        self.lin2 = FullyConnectedTensorProduct(irreps_mid, self.irreps_node_attr, self.irreps_out)

    def forward(self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_features) -> torch.Tensor:
        weight = self.fc(edge_features)

        x = node_input

        s = self.sc(x, node_attr)
        x = self.lin1(x, node_attr)

        edge_features = self.tp(x[edge_src], edge_attr, weight)
        x = scatter(edge_features, edge_dst, dim=0, dim_size=x.shape[0]).div(self.num_neighbors**0.5)

        x = self.lin2(x, node_attr)

        c_s, c_x = math.sin(math.pi / 8), math.cos(math.pi / 8)
        m = self.sc.output_mask
        c_x = (1 - m) + c_x * m
        return c_s * s + c_x * x


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


class Compose(torch.nn.Module):
    def __init__(self, first, second):
        super().__init__()
        self.first = first
        self.second = second
        self.irreps_in = self.first.irreps_in
        self.irreps_out = self.second.irreps_out

    def forward(self, *input):
        x = self.first(*input)
        return self.second(x)



class GraphNet_EQ(torch.nn.Module):
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
        radial_layers,
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
        embed_dim = 100
        nmax_atoms = 20
        self.irreps_in = o3.Irreps(irreps_in) if irreps_in is not None else None
        self.irreps_hidden = o3.Irreps(irreps_hidden)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr) if irreps_node_attr is not None else o3.Irreps("0e")
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)

        self.input_has_node_in = (irreps_in is not None)
        self.input_has_node_attr = (irreps_node_attr is not None)

        self.ext_z = ExtractIr(self.irreps_node_attr, '0e')
        number_of_edge_features = number_of_basis + 2 * self.irreps_node_attr.count('0e')

        irreps = self.irreps_in if self.irreps_in is not None else o3.Irreps("0e")
        self.node_embedder = torch.nn.Embedding(nmax_atoms,embed_dim)
        irreps = o3.Irreps("{:}x0e".format(embed_dim))

        act = {
            1: torch.nn.functional.silu,
            -1: torch.tanh,
        }
        act_gates = {
            1: torch.sigmoid,
            -1: torch.tanh,
        }

        self.tp_self = torch.nn.ModuleList()
        for _ in range(layers):
            tp = o3.FullyConnectedTensorProduct(self.irreps_hidden,self.irreps_hidden,self.irreps_hidden)
            # tp = TensorProduct(
            #     self.irreps_hidden, self.irreps_hidden, self.irreps_hidden,
            #     [
            #         (0, 0, 0, "uuu", True),
            #         (1, 1, 0, "uuu", True),
            #         (2, 2, 0, "uuu", True),
            #         (3, 3, 0, "uuu", True),
            #         (0, 1, 1, "uuu", True),
            #         (2, 3, 1, "uuu", True),
            #         (2, 2, 2, "uuu", True),
            #         (0, 2, 2, "uuu", True),
            #         (1, 3, 2, "uuu", True),
            #         (2, 3, 3, "uuu", True),
            #         (1, 2, 3, "uuu", True),
            #         (0, 3, 3, "uuu", True),
            #     ])
            self.tp_self.append(tp)

        self.layers = torch.nn.ModuleList()

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
                number_of_edge_features,
                radial_layers,
                radial_neurons,
                num_neighbors
            )
            irreps = gate.irreps_out
            self.layers.append(Compose(conv, gate))

        self.layers.append(
            Convolution(
                irreps,
                self.irreps_node_attr,
                self.irreps_edge_attr,
                self.irreps_out,
                number_of_edge_features,
                radial_layers,
                radial_neurons,
                num_neighbors
            )
        )

    def doublelayer(self,x,c1,c2,g1,g2,g3):
        x = g1(x)
        x = c1(x)
        x = g2(x)
        x = c2(x)
        x = g3(x)
        return x

    def nodeGrad(self, x, esrc,edst):
        W = torch.randn((1,x.shape[-1]))
        g = W * (x[edst, :] - x[esrc, :])
        return g

    def nodeAve(self, x, esrc, edst):
        W = torch.randn((1,x.shape[-1]))
        g = (x[edst, :] + x[esrc, :]) / 2.0

        # edge_features = self.tp(x[esrc], edge_attr, weight)
        # x = scatter(edge_features, edge_dst, dim=0, dim_size=x.shape[0]).div(self.num_neighbors**0.5)


        return g

    def edgeDiv(self, x, g, esrc, edst):
        '''
        Combines all the edge information leading into each node in a way similar to how a divergence operation would work.
        Hence g will have the last dimension equal to number of edges, while x will have the dimension equal to the number of nodes.
        :param g:
        :param W:
        :return:
        '''

        W = torch.randn((1,g.shape[-1]))
        x *= 0
        x.index_add_(0, edst, W*g)
        x.index_add_(0, esrc, -W*g)
        return x

    def edgeAve(self, x, g, esrc, edst):
        '''
        Combines all the edge information leading into each node in a way similar to how an average operation would work.
        Hence g will have the last dimension equal to number of edges, while x will have the dimension equal to the number of nodes.
        :param g:
        :param W:
        :return:
        '''
        x *= 0
        x.index_add_(0, edst, g)
        x.index_add_(0, esrc, g)
        x /= 2
        return x

    def atomwise_interaction(self,x):
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
        xmix = self.tp_self[0](x,x)

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
            basis='gaussian',
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

        scalar_z = self.ext_z(z)
        edge_features = torch.cat([edge_length_embedded, scalar_z[edge_src], scalar_z[edge_dst]], dim=1)

        x = self.node_embedder(x.to(dtype=torch.int64)).squeeze()
        for i,lay in enumerate(self.layers):
            y = lay(x, z, edge_src, edge_dst, edge_attr, edge_features)
            if y.shape == x.shape:
                x = x + h*y
                # edge_ft = self.nodeAve(x, edge_src, edge_dst)
                # x = self.edgeDiv(x, edge_ft, edge_src, edge_dst)
                # x = self.atomwise_interaction(x)
            else:
                x = y
            # print(f'mean(abs(x))={torch.abs(x).mean():2.2f},norm={x.norm():2.2f}')

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
    irreps_hidden = o3.Irreps("50x0e+50x0o+50x1e+50x1o")
    irreps_out = o3.Irreps("1x0e")
    irreps_node_attr = o3.Irreps("1x0e")
    irreps_edge_attr = o3.Irreps("1x0e+1x1o")
    layers = 6
    max_radius = 5
    number_of_basis = 10
    radial_layers = 2
    radial_neurons = 20
    num_neighbors = 15
    num_nodes = na
    model = GraphNet_EQ(irreps_in=irreps_in, irreps_hidden=irreps_hidden, irreps_out=irreps_out, irreps_node_attr=irreps_node_attr, irreps_edge_attr=irreps_edge_attr, layers=layers, max_radius=max_radius,
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

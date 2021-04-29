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
from src.Equivariant.EQ_operations import *



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
        irreps_in_n,
        irreps_in_e,
        irreps_hidden,
        irreps_out_n,
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
        self.nlayers = layers
        embed_dim = 8
        nmax_atoms = 20
        self.h = 0.1
        self.irreps_in_n = o3.Irreps(irreps_in_n) if irreps_in_n is not None else None
        self.irreps_in_e = o3.Irreps(irreps_in_e) if irreps_in_e is not None else None
        self.irreps_hidden = o3.Irreps(irreps_hidden)
        self.irreps_out_n = o3.Irreps(irreps_out_n)

        self.node_embedder = torch.nn.Embedding(nmax_atoms,embed_dim)
        irreps_n = o3.Irreps("{:}x0e".format(embed_dim))
        irreps_e = self.irreps_in_e

        self.doublelayers = torch.nn.ModuleList()
        dl_xn = DoubleLayer(irreps_n, self.irreps_hidden, self.irreps_hidden)
        dl_xe = DoubleLayer(irreps_e, self.irreps_hidden, self.irreps_hidden)
        self.doublelayers.append(dl_xn)
        self.doublelayers.append(dl_xe)

        self.filters = torch.nn.ModuleList()

        self.filters.append(Filter(number_of_basis,radial_layers,radial_neurons,dl_xe.irreps_out))
        self.filters.append(Filter(number_of_basis,radial_layers,radial_neurons,dl_xe.irreps_out))
        # self.filters.append(FullyConnectedNet(
        #     [number_of_basis] + radial_layers * [radial_neurons] + [dl_xe.irreps_out.dim], torch.nn.functional.silu))
        # self.filters.append(FullyConnectedNet(
        #     [number_of_basis] + radial_layers * [radial_neurons] + [dl_xe.irreps_out.dim], torch.nn.functional.silu))
        irreps = dl_xn.irreps_out + 2 * dl_xe.irreps_out
        self.cats = torch.nn.ModuleList()
        cat = Concatenate(irreps)
        self.cats.append(cat)
        irreps = cat.irreps_out

        for _ in range(layers):
            # self.filters.append(FullyConnectedNet(
            #     [number_of_basis] + radial_layers * [radial_neurons] + [irreps.dim],
            #     torch.nn.functional.silu))
            # self.filters.append(FullyConnectedNet(
            #     [number_of_basis] + radial_layers * [radial_neurons] + [irreps.dim],
            #     torch.nn.functional.silu))
            self.filters.append(Filter(number_of_basis, radial_layers, radial_neurons, irreps))
            self.filters.append(Filter(number_of_basis, radial_layers, radial_neurons, irreps))
            cat = Concatenate(2*irreps)
            self.cats.append(cat)
            irreps2 = cat.irreps_out
            dl = DoubleLayer(irreps2, self.irreps_hidden, irreps2)
            self.doublelayers.append(dl)
            irreps2 = dl.irreps_out
            self.filters.append(Filter(number_of_basis, radial_layers, radial_neurons, irreps2))
            # self.filters.append(FullyConnectedNet(
            #     [number_of_basis] + radial_layers * [radial_neurons] + [irreps2.dim],
            #     torch.nn.functional.silu))


        self.si_close = SelfInteraction(irreps,self.irreps_out_n)


    def nodeGrad(self, x, esrc,edst,W):
        # W = torch.randn((1,x.shape[-1]))
        g = W * (x[edst, :] - x[esrc, :])
        return g

    def nodeAve(self, x, esrc, edst, W):
        # W = torch.randn((1,x.shape[-1]))
        g = W * (x[edst, :] + x[esrc, :]) / 2.0
        return g

    def edgeDiv(self, g, nnodes, esrc, edst, W):
        '''
        Combines all the edge information leading into each node in a way similar to how a divergence operation would work.
        Hence g will have the last dimension equal to number of edges, while x will have the dimension equal to the number of nodes.
        :param g:
        :param W:
        :return:
        '''

        # W = torch.randn((1,g.shape[-1]))
        x = torch.zeros((nnodes,g.shape[-1]),dtype=torch.float32,device=g.device)
        x.index_add_(0, edst, W*g)
        x.index_add_(0, esrc, -W*g)
        return x

    def edgeAve(self, g, nnodes, esrc, edst, W):
        '''
        Combines all the edge information leading into each node in a way similar to how an average operation would work.
        Hence g will have the last dimension equal to number of edges, while x will have the dimension equal to the number of nodes.
        :param g:
        :param W:
        :return:
        '''
        x = torch.zeros((nnodes,g.shape[-1]),dtype=torch.float32,device=g.device)
        x.index_add_(0, edst, W*g)
        x.index_add_(0, esrc, W*g)
        x /= 2
        return x



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

        edge_index = radius_graph(data['pos'], self.max_radius, batch)
        edge_src = edge_index[0]
        edge_dst = edge_index[1]
        edge_vec = data['pos'][edge_src] - data['pos'][edge_dst]
        edge_sh = o3.spherical_harmonics(self.irreps_in_e, edge_vec, True, normalization='component')
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedded = soft_one_hot_linspace(
            x=edge_length,
            start=0.0,
            end=self.max_radius,
            number=self.number_of_basis,
            basis='bessel',
            cutoff=False
        ).mul(self.number_of_basis**0.5)
        xe = smooth_cutoff(edge_length / self.max_radius)[:, None] * edge_sh
        xn = data['x']
        nnodes = xn.shape[0]

        xn = self.node_embedder(xn.to(dtype=torch.int64)).squeeze()
        xn = self.doublelayers[0](xn)
        xe = self.doublelayers[1](xe)


        Wxe = self.filters[0](edge_length_embedded)
        xe_eD = self.edgeDiv(xe,nnodes,edge_src,edge_dst,Wxe)

        Wxe = self.filters[1](edge_length_embedded)
        xe_eA = self.edgeAve(xe,nnodes,edge_src,edge_dst,Wxe)

        xn = self.cats[0]([xn,xe_eD,xe_eA], dim=1)

        for i in range(self.nlayers):

            W = self.filters[i*3+2](edge_length_embedded)
            gradX = self.nodeGrad(xn,edge_src,edge_dst,W)

            W = self.filters[i*3+3](edge_length_embedded)
            aveX = self.nodeAve(xn,edge_src,edge_dst,W)

            dxe = self.cats[i+1]([gradX, aveX], dim=1)
            assert ~dxe.isnan().any()
            dxe = self.doublelayers[i+2](dxe)
            assert ~dxe.isnan().any()
            dxe = self.cats[i+1].reverse_idx(dxe, dim=1)

            W = self.filters[i*3+4](edge_length_embedded)
            xn_div = self.edgeDiv(dxe[:,:dxe.shape[-1]//2],nnodes,edge_src,edge_dst,W[:,:dxe.shape[-1]//2])
            assert ~xn_div.isnan().any()
            xn_ave = self.edgeAve(dxe[:,dxe.shape[-1]//2:],nnodes,edge_src,edge_dst,W[:,dxe.shape[-1]//2:])
            assert ~xn_ave.isnan().any()

            xn = xn - self.h * (xn_div + xn_ave)
            assert ~xn.isnan().any()

        # if (xn < 1e-6).any():
        #     print("stop here")
        xn = self.si_close(xn)


        if self.reduce_output:
            return scatter(xn, batch, dim=0).div(self.num_nodes**0.5)
        else:
            return xn


def test_equivariance(model,irreps_in,irreps_out,data=None):
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

    irreps_in_n = o3.Irreps("1x0e")
    irreps_in_e = o3.Irreps("1x0e+1x1o")
    irreps_hidden = o3.Irreps("10x0e+10x1o")
    irreps_out_n = o3.Irreps("1x0e")
    layers = 2
    max_radius = 5
    number_of_basis = 10
    radial_layers = 2
    radial_neurons = 20
    num_neighbors = 15
    num_nodes = na
    model = GraphNet_EQ(irreps_in_n=irreps_in_n, irreps_in_e=irreps_in_e, irreps_hidden=irreps_hidden, irreps_out_n=irreps_out_n, layers=layers, max_radius=max_radius,
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

    # test_equivariance(model,irreps_in_n,irreps_out_n)


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

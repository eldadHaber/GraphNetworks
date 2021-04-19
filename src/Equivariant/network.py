import torch
# import torch_cluster
from torch_cluster import radius_graph
from torch_scatter import scatter
from e3nn.o3 import FullyConnectedTensorProduct
from e3nn import o3



class eq_network(torch.nn.Module):
    def __init__(self,n_atoms, irreps_out) -> None:
        super().__init__()
        n_embeddings = 100
        basis_functions = 3
        self.irreps_sh = o3.Irreps.spherical_harmonics(1)
        self.irreps_out = irreps_out
        irreps_mid = o3.Irreps("64x0e + 24x1e + 24x1o + 16x2e + 16x2o")

        self.tp1 = FullyConnectedTensorProduct(
            irreps_in1=self.irreps_sh,
            irreps_in2=self.irreps_sh,
            irreps_out=irreps_mid,
        )
        self.tp2 = FullyConnectedTensorProduct(
            irreps_in1=irreps_mid,
            irreps_in2=self.irreps_sh,
            irreps_out=irreps_out,
        )

        self.embedding = torch.nn.Embedding(n_embeddings, basis_functions)
        self.atomwise_interaction1 = o3.ElementwiseTensorProduct(self.irreps_sh,self.irreps_sh)
        return



    def forward(self, node_features, r) -> torch.Tensor:
        nb,na,ndr = r.shape
        ndf = node_features.shape[-1]
        r_vec = r.reshape(-1, ndr)
        batch_vec = torch.arange(nb).repeat_interleave(na)
        node_features_vec = node_features.reshape(-1,ndf)

        num_neighbors = 2  # typical number of neighbors
        num_nodes = 4  # typical number of nodes

        edge_src, edge_dst = radius_graph(
            x=r_vec,
            r=3.1,
            batch=batch_vec
        )  # tensors of indices representing the graph
        edge_vec = r_vec[edge_src] - r_vec[edge_dst]
        edge_sh = o3.spherical_harmonics(
            l=self.irreps_sh,
            x=edge_vec,
            normalize=False,  # here we don't normalize otherwise it would not be a polynomial
            normalization='component'
        )

        # First we create an embedding of the node features
        node_features_embedded = self.embedding(node_features_vec)

        # Next we get a spherical harmonic representation of the embedding.
        node_features_embedded_sh = o3.spherical_harmonics(self.irreps_sh, node_features_embedded, normalize=False, normalization='component')
        # node_features_embedded_sh = o3.spherical_harmonics(self.irreps_sh, node_features_vec.to(dtype=torch.float32), normalize=False, normalization='component')

        # We convert edge features into node features
        node_features = scatter(edge_sh, edge_dst, dim=0).div(num_neighbors**0.5)

        # We concatenate the nodes
        node_features = torch.cat((node_features[:,None,:], node_features_embedded_sh),dim=1)


        # For each edge, tensor product the features on the source node with the spherical harmonics
        edge_features = self.tp1(node_features[edge_src], edge_sh[:,None,:])
        node_features = scatter(edge_features, edge_dst, dim=0).div(num_neighbors**0.5)

        edge_features = self.tp2(node_features[edge_src], edge_sh[:,None,:])
        node_features = scatter(edge_features, edge_dst, dim=0).div(num_neighbors**0.5)

        # For each graph, all the node's features are summed'
        # res = scatter(node_features, batch, dim=0).div(num_nodes ** 0.5)
        node_features = torch.sum(node_features,dim=1)
        node_features_reshaped = node_features.reshape(nb,na,ndr)
        # edge_features_reshaped = edge_features.reshape(nb,-1,nd)
        return node_features_reshaped



if __name__ == '__main__':
    na = 7 #Number of atoms
    nb = 2 #Number of batches
    nf = 9 #Number of features
    irreps_in = o3.Irreps("1x1e")
    irreps_in2 = o3.Irreps("9x0e")
    irreps_out = o3.Irreps("1x1e")
    # irreps_out = "{:}x1e".format(na)

    node_features = torch.randint(0,10,(nb,na,nf))
    R = torch.randn((nb, na, 3), dtype=torch.float32)
    F = torch.randn((1,4,3),dtype=torch.float32)

    model = eq_network(n_atoms=na,irreps_out=irreps_out)

    #First we propagate the data through the model
    F_pred = model(node_features,R)

    #Test equivariance
    rot = o3.rand_matrix(nb)
    D_in = irreps_in.D_from_matrix(rot)
    D_in2 = irreps_in2.D_from_matrix(rot)
    D_out = irreps_out.D_from_matrix(rot)

    # rotate before
    f_before = model(node_features, R @ rot.transpose(1,2))
    # f_before = model(node_features @ D_in.T, R @ rot.transpose(1,2))

    # rotate after
    f_after = model(node_features, R) @ D_out.transpose(1,2)

    torch.allclose(f_before, f_after, rtol=1e-4, atol=1e-4)


    print("done")


    # error = f(rotated_data) - f(data)
    # print(f"Equivariance error = {error.abs().max().item():.1e}")

#Define a network

#Define a simple data input (data, labels)

#Ensure that the network is equivariant
#Try to train a network to hit those labels


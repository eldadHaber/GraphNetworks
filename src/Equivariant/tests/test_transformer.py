import torch
# import torch_cluster
from torch_cluster import radius_graph
from torch_scatter import scatter
from e3nn import o3, nn
from e3nn.math import soft_one_hot_linspace, soft_unit_step
import matplotlib.pyplot as plt


def transformer(f, pos):
    edge_src, edge_dst = radius_graph(pos, max_radius)
    edge_vec = pos[edge_src] - pos[edge_dst]
    edge_length = edge_vec.norm(dim=1)

    edge_length_embedded = soft_one_hot_linspace(edge_length, 0.0, max_radius, number_of_basis, 'smooth_finite', False)
    edge_length_embedded = edge_length_embedded.mul(number_of_basis**0.5)
    edge_weight_cutoff = soft_unit_step(10 * (1 - edge_length / max_radius))

    edge_sh = o3.spherical_harmonics(irreps_sh, edge_vec, True, normalization='component')

    q = h_q(f)
    k = tp_k(f[edge_src], edge_sh, fc_k(edge_length_embedded))
    v = tp_v(f[edge_src], edge_sh, fc_v(edge_length_embedded))

    # dot = o3.FullyConnectedTensorProduct(irreps_query, irreps_key, "0e")
    exp = edge_weight_cutoff[:, None] * dot(q[edge_dst], k).exp()
    z = scatter(exp, edge_dst, dim=0, dim_size=len(f))
    z[z == 0] = 1
    alpha = exp / z[edge_dst]

    return scatter(alpha.sqrt() * v, edge_dst, dim=0, dim_size=len(f))



# Just define arbitrary irreps
irreps_input = o3.Irreps("10x0e + 5x1o + 2x2e")
irreps_query = o3.Irreps("11x0e + 4x1o")
irreps_key = o3.Irreps("12x0e + 3x1o")
irreps_output = o3.Irreps("14x0e + 6x1o")  # also irreps of the values


# num_nodes = 20
#
# pos = torch.randn(num_nodes, 3)
# f = irreps_input.randn(num_nodes, -1)
#
# # create graph
# max_radius = 1.3
# edge_src, edge_dst = radius_graph(pos, max_radius)
# edge_vec = pos[edge_src] - pos[edge_dst]
# edge_length = edge_vec.norm(dim=1)
#
#
# ir_in = o3.Irreps("2x0e + 4x1o")
# ir_qr = o3.Irreps("3x0e + 1x1e")
# h1 = o3.Linear(ir_in,ir_qr)
#
#
# h_q = o3.Linear(irreps_input, irreps_query)
#
# number_of_basis = 10
# edge_length_embedded = soft_one_hot_linspace(edge_length, 0.0, max_radius, number_of_basis, 'smooth_finite', False)
# edge_length_embedded = edge_length_embedded.mul(number_of_basis**0.5)
#
# edge_weight_cutoff = soft_unit_step(10 * (1 - edge_length / max_radius))
#
#
# irreps_sh = o3.Irreps.spherical_harmonics(3)
# edge_sh = o3.spherical_harmonics(irreps_sh, edge_vec, True, normalization='component')
#
# tp_k = o3.FullyConnectedTensorProduct(irreps_input, irreps_sh, irreps_key, shared_weights=False)
# fc_k = nn.FullyConnectedNet([number_of_basis, 16, tp_k.weight_numel], act=torch.nn.functional.silu)
#
# tp_v = o3.FullyConnectedTensorProduct(irreps_input, irreps_sh, irreps_output, shared_weights=False)
# fc_v = nn.FullyConnectedNet([number_of_basis, 16, tp_v.weight_numel], act=torch.nn.functional.silu)
#
#
# # compute the queries (per node), keys (per edge) and values (per edge)
# q = h_q(f)
# k = tp_k(f[edge_src], edge_sh, fc_k(edge_length_embedded))
# v = tp_v(f[edge_src], edge_sh, fc_v(edge_length_embedded))
#
#
#
# dot = o3.FullyConnectedTensorProduct(irreps_query, irreps_key, "0e")

# # compute the softmax (per edge)
# exp = edge_weight_cutoff[:, None] * dot(q[edge_dst], k).exp()  # compute the numerator
# z = scatter(exp, edge_dst, dim=0, dim_size=len(f))  # compute the denominator (per nodes)
# z[z == 0] = 1  # to avoid 0/0 when all the neighbors are exactly at the cutoff
# alpha = exp / z[edge_dst]
#
# # compute the outputs (per node)
# f_out = scatter(alpha.sqrt() * v, edge_dst, dim=0, dim_size=len(f))



f = irreps_input.randn(3, -1)

xs = torch.linspace(-1.3, -1.0, 200)
outputs = []

for x in xs:
    pos = torch.tensor([
        [0.0, 0.5, 0.0],       # this node always sees...
        [0.0, -0.5, 0.0],      # ...this node
        [x.item(), 0.0, 0.0],  # this node moves slowly
    ])

    with torch.no_grad():
        outputs.append(transformer(f, pos))

outputs = torch.stack(outputs)
plt.plot(xs, outputs[:, 0, [0, 1, 14, 15, 16]], 'k')  # plots 2 scalars and 1 vector
plt.plot(xs, outputs[:, 1, [0, 1, 14, 15, 16]], 'g')
plt.plot(xs, outputs[:, 2, [0, 1, 14, 15, 16]], 'r')
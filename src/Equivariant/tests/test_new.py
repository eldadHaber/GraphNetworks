import torch
from e3nn.nn.models.v2103.gate_points_networks import NetworkForAGraphWithAttributes
from torch_cluster import radius_graph

max_radius = 3.0

net = NetworkForAGraphWithAttributes(
    irreps_node_input="0e+1e",
    irreps_node_attr="0e+1e",
    irreps_edge_attr="0e+1e",  # attributes in extra of the spherical harmonics
    irreps_node_output="0e+1e",
    max_radius=max_radius,
    num_neighbors=4.0,
    num_nodes=5.0,
)

num_nodes = 5
pos = torch.randn(num_nodes, 4)
edge_index = radius_graph(pos, max_radius)
num_edges = edge_index.shape[1]

net({
    'pos': pos,
    'edge_index': edge_index,
    'node_input': torch.randn(num_nodes, 4),
    'node_attr': torch.randn(num_nodes, 4),
    'edge_attr': torch.randn(num_edges, 4),
})

print("done")
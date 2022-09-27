import torch_geometric
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import ArgumentParser
from torch_geometric.nn import GCNConv
from ..model.gcl import E_GCL, unsorted_segment_sum

class E_GCL_mask(E_GCL):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_attr_dim=0, act_fn=nn.ReLU(), recurrent=True, coords_weight=1.0, attention=False):
        E_GCL.__init__(self, input_nf, output_nf, hidden_nf, edges_in_d=edges_in_d, nodes_att_dim=nodes_attr_dim, act_fn=act_fn, recurrent=recurrent, coords_weight=coords_weight, attention=attention)

        del self.coord_mlp
        self.act_fn = act_fn

    def coord_model(self, coord, edge_index, coord_diff, edge_feat, edge_mask):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat) * edge_mask
        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        coord += agg*self.coords_weight
        return coord

    def forward(self, h, edge_index, coord, node_mask, edge_mask, edge_attr=None, node_attr=None, n_nodes=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)

        edge_feat = edge_feat * edge_mask

        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat, edge_mask)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr

class EGNN_3d_enc(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, 
                 coords_weight=1.0, attention=False, node_attr=1):
        super(EGNN_3d_enc, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers

        ### Encoder
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        self.node_attr = node_attr
        if node_attr:
            n_node_attr = in_node_nf
        else:
            n_node_attr = 0
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL_mask(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, nodes_attr_dim=n_node_attr, act_fn=act_fn, recurrent=True, coords_weight=coords_weight, attention=attention))

        self.node_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                      act_fn,
                                      nn.Linear(self.hidden_nf, self.hidden_nf))

        self.graph_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                       act_fn,
                                       nn.Linear(self.hidden_nf, 1))
        self.to(self.device)

    def forward(self, h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes):
        h = self.embedding(h0)
        coord = 0
        for i in range(0, self.n_layers):
            if self.node_attr:
                h, coord, edge_attr = self._modules["gcl_%d" % i](h, edges, x, node_mask, edge_mask, edge_attr=edge_attr, node_attr=h0, n_nodes=n_nodes)
            else:
                h, coord, edge_attr = self._modules["gcl_%d" % i](h, edges, x, node_mask, edge_mask, edge_attr=edge_attr,
                                                      node_attr=None, n_nodes=n_nodes)
        return coord

class simple_gcn(nn.Module):
    def __init__(self, in_dim, emb_dim, out_dim, depth = 5):
        super(simple_gcn, self).__init__()
        self.in_dim = in_dim
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.depth = depth
        self.gcn_convs= []
        self.dropout = []   
        for i in range(self.depth):
            if i == 0:
                self.gcn_convs.append(GCNConv(self.in_dim, self.emb_dim))
            elif i == self.depth -1:
                self.gcn_convs.append(GCNConv(self.emb_dim, self.out_dim))
            else:
                self.gcn_convs.append(GCNConv(self.emb_dim, self.emb_dim))
            

    def forward(self, data, tensor_3d, add_3d):
        x, edge_idx = data.x, data.edge_index
        x = torch.cat((x, tensor_3d), dim = 1) if add_3d else x 
        for i in range(self.depth):
            h = self.gcn_convs[i](x, edge_idx)
            if i != self.depth - 1:
                h = F.dropout(F.relu(h), 0.2)
            else: h = F.relu(h)

        return h.squeeze(1)

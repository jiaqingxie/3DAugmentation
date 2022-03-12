
import torch.nn as nn

import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform

from conv import GNN_node, GNN_node_Virtualnode, GNN_node_discriminator

from torch_scatter import scatter_mean
class Discriminatorfor3D(torch.nn.Module):
    def __init__(self, num_tasks = 2, num_layers = 5, emb_dim = 300,nhead=10,
                    gnn_type = 'gin', virtual_node = True, residual = False, drop_ratio = 0, JK = "last", graph_pooling = "sum"):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''
        super(Discriminatorfor3D, self).__init__()
        self.threed2embedding=nn.Linear(3,emb_dim)
        self.threedEncoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(emb_dim,nhead=nhead,dim_feedforward=4*emb_dim),num_layers)
        
        self.classifier=nn.Linear(emb_dim,num_tasks)
        self.activation=nn.Softmax(dim=-1)
    def forward(self, xyz):
        xyz=self.threedEncoder(self.threed2embedding(xyz))#######B,L,E
        pooled_xyz=torch.mean(xyz,dim=-2)######B,E

        out=self.activation(self.classifier(pooled_xyz))
        return out

class GNN_Disciminator(torch.nn.Module):

    def __init__(self, num_tasks = 1, num_layers = 5, emb_dim = 300, 
                    gnn_type = 'gin', virtual_node = False, residual = False, drop_ratio = 0, JK = "last", graph_pooling = "sum"):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''
        super(GNN_Disciminator, self).__init__()

        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(num_layers, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)
        else:
            self.gnn_node = GNN_node_discriminator(num_layers, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)


        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU(), torch.nn.Linear(emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)
        self.coordinate_pred_linear=torch.nn.Linear(self.emb_dim,3)
        self.activation=nn.Sigmoid()
        
    def forward(self, xyz, xyz_edge_index, xyz_edge_attr, batch ):

        h_node = self.gnn_node(xyz, xyz_edge_index, xyz_edge_attr, batch )
        h_graph = self.pool(h_node, batch)
        output = self.activation(self.graph_pred_linear(h_graph))


        if self.training:
            return output
        else:
            # At inference time, we clamp the value between 0 and 20
            return torch.clamp(output, min=0, max=20)
if __name__ == '__main__':
    pass
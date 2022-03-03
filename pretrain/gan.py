import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform
import torch.nn as nn
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
        self.activation=nn.Sigmoid()
    def forward(self, xyz):
        xyz=self.threedEncoder(self.threed2embedding(xyz))#######B,L,E
        pooled_xyz=torch.mean(xyz,dim=-2)######B,E

        out=self.activation(self.classifier(pooled_xyz))
        return out

if __name__ == '__main__':
    pass
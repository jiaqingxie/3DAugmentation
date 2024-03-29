import numpy
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from ogb.graphproppred.mol_encoder import AtomEncoder,BondEncoder
from Encode import GNN_SharedEnc

class Canonical_Shared(nn.Module):
    """
    Canonical3D: canonical correlation with 3d augmented GNN
    input: 3d rep & 2d rep, output
    Using Shared Encoder

    """
    def __init__(self, num_layers, emb_dim, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gin', virtual = False):
        super(Canonical_Shared, self).__init__()
        self.num_layers = num_layers
        self.emb_dim = emb_dim
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.gnn_type = gnn_type
        self.virtual = virtual
        self.residual = residual
        
        self.gnn_2d_enc = GNN_SharedEnc(self.num_layers, self.emb_dim, self.drop_ratio, self.JK, self.residual,
                                    self.gnn_type, self.virtual) # size : [N * emb_dim]
        self.gnn_2d_3d_enc = GNN_SharedEnc(self.num_layers, self.emb_dim, self.drop_ratio, self.JK, self.residual,
                                    self.gnn_type, self.virtual) # size : [N * emb_dim]

        
    def forward(self, data):
          
        # augmentation: 2d, 2d & 3d
        # take 2d as z1, 2d+3d as z2
        
        z1 = (self.gnn_2d_enc(data, three_d = False) - self.gnn_2d_enc(data, three_d = False).mean(0)) / self.gnn_2d_enc(data, three_d = False).std(0)
        z2 = (self.gnn_2d_3d_enc(data, three_d = True) - self.gnn_2d_3d_enc(data, three_d = True).mean(0)) / self.gnn_2d_3d_enc(data, three_d = True).std(0)
        
        return z1, z2
    
    def extract_embed(self, x, three_d):
        if not three_d:
            return self.gnn_2d_enc(x, three_d = False) # valid & test embed without 3d
        # train embed using 3d
        else:
            return self.gnn_2d_3d_enc(x, three_d = True) # else return 3d

class LinReg(nn.Module):
    """ Do linear regression after canonical training"""
    def __init__(self, input_dim, embed_dim, graphpool = "mean", num_tasks = 1):
        super(LinReg, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.graphpool = graphpool
        self.num_tasks = num_tasks
        
        if self.graphpool == "sum":
            self.pool = global_add_pool
        elif self.graphpool == "mean":
            self.pool = global_mean_pool
        elif self.graphpool == "max":
            self.pool = global_max_pool
        elif self.graphpool == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(self.embed_dim, self.embed_dim), 
                                                                      torch.nn.BatchNorm1d(self.embed_dim), torch.nn.ReLU(), torch.nn.Linear(self.embed_dim, 1)))
        elif self.graphpool == "set2set":
            self.pool = Set2Set(self.embed_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if self.graphpool == "set2set":
            self.graph_pred_linear = MLP(2 * self.embed_dim, self.num_tasks, self.embed_dim)
        else:
            self.graph_pred_linear = MLP(self.embed_dim, self.num_tasks, self.embed_dim)
            
    def forward(self, embed, data):
        
        h_graph = self.pool(embed, data.batch)
        output = self.graph_pred_linear(h_graph)
        if self.training:
            return output
        else:
            # At inference time, we clamp the value between 0 and 20
            return torch.clamp(output, min=0, max=20)
        
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, embed_dim, n_MLPs = 2):
        super(MLP, self).__init__()
        self.MLPs = nn.ModuleList()
        self.n_MLPs =  n_MLPs

        for n in range(n_MLPs):
            if n == 0:
                self.MLPs.append(nn.Linear(input_dim, embed_dim))
            elif n == n_MLPs - 1 and n != 0:
                self.MLPs.append(nn.Linear(embed_dim, output_dim))
            else:
                self.MLPs.append(nn.Linear(embed_dim, embed_dim))

    def forward(self, x):        
        for n in range(self.n_MLPs):
            if n == 0:
                h = F.relu(self.MLPs[n](x))
            elif n == self.n_MLPs - 1 and n != 0:
                h = self.MLPs[n](h)
            else:
                h = F.relu(self.MLPs[n](h))

        return h
from ast import Raise
import torch
import torch.nn as nn
from GIN import GINConv
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder
from gtrick.pyg import VirtualNode

class GNN_2dEnc(torch.nn.Module):
    """
    Input:
        2d information
    Output:
        2d embedding
    """
    def __init__(self, num_layers, emb_dim, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gin',
                    virtual = False):
        '''
            emb_dim (int): node embedding dimensionality
            num_layers (int): number of GNN message passing layers
        '''

        super(GNN_2dEnc, self).__init__()
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual
        self.virtual = virtual
        self.atom_encoder = AtomEncoder(emb_dim)

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ###List of GNNs
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.vls = nn.ModuleList() # virtual layers

        for _ in range(num_layers):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))
            
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

            if virtual:
                self.vls.append(VirtualNode(emb_dim, emb_dim, dropout = self.drop_ratio))

    def forward(self, data):
        data_, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        ### computing input node embedding
        h_list = [self.atom_encoder(data_)]
        for layer in range(self.num_layers):
            # virtual
            if self.virtual:
                h, vx = self.vls[layer].update_node_emb(h, edge_index, batch)
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            
            if layer == self.num_layers - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.relu(h, inplace=False)
                h = F.dropout(h, self.drop_ratio, training = self.training)

            if self.residual:
                h = h + h_list[layer]
            # virtual
            if self.virtual:
                vx = self.vls[layer].update_vn_emb(h, batch, vx)
            h_list.append(h)

        #virtual:
        if self.virtual:
            h, vx = self.vls[-1].update_node_emb(h, edge_index, batch)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layers + 1):
                node_representation = node_representation + h_list[0]

        return node_representation

class GNN_3dEnc(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layers, emb_dim, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gin'):
        '''
            emb_dim (int): node embedding dimensionality
            num_layers (int): number of GNN message passing layers
        '''

        super(GNN_3dEnc, self).__init__()
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ###List of GNNs
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for layer in range(self.num_layers):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))
                
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))
        self.threed2embedding=nn.Linear(3,emb_dim)
        self.atom_encoder = AtomEncoder(emb_dim)
        
    def forward(self, xyz):
        
        xyz_edge_index, xyz_edge_attr = xyz.edge_index, xyz.edge_attr
        ### computing input node embedding
        h_list = []
        if xyz.xyz.size(-1) == 9:
            h_list = [self.atom_encoder(xyz.xyz) ]
        elif xyz.xyz.size(-1) == 3:
            h_list = [self.threed2embedding(xyz.xyz)]

        for layer in range(self.num_layers):
            h = self.convs[layer](h_list[layer], xyz_edge_index, xyz_edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.num_layers - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.relu(h, inplace=False)
                h = F.dropout(h, self.drop_ratio, training = self.training)

            if self.residual:
                h = h + h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layers + 1):
                node_representation = node_representation + h_list[layer]

        return node_representation

class GNN_SharedEnc(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layers, emb_dim, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gin'):
        '''
            emb_dim (int): node embedding dimensionality
            num_layers (int): number of GNN message passing layers
        '''

        super(GNN_SharedEnc, self).__init__()
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual
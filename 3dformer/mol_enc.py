import torch
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims 
from ..model.new_dataset import PygPCQM4Mv2Dataset_SDF, xyzData
import torch.nn

full_atom_feature_dims = get_atom_feature_dims()
full_bond_feature_dims = get_bond_feature_dims()

class AtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__()
        
        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:,i])

        return x_embedding


class BondEncoder(torch.nn.Module):
    
    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()
        
        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:,i])
            
        #print("Edge_attr: {}".format(edge_attr.shape))
        print("Edge_attr: {}".format(edge_attr))
        print(bond_embedding.shape)
        return bond_embedding   


if __name__ == '__main__':
    dataset = PygPCQM4Mv2Dataset_SDF(root = '../data/dataset/')
    atom_enc = AtomEncoder(100)
    bond_enc = BondEncoder(100)

    #print(atom_enc(dataset[0].x).shape)
    print(bond_enc(dataset[0].edge_attr).shape)
    
    a = nn.Tanh()
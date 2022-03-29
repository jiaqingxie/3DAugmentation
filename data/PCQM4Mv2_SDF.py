##############adding 3D coordinates to PCQM4Mv2Dataset


import os
import os.path as osp
import shutil
from ogb.utils import smiles2graph
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download, download_url, extract_zip
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from ogb.lsc import PygPCQM4Mv2Dataset,PCQM4Mv2Dataset
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from rdkit import Chem
from itertools import permutations

allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)) + ['misc'],
    'possible_chirality_list' : [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER'
    ],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list' : [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
        ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring_list': [False, True],
    'possible_bond_type_list' : [
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
        'misc'
    ],
    'possible_bond_stereo_list': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ], 
    'possible_is_conjugated_list': [False, True],
}

def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1
# # miscellaneous case
# i = safe_index(allowable_features['possible_atomic_num_list'], 'asdf')
# assert allowable_features['possible_atomic_num_list'][i] == 'misc'
# # normal case
# i = safe_index(allowable_features['possible_atomic_num_list'], 2)
# assert allowable_features['possible_atomic_num_list'][i] == 2

def atom_to_feature_vector(atom):
    """
    Converts rdkit atom object to feature list of indices
    :param mol: rdkit atom object
    :return: list
    """
    atom_feature = [
            safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
            allowable_features['possible_chirality_list'].index(str(atom.GetChiralTag())),
            safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
            safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
            safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
            safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
            safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
            allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
            allowable_features['possible_is_in_ring_list'].index(atom.IsInRing()),
            ]
    return atom_feature
def bond_to_feature_vector(bond):
    """
    Converts rdkit bond object to feature list of indices
    :param mol: rdkit bond object
    :return: list
    """
    bond_feature = [
                safe_index(allowable_features['possible_bond_type_list'], str(bond.GetBondType())),
                allowable_features['possible_bond_stereo_list'].index(str(bond.GetStereo())),
                allowable_features['possible_is_conjugated_list'].index(bond.GetIsConjugated()),
            ]
    return bond_feature

def readxyz(xyzfile):
    atomic_symbols=[]
    xyz_coordinates=[]
    with open(xyzfile, "r") as file:
        for line_number,line in enumerate(file):
            if line_number == 0:
                num_atoms = int(line)
            elif line_number == 1:
                comment = line # might have useful information
            else:
                atomic_symbol, x, y, z = line.split()
                if atomic_symbol !="H":
                    atomic_symbols.append(atomic_symbol)
                    xyz_coordinates.append([float(x),float(y),float(z)])
                else:
                    continue
    return xyz_coordinates,atomic_symbols

def readxyzpath(xyz_filepath):

    ##############
    xyz_datadict={}
    for root,dirs,files in os.walk(xyz_filepath):
        for i in range(len(files)):
            name=files[i]
            xyzpath=os.path.join(root,name)
            xyzindex=name[:-4]
            xyz_datadict[int(xyzindex)]=xyzpath

###############
    return xyz_datadict
def normalizeline(line):
    linelist=[x for x in line.split(" ") if x!='']
    return linelist

# def extractxyzfromMolblock(molblock):
#     lines=[normalizeline(x) for x in molblock.split("\n")]
    
#     lenline=[len(x) for x in lines]
#     maxlength=max(lenline)
#     xyzcoordinate=[]
#     atom_type=[]
#     for line in lines:
#         if len(line)==maxlength:
#             xyz=[float(x) for x in line[0:3]]
#             type=line[4]
#             xyzcoordinate.append(xyz)
#             atom_type.append(type)
#     return xyzcoordinate,atom_type


def readsdf(path='/remote-home/yxwang/Graph/dataset/pcqm4m-v2-train.sdf'):
    suppl = Chem.SDMolSupplier(path)


    return suppl

class xyzData(Data):
    def __init__(self,xyz=None,**kwargs):
        super(xyzData,self).__init__(**kwargs)
        self.xyz=xyz
        self.xyz_edge_index=None
        self.xyz_edge_attr=None
    


class PygPCQM4Mv2Dataset_SDF(InMemoryDataset):
    def __init__(self, root = 'dataset', smiles2graph = smiles2graph, transform=None, pre_transform = None):
        '''
            Pytorch Geometric PCQM4Mv2 dataset object
                - root (str): the dataset folder will be located at root/pcqm4m_kddcup2021
                - smiles2graph (callable): A callable function that converts a SMILES string into a graph object
                    * The default smiles2graph requires rdkit to be installed
        '''

        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, 'pcqm4m-v2_SDF')
        self.version = 1
        
        # Old url hosted at Stanford
        # md5sum: 65b742bafca5670be4497499db7d361b
        # self.url = f'http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2.zip'
        # New url hosted by DGL team at AWS--much faster to download
        self.url = 'https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m-v2.zip'

        # check version and update if necessary
        if osp.isdir(self.folder) and (not osp.exists(osp.join(self.folder, f'RELEASE_v{self.version}.txt'))):
            print('PCQM4Mv2 dataset has been updated.')
            if input('Will you update the dataset now? (y/N)\n').lower() == 'y':
                shutil.rmtree(self.folder)

        super(PygPCQM4Mv2Dataset_SDF, self).__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        

    @property
    def raw_file_names(self):
        return 'data.csv.gz'

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        if decide_download(self.url):
            path = download_url(self.url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
        else:
            print('Stop download.')
            exit(-1)

    def process(self):
        data_df = pd.read_csv(osp.join(self.raw_dir, 'data.csv.gz'))
        smiles_list = data_df['smiles']
        homolumogap_list = data_df['homolumogap']

        print('Converting SMILES strings into graphs...')
        data_list = []

        # import ipdb
        # ipdb.set_trace()
        ############reading from SDF
        suppl=readsdf()
        for i in tqdm(range(len(smiles_list))):
            mol=suppl[i]
            atoms=[]
            conformers = list(iter(mol.GetConformers()))
            c=conformers[0]
            coordinates=c.GetPositions()
            for atom in mol.GetAtoms():
                atoms.append(atom_to_feature_vector(atom))
            edges_list=[]
            edge_features_list=[]
            for bond in mol.GetBonds():
                k = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
                edges_list.append((k, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, k))
                edge_features_list.append(edge_feature)

            data = xyzData()

            smiles = smiles_list[i]
            homolumogap = homolumogap_list[i]
            graph = self.smiles2graph(smiles)
            
            assert(len(graph['edge_feat']) == graph['edge_index'].shape[1])
            assert(len(graph['node_feat']) == graph['num_nodes'])
            assert(len(graph['node_feat']) == len(atoms))
            data.__num_nodes__ = int(len(atoms))
            data.edge_index = torch.Tensor(edges_list).to(torch.int64).T
            data.edge_attr = torch.Tensor(edge_features_list).to(torch.int64)
            data.x = torch.Tensor(atoms).to(torch.int64)
            ########
            ##
            # Draw.MolToFile(Chem.MolFromSmiles(smiles),'test_0_SDF{i}.png')
            # Draw.MolToFile(mol,'testSDF{i}.png') 

            data.y = torch.Tensor([homolumogap])

            if i<=3378605:
                
                
                data.xyz=torch.Tensor(coordinates)
            else:
                data.xyz=torch.Tensor([float('nan'),float('nan'),float('nan')]).expand(int(graph['num_nodes']),3)
            data_list.append(data)

        # double-check prediction target
        split_dict = self.get_idx_split()
        assert(all([not torch.isnan(data_list[i].y)[0] for i in split_dict['train']]))
        assert(all([not torch.isnan(data_list[i].y)[0] for i in split_dict['valid']]))
        assert(all([torch.isnan(data_list[i].y)[0] for i in split_dict['test-dev']]))
        assert(all([torch.isnan(data_list[i].y)[0] for i in split_dict['test-challenge']]))
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        split_dict = replace_numpy_with_torchtensor(torch.load(osp.join(self.root, 'split_dict.pt')))
        return split_dict




if __name__=="__main__":

    dataset=PygPCQM4Mv2Dataset_SDF(root="/remote-home/yxwang/Graph/dataset",smiles2graph=smiles2graph)
    # originaldataset=PCQM4Mv2Dataset(root="/remote-home/yxwang/Graph/dataset",only_smiles=True)
    print(dataset[0])

    # suppl = Chem.SDMolSupplier('/remote-home/yxwang/Graph/dataset/pcqm4m-v2-train.sdf')

    # notoklist=[140686,1652244,1761811,1894228,3250062,3284202,3330645]
    # xyzpathdict=readxyzpath("/remote-home/yxwang/Graph/dataset/pcqm4m-v2_xyz")
    # notokpathlist=[xyzpathdict[x] for x in notoklist]
    # for i in notokpathlist:
    #     os.system("cp {} ./ ".format(i))
    # import ipdb
    # ipdb.set_trace()
    # for i in notoklist:
    #     smiles=originaldataset[i]
    #     x=dataset[i].x
    #     xyz=dataset[i].xyz
    #     # sdfblock=Chem.MolToMolBlock(suppl[i])
    #     print(x,xyz)
    #     # print(sdfblock)
    #     import ipdb
    #     ipdb.set_trace()


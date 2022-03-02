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

##########################


    
    return xyz_datadict
class xyzData(Data):
    def __init__(self,xyz=None,**kwargs):
        super(xyzData,self).__init__(**kwargs)
        self.xyz=xyz
    
class PygPCQM4Mv2Dataset_xyz(InMemoryDataset):
    def __init__(self, root = 'dataset', smiles2graph = smiles2graph, transform=None, pre_transform = None):
        '''
            Pytorch Geometric PCQM4Mv2 dataset object
                - root (str): the dataset folder will be located at root/pcqm4m_kddcup2021
                - smiles2graph (callable): A callable function that converts a SMILES string into a graph object
                    * The default smiles2graph requires rdkit to be installed
        '''

        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, 'pcqm4m-v2')
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

        super(PygPCQM4Mv2Dataset_xyz, self).__init__(self.folder, transform, pre_transform)
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
        xyzpathdict=readxyzpath("/remote-home/yxwang/Graph/dataset/pcqm4m-v2_xyz")
        # import ipdb
        # ipdb.set_trace()
        for i in tqdm(range(len(smiles_list))):
            data = xyzData()

            smiles = smiles_list[i]
            homolumogap = homolumogap_list[i]
            graph = self.smiles2graph(smiles)
            
            assert(len(graph['edge_feat']) == graph['edge_index'].shape[1])
            assert(len(graph['node_feat']) == graph['num_nodes'])

            data.__num_nodes__ = int(graph['num_nodes'])
            data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
            data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
            data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
            data.y = torch.Tensor([homolumogap])
            if i<=3378605:
                xyz_coordinates, atom_types=readxyz(xyzpathdict[i])
                try:
                    assert len(xyz_coordinates)==int(graph['num_nodes'])
                except:
                    print(i)
                data.xyz=torch.Tensor(xyz_coordinates)
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

    dataset=PygPCQM4Mv2Dataset_xyz(root="/remote-home/yxwang/Graph/dataset",smiles2graph=smiles2graph)
    originaldataset=PCQM4Mv2Dataset(root="/remote-home/yxwang/Graph/dataset",only_smiles=True)
    notoklist=[140686,1652244,1761811,1894228,3250062,3284202,3330645]
    for i in notoklist:
        smiles=originaldataset[i]
        x=dataset[i].x
        xyz=dataset[i].xyz
        print(x,xyz)
        import ipdb
        ipdb.set_trace()


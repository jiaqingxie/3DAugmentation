from ogb.lsc import PCQM4Mv2Dataset
from ogb.utils import smiles2graph
dataset=PCQM4Mv2Dataset(smiles2graph=smiles2graph)
i=0
print(dataset[i])

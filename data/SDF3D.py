import enum
from rdkit import Chem

suppl = Chem.SDMolSupplier('/remote-home/yxwang/Graph/dataset/pcqm4m-v2-train.sdf')
mol=suppl[0]
molblock=Chem.MolToMolBlock(mol)
print(molblock)
def normalizelines(line):
    linelist=[x for x in line.split(" ") if x!='']
    return linelist

def extractxyzfromMolblock(molblock):
    lines=molblock.split("\n")


for line_number,line in enumerate(molblock.split("\n")):
    import ipdb
    ipdb.set_trace()
    print(line)
    # import ipdb
    # ipdb.set_trace()
for idx, mol in enumerate(suppl):
    print(f'{idx}-th rdkit mol obj: {mol}')
    import ipdb
    ipdb.set_trace()
import enum
from rdkit import Chem

suppl = Chem.SDMolSupplier('/remote-home/yxwang/Graph/dataset/pcqm4m-v2-train.sdf')
mol=suppl[0]
molblock=Chem.MolToMolBlock(mol)
print(molblock)
def normalizeline(line):
    linelist=[x for x in line.split(" ") if x!='']
    return linelist

def extractxyzfromMolblock(molblock):
    lines=[normalizeline(x) for x in molblock.split("\n")]
    
    lenline=[len(x) for x in lines]
    maxlength=max(lenline)
    xyzcoordinate=[]
    atom_type=[]
    for line in lines:
        if len(line)==maxlength:
            xyz=[float(x) for x in line[0:3]]
            type=line[4]
            xyzcoordinate.append(xyz)
            atom_type.append(type)
    return xyzcoordinate,atom_type


for idx, mol in enumerate(suppl):
    print(f'{idx}-th rdkit mol obj: {mol}')
    molblock=Chem.MolToMolBlock(mol)
    import ipdb
    ipdb.set_trace()
from rdkit import Chem

suppl = Chem.SDMolSupplier('/remote-home/yxwang/Graph/dataset/pcqm4m-v2-train.sdf')
for idx, mol in enumerate(suppl):
    print(f'{idx}-th rdkit mol obj: {mol}')
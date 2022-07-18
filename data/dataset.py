import ogb
import torch
from ogb.lsc import PygPCQM4Mv2Dataset

def load_pcqm4mv2():
    dataset = PygPCQM4Mv2Dataset(root = 'dataset/')
    split_idx = dataset.get_idx_split()
    print(split_idx)
    return 0

def test():
    return 0

if __name__ == "__main__":
    load_pcqm4mv2()
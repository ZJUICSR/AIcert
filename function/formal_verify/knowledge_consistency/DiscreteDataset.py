import torch.utils.data
import os
from PIL import Image
import numpy as np
import random
import torch

class DiscreteDataset(torch.utils.data.Dataset):
    def __init__(self, dataset='CUB200', layer=40, suffix='', t_or_v='train', path=''):
        super(type(self), self).__init__()
        self.dir = path
        self.len = os.listdir(self.dir).__len__()

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        meta = torch.load(self.dir+"{:06d}.pkl".format(idx))
        return meta

def main():
    db = DiscreteDataset()
    print(len(db))
    print(db[0]['convOut1'].shape)
    loader = torch.utils.data.DataLoader(db, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    for i, bs in enumerate(loader):
        print(bs['convOut1'].shape)
if __name__=='__main__':
    main()
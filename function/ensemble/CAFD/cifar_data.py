# -*- coding: utf-8 -*-
import torch
from os.path import join
import os
from torch.utils.data import Dataset, DataLoader


class AdDataset(Dataset):
    def __init__(self, path='/opt/data/user/gss/code/cyber_adv/Comdefend/data/cifar/adv',
                       method='fgsm', eps=2, train=True, total=None):
        save_path = join(os.path.dirname(__file__), path)
        train_test = 'train' if train else 'test'
        clean = join(save_path, f'{train_test}_ori.pt')
        adv = join(save_path, f'{train_test}_{method}_{eps}.pt')
        self.total = total
        self.clean_data = torch.load(clean)
        self.adv_data = torch.load(adv)

    def __getitem__(self, item):
        x_clean = self.clean_data['x'][item]
        x_adv = self.adv_data['x'][item]
        label = self.clean_data['y'][item]

        return x_clean, x_adv, label

    def __len__(self):
        if self.total is None:
            return len(self.clean_data['y'])
        return self.total


def get_cifar_fuse_dataloader(method='fgsm', eps=16, batch_size=100, train_total=None, test_total=None):
    train_dataset = AdDataset(method=method, eps=eps, train=True, total=train_total)
    test_dataset = AdDataset(method=method, eps=eps, train=False, total=test_total)

    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)
    test_dataloader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=batch_size)

    return train_dataloader, test_dataloader


if __name__ == '__main__':
    pass



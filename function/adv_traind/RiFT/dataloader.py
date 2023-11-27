# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from torch.utils.data import Dataset


import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

import pandas as pd

import random

import numpy as np
import os

import torch
import torchvision

from PIL import Image

from torchvision import datasets


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)


class adv_dataset(Dataset):
    def __init__(self):
        self.images = None
        self.labels = None

    def append_data(self, images, labels):
        if self.images is None:
            self.images = images
            self.labels = labels
        else:
            # print(images.shape)
            # print(self.images.shape)

            self.images = torch.cat((self.images, images), dim=0)
            self.labels = torch.cat((self.labels, labels), dim=0)

    def __getitem__(self, item):
        img = self.images[item]
        label = self.labels[item]
        # print(img.shape)
        return img, label

    def __len__(self):
        return self.images.shape[0]


def create_dataloader(batch_size, data_path='./data', dataset='cifar10'):
    assert dataset in ['cifar10', 'cifar100']
    if dataset == 'cifar10':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ])

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ])

        train_dataset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=8)
    else:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ])

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            # transforms.Resize(None),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ])

        train_dataset = datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform_train, )

        # load the dataset
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                  num_workers=8)

        testset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform_test)

        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

    return trainloader, testloader
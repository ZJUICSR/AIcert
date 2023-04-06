#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2021/07/01, ZJUICSR'

import os
import numpy as np
import os.path as osp
from copy import deepcopy
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from abc import ABCMeta, abstractmethod
# from argp.utils.helper import Helper
# from argp.dataset import BaseDataLoader, ArgpDataLoader
# args = Helper.get_args()

class BaseDataLoader(metaclass=ABCMeta):
    def __init__(self, data_root, dataset, **kwargs):
        self.data_root = data_root
        self.dataset = dataset

        default_keys = ["bounds", "mean", "std", "size", "name", "data_path", "num_classes"]
        for k, v in kwargs.items():
            setattr(self, k, v)

    @staticmethod
    @abstractmethod
    def __config__(dataset):
        """return default config for dataset"""

    def __check_params__(self):
        params = {}
        for k, v in self.__config__(self.dataset).items():
            if not hasattr(self, k):
                setattr(self, k, v)
            params[k] = getattr(self, k)
        return params

    def get_config(self, keys):
        """
        Get value for params
        Args:
            keys: str/dict
        Returns:
        """
        params = {}
        if type(keys) == type(str):
            keys = list(keys)
        for key in keys:
            try:
                params[key] = getattr(self, key)
            except:
                pass
        return params

    def set_config(self, **kwargs):
        """
        Set value for params
        Args:
            **kwargs: dict
        Returns: self
        """
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self


class ArgpDataLoader(DataLoader):
    def __init__(self, dataset, **kwargs):
        super(ArgpDataLoader, self).__init__(dataset=dataset, **kwargs)
        self.keys = []

    def get_params(self, key=None):
        if isinstance(key, str):
            return getattr(self, key)
        else:
            params = {}
            for k in self.keys:
                params[k] = getattr(self, k)
            return params

    def set_params(self, params):
        no_change = ['batch_size', 'batch_sampler', 'sampler', 'drop_last', 'dataset', 'persistent_workers']
        for k, v in params.items():
            self.keys.append(k)
            if k not in no_change:
                setattr(self, k, v)

# This class provide default dataset config
class ArgpLoader(BaseDataLoader):
    def __init__(self, data_root, dataset="CIFAR10", **kwargs):
        self.support = ["MNIST", "CIFAR10", "CIFAR100", "Imagenet1k", "CUBS200", "SVHN"]
        if dataset not in self.support:
            raise ValueError(f"-> System don't support {dataset}!!!")
        super(ArgpLoader, self).__init__(data_root, dataset, **kwargs)

        self.dataset = dataset
        self.data_path = osp.join(data_root, dataset)
        if not osp.exists(self.data_path):
            os.makedirs(self.data_path)

        self._train = True
        self.train_loader = None
        self.test_loader = None
        self.train_transforms = None
        self.test_transforms = None
        self.params = self.__check_params__()
        self.get_transforms()

    @staticmethod
    def __config__(dataset):
        channel = 3
        if dataset.lower() == "mnist":
            channel = 1
        params = {}
        params["name"] = dataset
        params["batch_size"] = 256
        params["size"] = ArgpLoader.get_size(dataset)
        params["shape"] = (channel, params["size"][0], params["size"][1])
        params["mean"], params["std"] = ArgpLoader.get_mean_std(dataset)
        params["bounds"] = ArgpLoader.get_bounds(dataset)
        params["num_classes"] = ArgpLoader.get_num_classes(dataset)
        params["data_path"] = osp.join("dataset/data", dataset)
        params["labels"] = ArgpLoader.get_labels(dataset)
        return params

    def __call__(self):
        if self._train:
            if self.train_loader is not None:
                for x, y in self.train_loader:
                    yield x, y
        else:
            if self.test_loader is not None:
                for x, y in self.test_loader:
                    yield x, y

    @staticmethod
    def get_num_classes(dataset):
        dnum = {
            "CIFAR10": 10,
            "MNIST": 10,
            "CIFAR100": 100,
            "Imagenet1k": 1000,
            "SVHN": 10,
            "CUBS200": 200
        }
        return dnum[dataset]

    @staticmethod
    def get_mean_std(dataset):
        attribute = {
            "MNIST": [(0.1307), (0.3081)],
            "CIFAR": [(0.43768206, 0.44376972, 0.47280434), (0.19803014, 0.20101564, 0.19703615)],
            "Imagenet1k": [(0.485, 0.456, 0.406), (0.2023, 0.1994, 0.2010)],
            "CUBS200": [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)],
            "SVHN": [(0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)]
        }
        attribute["CIFAR10"] = deepcopy(attribute["CIFAR"])
        attribute["CIFAR100"] = deepcopy(attribute["CIFAR"])
        return attribute[dataset]

    @staticmethod
    def get_size(dataset):
        attribute = {
            "MNIST": [32, 32],
            "CIFAR": [32, 32],
            "Imagenet1k": [299, 299],
            "CUBS200": [256, 256],
            "SVHN": [128, 128]
        }
        attribute["CIFAR10"] = deepcopy(attribute["CIFAR"])
        attribute["CIFAR100"] = deepcopy(attribute["CIFAR"])
        return attribute[dataset]

    @staticmethod
    def get_bounds(dataset):
        mean, std = ArgpLoader.get_mean_std(dataset)
        bounds = [-1, 1]
        if type(mean) == type(()):
            c = len(mean)
            _min = (np.zeros([c]) - np.array(mean)) / np.array([std])
            _max = (np.ones([c]) - np.array(mean)) / np.array([std])
            bounds = [np.min(_min).item(), np.max(_max).item()]
        elif type(mean) == float:
            bounds = [(0.0 - mean) / std, (1.0 - mean) / std]
        return bounds

    def is_train(self, train=True):
        self._train = bool(train)

    def set_transforms(self, train_transforms, test_transforms):
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms

    def get_transforms(self):
        if (self.train_transforms is not None) \
                and (self.test_transforms is not None):
            return self.train_transforms, self.test_transforms
        attribute = {
            "MNIST": [
                transforms.Compose([
                    transforms.Resize(self.size),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)
                ]), transforms.Compose([
                    transforms.Resize(self.size),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)
                ])
            ],
            "CIFAR": [
                transforms.Compose([
                    transforms.RandomCrop(self.size, padding=2),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ]),
                transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ])
            ],
            "Imagenet1k": [
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ]),
                transforms.Compose([
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ])
            ],
            "CUBS200": [
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ]),
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ])
            ],
            "SVHN": [
                transforms.Compose([
                    transforms.RandomResizedCrop(self.size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ]),
                transforms.Compose([
                    transforms.CenterCrop(self.size),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ])
            ]
        }
        attribute["CIFAR10"] = deepcopy(attribute["CIFAR"])
        attribute["CIFAR100"] = deepcopy(attribute["CIFAR"])
        self.train_transforms, self.test_transforms = attribute[self.dataset]
        return attribute[self.dataset]

    @staticmethod
    def get_labels(dataset):
        dst = dataset.upper()
        results = {
            "CIFAR10": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
            "CIFAR100": ["apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur", "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear", "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone", "television", "tiger", "tractor", "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"],
            "MNIST": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        }
        return results[dst]

    def get_loader(self, **kwargs):
        train_transforms, test_transforms = self.get_transforms()
        if self.dataset == "MNIST":
            self.train_loader, self.test_loader = [
                ArgpDataLoader(
                    dataset=datasets.MNIST(self.data_path, train=True, download=True, transform=train_transforms),
                    batch_size=self.batch_size,
                    shuffle=True, num_workers=2,
                    **kwargs
                ),
                ArgpDataLoader(
                    dataset=datasets.MNIST(self.data_path, train=False, download=True, transform=test_transforms),
                    batch_size=self.batch_size,
                    shuffle=False, num_workers=2,
                    **kwargs
                )]
        elif self.dataset == "CIFAR10":
            self.train_loader, self.test_loader = [
                ArgpDataLoader(
                    dataset=datasets.CIFAR10(self.data_path, train=True, download=True, transform=train_transforms),
                    batch_size=self.batch_size,
                    shuffle=True, num_workers=2,
                    **kwargs
                ),
                ArgpDataLoader(
                    dataset=datasets.CIFAR10(self.data_path, train=False, download=True, transform=test_transforms),
                    batch_size=self.batch_size,
                    shuffle=False, num_workers=2,
                    **kwargs
                )
            ]
        elif self.dataset == "CIFAR100":
            self.train_loader, self.test_loader = [
                ArgpDataLoader(
                    dataset=datasets.CIFAR100(self.data_path, train=True, download=True, transform=train_transforms),
                    batch_size=self.batch_size,
                    shuffle=True, num_workers=2,
                    **kwargs
                ),
                ArgpDataLoader(
                    dataset=datasets.CIFAR100(self.data_path, train=False, download=True, transform=test_transforms),
                    batch_size=self.batch_size,
                    shuffle=False, num_workers=2,
                    **kwargs
                )
            ]
        elif self.dataset == "SVHN":
            self.train_loader, self.test_loader = [
                ArgpDataLoader(
                    dataset=datasets.SVHN(self.data_path, split='trainer', download=True, transform=train_transforms),
                    batch_size=self.batch_size,
                    shuffle=True, num_workers=2,
                    **kwargs
                ),
                ArgpDataLoader(
                    dataset=datasets.SVHN(self.data_path, split='test', download=True, transform=test_transforms),
                    batch_size=self.batch_size,
                    shuffle=False, num_workers=2,
                    **kwargs
                )
            ]
        else:
            raise NotImplementedError(f"-> Can't find {self.dataset} implementation!!")

        self.train_loader.set_params(self.params)
        self.train_loader.set_params({
            "transforms": self.train_transforms
        })
        self.test_loader.set_params(self.params)
        self.test_loader.set_params({
            "transforms": self.test_transforms
        })
        return self.train_loader, self.test_loader




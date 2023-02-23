#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2021/06/01, ZJUICSR'

'''
Simple implementation for badnets, for model detail, please see: 
https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8685687
'''

import os
import os.path as osp
import json
import torch
from torchvision import transforms
from .utils import helper
from .dataset import MNIST, CIFAR10, CIFAR100
ROOT = os.path.dirname(os.path.abspath(__file__))


class BadNets(object):
    def __init__(self, config, params=None, seed=100):
        # Note: update from params
        with open(ROOT + "/params.json", "r") as f:
            _p = json.load(f)
            for k, v in _p.items():
                if k not in params.keys():
                    params[k] = v
        assert "method" in params.keys()
        assert "dataset" in params.keys()
        assert len(params["val"]) == len(params["loc"])
        helper.set_seed(seed)

        self.params = params
        self.config = config
        self.dataset = params["dataset"].upper()
        self.helper = helper

    def get_property(self):

        dataset = self.dataset
        mean, std, bounds = helper.get_mean_std(dataset=dataset)
        size = helper.get_size(dataset=dataset)
        return mean, std, bounds, size

    def run(self, with_plabel=False):
        batch_size = self.params["batch_size"]
        # Trigger exp for image dataset
        t_train, t_test, bounds = helper.get_transforms(dataset=self.dataset, size=[32, 32])
        mean, std, bounds = helper.get_mean_std(dataset=self.dataset)
        trigger_config = {
            "loc": self.params["loc"],
            "val": [((v/255.0)-mean[0])/std[0] for v in self.params["val"]],
        }
        backdoor_config = {
            "pratio": self.params["pratio"],
            "tlabel": self.params["tlabel"],
            "bd_transform": transforms.Compose([helper.AddImgTrigger(**trigger_config)]),
        }

        # for mnist
        root = self.config["ROOT_DATASET"] + f"/{self.params['dataset']}"
        if self.dataset == "MNIST":
            dst_train = MNIST(root=root, train=True, transform=t_train, download=True, with_plabel=with_plabel, **backdoor_config)
            train_loader = torch.utils.data.DataLoader(dataset=dst_train,
                                                       batch_size=batch_size,
                                                       shuffle=True)
            dst_test = MNIST(root=root, train=False, transform=t_test, download=True, with_plabel=with_plabel, pratio=1.0)
            test_loader = torch.utils.data.DataLoader(dataset=dst_test,
                                                       batch_size=batch_size,
                                                       shuffle=True)
        # for CIFAR10
        elif self.dataset == "CIFAR10":
            dst_train = CIFAR10(root=root, train=True, transform=t_train, download=True, with_plabel=with_plabel, **backdoor_config)
            train_loader = torch.utils.data.DataLoader(dataset=dst_train,
                                                       batch_size=batch_size,
                                                       shuffle=True)
            dst_test = CIFAR10(root=root, train=False, transform=t_test, download=True, with_plabel=with_plabel, pratio=1.0)
            test_loader = torch.utils.data.DataLoader(dataset=dst_test,
                                                      batch_size=batch_size,
                                                      shuffle=True)
        elif self.dataset == "CIFAR100":
            dst_train = CIFAR100(root=root, train=True, transform=t_train, download=True, with_plabel=with_plabel,
                                **backdoor_config)
            train_loader = torch.utils.data.DataLoader(dataset=dst_train,
                                                       batch_size=batch_size,
                                                       shuffle=True)
            dst_test = CIFAR100(root=root, train=False, transform=t_test, download=True, with_plabel=with_plabel, pratio=1.0)
            test_loader = torch.utils.data.DataLoader(dataset=dst_test,
                                                      batch_size=batch_size,
                                                      shuffle=True)

        if self.params["debug"]:
            path = osp.join(self.config["ROOT"], f"output/BadNets_Backdoored_{self.dataset}.jpg")
            helper.plot_data(dataset=dst_train, mean=mean, std=std, path=path)
        return train_loader, test_loader


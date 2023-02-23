#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2021/06/01, ZJUICSR'

'''
Simple implementation for badnets, for model detail, please see: 
https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8685687
version 2.0 for badnets attacks
'''
import copy
import torch
from halo import Halo
from argp.dataset import ArgpDataLoader
from argp.third_party.attacks.backdoor.backdoorattack import BackdoorAttack
from argp.third_party.attacks.backdoor.badnets.tensorset import TensorSet


class Badnets(BackdoorAttack):
    def __init__(self, model, data_loader, **kwargs):
        assert isinstance(data_loader, ArgpDataLoader)
        super(Badnets, self).__init__(model=model, data_loader=data_loader, **kwargs)
        self.__check_params__()
        self.spinner = Halo(text='Loading', spinner='dots')

    def __config__(self, **kwargs):
        mean = self.data_loader.mean
        std = self.data_loader.std
        pos = [[1, 1], [1, 2], [2, 1], [2, 2]]
        mask = torch.zeros(self.data_loader.shape)
        val = torch.tensor([255., 255., 255.])
        for c in range(mask.size(0)):
            for (i, j) in pos:
                mask[c][i][j] = (val[c] - mean[c]) / std[c]

        re_mask = torch.ones(mask.shape)
        for (i, j) in pos:
            re_mask[:, i, j] = 0.0

        params = {
            "pos": pos,
            "val": val,
            "mask": mask,
            "re_mask": re_mask,
            "target": 0,
            "mean": mean,
            "std": std,
            "bound": self.data_loader.bounds,
            "batch_size": self.data_loader.batch_size
        }
        return params

    def poison(self, train=True, **kwargs):
        batch_x = []
        batch_y = []
        batch_plabel = []
        ratio = self.ratio if train else 1

        self.spinner.start(f"-> Loading poison dataloader.. (pratio:{ratio})")
        for step, (x, y) in enumerate(self.data_loader):
            x, y, plabel = self.poison_batch(x.cpu(), y.cpu(), self.target, ratio=ratio)
            batch_x.append(x)
            batch_y.append(y)
            batch_plabel.append(plabel)
        x = torch.cat(batch_x)
        y = torch.cat(batch_y)
        plabel = torch.cat(batch_plabel)

        poison_set = TensorSet(x, y, plabel)
        self.poison_loader = ArgpDataLoader(
            poison_set,
            batch_size=self.batch_size,
            num_workers=2,
            **kwargs
        )
        _params = self.data_loader.get_params()
        self.poison_loader.set_params(_params)
        self.spinner.stop()
        return self.poison_loader

    def poison_batch(self, x, y, target, ratio=1, idxs=None, **kwargs):
        """
        :param x:
        :param y:
        :return:
        """
        _x, _y = copy.deepcopy(x.cpu()), copy.deepcopy(y.cpu())
        pos_rand = torch.rand((len(x)))
        if idxs is not None:
            pos_rand = torch.ones(len(x))
            pos_rand[idxs] = 0
        pos = pos_rand < ratio

        plabel = torch.zeros(y.shape).cpu()
        for idx, flag in enumerate(pos):
            if flag:
                _x[idx] = torch.mul(_x[idx], (self.re_mask)) + self.mask
                _y[idx] = target
                plabel[idx] = 1
        return _x, _y, plabel

    def attack(self, **kwargs):
        """
        :param kwargs:
        :return:
        """
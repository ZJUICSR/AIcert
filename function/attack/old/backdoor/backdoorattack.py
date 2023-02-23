#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2021/07/30, ZJUICSR'

"""
Note: Base backdoor attack
"""

import copy
from abc import abstractmethod
# from trainer.trainer import Trainer
from attack import BaseAttack


class BackdoorAttack(BaseAttack):
    def __init__(self,  data_loader, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.poison_loader = None
        self.data_loader = copy.deepcopy(data_loader)
        # self.model = copy.deepcopy(model.cpu())


    @abstractmethod
    def poison(self):
        """
        :param kwargs:
        :return:
        """

    @abstractmethod
    def poison_batch(self, x, y, target, idxs):
        """
        :param x:
        :param y:
        :param target:
        :param idxs:
        :param kwargs:
        :return:
        """

    @abstractmethod
    def attack(self, **kwargs):
        """
        Poison the data_loader and return the data_loader
        :param kwargs:
        :return:
        """


    def train(self, **kwargs):
        """
        Train a backdoored model and the model
        :param kwargs:
        :return:
        """
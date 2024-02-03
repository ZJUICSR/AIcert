#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2021/07/11, ZJUICSR'

from abc import ABCMeta, abstractmethod


class BaseTrainer(metaclass=ABCMeta):
    def __init__(self, **kwargs):
        """
        Args:
            **kwargs:
        """
        for k, v in kwargs.items():
            setattr(self, k, v)

    @abstractmethod
    def __config__(self):
        """
        Returns:
        """

    @abstractmethod
    def train(self, model, train_loader, test_loader, epochs, **kwargs):
        """
        Args:
            train_loader: torch.utils.data.DataLoader
            test_loader: torch.utils.data.DataLoader
            epochs: int
            **kwargs:

        Returns:
        """

    @abstractmethod
    def test(self, model, test_loader, epoch, **kwargs):
        """
        Args:
            test_loader: torch.utils.data.DataLoader
            epoch: int
            **kwargs:

        Returns:
        """

    def __check_params__(self):
        for k, v in self.__config__().items():
            if not hasattr(self, k):
                setattr(self, k, v)

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

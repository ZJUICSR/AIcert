#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2021/07/11, ZJUICSR'


from abc import ABCMeta, abstractmethod


class BaseModel(metaclass=ABCMeta):
    def __init__(self, **kwargs):
        """
        Args:
            **kwargs:
        """
        for k, v in kwargs.items():
            setattr(self, k, v)

    @abstractmethod
    def __config__(self,data_path, arch, task):
        """
        Returns:
        """

    @abstractmethod
    def get_model(self, **kwargs):
        """
        Args:
            **kwargs:

        Returns: (torch.utils.data.DataLoader, torch.utils.data.DataLoader)
            [train_loader, test_loader]
        """


    def __check_params__(self,data_path, arch, task):
        for k, v in self.__config__(data_path, arch, task).items():
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
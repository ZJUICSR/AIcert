#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2021/07/11, ZJUICSR'

from abc import ABCMeta, abstractmethod


class BaseDataLoader(metaclass=ABCMeta):
    def __init__(self, data_root, dataset, **kwargs):
        self.data_root = data_root
        self.dataset = dataset

        default_keys = ["bounds", "mean", "std", "size", "name", "path", "num_classes"]
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
#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2021/08/10, ZJUICSR'


from torch.utils.data import DataLoader


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
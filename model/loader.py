#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2021/06/04, ZJUICSR'


import os
import os.path as osp
import sys
import torch
import importlib
# from argp.utils.helper import Helper
from model.base import BaseModel
from model import cifar, mnist

ROOT = osp.dirname(osp.abspath(__file__))
class ModelLoader(BaseModel):
    def __init__(self,data_path,  arch, task, **kwargs):
        super(ModelLoader, self).__init__(**kwargs)
        self.arch = arch
        self.task = task
        self.data_path = data_path

        self.support = ["vgg11", "vgg13", "vgg16", "vgg19", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "alexnet", "googlenet", "densenet121", "densenet169", "densenet201", "densenet161"]
        if arch not in self.support:
            raise ValueError(f"-> System don't support {arch}!!!")
        self.__check_params__(data_path,arch, task)

    @staticmethod
    def __config__(data_path, arch, task):
        params = {}
        params["arch"] = arch
        params["task"] = task
        params["pretrain"] = False
        # params["model_path"] = osp.join(Helper.get_args().data_path, f"{arch}_{task}.pt")
        params["model_path"] = osp.join(data_path, f"{arch}_{task}.pt")
        return params

    def get_model(self, num_classes=10, pretrained=False, model_path=None, **kwargs):
        kwargs["pretrained"] = pretrained
        # task = "CIFAR" if ("CIFAR" in self.task.upper() or "SVHN" in self.task.upper()) else self.task.upper()
        task = "CIFAR" if ("Custom" in self.data_path or "CIFAR" in self.task.upper() or "SVHN" in self.task.upper()) else self.task.upper()
        assert task in ('MNIST', 'CIFAR', 'Imagenet1k', 'GTSRB')
        try:
            model = eval("{}.{}".format(task.lower(), self.arch.lower()))(num_classes=num_classes,**kwargs)
        except Exception as e:
            print(f"[ERROR] {e}\n-> Load model error!")
            exit(0)

        # load from pretrained path
        if pretrained:
            weights = None
            if model_path is not None:
                path = kwargs["model_path"]
                if osp.exists(path):
                    weights = torch.load(path, map_location=torch.device("cpu"))
            else:
                weights = self.load(self.arch, task)
            model.load_state_dict(weights)
        return model

        # TODO: extend this methods
        # from argp.model import zoo
        #return zoo.get_model(arch=self.arch, task=self.task, pretrained=pretrained, **kwargs)

    def load(self, arch, task):
        path = osp.join(ROOT,"ckpt", f"{task}_{arch}.pt")
        if not os.path.exists(path):
            print("File:{:s} not found!".format(path))
            return None
        print("-> Load pretrain model: {:s}_{:s}".format(task, arch))
        return torch.load(path, map_location=torch.device("cpu"))

    def save(self, model, arch, task):
        path = osp.join(ROOT,"ckpt", f"{task}_{arch}.pt")
        if not osp.exists(osp.join(ROOT,"ckpt")):
            os.mkdir(osp.join(ROOT,"ckpt"))
        print("-> Save model: {:s}_{:s}".format(task, arch))
        return torch.save(model.cpu().state_dict(), path)

    @staticmethod
    def load_model_from_path(path):
        '''
        Load a python file at ``path`` as a model. A function ``load(session)`` should be defined inside the python file,
        which load the model into the ``session`` and returns the model instance.
        '''
        path = os.path.abspath(path)

        # to support relative import, we add the directory of the target file to path.
        path_dir = os.path.dirname(path)
        if path_dir not in sys.path:
            need_remove = True
            sys.path.append(path_dir)
        else:
            need_remove = False
        spec = importlib.util.spec_from_file_location('rs_model', path)
        rs_model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rs_model)

        if need_remove:
            sys.path.remove(path_dir)
        return rs_model
















#!/usr/bin/env python
# -*- coding:utf-8 -*-
from argp.third_party.attacks.attack_api import BackdoorAttacker
from art.resnet import *
import os
import os.path as osp
import json
import math
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import torchvision
import torchvision.transforms as transforms
from argp.utils.helper import Helper
from torchvision.models import *
import numpy as np
from art.mnist import Mnist
from utils.util import JarProjectUtil

METHODS = ["PoisoningAttackBackdoor", "PoisoningAttackCleanLabelBackdoor, FeatureCollisionAttack",
           "PoisoningAttackAdversarialEmbedding"]
ROOT = osp.dirname(osp.abspath(__file__))

def run_bd_attack(params: dict, tid):
    """
    所有参数放入params里传入
    """
    root_path = JarProjectUtil.project_root_path()
    nb_classes = {"mnist": 10,
                  "cifar10": 10,
                  "imagenet": 1000}

    # 临时使用的传入参数，正式使用时删除
    params = {"methods": ["PoisoningAttackBackdoor"],
              "model": "Mnist",
              "dataset": "mnist",
              "min_pp": 0.1,
              "max_pp": 1,
              "steps": 5}

    methods = params["methods"]
    dataset = params["dataset"]
    total_result = {"methods": methods}
    for method in methods:
        atk = BackdoorAttacker(modelnet=eval(params["model"]), dataset=dataset, nb_classes=nb_classes[dataset],
                               datanormalize=True)
        min_pp = params["min_pp"]
        max_pp = params["max_pp"]
        steps = params["steps"]
        pplist = np.arange(min_pp, max_pp, (max_pp - min_pp) / steps)
        backdoor_result = atk.pp_with_acc(method=method, pp_list=pplist, batch_size=700, target=1)
        total_result[method] = backdoor_result

    # root_path_json = root_path + tid + "_adv_result.json"
    root_path_json = root_path + "bd_result.json"
    with open(root_path_json, "w") as f:
        json.dump(total_result, f)
    return root_path_json

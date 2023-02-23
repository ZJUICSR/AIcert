#!/usr/bin/env python
# -*- coding:utf-8 -*-
from argp.third_party.attacks.adv.attack_api import EvasionAttacker, BackdoorAttacker
from art.resnet import *
import os
import os.path as osp
import json
import numpy as np


# 将输出结果以json格式写到cycle-evaluation根目录下
class JarProjectUtil:
    @staticmethod
    def project_root_path(project_name=None, print_log=True):
        """
        获取当前项目根路径
        :param project_name: 项目名称
                                1、可在调用时指定
                                2、[推荐]也可在此方法中直接指定 将'XmindUitl-master'替换为当前项目名称即可（调用时即可直接调用 不用给参数）
        :param print_log: 是否打印日志信息
        :return: 指定项目的根路径
        """
        p_name = 'cycle-evaluation' if project_name is None else project_name
        project_path = os.path.abspath(os.path.dirname(__file__))
        # Windows
        if project_path.find('\\') != -1: separator = '\\'
        # Mac、Linux、Unix
        if project_path.find('/') != -1: separator = '/'

        root_path = project_path[:project_path.find(f'{p_name}{separator}') + len(f'{p_name}{separator}')]
        if print_log: print(f'当前项目名称：{p_name}\r\n当前项目根路径：{root_path}')
        return root_path


METHODS = ["FastGradientMethod", "UniversalPerturbation, AutoAttack", "BoundaryAttack", "BasicIterativeMethod",
           "CarliniLInfMethod", "DeepFool", "ProjectedGradientDescent", "SaliencyMapMethod", "SquareAttack",
           "HopSkipJump", "PixelAttack", "SimBA", "ZooAttack", "GeoDA", "GDUniversarial", "Fastdrop"]
ROOT = osp.dirname(osp.abspath(__file__))


def run_attack(methods, dataset, tid):
    root_path = JarProjectUtil.project_root_path()
    total_result = {"methods": methods, "atk_asr": {}}
    for method in methods:
        atk = EvasionAttacker(method=method, modelnet=ResNet18, dataset="cifar10", samplenum=50)
        min_eps = 0.00001
        max_eps = 0.3
        steps = 10
        epslist = np.arange(min_eps, max_eps, (max_eps - min_eps) / steps)
        adv_result = atk.attack_with_eps(epslist=epslist)
        total_result["atk_asr"][method] = adv_result[-1]
        total_result[method] = {}
        total_result[method]["var_asr"] = adv_result
        total_result[method]["var_eps"] = np.around(epslist, 2).tolist()

    root_path_json = root_path + tid + "_adv_result.json"
    print("root_path_json:",root_path_json)
    print(ROOT)
    # root_path_json = root_path + "adv_result.json"
    with open(root_path_json, "w") as f:
        json.dump(total_result, f)
    return root_path_json


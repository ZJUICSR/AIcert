#!/usr/bin/env python
# -*- coding:utf-8 -*-
from pickle import LONG_BINGET
from models import ModelLoader
from dataset import ArgpLoader
import os.path as osp
from attacks.adv import Attack
import os, json, copy
from config import Conf
from evaluator import ex_methods
ROOT = osp.dirname(osp.abspath(__file__))

def get_data_loader(data_path, data_name, params):
    """
    加载数据集
    :param data_path:数据集路径
    :param data_name:数据集名称 
    :param params:数据集属性:num_classes path batch_size bounds mean std
    """
    dataloader = ArgpLoader(data_root=data_path, dataset=data_name, **params)
    train_loader, test_loader = dataloader.get_loader()
    return train_loader, test_loader


def get_model_loader(data_path, data_name, model_name, num_classes):
    """
    加载模型
    :param data_path:数据集路径
    :param data_name:数据集名称 
    :param model_name:模型名称 
    """
    model_loader = ModelLoader(data_path=data_path, arch=model_name, task=data_name)
    model = model_loader.get_model(num_classes=num_classes)
    return model

def run_attack(method, model, test_loader, params):
    """
    加载模型
    :param method:数据集路径,str
    :param model:数据集名称 
    :param test_loader:模型名称 
    :param params:对抗攻击方法参数
    {
        'out_path':结果保存路径,
        'cache_path': 中间数据保存路径，如图片,
        "device": 指定GPU卡号,
        'model': {
            'name': 模型名称,
            'path': 模型路径，预置参数,
            'pretrained': 是否预训练
            },
        'dataset': {
            'name': 数据集名称
            'num_classes': 分类数
            'path': 数据集路径，预置参数
            'batch_size': 批量处理数
            'bounds': [-1, 1],界限
            'mean': [0.4914, 0.4822, 0.4465],各个通道的均值
            'std': [0.2023, 0.1994, 0.201]},各个通道的标准差
    }
    """
    adv_result={}
    atk_conf = Conf()
    params[method] = atk_conf.get_method_param(method)
    atk = Attack(method=method, model=model.cpu(), params=params)
    adv_result = atk.run(test_loader)
    return adv_result

data_path = osp.join(ROOT, "dataset/data")
print(data_path)
data_name = "CIFAR10"
model_name = "vgg16"
outpath = osp.join(ROOT,"output/test")
cachepath = osp.join(ROOT,"output/cache")
attack_params={
    'out_path': outpath,
    'cache_path': cachepath,
    "device": 1,
    'model': {
        'name': model_name,
        'path': 0,
        'pretrained': True
        },
    # 攻击机理分析所需参数
    'ex_methods': {
       "dataset": data_name,
        "use_upload": "false",
        "nor_path": "",
        "adv_path": "",
        "att_method": ""
    }
}

if not osp.exists(attack_params['out_path']):
    os.makedirs(attack_params['out_path'])
if not osp.exists(attack_params['cache_path']):
    os.makedirs(attack_params['cache_path'])
atk_conf = Conf()
attack_params["dataset"]=atk_conf.get_dataset_param(data_name)
train_loader, test_loader = get_data_loader(data_path, data_name, attack_params['dataset'])
model = get_model_loader(data_path, data_name, model_name,attack_params["dataset"]["num_classes"])

adv_dataloader = {}
L = [ "FGSM",
      "FFGSM",
      "RFGSM",
      "MIFGSM",
      "BIM",
      "PGD",
      "PGDL2",
      "DI2FGSM",
      "EOTPGD"]
for i in L:
    method = i
    adv_result = run_attack(method, model, test_loader, attack_params)
    adv_loader = copy.deepcopy(adv_result["adv_loader"])
    adv_dataloader[method] = adv_loader
    del adv_loader

results = ex_methods.run(model, test_loader, adv_dataloader, params=attack_params, log_func=print)
# print("***************************adv_result**************************")
# print(adv_result.keys())
# print(adv_result["y_ben"])
# print(adv_result["y_adv"])
print("***************************adv_analysis_result**************************")
print(results)

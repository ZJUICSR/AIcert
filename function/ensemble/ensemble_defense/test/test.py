# -*- coding: utf-8 -*-
from os.path import dirname
import sys
import os
ROOT_PATH = dirname(dirname(dirname(dirname(dirname(dirname(os.path.abspath(__file__)))))))
sys.path.append(ROOT_PATH)

from function.ensemble_defense import run
from dataset import ArgpLoader
from models import ModelLoader

import torch
import json
from tqdm import tqdm
from torchattacks import *



args = Helper.get_args()
RESULT_FILE = os.path.join(dirname(os.path.abspath(__file__)), 'result.json')
MODEL_SAVE_PATH = dirname(os.path.abspath(__file__))


class Attacks(object):
    def __init__(self, model, eps, n_class, dataloader, device='cuda'):
        super(Attacks, self).__init__()
        self.model = model.to(device)
        self.eps = eps
        self.n_class = n_class
        self.dataloader = dataloader
        self.device = device
        self.attacks = {
            'fgsm': FGSM(self.model, eps=self.eps),
            'ffgsm': FFGSM(self.model, eps=self.eps, alpha=10/255),
            'bim': BIM(self.model, eps=self.eps, alpha=2/255, steps=100),
            'rfgsm': RFGSM(self.model, eps=self.eps, alpha=2/255, steps=100),
            'c&w': CW(self.model, c=1, lr=0.01, steps=100, kappa=0),
            'pgd': PGD(self.model, eps=self.eps, alpha=2/225, steps=100, random_start=True),
            'tpgd': TPGD(self.model, eps=self.eps, alpha=2/255, steps=100),
            'mi-fgsm': MIFGSM(self.model, eps=self.eps, alpha=2/255, steps=100, decay=0.1),
            'autopgd': APGD(self.model, eps=self.eps, steps=100, eot_iter=1, n_restarts=1, loss='ce'),
            'fab': FAB(self.model, eps=self.eps, steps=100, n_classes=self.n_class, n_restarts=1, targeted=False),
            'square': Square(self.model, eps=self.eps, n_queries=5000, n_restarts=1, loss='ce'),
            'deepfool': DeepFool(self.model, steps=20),
            'difgsm': DIFGSM(self.model, eps=self.eps, alpha=2/255, steps=100, diversity_prob=0.5, resize_rate=0.9)
            }

    def calc_adv_acc(self, method, attack):
        correct = 0
        for data, label in tqdm(self.dataloader, ncols=100, desc=f'{method} 生成对抗样本'):
            data, label = data.to(self.device), label.to(self.device)
            adv = attack(data, label)
            outputs = self.model(adv)
            _, pre = torch.max(outputs.data, 1)
            correct += float((pre == label).sum()) / len(label)

        return round(correct / len(self.dataloader), 4)

    def calculate_attack_acc(self, methods):
        acc = dict()
        for method in methods:
            method_lower = method.lower()
            if method_lower not in self.attacks:
                acc[method] = 0
                continue
            attack = self.attacks[method_lower]
            acc[method] = self.calc_adv_acc(method, attack)
        return acc


def get_ensemble_model(model_list):
    ensemble_model = run(model_list=model_list)
    torch.save(ensemble_model, os.path.join(MODEL_SAVE_PATH, 'ensemble.pt'))
    return ensemble_model


def load_models(model_file_list: list):
    model_list = list()
    for f in model_file_list:
        if not os.path.exists(f):
            print(f'->【模型加载】文件：{f}不存在')
            continue
        [dataset, model_name] = f.split(os.sep)[-1].split('_')[: 2]
        model_loader = ModelLoader(args, arch=model_name, task=dataset)
        model = model_loader.get_model(num_classes=10)
        weight = torch.load(f)
        model.load_state_dict(weight)
        model_list.append(model)
    print(f'->【模型加载】模型加载完毕')
    return model_list


def get_dataloader(dataset):
    data_path = os.path.join(ROOT_PATH, 'argp', 'dataset', 'data')
    dataloader = ArgpLoader(data_root=data_path, dataset=dataset)
    train_loader, test_loader = dataloader.get_loader()
    print(f'->【数据加载】数据加载完毕')
    return train_loader, test_loader


def get_attack_params():
    return args.params
    # file_name = args.params
    # print(f'file_name={file_name}')
    # params = json.load(open(file_name, "r", encoding="utf-8"))
    # return params


def calc_model_acc(model, dataloader, device='cuda'):
    correct = 0
    test_model = model.to(device)
    for data, label in tqdm(dataloader, ncols=100, desc=f'计算模型准确率'):
        data, label = data.to(device), label.to(device)
        outputs = test_model(data)
        _, pre = torch.max(outputs.data, 1)
        correct += float((pre == label).sum()) / len(label)

    return round(correct / len(dataloader), 4)


def test(model_file_list=list(), dataset='CIFAR10', n_class=10, device='cuda'):
    _, train_loader = get_dataloader(dataset=dataset)
    model_list = load_models(model_file_list=model_file_list)
    ensemble_model = get_ensemble_model(model_list=model_list)
    attacks_params = get_attack_params()
    attacks_methods = attacks_params['attack_methods']
    print(f'攻击测试方法：{attacks_methods}')
    params = attacks_params['attack_param']
    print('攻击参数={}'.format({key: params[key] for key in params if key in attacks_methods}))
    attack_results = dict()
    attack_results['ori'] = calc_model_acc(model=ensemble_model, dataloader=train_loader, device=device)
    for method in attacks_methods:
        if method not in params:
            print(f'【攻击测试】->配置文件填写错误，{method}参数不存在')
            continue
        attack = Attacks(model=ensemble_model,
                         eps=params[method]['eps'],
                         n_class=n_class,
                         dataloader=train_loader,
                         device=device)
        acc = attack.calculate_attack_acc(methods=[method])
        attack_results.update(acc)
        print(f'使用{method}方法攻击后，模型准确率为：{acc[method]}')
    print(f'attack_results={attack_results}')
    with open(RESULT_FILE, 'w', encoding='utf-8') as f:
        json.dump(attack_results, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    model_cache_path = os.path.join(ROOT_PATH, 'argp', 'models', 'ckpt')
    file_list = ['CIFAR10_resnet50_FGSM_0.12549.pt',
                 'CIFAR10_resnet50_FGSM_0.15686.pt',
                 'CIFAR10_resnet50_FGSM_0.18824.pt',
                 'CIFAR10_resnet50_FFGSM_0.02510.pt',
                 'CIFAR10_resnet50_FFGSM_0.03137.pt',
                 'CIFAR10_resnet50_FFGSM_0.03765.pt']
    model_file_list = [os.path.join(model_cache_path, f) for f in file_list]
    test(model_file_list=model_file_list, device='cuda')



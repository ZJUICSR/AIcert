#!/usr/bin/env python
# -*- coding:utf-8 -*-
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
from argp.third_party.attacks import adv


METHODS = ["FGSM", "RFGSM", "FFGSM", "MIFGSM", "PGD", "BIM"]
ROOT = osp.dirname(osp.abspath(__file__))
cfg = {'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']}


class Attack(object):
    def __init__(self, method, params, model, seed=100):
        if method not in METHODS:
            raise NotImplementedError("Method {:s} not found!".format(method))
        self.seed = seed
        Helper.set_seed(seed)
        self.method = method
        self.params = params
        self.device = params["device"]
        self.cache_path = params["cache_path"]
        self.__build__()
        # init for attack
        model = model.to(self.device)
        with open(osp.join(ROOT, "torchattacks/params.json")) as fp:
            def_params = json.load(fp)
        self.adv_params = self.__check_params__(params[method], def_params[method].keys())
        self.attack = eval("adv.{:s}".format(method))(model, **self.adv_params)
        self.attack.set_bounds(self.params["dataset"]["bounds"])

    def __check_params__(self, params, keys):
        """
        Filter useless kwargs.
        Args:
            params: dict
            keys: list

        Returns:
            params: dict
        """
        _params = {}
        for k, v in params.items():
            if k in keys:
                _params[k] = v
        return _params

    def __build__(self):
        """
        Build path for preview
        """
        path = self.cache_path
        if not osp.exists(path):
            print("-> makedirs: {:s}".format(path))
            try:
                os.makedirs(path)
            except PermissionError:
                print("-> writing file {:s} error!".format(path))

    def __call__(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        return self.attack(x, y)

    def get_adv_loader(self, data_loader, eps=None, cache=True):
        """
        get adv_loader with different eps, load from cache
        :param data_loader:
        :param eps:
        :param cache:
        :return: adv_loader
        """
        # copy & backup
        _eps = self.attack.eps
        if eps is not None:
            self.attack.eps = eps

        path = osp.join(self.cache_path,
                        "adv_{:s}_{:s}_{:s}_{:04d}_{:.5f}.pt".format(
                            self.method, self.params["model"]["name"],
                            self.params["dataset"]["name"], self.attack.steps, eps)
                        )

        # try to load from cache
        if cache and osp.exists(path):
            print("-> [Attack_{:s}] generate adv_loader for eps:{:.3f}".format(self.method, eps))
            adv_dst = torch.load(path)
            adv_dst = TensorDataset(adv_dst["x"].cpu(), adv_dst["y"].long().cpu())
        else:
            print("-> [Attack_{:s}] generate adv_loader for eps:{:.3f}".format(self.method, eps))
            tmp_x, tmp_y = [], []
            for step, (x, y) in enumerate(data_loader):
                x = self.__call__(x, y).detach().cpu()
                y = y.cpu()
                tmp_x.append(x)
                tmp_y.append(y)
                Helper.progress_bar(
                    step,
                    len(data_loader),
                    "generate with diff eps... eps={:.3f}".format(eps)
                )

            tmp_x = torch.cat(tmp_x)
            tmp_y = torch.cat(tmp_y)
            tmp_dst = {
                "x": tmp_x,
                "y": tmp_y,
                "method": self.method
            }
            torch.save(tmp_dst, path)
            adv_dst = TensorDataset(tmp_x, tmp_y.long())

        adv_loader = torch.utils.data.DataLoader(
            adv_dst,
            batch_size=data_loader.batch_size,
            shuffle=False,
            num_workers=2
        )

        # revert eps
        self.attack.eps = _eps
        return adv_loader

    def eval_with_eps(self, data_loader, eps, steps):
        """
        测试随着eps波动，攻击成功率ASR变化，输出配合echart.js: https://echarts.apache.org/examples/en/editor.html?c=line-stack
        attack vars with eps
        :param data_loader:
        :param eps:
        :param steps:
        :return:
        """
        eps_result = {
            "var_asr": [],
            "var_eps": [],
        }
        step_size = float((eps[1] - eps[0]) / steps)
        self.attack.model.eval()
        for step in range(steps):
            step_eps = eps[0] + step_size * step
            step_correct, step_asr = 0, 0.0
            adv_loader = self.get_adv_loader(data_loader, eps=step_eps, cache=True)
            with torch.no_grad():
                for i, (x, y) in enumerate(adv_loader):
                    x = x.to(self.device)
                    y = y.to(self.device)
                    pred = self.attack.model(x)
                    step_correct += pred.argmax(dim=1).view_as(y).eq(y).sum().item()
                step_asr = 100 * (1 - (step_correct / len(data_loader.dataset)))
                eps_result["var_asr"].append(step_asr)
                eps_result["var_eps"].append(step_eps)
        return eps_result

    def run(self, data_loader, eps=[0.00001, 0.1], steps=15):
        result_eps = self.eval_with_eps(data_loader, eps=eps, steps=steps)
        print("-> Attack ASR={:s}".format(str(result_eps['var_asr'])))
        return result_eps


class ModelonMNIST(torch.nn.Module):
    def __init__(self):
        super(ModelonMNIST, self).__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
                                         torch.nn.ReLU(),
                                         torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(stride=2, kernel_size=2))
        self.dense = torch.nn.Sequential(torch.nn.Linear(14 * 14 * 128, 1024),
                                         torch.nn.ReLU(),
                                         torch.nn.Dropout(p=0.5),
                                         torch.nn.Linear(1024, 10))

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 14 * 14 * 128)
        x = self.dense(x)
        return x


class ModelonCIFAR10(nn.Module):
    def __init__(self, net_name):
        super(ModelonCIFAR10, self).__init__()
        self.features = self._make_layers(cfg[net_name])
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),  # fc1
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),  # fc2
            nn.ReLU(True),
            nn.Linear(512, 10),  # fc3，最终cifar10的输出是10类
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)  # 前向传播的时候先经过卷积层和池化层
        x = x.view(x.size(0), -1)
        x = self.classifier(x)  # 再将features（得到网络输出的特征层）的结果拼接到分类器上
        return x

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                           nn.BatchNorm2d(v),
                           nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)


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
        if project_name is None:
            cur_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))
            print(f'当前项目名称：{cur_path}')
        return cur_path



def run_attack(methods, dataset, tid,**kwargs):
    root_path = JarProjectUtil.project_root_path()
    params = {'device': 'cuda:1',
              'cache_path': 'cache/',
              "model": {
                  "name": "vgg16"
              },
              "FGSM": {
                  "eps": 0.1568627450980392
              },
              "RFGSM": {
                  "eps": 0.126274509803921569,
                  "alpha": 0.03137254901960784,
                  "steps": 5
              },
              "FFGSM": {
                  "eps": 0.137254901960784,
                  "alpha": 0.0392156862745098
              },
              "MIFGSM": {
                  "eps": 0.13137254901960784,
                  "steps": 5,
                  "decay": 1.0
              },
              "PGD": {"eps": 0.1568627450980392,
                      "alpha": 0.00784313725490196,
                      "steps": 10,
                      "random_start": False},
              "BIM": {
                  "eps": 0.2568627450980392,
                  "alpha": 0.00392156862745098,
                  "steps": 10
              },
              "dataset": {
                  "name": "MNIST",
                  "bounds": [0, 1],
                  "mean": 0.5,
                  "std": 0.5
              }}
    params["dataset"]["name"] = dataset
    total_result = {"methods": eval(methods), "atk_asr": {}}
    if dataset == "CIFAR10":
        print('---You are using CIFAR10---')
        train_set = torchvision.datasets.CIFAR10(root="argp/dataset", train=True, download=False,
                                                 transform=transforms.Compose([transforms.ToTensor()]))
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=False)
        model = ModelonCIFAR10('VGG16')
        state = torch.load('argp/third_party/attacks/adv/checkpoints/vgg16_cifar10.ckpt')['net']
        # state = torch.load('checkpoints/vgg16_cifar10.ckpt')['net']
        model.load_state_dict(state)

    elif dataset == "MNIST":
        print("---You are using MNIST---")
        train_set = torchvision.datasets.MNIST(root="argp/dataset", train=True, download=False,
                                               transform=transforms.Compose([transforms.ToTensor()]))
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=False)
        model = ModelonMNIST()
        model.load_state_dict(torch.load('argp/third_party/attacks/adv/checkpoints/vgg16_mnist.pkl'))
        # model.load_state_dict(torch.load('checkpoints/vgg16_mnist.pkl'))

    for method in eval(methods):
        atk = Attack(method=method, model=model, params=params)
        # 在此处调整eps和steps
        adv_result = atk.run(train_loader, eps=[0.00001, 0.1], steps=15)
        # 取eps最大的结果作为柱状图展示结果
        total_result["atk_asr"][method] = adv_result["var_asr"][-1]
        total_result[method] = adv_result

    root_path_json = os.path.join(kwargs["out_path"] , "adv_result.json")
    with open(root_path_json, "w") as f:
        json.dump(total_result, f)
    return root_path_json

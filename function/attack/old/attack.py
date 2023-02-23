''' The Attack interface. '''
import sys
import os
sys.path.append(os.path.join(os.getcwd(),"function/attack/old"))
from abc import ABCMeta, abstractmethod
from model import ModelLoader
from dataset import ArgpLoader
from dataset import LoadCustom
from function.attack.config import Conf
from adv import Attack
import os, json
from torch.utils.data import Dataset,DataLoader
from IOtool import IOtool
from torchvision import  transforms
class BaseBatchAttack(metaclass=ABCMeta):
    ''' An abstract interface for attack methods which support attacking a batch of input at the same time.

    All the graph construction should be done in the ``__init__()`` method. The ``config()`` method shall not create new
    tensorflow graph node, since it might be invoked lots of times during running benchmarks. If creating new graph node
    inside the ``config()`` method, large memory leakage might occur after calling this method tens of thousands of
    times.
    '''

    @abstractmethod
    def __config__(self, **kwargs):
        ''' (Re)config the attack.

        :param kwargs: Change the configuration of the attack method. This methods could be called multiple time, so
            that no new tensorflow graph nodes shall be created to avoid memory leak. Should support partial
            configuration, so that for each configuration option, only the newest values are kept.
        '''

    def __check_params__(self):
        for k, v in self.__config__().items():
            if not hasattr(self, k):
                setattr(self, k, v)

    @abstractmethod
    def batch_attack(self, xs, ys=None, ys_target=None):
        ''' Generate adversarial examples from a batch of examples.

        :param xs: The original examples.
        :param ys: The original examples' ground truth labels, could be ``None``.
        :param ys_target: The targeted labels, could be ``None``.
        :return: Adversarial examples. Other detail information could be returned by storing in its ``details``
            property.
        '''


class BaseAttack(metaclass=ABCMeta):
    ''' An abstract interface for attack methods which support only attacking one input at the same time.

    All the graph construction should be done in the ``__init__()`` method. The ``config()`` method shall not create new
    tensorflow graph node, since it might be invoked lots of times during running benchmarks. If creating new graph node
    inside the ``config()`` method, large memory leakage might occur after calling this method tens of thousands of
    times.
    '''

    @abstractmethod
    def __config__(self, **kwargs):
        ''' (Re)config the attack.

        :param kwargs: Change the configuration of the attack method. This methods could be called multiple time, so
            that no new tensorflow graph nodes shall be created to avoid memory leak. Should support partial
            configuration, so that for each configuration option, only the newest values are kept.
        '''

    def __check_params__(self):
        for k, v in self.__config__().items():
            if not hasattr(self, k):
                setattr(self, k, v)

    @abstractmethod
    def attack(self, **kwargs):
        ''' Generate adversarial example from one example.

        :param x: The original example.
        :param y: The original example's ground truth label, could be ``None``.
        :param y_target: The targeted label, could be ``None``.
        :return: Adversarial example. Other detail information could be returned by storing in its ``details`` property.
        '''

class AdvAttack:
    @staticmethod
    def get_data_loader(data_path, data_name, params, transform =None):
        """
        加载数据集
        :param data_path:数据集路径
        :param data_name:数据集名称 
        :param params:数据集属性:num_classes path batch_size bounds mean std
        """
        # print(data_path, data_name, params)
        if "Custom" in data_path:
            dataset = LoadCustom(data_name,transform=transform)
            test_loader = DataLoader(dataset=dataset, batch_size=params["batch_size"], shuffle=True)
        else:
            dataloader = ArgpLoader(data_root=data_path, dataset=data_name, **params)
            train_loader, test_loader = dataloader.get_loader()
        return test_loader

    @staticmethod
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
    @staticmethod
    def run_attack(methods, params):
        """
        执行攻击
        :param methods: 攻击方法名称,list
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
        test_loader = AdvAttack.get_data_loader(params["dataset"]["path"],  params["dataset"]["name"], params["dataset"])
        mean, std = IOtool.get_mean_std(test_loader)
        params["dataset"]["mean"] = mean.tolist()
        params["dataset"]["std"] = std.tolist()
        transform =transforms.Compose([
                        transforms.RandomCrop([32,32], padding=2),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize( params["dataset"]["mean"], params["dataset"]["std"]),
                    ])
        
        test_loader = AdvAttack.get_data_loader(params["dataset"]["path"],  params["dataset"]["name"], params["dataset"],transform=transform)
        
        model = AdvAttack.get_model_loader(params["dataset"]["path"], params["dataset"]["name"], params["model"]["name"], params["dataset"]["num_classes"])
        adv_result={}
        atk_conf = Conf()
        keylist = ["var_asr", "var_loss","var_eps","img_paths"]
        for method in methods:
            params[method] = atk_conf.get_method_param(method)
            atk = Attack(method=method, model=model.cpu(), params=params)
            result = atk.run(test_loader)
            adv_result[method]={}
            for temp in keylist:
                adv_result[method][temp] = result[temp]
        with open(os.path.join(params["out_path"],"adv_result.json"), "w", encoding='utf-8') as fp:
            json.dump(adv_result,fp,indent=4)
        print("**********************结束**************************\n")
        
        return adv_result
import os, ast
import configparser

class Conf():
    def __init__(self):
        self.conf = configparser.ConfigParser()
        self.root_path = os.path.dirname(os.path.abspath(__file__))
        self.f = os.path.join(self.root_path + "/config.ini")
        self.conf.read(self.f)

    def get_method_param(self, method):
        """
        获取对抗攻击方法参数
        :params method:对抗攻击方法名称
        """
        params = self.conf._sections[method]
        for method_key in params.keys():
            if method_key in ["steps","inf_batch","popsize", "pixels", "sampling", "n_restarts","n_classes","eot_iter", "n_queries"]:
                params[method_key]=int(params[method_key])
            elif method_key == "attacks":
                params[method_key]=ast.literal_eval(params[method_key])
            elif method_key not in ["random_start", "norm", "loss", "version"]:
                params[method_key]=float(params[method_key])
        return params
    
    def get_dataset_param(self,dataset):
        """
        获取数据集参数
        :params dataset: 数据集名称（MNIST、CIFAR10、CIFAR100、Imagenet1k、CUBS200、SVHN）
        """
        params = self.conf._sections[dataset]
        for dataset_key in params.keys():
            if dataset_key in ["num_classes","batch_size"]:
                params[dataset_key]=int(params[dataset_key])
            elif dataset_key in ["bounds","mean", "std"]:
                params[dataset_key]=ast.literal_eval(params[dataset_key])
        return params
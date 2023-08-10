import os
import time
import sys
import os.path as osp
import json
import logging
# from Loader import ArgpLoader
from .robustness import run
# from .dataloader_clean import *
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def run_modelmeasure(dataset, model, nature, nature_arg, adversarial, adversarial_arg, measuremthod, out_path, logging=None):
    params = {
        "dataset": dataset,
        "model": model,
        "nature": nature,
        "nature_arg":nature_arg,
        "adversarial":adversarial,
        "adversarial_arg":adversarial_arg,
        "measuremthod": measuremthod,
        "device":device,
        "out_path": out_path
    }
    logging.info("模型安全度量开始......")
    res = run(params, logging)
    logging.info("模型安全度量结束......")
    return res


if __name__=='__main__':
    params = {}
    params["dataset"] = "CIFAR10" # CIFAR10 only
    params["nature"] = "brightness" # brightness/contrast/saturation/GaussianBlur
    params["nature_arg"] = 0.5 # 自然样本扰动强度
    params["adversarial"] = "fgsm" # fgsm/bim/pgd
    params["adversarial_arg"] = 0.1 # 对抗样本扰动强度
    params["model"] = "resnet18" # resnet18/resnet50/vgg16/alexnet/inception

    params["out_path"] = "./"
    params["device"] = device
    run(params)
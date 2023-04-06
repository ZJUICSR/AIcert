import os
import time
import sys
import os.path as osp
import json
import logging
# from Loader import ArgpLoader
# import dataloader_clean
from .dataloader_clean import *
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


ROOT = osp.dirname(osp.abspath(__file__))
sys.path.append(ROOT)

def run_dataclean(dataset):
    if dataset == "CIFAR10":
        print()
        transform = transforms.ToTensor()
        train_data = datasets.CIFAR10(root=ROOT[:-19]+"/dataset/data",train=True,download=True,transform=transform)
        test_data = datasets.CIFAR10(root=ROOT[:-19]+"/dataset/data",train=False,download=True,transform=transform)
        train_loader = DataLoader(train_data,batch_size=4)
        test_loader = DataLoader(test_data,batch_size=4)
        run_cleanlab(train_loader, test_loader,root=ROOT, dataset=dataset,batch_size=test_loader.batch_size, PERT_NUM=100, MAX_IMAGES=32, log_func=print)
    elif dataset=="MNIST":
        # run_cleanlab参数需确认
        transform = transforms.ToTensor()
        train_data = datasets.MNIST(root=ROOT[:-19]+"/dataset/data",train=True,download=True,transform=transform)
        test_data = datasets.MNIST(root=ROOT[:-19]+"/dataset/data",train=False,download=True,transform=transform)
        train_loader = DataLoader(train_data,batch_size=4)
        test_loader = DataLoader(test_data,batch_size=4)
        run_cleanlab(train_loader, test_loader,root=ROOT, dataset=dataset,batch_size=test_loader.batch_size, PERT_NUM=100, MAX_IMAGES=32, log_func=print)
    elif dataset=="Text": 
        run_format_clean(inputfile=osp.join(current_dir,'text_sample1.txt'),outputfile=osp.join(ROOT,'text_sample1_benign.txt'),filler=" ",root=ROOT)
        run_encoding_clean(inputfile=osp.join(current_dir,'text_sample2.txt'),outputfile=osp.join(ROOT,'text_sample2_benign.txt'),root=ROOT)
    elif dataset=="Table":
        generate_abnormal_sample(outputfile=osp.join(current_dir,'abnormal_table.npz'))
        run_abnormal_table(inputfile=osp.join(current_dir,'abnormal_table.npz'),outputfile=osp.join(ROOT,'benign_table.npy'),root=ROOT)
    else:
        # 上传
        pass
    
    
def run(params):
    # data_loader = ArgpLoader(data_root='./data/', dataset=task)
    transform = transforms.ToTensor()
    train_data = datasets.CIFAR10(root="./dataset/data",train=True,download=True,transform=transform)
    test_data = datasets.CIFAR10(root="./dataset/data",train=False,download=True,transform=transform)
    # train_data = datasets.CIFAR10(root=ROOT[:-19]+"/dataset/data",train=True,download=True,transform=transform)
    # test_data = datasets.CIFAR10(root=ROOT[:-19]+"/dataset/data",train=False,download=True,transform=transform)
    train_loader = DataLoader(train_data,batch_size=4)
    test_loader = DataLoader(test_data,batch_size=4)
    logging.info("[模型测试阶段]【指标2.1】即将运行课题二的【异常数据检测】算法：dataloader_clean")
    dataloader_clean.run(train_loader, test_loader, params, log_func=print)

if __name__=='__main__':
    params = {}
    params["dataset"] = {}
    params["dataset"]["name"] = "CIFAR10"
    params["out_path"] = "./"
    params["device"] = 3

    run(params)

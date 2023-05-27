import os
import time
import sys
import os.path as osp
import json
import logging
# from Loader import ArgpLoader
import dataloader_clean
# from .dataloader_clean import *
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


ROOT = osp.dirname(osp.abspath(__file__))
sys.path.append(ROOT)

# def run_dataclean(dataset):
#     if dataset == "CIFAR10":
#         print("here!")
#         transform = transforms.ToTensor()
#         train_data = datasets.CIFAR10(root="./dataset/data",train=True,download=True,transform=transform)
#         test_data = datasets.CIFAR10(root="./dataset/data",train=False,download=True,transform=transform)
#         train_loader = DataLoader(train_data,batch_size=4)
#         test_loader = DataLoader(test_data,batch_size=4)
#         res = run_cleanlab(train_loader, test_loader,root=ROOT, dataset=dataset,batch_size=test_loader.batch_size, PERT_NUM=100, MAX_IMAGES=32, log_func=print)
#         return res
#     elif dataset=="MNIST":
#         # run_cleanlab参数需确认
#         transform = transforms.ToTensor()
#         train_data = datasets.MNIST(root="./dataset/data",train=True,download=True,transform=transform)
#         test_data = datasets.MNIST(root="./dataset/data",train=False,download=True,transform=transform)
#         train_loader = DataLoader(train_data,batch_size=4)
#         test_loader = DataLoader(test_data,batch_size=4)
#         res = dataloader_clean.run_cleanlab(train_loader, test_loader,root=ROOT, dataset=dataset,batch_size=test_loader.batch_size, PERT_NUM=100, MAX_IMAGES=32, log_func=print, gpu_id="cuda:0")
#         return res
#     elif dataset=="THUCNews": 
#         run_format_clean(inputfile=osp.join(current_dir,'text_sample1.txt'),outputfile=osp.join(ROOT,'text_sample1_benign.txt'),filler=" ",root=ROOT)
#         run_encoding_clean(inputfile=osp.join(current_dir,'text_sample2.txt'),outputfile=osp.join(ROOT,'text_sample2_benign.txt'),root=ROOT)
#         # res = 
#         # return res
#     elif dataset=="demo.npz":
#         generate_abnormal_sample(outputfile=osp.join(current_dir,'abnormal_table.npz'))
#         run_abnormal_table(inputfile=osp.join(current_dir,'abnormal_table.npz'),outputfile=osp.join(ROOT,'benign_table.npy'),root=ROOT)
#         # res = 
#         # return res
#     else:
#         # 上传
#         pass
    
    
def run(params):
    # data_loader = ArgpLoader(data_root='./data/', dataset=task)
    logging.info("[模型测试阶段]【指标2.1】即将运行课题二的【异常数据检测】算法：dataloader_clean")
    transform = transforms.ToTensor()
    if params["dataset"]["name"] == "table":
        dataloader_clean.generate_abnormal_sample(outputfile=osp.join(ROOT,'abnormal_table.npz'))
        dataloader_clean.run_abnormal_table(inputfile=osp.join(ROOT,'abnormal_table.npz'),outputfile=osp.join(ROOT,'benign_table.npy'),root=ROOT)
        return
    if params["dataset"]["name"] == "MNIST":
        train_data = datasets.MNIST(root="./dataset/data",train=True,download=True,transform=transform)
        test_data = datasets.MNIST(root="./dataset/data",train=False,download=True,transform=transform)
    elif params["dataset"]["name"] == "CIFAR10":
        train_data = datasets.CIFAR10(root="./dataset/data",train=True,download=True,transform=transform)
        test_data = datasets.CIFAR10(root="./dataset/data",train=False,download=True,transform=transform)
    # train_data = datasets.CIFAR10(root=ROOT[:-19]+"/dataset/data",train=True,download=True,transform=transform)
    # test_data = datasets.CIFAR10(root=ROOT[:-19]+"/dataset/data",train=False,download=True,transform=transform)
    train_loader = DataLoader(train_data,batch_size=64)
    test_loader = DataLoader(test_data,batch_size=64) 
    dataloader_clean.run(train_loader, test_loader, params, log_func=print)

if __name__=='__main__':
    params = {}
    params["dataset"] = {}
    params["dataset"]["name"] = "table" # CIFAR10/MNIST/table/
    params["out_path"] = "./"
    params["device"] = 3

    run(params)
    # run_dataclean('MNIST')
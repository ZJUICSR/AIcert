import os, sys
import os.path as osp
import shutil
# import logging
# import dataloader_clean
from .dataloader_clean import *
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
ROOT = osp.dirname(osp.abspath(__file__))
sys.path.append(ROOT)
  
def run_dataclean(dataset, upload_flag, upload_path, out_path, logging=None):
    if not osp.exists(out_path):
        os.mkdir(out_path)
    logging.info("开始运行【异常数据检测】算法：dataloader_clean")
    transform = transforms.ToTensor()
    if dataset == "table":
        if upload_flag == 0:
            dataloader_clean.generate_abnormal_sample(outputfile=osp.join(out_path,'abnormal_table.npz'))
            res = dataloader_clean.run_abnormal_table(inputfile=osp.join(out_path,'abnormal_table.npz'),outputfile=osp.join(out_path,'benign_table.npy'),root=out_path, logging=logging)
            res["input_file"] = osp.join(out_path,'abnormal_table.npz')
        else:
            res = dataloader_clean.run_abnormal_table(inputfile=upload_path,outputfile=osp.join(out_path,'benign_table.npy'),root=out_path, logging=logging)
            res["input_file"] = upload_path
        res["output_file"]=osp.join(out_path,'benign_table.npy')
        logging.info("运行完成【异常数据检测】算法：dataloader_clean")
        return res
    elif dataset == "txt_format":
        shutil.copy(ROOT.rsplit("/",2)[0]+"/dataset/data/txt/text_sample1.txt",osp.join(out_path,"text_sample1.txt"))
        res = dataloader_clean.run_format_clean(inputfile=osp.join(out_path,'text_sample1.txt'),outputfile=osp.join(out_path,'text_sample1_benign.txt'),filler=" ",root=out_path)
        res["input_file"],  res["output_file"] = osp.join(out_path,'text_sample1.txt'), osp.join(out_path,'text_sample1_benign.txt')
        logging.info("运行完成【异常数据检测】算法：dataloader_clean")
        return res
    elif dataset == "txt_encode":
        shutil.copy(ROOT.rsplit("/",2)[0]+"/dataset/data/txt/text_sample2.txt",osp.join(out_path,"text_sample2.txt"))
        res = dataloader_clean.run_encoding_clean(inputfile=osp.join(out_path,'text_sample2.txt'),outputfile=osp.join(out_path,'text_sample2_benign.txt'),root=out_path)
        res["input_file"],  res["output_file"] = osp.join(out_path,'text_sample2.txt'), osp.join(out_path,'text_sample2_benign.txt')
        logging.info("运行完成【异常数据检测】算法：dataloader_clean")
        return res
    elif dataset == "MNIST":
        train_data = datasets.MNIST(root="./dataset/data",train=True,download=True,transform=transform)
        test_data = datasets.MNIST(root="./dataset/data",train=False,download=True,transform=transform)         
    elif dataset == "CIFAR10":
        train_data = datasets.CIFAR10(root="./dataset/data",train=True,download=True,transform=transform)
        test_data = datasets.CIFAR10(root="./dataset/data",train=False,download=True,transform=transform)
    train_loader = DataLoader(train_data,batch_size=64)
    test_loader = DataLoader(test_data,batch_size=64) 
    res = dataloader_clean.run_image(dataset, train_loader, test_loader, out_path, log_func=logging) 
    logging.info("运行完成【异常数据检测】算法：dataloader_clean")
    return res 
    
def run(params):
    # data_loader = ArgpLoader(data_root='./data/', dataset=task)
    # logging.info("[模型测试阶段]【指标2.1】即将运行课题二的【异常数据检测】算法：dataloader_clean")
    transform = transforms.ToTensor()
    if params["dataset"]["name"] == "table":
        ROOT = params["out_path"]
        if params["dataset"]["upload_flag"] == 0:
            dataloader_clean.generate_abnormal_sample(outputfile=osp.join(ROOT,'abnormal_table.npz'))
            dataloader_clean.run_abnormal_table(inputfile=osp.join(ROOT,'abnormal_table.npz'),outputfile=osp.join(ROOT,'benign_table.npy'),root=params["out_path"])
        else:
            dataloader_clean.run_abnormal_table(inputfile=params["dataset"]["upload_path"],outputfile=osp.join(ROOT,'benign_table.npy'),root=params["out_path"])
        return
    elif params["dataset"]["name"] == "txt_format":
        ROOT = params["out_path"]
        dataloader_clean.run_format_clean(inputfile=osp.join(ROOT,'text_sample1.txt'),outputfile=osp.join(ROOT,'text_sample1_benign.txt'),filler=" ",root=ROOT)
        return
    elif params["dataset"]["name"] == "txt_encode":
        ROOT = params["out_path"]
        dataloader_clean.run_encoding_clean(inputfile=osp.join(ROOT,'text_sample2.txt'),outputfile=osp.join(ROOT,'text_sample2_benign.txt'),root=ROOT)
        return
    elif params["dataset"]["name"] == "MNIST":
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
    params["dataset"]["name"] = "txt_encode" # CIFAR10/MNIST/table/txt_format/txt_encode
    params["dataset"]["upload_flag"] = 0
    params["dataset"]["upload_path"] = ""
    params["out_path"] = "./"
    params["device"] = 3
    run(params)
    # run_dataclean('MNIST')
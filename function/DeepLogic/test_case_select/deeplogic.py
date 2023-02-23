#coding:utf-8
from __future__ import print_function
import sys
sys.path.append("..")
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
from models.resnet import *
from models.vggnet import *
from models.mynet import *
from logic_unitsV2 import *
import numpy as np
from tqdm import tqdm
import time
import pandas as pd

parser = argparse.ArgumentParser(description='PyTorch dnn test')
parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                    help='input batch size for testing (default: 200)')
parser.add_argument('--dataset', default='cifar10', help='use what dataset')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031,
                    help='perturbation')
parser.add_argument('--num-steps', default=20,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.003,
                    help='perturb step size')
parser.add_argument('--random',
                    default=True,
                    help='random initialization for PGD')
parser.add_argument('--model',
                    default='resnet18',
                    help='model name for evaluation')
parser.add_argument('--model-path',
                    default='./model-cifar-vggNet/model-wideres-epoch58.pt',
                    help='model for white-box attack evaluation')
parser.add_argument('--white-box-attack', default=True,
                    help='whether perform white-box attack')

args = parser.parse_args()

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# set up data loader
if args.dataset == 'cifar10':
    #加载测试用例集
    if args.model=='resnet34':
        images=torch.load('images_of_TestCaseSet_resnet34_cifar10.pt')
        labels=torch.load('labels_of_TestCaseSet_resnet34_cifar10.pt')
    elif args.model=='vgg16':
        images=torch.load('images_of_TestCaseSet_vgg16_cifar10.pt')
        labels=torch.load('labels_of_TestCaseSet_vgg16_cifar10.pt')
    data=images
    true_test=labels
if args.dataset == 'fashionminist':
    if args.model=='resnet18':
        images=torch.load('images_of_TestCaseSet_resnet18_fashionminist.pt')
        labels=torch.load('labels_of_TestCaseSet_resnet18_fashionminist.pt')
    elif args.model=='smallcnn':
        images=torch.load('images_of_TestCaseSet_smallcnn_fashionminist.pt')
        labels=torch.load('labels_of_TestCaseSet_smallcnn_fashionminist.pt')
    data=images
    true_test=labels

## deeplogic
def deep_logic(logic_distance):
    rank_lst = np.argsort(logic_distance)  # 按照值从小到大排序,因此序号越小代表值越小代表越好
    return rank_lst
def error_level(pred_test_prob,true_test):#新增评价指标：严重性指标     
    error_level=[]
    pred_test_sort=np.argsort(-pred_test_prob, axis=1)
    for i in range(len(pred_test_prob)):
        if pred_test_sort[i][0]==true_test[i]:
            error_level.append(0)
        elif pred_test_sort[i][1]==true_test[i]:
            error_level.append(5)
        elif pred_test_sort[i][2]==true_test[i]:
            error_level.append(10)
        else:
            error_level.append(100)
    return error_level
def deeplogic_test(model, device,model_name,dataset_name,data,true_test):
    if args.dataset == 'cifar10':
        unit_topk=50
    if args.dataset == 'fashionminist':
        unit_topk=10
    model.eval()
    
    batch_size=128
    datalist=torch.split(data, batch_size, dim=0)
    labellist=torch.split(true_test, batch_size, dim=0)
    
    all_test_logic_distance=[]
    pred_test_prob=[]
    
    logicV2 = Logic(model,model_name)
    bcount=0
    start = time.time()
    for data_batch in datalist:
        label_batch=labellist[bcount]
        bcount+=1
        output=model(data_batch.to(device))
        prob = F.softmax(output)
        pred_one=prob.cpu().detach()
        pred_test_prob.append(pred_one)      
        
        logic_units,_=logicV2.cal_logic_units(data_batch,label_batch,unit_topk)#按照batch计算逻辑神经元
        N=logic_units.shape[0]
        batch_distance=[]
        for j in range(N):
            like_degree=logicV2.get_logic_similarity(set(logic_units[j]),label_batch[j])#计算逻辑度（测试优先级排序根据逻辑度大小排列）
            batch_distance.append(like_degree)
        all_test_logic_distance.append(torch.tensor(batch_distance,dtype=torch.double))    

    all_test_logic_distance=torch.cat(all_test_logic_distance,dim=0)
    all_test_logic_distance=all_test_logic_distance.numpy()
       
    pred_test_prob=torch.cat(pred_test_prob,dim=0)
    pred_test_prob=pred_test_prob.numpy()
    pred_test=np.argmax(pred_test_prob, axis=1)
    
    rank_lst = deep_logic(all_test_logic_distance)
    end = time.time()
    rank_lst_time = end-start
    df = pd.DataFrame([])

    true_test=true_test.cpu().numpy()
    df['right'] = (pred_test == true_test).astype('int')
    df['cam'] = 0
    df['cam'].loc[rank_lst] = list(range(1, len(rank_lst) + 1))
    df['cam_time'] = rank_lst_time
    df['rate'] = 0
    df['ctm'] = 0
    df['ctm'].loc[rank_lst] = list(range(1, len(rank_lst) + 1))
    df['ctm_time'] = rank_lst_time
    
    df['error_level']=error_level(pred_test_prob,true_test)
        
    if dataset_name=='cifar':
        df.to_csv('./all_output/output_cifar/{}/{}_deeplogic_0.csv'.format(model_name,dataset_name))
    if dataset_name=='fashionminist':
        df.to_csv('./all_output/output_fashionminist/{}/{}_deeplogic_0.csv'.format(model_name,dataset_name))

if __name__=='__main__':

    if args.white_box_attack:
        # white-box attack
        print('deeplogic')
        if args.model=='vgg16':
            model = vgg16_bn().to(device)
            model_name='vgg16'
            dataset_name='cifar'
        elif args.model=='resnet34':
            model = ResNet34().to(device)
            model_name='resnet34'
            dataset_name='cifar'
        elif args.model=='resnet18':
            model = ResNet18().to(device)
            model_name='resnet18'
            dataset_name='fashionminist'      
        elif args.model=='smallcnn':
            model = SmallCNN().to(device)
            model_name='smallcnn'
            dataset_name='fashionminist'
        model.load_state_dict(torch.load(args.model_path))

        deeplogic_test(model, device,model_name,dataset_name,data,true_test)
        
        

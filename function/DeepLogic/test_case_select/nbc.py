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
        images=torch.load('dataset/data/ckpt/images_of_TestCaseSet_resnet34_cifar10.pt')
        labels=torch.load('dataset/data/ckpt/labels_of_TestCaseSet_resnet34_cifar10.pt')
    elif args.model=='vgg16':
        images=torch.load('dataset/data/ckpt/images_of_TestCaseSet_vgg16_cifar10.pt')
        labels=torch.load('dataset/data/ckpt/labels_of_TestCaseSet_vgg16_cifar10.pt')
    data=images
    true_test=labels
if args.dataset == 'fashionminist':
    if args.model=='resnet18':
        images=torch.load('dataset/data/ckpt/images_of_TestCaseSet_resnet18_fashionminist.pt')
        labels=torch.load('dataset/data/ckpt/labels_of_TestCaseSet_resnet18_fashionminist.pt')
    elif args.model=='smallcnn':
        images=torch.load('dataset/data/ckpt/images_of_TestCaseSet_smallcnn_fashionminist.pt')
        labels=torch.load('dataset/data/ckpt/labels_of_TestCaseSet_smallcnn_fashionminist.pt')
    data=images
    true_test=labels    

class NeuronsActivate:
    def __init__(self, model,data,threshold):
        self.model = model
        self.data = data
        self.threshold = threshold
    def get_neurons_activate(self):
        sample_num=self.data.shape[0] #样本个数
        neurons_activate_dict=torch.zeros(sample_num,1).to(device)
        layer_dict = self.get_model_layers()
        for layer, module in layer_dict.items():
            outputs = torch.squeeze(self.extract_outputs(module))
            scaled_outputs = self.scale(outputs)
            sample_layer_outputs=scaled_outputs.view(sample_num,-1)  #sample_layer_outputs表示所有样本的某层输出--神经元激活值
            activation=torch.gt(sample_layer_outputs, self.threshold)  #大于门限则激活
            neurons_activate_dict=torch.cat([neurons_activate_dict, activation], dim=1)
        return neurons_activate_dict.detach().cpu().numpy()
    def step_through_model(self, model,prefix=''):
        for name, module in model.named_children():
            path = '{}/{}'.format(prefix, name)
            if (isinstance(module, nn.Conv1d)
                or isinstance(module, nn.Conv2d)
                or isinstance(module, nn.Linear)): # test for dataset
                yield (path, name, module)
            else:
                yield from self.step_through_model(module, path)
    def get_model_layers(self, cross_section_size=0):
        layer_dict = {}
        i = 0
        for (path, name, module) in self.step_through_model(self.model):
            layer_dict[str(i) + path] = module
            i += 1
        if cross_section_size > 0:
            target_layers = list(layer_dict)[0::cross_section_size] 
            layer_dict = { target_layer: layer_dict[target_layer] for target_layer in target_layers }
        return layer_dict

    def scale(self, out, rmax=1, rmin=0):
        output_std = (out - out.min()) / (out.max() - out.min())
        output_scaled = output_std * (rmax - rmin) + rmin
        return output_scaled

    def extract_outputs(self,module, force_relu=True):
        outputs = []      
        def hook(module, input, output):
            if force_relu:
                outputs.append(torch.relu(output))   
            else:
                outputs.append(output)
        handle = module.register_forward_hook(hook)     
        self.model(self.data)
        handle.remove()
        return torch.stack(outputs)
#按照样本的神经元覆盖率进行排序
class NBC():
    def __init__(self,model,test,std):#model模型，test测试用例集，std
        self.test = test
        self.std = std
        self.lst = []
        index_lst = []
        self.lst = list(zip(index_lst, self.lst))
        
        self.neuron_activate = []
        self.neuron_num = 0

    def fit(self):
        batch_size=128
        datalist=torch.split(data, batch_size, dim=0)
        neurons_activate=[]
        for data_batch in datalist:         
            na=NeuronsActivate(model,data_batch,0.0)
            batch_neurons_activate=na.get_neurons_activate()
            threshold_upper=np.mean(np.max(batch_neurons_activate, axis=1) + self.std * np.std(batch_neurons_activate, axis=1))
            threshold_lower=np.mean(np.min(batch_neurons_activate, axis=1) - self.std * np.std(batch_neurons_activate, axis=1))
            upper = (batch_neurons_activate > threshold_upper)  # upper是一个TrueFalse 矩阵,shape(用例数,神经元数)
            lower = (batch_neurons_activate < threshold_lower)  # lower是一个TrueFalse 矩阵,shape(用例数,神经元数)
            batch_coverage = np.sum(upper,axis=1)+np.sum(lower,axis=1)  # 统计激活了的神经元的个数

            neurons_activate.append(batch_coverage)           
            
        self.neuron_activate=np.concatenate(neurons_activate, axis=0)

    def rank_fast(self):
        rank_lst = np.argsort(self.neuron_activate)  # 按照值从小到大排序,因此序号越小代表值越小代表越好
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
def nbc_test(model, device,model_name,dataset_name,data,true_test):
    model.eval()
    
    batch_size=128
    datalist=torch.split(data, batch_size, dim=0)

    pred_test_prob=[]
    start = time.time()
    for data_batch in datalist:
        output=model(data_batch.to(device))
        prob = F.softmax(output)
        pred_one=prob.cpu().detach()
        pred_test_prob.append(pred_one)
    pred_test_prob=torch.cat(pred_test_prob,dim=0)
    pred_test_prob=pred_test_prob.numpy()
    
    pred_test=np.argmax(pred_test_prob, axis=1)
    true_test=true_test.cpu().numpy()
    
    ac = NBC(model,data.to(device),0.4)
    rate = ac.fit()
    rank_lst = ac.rank_fast()
    end = time.time()
    rank_lst_time = end - start

    df = pd.DataFrame([])
    df['right'] = (pred_test == true_test).astype('int')
    df['cam'] = 0
    df['cam'].loc[rank_lst] = list(range(1, len(rank_lst) + 1))
    df['cam_time'] = rank_lst_time
    df['rate'] = rate
    df['ctm'] = 0
    df['ctm'].loc[rank_lst] = list(range(1, len(rank_lst) + 1))
    df['ctm_time'] = rank_lst_time
    
    df['error_level']=error_level(pred_test_prob,true_test)
    
    if dataset_name=='cifar':
        df.to_csv('output/cache/test_case_select/all_output/output_cifar/{}/{}_nbc_0.csv'.format(model_name,dataset_name))
    if dataset_name=='fashionminist':
        df.to_csv('output/cache/test_case_select/all_output/output_fashionminist/{}/{}_nbc_0.csv'.format(model_name,dataset_name))

if __name__=='__main__':

    if args.white_box_attack:
        # white-box attack
        print('nbc')
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

        nbc_test(model, device,model_name,dataset_name,data,true_test)
        
        

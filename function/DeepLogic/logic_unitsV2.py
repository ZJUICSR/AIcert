#coding:utf-8
import os
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import datasets,transforms
import torchvision.models as models
from PIL import Image,ImageOps
import numpy as np
from matplotlib import pyplot as plt
import linecache

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N_CLASSES = 10

def logic_similarity(A, B):# 求集合 A 和集合 B 的Jaccard相似性
    nominator = A.intersection(B)
    # 求集合 A 和集合 B 的并集
    denominator = A.union(B)
    # 计算比率
    similarity = len(nominator)/len(denominator)
    return similarity

def logic_distance(A, B):# 求集合 A 和集合 B 的Jaccard距离
    nominator = A.symmetric_difference(B)
    # 求集合 A 和集合 B 的并集
    denominator = A.union(B)
    # 计算比率
    distance = len(nominator)/len(denominator)
    return distance

feat_result_input_std = []
feat_result_output_std = []

def get_features_hook_std(module, data_input, data_output):
    feat_result_input_std.append(data_input)
    feat_result_output_std.append(data_output)
    
#核心类：逻辑神经元相关计算
class Logic(object):
    def __init__(self, model:models,model_name='vgg16',device:torch.device=device):
        self.model_name=model_name
        self.device = device
        self.model = model.to(device).eval()
    
    #计算单个样本逻辑神经元序列(smallcnn)
    def cal_logic_units_smallcnn(self,data,label,logic_topk):
        self.model.feature_extractor.relu4.register_forward_hook(get_features_hook_std)#钩子获取模型运算中间结果

        count_samples = 0

        # clear feature blobs
        feat_result_input_std.clear()
        feat_result_output_std.clear()

        data, label = data.to(device) ,label.to(device)

        output = self.model(data)
        pred1 = torch.max(output, dim=1)[1]

        idx = torch.tensor(np.arange(data.shape[0]))
        #idx = np.where(label.cpu().numpy() != pred1.cpu().numpy())[0]#类别=0的样本
        #idx = torch.tensor(idx)
        count_samples += len(idx)
        idx=torch.tensor(idx).type(torch.long)
        if len(idx) > 0:
            feat1 = feat_result_input_std[0]
            feat2 = feat_result_output_std[0]
            feat_in = feat1[0][idx]
            feat_out = feat2[idx]
            if len(feat_out.shape) == 4:
                N, C, H, W = feat_out.shape
                feat_out = feat_out.view(N, C, H * W)
                feat_out = torch.mean(feat_out, dim=-1)
            N, C = feat_out.shape  
            feat_out_idx=torch.argsort(feat_out,descending=True)
            logic_units=torch.split(feat_out_idx,logic_topk,dim=1)[0]
        return logic_units.cpu().detach().numpy(),pred1.cpu().detach()
    #计算单个样本逻辑神经元序列(resnet18)
    def cal_logic_units_resnet18(self,data,label,logic_topk):
        self.model.layer4.register_forward_hook(get_features_hook_std)#钩子获取模型运算中间结果

        count_samples = 0

        # clear feature blobs
        feat_result_input_std.clear()
        feat_result_output_std.clear()

        data, label = data.to(device) ,label.to(device)

        output = self.model(data)
        pred1 = torch.max(output, dim=1)[1]

        idx = torch.tensor(np.arange(data.shape[0]))
        #idx = np.where(label.cpu().numpy() != pred1.cpu().numpy())[0]#类别=0的样本
        #idx = torch.tensor(idx)
        count_samples += len(idx)
        idx=torch.tensor(idx).type(torch.long)
        if len(idx) > 0:
            feat1 = feat_result_input_std[0]
            feat2 = feat_result_output_std[0]
            feat_in = feat1[0][idx]
            feat_out = feat2[idx]
            if len(feat_out.shape) == 4:
                N, C, H, W = feat_out.shape
                feat_out = feat_out.view(N, C, H * W)
                feat_out = torch.mean(feat_out, dim=-1)
            N, C = feat_out.shape  
            feat_out_idx=torch.argsort(feat_out,descending=True)
            logic_units=torch.split(feat_out_idx,logic_topk,dim=1)[0]
        return logic_units.cpu().detach().numpy(),pred1.cpu().detach()       
    #计算单个样本逻辑神经元序列(resnet34)
    def cal_logic_units_resnet34(self,data,label,logic_topk):
        self.model.layer4.register_forward_hook(get_features_hook_std)#钩子获取模型运算中间结果

        count_samples = 0

        # clear feature blobs
        feat_result_input_std.clear()
        feat_result_output_std.clear()

        data, label = data.to(device) ,label.to(device)

        output = self.model(data)
        pred1 = torch.max(output, dim=1)[1]

        idx = torch.tensor(np.arange(data.shape[0]))
        #idx = np.where(label.cpu().numpy() != pred1.cpu().numpy())[0]#类别=0的样本
        #idx = torch.tensor(idx)
        count_samples += len(idx)
        idx=torch.tensor(idx).type(torch.long)
        if len(idx) > 0:
            feat1 = feat_result_input_std[0]
            feat2 = feat_result_output_std[0]
            feat_in = feat1[0][idx]
            feat_out = feat2[idx]
            if len(feat_out.shape) == 4:
                N, C, H, W = feat_out.shape
                feat_out = feat_out.view(N, C, H * W)
                feat_out = torch.mean(feat_out, dim=-1)
            N, C = feat_out.shape  
            feat_out_idx=torch.argsort(feat_out,descending=True)
            logic_units=torch.split(feat_out_idx,logic_topk,dim=1)[0]
        return logic_units.cpu().detach().numpy(),pred1.cpu().detach()     
    #计算单个样本逻辑神经元序列(vgg16)
    def cal_logic_units_vgg16(self,data,label,logic_topk):#一批数据和标签，逻辑神经元取topk

        self.model.features[42].register_forward_hook(get_features_hook_std)#钩子获取模型运算中间结果

        count_samples = 0

        # clear feature blobs
        feat_result_input_std.clear()
        feat_result_output_std.clear()

        data, label = data.to(device) ,label.to(device)

        output = self.model(data)
        pred1 = torch.max(output, dim=1)[1]

        idx = torch.tensor(np.arange(data.shape[0]))
        #idx = np.where(label.cpu().numpy() != pred1.cpu().numpy())[0]#类别=0的样本
        #idx = torch.tensor(idx)
        count_samples += len(idx)
        idx=torch.tensor(idx).type(torch.long)
        if len(idx) > 0:
            feat1 = feat_result_input_std[0]
            feat2 = feat_result_output_std[0]
            feat_in = feat1[0][idx]
            feat_out = feat2[idx]
            if len(feat_out.shape) == 4:
                N, C, H, W = feat_out.shape
                feat_out = feat_out.view(N, C, H * W)
                feat_out = torch.mean(feat_out, dim=-1)
            N, C = feat_out.shape  
            feat_out_idx=torch.argsort(feat_out,descending=True)
            logic_units=torch.split(feat_out_idx,logic_topk,dim=1)[0]
        return logic_units.cpu().detach().numpy(),pred1.cpu().detach()

    def cal_logic_units(self,data,label,logic_topk):
        if self.model_name=='vgg16':
            return self.cal_logic_units_vgg16(data,label,logic_topk)
        elif self.model_name=='resnet34':
            return self.cal_logic_units_resnet34(data,label,logic_topk)
        elif self.model_name=='resnet18':
            return self.cal_logic_units_resnet18(data,label,logic_topk)
        elif self.model_name=='smallcnn':
            return self.cal_logic_units_smallcnn(data,label,logic_topk)
        
    #查询某个类的逻辑神经元序列
    def get_unit_of_class(self,vlabel):
        if self.model_name=='vgg16':
            #classname = class_label[str(vlabel)][1]
            oline = linecache.getline('dataset/data/repository/vgg16/unit.txt', vlabel + 1)
            osplit=oline.replace('\n','').split(':')[2]
            result=osplit.split(',')
            return list(map(int, result))
        elif self.model_name=='resnet34':
            #classname = class_label[str(vlabel)][1]
            oline = linecache.getline('dataset/data/repository/resnet34/unit.txt', vlabel + 1)
            osplit=oline.replace('\n','').split(':')[2]
            result=osplit.split(',')
            return list(map(int, result))
        elif self.model_name=='resnet18':
            #classname = class_label[str(vlabel)][1]
            oline = linecache.getline('dataset/data/repository/resnet18/unit.txt', vlabel + 1)
            osplit=oline.replace('\n','').split(':')[2]
            result=osplit.split(',')
            return list(map(int, result))
        elif self.model_name=='smallcnn':
            #classname = class_label[str(vlabel)][1]
            oline = linecache.getline('dataset/data/repository/smallcnn/unit.txt', vlabel + 1)
            osplit=oline.replace('\n','').split(':')[2]
            result=osplit.split(',')
            return list(map(int, result))
        
    #计算当前样本逻辑神经元与所属类逻辑神经元相似度--逻辑度大小
    def get_logic_similarity(self,sample_logic_units,vlabel):
        if self.model_name=='vgg16':
            global_class_units_file='dataset/data/repository/vgg16/unit.txt'
        elif self.model_name=='resnet34':
            global_class_units_file='dataset/data/repository/resnet34/unit.txt'
        elif self.model_name=='resnet18':
            global_class_units_file='dataset/data/repository/resnet18/unit.txt'
        elif self.model_name=='smallcnn':
            global_class_units_file='dataset/data/repository/smallcnn/unit.txt'
        #当前样本对应类的相似性
        oline = linecache.getline(global_class_units_file, vlabel + 1)
        osplit=oline.replace('\n','').split(':')[2]
        result=osplit.split(',')
        class_logic_units=set(map(int, result))
        k_union=sample_logic_units.intersection(class_logic_units)
        now_like=len(k_union)
        #print(vlabel,len(k_union))
        #当前样本与其他类的距离之和
        other_like=0
        for i in range(N_CLASSES):
            if i!=vlabel:
                oline = linecache.getline(global_class_units_file, i + 1)
                osplit=oline.replace('\n','').split(':')[2]
                result=osplit.split(',')
                class_logic_units=set(map(int, result))
                k_union=sample_logic_units.intersection(class_logic_units)
                other_like=other_like+len(k_union)
                #print(vlabel,len(k_union))
        like_degree=now_like*1.0/other_like
        return like_degree



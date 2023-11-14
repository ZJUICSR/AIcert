#coding:utf-8
from __future__ import print_function
import os,sys
sys.path.append(os.path.dirname(__file__).rsplit('/',1)[0])
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
# from models.resnet import *
# from models.vggnet import *
# from models.mynet import *
from model.model_net.resnet import *
from model.model_net.vggnet import *
from model.model_net.mynet import *
from logic_unitsV2 import *
import numpy as np
from tqdm import tqdm
import time
import pandas as pd
import glob
import os
import csv
import random
import pandas as pd


# parser = argparse.ArgumentParser(description='PyTorch dnn test')
# parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
#                     help='input batch size for testing (default: 200)')
# parser.add_argument('--dataset', default='cifar10', help='use what dataset')
# parser.add_argument('--no-cuda', action='store_true', default=False,
#                     help='disables CUDA training')
# parser.add_argument('--epsilon', default=0.031,
#                     help='perturbation')
# parser.add_argument('--num-steps', default=20,
#                     help='perturb number of steps')
# parser.add_argument('--step-size', default=0.003,
#                     help='perturb step size')
# parser.add_argument('--random',
#                     default=True,
#                     help='random initialization for PGD')
# parser.add_argument('--model',
#                     default='resnet18',
#                     help='model name for evaluation')
# parser.add_argument('--model-path',
#                     default='./model-cifar-vggNet/model-wideres-epoch58.pt',
#                     help='model for white-box attack evaluation')
# parser.add_argument('--white-box-attack', default=True,
#                     help='whether perform white-box attack')

# args = parser.parse_args()

# settings
# use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
# kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


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
    if dataset_name == 'cifar':
        unit_topk=50
    if dataset_name == 'fashionminist':
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

    df.to_csv('output/cache/test_case_select/all_output/output_cifar/{}/{}_deeplogic_0.csv'.format(model_name,dataset_name))

from test_case_select.show_result import *
def evaluate_deeplogic(dataset, modelname, out_path='./', logging=None):
        # white-box attack
        if dataset == 'cifar10':
            # 加载测试用例集
            if modelname == 'resnet34':
                images = torch.load('dataset/data/ckpt/images_of_TestCaseSet_resnet34_cifar10.pt')
                labels = torch.load('dataset/data/ckpt/labels_of_TestCaseSet_resnet34_cifar10.pt')
            elif modelname == 'vgg16':
                images = torch.load('dataset/data/ckpt/images_of_TestCaseSet_vgg16_cifar10.pt')
                labels = torch.load('dataset/data/ckpt/labels_of_TestCaseSet_vgg16_cifar10.pt')
            data = images
            true_test = labels
        if dataset == 'fashionmnist':
            if modelname == 'resnet18':
                images = torch.load('dataset/data/ckpt/images_of_TestCaseSet_resnet18_fashionminist.pt')
                labels = torch.load('dataset/data/ckpt/labels_of_TestCaseSet_resnet18_fashionminist.pt')
            elif modelname == 'smallcnn':
                images = torch.load('dataset/data/ckpt/images_of_TestCaseSet_smallcnn_fashionminist.pt')
                labels = torch.load('dataset/data/ckpt/labels_of_TestCaseSet_smallcnn_fashionminist.pt')
            data = images
            true_test = labels

        if modelname=='vgg16':
            model = vgg16_bn().to(device)
            model_name='vgg16'
            dataset_name='cifar'
            model_path='model/ckpt/Standard-cifar10-model-vgg16-epoch300.pt'
        elif modelname=='resnet34':
            model = ResNet34().to(device)
            model_name='resnet34'
            dataset_name='cifar'
            model_path = 'model/ckpt/Standard-cifar10-model-resnet34-epoch300.pt'
        elif modelname=='resnet18':
            model = ResNet18().to(device)
            model_name='resnet18'
            dataset_name='fashionminist'
            model_path = 'model/ckpt/Standard-fashionminist-model-resnet18-epoch300.pt'
        model.load_state_dict(torch.load(model_path))

        deeplogic_test(model, device,model_name,dataset_name,data,true_test)
        apfd=statistic_apfd(model_name)

        res=show(dataset_name,model_name,out_path,apfd)
        with open(res["data"], 'r') as f:
            result = json.load(f)
            # print(result)
        res["result"] = result
        return res


df = pd.DataFrame(columns=['metrics', 'model', 'time','apfd','rda'])
df['metrics']=range(32)
row=0
logic_apfd=0.0
# 获得apfd figure 表格,用于图像绘制

class Item:
    def __init__(self, item, header, mode, metric):
        self.item = item
        self.header = header
        self.mode = mode
        self.metric = metric

    def get_order(self):
        order = int(self.item[self.header[self.mode]])
        if order == 0 and self.mode == "cam":
            if "ctm" in self.header:
                if self.metric == "lsc" or self.metric == "dsc":
                    order = 500000 + random.randint(1, 500000)
                else:
                    order = 500000 + int(self.item[self.header["ctm"]])
            else:
                print("!!!!!! cam has order 0, but the sheet does not have ctm")
                order = 500000
        return order

    def get_best_order(self):
        right = int(self.item[self.header["right"]])
        if right == 1:
            return 1000
        else:
            return 0

    def get_worst_order(self):
        right = int(self.item[self.header["right"]])
        if right == 1:
            return 0
        else:
            return 1000


def get_order(item):
    return item.get_order()


def get_best_order(item):
    return item.get_best_order()


def get_worst_order(item):
    return item.get_worst_order()


metric_index = {
    "nac": 0,
    "nbc": 1,
    "deepgini": 2,
    "tknc": 3,
    "dsc": 4,
    "random": 5,
    "deeplogic": 6,
    "entropy": 7,
    "mcp":8
}

metric_conf = [
    ["cam"],
    ["cam"],
    ["cam"],
    ["cam"],
    ["cam"],
    ["cam"],
    ["cam"],
    ["cam"],
    ["cam"]
]


def calc_apfd(items):
    n_tests = len(items)
    sigma_o = 0
    k_mis_tests = 0
    o = 0
    for i in items:
        o = o + 1
        if int(i.item[i.header["right"]]) == 0:
            sigma_o = sigma_o + o
            k_mis_tests = k_mis_tests + 1

    apfd = 1 - (1.0 * sigma_o / (k_mis_tests * n_tests)) + 1.0 / (2 * n_tests)
    return apfd


def best(items):
    items.sort(key=get_best_order)
    return calc_apfd(items)


def worst(items):
    items.sort(key=get_worst_order)
    return calc_apfd(items)


def get_apfd(inputfile, method, sortmode, metric, verbose=False):
    items = []
    header_map = {}
    csv_file = csv.reader(open(inputfile, 'r'))
    i = 0
    for line in csv_file:
        if i == 0:
            i += 1
            j = 0
            for x in line:
                header_map[x] = j
                j += 1
            if sortmode not in header_map.keys():
                print("=======================================")
                print(method + " does not have mode " + sortmode)
                print("=======================================")
                return None, None, None
            if "right" not in header_map.keys():
                print("=======================================")
                print(method + " does not col right")
                print("=======================================")
                return None, None, None
        else:
            items.append(Item(line, header_map, sortmode, metric))

    best_apfd = best(items)
    worst_apfd = worst(items)

    items.sort(key=get_order)
    orig_apfd = calc_apfd(items)

    norm_apfd = (orig_apfd - worst_apfd) / (best_apfd - worst_apfd)
    #if verbose:
    #    print("best : " + str(best_apfd))
    #    print("worst : " + str(worst_apfd))

    #    print(sortmode + " orig apfd : " + str(orig_apfd))
    #    print(sortmode + " norm apfd : " + str(norm_apfd))
    return norm_apfd, items, header_map

def calc_time_rda(items):
    time=0
    n_tests = len(items)
    sigma_o = 0
    k_mis_tests = 0
    o = 0
    for i in items:
        time=float(i.item[i.header["cam_time"]])
        o = int(i.item[i.header["error_level"]])*int(i.item[i.header["cam"]])
        if int(i.item[i.header["right"]]) == 0:
            sigma_o = sigma_o + o
            k_mis_tests = k_mis_tests + 1

    #print('分子',sigma_o)
    rda = (1.0 * sigma_o / (k_mis_tests * n_tests)) + 1.0 / (2 * n_tests)
    return rda,time

def get_time_rda(inputfile, method, sortmode, metric, verbose):
    items = []
    header_map = {}
    csv_file = csv.reader(open(inputfile, 'r'))
    i = 0
    for line in csv_file:
        #print(line)
        if i == 0:
            i += 1
            j = 0
            for x in line:
                header_map[x] = j
                j += 1
            if sortmode not in header_map.keys():
                print("=======================================")
                print(method + " does not have mode " + sortmode)
                print("=======================================")
            if "right" not in header_map.keys():
                print("=======================================")
                print(method + " does not col right")
                print("=======================================")
        else:
            items.append(Item(line, header_map, sortmode, metric))
    items.sort(key=get_order)
    orig_rda,time = calc_time_rda(items)
    return orig_rda,time

def compute(csvname, abspath, model_name,target_modelname,outputdir="", to_csv=False, verbose=False):
    conf = csvname.split("_")
    dataset = conf[0]
    withadv = conf[1] == "adv"
    if withadv:
        metric = conf[2].lower()
        metric_param = "_".join(conf[3:])
    else:
        metric = conf[1].lower()
        metric_param = "_".join(conf[2:])
    #if verbose:
    #    print("dataset: " + dataset + "; withadv: " + str(withadv) + "; metric: " + metric + "; param: " + metric_param)

    inputfile = abspath
    sortmodes = metric_conf[metric_index[metric]]
    res = {"cam": "N/A", "ctm": "N/A"}
    for sortmode in sortmodes:
        method = sortmode + "_" + os.path.basename(inputfile)
        outputfile = outputdir + method

        # if metric == "kmnc" and sortmode == "cam" and withadv == True:
        #     # continue
        #     print(1)
        norm_apfd, items, header_map = get_apfd(inputfile, method, sortmode, metric, verbose)
        if norm_apfd is None:
            continue
        res[sortmode] = norm_apfd
        orig_rda,usetime = get_time_rda(inputfile, method, sortmode, metric, verbose)
        if to_csv:
            with open(outputfile, "w") as o:
                o.write(method + "\n")
                sum = 0
                for i in items:
                    if int(i.item[header_map["right"]]) == 0:
                        sum += 1
                    o.write(str(sum) + "\n")
    global row
    global df
    global logic_apfd
    df['metrics'].loc[row] = metric
    df['model'].loc[row] = model_name
    df['time'].loc[row]= usetime
    df['apfd'].loc[row] = norm_apfd
    df['rda'].loc[row] = orig_rda
    row+=1
    if metric=='deeplogic' and model_name==target_modelname:
        logic_apfd=norm_apfd
    return res


def statistic_apfd(target_modelname):
    input_base_path = "output/cache/test_case_select/all_output"
    output_base_path = "output/cache/test_case_select/result/apfd_figure_csv"
    dir_list = ["output_cifar","output_fashionminist","output_imagenet"]
    for path_dir in dir_list:
        dataset_name = os.path.basename(path_dir)[7:]
        lst = glob.glob(input_base_path + '/' + path_dir + '/*')
        for inputdir in lst:  # 遍历每个模型
            model_name = os.path.basename(inputdir)
            outputdir = output_base_path + "/" + dataset_name + "/" + model_name + "/"
            if not os.path.exists(outputdir):
                os.makedirs(outputdir)
            #print(inputdir, outputdir)
            for filename in os.listdir(inputdir):
                if filename.endswith(".csv"):
                    abspath = os.path.join(inputdir, filename)
                    #print("analyzing " + filename + "...")
                    res = compute(filename, abspath,model_name,target_modelname, outputdir=outputdir, to_csv=True, verbose=True)
                    #print(res)

                    #print("")
    #print(df)
    df.sort_values("metrics",inplace=True)
    df.to_csv('output/cache/test_case_select/table.csv')
    return logic_apfd

        
        

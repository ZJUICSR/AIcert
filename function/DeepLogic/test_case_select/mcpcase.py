#coding:utf-8
from __future__ import print_function
import sys
sys.path.append("..")
import os
import argparse
from models.resnet import *
from models.vggnet import *
from models.mynet import *
import numpy as np
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

def test(testset,true_label, model, num_classes=10):
    batch_size = 128
    testsize = testset.shape[0]

    datalist=torch.split(testset, batch_size, dim=0)
    labellist=torch.split(true_label, batch_size, dim=0)
    correct = 0
    total = 0
    model.eval()

    # test
    correct_array = np.zeros((testsize, ), dtype=int)
    logits = np.zeros((testsize, num_classes), dtype=float)
    with torch.no_grad():
        batch_idx=0
        #for batch_idx, (inputs, labels) in enumerate(testloader):
        for data_batch in datalist:
            label_batch=labellist[batch_idx] 
            inputs, labels = data_batch.to(device), label_batch.to(device)
            outputs= model(inputs)
            _, pred = outputs.max(1)
            logits[(batch_idx)*batch_size: (batch_idx+1)*batch_size] = outputs.cpu().numpy()
            correct_array[(batch_idx)*batch_size: (batch_idx+1)*batch_size] = pred.eq(labels).cpu().numpy().astype(int)
            correct += pred.eq(labels).sum().item() 
            total += labels.size(0)
            batch_idx+=1

    return correct_array, logits    
## mcpcase
def mcpcase(model2test,test_inputs,true_label):
    import mcp
    import csv, ast
    import copy
    from scipy.special import softmax
    p_budget_lst = [1, 2, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100] # percentage of budget 
    num_classes=10
    unlabeled_indices = np.arange(len(test_inputs))
    # gt
    correct_array, _ = test(test_inputs,true_label, model2test, num_classes=num_classes)
    misclass_array = (correct_array==0).astype(int)

    _, logits = test(test_inputs,true_label, model2test, num_classes=num_classes)
    prob = softmax(logits, axis=1)
    dicratio=[[] for i in range(num_classes*num_classes)]
    dicindex=[[] for i in range(num_classes*num_classes)]

    rank_lst=[]

    for i in range(len(prob)):
        act=prob[i]
        max_index,sec_index,ratio = mcp.get_boundary_priority(act)#max_index 
        dicratio[max_index*num_classes+sec_index].append(ratio)
        dicindex[max_index*num_classes+sec_index].append(i)

    for p_budget in p_budget_lst:
        print("\n ###### budget percent is {} ######".format(p_budget))
        model2test_temp = copy.deepcopy(model2test)
        dicindex_temp = copy.deepcopy(dicindex)
        dicratio_temp = copy.deepcopy(dicratio)

        budget = int(p_budget*len(unlabeled_indices)/100.0)

        if p_budget == 100:
            selected = list(np.arange(len(unlabeled_indices)))
        else:
            selected = mcp.select_from_firstsec_dic(budget, dicratio_temp, dicindex_temp, num_classes=num_classes)
        newselect=list((set(selected))^(set(rank_lst)))
        rank_lst.extend(list(newselect))
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
def mcp_test(model, device,model_name,dataset_name,data,true_test):
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
    rank_lst = mcpcase(model,data,true_test)
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
        df.to_csv('./all_output/output_cifar/{}/{}_mcp_0.csv'.format(model_name,dataset_name))
    if dataset_name=='fashionminist':
        df.to_csv('./all_output/output_fashionminist/{}/{}_mcp_0.csv'.format(model_name,dataset_name))

if __name__=='__main__':

    if args.white_box_attack:
        # white-box attack
        print('mcp')
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

        mcp_test(model, device,model_name,dataset_name,data,true_test)
        
        

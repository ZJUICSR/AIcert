#coding:utf-8
import sys
sys.path.append("..")
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from torch.autograd import Variable
from models.resnet import *
from models.vggnet import *
from models.mynet import *
import torch.optim as optim
import numpy as np
import foolbox as fb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon,
                  num_steps,
                  step_size,
                  random=True):
    #out = model(X)
    #err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    #err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    #print('err pgd (white-box): ', err_pgd)
    return X_pgd
def one_hot_tensor(y_batch_tensor, num_classes, device):
    y_tensor = torch.cuda.FloatTensor(y_batch_tensor.size(0),
                                      num_classes).fill_(0)
    y_tensor[np.arange(len(y_batch_tensor)), y_batch_tensor] = 1.0
    return y_tensor

class CWLoss(nn.Module):
    def __init__(self, num_classes, margin=50, reduce=True):
        super(CWLoss, self).__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.reduce = reduce
        return

    def forward(self, logits, targets):
        """
        :param inputs: predictions
        :param targets: target labels
        :return: loss
        """
        onehot_targets = one_hot_tensor(targets, self.num_classes,
                                        targets.device)

        self_loss = torch.sum(onehot_targets * logits, dim=1)
        other_loss = torch.max(
            (1 - onehot_targets) * logits - onehot_targets * 1000, dim=1)[0]

        loss = -torch.sum(torch.clamp(self_loss - other_loss + self.margin, 0))

        if self.reduce:
            sample_num = onehot_targets.shape[0]
            loss = loss / sample_num

        return loss
    
def _cw_whitebox(model,
                 X,
                 y,
                 epsilon,
                 num_steps,
                 step_size,
                 random=True):
    #out = model(X)
    #err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = CWLoss(10)(model(X_pgd), y)   #cifar10--num_classes=10,imagnet--num_classes=1000
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    #err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    #print('err cw (white-box): ', err_pgd)
    return X_pgd

def get_testcase(model_name,dataset_name,model,val_dataloader):
    
    all_test_data=[]
    all_test_label=[]
    
    clean_set_num=70 #控制生成测试集的小组数目
    for data, label in val_dataloader:
        if clean_set_num>0:
            if dataset_name=='fashionminist':              
                # 因为FashionMNIST输入为单通道图片，需要转换为三通道
                data = np.array(data)
                data = data.transpose((1, 0, 2, 3))  # array 转置
                data = np.concatenate((data, data, data), axis=0)  # 维度拼接
                data = data.transpose((1, 0, 2, 3))  # array 转置回来
                data = torch.tensor(data)  # 将 numpy 数据格式转为 tensor  
            data, label = data.to(device), label.to(device)
            all_test_data.append(data)#clean
            all_test_label.append(label)
            clean_set_num=clean_set_num-1
        else:
            break
            
    adv_set_num=20#控制生成测试集的小组数目
    for data, label in val_dataloader:
        if adv_set_num>0:
            if dataset_name=='fashionminist':              
                # 因为FashionMNIST输入为单通道图片，需要转换为三通道
                data = np.array(data)
                data = data.transpose((1, 0, 2, 3))  # array 转置
                data = np.concatenate((data, data, data), axis=0)  # 维度拼接
                data = data.transpose((1, 0, 2, 3))  # array 转置回来
                data = torch.tensor(data)  # 将 numpy 数据格式转为 tensor            
            data, label = data.to(device), label.to(device)
            #X, y = Variable(data, requires_grad=True), Variable(label)
            X, y = Variable(data), Variable(label)
            fmodel = fb.PyTorchModel(model,bounds=(0,1))
            
            attack = fb.attacks.FGSM()
            _, one_fgsm, success = attack(fmodel,X, y, epsilons=0.093)
            all_test_data.append(one_fgsm)#fgsm
            all_test_label.append(label)
            #one_fgsm = _pgd_whitebox(model, X, y, epsilon=8/255.0, num_steps=1,step_size=8/255.0)
            adv_set_num=adv_set_num-1
        else:
            break
    adv_set_num=20#控制生成测试集的小组数目
    for data, label in val_dataloader:
        if adv_set_num>0:
            if dataset_name=='fashionminist':              
                # 因为FashionMNIST输入为单通道图片，需要转换为三通道
                data = np.array(data)
                data = data.transpose((1, 0, 2, 3))  # array 转置
                data = np.concatenate((data, data, data), axis=0)  # 维度拼接
                data = data.transpose((1, 0, 2, 3))  # array 转置回来
                data = torch.tensor(data)  # 将 numpy 数据格式转为 tensor            
            data, label = data.to(device), label.to(device)
            #X, y = Variable(data, requires_grad=True), Variable(label)
            X, y = Variable(data), Variable(label)
            fmodel = fb.PyTorchModel(model,bounds=(0,1))
            
            attack=fb.attacks.L2ProjectedGradientDescentAttack(steps=20)
            _, one_pgd, success = attack(fmodel,X, y, epsilons=0.1)
            all_test_data.append(one_pgd)
            all_test_label.append(label)                
            #one_pgd = _pgd_whitebox(model, X, y, epsilon=0.031,num_steps=10, step_size=0.003)
            adv_set_num=adv_set_num-1
        else:
            break

    adv_set_num=20#控制生成测试集的小组数目
    for data, label in val_dataloader:
        if adv_set_num>0:
            if dataset_name=='fashionminist':              
                # 因为FashionMNIST输入为单通道图片，需要转换为三通道
                data = np.array(data)
                data = data.transpose((1, 0, 2, 3))  # array 转置
                data = np.concatenate((data, data, data), axis=0)  # 维度拼接
                data = data.transpose((1, 0, 2, 3))  # array 转置回来
                data = torch.tensor(data)  # 将 numpy 数据格式转为 tensor            
            data, label = data.to(device), label.to(device)
            #X, y = Variable(data, requires_grad=True), Variable(label)
            X, y = Variable(data), Variable(label)
            fmodel = fb.PyTorchModel(model,bounds=(0,1))

            attack = fb.attacks.L2CarliniWagnerAttack()
            _, one_cw, success = attack(fmodel,X, y, epsilons=0.2)
            #one_cw = _cw_whitebox(model, X, y, epsilon=0.031,num_steps=10, step_size=0.003)
            all_test_data.append(one_cw)#cw
            all_test_label.append(label)
            adv_set_num=adv_set_num-1
        else:
            break

    all_test_data=torch.cat(all_test_data,dim=0)
    all_test_label=torch.cat(all_test_label,dim=0)
    print(all_test_data.shape)
    print(all_test_label.shape)
    torch.save(all_test_data,'images_of_TestCaseSet_{}_{}.pt'.format(model_name,dataset_name))
    torch.save(all_test_label,'labels_of_TestCaseSet_{}_{}.pt'.format(model_name,dataset_name))

  
batch_size=128

'''
#vgg16+cifar10
val_dataset = datasets.CIFAR10(root='../dataset/data', train=False,download=False, transform=transforms.ToTensor())
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
model = vgg16_bn().to(device)
model.load_state_dict((torch.load('../adv_train/model-vgg16-cifar10/Standard-cifar10-model-vgg16-epoch300.pt')))#评估普通模型-干净样本准确率
model = model.to(device).eval()
get_testcase("vgg16","cifar10",model,val_dataloader)
print("vgg16+cifar10 testcaseset ok!!!")

#resnet34+cifar10
val_dataset = datasets.CIFAR10(root='../dataset/data', train=False,download=False, transform=transforms.ToTensor())
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
model = ResNet34().to(device)
model.load_state_dict((torch.load('../adv_train/model-resnet34-cifar10/Standard-cifar10-model-resnet34-epoch300.pt')))#评估普通模型-干净样本准确率
model = model.to(device).eval()
get_testcase("resnet34","cifar10",model,val_dataloader)
print("resnet34+cifar10 testcaseset ok!!!")
'''
#resnet18+fashionminist
val_dataset = datasets.FashionMNIST(root='../dataset/data', train=False,download=False, transform=transforms.ToTensor())
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
model = ResNet18().to(device)
model.load_state_dict((torch.load('model/ckpt/Standard-fashionminist-model-resnet18-epoch300.pt')))#评估普通模型-干净样本准确率
model = model.to(device).eval()
get_testcase("resnet18","fashionminist",model,val_dataloader)
print("resnet18+fashionminist testcaseset ok!!!")

#smallcnn+fashionminist
val_dataset = datasets.FashionMNIST(root='../dataset/data', train=False, download=True, transform=transforms.ToTensor())
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
model = SmallCNN().to(device)
model.load_state_dict((torch.load('model/ckpt/Standard-fashionminist-model-smallcnn-epoch300.pt')))#评估普通模型-干净样本准确率
model = model.to(device).eval()
get_testcase("smallcnn","fashionminist",model,val_dataloader)
print("smallcnn+fashionminist testcaseset ok!!!")
import json


import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from thop import profile
from torch.nn import CrossEntropyLoss
import sys,os
from  .models import Models
import torch
cur_path=os.path.abspath(__file__)
parent=os.path.dirname(cur_path)
parent=os.path.dirname(parent)
dict=(os.path.join(parent,'auto_LiRPA'))
print(parent)
sys.path.append(parent)

# from  auto_LiRPA.perturbations import *
from  auto_LiRPA import *
import numpy as np
# from  auto_LiRPA.utils import  logger
from PIL import Image

def check_safe(lb,ub,predicted):
    target_low=lb[predicted]
    for i,up in enumerate(ub):
        if i!=predicted and target_low<up:
            return False
    return True

def auto_verify_img(net,data,eps,input_file='./test.jpg',device='cuda'):
    bound_opts=None
    base=os.path.join(os.getcwd(),"model","ckpt")
    auto_LiRPA_pretrain=os.path.join(base,'verify_checkpoints')
    if net=='cnn_7layer_bn':
        model='cnn_7layer_bn'
    elif net=='Densenet' and data=='CIFAR':
        model='Densenet_cifar_32'
    elif net=='Resnet' and data=='MNIST':
        model='resnet'
    elif net=='Resnet' and data=='CIFAR':
        model='ResNeXt_cifar' 
    elif net=='Wide Resnet' and data=='CIFAR':
        model='wide_resnet_cifar_bn_wo_pooling'

    if model=='cnn_7layer_bn' and data=='CIFAR':
        load=f'{auto_LiRPA_pretrain}/cnn_7layer_bn_cifar'
    elif model=='cnn_7layer_bn' and data=='MNIST':
        load=f'{auto_LiRPA_pretrain}/natural_cnn_7layer_bn_mnist'
    elif model=='Densenet_cifar_32':
        load=f'{auto_LiRPA_pretrain}/Densenet_cifar'
    elif model=='resnet':
        load=f'{auto_LiRPA_pretrain}/ResNet_mnist'
    elif model=='ResNeXt_cifar':
        load=f'{auto_LiRPA_pretrain}/ResNeXt_cifar'
    elif model=='wide_resnet_cifar_bn_wo_pooling':
        load=f'{auto_LiRPA_pretrain}/wide_resnet_cifar_bn_wo_pooling_dropout'
    if data == 'MNIST':
        model_ori = Models[model](in_ch=1, in_dim=28)
    else:
        print(model)
        model_ori = Models[model](in_ch=3, in_dim=32)
    checkpoint = torch.load(load)
    epoch, state_dict = checkpoint['epoch'], checkpoint['state_dict']
    opt_state = None
    try:
        opt_state = checkpoint['optimizer']
    except KeyError:
        print('no opt_state found')
    for k, v in state_dict.items():
        assert torch.isnan(v).any().cpu().numpy() == 0 and torch.isinf(v).any().cpu().numpy() == 0
    model_ori.load_state_dict(state_dict)
    # utils.logger.info('Checkpoint loaded: {}'.format(load))
    if data == 'MNIST':
        dummy_input = torch.randn(1, 1, 28, 28)
        transform=transforms.ToTensor()
    elif data == 'CIFAR':
        dummy_input = torch.randn(1, 3, 32, 32)
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

        transform=transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    #transforms.RandomCrop(32, 4, padding_mode='edge'),
                    transforms.ToTensor(),
                    normalize])
    

    image = Image.open(input_file).convert('RGB')
    print(input_file)
    
    
    image= transform(image)
    image=image.view(1,image.shape[0],image.shape[1],image.shape[2]).cuda()
    model_ori=model_ori.cuda()

    model = BoundedModule(model_ori, torch.zeros_like(image), bound_opts={'relu':bound_opts}, device=device)

    model.eval()
    
    
    _, predicted = torch.max(model(image), 1) 
    norm = np.inf
    ptb = PerturbationLpNorm(norm = norm, eps = eps)
    bounded_image = BoundedTensor(image, ptb)
    print('Model prediction:', model(bounded_image))
    with torch.no_grad():  # If gradients of the bounds are not needed, we can use no_grad to save memory.
        lb2, ub2 = model.compute_bounds(x=(bounded_image,), method='CROWN-IBP')
        lb1, ub1 = model.compute_bounds(x=(bounded_image,), method='IBP')
        #lb, ub = model.compute_bounds(x=(bounded_image,), method='CROWN')
    score_IBP,score_CROWN=1,1
    for i in range(0,101,1):
        eps=i/100
        ptb = PerturbationLpNorm(norm = norm, eps = eps)
        with torch.no_grad():
            bounded_image = BoundedTensor(image, ptb)
            lb, ub = model.compute_bounds(x=(bounded_image,), method='CROWN-IBP')
            lb=lb.cpu().numpy()[0].tolist()
            ub=ub.cpu().numpy()[0].tolist()
            print(eps,check_safe(lb,ub,predicted))
            if not check_safe(lb,ub,predicted):
                break
            score_CROWN=int(eps*100)
    for i in range(0,101,1):
        eps=i/100
        ptb = PerturbationLpNorm(norm = norm, eps = eps)
        with torch.no_grad():
            bounded_image = BoundedTensor(image, ptb)
            lb, ub = model.compute_bounds(x=(bounded_image,), method='IBP')
            lb=lb.cpu().numpy()[0].tolist()
            ub=ub.cpu().numpy()[0].tolist()
            if not check_safe(lb,ub,predicted):
                break
            score_IBP=int(eps*100)


    return lb1.cpu().numpy()[0].tolist(),ub1.cpu().numpy()[0].tolist(),lb2.cpu().numpy()[0].tolist(),ub2.cpu().numpy()[0].tolist(),predicted.cpu().numpy().tolist()[0],score_IBP,score_CROWN
# verify_img('ResNeXt_cifar','CIFAR',0.1)
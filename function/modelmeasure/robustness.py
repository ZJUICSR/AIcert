import os.path as osp
import json
import time
import copy
import numpy as np
import shutil
import torch
from torch.autograd import Variable
# from torch.autograd.gradcheck import zero_gradients
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms, models
from tqdm import trange
import matplotlib.pyplot as plt
import collections

current_dir = osp.dirname(osp.realpath(__file__))
output_dict = {}



def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, collections.abc.Iterable):
        for elem in x:
            zero_gradients(elem)


def handle_imagenet(image, handler):
    """[summary]

    Args:
        image ([type]): [description]
        handler ([type]): [description]

    Returns:
        [type]: [description]
    """
    if handler == 0:
        image[:, 0, :, :] = image[:, 0, :, :] * 0.229 + 0.485
        image[:, 1, :, :] = image[:, 1, :, :] * 0.224 + 0.456
        image[:, 2, :, :] = image[:, 2, :, :] * 0.225 + 0.406
    elif handler == 1:
        image[:, 0, :, :] = (image[:, 0, :, :] - 0.485) / 0.229
        image[:, 1, :, :] = (image[:, 1, :, :] - 0.456) / 0.224
        image[:, 2, :, :] = (image[:, 2, :, :] - 0.406) / 0.225
    
    return image

def bim(device, model, sample, label, is_imagenet, tmp_eps):
    """Basic Iterative Method（BIM）对抗攻击算法
    非定向迭代攻击
    论文：A. Kurakin, I. Goodfellow, and S. Bengio, “Adversarial examples in the physical world,” in ICLR, 2017.

    Args:
        device {torch.device} -- CUDA或CPU
        model (torch.nn.Module): MNIST、CIFAR-10、ImageNet数据集的神经网络模型（基于PyTorch框架）
        sample (numpy.ndarray): 图像样本数据
        label (int): 图像样本真实标签
        is_imagenet (bool): 是否为ImageNet数据集
        args ([argparse.ArgumentParser, attacks.AttackParameters]): 攻击参数（扰动程度等）

    Returns:
        torch.tensor: BIM对抗样本
    """
#     tmp_eps = args.epsilon
    alpha = 0.005
    model.to(device)
    adversarial_example = copy.copy(sample)
    adversarial_example = Variable(torch.FloatTensor(adversarial_example).to(device), requires_grad=True)
    label = label.to(device)

    while tmp_eps > 0:
        output = model(adversarial_example)
        loss = F.nll_loss(F.log_softmax(output, 1), label)
        model.zero_grad()
        loss.backward()
        sign_data_grad = adversarial_example.grad.data.sign()

        with torch.no_grad():
            if is_imagenet:
                adversarial_example = handle_imagenet(image=adversarial_example, handler=0)
            
            adversarial_example = adversarial_example + min(tmp_eps, alpha) * sign_data_grad
            adversarial_example = torch.clamp(adversarial_example, 0, 1)
            
            if is_imagenet:
                adversarial_example = handle_imagenet(image=adversarial_example, handler=1)
            
            adversarial_example = Variable(torch.FloatTensor(adversarial_example.detach().cpu().numpy()).to(device), requires_grad=True)
        
        tmp_eps -= min(tmp_eps, alpha)
#         adversarial_example.grad.zero_()
    return adversarial_example

def pgd(device, model, sample, label, is_imagenet, epsilon):
    """Projected Gradient Descent（PGD）对抗攻击算法
    非定向迭代攻击
    论文：A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu, “Towards deep learning models resistant to adversarial attacks,” in ICLR, 2018.

    Args:
        device {torch.device} -- CUDA或CPU
        model (torch.nn.Module): MNIST、CIFAR-10、ImageNet数据集的神经网络模型（基于PyTorch框架）
        sample (numpy.ndarray): 图像样本数据
        label (int): 图像样本真实标签
        is_imagenet (bool): 是否为ImageNet数据集
        args ([argparse.ArgumentParser, attacks.AttackParameters]): 攻击参数（扰动程度等）

    Returns:
        torch.tensor: PGD对抗样本
    """
    epsilon_iter = 0.005
    num_steps = 50
    epsilon_temp = epsilon

    adversarial_example = np.copy(sample)
    # randomly chosen starting points inside the L_\inf ball around the
    if is_imagenet:
        adversarial_example = handle_imagenet(image=adversarial_example, handler=0)
        sample = handle_imagenet(image=sample, handler=0)

    adversarial_example = adversarial_example + np.random.uniform(-epsilon, epsilon, adversarial_example.shape).astype('float32')
    adversarial_example = np.clip(adversarial_example, 0, 1)

    if is_imagenet:
        adversarial_example = handle_imagenet(image=adversarial_example, handler=1)
        
    var_label = Variable(torch.LongTensor(label).to(device), requires_grad=False)

    if epsilon == 0.095: 
        epsilon = 0.115
    
    for index in range(num_steps):
        zero_gradients(adversarial_example)
        var_sample = Variable(torch.FloatTensor(adversarial_example).to(device), requires_grad=True)
        
        output = model(var_sample)
        loss_fun = torch.nn.CrossEntropyLoss()
        # loss = loss_fun(output, torch.max(var_label, 1)[1])
        loss = loss_fun(output, var_label)
        loss.backward()

        gradient_sign = var_sample.grad.data.cpu().sign().numpy()
        
        if is_imagenet:
            adversarial_example = handle_imagenet(image=adversarial_example, handler=0)
        
        adversarial_example = adversarial_example + epsilon_iter * gradient_sign
        adversarial_example = np.clip(adversarial_example, sample - epsilon, sample + epsilon)
        adversarial_example = np.clip(adversarial_example, 0, 1)
    
        if is_imagenet:
            adversarial_example = handle_imagenet(image=adversarial_example, handler=1)
    
    epsilon = epsilon_temp
        
    return torch.FloatTensor(adversarial_example).to(device)

def fgsm(device, model, sample, label, is_imagenet, epsilon):
    """Fast Gradient Sign Method（FGSM）对抗攻击算法
    非定向单步攻击
    论文：I.J. Goodfellow, J. Shlens, and C. Szegedy, “Explaining and harnessing adversarial examples,” in ICLR, 2015.

    Args:
        device {torch.device} -- CUDA或CPU
        model (torch.nn.Module): MNIST、CIFAR-10、ImageNet数据集的神经网络模型（基于PyTorch框架）
        sample (numpy.ndarray): 图像样本数据
        label (int): 图像样本真实标签
        is_imagenet (bool): 是否为ImageNet数据集
        args ([argparse.ArgumentParser, attacks.AttackParameters]): 攻击参数（扰动程度等）

    Returns:
        torch.tensor: FGSM对抗样本
    """
    var_sample = Variable(torch.FloatTensor(sample).to(device), requires_grad=True)
    var_label = Variable(torch.LongTensor(label).to(device))

    output = model(var_sample)
    loss = F.nll_loss(F.log_softmax(output, 1), var_label)
    model.zero_grad()
    loss.backward()
    sign_data_grad = var_sample.grad.data.sign()
    
    if is_imagenet:
        var_sample = handle_imagenet(image=var_sample, handler=0) 

    adversarial_example = var_sample + epsilon * sign_data_grad
    adversarial_example = torch.clamp(adversarial_example, 0, 1)

    if is_imagenet:
        adversarial_example = handle_imagenet(image=adversarial_example, handler=1) 
    
    return adversarial_example

def deepfool(device, model, sample, is_imagenet, epsilon):
    """DeepFool对抗攻击算法
    非定向迭代攻击
    论文：S.-M. Moosavi-Dezfooli, A. Fawzi, and P. Frossard, “Deepfool: A simple and accurate method to fool deep neural networks,” in CVPR, 2016.

    Args:
        device {torch.device} -- CUDA或CPU
        model (torch.nn.Module): MNIST、CIFAR-10、ImageNet数据集的神经网络模型（基于PyTorch框架）
        sample (numpy.ndarray): 图像样本数据
        label (int): 图像样本真实标签
        is_imagenet (bool): 是否为ImageNet数据集
        args ([argparse.ArgumentParser, attacks.AttackParameters]): 攻击参数（扰动程度等）

    Returns:
        torch.tensor: DeepFool对抗样本
    """
    overshoot = 0.02
    max_iter = 50
    if is_imagenet:
        num_classes = 200
    else:
        num_classes = 10
    epsilon_temp = epsilon
    
    sample = torch.FloatTensor(sample).to(device)
    var_sample = Variable(sample, requires_grad=True)
    prediction = model(var_sample)
    original = torch.max(prediction, 1)[1]
    current = original
    
    if is_imagenet:
        sample = handle_imagenet(image=sample, handler=0)
    
    if epsilon == 0.095: 
        epsilon = 0.6
    
    # indices of predication in descending order
    I = np.argsort(prediction.data.cpu().numpy() * -1)
    perturbation_r_tot = np.zeros(sample.shape, dtype=np.float32)
    iteration = 0

    while 0==(original == current).sum() and iteration < max_iter:
        # predication for the adversarial example in i-th iteration
        zero_gradients(var_sample)
        f_kx = model(var_sample)
        current = torch.max(f_kx, 1)[1]

        # gradient of the original example
        f_kx[0, I[0, 0]].backward(retain_graph=True)
        grad_original = np.copy(var_sample.grad.data.cpu().numpy())

        # calculate the w_k and f_k for every class label
        closest_dist = 1e10
        for k in range(1, num_classes):
            # gradient of adversarial example for k-th label
            zero_gradients(var_sample)
            f_kx[0, I[0, k]].backward(retain_graph=True)
            grad_current = var_sample.grad.data.cpu().numpy().copy()
            # update w_k and f_k
            w_k = grad_current - grad_original
            f_k = (f_kx[0, I[0, k]] - f_kx[0, I[0, 0]]).detach().data.cpu().numpy()
            # find the closest distance and the corresponding w_k
            dist_k = np.abs(f_k) / (np.linalg.norm(w_k.flatten()) + 1e-10)
            if dist_k < closest_dist:
                closest_dist = dist_k
                closest_w = w_k

        # accumulation of perturbation
        r_i = (closest_dist + 1e-4) * closest_w / np.linalg.norm(closest_w)
        perturbation_r_tot = perturbation_r_tot + r_i

        var_perturbation = torch.FloatTensor(torch.from_numpy((1 + overshoot) * perturbation_r_tot)).to(device)
        
        tmp_sample = var_perturbation + sample
        tmp_sample = torch.clamp(tmp_sample, 0, 1)
            
        if is_imagenet:
            tmp_sample = handle_imagenet(image=tmp_sample, handler=1)
        
        var_sample = Variable(tmp_sample, requires_grad=True)
        iteration += 1

    var_perturbation = torch.FloatTensor(torch.from_numpy((1 + overshoot) * perturbation_r_tot)).to(device)
    var_perturbation = torch.clamp(var_perturbation, -epsilon, epsilon)
    
    adversarial_example = var_perturbation + sample
    adversarial_example = torch.clamp(adversarial_example, 0, 1)
        
    if is_imagenet:
        adversarial_example = handle_imagenet(image=adversarial_example, handler=1)
    
    epsilon = epsilon_temp
    
    return adversarial_example


def jsma(device, model, sample, label, is_imagenet, epsilon):
    """JSMA对抗攻击算法
    非定向单步攻击
    论文：Nicolas Papernot et al. “The Limitations of Deep Learning in Adversarial Settings” in S&P, 2016.

    Args:
        device {torch.device} -- CUDA或CPU
        model (torch.nn.Module): MNIST、CIFAR-10、ImageNet数据集的神经网络模型（基于PyTorch框架）
        sample (numpy.ndarray): 图像样本数据
        label (int): 图像样本真实标签
        is_imagenet (bool): 是否为ImageNet数据集
        args ([argparse.ArgumentParser, attacks.AttackParameters]): 攻击参数（扰动程度等）

    Returns:
        torch.tensor: FGSM对抗样本
    """
    var_sample = Variable(torch.FloatTensor(sample).to(device), requires_grad=True)
    var_label = Variable(torch.LongTensor(label).to(device))

    output = model(var_sample)
    loss = F.nll_loss(F.log_softmax(output, 1), var_label)
    model.zero_grad()
    loss.backward()
    sign_data_grad = var_sample.grad.data.sign()
    
    if is_imagenet:
        var_sample = handle_imagenet(image=var_sample, handler=0) 

    adversarial_example = var_sample + epsilon * sign_data_grad
    adversarial_example = torch.clamp(adversarial_example, 0, 1)

    if is_imagenet:
        adversarial_example = handle_imagenet(image=adversarial_example, handler=1) 
    
    return adversarial_example

def sample_gen(method,model,device,dataloader,arg,BATCH_SIZE=8):
    '''
    Parameters
    ------
    method: {'bim', 'fgsm', 'pgd','jsma', 'deepfool', 'GaussianBlur', 'brightness', 'contrast', 'saturation', }
    model: 待评估模型
    device: {'cuda','cpu'}
    dataloader: torch.utils.data.DataLoader
    arg: float
        different hyperparameter for different method.
    BATCH_SIZE: int
        batch size of dataloader

    Returns
    ------
    adv_list: list
        list of adversarial samples
    '''
#     model = torch.load(model_pkl_path)
    model = model.to(device)
    ads_list = []
    method_dict = {
        'bim':bim,
        'fgsm':fgsm,
        'pgd':pgd,
        'jsma':jsma,
    }
    for x_bat,y_bat in dataloader:
        sample=x_bat
        label=y_bat

        if method == 'GaussianBlur':
            nature_transforms=transforms.GaussianBlur(3, sigma=arg)
            ads = nature_transforms(sample)
        elif method == 'brightness':
            nature_transforms = transforms.ColorJitter(brightness=arg, contrast=0, saturation=0, hue=0)
            ads = nature_transforms(sample)
        elif method == 'contrast':
            nature_transforms = transforms.ColorJitter(brightness=0, contrast=arg, saturation=0, hue=0)
            ads = nature_transforms(sample)
        elif method == 'saturation':
            nature_transforms = transforms.ColorJitter(brightness=0, contrast=0, saturation=arg, hue=0)
            ads = nature_transforms(sample)
        elif method == 'deepfool':
            ads = deepfool(device,model,sample,False,arg)
        else:
            ads = method_dict[method](device,model,sample,label,False,arg)

        ads = ads.to(device)
        m = nn.Softmax(dim=0)
        conf = m(model(ads))
        ads = ads.cpu()
        
        ads_list.append(ads)   

    return ads_list


def draw_fig(model,train_loader,device):
    BATCH_SIZE=8
    #     model = torch.load(model_pkl_path)
    model = model.to(device)

    for x_bat,y_bat in train_loader:
        sample=x_bat
        label=y_bat
        break
        
    ads_list = []    
    nature_transforms=transforms.GaussianBlur(3, sigma=1)
    ads_list.append(nature_transforms(sample))
    nature_transforms = transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0)
    ads_list.append(nature_transforms(sample))
    nature_transforms = transforms.ColorJitter(brightness=0, contrast=2, saturation=0, hue=0)
    ads_list.append(nature_transforms(sample))
    nature_transforms = transforms.ColorJitter(brightness=0, contrast=0, saturation=2, hue=0)
    ads_list.append(nature_transforms(sample))

    # ads_list.append(jsma(device,model,sample,label,False,0.1))
    ads_list.append(deepfool(device,model,sample,False,0.5))
    ads_list.append(bim(device,model,sample,label,False,1))
    ads_list.append(fgsm(device,model,sample,label,False,0.5))
    ads_list.append(pgd(device,model,sample,label,False,0.5))

    ads_list_new = []
    for ads in ads_list:
        ads = ads.to(device)
        m = nn.Softmax(dim=0)
        conf = m(model(ads))
        ads = ads.cpu()
        ads_list_new.append(ads)

    myfont = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 20,
        }
    fig, ax = plt.subplots(len(ads_list_new)+1, BATCH_SIZE,sharex='col',sharey='row',figsize=(20,30))
    for i in range(BATCH_SIZE):
        ax[0,i].imshow(sample[i].transpose(2,0).detach().numpy())
    #         ax[0,i].set_title('label:{}'.format(label[i]),myfont)
        for j in range(len(ads_list_new)):
            ax[j+1,i].imshow(ads_list_new[j][i].transpose(2,0).detach().numpy())
    #         ax[1,i].set_title('label:{}'.format(ads_label[i]),myfont)
    fig.tight_layout()
    plt.savefig(osp.join(current_dir,'sample.png'))

def draw_fig_choose(model,train_loader, naturemethod, nature_args, adv_method, adv_args, out_path, device):
    BATCH_SIZE=8
    #     model = torch.load(model_pkl_path)
    model = model.to(device)

    for x_bat,y_bat in train_loader:
        sample=x_bat
        label=y_bat
        break
    
    ads_list = []    
    if naturemethod == "GaussianBlur":
        nature_transforms=transforms.GaussianBlur(3, sigma=nature_args)
        ads_list.append(nature_transforms(sample))
    elif naturemethod == "brightness":
        nature_transforms = transforms.ColorJitter(brightness=nature_args, contrast=0, saturation=0, hue=0)
        ads_list.append(nature_transforms(sample))
    elif naturemethod == "contrast":
        nature_transforms = transforms.ColorJitter(brightness=0, contrast=nature_args, saturation=0, hue=0)
        ads_list.append(nature_transforms(sample))
    elif naturemethod == "saturation":
        nature_transforms = transforms.ColorJitter(brightness=0, contrast=0, saturation=nature_args, hue=0)
        ads_list.append(nature_transforms(sample))  
        
    if adv_method == "deepfool":
        ads_list.append(deepfool(device,model,sample,False,adv_args))
    elif adv_method == "bim":
        ads_list.append(bim(device,model,sample,label,False,adv_args))
    elif adv_method == "fgsm":
        ads_list.append(fgsm(device,model,sample,label,False,adv_args))
    elif adv_method == "pgd":
        ads_list.append(pgd(device,model,sample,label,False,adv_args))

    ads_list_new = []
    for ads in ads_list:
        ads = ads.to(device)
        m = nn.Softmax(dim=0)
        conf = m(model(ads))
        ads = ads.cpu()
        ads_list_new.append(ads)

    myfont = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 20,
        }
    fig, ax = plt.subplots(len(ads_list_new)+1, BATCH_SIZE,sharex='col',sharey='row',figsize=(20,10))
    for i in range(BATCH_SIZE):
        ax[0,i].imshow(sample[i].transpose(2,0).detach().numpy())
    #         ax[0,i].set_title('label:{}'.format(label[i]),myfont)
        for j in range(len(ads_list_new)):
            ax[j+1,i].imshow(ads_list_new[j][i].transpose(2,0).detach().numpy())
    #         ax[1,i].set_title('label:{}'.format(ads_label[i]),myfont)
    fig.tight_layout()
    import os
    if not osp.exists(out_path):
        os.mkdir(out_path)
    plt.savefig(osp.join(out_path,'sample.png'))


def safety_metric(method,model,device,dataloader,arg,BATCH_SIZE=128):
    model = model.to(device)
    method_dict = {
        'bim':bim,
        'fgsm':fgsm,
        'pgd':pgd
    }
    wrong_label = 0
    for x_bat,y_bat in dataloader:
        sample=x_bat
        label=y_bat 

        if method == 'deepfool':
            ads = deepfool(device,model,sample,False,arg)
        else:
            ads = method_dict[method](device,model,sample,label,False,arg)
        ads = ads.to(device)
        m = nn.Softmax(dim=0)
        conf = m(model(ads))
        ads = ads.cpu()

        ads_out=torch.max(conf,dim=1)
        ads_label=ads_out.indices.cpu().detach().numpy()
        wrong_label += (ads_label!=label.numpy()).sum()
    misclassify_rate = (wrong_label/len(dataloader.dataset))*100
    return 100-misclassify_rate

def robustness_metric(method,model,device,dataloader,arg,BATCH_SIZE=128):
    model = model.to(device)
    wrong_label = 0
    for x_bat,y_bat in dataloader:
        sample=x_bat
        label=y_bat     
        if method == 'GaussianBlur':
            nature_transforms=transforms.GaussianBlur(3, sigma=arg)
            ads = nature_transforms(sample)
        elif method == 'brightness':
            nature_transforms = transforms.ColorJitter(brightness=arg, contrast=0, saturation=0, hue=0)
            ads = nature_transforms(sample)
        elif method == 'contrast':
            nature_transforms = transforms.ColorJitter(brightness=0, contrast=arg, saturation=0, hue=0)
            ads = nature_transforms(sample)
        elif method == 'saturation':
            nature_transforms = transforms.ColorJitter(brightness=0, contrast=0, saturation=arg, hue=0)
            ads = nature_transforms(sample)
        ads = ads.to(device)
        m = nn.Softmax(dim=0)
        conf = m(model(ads))
        ads = ads.cpu()

        ads_out=torch.max(conf,dim=1)
        ads_label=ads_out.indices.cpu().detach().numpy()
        wrong_label += (ads_label!=label.numpy()).sum()
    misclassify_rate = (wrong_label/len(dataloader.dataset))*100
    return 100-misclassify_rate


def generalization_metric(model,device,dataloader,mapping=None,BATCH_SIZE=128):
    '''
    Parameters
    ------
    method: {'bim', 'fgsm', 'pgd','jsma', 'deepfool', 'GaussianBlur', 'brightness', 'contrast', 'saturation', }
    model: 待评估模型
    device: {'cuda','cpu'}
    dataloader: torch.utils.data.DataLoader
    mapping: dict
        标签映射字典
    BATCH_SIZE: int
        batch size of dataloader

    Returns
    ------
    adv_list: list
        list of adversarial samples
    '''
    model = model.to(device)
    wrong_label = 0
    for x_bat,y_bat in dataloader:
        sample = x_bat
        if mapping != None:
            label = np.array(list(map(mapping.get,y_bat.numpy())))
        else:
            label = y_bat.numpy()
        
        
        sample = sample.to(device)
        
        m = nn.Softmax(dim=0)
        conf = m(model(sample))
        
        out=torch.max(conf,dim=1)
        predict_label=out.indices.cpu().detach().numpy()
        wrong_label += (predict_label!=label).sum()
    misclassify_rate = (wrong_label/len(dataloader.dataset))*100
    return 100-misclassify_rate



def run(params, logging=None):
    BATCH_SIZE = 32
    logging.info("模型加载中......")
    VGG16 = torchvision.models.vgg16(pretrained=True)
    resnet18 = models.resnet18(pretrained=True)
    resnet50 = models.wide_resnet50_2(pretrained=True)
    alexnet = models.alexnet(pretrained=True)
    inception = models.inception_v3(pretrained=True)
    model_dict = {
        'vgg16':VGG16,
        'resnet18':resnet18,
        'resnet50':resnet50,
        'alexnet':alexnet,
        'inception':inception
    }
    
    model = model_dict[params["model"]]
    logging.info("模型加载完成......")
    transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.ToTensor()
            ])
    cifar10_traindata = datasets.CIFAR10(root='./dataset/data/',train=True,download=True,transform=transform)
    cifar10_testdata = datasets.CIFAR10(root='./dataset/data/',train=False,download=True,transform=transform)
    train_loader = DataLoader(cifar10_traindata, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(cifar10_testdata, batch_size=BATCH_SIZE)

    stl10_data = datasets.STL10(root='./dataset/data/',split='train',download=True,transform=transform)
    generalize_loader = DataLoader(stl10_data, batch_size=BATCH_SIZE, shuffle=True)
    # draw_fig(model,train_loader,params["device"])
    draw_fig_choose(model, train_loader, params["nature"], params["nature_arg"], params["adversarial"], params["adversarial_arg"], params["out_path"], params["device"])

    for i in params["measuremthod"]:
        if i == "safety":
            logging.info("模型安全性度量中......")
            output_dict["safety"] = safety_metric(params["adversarial"],model,params["device"],test_loader,params["adversarial_arg"])
            logging.info("模型安全性度量完成......")
        elif i == "robustness":
            logging.info("模型鲁棒性度量中......")
            output_dict["robustness"] = robustness_metric(params["nature"],model,params["device"],test_loader,params["nature_arg"])
            logging.info("模型鲁棒性度量完成......")
        elif i == "generalization":
            logging.info("模型泛化性度量中......")
            output_dict["generalization"] = generalization_metric(model,params["device"],generalize_loader)
            logging.info("模型泛化性度量完成......")
    output_dict["result"] = str(osp.join(params["out_path"],'sample.png'))

    return output_dict
    # json_path = osp.join(params["out_path"], "output.json")
    # with open(json_path, 'w') as f:
    #     json.dump(output_dict, f)   
# print(safety_metric('fgsm',resnet18,device,test_loader,0.1))
# print(robustness_metric('brightness',resnet18,device,test_loader,1))
# print(generalization_metric(resnet18,device,generalize_loader))
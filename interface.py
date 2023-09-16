import os
import torch
from torchvision.models import vgg16
from function.defense.jpeg import Jpeg
from function.defense.twis import Twis
from function.defense.region_based import RegionBased
from function.defense.pixel_deflection import Pixel_Deflection
from function.defense.feature_squeeze import feature_squeeze
from function.defense.preprocessor.preprocessor import *
from function.defense.trainer.trainer import *
from function.defense.detector.poison.detect_poison import *
from function.defense.transformer.poisoning.transformer_poison import *
from function.defense.sage.sage import *
from function.defense.models import *

def detect(adv_dataset, adv_model, adv_method, adv_nums, defense_methods, adv_examples=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
    if torch.cuda.is_available():
        print("got GPU")
    if adv_dataset == 'CIFAR10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        if adv_model == 'ResNet18':
            model = ResNet18()
            checkpoint = torch.load('/mnt/data2/yxl/AI-platform/model/model-cifar-resnet18/model-res-epoch85.pt')
        elif adv_model == 'VGG16':
            model = vgg16()
            model.classifier[6] = nn.Linear(4096, 10)
            checkpoint = torch.load('/mnt/data2/yxl/AI-platform/model/model-cifar-vgg16/model-vgg16-epoch85.pt')
        else:
            raise Exception('CIFAR10 can only use ResNet18 and VGG16!')
        model.load_state_dict(checkpoint)
        model = model.to(device)
    elif adv_dataset == 'MNIST':
        mean = (0.1307,)
        std = (0.3081,)
        if adv_model == 'SmallCNN':
            model = SmallCNN()
            checkpoint = torch.load('/mnt/data2/yxl/AI-platform/model/model-mnist-smallCNN/model-nn-epoch61.pt')
        elif adv_model == 'VGG16':
            model = vgg16()
            model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
            model.classifier[6] = nn.Linear(4096, 10)
            checkpoint = torch.load('/mnt/data2/yxl/AI-platform/model/model-mnist-vgg16/model-vgg16-epoch32.pt')
        else:
            raise Exception('MNIST can only use SmallCNN and VGG16!')
        model.load_state_dict(checkpoint)
        model = model.to(device).eval()    

    if defense_methods not in ['Pixel Defend', 'Pixel Defend Enhanced'] and adv_method == 'BPDA':
        raise Exception('BPDA can only use to attack Pixel Defend and Pixel Defend Enhanced!')
    if defense_methods == 'JPEG':
        detector =  Jpeg(model, mean, std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'Feature Squeeze':
        detector =  feature_squeeze(model, mean, std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'Twis':
        detector =  Twis(model, mean, std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'Rgioned-based':
        detector =  RegionBased(model, mean, std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'Pixel Deflection':
        detector =  Pixel_Deflection(model, mean, std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'Label Smoothing':
        detector =  Label_smoothing(model, mean, std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'Spatial Smoothing':
        detector =  Spatial_smoothing(model, mean, std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'Gaussian Data Augmentation':
        detector =  Gaussian_augmentation(model, mean, std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'Total Variance Minimization':
        detector =  Total_var_min(model, mean, std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'Pixel Defend':
        detector =  Pixel_defend(model, mean, std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'Pixel Defend Enhanced':
        detector =  Pixel_defend_enhanced(model, mean, std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'InverseGAN':
        detector =  Inverse_gan(model, mean, std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'DefenseGAN':
        detector =  Defense_gan(model, mean, std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'Madry':
        detector =  Madry(model, mean, std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'FastAT':
        detector =  FastAT(model, mean, std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'TRADES':
        detector =  Trades(model, mean, std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'FreeAT':
        detector =  FreeAT(model, mean, std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'MART':
        detector =  Mart(model, mean, std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'CARTL':
        detector =  Cartl(model, mean, std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'Activation':
        detector =  Activation_defence(model, mean, std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'Spectral Signature':
        detector =  Spectral_signature(model, mean, std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'Provenance':
        detector =  Provenance_defense(model, mean, std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'Neural Cleanse L1':
        detector =  Neural_cleanse_l1(model, mean, std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'Neural Cleanse L2':
        detector =  Neural_cleanse_l2(model, mean, std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'Neural Cleanse Linf':
        detector =  Neural_cleanse_linf(model, mean, std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'SAGE':
        detector =  Sage(model, mean, std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)

    _, _, detect_rate, no_defense_accuracy = detector.detect()
    return detect_rate, no_defense_accuracy
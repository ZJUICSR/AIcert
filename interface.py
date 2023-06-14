import os
import torch
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

def detect(adv_dataset, adv_method, adv_nums, defense_methods):
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
    if torch.cuda.is_available():
        print("got GPU")
    if adv_dataset == 'CIFAR10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        model = ResNet18()
        checkpoint = torch.load('/mnt/data2/yxl/AI-platform/model/model-cifar-wideResNet/model-wideres-epoch85.pt')
        model.load_state_dict(checkpoint)
        model = model.to(device)
    elif adv_dataset == 'MNIST':
        mean = (0.1307,)
        std = (0.3081,)
        model = SmallCNN()
        checkpoint = torch.load('/mnt/data2/yxl/AI-platform/model/model-mnist-smallCNN/model-nn-epoch61.pt')
        model.load_state_dict(checkpoint)
        model = model.to(device).eval()    

    if defense_methods == 'JPEG':
        detector =  Jpeg(model, mean, std, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'Feature Squeeze':
        detector =  feature_squeeze(model, mean, std, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'Twis':
        detector =  Twis(model, mean, std, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'Rgioned-based':
        detector =  RegionBased(model, mean, std, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'Pixel Deflection':
        detector =  Pixel_Deflection(model, mean, std, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'Label Smoothing':
        detector =  Label_smoothing(model, mean, std, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'Spatial Smoothing':
        detector =  Spatial_smoothing(model, mean, std, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'Gaussian Data Augmentation':
        detector =  Gaussian_augmentation(model, mean, std, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'Total Variance Minimization':
        detector =  Total_var_min(model, mean, std, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'Pixel Defend':
        detector =  Pixel_defend(model, mean, std, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'InverseGAN':
        detector =  Inverse_gan(model, mean, std, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'DefenseGAN':
        detector =  Defense_gan(model, mean, std, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'Madry':
        detector =  Madry(model, mean, std, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'FastAT':
        detector =  FastAT(model, mean, std, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'TRADES':
        detector =  Trades(model, mean, std, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'FreeAT':
        detector =  FreeAT(model, mean, std, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'MART':
        detector =  Mart(model, mean, std, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'CARTL':
        detector =  Cartl(model, mean, std, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'SAGE':
        detector =  Sage(model, mean, std, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'Activation':
        detector =  Activation_defence(model, mean, std, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'Spectral Signature':
        detector =  Spectral_signature(model, mean, std, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'Provenance':
        detector =  Provenance_defense(model, mean, std, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'Neural Cleanse':
        detector =  Neural_cleanse(model, mean, std, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)

    _, _, detect_rate = detector.detect()
    return detect_rate
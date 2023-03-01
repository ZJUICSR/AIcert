import os
import torch
from control.defense.jpeg import Jpeg
from control.defense.twis import Twis
from control.defense.region_based import RegionBased
from control.defense.pixel_deflection import Pixel_Deflection
from control.defense.feature_squeeze import feature_squeeze
from control.defense.preprocessor.preprocessor import *
from control.defense.trainer.trainer import *
from control.defense.detector.poison.detect_poison import *
from control.defense.transformer.poisoning.transformer_poison import *
from control.defense.models import *

def detect(adv_dataset, adv_method, adv_nums, defense_methods):
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
    if torch.cuda.is_available():
        print("got GPU")
    if adv_dataset == 'CIFAR10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        model = ResNet18()
        checkpoint = torch.load('/mnt/data2/yxl/AI-platform/trades-clean/model-cifar-wideResNet/model-wideres-epoch85.pt')
        model.load_state_dict(checkpoint)
        model = model.to(device)
    elif adv_dataset == 'MNIST':
        mean = (0.1307,)
        std = (0.3081,)
        model = Net()
        checkpoint = torch.load('/mnt/data2/yxl/AI-platform/trades-clean/model-mnist-smallCNN/model-nn-epoch61.pt')
        model.load_state_dict(checkpoint)
        model = model.to(device).eval()    
    elif adv_dataset == 'CIFAR100':
        mean = [0.5071, 0.4865, 0.4409]
        std = [0.2673, 0.2564, 0.2762]
        model = resnet18(100)
        checkpoint = torch.load('/mnt/data/detect_project/AI-platform/control/defense/trained_models/cifar100.ckpt', map_location='cuda')
        model.load_state_dict(checkpoint)
        model = model.to(device)
    elif adv_dataset == 'Imagenette':
        mean = [0, 0, 0]
        std = [1, 1, 1]
        model = resnet18(10)
        checkpoint = torch.load('/mnt/data/detect_project/AI-platform/control/defense/trained_models/cifar100.ckpt', map_location='cuda')
        model.load_state_dict(checkpoint)
        model = model.to(device)

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
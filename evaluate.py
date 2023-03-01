import os
import torch
from control.defense.models import *
from control.defense.jpeg import Jpeg
from control.defense.twis import Twis
from control.defense.region_based import RegionBased
from control.defense.pixel_deflection import Pixel_Deflection
from control.defense.feature_squeeze import feature_squeeze
from control.defense.preprocessor.preprocessor import *
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if torch.cuda.is_available():
    print("got GPU")
adv_dataset_list = ['CIFAR10', 'MNIST']
defense_list = [feature_squeeze, Spatial_smoothing, Twis, Jpeg, Total_var_min, Pixel_Deflection, RegionBased]
adv_method_list = ['FGSM', 'PGD']
for adv_dataset in adv_dataset_list:
    print(adv_dataset)
    if adv_dataset == 'CIFAR10':
        mean=[0.4914, 0.4822, 0.4465]
        std=[0.2023, 0.1994, 0.2010]
        model = ResNet18()
        checkpoint = torch.load('/mnt/data2/yxl/AI-platform/trades-clean/model-cifar-wideResNet/model-wideres-epoch85.pt')
    elif adv_dataset == 'MNIST':
        mean = (0.1307,)
        std = (0.3081,)
        model = SmallCNN()
        checkpoint = torch.load('/mnt/data2/yxl/AI-platform/trades-clean/model-mnist-smallCNN/model-nn-epoch61.pt')
    model.load_state_dict(checkpoint)
    model = model.to(device)
    for defense in defense_list:
        print(defense.__name__)
        if defense.__name__ == 'Jpeg' and adv_dataset == 'MNIST':
            continue
        for adv_method in adv_method_list:
            print(adv_method)
            deflection = defense(model = model, mean = mean, std = std, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=10000, device=device)
            deflection.detect()
            deflection.print_res()

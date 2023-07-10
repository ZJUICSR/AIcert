import os
import torch
import sys
sys.path.append('../..')
from function.defense.models import *
from function.defense.jpeg import *
from function.defense.feature_squeeze import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if torch.cuda.is_available():
    print("got GPU")

adv_dataset = 'CIFAR10'
adv_method = 'PGD'
if adv_dataset == 'CIFAR10':
    model = ResNet18()
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    # # checkpoint = torch.load('/mnt/data/yxl/AI-platform/trades/model-cifar-wideResNet/model-wideres-epoch91.pt')
    checkpoint = torch.load('/mnt/data2/yxl/AI-platform/model/model-cifar-resnet18/model-res-epoch85.pt') 
elif adv_dataset == 'MNIST':
    model = SmallCNN()
    mean = (0.1307,)
    std = (0.3081,)
    # checkpoint = torch.load('/mnt/data/yxl/AI-platform/trades/model-mnist-smallCNN/model-nn-epoch82.pt')
    checkpoint = torch.load('/mnt/data2/yxl/AI-platform/model/model-mnist-smallCNN/model-nn-epoch61.pt')
model.load_state_dict(checkpoint)
model = model.to(device)
deflection = feature_squeeze(model = model, adv_nums=100, mean = mean, std = std, adv_method=adv_method, adv_dataset=adv_dataset, device=device)#, adv_examples='/mnt/data2/yxl/AI-platform/dataset/adv_CIFAR10_FGSM.pt')
deflection.detect()
# print(deflection.no_defense_accuracy)
deflection.print_res()

import os
import torch
import sys
from trainer import *
sys.path.append("..")
from resnet import ResNet18
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if torch.cuda.is_available():
    print("got GPU")

model = ResNet18()
checkpoint = torch.load('../ckpt-cifar10-32-resnet18_0.9513.pth', map_location='cuda')
model.load_state_dict(checkpoint['net'])
model = model.to(device)   
print(next(model.parameters()).device)
deflection = FastAT(model, mean, std, adv_method='PGD', adv_dataset='CIFAR10', adv_nums=100, device=device)
deflection.detect()
deflection.print_res()

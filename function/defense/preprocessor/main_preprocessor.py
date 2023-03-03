import os
import torch
from function.defense.models.resnet import ResNet18
from function.defense.preprocessor.preprocessor import *
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if torch.cuda.is_available():
    print("got GPU")

model = ResNet18()
checkpoint = torch.load('/mnt/data2/yxl/AI-platform/function/defense/trained_models/ckpt-cifar10-32-resnet18_0.9513.pth', map_location='cuda')
model.load_state_dict(checkpoint['net'])
model = model.to(device)   
print(next(model.parameters()).device)
deflection = Pixel_defend(model = model, mean = mean, std = std, adv_method='PGD', adv_dataset='CIFAR10', adv_nums=1000, device=device)
deflection.detect()
deflection.print_res()

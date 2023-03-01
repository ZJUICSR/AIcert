import os
import torch
from control.defense.models import *
from control.defense.preprocessor.preprocessor import *
from control.defense.utils.generate_aes import generate_adv_examples
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if torch.cuda.is_available():
    print("got GPU")

# model = ResNet18()
# checkpoint = torch.load('/mnt/data2/yxl/AI-platform/trades/model-cifar-wideResNet/model-wideres-epoch91.pt')
# # checkpoint = torch.load('/mnt/data2/yxl/AI-platform/trades-clean/model-cifar-wideResNet/model-wideres-epoch85.pt')
# model.load_state_dict(checkpoint)
# model = model.to(device)   
mean = (0.1307,)
std = (0.3081,)
model = SmallCNN()
# checkpoint = torch.load('/mnt/data2/yxl/AI-platform/trades/model-mnist-smallCNN/model-nn-epoch82.pt')
checkpoint = torch.load('/mnt/data2/yxl/AI-platform/trades-clean/model-mnist-smallCNN/model-nn-epoch61.pt')
model.load_state_dict(checkpoint)
model = model.to(device)
deflection = Pixel_defend(model = model, mean = mean, std = std, adv_method='FGSM', adv_dataset='MNIST', adv_nums=10000, device=device)
deflection.detect()
deflection.print_res()

import os
import torch
from detect_poison import *

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
deflection = Provenance_defense(model = None, mean = mean, std = std, adv_method='FGSM', adv_dataset='CIFAR10', adv_nums=1000, device=device)
deflection.detect()
print(deflection.no_defense_accuracy)
deflection.print_res()

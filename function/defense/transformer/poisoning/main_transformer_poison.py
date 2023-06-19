import os
import torch
from transformer_poison import *

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if torch.cuda.is_available():
    print("got GPU")
deflection = Neural_cleanse_l1(model = None, mean = mean, std = std, adv_method='PGD', adv_dataset='MNIST', adv_nums=1000, device=device)
deflection.detect()
print(deflection.no_defense_accuracy)
deflection.print_res()

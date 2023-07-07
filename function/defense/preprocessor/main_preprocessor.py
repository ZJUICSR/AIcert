import os
import torch
import sys
sys.path.append('../../..')
from function.defense.models import *
from function.defense.preprocessor.preprocessor import *
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if torch.cuda.is_available():
    print("got GPU")

adv_dataset = 'CIFAR10'
adv_method = 'FGSM'
if adv_dataset == 'CIFAR10':
    model = ResNet18()
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    # # checkpoint = torch.load('/mnt/data/yxl/AI-platform/trades/model-cifar-wideResNet/model-wideres-epoch91.pt')
<<<<<<< HEAD
    checkpoint = torch.load('./model/model-cifar-resnet18/model-res-epoch85.pt') 
=======
    checkpoint = torch.load('/mnt/data2/yxl/AI-platform/model/model-cifar-resnet18/model-res-epoch85.pt') 
>>>>>>> c20e6977dccd1931b9978fef485a78c65af4c528
elif adv_dataset == 'MNIST':
    model = SmallCNN()
    mean = 0.1307
    std = 0.3081
    # checkpoint = torch.load('/mnt/data/yxl/AI-platform/trades/model-mnist-smallCNN/model-nn-epoch82.pt')
<<<<<<< HEAD
    checkpoint = torch.load('./model/model-mnist-smallCNN/model-nn-epoch61.pt')
=======
    checkpoint = torch.load('/mnt/data2/yxl/AI-platform/model/model-mnist-smallCNN/model-nn-epoch61.pt')
>>>>>>> c20e6977dccd1931b9978fef485a78c65af4c528
model.load_state_dict(checkpoint)
model = model.to(device)
deflection = Defense_gan(model = model, mean = mean, std = std, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=100, device=device)#, adv_examples='data/' + adv_dataset + '/adv_' + adv_dataset + '_' + adv_method + '.npy')
deflection.detect()
deflection.print_res()

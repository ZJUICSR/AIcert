import os
import torch
import sys
sys.path.append('../../..')
from function.defense.models import *
from function.defense.preprocessor.preprocessor import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if torch.cuda.is_available():
    print("got GPU")

adv_dataset = 'CIFAR10'
adv_method = 'PGD'
adv_model = 'VGG16'
if adv_dataset == 'CIFAR10':
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    checkpoint = torch.load('./model/model-cifar-resnet18/model-res-epoch85.pt') 

elif adv_dataset == 'MNIST':
    model = SmallCNN()
    mean = 0.1307
    std = 0.3081
    # checkpoint = torch.load('/mnt/data/yxl/AI-platform/trades/model-mnist-smallCNN/model-nn-epoch82.pt')
    checkpoint = torch.load('./model/model-mnist-smallCNN/model-nn-epoch61.pt')
    if adv_model == 'ResNet18':
        model = ResNet18()
        checkpoint = torch.load('./model/model-cifar-resnet18/model-res-epoch85.pt')
    elif adv_model == 'VGG16':
        model = vgg16()
        model.classifier[6] = nn.Linear(4096, 10)
        checkpoint = torch.load('./model/model-cifar-vgg16/model-vgg16-epoch85.pt')
    else:
        raise Exception('CIFAR10 can only use ResNet18 and VGG16!')
    model.load_state_dict(checkpoint)
    model = model.to(device)
elif adv_dataset == 'MNIST':
    mean = (0.1307,)
    std = (0.3081,)
    if adv_model == 'SmallCNN':
        model = SmallCNN()
        checkpoint = torch.load('./model/model-mnist-smallCNN/model-nn-epoch61.pt')
    elif adv_model == 'VGG16':
        model = vgg16()
        model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        model.classifier[6] = nn.Linear(4096, 10)
        checkpoint = torch.load('./model/model-mnist-vgg16/model-vgg16-epoch32.pt')
    else:
        raise Exception('MNIST can only use SmallCNN and VGG16!')
    model.load_state_dict(checkpoint)
    model = model.to(device).eval()  
model.load_state_dict(checkpoint)
model = model.to(device)
deflection = Defense_gan(model = model, adv_nums=100, mean = mean, std = std, adv_method=adv_method, adv_dataset=adv_dataset, device=device, adv_examples=None)#, adv_examples='data/' + adv_dataset + '/adv_' + adv_dataset + '_' + adv_method + '.npy')
deflection.detect()
deflection.print_res()

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets
# from .imagenette import ImageNette
import torchvision.transforms as transforms

def get_clean_loader(dataset):
    if dataset == 'CIFAR10':
        num_classes = 10
        tf_test = transforms.Compose([transforms.ToTensor(),
                                #   transforms.Normalize(
                                #     mean=[0.4914, 0.4822, 0.4465],
                                #     std=[0.2023, 0.1994, 0.2010],)
                                ])
        cleanset = datasets.CIFAR10(
            root='/mnt/data2/yxl/AI-platform/dataset/CIFAR10', train=False, download=True, transform=tf_test)
        clean_loader = DataLoader(dataset=cleanset,
                                   batch_size=128,
                                   shuffle=False,
                                   )
    elif dataset == 'MNIST':
        num_classes = 10
        tf_test = transforms.Compose([
                    transforms.ToTensor(),
                    # transforms.Normalize((0.1307, ), (0.3081, ))
            ])
        cleanset = datasets.MNIST(
            root = '/mnt/data2/yxl/AI-platform/data', train=False, download=True, transform=tf_test
        )
        clean_loader = DataLoader(dataset=cleanset,
                                   batch_size=128,
                                   shuffle=False,
                                   )
    
    return clean_loader, num_classes
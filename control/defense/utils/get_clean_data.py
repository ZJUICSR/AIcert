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
            root='/mnt/data2/yxl/AI-platform/data/CIFAR10', train=False, download=True, transform=tf_test)
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
    elif dataset == 'CIFAR100':
        num_classes = 100
        tf_test = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize(
                                    mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343], std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
                                )])
        cleanset = datasets.CIFAR100(
            root='/mnt/data2/yxl/AI-platform/data/CIFAR100', train=False, download=True, transform=tf_test)
        clean_loader = DataLoader(dataset=cleanset,
                                   batch_size=128,
                                   shuffle=False,
                                   )
    # elif dataset == 'ImageNette':
    #     num_classes = 10
    #     tf_test = transforms.Compose([
    #         transforms.Resize((224, 224)),
    #         transforms.ToTensor(),
    #     ])
    #     cleanset = ImageNette(False, transform=None)
    #     clean_data = DatasetCL(nums_needed=nums_needed, full_dataset=cleanset, transform=tf_test)
    #     clean_loader = DataLoader(dataset=clean_data,
    #                                batch_size=32,
    #                                shuffle=False,
    #                                )
    return clean_loader, num_classes
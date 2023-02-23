import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms


class AddImgTrigger(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, img):
        return self.add_backdoor(img, **self.kwargs)

    def add_backdoor(self, img, **kwargs):
        trigger_loc = kwargs["loc"]
        trigger_ptn = kwargs["val"]
        for i, (m, n) in enumerate(trigger_loc):
            img[:, m, n].fill_(trigger_ptn[i])
        return img

def get_bounds(dataset):
    mean, std = get_mean_std(dataset)
    bounds = (-1. * mean[0]) * std[0], (1. - mean[0]) * std[0]
    return bounds


def get_size(dataset):
    dname = dataset.lower()
    if dname == "mnist":
        size = (28, 28)
    elif "cifar" in dname:
        size = (32, 32)
    else:
        size = (299, 299)
    return size


def get_mean_std(dataset):
    dname = dataset.lower()
    if "mnist" in dname:
        mean = (0.1307,)
        std = (0.3081,)
    elif dname == "svhn":
        mean = (0.43768206, 0.44376972, 0.47280434)
        std = (0.19803014, 0.20101564, 0.19703615)
    elif "cifar" in dname:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif dname == "imagenet1k":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif dname == "cubs200":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        raise ValueError(f"{dataset} can't be found!")
    bounds = (-1. * mean[0]) * std[0], (1. - mean[0]) * std[0]
    return mean, std, bounds


def get_transforms(dataset, size=(32, 32)):
    dname = dataset.lower()
    mean, std, bounds = get_mean_std(dataset)
    if "mnist" in dname:
        size = (28, 28)
        t_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        t_test = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif dname == "svhn":
        t_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        t_test = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    elif "cifar" in dname:
        t_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(size, padding=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        t_test = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    elif dname == "imagenet1k":
        t_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        t_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    elif dname == "cubs200":
        t_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        t_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        raise ValueError(f"{dataset} can't be found!")
    return t_train, t_test, bounds


def set_seed(seed=100):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_data(dataset, mean, std, path):
    from torchvision.utils import save_image
    data_loader = DataLoader(dataset, batch_size=40)
    batch_data, targets = next(iter(data_loader))

    for i, (m, s) in enumerate(zip(mean, std)):
        batch_data[:, i, :, :] = batch_data[:, i, :, :] * s + m
    print(f"-> Saving backdoored image: {path}")
    save_image(batch_data, format="JPEG", fp=path)

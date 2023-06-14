from typing import Tuple
import os
import sys #
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
sys.path.append('../../..') #
from ....src import settings
from ..logging_utils import logger
from .dataset_utils import SubsetDataset, GTSRB

DATA_DIR = "/mnt/data2/yxl/AI-platform/dataset"

# default mean of cifar100
CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
# default std of cifar100
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
# default mean of cifar10
CIFAR10_TRAIN_MEAN = (0.49139765, 0.48215759, 0.44653141)
# default std of cifar10
CIFAR10_TRAIN_STD = (0.24703199, 0.24348481, 0.26158789)

MNIST_TRAIN_STD = (0.3081, 0.3081, 0.3081)
MNIST_TRAIN_MEAN = (0.1307, 0.1307, 0.1307)

SVHN_TRAIN_STD = (0.19803032, 0.20101574, 0.19703609)
SVHN_TRAIN_MEAN = (0.4376817, 0.4437706, 0.4728039)

GTSRB_TRAIN_MEAN = (0.3403, 0.3121, 0.3214)
GTSRB_TRAIN_STD = (0.2724, 0.2608, 0.2669)


def get_mean_and_std(dataset: str) -> Tuple[Tuple, Tuple]:
    if dataset == "cifar100":
        logger.debug(f"get mean and std of cifar100")
        mean = CIFAR100_TRAIN_MEAN
        std = CIFAR100_TRAIN_STD
    elif dataset == "cifar10":
        # logger.debug(f"get mean and std of cifar10")
        logger.warning("Using mean and std of cifar100 for dataset cifar10!")
        mean = CIFAR100_TRAIN_MEAN
        std = CIFAR100_TRAIN_STD
    elif dataset == "mnist":
        # logger.debug(f"get mean and std of mnist")
        logger.warning(f"Using mean and std of svhn for mnist")
        # mean = MNIST_TRAIN_MEAN
        mean = SVHN_TRAIN_MEAN
        # std = MNIST_TRAIN_STD
        std = SVHN_TRAIN_STD
    elif dataset == "svhn":
        logger.debug(f"get mean and std of svhn")
        mean = SVHN_TRAIN_MEAN
        std = SVHN_TRAIN_STD
        # logger.warning("Using mean and std of gtsrb!")
        # mean = GTSRB_TRAIN_MEAN
        # std = GTSRB_TRAIN_STD
    elif dataset == "svhntl":
        # logger.debug(f"get mean and std of cifar10")
        logger.warning("Using mean and std of cifar100 for dataset svhn!")
        mean = CIFAR100_TRAIN_MEAN
        std = CIFAR100_TRAIN_STD
    elif dataset == "gtsrb":
        logger.warning("Using mean and std as cifar10 for dataset gtsrb!")
        return get_mean_and_std("cifar10")
        # logger.warning("Using mean and std of gtsrb!")
        # mean = GTSRB_TRAIN_MEAN
        # std = GTSRB_TRAIN_STD
    else:
        raise ValueError(f'dataset "{dataset}" is not supported!')

    return mean, std


def get_subset_cifar_train_dataloader(partition_ratio: float, dataset=settings.dataset_name,
                                      batch_size=settings.batch_size, num_workers=settings.num_worker,
                                      shuffle=True, normalize=True):
    logger.info("load whole loader")
    whole_cifar_dataloader = get_cifar_train_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        # false to keep same
        shuffle=False,
        normalize=normalize
    )
    subset_dataset = SubsetDataset(whole_cifar_dataloader, partition_ratio)
    logger.info(f"subset size: {len(subset_dataset)}")
    return DataLoader(subset_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)


def get_cifar_train_dataloader(dataset=settings.dataset_name, batch_size=settings.batch_size,
                               num_workers=settings.num_worker, shuffle=True, normalize=True):
    if dataset == "cifar100":
        _data = torchvision.datasets.CIFAR100
        path = "CIFAR100"
        logger.info("load cifar100 train dataset")
    elif dataset == "cifar10":
        _data = torchvision.datasets.CIFAR10
        path = "CIFAR10"
        logger.info("load cifar10 train dataset")
    else:
        raise ValueError(f'dataset "{dataset}" is not supported!')

    mean, std = get_mean_and_std(dataset=dataset)
    logger.debug(f"dataset mean: {mean}, dataset std: {std}")

    compose_list = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ]
    if normalize:
        compose_list.append(transforms.Normalize(mean, std))
    transform_train = transforms.Compose(compose_list)

    train_dataset = _data(root=os.path.join(DATA_DIR, path), train=True, download=True,
                          transform=transform_train)

    train_loader = DataLoader(train_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return train_loader


def get_cifar_test_dataloader(dataset=settings.dataset_name, batch_size=settings.batch_size,
                              num_workers=settings.num_worker, shuffle=False, normalize=True):
    if dataset == "cifar100":
        _data = torchvision.datasets.CIFAR100
        path = "CIFAR100"
        logger.info("load cifar100 test dataset")
    elif dataset == "cifar10":
        _data = torchvision.datasets.CIFAR10
        path = "CIFAR10"
        logger.info("load cifar10 test dataset")
    else:
        raise ValueError(f'dataset "{dataset}" is not supported!')

    mean, std = get_mean_and_std(dataset=dataset)

    compose_list = [
        transforms.ToTensor(),
    ]
    if normalize:
        compose_list.append(transforms.Normalize(mean, std))
    transform_test = transforms.Compose(compose_list)
    test = _data(root=os.path.join(DATA_DIR, path), train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return test_loader


def get_mnist_train_dataloader(batch_size=settings.batch_size, num_workers=settings.num_worker,
                               shuffle=True, normalize=True):
    compose_list = [
        # resize original mnist size(28 * 28) to 32 * 32
        transforms.Resize(32),
        # 1 * 32 * 32 to 3 * 32 * 32
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ]
    if normalize:
        mean, std = get_mean_and_std("mnist")
        compose_list.append(transforms.Normalize(mean, std))
    transform = transforms.Compose(compose_list)

    train_data = torchvision.datasets.MNIST(root=os.path.join(DATA_DIR, "MNIST"), train=True,
                                            download=True, transform=transform)
    train_loader = DataLoader(train_data, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return train_loader


def get_mnist_test_dataloader(batch_size=settings.batch_size, num_workers=settings.num_worker,
                              shuffle=False, normalize=True):
    """3*32*32 mnist tensor"""
    compose_list = [
        transforms.Resize(32),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ]
    if normalize:
        mean, std = get_mean_and_std("mnist")
        compose_list.append(transforms.Normalize(mean, std))
    transform = transforms.Compose(compose_list)

    test_data = torchvision.datasets.MNIST(root=os.path.join(DATA_DIR, "MNIST"), train=False,
                                           download=True, transform=transform)
    test_loader = DataLoader(test_data, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return test_loader


def get_mnist_test_dataloader_one_channel(batch_size=settings.batch_size, num_workers=settings.num_worker,
                                          shuffle=False, normalize=True):
    """1*28*28 mnist tensor"""
    compose_list = [
        transforms.ToTensor(),
    ]
    if normalize:
        mean, std = (0.1307,), (0.3081,)
        compose_list.append(transforms.Normalize(mean, std))
    transform = transforms.Compose(compose_list)

    test_data = torchvision.datasets.MNIST(root=os.path.join(DATA_DIR, "MNIST"), train=False,
                                           download=True, transform=transform)
    test_loader = DataLoader(test_data, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return test_loader


def get_mnist_train_dataloader_one_channel(batch_size=settings.batch_size, num_workers=settings.num_worker,
                                           shuffle=True, normalize=True):
    compose_list = [
        transforms.ToTensor(),
    ]
    if normalize:
        mean, std = (0.1307,), (0.3081,)
        compose_list.append(transforms.Normalize(mean, std))
    transform = transforms.Compose(compose_list)

    train_data = torchvision.datasets.MNIST(root=os.path.join(DATA_DIR, "MNIST"), train=True,
                                            download=True, transform=transform)
    train_loader = DataLoader(train_data, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return train_loader


def get_svhn_train_dataloder(batch_size=settings.batch_size, num_workers=settings.num_worker,
                             shuffle=True, normalize=True, dataset_norm_type="svhn"):
    compose_list = [
        transforms.ToTensor(),
    ]
    if normalize:
        mean, std = get_mean_and_std(dataset_norm_type)
        compose_list.append(transforms.Normalize(mean, std))
    transform = transforms.Compose(compose_list)

    train_data = torchvision.datasets.SVHN(root=os.path.join(DATA_DIR, "SVHN"), split="train",
                                           download=True, transform=transform)
    train_loader = DataLoader(train_data, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return train_loader


def get_svhn_test_dataloader(batch_size=settings.batch_size, num_workers=settings.num_worker,
                             shuffle=False, normalize=True, dataset_norm_type="svhn"):
    compose_list = [
        transforms.ToTensor(),
    ]
    if normalize:
        mean, std = get_mean_and_std(dataset_norm_type)
        compose_list.append(transforms.Normalize(mean, std))
    transform = transforms.Compose(compose_list)

    test_data = torchvision.datasets.SVHN(root=os.path.join(DATA_DIR, "SVHN"), split="test",
                                          download=True, transform=transform)
    test_loader = DataLoader(test_data, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return test_loader


def get_gtsrb_train_dataloder(batch_size=settings.batch_size, num_workers=settings.num_worker,
                              shuffle=True, normalize=True):
    compose_list = [
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ]
    if normalize:
        mean, std = get_mean_and_std("gtsrb")
        compose_list.append(transforms.Normalize(mean, std))
    transform = transforms.Compose(compose_list)

    train_data = GTSRB(root=os.path.join(DATA_DIR, "GTSRB"),
                       train=True, transform=transform)
    train_loader = DataLoader(train_data, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return train_loader


def get_gtsrb_test_dataloder(batch_size=settings.batch_size, num_workers=settings.num_worker,
                             shuffle=False, normalize=True):
    compose_list = [
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ]
    if normalize:
        mean, std = get_mean_and_std("gtsrb")
        compose_list.append(transforms.Normalize(mean, std))
    transform = transforms.Compose(compose_list)

    test_data = GTSRB(root=os.path.join(DATA_DIR, "GTSRB"),
                      train=False, transform=transform)
    test_loader = DataLoader(test_data, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return test_loader

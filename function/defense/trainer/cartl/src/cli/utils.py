from ...src.utils.data_utils.get_dataloader import get_subset_cifar_train_dataloader
from typing import Optional

from torch.nn import Module
from torch.utils.data import DataLoader

from ...src.networks import (resnet18, resnet34, resnet50, wrn34_10, wrn28_10, wrn28_4, wrn34_4,
                          parseval_retrain_wrn28_10, parseval_retrain_wrn34_10, parseval_retrain_wrn28_4,
                          parseval_resnet18, SupportedAllModuleType)

from ...src.utils import (get_cifar_test_dataloader, get_cifar_train_dataloader,
                       get_mnist_test_dataloader, get_mnist_train_dataloader,
                       get_svhn_test_dataloader, get_svhn_train_dataloder,
                       get_gtsrb_test_dataloder, get_gtsrb_train_dataloder)


SupportNormalModelList = ['res18', 'res34', 'res50', 'wrn34', 'wrn34(4)', 'wrn28', 'wrn28(4)']
SupportParsevalModelList = ['pres18', 'pwrn34', 'pwrn28', 'pwrn28(4)']
SupportModelList = SupportNormalModelList + SupportParsevalModelList
DefaultModel = 'res18'


PartationDatasetList = ['cifar10(0.5)', 'cifar10(0.2)', 'cifar10(0.1)']
SupportDatasetList = ['cifar10', 'cifar100', 'mnist', 'svhn', 'svhntl', 'gtsrb'] + PartationDatasetList
DefaultDataset = 'mnist'


def get_model(model: str, num_classes: int, k: Optional[int] = None) -> SupportedAllModuleType:
    if model not in SupportModelList:
        raise ValueError("model not supported")
    if model == 'res18':
        return resnet18(num_classes=num_classes)
    elif model == "res34":
        return resnet34(num_classes=num_classes)
    elif model == "res50":
        return resnet50(num_classes=num_classes)
    elif model == 'pres18':
        return parseval_resnet18(k=k, num_classes=num_classes)
    elif model == 'wrn34':
        return wrn34_10(num_classes=num_classes)
    elif model == 'wrn34(4)':
        return wrn34_4(num_classes=num_classes)
    elif model == 'pwrn34':
        return parseval_retrain_wrn34_10(k=k, num_classes=num_classes)
    elif model == 'wrn28':
        return wrn28_10(num_classes=num_classes)
    elif model == 'wrn28(4)':
        return wrn28_4(num_classes=num_classes)
    elif model == 'pwrn28':
        return parseval_retrain_wrn28_10(k=k, num_classes=num_classes)
    elif model == 'pwrn28(4)':
        return parseval_retrain_wrn28_4(k=k, num_classes=num_classes)


def get_train_dataset(dataset: str) -> DataLoader:
    if dataset not in SupportDatasetList:
        raise ValueError("dataset not supported")
    if dataset.startswith("cifar"):
        if dataset in {'cifar10', 'cifar100'}:
            return get_cifar_train_dataloader(dataset=dataset)
        else: # very ugly hack
            import re
            [_, ds, ratio,  _] = re.split(r"(\w*)\(([\d\.]*)\)", dataset)
            return get_subset_cifar_train_dataloader(float(ratio), ds)
    elif dataset == 'mnist':
        return get_mnist_train_dataloader()
    elif dataset.startswith('svhn'):
        # 'svhn': using mean and std of 'svhn'
        # 'svhn': using mean and std of 'cifar100'
        return get_svhn_train_dataloder(dataset_norm_type=dataset)
    elif dataset == "gtsrb":
        return get_gtsrb_train_dataloder()
    else:
        raise ValueError(f"dataset `{dataset} is not supported`")


def get_test_dataset(dataset: str) -> DataLoader:
    if dataset not in SupportDatasetList:
        raise ValueError("dataset not supported")
    if dataset.startswith("cifar"):
        import re # very ugly hack
        [_, ds, _] = re.split("(cifar[\d]+).*", dataset)
        return get_cifar_test_dataloader(dataset=ds)
    elif dataset == 'mnist':
        return get_mnist_test_dataloader()
    elif dataset.startswith('svhn'):
        # 'svhn': using mean and std of 'svhn'
        # 'svhn': using mean and std of 'cifar100'
        return get_svhn_test_dataloader(dataset_norm_type=dataset)
    elif dataset == "gtsrb":
        return get_gtsrb_test_dataloder()
    else:
        raise ValueError(f"dataset `{dataset} is not supported`")
"""get subset of pytorch dataset"""
from torchvision.datasets import VisionDataset

from torch.utils.data import Dataset, DataLoader
import torch
from torch import Tensor

import pandas as pd

from PIL import Image

from typing import Tuple, Dict, Any
import math
import os


def calculate_categories_size(data_loader: DataLoader) -> Dict[int, int]:
    """calculate data size of each category"""
    categories_size = dict()
    for _, labels in data_loader:
        for label in labels:
            label = label.item()
            try:
                categories_size[label] += 1
            except KeyError:
                categories_size[label] = 1

    return categories_size


class SubsetDataset(Dataset):

    def __init__(self, whole_data_loader: DataLoader, partition_ratio: float):
        """get subset of targeted dataset

        Args:
            whole_data_loader: original whole data loader,
            partition_ratio: proportion of partitioned subset

        Steps:
            1. calculate the original dataset size of each category
            2. calculate the dataset size after scaling according to the partition ratio
            3. generate subset dataset

        Notes:
            1. we presume that dataset size of each category of original data loader is same
            2. the original data loader should not be shuffled lest influence to the experiment
        """
        dataset_len = len(whole_data_loader.dataset) * partition_ratio
        whole_categories_size = list(calculate_categories_size(whole_data_loader).values())
        # check if all categories have the same dataset size
        if not all(x == whole_categories_size[0] for x in whole_categories_size):
            raise ValueError("size of categories are not same!")

        if not self._is_integer(dataset_len):
            raise ValueError("length of subset must be integer, choose the correct `partition ratio`!")
        else:
            self._dataset_len = math.ceil(dataset_len)

        subset_category_size = whole_categories_size[0] * partition_ratio
        if not self._is_integer(subset_category_size):
            raise ValueError("dataset size of subset category must be integer, choose the correct `partition ratio`!")
        else:
            subset_category_size = math.ceil(subset_category_size)

        inputs_tensor_list = []
        labels_tensor_list = []

        subset_categories_size = {}
        # todo
        # traverse the whole iterator may be slow
        for inputs, labels in whole_data_loader:
            for _input, label in zip(inputs, labels):
                label_item = label.item()
                try:
                    subset_categories_size[label_item] += 1
                except KeyError:
                    subset_categories_size[label_item] = 1
                if subset_categories_size[label_item] <= subset_category_size:
                    inputs_tensor_list.append(_input)
                    labels_tensor_list.append(label)

        self._inputs = torch.stack(inputs_tensor_list, dim=0)
        self._labels = torch.stack(labels_tensor_list, dim=0)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        inputs = self._inputs[idx]
        labels = self._labels[idx]

        return inputs, labels

    def __len__(self):
        return self._dataset_len

    @staticmethod
    def _is_integer(number: Any, threshold: float = 1e-10) -> bool:
        if abs(number-math.ceil(number)) > threshold:
            return False
        return True


class GTSRB(VisionDataset):

    train_csv_path: str = "Train.csv"
    test_csv_path: str = "Test.csv"

    def __init__(self, root: str, *,
                 train=True,
                 transform=None,
                 target_transform=None):
        super(GTSRB, self).__init__(root, transform=transform,
                                    target_transform=target_transform)

        self.train = train

        if self.train:
            csv_file_path = self.train_csv_path
        else:
            csv_file_path = self.test_csv_path
        csv_file_path = os.path.join(self.root, csv_file_path)

        if not os.path.exists(csv_file_path):
            raise ValueError(
                f"gtsrb dataset not found at {self.root}, "
                f"please download at `https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign`"
            )

        self._csv_data = pd.read_csv(csv_file_path)

        self._data = []
        self._targets = []
        self._prepare_data_and_targets()

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        img, target = self._data[index], self._targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _prepare_data_and_targets(self):
        print(f"prepare gtsrb {'train' if self.train else 'test'} dataset")
        for line in self._csv_data.iloc:
            self._data.append(
                Image.open(
                    os.path.join(self.root, line[-1])
                ).copy()
            )

            self._targets.append(line[-2])

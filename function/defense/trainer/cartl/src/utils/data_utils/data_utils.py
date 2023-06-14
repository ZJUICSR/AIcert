from typing import Dict, Any, Union
import json

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
import torch
from torch import Tensor
from torch.nn import Module
import numpy as np
import matplotlib.pyplot as plt


def clamp(t: Tensor, lower_limit, upper_limit):
    return torch.max(torch.min(t, upper_limit), lower_limit)


def evaluate_accuracy(model: Module, test_loader: DataLoader,
                      device: Union[str, torch.device], debug=False) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            _, y_hats = model(inputs).max(1)
            match = (y_hats == labels)
            correct += len(match.nonzero())

    if debug:
        print(f"Testing: {len(test_loader.dataset)}")
        print(f"correct: {correct}")
        print(f"accuracy: {100 * correct / len(test_loader.dataset):.3f}%")

    model.train()

    return correct / len(test_loader.dataset)


def get_fc_out_features(model: Module):
    if not hasattr(model, "fc"):
        raise AttributeError("model doesn't have attribute as fully connected layer!")

    return model.fc.out_features


def compute_mean_std(dataset: Tensor):
    """compute the mean and std of `n * 3 * weight * height` dataset

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = np.dstack([dataset[i][0, :, :] for i in range(len(dataset))])
    data_g = np.dstack([dataset[i][1, :, :] for i in range(len(dataset))])
    data_b = np.dstack([dataset[i][2, :, :] for i in range(len(dataset))])
    mean = np.mean(data_r), np.mean(data_g), np.mean(data_b)
    std = np.std(data_r), np.std(data_g), np.std(data_b)

    return mean, std


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def attack_loss(X, y, model: torch.nn.Module, attacker, eps=0.1, loss=torch.nn.CrossEntropyLoss()) -> torch.Tensor:
    y_hat = model(X)
    adv_y_hat = attacker(model, eps=eps).cal_perturbation(X, y)

    return loss(y_hat, y) + loss(adv_y_hat, y)


def grey_to_img(img: Tensor):
    # change (3, H, W) to (H, W, 3)
    size = img.shape
    if len(size) == 3:
        area = size[1] * size[2]
        red = img[0].reshape(area, 1)
        green = img[1].reshape(area, 1)
        blue = img[2].reshape(area, 1)
        new_img = np.hstack([red, green, blue]).reshape((size[1], size[2], 3))
    elif len(size) == 2:
        new_img = img
    else:
        raise NotImplementedError(f"img shape: {size} is not supported")
    plt.imshow(new_img)
    plt.axis("off")
    plt.show()


def load_json(json_path: str) -> Dict[Any, Any]:
    with open(json_path, "r", encoding="utf8") as f:
        return json.loads(f.read())

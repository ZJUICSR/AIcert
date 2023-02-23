import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import json
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from typing import Optional, Tuple, TYPE_CHECKING
from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin
from art.utils import projection, get_labels_np_array, check_and_transform_label_format
import logging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE

class GDUniversarial(EvasionAttack):
    # 暂时添加，参数check，等待修改
    attack_params = EvasionAttack.attack_params + [
        "max_iter",
        "batch_size",
        "patience_interval",
        "sat_threshold",
        "sat_min",
        "eps"
    ]
    _estimator_requirements = (BaseEstimator, ClassifierMixin)

    def __init__(
        self,
        classifier: "CLASSIFIER_TYPE",
        batch_size: int = 32,
        patience_interval = 10000,
        max_iter = 20000,
        sat_threshold = 0.00001,
        sat_min = 0.5,
        eps = 0.8,
        lr = 0.01
    ) -> None:
        super().__init__(estimator=classifier)
        # self.model = model
        # self.data_loader, self.size = self.get_data_loader(dataset, batch_size=64, shuffle=False)
        # self.device = device
        self.asr = 0
        self.batch_size = batch_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.patience_interval = patience_interval
        self.max_iter = max_iter
        self.sat_threshold = sat_threshold
        self.sat_min = sat_min,
        self.eps = eps
        self.lr = lr
    
    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
    # def perturb(self):
        if y is not None:
            y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes)
        if y is None:
            # Use model predictions as true labels
            logger.info("Using model predictions as true labels.")
            y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))
        
        self.nb_instances = len(x)

        disable_tqdm=False
        debug = False
        # size = self.size

        sat_prev = 0
        sat = 0
        sat_change = 0
        sat_should_rescale = False

        iter_since_last_fooling = 0
        iter_since_last_best = 0
        best_fooling_rate = 0

        xi_min = 0
        xi_max = self.eps
        # delta = (xi_min - xi_max) * torch.rand(size, device=self.device) + xi_max
        delta = (xi_min - xi_max) * torch.rand(np.shape(x[[0]]), device=self.device) + xi_max
        best_delta = delta.data.cpu().numpy()
        delta.requires_grad = True

        print(f"Initial norm: {torch.norm(delta, p=np.inf)}")

        optimizer = optim.Adam([delta], lr=self.lr)

        for i in tqdm(range(self.max_iter), disable=disable_tqdm):
            iter_since_last_fooling += 1
            optimizer.zero_grad()
            # loss = self.l2_layer_loss(self.model, delta)
            loss = self.l2_layer_loss(self.estimator, delta)
            loss.backward()
            optimizer.step()

            if i % 1000 == 0:
                print(f"Iter {i}, Loss: {loss.item()}")
                if debug:
                    print(f"Norm before clip: {torch.norm(delta, p=np.inf)}")

            # clip delta after each step
            with torch.no_grad():
                delta.clamp_(xi_min, xi_max)

            # compute rate of saturation on a clamped delta
            sat_prev = np.copy(sat)
            sat = self.get_rate_of_saturation(delta.cpu().detach().numpy(), xi_max)
            sat_change = np.abs(sat - sat_prev)

            if sat_change < self.sat_threshold and sat > self.sat_min:
                if debug:
                    print(f"Saturated delta in iter {i} with {sat} > {self.sat_min}\nChange in saturation: {sat_change} < {self.sat_threshold}\n")
                sat_should_rescale = True

            # fooling rate is measured every 200 iterations if saturation threshold is crossed
            # otherwise, fooling rate is measured every 400 iterations
            if iter_since_last_fooling > 400 or (sat_should_rescale and iter_since_last_fooling > 200):
                iter_since_last_fooling = 0
                print("Getting latest fooling rate...")
                current_fooling_rate = self.get_fooling_rate(x, y, delta)
                print(f"Latest fooling rate: {current_fooling_rate}")

                if current_fooling_rate > best_fooling_rate:
                    print(f"Best fooling rate thus far: {current_fooling_rate}")
                    best_fooling_rate = current_fooling_rate
                    best_delta = delta.data.cpu().numpy()
                else:
                    iter_since_last_best += 1

                # if the best fooling rate has not been overcome after patience_interval iterations
                # then training is considered complete
                if iter_since_last_best == self.patience_interval:
                    break

            if sat_should_rescale:
                with torch.no_grad():
                    delta.data = delta.data / 2
                sat_should_rescale = False

        self.asr = best_fooling_rate

        print(f"Training complete.\nLast delta Iter: {i}, Loss: {loss}, Fooling rate: {best_fooling_rate}")
        # return delta, self.asr
        return np.add(x, best_delta)

    def get_rate_of_saturation(self, delta, xi):
        """
        Returns the proportion of pixels in delta
        that have reached the max-norm limit xi
        """
        return np.sum(np.equal(np.abs(delta), xi)) / np.size(delta)

    def l2_layer_loss(self, estimator, delta):
        loss = torch.tensor(0.)
        activations = []
        remove_handles = []

        def activation_recorder_hook(self, input, output):
            activations.append(output)
            return None

        for conv_layer in self.get_conv_layers(estimator.model):
            handle = conv_layer.register_forward_hook(activation_recorder_hook)
            remove_handles.append(handle)

        estimator.model.eval()
        estimator.model.zero_grad()
        estimator.model(delta)

        # unregister hook so activation tensors have no references
        for handle in remove_handles:
            handle.remove()

        loss = -sum(list(map(lambda activation: torch.log(torch.sum(torch.square(activation)) / 2), activations)))
        return loss

    def get_conv_layers(self, model):
        return [module for module in model.modules() if type(module) == nn.Conv2d]

    def get_fooling_rate(self, x, y, delta):
        y_adv = np.argmax(self.estimator.predict(np.add(x, delta.cpu().detach().numpy()), batch_size=self.batch_size), axis=1)
        fooling_rate = np.sum(np.argmax(y, axis=1) != y_adv) / self.nb_instances
        return fooling_rate

    # def get_fooling_rate(self, x, y, delta):
    #     """
    #     Computes the fooling rate of the UAP delta on the dataset.
    #     Fooling rate is a measure of change in the model's output
    #     caused by the perturbation. In this case, fooling rate is
    #     the proportion of outputs that are changed by adding delta.
    #     Ex. delta = torch.zeros() should have a fooling rate of 0.0
    #     """
    #     flipped = 0
    #     total = 0
        
    #     # self.model.eval()

    #     # 唯一用到数据的位置待实现
    #     with torch.no_grad():
    #         for batch in tqdm(self.data_loader, disable=disable_tqdm):
    #             images, labels = batch
    #             images = images.to(self.device)
    #             adv_images = torch.add(delta, images).clamp(0, 1)

    #             # outputs = self.model(images)
    #             outputs = self.estimator(images)
    #             # adv_outputs = self.model(adv_images)
    #             adv_outputs = self.estimator(adv_images)

    #             _, predicted = torch.max(outputs.data, 1)
    #             _, adv_predicted = torch.max(adv_outputs.data, 1)

    #             total += images.size(0)
    #             flipped += (predicted != adv_predicted).sum().item()

    #     return flipped / total


    # def get_data_loader(self, dataset_name, batch_size=64, shuffle=False):
    #     """
    #     Returns a DataLoader with validation images for dataset_name
    #     """
        
    #     if dataset_name == 'CIFAR10':
    #         transform = transforms.Compose([transforms.ToTensor(),
    #                               transforms.Normalize(
    #                                 mean=[0.4914, 0.4822, 0.4465],
    #                                 std=[0.2023, 0.1994, 0.2010],
    #                             )])
    #         size = (1, 3, 32, 32)
    #         val_dataset = datasets.CIFAR10(root='/mnt/data/dataset/cifar10', train=False, transform=transform)
            
    #     elif dataset_name == 'MNIST':
    #         transform = transforms.Compose([
    #                 transforms.ToTensor(),
    #                 transforms.Normalize((0.1307, ), (0.3081, ))
    #         ])
    #         size = (1, 1, 28, 28)
    #         val_dataset = datasets.MNIST(root='/mnt/data/dataset/mnist', train=False, transform=transform)

    #     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    #     return val_loader, size
# import logging
import math
import torch
from typing import Optional, Union, TYPE_CHECKING
import numpy as np
from tqdm.auto import trange
from function.attack.attacks.config import MY_NUMPY_DTYPE
from function.attack.attacks.attack import EvasionAttack
from function.attack.estimators.estimator import BaseEstimator, LossGradientsMixin
from function.attack.estimators.classification.classifier import ClassifierMixin
from function.attack.attacks.utils import check_and_transform_label_format, projection, random_sphere, is_probability, get_labels_np_array
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
from function.attack.attacks.utils import get_labels_np_array, check_and_transform_label_format, compute_success_array, random_sphere, compute_success
if TYPE_CHECKING:
    from function.attack.attacks.utils import CLASSIFIER_LOSS_GRADIENTS_TYPE

# logger = logging.getLogger(__name__)
# 无穷范数下的攻击
class GDUAP(EvasionAttack):
    attack_params = EvasionAttack.attack_params + [
        "eps",
        "max_iter",
        "batch_size",
        "sat_min",
        "sat_threshold",
        "patience_interval",
    ]
    _estimator_requirements = (BaseEstimator, LossGradientsMixin, ClassifierMixin)

    def __init__(
        self,
        estimator: "CLASSIFIER_LOSS_GRADIENTS_TYPE",
        eps: float = 20/255,
        max_iter: int = 10000,
        batch_size: int = 128,
        sat_min = 0.35,
        sat_threshold = 0.00001,
        patience_interval = 10,
    ):
        super().__init__(estimator=estimator)
        # self.norm = norm
        # self.estimator = estimator
        self.eps = eps
        # self.eps_step = eps_step
        self.max_iter = max_iter
        # self.targeted = targeted
        # self.nb_random_init = nb_random_init
        self.batch_size = batch_size
        # self.loss_type = loss_type
        # self.verbose = verbose
        # self._check_params()
        self.sat_min = sat_min
        self.sat_threshold = sat_threshold
        self.patience_interval = patience_interval
    
    def get_conv_layers(self, model):
        return [module for module in model.modules() if type(module) == nn.Conv2d]

    def l2_layer_loss(self, model, delta):
        loss = torch.tensor(0.)
        activations = []
        remove_handles = []

        def activation_recorder_hook(self, input, output):
            activations.append(output)
            return None

        for conv_layer in self.get_conv_layers(model):
            handle = conv_layer.register_forward_hook(activation_recorder_hook)
            remove_handles.append(handle)

        model.eval()
        model.zero_grad()
        model(delta)

        # unregister hook so activation tensors have no references
        for handle in remove_handles:
            handle.remove()

        loss = -sum(list(map(lambda activation: torch.log(torch.sum(torch.square(activation)) / 2), activations)))
        return loss

    def get_rate_of_saturation(self, delta, xi):
        return np.sum(np.equal(np.abs(delta), xi)) / np.size(delta)

    def get_fooling_rate(self, delta, x, y):
        
        # dataset = torch.utils.data.TensorDataset(
        #     torch.from_numpy(x.astype(MY_NUMPY_DTYPE)),
        #     torch.from_numpy(y.astype(MY_NUMPY_DTYPE)),
        # )

        # data_loader = torch.utils.data.DataLoader(
        #     dataset=dataset, batch_size=self.batch_size, shuffle=False, drop_last=False
        # )

        # flipped = 0
        # total = 0
        # model.eval()

        # with torch.no_grad():
        #     for batch in tqdm(data_loader):
        #         images, labels = batch
        #         images = images.to(device)
        #         adv_images = torch.add(delta, images).clamp(0, 1)

        #         outputs = model(normalize(images))
        #         adv_outputs = model(normalize(adv_images))
        #         _, predicted = torch.max(outputs.data, 1)
        #         _, adv_predicted = torch.max(adv_outputs.data, 1)

        #         total += images.size(0)
        #         flipped += (predicted != adv_predicted).sum().item()

        attack_success = compute_success(
            self.estimator,
            x,
            y,
            x+delta,
            False,
            batch_size=self.batch_size,
        )
        
        return attack_success

    # def gd_universal_adversarial_perturbation(self, model, model_name, train_type, batch_size, device, dataset_name, patience_interval, id):
    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        
        if y is not None:
            y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes)
        else:
            if self.targeted:
                raise ValueError("Target labels `y` need to be provided for a targeted attack.")
            y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size)).astype(int)
    
        sat_prev = 0
        sat = 0
        sat_change = 0
        sat_should_rescale = False
        iter_since_last_fooling = 0
        iter_since_last_best = 0
        best_fooling_rate = 0
        xi_min = -self.eps
        xi_max = self.eps

        # 首先随机生成范围内的随机噪声
        input_shape = list(self.estimator.input_shape)
        input_shape.insert(0,1)
        delta = (xi_min - xi_max) * torch.rand(tuple(input_shape), device=self.estimator.device) + xi_max

        delta.requires_grad = True

        optimizer = optim.Adam([delta])

        # 噪声优化迭代
        for i in tqdm(range(self.max_iter)):
            iter_since_last_fooling += 1
            optimizer.zero_grad()
            loss = self.l2_layer_loss(self.estimator.model, delta)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                delta.clamp_(xi_min, xi_max)
                # test = torch.from_numpy(x).to(device=self.estimator.device)
                # delta = torch.clamp(torch.from_numpy(x).to(device=self.estimator.device)+delta, self.estimator.clip_values[0], self.estimator.clip_values[1]) - torch.from_numpy(x).to(device=self.estimator.device)
            
            sat_prev = np.copy(sat)
            sat = self.get_rate_of_saturation(delta.cpu().detach().numpy(), xi_max)
            sat_change = np.abs(sat - sat_prev)

            if sat_change < self.sat_threshold and sat > self.sat_min:
                sat_should_rescale = True
            
            if iter_since_last_fooling > 400 or (sat_should_rescale and iter_since_last_fooling > 200):
                iter_since_last_fooling = 0
                print("Getting latest fooling rate...")
                current_fooling_rate = self.get_fooling_rate(delta.cpu().detach().numpy(), x, y)
                print(f"Latest fooling rate: {current_fooling_rate}")

                if current_fooling_rate > best_fooling_rate:
                    best_fooling_rate = current_fooling_rate
                else:
                    iter_since_last_best += 1
                
                if iter_since_last_best == self.patience_interval:
                    break

            if sat_should_rescale:
                with torch.no_grad():
                    delta.data = delta.data / 3
                sat_should_rescale = False
        
        return np.clip(delta.cpu().detach().numpy() + x, self.estimator.clip_values[0], self.estimator.clip_values[1])
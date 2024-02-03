from __future__ import absolute_import, division, print_function, unicode_literals
from functools import reduce
import logging
from typing import Optional, Tuple, Union, TYPE_CHECKING
import numpy as np
from tqdm.auto import trange
from function.attack.attacks.attack import PoisoningAttackWhiteBox
from function.attack.estimators import BaseEstimator, NeuralNetworkMixin
from function.attack.estimators.classification.classifier import ClassifierMixin
from function.attack.estimators.classification.pytorch import PyTorchClassifier
from function.attack.estimators.classification.keras import KerasClassifier
if TYPE_CHECKING:
    from function.attack.attacks.utils import CLASSIFIER_NEURALNETWORK_TYPE
logger = logging.getLogger(__name__)

class FeatureCollisionAttack(PoisoningAttackWhiteBox):
    attack_params = PoisoningAttackWhiteBox.attack_params + [
        "feature_layer",
        "learning_rate",
        "decay_coeff",
        "stopping_tol",
        "obj_threshold",
        "num_old_obj",
        "max_iter",
        "similarity_coeff",
        "watermark",
    ]
    _estimator_requirements = (BaseEstimator, NeuralNetworkMixin, ClassifierMixin)

    def __init__(
        self,
        classifier: "CLASSIFIER_NEURALNETWORK_TYPE",
        target: np.ndarray,
        feature_layer: Union[str, int],
        learning_rate: float = 500 * 255.0,
        decay_coeff: float = 0.5,
        stopping_tol: float = 1e-10,
        obj_threshold: Optional[float] = None,
        num_old_obj: int = 40,
        max_iter: int = 120,
        similarity_coeff: float = 256.0,
        watermark: Optional[float] = None,
        verbose: bool = True,
    ):
        super().__init__(classifier=classifier)  # type: ignore
        self.target = target
        self.feature_layer = feature_layer
        self.learning_rate = learning_rate
        self.decay_coeff = decay_coeff
        self.stopping_tol = stopping_tol
        self.obj_threshold = obj_threshold
        self.num_old_obj = num_old_obj
        self.max_iter = max_iter
        self.similarity_coeff = similarity_coeff
        self.watermark = watermark
        self.verbose = verbose
        self._check_params()

        if isinstance(self.estimator, PyTorchClassifier):
            self.target_feature_rep = self.estimator.get_activations(self.target, self.feature_layer, 1, framework=True)
            self.poison_feature_rep = self.estimator.get_activations(self.target, self.feature_layer, 1, framework=True)
        else:
            raise ValueError("Type of estimator currently not supported.")
        self.attack_loss = tensor_norm(self.poison_feature_rep - self.target_feature_rep)
    
    # 需要在外部指定哪些样本被投毒
    # 只应该对一个类进行攻击
    def poison(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs):
        num_poison = len(x)
        final_attacks = []
        if num_poison == 0:  # pragma: no cover
            raise ValueError("Must input at least one poison point")
        target_features = self.estimator.get_activations(self.target, self.feature_layer, 1)
        for index in trange(len(x), desc="Feature collision", disable=True):
            init_attack = x[index]
            old_attack = np.expand_dims(np.copy(init_attack), axis=0)
            poison_features = self.estimator.get_activations(old_attack, self.feature_layer, 1)
            old_objective = self.objective(poison_features, target_features, init_attack, old_attack)
            last_m_objectives = [old_objective]
            for i in trange(self.max_iter, desc="Feature collision", disable=True):
                # forward step
                new_attack = self.forward_step(old_attack)

                # backward step
                new_attack = self.backward_step(np.expand_dims(init_attack, axis=0), poison_features, new_attack)

                rel_change_val = np.linalg.norm(new_attack - old_attack) / np.linalg.norm(new_attack)
                if (  # pragma: no cover
                    rel_change_val < self.stopping_tol or self.obj_threshold and old_objective <= self.obj_threshold
                ):
                    logger.info("stopped after %d iterations due to small changes", i)
                    break

                np.expand_dims(new_attack, axis=0)
                new_feature_rep = self.estimator.get_activations(new_attack, self.feature_layer, 1)
                new_objective = self.objective(new_feature_rep, target_features, init_attack, new_attack)

                avg_of_last_m = sum(last_m_objectives) / float(min(self.num_old_obj, i + 1))

                # Increasing objective means then learning rate is too big.  Chop it, and throw out the latest iteration
                if new_objective >= avg_of_last_m and (i % self.num_old_obj / 2 == 0):
                    self.learning_rate *= self.decay_coeff
                else:  # pragma: no cover
                    old_attack = new_attack
                    old_objective = new_objective

                if i < self.num_old_obj - 1:
                    last_m_objectives.append(new_objective)
                else:  # pragma: no cover
                    # first remove the oldest obj then append the new obj
                    del last_m_objectives[0]
                    last_m_objectives.append(new_objective)

            # Watermarking
            watermark = self.watermark * self.target if self.watermark else 0
            final_poison = np.clip(old_attack + watermark, *self.estimator.clip_values)
            final_attacks.append(final_poison)

        return np.vstack(final_attacks)

    def forward_step(self, poison: np.ndarray) -> np.ndarray:
        # if isinstance(self.estimator, KerasClassifier):
        #     (attack_grad,) = self.estimator.custom_loss_gradient(
        #         self.attack_loss,
        #         [self.poison_placeholder, self.target_placeholder],
        #         [poison, self.target],
        #         name="feature_collision_" + str(self.feature_layer),
        #     )
        # elif isinstance(self.estimator, PyTorchClassifier):
        #     attack_grad = self.estimator.custom_loss_gradient(self.attack_loss, poison, self.target, self.feature_layer)

        attack_grad = self.estimator.custom_loss_gradient(self.attack_loss, poison, self.target, self.feature_layer)
        poison -= self.learning_rate * attack_grad[0]

        return poison

    def backward_step(self, base: np.ndarray, feature_rep: np.ndarray, poison: np.ndarray) -> np.ndarray:
        num_features = reduce(lambda x, y: x * y, base.shape)
        dim_features = feature_rep.shape[-1]
        beta = self.similarity_coeff * (dim_features / num_features) ** 2
        poison = (poison + self.learning_rate * beta * base) / (1 + beta * self.learning_rate)
        low, high = self.estimator.clip_values
        return np.clip(poison, low, high)

    def objective(
        self, poison_feature_rep: np.ndarray, target_feature_rep: np.ndarray, base_image: np.ndarray, poison: np.ndarray
    ) -> float:
        num_features = base_image.size
        num_activations = poison_feature_rep.size
        beta = self.similarity_coeff * (num_activations / num_features) ** 2
        return np.linalg.norm(poison_feature_rep - target_feature_rep) + beta * np.linalg.norm(poison - base_image)

    def _check_params(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be strictly positive")
        if not isinstance(self.feature_layer, (str, int)):
            raise TypeError("Feature layer should be a string or int")
        if self.decay_coeff <= 0:
            raise ValueError("Decay coefficient must be positive")
        if self.stopping_tol <= 0:
            raise ValueError("Stopping tolerance must be positive")
        if self.obj_threshold and self.obj_threshold <= 0:
            raise ValueError("Objective threshold must be positive")
        if self.num_old_obj <= 0:
            raise ValueError("Number of old stored objectives must be positive")
        if self.max_iter <= 0:
            raise ValueError("Maximum number of iterations must be 1 or larger")
        if self.watermark and not (isinstance(self.watermark, float) and 0 <= self.watermark < 1):
            raise ValueError("Watermark must be between 0 and 1")
        if not isinstance(self.verbose, bool):
            raise ValueError("The argument `verbose` has to be of type bool.")

def get_class_name(obj: object) -> str:
    module = obj.__class__.__module__
    if module is None or module == str.__class__.__module__:
        return obj.__class__.__name__
    return module + "." + obj.__class__.__name__

def tensor_norm(tensor, norm_type: Union[int, float, str] = 2):  # pylint: disable=R1710
    tf_tensor_types = ("tensorflow.python.framework.ops.Tensor", "tensorflow.python.framework.ops.EagerTensor")
    torch_tensor_types = ("torch.Tensor", "torch.float", "torch.double", "torch.long")
    mxnet_tensor_types = ()
    supported_types = tf_tensor_types + torch_tensor_types + mxnet_tensor_types
    tensor_type = get_class_name(tensor)
    if tensor_type not in supported_types:  # pragma: no cover
        raise TypeError("Tensor type `" + tensor_type + "` is not supported")
    if tensor_type in tf_tensor_types:
        import tensorflow as tf
        return tf.norm(tensor, ord=norm_type)
    if tensor_type in torch_tensor_types:  # pragma: no cover
        import torch
        return torch.norm
    # if tensor_type in mxnet_tensor_types:  # pragma: no cover
    #     import mxnt
    #     return mxnet.ndarray.norm(tensor, ord=norm_type)

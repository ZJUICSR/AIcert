# from __future__ import absolute_import, division, print_function, unicode_literals
# import logging
from typing import Optional, Union, TYPE_CHECKING
import numpy as np
from function.attack.attacks.config import MY_NUMPY_DTYPE
from function.attack.attacks.attack import EvasionAttack
from function.attack.estimators.estimator import BaseEstimator, LossGradientsMixin
from function.attack.estimators.classification.classifier import ClassifierMixin
from function.attack.attacks.utils import (
    get_labels_np_array,
    check_and_transform_label_format,
)

if TYPE_CHECKING:
    from function.attack.attacks.utils import CLASSIFIER_LOSS_GRADIENTS_TYPE

class FastGradientMethod(EvasionAttack):

    attack_params = EvasionAttack.attack_params + [
        "norm",
        "eps",
        "targeted",
        "batch_size",
    ]

    _estimator_requirements = (BaseEstimator, LossGradientsMixin)

    def __init__(
        self,
        estimator: "CLASSIFIER_LOSS_GRADIENTS_TYPE",
        norm: Union[int, float, str] = np.inf,
        eps: Union[int, float, np.ndarray] = 8/255,
        targeted: bool = False,
        batch_size: int = 128,
    ) -> None:
        # super().__init__(estimator=estimator, summary_writer=summary_writer)
        super().__init__(estimator=estimator)
        self.norm = norm
        self.eps = eps
        # self.eps_step = eps_step
        self._targeted = targeted
        # self.num_random_init = num_random_init
        self.batch_size = batch_size
        # self.minimal = minimal
        self._project = True
        # FastGradientMethod._check_params(self)

        self._batch_id = 0
        self._i_max_iter = 0

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        # mask = self._get_mask(x, **kwargs)

        # Ensure eps is broadcastable
        # self._check_compatibility_input_and_eps(x=x)

        if isinstance(self.estimator, ClassifierMixin):
            
            if self.targeted:
                if y is None:
                    raise ValueError("Target labels `y` only need to be provided for a targeted attack.")
                else:
                    y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes)
            else:
                y_array = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))

            if self.estimator.nb_classes > 2:
                y_array = y_array / np.sum(y_array, axis=1, keepdims=True)

            # Return adversarial examples computed with minimal perturbation if option is active
            adv_x_best = x
            
            adv_x = self._compute(
                    x,
                    y_array,
                    self.eps,
            )
            adv_x_best = adv_x

        else:
            # if self.minimal:  # pragma: no cover
            #     raise ValueError("Minimal perturbation is only supported for classification.")

            if y is None:
                # Throw error if attack is targeted, but no targets are provided
                if self.targeted:  # pragma: no cover
                    raise ValueError("Target labels `y` need to be provided for a targeted attack.")

                # Use model predictions as correct outputs
                # logger.info("Using model predictions as correct labels for FGM.")
                y_array = self.estimator.predict(x, batch_size=self.batch_size)
            else:
                y_array = y

            adv_x_best = self._compute(
                x,
                y_array,
                self.eps,
                self._project,
                # self.num_random_init > 0,
            )

        if self.summary_writer is not None:
            self.summary_writer.reset()

        return adv_x_best

    
    # 计算噪声，无穷范数约束下直接返回梯度符号
    def _compute_perturbation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        # mask: Optional[np.ndarray],
        # decay: Optional[float] = None,
        # momentum: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        # Pick a small scalar to avoid division by 0
        # tol = 10e-8

        # 梯度计算
        grad = self.estimator.loss_gradient(x, y) * (1 - 2 * int(self.targeted))

      

        # Check for NaN before normalisation an replace with 0
        if grad.dtype != object and np.isnan(grad).any():  # pragma: no cover
            # logger.warning("Elements of the loss gradient are NaN and have been replaced with 0.0.")
            grad = np.where(np.isnan(grad), 0.0, grad)
        else:
            for i, _ in enumerate(grad):
                grad_i_array = grad[i].astype(np.float32)
                if np.isnan(grad_i_array).any():
                    grad[i] = np.where(np.isnan(grad_i_array), 0.0, grad_i_array).astype(object)

        # # Apply mask
        # if mask is not None:
        #     grad = np.where(mask == 0.0, 0.0, grad)

        # Apply norm bound
        def _apply_norm(norm, grad, object_type=False):
            # if (grad.dtype != object and np.isinf(grad).any()) or np.isnan(  # pragma: no cover
            #     grad.astype(np.float32)
            # ).any():
            #     logger.info("The loss gradient array contains at least one positive or negative infinity.")

            if norm in [np.inf, "inf"]:
                grad = np.sign(grad)
           
            return grad

        if x.dtype == object:
            for i_sample in range(x.shape[0]):
                grad[i_sample] = _apply_norm(self.norm, grad[i_sample], object_type=True)
                assert x[i_sample].shape == grad[i_sample].shape
        else:
            grad = _apply_norm(self.norm, grad)

        assert x.shape == grad.shape

        return grad
    
    # 使用梯度符号乘以最大的无穷范数约束，叠加到样本上得到对抗样本
    def _apply_perturbation(
        self, x: np.ndarray, perturbation: np.ndarray, eps: Union[int, float, np.ndarray]
    ) -> np.ndarray:

        perturbation_step = eps * perturbation
        if perturbation_step.dtype != object:
            perturbation_step[np.isnan(perturbation_step)] = 0
        else:
            for i, _ in enumerate(perturbation_step):
                perturbation_step_i_array = perturbation_step[i].astype(np.float32)
                if np.isnan(perturbation_step_i_array).any():
                    perturbation_step[i] = np.where(
                        np.isnan(perturbation_step_i_array), 0.0, perturbation_step_i_array
                    ).astype(object)

        x = x + perturbation_step
        if self.estimator.clip_values is not None:
            clip_min, clip_max = self.estimator.clip_values
            if x.dtype == object:
                for i_obj in range(x.shape[0]):
                    x[i_obj] = np.clip(x[i_obj], clip_min, clip_max)
            else:
                x = np.clip(x, clip_min, clip_max)

        return x
    
    # 批量计算对抗样本
    def _compute(
        self,
        x: np.ndarray,
        # x_init: np.ndarray,
        y: np.ndarray,
        # mask: Optional[np.ndarray],
        eps: Union[int, float, np.ndarray],
        
    ) -> np.ndarray:
        
        if x.dtype == object:
            x_adv = x.copy()
        else:
            x_adv = x.astype(MY_NUMPY_DTYPE)
        # Compute perturbation with implicit batching
        for batch_id in range(int(np.ceil(x.shape[0] / float(self.batch_size)))):
            # if batch_id_ext is None:
            #     self._batch_id = batch_id
            # else:
            #     self._batch_id = batch_id_ext
            self._batch_id = batch_id
            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            batch_index_2 = min(batch_index_2, x.shape[0])
            batch = x_adv[batch_index_1:batch_index_2]
            batch_labels = y[batch_index_1:batch_index_2]

            # Get perturbation
            perturbation = self._compute_perturbation(batch, batch_labels)

            # batch_eps = eps
            batch_eps_step = eps
            # Apply perturbation and clip
            x_adv[batch_index_1:batch_index_2] = self._apply_perturbation(batch, perturbation, batch_eps_step)

        return x_adv
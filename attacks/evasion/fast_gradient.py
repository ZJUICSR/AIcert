# from __future__ import absolute_import, division, print_function, unicode_literals
# import logging
from typing import Optional, Union, TYPE_CHECKING
import numpy as np
from attacks.config import MY_NUMPY_DTYPE
from attacks.attack import EvasionAttack
from estimators.estimator import BaseEstimator, LossGradientsMixin
from estimators.classification.classifier import ClassifierMixin
from attacks.utils import (
    get_labels_np_array,
    check_and_transform_label_format,
)
# from art.summary_writer import SummaryWriter

if TYPE_CHECKING:
    from attacks.utils import CLASSIFIER_LOSS_GRADIENTS_TYPE

# logger = logging.getLogger(__name__)

class FastGradientMethod(EvasionAttack):
    
    # attack_params = EvasionAttack.attack_params + [
    #     "norm",
    #     "eps",
    #     "eps_step",
    #     "targeted",
    #     "num_random_init",
    #     "batch_size",
    #     "minimal",
    #     "summary_writer",
    # ]

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
        # eps_step: Union[int, float, np.ndarray] = 0.03,
        targeted: bool = False,
        # num_random_init: int = 0,
        batch_size: int = 128,
        # minimal: bool = False,
        # summary_writer: Union[str, bool, SummaryWriter] = False,
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
            # if y is not None:
            #     y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes)
            
            # # if y is not None and not self.targeted:
            # #     raise ValueError("Target labels `y` only need to be provided for a targeted attack.")

            # if y is None:
            #     # Throw error if attack is targeted, but no targets are provided
            #     if self.targeted:  # pragma: no cover
            #         raise ValueError("Target labels `y` need to be provided for a targeted attack.")

            #     # Use model predictions as correct outputs
            #     # logger.info("Using model predictions as correct labels for FGM.")
            #     y_array = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))  # type: ignore
            # else:
            #     y_array = y
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
            # if self.minimal:
            #     # logger.info("Performing minimal perturbation FGM.")
            #     # adv_x_best = self._minimal_perturbation(x, y_array, mask)
            #     rate_best = 100 * compute_success(
            #         self.estimator,  # type: ignore
            #         x,
            #         y_array,
            #         adv_x_best,
            #         self.targeted,
            #         batch_size=self.batch_size,  # type: ignore
            #     )
            # else:
            #     rate_best = 0.0
            #     for _ in range(max(1, self.num_random_init)):
            #         adv_x = self._compute(
            #             x,
            #             x,
            #             y_array,
            #             mask,
            #             self.eps,
            #             self.eps,
            #             self._project,
            #             self.num_random_init > 0,
            #         )

            #         if self.num_random_init > 1:
            #             rate = 100 * compute_success(
            #                 self.estimator,  # type: ignore
            #                 x,
            #                 y_array,
            #                 adv_x,
            #                 self.targeted,
            #                 batch_size=self.batch_size,  # type: ignore
            #             )
            #             if rate > rate_best:
            #                 rate_best = rate
            #                 adv_x_best = adv_x
            #         else:
            #             adv_x_best = adv_x

                        #     rate_best = 0.0

            # for _ in range(max(1, self.num_random_init)):
            #     adv_x = self._compute(
            #         x,
            #         x,
            #         y_array,
            #         self.eps,
            #         self.eps,
            #         self._project,
            #     )
            #     adv_x_best = adv_x

            # logger.info(
            #     "Success rate of FGM attack: %.2f%%",
            #     rate_best
            #     if rate_best is not None
            #     else 100
            #     * compute_success(
            #         self.estimator,  # type: ignore
            #         x,
            #         y_array,
            #         adv_x_best,
            #         self.targeted,
            #         batch_size=self.batch_size,
            #     ),
            # )
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

    # def _check_params(self) -> None:

    #     if self.norm not in [1, 2, np.inf, "inf"]:
    #         raise ValueError('Norm order must be either 1, 2, `np.inf` or "inf".')

    #     if not (
    #         isinstance(self.eps, (int, float))
    #         and isinstance(self.eps_step, (int, float))
    #         or isinstance(self.eps, np.ndarray)
    #         and isinstance(self.eps_step, np.ndarray)
    #     ):
    #         raise TypeError(
    #             "The perturbation size `eps` and the perturbation step-size `eps_step` must have the same type of `int`"
    #             ", `float`, or `np.ndarray`."
    #         )

    #     if isinstance(self.eps, (int, float)):
    #         if self.eps < 0:
    #             raise ValueError("The perturbation size `eps` has to be nonnegative.")
    #     else:
    #         if (self.eps < 0).any():
    #             raise ValueError("The perturbation size `eps` has to be nonnegative.")

    #     if isinstance(self.eps_step, (int, float)):
    #         if self.eps_step <= 0:
    #             raise ValueError("The perturbation step-size `eps_step` has to be positive.")
    #     else:
    #         if (self.eps_step <= 0).any():
    #             raise ValueError("The perturbation step-size `eps_step` has to be positive.")

    #     if isinstance(self.eps, np.ndarray) and isinstance(self.eps_step, np.ndarray):
    #         if self.eps.shape != self.eps_step.shape:
    #             raise ValueError(
    #                 "The perturbation size `eps` and the perturbation step-size `eps_step` must have the same shape."
    #             )

    #     if not isinstance(self.targeted, bool):
    #         raise ValueError("The flag `targeted` has to be of type bool.")

        # if not isinstance(self.num_random_init, int):
        #     raise TypeError("The number of random initialisations has to be of type integer")

        # if self.num_random_init < 0:
        #     raise ValueError("The number of random initialisations `random_init` has to be greater than or equal to 0.")

        # if self.batch_size <= 0:
        #     raise ValueError("The batch size `batch_size` has to be positive.")

        # if not isinstance(self.minimal, bool):
        #     raise ValueError("The flag `minimal` has to be of type bool.")
    
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

        # # Write summary
        # if self.summary_writer is not None:  # pragma: no cover
        #     self.summary_writer.update(
        #         batch_id=self._batch_id,
        #         global_step=self._i_max_iter,
        #         grad=grad,
        #         patch=None,
        #         estimator=self.estimator,
        #         x=x,
        #         y=y,
        #         targeted=self.targeted,
        #     )

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
            # elif norm == 1:
            #     if not object_type:
            #         ind = tuple(range(1, len(x.shape)))
            #     else:
            #         ind = None
            #     grad = grad / (np.sum(np.abs(grad), axis=ind, keepdims=True) + tol)
            # elif norm == 2:
            #     if not object_type:
            #         ind = tuple(range(1, len(x.shape)))
            #     else:
            #         ind = None
            #     grad = grad / (np.sqrt(np.sum(np.square(grad), axis=ind, keepdims=True)) + tol)
            return grad

        # Add momentum
        # if decay is not None and momentum is not None:
        #     grad = _apply_norm(norm=1, grad=grad)
        #     grad = decay * momentum + grad
        #     momentum += grad

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
        # eps_step: Union[int, float, np.ndarray],
        # project: bool,
        # random_init: bool,
        # batch_id_ext: Optional[int] = None,
        # decay: Optional[float] = None,
        # momentum: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        # if random_init:
        #     n = x.shape[0]
        #     m = np.prod(x.shape[1:]).item()
        #     random_perturbation = random_sphere(n, m, eps, self.norm).reshape(x.shape).astype(MY_NUMPY_DTYPE)
        #     # if mask is not None:
        #     #     random_perturbation = random_perturbation * (mask.astype(MY_NUMPY_DTYPE))
        #     # x_adv = x.astype(MY_NUMPY_DTYPE) + random_perturbation

        #     if self.estimator.clip_values is not None:
        #         clip_min, clip_max = self.estimator.clip_values
        #         x_adv = np.clip(x_adv, clip_min, clip_max)
        # else:
        #     if x.dtype == object:
        #         x_adv = x.copy()
        #     else:
        #         x_adv = x.astype(MY_NUMPY_DTYPE)

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

            # mask_batch = mask
            # if mask is not None:
            #     # Here we need to make a distinction: if the masks are different for each input, we need to index
            #     # those for the current batch. Otherwise (i.e. mask is meant to be broadcasted), keep it as it is.
            #     if len(mask.shape) == len(x.shape):
            #         mask_batch = mask[batch_index_1:batch_index_2]

            # Get perturbation
            perturbation = self._compute_perturbation(batch, batch_labels)

            # Compute batch_eps and batch_eps_step
            # if isinstance(eps, np.ndarray) and isinstance(eps_step, np.ndarray):
            #     if len(eps.shape) == len(x.shape) and eps.shape[0] == x.shape[0]:
            #         batch_eps = eps[batch_index_1:batch_index_2]
            #         batch_eps_step = eps_step[batch_index_1:batch_index_2]

            #     else:
            #         batch_eps = eps
            #         batch_eps_step = eps_step
            # else:
            #     batch_eps = eps
            #     batch_eps_step = eps_step

            # batch_eps = eps
            batch_eps_step = eps
            # Apply perturbation and clip
            x_adv[batch_index_1:batch_index_2] = self._apply_perturbation(batch, perturbation, batch_eps_step)

        return x_adv
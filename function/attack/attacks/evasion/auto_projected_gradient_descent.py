# import logging..
import math
from typing import Optional, Union, TYPE_CHECKING
import numpy as np
from tqdm.auto import trange
from function.attack.attacks.config import MY_NUMPY_DTYPE
from function.attack.attacks.attack import EvasionAttack
from function.attack.estimators.estimator import BaseEstimator, LossGradientsMixin
from function.attack.estimators.classification.classifier import ClassifierMixin
from function.attack.attacks.utils import check_and_transform_label_format, projection, random_sphere, is_probability, get_labels_np_array

if TYPE_CHECKING:
    from function.attack.attacks.utils import CLASSIFIER_LOSS_GRADIENTS_TYPE

# logger = logging.getLogger(__name__)


class AutoProjectedGradientDescent(EvasionAttack):
    attack_params = EvasionAttack.attack_params + [
        "norm",
        "eps",
        "eps_step",
        "max_iter",
        "targeted",
        "nb_random_init",
        "batch_size",
        "loss_type",
        # "verbose",
    ]
    _estimator_requirements = (BaseEstimator, LossGradientsMixin, ClassifierMixin)
    _predefined_losses = [None, "cross_entropy", "difference_logits_ratio"]

    def __init__(
        self,
        estimator: "CLASSIFIER_LOSS_GRADIENTS_TYPE",
        norm: Union[int, float, str] = np.inf,
        eps: float = 0.3,
        eps_step: float = 0.1,
        max_iter: int = 10,
        targeted: bool = False,
        nb_random_init: int = 5,
        batch_size: int = 128,
        loss_type: Optional[str] = None,
        # verbose: bool = True,
    ):
        from function.attack.estimators.classification import PyTorchClassifier

        if loss_type not in self._predefined_losses:
            raise ValueError(
                f"The argument loss_type has an invalid value. The following options for `loss_type` are currently "
                f"supported: {self._predefined_losses}"
            )

        if loss_type is None:
            if hasattr(estimator, "predict") and is_probability(
                estimator.predict(x=np.ones(shape=(1, *estimator.input_shape), dtype=np.float32))
            ):
                raise ValueError(  # pragma: no cover
                    "AutoProjectedGradientDescent is expecting logits as estimator output, the provided "
                    "estimator seems to predict probabilities."
                )

            estimator_apgd = estimator
        else:
            import torch

            if loss_type == "cross_entropy":
                if is_probability(
                    estimator.predict(x=np.ones(shape=(1, *estimator.input_shape), dtype=np.float32))
                ):
                    raise ValueError(  # pragma: no cover
                        "The provided estimator seems to predict probabilities. If loss_type='cross_entropy' "
                        "the estimator has to to predict logits."
                    )

                self._loss_object = torch.nn.CrossEntropyLoss(reduction="mean")
            elif loss_type == "difference_logits_ratio":
                if is_probability(
                    estimator.predict(x=np.ones(shape=(1, *estimator.input_shape), dtype=MY_NUMPY_DTYPE))
                ):
                    raise ValueError(  # pragma: no cover
                        "The provided estimator seems to predict probabilities. "
                        "If loss_type='difference_logits_ratio' the estimator has to to predict logits."
                    )

                class DifferenceLogitsRatioPyTorch:
                    """
                    Callable class for Difference Logits Ratio loss in PyTorch.
                    """

                    def __init__(self):
                        self.reduction = "mean"

                    def __call__(self, y_pred, y_true):  # type: ignore
                        if isinstance(y_true, np.ndarray):
                            y_true = torch.from_numpy(y_true)
                        if isinstance(y_pred, np.ndarray):
                            y_pred = torch.from_numpy(y_pred)

                        y_true = y_true.float()

                        i_y_true = torch.argmax(y_true, axis=1)
                        i_y_pred_arg = torch.argsort(y_pred, axis=1)
                        i_z_i_list = []

                        for i in range(y_true.shape[0]):
                            if i_y_pred_arg[i, -1] != i_y_true[i]:
                                i_z_i_list.append(i_y_pred_arg[i, -1])
                            else:
                                i_z_i_list.append(i_y_pred_arg[i, -2])

                        i_z_i = torch.stack(i_z_i_list)

                        z_1 = y_pred[:, i_y_pred_arg[:, -1]]
                        z_3 = y_pred[:, i_y_pred_arg[:, -3]]
                        z_i = y_pred[:, i_z_i]
                        z_y = y_pred[:, i_y_true]

                        z_1 = torch.diagonal(z_1)
                        z_3 = torch.diagonal(z_3)
                        z_i = torch.diagonal(z_i)
                        z_y = torch.diagonal(z_y)

                        dlr = -(z_y - z_i) / (z_1 - z_3)

                        return torch.mean(dlr.float())

                self._loss_object = DifferenceLogitsRatioPyTorch()

            # estimator_apgd = PyTorchClassifier(
            #     model=estimator.model,
            #     loss=self._loss_object,
            #     input_shape=estimator.input_shape,
            #     nb_classes=estimator.nb_classes,
            #     optimizer=None,
            #     channels_first=estimator.channels_first,
            #     clip_values=estimator.clip_values,
            #     preprocessing_defences=estimator.preprocessing_defences,
            #     postprocessing_defences=estimator.postprocessing_defences,
            #     preprocessing=None,
            #     device_type=str(estimator._device),
            #     device=self.estimator.device,
            # )

        super().__init__(estimator=estimator)
        self.norm = norm
        self.eps = eps
        self.eps_step = eps_step
        self.max_iter = max_iter
        self.targeted = targeted
        self.nb_random_init = nb_random_init
        self.batch_size = batch_size
        self.loss_type = loss_type
        # self.verbose = verbose
        # self._check_params()

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        # mask = kwargs.get("mask")

        if y is not None:
            y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes)
        else:
            if self.targeted:
                raise ValueError("Target labels `y` need to be provided for a targeted attack.")
            y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size)).astype(int)

        # if self.estimator.nb_classes == 2 and y.shape[1] == 1:
        #     raise ValueError(
        #         "This attack has not yet been tested for binary classification with a single output classifier."
        #     )

        x_adv = x.astype(MY_NUMPY_DTYPE)

        for _ in trange(max(1, self.nb_random_init), desc="AutoPGD - restart"):
            # Determine correctly predicted samples
            y_pred = self.estimator.predict(x_adv)
            if self.targeted:
                sample_is_robust = np.argmax(y_pred, axis=1) != np.argmax(y, axis=1)
            elif not self.targeted:
                sample_is_robust = np.argmax(y_pred, axis=1) == np.argmax(y, axis=1)

            if np.sum(sample_is_robust) == 0:
                break

            x_robust = x_adv[sample_is_robust]
            y_robust = y[sample_is_robust]
            x_init = x[sample_is_robust]

            n = x_robust.shape[0]
            m = np.prod(x_robust.shape[1:]).item()
            random_perturbation = (
                random_sphere(n, m, self.eps, self.norm).reshape(x_robust.shape).astype(MY_NUMPY_DTYPE)
            )

            x_robust = x_robust + random_perturbation

            if self.estimator.clip_values is not None:
                clip_min, clip_max = self.estimator.clip_values
                x_robust = np.clip(x_robust, clip_min, clip_max)

            perturbation = projection(x_robust - x_init, self.eps, self.norm)
            x_robust = x_init + perturbation

            # Compute perturbation with implicit batching
            for batch_id in trange(
                int(np.ceil(x_robust.shape[0] / float(self.batch_size))),
                desc="AutoPGD - batch",
                leave=False,
            ):
                self.eta = 2 * self.eps_step
                batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
                x_k = x_robust[batch_index_1:batch_index_2].astype(MY_NUMPY_DTYPE)
                x_init_batch = x_init[batch_index_1:batch_index_2].astype(MY_NUMPY_DTYPE)
                y_batch = y_robust[batch_index_1:batch_index_2]

                p_0 = 0
                p_1 = 0.22
                var_w = [p_0, p_1]

                while True:
                    p_j_p_1 = var_w[-1] + max(var_w[-1] - var_w[-2] - 0.03, 0.06)
                    if p_j_p_1 > 1:
                        break
                    var_w.append(p_j_p_1)

                var_w = [math.ceil(p * self.max_iter) for p in var_w]

                eta = self.eps_step
                self.count_condition_1 = 0

                for k_iter in trange(self.max_iter, desc="AutoPGD - iteration", leave=False):

                    # Get perturbation, use small scalar to avoid division by 0
                    tol = 10e-8

                    # Get gradient wrt loss; invert it if attack is targeted
                    grad = self.estimator.loss_gradient(x_k, y_batch) * (1 - 2 * int(self.targeted))

                    # Apply norm bound
                    if self.norm in [np.inf, "inf"]:
                        grad = np.sign(grad)
                    elif self.norm == 1:
                        ind = tuple(range(1, len(x_k.shape)))
                        grad = grad / (np.sum(np.abs(grad), axis=ind, keepdims=True) + tol)
                    elif self.norm == 2:
                        ind = tuple(range(1, len(x_k.shape)))
                        grad = grad / (np.sqrt(np.sum(np.square(grad), axis=ind, keepdims=True)) + tol)
                    assert x_k.shape == grad.shape

                    perturbation = grad

                    # if mask is not None:
                    #     perturbation = perturbation * (mask.astype(MY_NUMPY_DTYPE))

                    # Apply perturbation and clip
                    z_k_p_1 = x_k + eta * perturbation

                    if self.estimator.clip_values is not None:
                        clip_min, clip_max = self.estimator.clip_values
                        z_k_p_1 = np.clip(z_k_p_1, clip_min, clip_max)

                    if k_iter == 0:
                        x_1 = z_k_p_1
                        perturbation = projection(x_1 - x_init_batch, self.eps, self.norm)
                        x_1 = x_init_batch + perturbation

                        f_0 = self.estimator.compute_loss(x=x_k, y=y_batch, reduction="mean")
                        f_1 = self.estimator.compute_loss(x=x_1, y=y_batch, reduction="mean")

                        self.eta_w_j_m_1 = eta
                        self.f_max_w_j_m_1 = f_0

                        if f_1 >= f_0:
                            self.f_max = f_1
                            self.x_max = x_1
                            self.x_max_m_1 = x_init_batch
                            self.count_condition_1 += 1
                        else:
                            self.f_max = f_0
                            self.x_max = x_k.copy()
                            self.x_max_m_1 = x_init_batch

                        # Settings for next iteration k
                        x_k_m_1 = x_k.copy()
                        x_k = x_1

                    else:
                        perturbation = projection(z_k_p_1 - x_init_batch, self.eps, self.norm)
                        z_k_p_1 = x_init_batch + perturbation

                        alpha = 0.75

                        x_k_p_1 = x_k + alpha * (z_k_p_1 - x_k) + (1 - alpha) * (x_k - x_k_m_1)

                        if self.estimator.clip_values is not None:
                            clip_min, clip_max = self.estimator.clip_values
                            x_k_p_1 = np.clip(x_k_p_1, clip_min, clip_max)

                        perturbation = projection(x_k_p_1 - x_init_batch, self.eps, self.norm)
                        x_k_p_1 = x_init_batch + perturbation

                        f_k_p_1 = self.estimator.compute_loss(x=x_k_p_1, y=y_batch, reduction="mean")

                        if f_k_p_1 == 0.0:
                            x_k = x_k_p_1.copy()
                            break

                        if (not self.targeted and f_k_p_1 > self.f_max) or (self.targeted and f_k_p_1 < self.f_max):
                            self.count_condition_1 += 1
                            self.x_max = x_k_p_1
                            self.x_max_m_1 = x_k
                            self.f_max = f_k_p_1

                        if k_iter in var_w:

                            rho = 0.75

                            condition_1 = self.count_condition_1 < rho * (k_iter - var_w[var_w.index(k_iter) - 1])
                            condition_2 = self.eta_w_j_m_1 == eta and self.f_max_w_j_m_1 == self.f_max

                            if condition_1 or condition_2:
                                eta = eta / 2
                                x_k_m_1 = self.x_max_m_1
                                x_k = self.x_max
                            else:
                                x_k_m_1 = x_k
                                x_k = x_k_p_1.copy()

                            self.count_condition_1 = 0
                            self.eta_w_j_m_1 = eta
                            self.f_max_w_j_m_1 = self.f_max

                        else:
                            x_k_m_1 = x_k
                            x_k = x_k_p_1.copy()

                y_pred_adv_k = self.estimator.predict(x_k)
                if self.targeted:
                    sample_is_not_robust_k = np.invert(np.argmax(y_pred_adv_k, axis=1) != np.argmax(y_batch, axis=1))
                elif not self.targeted:
                    sample_is_not_robust_k = np.invert(np.argmax(y_pred_adv_k, axis=1) == np.argmax(y_batch, axis=1))

                x_robust[batch_index_1:batch_index_2][sample_is_not_robust_k] = x_k[sample_is_not_robust_k]

            x_adv[sample_is_robust] = x_robust

        return x_adv

    # def _check_params(self) -> None:
    #     if self.norm not in [1, 2, np.inf, "inf"]:
    #         raise ValueError('The argument norm has to be either 1, 2, np.inf, or "inf".')

    #     if not isinstance(self.eps, (int, float)) or self.eps <= 0.0:
    #         raise ValueError("The argument eps has to be either of type int or float and larger than zero.")

    #     if not isinstance(self.eps_step, (int, float)) or self.eps_step <= 0.0:
    #         raise ValueError("The argument eps_step has to be either of type int or float and larger than zero.")

    #     if not isinstance(self.max_iter, int) or self.max_iter <= 0:
    #         raise ValueError("The argument max_iter has to be of type int and larger than zero.")

    #     if not isinstance(self.targeted, bool):
    #         raise ValueError("The argument targeted has to be of bool.")

    #     if not isinstance(self.nb_random_init, int) or self.nb_random_init <= 0:
    #         raise ValueError("The argument nb_random_init has to be of type int and larger than zero.")

    #     if not isinstance(self.batch_size, int) or self.batch_size <= 0:
    #         raise ValueError("The argument batch_size has to be of type int and larger than zero.")

    #     # if self.loss_type not in self._predefined_losses:
    #     #     raise ValueError("The argument loss_type has to be either {}.".format(self._predefined_losses))

    #     if not isinstance(self.verbose, bool):
    #         raise ValueError("The argument `verbose` has to be of type bool.")

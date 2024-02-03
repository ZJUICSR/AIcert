from __future__ import absolute_import, division, print_function, unicode_literals
# import logging
import numpy as np
from tqdm.auto import trange
from function.attack.attacks.config import MY_NUMPY_DTYPE
from function.attack.estimators.estimator import BaseEstimator
from function.attack.estimators.classification.classifier import ClassGradientsMixin
from function.attack.attacks.attack import EvasionAttack
from function.attack.attacks.utils import is_probability
from function.attack.attacks.utils import projection
from typing import Any, Dict, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from utils import CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE

# DeepFool攻击，第一个考虑精确寻找对抗样本的攻击，其默认寻找二范数约束下的最小对抗扰动
class DeepFool(EvasionAttack):
    attack_params = EvasionAttack.attack_params + [
        "max_iter",
        "eta",
        # "nb_grads",
        "batch_size",
        # "verbose",
        "norm",
    ]
    _estimator_requirements = (BaseEstimator, ClassGradientsMixin)

    def __init__(
        self,
        classifier: "CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE",
        max_iter: int = 50,
        eta: float = 0.02,
        # nb_grads: int = 10,
        batch_size: int = 128,
        norm: Union[int, float, str] = np.inf,
        # verbose: bool = True,
    ) -> None:
        super().__init__(estimator=classifier)
        self.max_iter = max_iter
        self.eta = eta
        # self.nb_grads = nb_grads
        self.batch_size = batch_size
        # self.verbose = verbose
        # self._check_params()
        # if self.estimator.clip_values is None:
        #     logger.warning(
        #         "The `clip_values` attribute of the estimator is `None`, therefore this instance of DeepFool will by "
        #         "default generate adversarial perturbations scaled for input values in the range [0, 1] but not clip "
        #         "the adversarial example."
        #     )
        self.norm = norm

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        x_adv = x.astype(MY_NUMPY_DTYPE)
        preds = self.estimator.predict(x, batch_size=self.batch_size)

        # if self.estimator.nb_classes == 2 and preds.shape[1] == 1:
        #     raise ValueError(  # pragma: no cover
        #         "This attack has not yet been tested for binary classification with a single output classifier."
        #     )

        # if is_probability(preds[0]):
        #     logger.warning(
        #         "It seems that the attacked model is predicting probabilities. DeepFool expects logits as model output "
        #         "to achieve its full attack strength."
        #     )

        # # Determine the class labels for which to compute the gradients
        # use_grads_subset = self.nb_grads < self.estimator.nb_classes
        # if use_grads_subset:
        #     # TODO compute set of unique labels per batch
        #     grad_labels = np.argsort(-preds, axis=1)[:, : self.nb_grads]
        #     labels_set = np.unique(grad_labels)
        # else:
        #     labels_set = np.arange(self.estimator.nb_classes)
        labels_set = np.arange(self.estimator.nb_classes)
        sorter = np.arange(len(labels_set))

        # 取一个极小的值在维持范数限制的时候避免除以0
        tol = 1e-7

        # Compute perturbation with implicit batching
        for batch_id in trange(
            int(np.ceil(x_adv.shape[0] / float(self.batch_size))), desc="DeepFool"
        ):
            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            batch = x_adv[batch_index_1:batch_index_2].copy()

            # Get predictions and gradients for batch
            f_batch = preds[batch_index_1:batch_index_2]
            fk_hat = np.argmax(f_batch, axis=1)
            # if use_grads_subset:
            #     # Compute gradients only for top predicted classes
            #     grd = np.array([self.estimator.class_gradient(batch, label=int(label_i)) for label_i in labels_set])
            #     grd = np.squeeze(np.swapaxes(grd, 0, 2), axis=0)
            # else:
            #     # Compute gradients for all classes
            #     grd = self.estimator.class_gradient(batch)
            grd = self.estimator.class_gradient(batch)
            
            # Get current predictions
            active_indices = np.arange(len(batch))
            current_step = 0
            while active_indices.size > 0 and current_step < self.max_iter:
                # Compute difference in predictions and gradients only for selected top predictions
                labels_indices = sorter[np.searchsorted(labels_set, fk_hat, sorter=sorter)]
                grad_diff = grd - grd[np.arange(len(grd)), labels_indices][:, None]
                f_diff = f_batch[:, labels_set] - f_batch[np.arange(len(f_batch)), labels_indices][:, None]

                # 这里的DeepFool仅仅考虑了二范数下的扰动
                if self.norm == 2:
                    norm = np.linalg.norm(grad_diff.reshape(len(grad_diff), len(labels_set), -1), axis=2, ord=2) + tol
                    value = np.abs(f_diff) / norm
                    value[np.arange(len(value)), labels_indices] = np.inf
                    l_var = np.argmin(value, axis=1)
                    absolute1 = abs(f_diff[np.arange(len(f_diff)), l_var])
                    draddiff = grad_diff[np.arange(len(grad_diff)), l_var].reshape(len(grad_diff), -1)
                    pow1 = (pow(np.linalg.norm(draddiff, axis=1, ord=2),2,)+ tol)
                    r_var = absolute1 / pow1
                    r_var = r_var.reshape((-1,) + (1,) * (len(x.shape) - 1))
                    r_var = r_var * grad_diff[np.arange(len(grad_diff)), l_var]
                elif self.norm == 1:
                    norm = np.linalg.norm(grad_diff.reshape(len(grad_diff), len(labels_set), -1), axis=2, ord=2) + tol
                    value = np.abs(f_diff) / norm
                    value[np.arange(len(value)), labels_indices] = np.inf
                    l_var = np.argmin(value, axis=1)
                    absolute1 = abs(f_diff[np.arange(len(f_diff)), l_var])
                    draddiff = grad_diff[np.arange(len(grad_diff)), l_var].reshape(len(grad_diff), -1)
                    pow1 = (pow(np.linalg.norm(draddiff, axis=1, ord=2),2,)+ tol)
                    r_var = absolute1 / pow1
                    r_var = r_var.reshape((-1,) + (1,) * (len(x.shape) - 1))
                    r_var = r_var * grad_diff[np.arange(len(grad_diff)), l_var]
                elif self.norm in ["inf", np.inf]:
                    norm = np.linalg.norm(grad_diff.reshape(len(grad_diff), len(labels_set), -1), axis=2, ord=1) + tol
                    value = np.abs(f_diff) / norm
                    value[np.arange(len(value)), labels_indices] = np.inf
                    l_var = np.argmin(value, axis=1)
                    absolute1 = abs(f_diff[np.arange(len(f_diff)), l_var])
                    draddiff = grad_diff[np.arange(len(grad_diff)), l_var].reshape(len(grad_diff), -1)
                    pow1 = (np.linalg.norm(draddiff, axis=1, ord=1)+ tol)
                    r_var = absolute1 / pow1
                    r_var = r_var.reshape((-1,) + (1,) * (len(x.shape) - 1))
                    r_var = r_var * np.sign(grad_diff[np.arange(len(grad_diff)), l_var])

                # 叠加以上得到的噪声并且将对抗样本值clip到合理的范围
                if self.estimator.clip_values is not None:
                    batch[active_indices] = np.clip(
                        batch[active_indices]
                        + r_var[active_indices] * (self.estimator.clip_values[1] - self.estimator.clip_values[0]),
                        self.estimator.clip_values[0],
                        self.estimator.clip_values[1],
                    )
                else:
                    batch[active_indices] += r_var[active_indices]

                # Recompute prediction for new x
                f_batch = self.estimator.predict(batch)
                fk_i_hat = np.argmax(f_batch, axis=1)

                # Recompute gradients for new x
                # if use_grads_subset:
                #     # Compute gradients only for (originally) top predicted classes
                #     grd = np.array([self.estimator.class_gradient(batch, label=int(label_i)) for label_i in labels_set])
                #     grd = np.squeeze(np.swapaxes(grd, 0, 2), axis=0)
                # else:
                #     # Compute gradients for all classes
                #     grd = self.estimator.class_gradient(batch)
                grd = self.estimator.class_gradient(batch)

                # 每次仅仅对还没有成功攻击的样本执行DeepFool算法
                active_indices = np.where(fk_i_hat == fk_hat)[0]

                current_step += 1

            # 使用eta参数是为了可以跨越到决策边界的另外一边
            x_adv1 = x_adv[batch_index_1:batch_index_2]
            x_adv2 = (1 + self.eta) * (batch - x_adv[batch_index_1:batch_index_2])
            x_adv[batch_index_1:batch_index_2] = x_adv1 + x_adv2
            if self.estimator.clip_values is not None:
                np.clip(
                    x_adv[batch_index_1:batch_index_2],
                    self.estimator.clip_values[0],
                    self.estimator.clip_values[1],
                    out=x_adv[batch_index_1:batch_index_2],
                )

        return x_adv

    # def _check_params(self) -> None:
    #     if not isinstance(self.max_iter, int) or self.max_iter <= 0:
    #         raise ValueError("The number of iterations must be a positive integer.")

    #     if not isinstance(self.nb_grads, int) or self.nb_grads <= 0:
    #         raise ValueError("The number of class gradients to compute must be a positive integer.")

    #     if self.eta < 0:
    #         raise ValueError("The overshoot parameter must not be negative.")

    #     if self.batch_size <= 0:
    #         raise ValueError("The batch size `batch_size` has to be positive.")

    #     if not isinstance(self.verbose, bool):
    #         raise ValueError("The argument `verbose` has to be of type bool.")
# import logging
from typing import List, Optional, Union, Tuple, TYPE_CHECKING
import numpy as np
from function.attack.attacks.config import MY_NUMPY_DTYPE
from function.attack.attacks.attack import EvasionAttack
from function.attack.attacks.evasion.auto_projected_gradient_descent import AutoProjectedGradientDescent
from function.attack.attacks.evasion.deepfool import DeepFool
from function.attack.attacks.evasion.square_attack import SquareAttack
from function.attack.estimators.estimator import BaseEstimator
from function.attack.estimators.classification.classifier import ClassifierMixin
from function.attack.attacks.utils import get_labels_np_array, check_and_transform_label_format
if TYPE_CHECKING:
    from function.attack.attacks.utils import CLASSIFIER_TYPE

# logger = logging.getLogger(__name__)

class AutoAttack(EvasionAttack):
    attack_params = EvasionAttack.attack_params + [
        "norm",
        "eps",
        "eps_step",
        # "attacks",
        "batch_size",
        # "estimator_orig",
        "targeted",
    ]

    _estimator_requirements = (BaseEstimator, ClassifierMixin)

    def __init__(
        self,
        estimator: "CLASSIFIER_TYPE",
        norm: Union[int, float, str] = np.inf,
        eps: float = 8/255,
        eps_step: float = 0.01,
        # attacks: Optional[List[EvasionAttack]] = None,
        batch_size: int = 128,
        # estimator_orig: Optional["CLASSIFIER_TYPE"] = None,
        targeted: bool = False,
    ):
        super().__init__(estimator=estimator)
        
        attacks = []
        attacks.append(
            AutoProjectedGradientDescent(
                estimator=estimator,
                norm=norm,
                eps=eps,
                eps_step=eps_step,
                max_iter=1,
                targeted=targeted,
                nb_random_init=5,
                batch_size=batch_size,
                loss_type="cross_entropy",
            )
        )
        attacks.append(
            AutoProjectedGradientDescent(
                estimator=estimator,
                norm=norm,
                eps=eps,
                eps_step=eps_step,
                max_iter=1,
                targeted=targeted,
                nb_random_init=5,
                batch_size=batch_size,
                loss_type="difference_logits_ratio",
            )
        )
        attacks.append(
            DeepFool(
                classifier=estimator,
                max_iter=100,
                # nb_grads=10,
                batch_size=batch_size,
            )
        )
        attacks.append(
            SquareAttack(estimator=estimator,norm=norm,targeted=targeted)
        )

        self.norm = norm
        self.eps = eps
        self.eps_step = eps_step
        self.attacks = attacks
        self.batch_size = batch_size
        self._targeted = targeted

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        x_adv = x.astype(MY_NUMPY_DTYPE)
        if y is not None:
            y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes)
        else:
            y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))
        
        y_pred = self.estimator.predict(x.astype(MY_NUMPY_DTYPE))
        sample_is_robust = np.argmax(y_pred, axis=1) == np.argmax(y, axis=1)

        if self.targeted:
            for attack in self.attacks:
                if np.sum(sample_is_robust) == 0:
                    break
                setattr(attack, "targeted", True)
                x_adv, sample_is_robust = self._run_attack(
                    x=x_adv,
                    y=y,
                    sample_is_robust=sample_is_robust,
                    attack=attack,
                    **kwargs,
                )
        else:
            y_t = np.array([range(y.shape[1])] * y.shape[0])
            y_idx = np.argmax(y, axis=1)
            y_idx = np.expand_dims(y_idx, 1)
            y_t = y_t[y_t != y_idx]
            targeted_labels = np.reshape(y_t, (y.shape[0], -1))

            for attack in self.attacks:
                setattr(attack, "targeted", False)
                for i in range(self.estimator.nb_classes - 1):
                    if np.sum(sample_is_robust) == 0:
                        break
                    target = check_and_transform_label_format(
                        targeted_labels[:, i], nb_classes=self.estimator.nb_classes
                    )
                    x_adv, sample_is_robust = self._run_attack(
                        x=x_adv,
                        y=target,
                        sample_is_robust=sample_is_robust,
                        attack=attack,
                        **kwargs,
                    )
        return x_adv
    
    def _run_attack(
        self,
        x: np.ndarray,
        y: np.ndarray,
        sample_is_robust: np.ndarray,
        attack: EvasionAttack,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        x_robust = x[sample_is_robust]
        y_robust = y[sample_is_robust]
        x_robust_adv = attack.generate(x=x_robust, y=y_robust, **kwargs)
        y_pred_robust_adv = self.estimator.predict(x_robust_adv)
        rel_acc = 1e-4
        order = np.inf if self.norm == "inf" else self.norm
        norm_is_smaller_eps = (1 - rel_acc) * np.linalg.norm(
            (x_robust_adv - x_robust).reshape((x_robust_adv.shape[0], -1)), axis=1, ord=order
        ) <= self.eps
        if attack.targeted:
            samples_misclassified = np.argmax(y_pred_robust_adv, axis=1) == np.argmax(y_robust, axis=1)
        else:
            samples_misclassified = np.argmax(y_pred_robust_adv, axis=1) != np.argmax(y_robust, axis=1)
        sample_is_not_robust = np.logical_and(samples_misclassified, norm_is_smaller_eps)
        x_robust[sample_is_not_robust] = x_robust_adv[sample_is_not_robust]
        x[sample_is_robust] = x_robust
        sample_is_robust[sample_is_robust] = np.invert(sample_is_not_robust)
        return x, sample_is_robust
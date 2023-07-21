from __future__ import absolute_import, division, print_function, unicode_literals
# import logging
from typing import Union, TYPE_CHECKING
import numpy as np
from function.attack.attacks.evasion.projected_gradient_descent.projected_gradient_descent import ProjectedGradientDescent
if TYPE_CHECKING:
    from function.attack.attacks.utils import CLASSIFIER_LOSS_GRADIENTS_TYPE

# 基于FGSM的基本迭代方法，与PGD的主要区别在于其没有随机开始，即num_random_init默认为0
class BasicIterativeMethod(ProjectedGradientDescent):
    attack_params = ProjectedGradientDescent.attack_params

    def __init__(
        self,
        estimator: "CLASSIFIER_LOSS_GRADIENTS_TYPE",
        eps: Union[int, float, np.ndarray] = 8/255,
        eps_step: Union[int, float, np.ndarray] = 0.01,
        max_iter: int = 20,
        targeted: bool = False,
        batch_size: int = 128,
        # verbose: bool = True,
    ) -> None:
        super().__init__(
            estimator=estimator,
            norm=np.inf,
            eps=eps,
            eps_step=eps_step,
            max_iter=max_iter,
            targeted=targeted,
            num_random_init=0,
            batch_size=batch_size,
            # verbose=verbose,
        )
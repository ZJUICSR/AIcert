from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from typing import Union, TYPE_CHECKING
import numpy as np
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import ProjectedGradientDescent
if TYPE_CHECKING:
    from art.utils import CLASSIFIER_LOSS_GRADIENTS_TYPE
logger = logging.getLogger(__name__)


class BasicIterativeMethod(ProjectedGradientDescent):
    attack_params = ProjectedGradientDescent.attack_params

    def __init__(
        self,
        estimator: "CLASSIFIER_LOSS_GRADIENTS_TYPE",
        eps: Union[int, float, np.ndarray] = 0.3,
        eps_step: Union[int, float, np.ndarray] = 0.03,
        max_iter: int = 20,
        targeted: bool = False,
        batch_size: int = 32,
        verbose: bool = True,
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
            verbose=verbose,
        )
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from typing import Optional, Union, TYPE_CHECKING
import numpy as np
from art.estimators.classification.pytorch import PyTorchClassifier
from art.estimators.classification.tensorflow import TensorFlowV2Classifier
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.attacks.attack import EvasionAttack
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent_numpy import (
    ProjectedGradientDescentNumpy,
)
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent_pytorch import (
    ProjectedGradientDescentPyTorch,
)
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent_tensorflow_v2 import (
    ProjectedGradientDescentTensorFlowV2,
)
from art.summary_writer import SummaryWriter

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_LOSS_GRADIENTS_TYPE, OBJECT_DETECTOR_TYPE

logger = logging.getLogger(__name__)


class ProjectedGradientDescent(EvasionAttack):
    attack_params = EvasionAttack.attack_params + [
        "norm",
        "eps",
        "eps_step",
        "decay",
        "targeted",
        "num_random_init",
        "batch_size",
        "max_iter",
        "random_eps",
        "summary_writer",
        "verbose",
    ]

    _estimator_requirements = (BaseEstimator, LossGradientsMixin)

    def __init__(
        self,
        estimator: Union["CLASSIFIER_LOSS_GRADIENTS_TYPE", "OBJECT_DETECTOR_TYPE"],
        norm: Union[int, float, str] = np.inf,
        eps: Union[int, float, np.ndarray] = 0.3,
        eps_step: Union[int, float, np.ndarray] = 0.03,
        decay: Optional[float] = None,
        max_iter: int = 20,
        targeted: bool = False,
        num_random_init: int = 0,
        batch_size: int = 4096,
        random_eps: bool = False,
        summary_writer: Union[str, bool, SummaryWriter] = False,
        verbose: bool = True,
    ):
        super().__init__(estimator=estimator, summary_writer=False)

        self.norm = norm
        self.eps = eps
        self.eps_step = eps_step
        self.max_iter = max_iter
        self.targeted = targeted
        self.num_random_init = num_random_init
        self.batch_size = batch_size
        self.random_eps = random_eps
        self.verbose = verbose
        self.decay = decay
        ProjectedGradientDescent._check_params(self)

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        self._attack: Union[
            ProjectedGradientDescentPyTorch, ProjectedGradientDescentTensorFlowV2, ProjectedGradientDescentNumpy
        ]
        if isinstance(self.estimator, PyTorchClassifier) and self.estimator.all_framework_preprocessing:
            self._attack = ProjectedGradientDescentPyTorch(
                estimator=self.estimator,  # type: ignore
                norm=self.norm,
                eps=self.eps,
                eps_step=self.eps_step,
                decay=self.decay,
                max_iter=self.max_iter,
                targeted=self.targeted,
                num_random_init=self.num_random_init,
                batch_size=self.batch_size,
                random_eps=self.random_eps,
                summary_writer=self.summary_writer,
                verbose=self.verbose,
            )
        logger.info("Creating adversarial samples.")
        return self._attack.generate(x=x, y=y, **kwargs)

    # @property
    # def summary_writer(self):
    #     """The summary writer."""
    #     return self._attack.summary_writer

    def set_params(self, **kwargs) -> None:
        super().set_params(**kwargs)
        self._attack.set_params(**kwargs)

    def _check_params(self) -> None:

        if self.norm not in [1, 2, np.inf, "inf"]:
            raise ValueError('Norm order must be either 1, 2, `np.inf` or "inf".')

        if not (
            isinstance(self.eps, (int, float))
            and isinstance(self.eps_step, (int, float))
            or isinstance(self.eps, np.ndarray)
            and isinstance(self.eps_step, np.ndarray)
        ):
            raise TypeError(
                "The perturbation size `eps` and the perturbation step-size `eps_step` must have the same type of `int`"
                ", `float`, or `np.ndarray`."
            )

        if isinstance(self.eps, (int, float)):
            if self.eps < 0:
                raise ValueError("The perturbation size `eps` has to be nonnegative.")
        else:
            if (self.eps < 0).any():
                raise ValueError("The perturbation size `eps` has to be nonnegative.")

        if isinstance(self.eps_step, (int, float)):
            if self.eps_step <= 0:
                raise ValueError("The perturbation step-size `eps_step` has to be positive.")
        else:
            if (self.eps_step <= 0).any():
                raise ValueError("The perturbation step-size `eps_step` has to be positive.")

        if isinstance(self.eps, np.ndarray) and isinstance(self.eps_step, np.ndarray):
            if self.eps.shape != self.eps_step.shape:
                raise ValueError(
                    "The perturbation size `eps` and the perturbation step-size `eps_step` must have the same shape."
                )

        if not isinstance(self.targeted, bool):
            raise ValueError("The flag `targeted` has to be of type bool.")

        if not isinstance(self.num_random_init, int):
            raise TypeError("The number of random initialisations has to be of type integer.")

        if self.num_random_init < 0:
            raise ValueError("The number of random initialisations `random_init` has to be greater than or equal to 0.")

        if self.batch_size <= 0:
            raise ValueError("The batch size `batch_size` has to be positive.")

        if self.max_iter < 0:
            raise ValueError("The number of iterations `max_iter` has to be a nonnegative integer.")

        if not isinstance(self.verbose, bool):
            raise ValueError("The verbose has to be a Boolean.")
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from typing import Callable, List, Optional, Tuple, Union
import numpy as np
from art.attacks.attack import PoisoningAttackBlackBox
logger = logging.getLogger(__name__)

class PoisoningAttackBackdoor(PoisoningAttackBlackBox):
    """
    最基本的后门攻击方法
    """
    attack_params = PoisoningAttackBlackBox.attack_params + ["perturbation"]
    _estimator_requirements = ()

    def __init__(self, perturbation: Union[Callable, List[Callable]]) -> None:
        super().__init__()
        self.perturbation = perturbation
        self._check_params()

    def poison (
        self, x: np.ndarray, y: Optional[np.ndarray] = None, broadcast=True, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        if y is None:  # pragma: no cover
            raise ValueError("Target labels `y` need to be provided for a targeted attack.")

        if broadcast:
            y_attack = np.broadcast_to(y, (x.shape[0], y.shape[0]))
        else:
            y_attack = np.copy(y)

        num_poison = len(x)
        if num_poison == 0:  # pragma: no cover
            raise ValueError("Must input at least one poison point.")
        poisoned = np.copy(x)

        if callable(self.perturbation):
            return self.perturbation(poisoned), y_attack

        for perturb in self.perturbation:
            poisoned = perturb(poisoned)

        return poisoned, y_attack
    # 参数检查
    def _check_params(self) -> None:
        if not (callable(self.perturbation) or all((callable(perturb) for perturb in self.perturbation))):
            raise ValueError("Perturbation must be a function or a list of functions.")
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from typing import Optional, Tuple, TYPE_CHECKING, Union
import numpy as np
from art.attacks.attack import PoisoningAttackBlackBox
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import ProjectedGradientDescent
from art.attacks.poisoning.backdoor_attack import PoisoningAttackBackdoor

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_LOSS_GRADIENTS_TYPE


logger = logging.getLogger(__name__)


class PoisoningAttackCleanLabelBackdoor(PoisoningAttackBlackBox):
    attack_params = PoisoningAttackBlackBox.attack_params + [
        "backdoor",
        "proxy_classifier",
        "target",
        "pp_poison",
        "norm",
        "eps",
        "eps_step",
        "max_iter",
        "num_random_init",
        "batch_size"
    ]
    _estimator_requirements = ()

    def __init__(
        self,
        backdoor: PoisoningAttackBackdoor,
        proxy_classifier: "CLASSIFIER_LOSS_GRADIENTS_TYPE",
        target: np.ndarray,
        pp_poison: float = 0.33,
        norm: Union[int, float, str] = np.inf,
        eps: float = 0.3,
        eps_step: float = 0.03,
        max_iter: int = 10,
        num_random_init: int = 1,
        batch_size = 256
    ) -> None:
        super().__init__()
        self.backdoor = backdoor
        self.proxy_classifier = proxy_classifier
        self.target = target
        self.pp_poison = pp_poison
        self.attack = ProjectedGradientDescent (
            proxy_classifier,
            norm=norm,
            eps=eps,
            eps_step=eps_step,
            max_iter=max_iter,
            targeted=False,
            num_random_init=num_random_init,
            batch_size=batch_size,
        )
        self._check_params()

    def poison(
        self, x: np.ndarray, y: Optional[np.ndarray] = None, broadcast: bool = True, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        data = np.copy(x)
        estimated_labels = self.proxy_classifier.predict(data) if y is None else np.copy(y)

        # Selected target indices to poison
        all_indices = np.arange(len(data))
        target_indices = all_indices[np.all(estimated_labels == self.target, axis=1)]
        num_poison = int(self.pp_poison * len(target_indices))
        selected_indices = np.random.choice(target_indices, num_poison)

        # Run untargeted PGD on selected points, making it hard to classify correctly
        perturbed_input = self.attack.generate(data[selected_indices])
        no_change_detected = np.array(
            [
                np.all(data[selected_indices][poison_idx] == perturbed_input[poison_idx])
                for poison_idx in range(len(perturbed_input))
            ]
        )

        if any(no_change_detected):  # pragma: no cover
            logger.warning("Perturbed input is the same as original data after PGD. Check params.")
            idx_no_change = np.arange(len(no_change_detected))[no_change_detected]
            logger.warning("%d indices without change: %s", len(idx_no_change), idx_no_change)

        # Add backdoor and poison with the same label
        poisoned_input, _ = self.backdoor.poison(perturbed_input, self.target, broadcast=broadcast)
        data[selected_indices] = poisoned_input
        return data, estimated_labels, selected_indices

    def _check_params(self) -> None:
        if not isinstance(self.backdoor, PoisoningAttackBackdoor):
            raise ValueError("Backdoor must be of type PoisoningAttackBackdoor")
        if not isinstance(self.attack, ProjectedGradientDescent):
            raise ValueError("There was an issue creating the PGD attack")
        if not 0 < self.pp_poison < 1:
            raise ValueError("pp_poison must be between 0 and 1")
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import random
import types
from typing import Any, Dict, Optional, Union, TYPE_CHECKING
import numpy as np
from tqdm.auto import tqdm
from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin
from art.utils import projection, get_labels_np_array, check_and_transform_label_format
if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE
logger = logging.getLogger(__name__)


class UniversalPerturbation(EvasionAttack):
    attacks_dict = {
        "carlini": "art.attacks.evasion.carlini.CarliniL2Method",
        "carlini_inf": "art.attacks.evasion.carlini.CarliniLInfMethod",
        "deepfool": "art.attacks.evasion.deepfool.DeepFool",
        "ead": "art.attacks.evasion.elastic_net.ElasticNet",
        "fgsm": "art.attacks.evasion.fast_gradient.FastGradientMethod",
        "bim": "art.attacks.evasion.iterative_method.BasicIterativeMethod",
        "pgd": "art.attacks.evasion.projected_gradient_descent.projected_gradient_descent.ProjectedGradientDescent",
        "newtonfool": "art.attacks.evasion.newtonfool.NewtonFool",
        "jsma": "art.attacks.evasion.saliency_map.SaliencyMapMethod",
        "vat": "art.attacks.evasion.virtual_adversarial.VirtualAdversarialMethod",
        "simba": "art.attacks.evasion.simba.SimBA",
    }
    attack_params = EvasionAttack.attack_params + [
        "attacker",
        "attacker_params",
        "delta",
        "max_iter",
        "eps",
        "norm",
        "batch_size",
        "verbose",
    ]
    _estimator_requirements = (BaseEstimator, ClassifierMixin)

    def __init__(
        self,
        classifier: "CLASSIFIER_TYPE",
        attacker: str = "pgd",
        attacker_params: Optional[Dict[str, Any]] = None,
        delta: float = 0.2,
        max_iter: int = 10,
        eps: float = 10.0,
        norm: Union[int, float, str] = np.inf,
        batch_size: int = 100,
        verbose: bool = True,
    ) -> None:
        super().__init__(estimator=classifier)
        self.attacker = attacker
        self.attacker_params = attacker_params
        self.delta = delta
        self.max_iter = max_iter
        self.eps = eps
        self.norm = norm
        self.batch_size = batch_size
        self.verbose = verbose
        self._check_params()

        # Attack properties
        self._fooling_rate: Optional[float] = None
        self._converged: Optional[bool] = None
        self._noise: Optional[np.ndarray] = None

    @property
    def fooling_rate(self) -> Optional[float]:
        return self._fooling_rate

    @property
    def converged(self) -> Optional[bool]:
        return self._converged

    @property
    def noise(self) -> Optional[np.ndarray]:
        return self._noise

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        logger.info("Computing universal perturbation based on %s attack.", self.attacker)

        if y is not None:
            y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes)

        if y is None:
            # Use model predictions as true labels
            logger.info("Using model predictions as true labels.")
            y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))

        if self.estimator.nb_classes == 2 and y.shape[1] == 1:
            raise ValueError(
                "This attack has not yet been tested for binary classification with a single output classifier."
            )

        y_index = np.argmax(y, axis=1)

        # Init universal perturbation
        noise = np.zeros_like(x[[0]])
        fooling_rate = 0.0
        nb_instances = len(x)

        # Instantiate the middle attacker
        attacker = self._get_attack(self.attacker, self.attacker_params)

        # Generate the adversarial examples
        nb_iter = 0
        pbar = tqdm(total=self.max_iter, desc="Universal perturbation", disable=not self.verbose)

        while fooling_rate < 1.0 - self.delta and nb_iter < self.max_iter:
            # Go through all the examples randomly
            rnd_idx = random.sample(range(nb_instances), nb_instances)

            # Go through the data set and compute the perturbation increments sequentially
            for j, ex in enumerate(x[rnd_idx]):
                x_i = ex[None, ...]

                current_label = np.argmax(self.estimator.predict(x_i + noise)[0])
                original_label = y_index[rnd_idx][j]

                if current_label == original_label:
                    # Compute adversarial perturbation
                    adv_xi = attacker.generate(x_i + noise, y=y[rnd_idx][[j]])
                    new_label = np.argmax(self.estimator.predict(adv_xi)[0])

                    # If the class has changed, update v
                    if current_label != new_label:
                        noise = adv_xi - x_i

                        # Project on L_p ball
                        noise = projection(noise, self.eps, self.norm)
            nb_iter += 1
            pbar.update(1)

            # Apply attack and clip
            x_adv = x + noise
            if self.estimator.clip_values is not None:
                clip_min, clip_max = self.estimator.clip_values
                x_adv = np.clip(x_adv, clip_min, clip_max)

            # Compute the error rate
            y_adv = np.argmax(self.estimator.predict(x_adv, batch_size=1), axis=1)
            fooling_rate = np.sum(y_index != y_adv) / nb_instances

        pbar.close()
        self._fooling_rate = fooling_rate
        self._converged = nb_iter < self.max_iter
        self._noise = noise
        logger.info("Success rate of universal perturbation attack: %.2f%%", 100 * fooling_rate)

        return x_adv

    def _get_attack(self, a_name: str, params: Optional[Dict[str, Any]] = None) -> EvasionAttack:
        try:
            attack_class = self._get_class(self.attacks_dict[a_name])
            a_instance = attack_class(self.estimator)  # type: ignore

            if params:
                a_instance.set_params(**params)

            return a_instance
        except KeyError:
            raise NotImplementedError(f"{a_name} attack not supported") from KeyError

    @staticmethod
    def _get_class(class_name: str) -> types.ModuleType:
        sub_mods = class_name.split(".")
        module_ = __import__(".".join(sub_mods[:-1]), fromlist=sub_mods[-1])
        class_module = getattr(module_, sub_mods[-1])

        return class_module

    def _check_params(self) -> None:
        if not isinstance(self.delta, (float, int)) or self.delta < 0 or self.delta > 1:
            raise ValueError("The desired accuracy must be in the range [0, 1].")

        if not isinstance(self.max_iter, int) or self.max_iter <= 0:
            raise ValueError("The number of iterations must be a positive integer.")

        if not isinstance(self.eps, (float, int)) or self.eps <= 0:
            raise ValueError("The eps coefficient must be a positive float.")

        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("The batch_size must be a positive integer.")

        if not isinstance(self.verbose, bool):
            raise ValueError("The argument `verbose` has to be of type bool.")
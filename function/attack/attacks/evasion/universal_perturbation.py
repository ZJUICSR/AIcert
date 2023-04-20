from __future__ import absolute_import, division, print_function, unicode_literals
import random
import types
from typing import Any, Dict, Optional, Union, TYPE_CHECKING
import numpy as np
from tqdm.auto import tqdm
from function.attack.attacks.attack import EvasionAttack
from function.attack.estimators.estimator import BaseEstimator
from function.attack.estimators.classification.classifier import ClassifierMixin
from function.attack.attacks.utils import projection, get_labels_np_array, check_and_transform_label_format
if TYPE_CHECKING:
    from function.attack.attacks.utils import CLASSIFIER_TYPE


class UniversalPerturbation(EvasionAttack):
    attacks_dict = {
        "FGSM": "attacks.evasion.fast_gradient.FastGradientMethod",
        "BIM": "attacks.evasion.iterative_method.BasicIterativeMethod",
        "PGD": "attacks.evasion.projected_gradient_descent.projected_gradient_descent.ProjectedGradientDescent",
        "DeepFool": "attacks.evasion.deepfool.DeepFool",
        "JSMA": "attacks.evasion.saliency_map.SaliencyMapMethod",
        "Carlini_l2": "attacks.evasion.carlini.CarliniL2Method",
        "Carlini_inf": "attacks.evasion.carlini.CarliniLInfMethod",
        "Simba": "attacks.evasion.simba.SimBA",
    }
    attack_params = EvasionAttack.attack_params + [
        # "attacker",
        # "attacker_params",
        "max_iter",
        "eps",
        "norm",
        "batch_size",
    ]
    _estimator_requirements = (BaseEstimator, ClassifierMixin)

    def __init__(
        self,
        classifier: "CLASSIFIER_TYPE",
        attacker: str = "PGD",
        attacker_params: Optional[Dict[str, Any]] = None,
        max_iter: int = 20,
        eps: float = 20/255,
        norm: Union[int, float, str] = np.inf,
        batch_size: int = 128,
        # verbose: bool = True,
    ) -> None:
        super().__init__(estimator=classifier)
        self.attacker = attacker
        self.attacker_params = attacker_params
        self.max_iter = max_iter
        self.eps = eps
        self.norm = norm
        self.batch_size = batch_size
        # self.verbose = verbose
        # self._check_params()
        self.delta = 0
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
        # logger.info("Computing universal perturbation based on %s attack.", self.attacker)

        if y is not None:
            y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes)

        if y is None:
            # Use model predictions as true labels
            # logger.info("Using model predictions as true labels.")
            y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))

        # if self.estimator.nb_classes == 2 and y.shape[1] == 1:
        #     raise ValueError(
        #         "This attack has not yet been tested for binary classification with a single output classifier."
        #     )

        y_index = np.argmax(y, axis=1)

        # Init universal perturbation
        noise = np.zeros_like(x[[0]])
        fooling_rate = 0.0
        nb_instances = len(x)

        # Instantiate the middle attacker
        attacker = self._get_attack(self.attacker, self.attacker_params)

        # Generate the adversarial examples
        nb_iter = 0
        pbar = tqdm(total=self.max_iter, desc="Universal perturbation")

        while fooling_rate < 1.0 - self.delta and nb_iter < self.max_iter:
            # Go through all the examples randomly
            rnd_idx = random.sample(range(nb_instances), nb_instances)

            x_i = x[rnd_idx]
            current_label = np.argmax(self.estimator.predict(x_i + noise), axis=1)
            original_label = y_index[rnd_idx]

            right_index = np.where(current_label == original_label)
            if len(right_index) != 0:
                adv_xi = attacker.generate(x_i + noise, y=original_label)
                new_label = np.argmax(self.estimator.predict(adv_xi), axis=1)
                change_index = np.where(new_label != current_label)
                if len(change_index) != 0:
                    noise = np.sum(adv_xi - x_i, axis=0)
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
        # logger.info("Success rate of universal perturbation attack: %.2f%%", 100 * fooling_rate)
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
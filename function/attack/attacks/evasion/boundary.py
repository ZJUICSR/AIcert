from __future__ import absolute_import, division, print_function, unicode_literals
from typing import Optional, Tuple, TYPE_CHECKING
import numpy as np
from tqdm.auto import tqdm
from function.attack.attacks.attack import EvasionAttack
from function.attack.attacks.config import MY_NUMPY_DTYPE
from function.attack.estimators.estimator import BaseEstimator
from function.attack.estimators.classification.classifier import ClassifierMixin
from function.attack.attacks.utils import to_categorical, check_and_transform_label_format, get_labels_np_array
if TYPE_CHECKING:
    from function.attack.attacks.utils import CLASSIFIER_TYPE

class BoundaryAttack(EvasionAttack):
    attack_params = EvasionAttack.attack_params + [
        "targeted",
        "delta",
        "eps",
        "step_adapt",
        "max_iter",
        "num_trial",
        "sample_size",
        "init_size",
        "batch_size",
    ]

    _estimator_requirements = (BaseEstimator, ClassifierMixin)

    def __init__(
        self,
        estimator: "CLASSIFIER_TYPE",
        batch_size: int = 64,
        targeted: bool = False,
        delta: float = 0.01,
        eps: float = 0.01,
        step_adapt: float = 0.667,
        max_iter: int = 1000,
        num_trial: int = 25,
        sample_size: int = 20,
        init_size: int = 100,
        min_epsilon: float = 0.0,
    ) -> None:
        super().__init__(estimator=estimator)

        self._targeted = targeted
        self.delta = delta
        self.epsilon = eps
        self.step_adapt = step_adapt
        self.max_iter = max_iter
        self.num_trial = num_trial
        self.sample_size = sample_size
        self.init_size = init_size
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.curr_adv: Optional[np.ndarray] = None

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        if y is None:
            if self.targeted:  # pragma: no cover
                raise ValueError("Target labels `y` need to be provided for a targeted attack.")

            # Use model predictions as correct outputs
            y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))  # type: ignore

        y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes, return_one_hot=False)

        if y is not None and self.estimator.nb_classes == 2 and y.shape[1] == 1:
            raise ValueError(  # pragma: no cover
                "This attack has not yet been tested for binary classification with a single output classifier."
            )

        # Get clip_min and clip_max from the classifier or infer them from data
        if self.estimator.clip_values is not None:
            clip_min, clip_max = self.estimator.clip_values
        else:
            clip_min, clip_max = np.min(x), np.max(x)

        # Prediction from the original images
        preds = np.argmax(self.estimator.predict(x, batch_size=self.batch_size), axis=1)

        # Prediction from the initial adversarial examples if not None
        x_adv_init = kwargs.get("x_adv_init")

        if x_adv_init is not None:
            init_preds = np.argmax(self.estimator.predict(x_adv_init, batch_size=self.batch_size), axis=1)
        else:
            init_preds = [None] * len(x)
            x_adv_init = [None] * len(x)

        # Assert that, if attack is targeted, y is provided
        if self.targeted and y is None:  # pragma: no cover
            raise ValueError("Target labels `y` need to be provided for a targeted attack.")

        # Some initial setups
        x_adv = x.astype(MY_NUMPY_DTYPE)
        print(len(x_adv))

        # Generate the adversarial samples
        for ind, val in enumerate(tqdm(x_adv, desc="Boundary attack")):
            if self.targeted:
                x_adv[ind] = self._perturb(
                    x=val,
                    y=y[ind],
                    y_p=preds[ind],
                    init_pred=init_preds[ind],
                    adv_init=x_adv_init[ind],
                    clip_min=clip_min,
                    clip_max=clip_max,
                )
            else:
                x_adv[ind] = self._perturb(
                    x=val,
                    y=-1,
                    y_p=preds[ind],
                    init_pred=init_preds[ind],
                    adv_init=x_adv_init[ind],
                    clip_min=clip_min,
                    clip_max=clip_max,
                )

        y = to_categorical(y, self.estimator.nb_classes)
        return x_adv

    def _perturb(
        self,
        x: np.ndarray,
        y: int,
        y_p: int,
        init_pred: int,
        adv_init: np.ndarray,
        clip_min: float,
        clip_max: float,
    ) -> np.ndarray:
        # First, create an initial adversarial sample
        initial_sample = self._init_sample(x, y, y_p, init_pred, adv_init, clip_min, clip_max)

        # If an initial adversarial example is not found, then return the original image
        if initial_sample is None:
            return x

        # If an initial adversarial example found, then go with boundary attack
        x_adv = self._attack(
            initial_sample[0],
            x,
            y_p,
            initial_sample[1],
            self.delta,
            self.epsilon,
            clip_min,
            clip_max,
        )

        return x_adv

    def _attack(
        self,
        initial_sample: np.ndarray,
        original_sample: np.ndarray,
        y_p: int,
        target: int,
        initial_delta: float,
        initial_epsilon: float,
        clip_min: float,
        clip_max: float,
    ) -> np.ndarray:
        # Get initialization for some variables
        x_adv = initial_sample
        self.curr_delta = initial_delta
        self.curr_epsilon = initial_epsilon

        self.curr_adv = x_adv

        # Main loop to wander around the boundary
        for _ in range(self.max_iter):
            # Trust region method to adjust delta
            for _ in range(self.num_trial):
                potential_advs = []
                for _ in range(self.sample_size):
                    potential_adv = x_adv + self._orthogonal_perturb(self.curr_delta, x_adv, original_sample)
                    potential_adv = np.clip(potential_adv, clip_min, clip_max)
                    potential_advs.append(potential_adv)

                preds = np.argmax(
                    self.estimator.predict(np.array(potential_advs), batch_size=self.batch_size),
                    axis=1,
                )

                if self.targeted:
                    satisfied = preds == target
                else:
                    satisfied = preds != y_p

                delta_ratio = np.mean(satisfied)

                if delta_ratio < 0.2:
                    self.curr_delta *= self.step_adapt
                elif delta_ratio > 0.5:
                    self.curr_delta /= self.step_adapt

                if delta_ratio > 0:
                    x_advs = np.array(potential_advs)[np.where(satisfied)[0]]
                    break
            else:  # pragma: no cover
                # logger.warning("Adversarial example found but not optimal.")
                return x_adv

            # Trust region method to adjust epsilon
            for _ in range(self.num_trial):
                perturb = np.repeat(np.array([original_sample]), len(x_advs), axis=0) - x_advs
                perturb *= self.curr_epsilon
                potential_advs = x_advs + perturb
                potential_advs = np.clip(potential_advs, clip_min, clip_max)
                preds = np.argmax(
                    self.estimator.predict(potential_advs, batch_size=self.batch_size),
                    axis=1,
                )

                if self.targeted:
                    satisfied = preds == target
                else:
                    satisfied = preds != y_p

                epsilon_ratio = np.mean(satisfied)

                if epsilon_ratio < 0.2:
                    self.curr_epsilon *= self.step_adapt
                elif epsilon_ratio > 0.5:
                    self.curr_epsilon /= self.step_adapt

                if epsilon_ratio > 0:
                    x_adv = self._best_adv(original_sample, potential_advs[np.where(satisfied)[0]])
                    self.curr_adv = x_adv
                    break
            else:  # pragma: no cover
                # logger.warning("Adversarial example found but not optimal.")
                return self._best_adv(original_sample, x_advs)

            if self.curr_epsilon < self.min_epsilon:
                return x_adv

        return x_adv

    def _orthogonal_perturb(self, delta: float, current_sample: np.ndarray, original_sample: np.ndarray) -> np.ndarray:
        # Generate perturbation randomly
        perturb = np.random.randn(*self.estimator.input_shape).astype(MY_NUMPY_DTYPE)

        # Rescale the perturbation
        perturb /= np.linalg.norm(perturb)
        perturb *= delta * np.linalg.norm(original_sample - current_sample)

        # Project the perturbation onto sphere
        direction = original_sample - current_sample

        direction_flat = direction.flatten()
        perturb_flat = perturb.flatten()

        direction_flat /= np.linalg.norm(direction_flat)
        perturb_flat -= np.dot(perturb_flat, direction_flat.T) * direction_flat
        perturb = perturb_flat.reshape(self.estimator.input_shape)

        hypotenuse = np.sqrt(1 + delta ** 2)
        perturb = ((1 - hypotenuse) * (current_sample - original_sample) + perturb) / hypotenuse
        return perturb

    def _init_sample(
        self,
        x: np.ndarray,
        y: int,
        y_p: int,
        init_pred: int,
        adv_init: np.ndarray,
        clip_min: float,
        clip_max: float,
    ) -> Optional[Tuple[np.ndarray, int]]:
        nprd = np.random.RandomState()
        initial_sample = None

        if self.targeted:
            # Attack satisfied
            if y == y_p:
                return None

            # Attack unsatisfied yet and the initial image satisfied
            if adv_init is not None and init_pred == y:
                return adv_init.astype(MY_NUMPY_DTYPE), init_pred

            # Attack unsatisfied yet and the initial image unsatisfied
            for _ in range(self.init_size):
                random_img = nprd.uniform(clip_min, clip_max, size=x.shape).astype(x.dtype)
                random_class = np.argmax(
                    self.estimator.predict(np.array([random_img]), batch_size=self.batch_size),
                    axis=1,
                )[0]

                if random_class == y:
                    initial_sample = random_img, random_class

                    # logger.info("Found initial adversarial image for targeted attack.")
                    break
            else:
                pass
                # logger.warning("Failed to draw a random image that is adversarial, attack failed.")

        else:
            # The initial image satisfied
            if adv_init is not None and init_pred != y_p:
                return adv_init.astype(MY_NUMPY_DTYPE), init_pred

            # The initial image unsatisfied
            for _ in range(self.init_size):
                random_img = nprd.uniform(clip_min, clip_max, size=x.shape).astype(x.dtype)
                random_class = np.argmax(
                    self.estimator.predict(np.array([random_img]), batch_size=self.batch_size),
                    axis=1,
                )[0]

                if random_class != y_p:
                    initial_sample = random_img, random_class

                    # logger.info("Found initial adversarial image for untargeted attack.")
                    break
            else:  # pragma: no cover
                # logger.warning("Failed to draw a random image that is adversarial, attack failed.")
                pass

        return initial_sample

    @staticmethod
    def _best_adv(original_sample: np.ndarray, potential_advs: np.ndarray) -> np.ndarray:
        shape = potential_advs.shape
        min_idx = np.linalg.norm(original_sample.flatten() - potential_advs.reshape(shape[0], -1), axis=1).argmin()
        return potential_advs[min_idx]
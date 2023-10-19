from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from typing import Optional, Union, TYPE_CHECKING
import numpy as np
from function.attack.estimators.classification.pytorch import PyTorchClassifier
from function.attack.estimators.classification.tensorflow import TensorFlowV2Classifier
from function.attack.estimators.estimator import BaseEstimator, LossGradientsMixin
from function.attack.attacks.attack import EvasionAttack
from function.attack.attacks.evasion.fast_gradient import FastGradientMethod
from function.attack.attacks.config import MY_NUMPY_DTYPE
from function.attack.attacks.utils import compute_success, get_labels_np_array, check_and_transform_label_format, compute_success_array, random_sphere
from scipy.stats import truncnorm
from tqdm.auto import tqdm
from function.attack.estimators.classification.classifier import ClassifierMixin

if TYPE_CHECKING:
    from function.attack.attacks.utils import CLASSIFIER_LOSS_GRADIENTS_TYPE, OBJECT_DETECTOR_TYPE
    import torch
    from function.attack.estimators.classification.pytorch import PyTorchClassifier

logger = logging.getLogger(__name__)

class ProjectedGradientDescentCommon():
    attack_params = FastGradientMethod.attack_params + ["max_iter", "random_eps", "verbose"]
    _estimator_requirements = (BaseEstimator, LossGradientsMixin)

    def __init__(
        self,
        estimator: Union["CLASSIFIER_LOSS_GRADIENTS_TYPE", "OBJECT_DETECTOR_TYPE"],
        norm: Union[int, float, str] = np.inf,
        eps: Union[int, float, np.ndarray] = 0.3,
        eps_step: Union[int, float, np.ndarray] = 0.1,
        decay: Optional[float] = None,
        max_iter: int = 100,
        targeted: bool = False,
        num_random_init: int = 0,
        batch_size: int = 32,
        random_eps: bool = False,
        verbose: bool = True,
    ) -> None:
        """
        Create a :class:`.ProjectedGradientDescentCommon` instance.

        :param estimator: A trained classifier.
        :param norm: The norm of the adversarial perturbation supporting "inf", np.inf, 1 or 2.
        :param eps: Maximum perturbation that the attacker can introduce.
        :param eps_step: Attack step size (input variation) at each iteration.
        :param random_eps: When True, epsilon is drawn randomly from truncated normal distribution. The literature
            suggests this for FGSM based training to generalize across different epsilons. eps_step is
            modified to preserve the ratio of eps / eps_step. The effectiveness of this method with PGD
            is untested (https://arxiv.org/pdf/1611.01236.pdf).
        :param decay: Decay factor for accumulating the velocity vector when using momentum.
        :param max_iter: The maximum number of iterations.
        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False).
        :param num_random_init: Number of random initialisations within the epsilon ball. For num_random_init=0
            starting at the original input.
        :param batch_size: Size of the batch on which adversarial samples are generated.
        :param summary_writer: Activate summary writer for TensorBoard.
                               Default is `False` and deactivated summary writer.
                               If `True` save runs/CURRENT_DATETIME_HOSTNAME in current directory.
                               If of type `str` save in path.
                               If of type `SummaryWriter` apply provided custom summary writer.
                               Use hierarchical folder structure to compare between runs easily. e.g. pass in
                               ‘runs/exp1’, ‘runs/exp2’, etc. for each new experiment to compare across them.
        :param verbose: Show progress bars.
        """
        # super().__init__(
        #     estimator=estimator,  # type: ignore
        #     norm=norm,
        #     eps=eps,
        #     eps_step=eps_step,
        #     targeted=targeted,
        #     num_random_init=num_random_init,
        #     batch_size=batch_size,
        #     minimal=False,
        #     summary_writer=summary_writer,
        # )
        # add
        self.norm = norm
        self.estimator = estimator
        self.eps = eps
        self.eps_step = eps_step
        self.targeted = targeted
        self.num_random_init=num_random_init
        self.batch_size = batch_size
        
        self.decay = decay
        self.max_iter = max_iter
        self.random_eps = random_eps
        self.verbose = verbose
        # ProjectedGradientDescentCommon._check_params(self)

        lower: Union[int, float, np.ndarray]
        upper: Union[int, float, np.ndarray]
        var_mu: Union[int, float, np.ndarray]
        sigma: Union[int, float, np.ndarray]

        if self.random_eps:
            if isinstance(eps, (int, float)):
                lower, upper = 0, eps
                var_mu, sigma = 0, (eps / 2)
            else:
                lower, upper = np.zeros_like(eps), eps
                var_mu, sigma = np.zeros_like(eps), (eps / 2)

            self.norm_dist = truncnorm((lower - var_mu) / sigma, (upper - var_mu) / sigma, loc=var_mu, scale=sigma)

    def _random_eps(self):
        """
        Check whether random eps is enabled, then scale eps and eps_step appropriately.
        """
        if self.random_eps:
            ratio = self.eps_step / self.eps

            if isinstance(self.eps, (int, float)):
                self.eps = np.round(self.norm_dist.rvs(1)[0], 10)
            else:
                self.eps = np.round(self.norm_dist.rvs(size=self.eps.shape), 10)

            self.eps_step = ratio * self.eps

    def _set_targets(self, x: np.ndarray, y: Optional[np.ndarray], classifier_mixin: bool = True) -> np.ndarray:
        """
        Check and set up targets.

        :param x: An array with the original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :param classifier_mixin: Whether the estimator is of type `ClassifierMixin`.
        :return: The targets.
        """
        if classifier_mixin:
            if y is not None:
                y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes)

        if y is None:
            # Throw error if attack is targeted, but no targets are provided
            if self.targeted:  # pragma: no cover
                raise ValueError("Target labels `y` need to be provided for a targeted attack.")

            # Use model predictions as correct outputs
            if classifier_mixin:
                targets = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))
            else:
                targets = self.estimator.predict(x, batch_size=self.batch_size)

        else:
            targets = y

        return targets

    def _check_params(self) -> None:  # pragma: no cover

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
                raise ValueError("The perturbation size `eps` has to be non-negative.")
        else:
            if (self.eps < 0).any():
                raise ValueError("The perturbation size `eps` has to be non-negative.")

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
            raise ValueError("The number of iterations `max_iter` has to be a non-negative integer.")

        if self.decay is not None and self.decay < 0.0:
            raise ValueError("The decay factor `decay` has to be a nonnegative float.")

        if not isinstance(self.verbose, bool):
            raise ValueError("The verbose has to be a Boolean.")

class ProjectedGradientDescentPyTorch(ProjectedGradientDescentCommon):
    _estimator_requirements = (BaseEstimator, LossGradientsMixin, ClassifierMixin)  # type: ignore

    def __init__(
        self,
        estimator: "PyTorchClassifier",
        norm: Union[int, float, str] = np.inf,
        eps: Union[int, float, np.ndarray] = 0.3,
        eps_step: Union[int, float, np.ndarray] = 0.1,
        decay: Optional[float] = None,
        max_iter: int = 100,
        targeted: bool = False,
        num_random_init: int = 0,
        batch_size: int = 32,
        random_eps: bool = False,
        verbose: bool = True,
    ):
        """
        Create a :class:`.ProjectedGradientDescentPyTorch` instance.

        :param estimator: An trained estimator.
        :param norm: The norm of the adversarial perturbation. Possible values: "inf", np.inf, 1 or 2.
        :param eps: Maximum perturbation that the attacker can introduce.
        :param eps_step: Attack step size (input variation) at each iteration.
        :param random_eps: When True, epsilon is drawn randomly from truncated normal distribution. The literature
                           suggests this for FGSM based training to generalize across different epsilons. eps_step is
                           modified to preserve the ratio of eps / eps_step. The effectiveness of this method with PGD
                           is untested (https://arxiv.org/pdf/1611.01236.pdf).
        :param max_iter: The maximum number of iterations.
        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False).
        :param num_random_init: Number of random initialisations within the epsilon ball. For num_random_init=0 starting
                                at the original input.
        :param batch_size: Size of the batch on which adversarial samples are generated.
        :param summary_writer: Activate summary writer for TensorBoard.
                               Default is `False` and deactivated summary writer.
                               If `True` save runs/CURRENT_DATETIME_HOSTNAME in current directory.
                               If of type `str` save in path.
                               If of type `SummaryWriter` apply provided custom summary writer.
                               Use hierarchical folder structure to compare between runs easily. e.g. pass in
                               ‘runs/exp1’, ‘runs/exp2’, etc. for each new experiment to compare across them.
        :param verbose: Show progress bars.
        """
        if not estimator.all_framework_preprocessing:
            raise NotImplementedError(
                "The framework-specific implementation only supports framework-specific preprocessing."
            )

        super().__init__(
            estimator=estimator,
            norm=norm,
            eps=eps,
            eps_step=eps_step,
            decay=decay,
            max_iter=max_iter,
            targeted=targeted,
            num_random_init=num_random_init,
            batch_size=batch_size,
            random_eps=random_eps,
            verbose=verbose,
        )

        self._batch_id = 0
        self._i_max_iter = 0

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :param mask: An array with a mask broadcastable to input `x` defining where to apply adversarial perturbations.
                     Shape needs to be broadcastable to the shape of x and can also be of the same shape as `x`. Any
                     features for which the mask is zero will not be adversarially perturbed.
        :type mask: `np.ndarray`
        :return: An array holding the adversarial examples.
        """
        import torch  # lgtm [py/repeated-import]

        # mask = self._get_mask(x, **kwargs)

        # Ensure eps is broadcastable
        # self._check_compatibility_input_and_eps(x=x)

        # Check whether random eps is enabled
        self._random_eps()

        # Set up targets
        targets = self._set_targets(x, y)

        # # Create dataset
        # if mask is not None:
        #     # Here we need to make a distinction: if the masks are different for each input, we need to index
        #     # those for the current batch. Otherwise (i.e. mask is meant to be broadcasted), keep it as it is.
        #     if len(mask.shape) == len(x.shape):
        #         dataset = torch.utils.data.TensorDataset(
        #             torch.from_numpy(x.astype(MY_NUMPY_DTYPE)),
        #             torch.from_numpy(targets.astype(MY_NUMPY_DTYPE)),
        #             torch.from_numpy(mask.astype(MY_NUMPY_DTYPE)),
        #         )

        #     else:
        #         dataset = torch.utils.data.TensorDataset(
        #             torch.from_numpy(x.astype(MY_NUMPY_DTYPE)),
        #             torch.from_numpy(targets.astype(MY_NUMPY_DTYPE)),
        #             torch.from_numpy(np.array([mask.astype(MY_NUMPY_DTYPE)] * x.shape[0])),
        #         )

        # else:
        #     dataset = torch.utils.data.TensorDataset(
        #         torch.from_numpy(x.astype(MY_NUMPY_DTYPE)),
        #         torch.from_numpy(targets.astype(MY_NUMPY_DTYPE)),
        #     )

        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(x.astype(MY_NUMPY_DTYPE)),
            torch.from_numpy(targets.astype(MY_NUMPY_DTYPE)),
        )

        data_loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=self.batch_size, shuffle=False, drop_last=False
        )

        # Start to compute adversarial examples
        adv_x = x.astype(MY_NUMPY_DTYPE)

        # Compute perturbation with batching
        for (batch_id, batch_all) in enumerate(
            tqdm(data_loader, desc="PGD - Batches", leave=False, disable=not self.verbose)
        ):

            self._batch_id = batch_id

            # if mask is not None:
            #     (batch, batch_labels, mask_batch) = batch_all[0], batch_all[1], batch_all[2]
            # else:
            #     (batch, batch_labels, mask_batch) = batch_all[0], batch_all[1], None
            (batch, batch_labels, mask_batch) = batch_all[0], batch_all[1], None

            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size

            # Compute batch_eps and batch_eps_step
            if isinstance(self.eps, np.ndarray) and isinstance(self.eps_step, np.ndarray):
                if len(self.eps.shape) == len(x.shape) and self.eps.shape[0] == x.shape[0]:
                    batch_eps = self.eps[batch_index_1:batch_index_2]
                    batch_eps_step = self.eps_step[batch_index_1:batch_index_2]

                else:
                    batch_eps = self.eps
                    batch_eps_step = self.eps_step

            else:
                batch_eps = self.eps
                batch_eps_step = self.eps_step

            for rand_init_num in range(max(1, self.num_random_init)):
                if rand_init_num == 0:
                    # first iteration: use the adversarial examples as they are the only ones we have now
                    adv_x[batch_index_1:batch_index_2] = self._generate_batch(
                        x=batch, targets=batch_labels, mask=mask_batch, eps=batch_eps, eps_step=batch_eps_step
                    )
                else:
                    adversarial_batch = self._generate_batch(
                        x=batch, targets=batch_labels, mask=mask_batch, eps=batch_eps, eps_step=batch_eps_step
                    )

                    # return the successful adversarial examples
                    attack_success = compute_success_array(
                        self.estimator,
                        batch,
                        batch_labels,
                        adversarial_batch,
                        self.targeted,
                        batch_size=self.batch_size,
                    )
                    adv_x[batch_index_1:batch_index_2][attack_success] = adversarial_batch[attack_success]

        # logger.info(
        #     "Success rate of attack: %.2f%%",
        #     100 * compute_success(self.estimator, x, targets, adv_x, self.targeted, batch_size=self.batch_size),
        # )

        # if self.summary_writer is not None:
        #     self.summary_writer.reset()

        return adv_x

    def _generate_batch(
        self,
        x: "torch.Tensor",
        targets: "torch.Tensor",
        mask: "torch.Tensor",
        eps: Union[int, float, np.ndarray],
        eps_step: Union[int, float, np.ndarray],
    ) -> np.ndarray:
        """
        Generate a batch of adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :param targets: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)`.
        :param mask: An array with a mask to be applied to the adversarial perturbations. Shape needs to be
                     broadcastable to the shape of x. Any features for which the mask is zero will not be adversarially
                     perturbed.
        :param eps: Maximum perturbation that the attacker can introduce.
        :param eps_step: Attack step size (input variation) at each iteration.
        :return: Adversarial examples.
        """
        import torch  # lgtm [py/repeated-import]

        inputs = x.to(self.estimator.device)
        targets = targets.to(self.estimator.device)
        adv_x = torch.clone(inputs)
        momentum = torch.zeros(inputs.shape)

        if mask is not None:
            mask = mask.to(self.estimator.device)

        for i_max_iter in range(self.max_iter):
            self._i_max_iter = i_max_iter
            adv_x = self._compute_pytorch(
                adv_x, inputs, targets, mask, eps, eps_step, self.num_random_init > 0 and i_max_iter == 0, momentum
            )

        return adv_x.cpu().detach().numpy()

    def _compute_perturbation_pytorch(  # pylint: disable=W0221
        self, x: "torch.Tensor", y: "torch.Tensor", mask: Optional["torch.Tensor"], momentum: "torch.Tensor"
    ) -> "torch.Tensor":
        """
        Compute perturbations.

        :param x: Current adversarial examples.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :param mask: An array with a mask broadcastable to input `x` defining where to apply adversarial perturbations.
                     Shape needs to be broadcastable to the shape of x and can also be of the same shape as `x`. Any
                     features for which the mask is zero will not be adversarially perturbed.
        :return: Perturbations.
        """
        import torch  # lgtm [py/repeated-import]

        # Pick a small scalar to avoid division by 0
        tol = 10e-8

        # Get gradient wrt loss; invert it if attack is targeted
        grad = self.estimator.loss_gradient(x=x, y=y) * (1 - 2 * int(self.targeted))

        # Write summary
        # if self.summary_writer is not None:  # pragma: no cover
        #     self.summary_writer.update(
        #         batch_id=self._batch_id,
        #         global_step=self._i_max_iter,
        #         grad=grad.cpu().detach().numpy(),
        #         patch=None,
        #         estimator=self.estimator,
        #         x=x.cpu().detach().numpy(),
        #         y=y.cpu().detach().numpy(),
        #         targeted=self.targeted,
        #     )

        # Check for nan before normalisation an replace with 0
        if torch.any(grad.isnan()):  # pragma: no cover
            logger.warning("Elements of the loss gradient are NaN and have been replaced with 0.0.")
            grad[grad.isnan()] = 0.0

        # Apply mask
        if mask is not None:
            grad = torch.where(mask == 0.0, torch.tensor(0.0).to(self.estimator.device), grad)

        # Apply momentum
        if self.decay is not None:
            ind = tuple(range(1, len(x.shape)))
            grad = grad / (torch.sum(grad.abs(), dim=ind, keepdims=True) + tol)  # type: ignore
            grad = self.decay * momentum + grad
            # Accumulate the gradient for the next iter
            momentum += grad

        # Apply norm bound
        if self.norm in ["inf", np.inf]:
            grad = grad.sign()

        elif self.norm == 1:
            ind = tuple(range(1, len(x.shape)))
            grad = grad / (torch.sum(grad.abs(), dim=ind, keepdims=True) + tol)  # type: ignore

        elif self.norm == 2:
            ind = tuple(range(1, len(x.shape)))
            grad = grad / (torch.sqrt(torch.sum(grad * grad, axis=ind, keepdims=True)) + tol)  # type: ignore

        assert x.shape == grad.shape

        return grad

    def _apply_perturbation_pytorch(  # pylint: disable=W0221
        self, x: "torch.Tensor", perturbation: "torch.Tensor", eps_step: Union[int, float, np.ndarray]
    ) -> "torch.Tensor":
        """
        Apply perturbation on examples.

        :param x: Current adversarial examples.
        :param perturbation: Current perturbations.
        :param eps_step: Attack step size (input variation) at each iteration.
        :return: Adversarial examples.
        """
        import torch  # lgtm [py/repeated-import]

        eps_step = np.array(eps_step, dtype=MY_NUMPY_DTYPE)
        perturbation_step = torch.tensor(eps_step).to(self.estimator.device) * perturbation
        perturbation_step[torch.isnan(perturbation_step)] = 0
        x = x + perturbation_step
        if self.estimator.clip_values is not None:
            clip_min, clip_max = self.estimator.clip_values
            x = torch.max(
                torch.min(x, torch.tensor(clip_max).to(self.estimator.device)),
                torch.tensor(clip_min).to(self.estimator.device),
            )

        return x

    def _compute_pytorch(
        self,
        x: "torch.Tensor",
        x_init: "torch.Tensor",
        y: "torch.Tensor",
        mask: "torch.Tensor",
        eps: Union[int, float, np.ndarray],
        eps_step: Union[int, float, np.ndarray],
        random_init: bool,
        momentum: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Compute adversarial examples for one iteration.

        :param x: Current adversarial examples.
        :param x_init: An array with the original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236).
        :param mask: An array with a mask broadcastable to input `x` defining where to apply adversarial perturbations.
                     Shape needs to be broadcastable to the shape of x and can also be of the same shape as `x`. Any
                     features for which the mask is zero will not be adversarially perturbed.
        :param eps: Maximum perturbation that the attacker can introduce.
        :param eps_step: Attack step size (input variation) at each iteration.
        :param random_init: Random initialisation within the epsilon ball. For random_init=False starting at the
                            original input.
        :return: Adversarial examples.
        """
        import torch  # lgtm [py/repeated-import]

        if random_init:
            n = x.shape[0]
            m = np.prod(x.shape[1:]).item()

            random_perturbation_array = random_sphere(n, m, eps, self.norm).reshape(x.shape).astype(MY_NUMPY_DTYPE)
            random_perturbation = torch.from_numpy(random_perturbation_array).to(self.estimator.device)

            if mask is not None:
                random_perturbation = random_perturbation * mask

            x_adv = x + random_perturbation

            if self.estimator.clip_values is not None:
                clip_min, clip_max = self.estimator.clip_values
                x_adv = torch.max(
                    torch.min(x_adv, torch.tensor(clip_max).to(self.estimator.device)),
                    torch.tensor(clip_min).to(self.estimator.device),
                )

        else:
            x_adv = x

        # Get perturbation
        perturbation = self._compute_perturbation_pytorch(x_adv, y, mask, momentum)

        # Apply perturbation and clip
        x_adv = self._apply_perturbation_pytorch(x_adv, perturbation, eps_step)

        # Do projection
        perturbation = self._projection(x_adv - x_init, eps, self.norm)

        # Recompute x_adv
        x_adv = perturbation + x_init

        return x_adv

    def _projection(
        self, values: "torch.Tensor", eps: Union[int, float, np.ndarray], norm_p: Union[int, float, str]
    ) -> "torch.Tensor":
        """
        Project `values` on the L_p norm ball of size `eps`.

        :param values: Values to clip.
        :param eps: Maximum norm allowed.
        :param norm_p: L_p norm to use for clipping supporting 1, 2, `np.Inf` and "inf".
        :return: Values of `values` after projection.
        """
        import torch  # lgtm [py/repeated-import]

        # Pick a small scalar to avoid division by 0
        tol = 10e-8
        values_tmp = values.reshape(values.shape[0], -1)

        if norm_p == 2:
            if isinstance(eps, np.ndarray):
                raise NotImplementedError(
                    "The parameter `eps` of type `np.ndarray` is not supported to use with norm 2."
                )

            values_tmp = (
                values_tmp
                * torch.min(
                    torch.tensor([1.0], dtype=torch.float32).to(self.estimator.device),
                    eps / (torch.norm(values_tmp, p=2, dim=1) + tol),
                ).unsqueeze_(-1)
            )

        elif norm_p == 1:
            if isinstance(eps, np.ndarray):
                raise NotImplementedError(
                    "The parameter `eps` of type `np.ndarray` is not supported to use with norm 1."
                )

            values_tmp = (
                values_tmp
                * torch.min(
                    torch.tensor([1.0], dtype=torch.float32).to(self.estimator.device),
                    eps / (torch.norm(values_tmp, p=1, dim=1) + tol),
                ).unsqueeze_(-1)
            )

        elif norm_p in [np.inf, "inf"]:
            if isinstance(eps, np.ndarray):
                eps = eps * np.ones_like(values.cpu())
                eps = eps.reshape([eps.shape[0], -1])  # type: ignore

            values_tmp = values_tmp.sign() * torch.min(
                values_tmp.abs(), torch.tensor([eps], dtype=torch.float32).to(self.estimator.device)
            )

        else:
            raise NotImplementedError(
                "Values of `norm_p` different from 1, 2 and `np.inf` are currently not supported."
            )

        values = values_tmp.reshape(values.shape)

        return values

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
        batch_size: int = 2048,
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
        self._attack: "ProjectedGradientDescentPyTorch"
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
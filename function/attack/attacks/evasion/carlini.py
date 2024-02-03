from __future__ import absolute_import, division, print_function, unicode_literals
from typing import Optional, Tuple, TYPE_CHECKING
import numpy as np
from tqdm.auto import trange
from ..optimizers import Adam
from function.attack.estimators.estimator import BaseEstimator
from function.attack.attacks.attack import EvasionAttack
from tqdm import tqdm
import torch
if TYPE_CHECKING:
    from function.attack.attacks.utils import CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE
from function.attack.estimators.estimator import BaseEstimator, NeuralNetworkMixin
from function.attack.estimators.classification.classifier import ClassifierMixin
from function.attack.attacks.config import MY_NUMPY_DTYPE

class CarliniWagner(EvasionAttack):
    attack_params = EvasionAttack.attack_params + [
        "lr",
        "confidence",
        "initial_const",
        "targeted",
        "binary_search_steps",
        "max_iterations",
        "batch_size",
    ]
    attack_params = list(set(attack_params))
    _estimator_requirements = (BaseEstimator, ClassifierMixin, NeuralNetworkMixin)

    def __init__(
            self, 
            estimator: "CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE",
            targeted=False,
            lr=1e-2,
            confidence=0,
            initial_const=1e-3,
            binary_search_steps=5,
            max_iterations=1000,
            batch_size = 128,
        ) -> None:
            super().__init__(estimator=estimator)
            self.lr = lr
            self.targeted = targeted
            self.confidence = confidence
            self.initial_const = initial_const
            self.binary_search_steps = binary_search_steps
            self.max_iterations = max_iterations
            self.myestimator = estimator
            self.batch_size = batch_size
    
    def carlini_wagner_l2(self, x, y):
        INF = float("inf")
        def compare(pred, label, is_logits=False):
            # Convert logits to predicted class if necessary
            if is_logits:
                pred_copy = pred.clone().detach()
                pred_copy[label] += -self.confidence if self.targeted else self.confidence
                pred = torch.argmax(pred_copy)

            return pred == label if self.targeted else pred != label

        if y is None:
            # Using model predictions as ground truth to avoid label leaking
            pred = self.estimator.model(x)
            y = torch.argmax(pred, 1)

        # Initialize some values needed for binary search on const
        lower_bound = [0.0] * len(x)
        upper_bound = [1e10] * len(x)
        const = x.new_ones(len(x), 1) * self.initial_const

        o_bestl2 = [INF] * len(x)
        o_bestscore = [-1.0] * len(x)
        x = torch.clamp(x, self.estimator.clip_values[0], self.estimator.clip_values[1])
        ox = x.clone().detach()  # save the original x
        o_bestattack = x.clone().detach()

        # Map images into the tanh-space
        x = (x - self.estimator.clip_values[0]) / (self.estimator.clip_values[1] - self.estimator.clip_values[0])
        x = torch.clamp(x, 0, 1)
        x = x * 2 - 1
        x = torch.arctanh(x * 0.999999)

        # Prepare some variables
        modifier = torch.zeros_like(x, requires_grad=True)
        y_onehot = torch.nn.functional.one_hot(y, self.estimator.nb_classes).to(torch.float)

        # Define loss functions and optimizer
        f_fn = lambda real, other, targeted: torch.max(
            ((other - real) if targeted else (real - other)) + self.confidence,
            torch.tensor(0.0).to(real.device),
        )
        l2dist_fn = lambda x, y: torch.pow(x - y, 2).sum(list(range(len(x.size())))[1:])
        optimizer = torch.optim.Adam([modifier], lr=self.lr)

        # Outer loop performing binary search on const
        for outer_step in tqdm(range(self.binary_search_steps)):
            # Initialize some values needed for the inner loop
            bestl2 = [INF] * len(x)
            bestscore = [-1.0] * len(x)

            # Inner loop performing attack iterations
            for i in range(self.max_iterations):
                # One attack step
                new_x = (torch.tanh(modifier + x) + 1) / 2
                new_x = new_x * (self.estimator.clip_values[1] - self.estimator.clip_values[0]) + self.estimator.clip_values[0]
                logits = self.estimator.model(new_x)

                real = torch.sum(y_onehot * logits, 1)
                other, _ = torch.max((1 - y_onehot) * logits - y_onehot * 1e4, 1)

                optimizer.zero_grad()
                f = f_fn(real, other, self.targeted)
                l2 = l2dist_fn(new_x, ox)
                loss = (const * f + l2).sum()
                loss.backward()
                optimizer.step()

                # Update best results
                for n, (l2_n, logits_n, new_x_n) in enumerate(zip(l2, logits, new_x)):
                    y_n = y[n]
                    succeeded = compare(logits_n, y_n, is_logits=True)
                    if l2_n < o_bestl2[n] and succeeded:
                        pred_n = torch.argmax(logits_n)
                        o_bestl2[n] = l2_n
                        o_bestscore[n] = pred_n
                        o_bestattack[n] = new_x_n
                        # l2_n < o_bestl2[n] implies l2_n < bestl2[n] so we modify inner loop variables too
                        bestl2[n] = l2_n
                        bestscore[n] = pred_n
                    elif l2_n < bestl2[n] and succeeded:
                        bestl2[n] = l2_n
                        bestscore[n] = torch.argmax(logits_n)

            # Binary search step
            for n in range(len(x)):
                y_n = y[n]
                if compare(bestscore[n], y_n) and bestscore[n] != -1:
                    # Success, divide const by two
                    upper_bound[n] = min(upper_bound[n], const[n])
                    if upper_bound[n] < 1e9:
                        const[n] = (lower_bound[n] + upper_bound[n]) / 2
                else:
                    # Failure, either multiply by 10 if no solution found yet
                    # or do binary search with the known upper bound
                    lower_bound[n] = max(lower_bound[n], const[n])
                    if upper_bound[n] < 1e9:
                        const[n] = (lower_bound[n] + upper_bound[n]) / 2
                    else:
                        const[n] *= 10

        return o_bestattack.detach().cpu().numpy()

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        adv_x = np.zeros(shape=x.shape)
        for batch_id in trange(
            int(np.ceil(x.shape[0] / float(self.batch_size))), desc="C&W"
        ):
            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            x_batch = torch.from_numpy(x[batch_index_1:batch_index_2].copy()).to(self.estimator.device)
            if isinstance(y,type(None)) :
                y_batch = None
            else:
                y_batch = torch.from_numpy(y[batch_index_1:batch_index_2].copy()).to(self.estimator.device)
            adv_x[batch_index_1:batch_index_2] = self.carlini_wagner_l2(x_batch, y_batch)
            x_batch.cpu()
            if y_batch != None:
                y_batch.cpu()
        return adv_x.astype(MY_NUMPY_DTYPE)
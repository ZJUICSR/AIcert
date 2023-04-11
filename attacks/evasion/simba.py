from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from typing import Optional, TYPE_CHECKING
import numpy as np
from scipy.fftpack import idct
from tqdm.auto import trange
from attacks.attack import EvasionAttack
from estimators.estimator import BaseEstimator, NeuralNetworkMixin
from estimators.classification.classifier import ClassifierMixin
from attacks.config import MY_NUMPY_DTYPE
if TYPE_CHECKING:
    from attacks.utils import CLASSIFIER_TYPE

class SimBA(EvasionAttack):
    attack_params = EvasionAttack.attack_params + [
        "attack",
        "max_iter",
        "epsilon",
        "order",
        "freq_dim",
        "stride",
        "targeted",
        "batch_size",
    ]

    _estimator_requirements = (BaseEstimator, ClassifierMixin, NeuralNetworkMixin)

    def __init__(
        self,
        classifier: "CLASSIFIER_TYPE",
        attack: str = "dct",
        max_iter: int = 3000,
        order: str = "random",
        eps: float = 8/255,
        freq_dim: int = 4,
        stride: int = 1,
        targeted: bool = False,
        batch_size: int = 128,
    ):
        super().__init__(estimator=classifier)

        self.attack = attack
        self.max_iter = max_iter
        self.epsilon = eps
        self.order = order
        self.freq_dim = freq_dim
        self.stride = stride
        self._targeted = targeted
        self.batch_size = batch_size
        # self._check_params()

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        x = x.astype(MY_NUMPY_DTYPE)
        x_adv = x.copy()

        y_prob_pred = self.estimator.predict(x, batch_size=self.batch_size)
        # print(y_prob_pred[0])
        # if not is_probability(y_prob_pred[0]):
        #     raise ValueError(
        #         "This attack requires an estimator predicting probabilities. It looks like the current "
        #         "estimator is not predicting probabilities"
        #     )

        if self.estimator.nb_classes == 2 and y_prob_pred.shape[1] == 1:
            raise ValueError(
                "This attack has not yet been tested for binary classification with a single output classifier."
            )

        if divmod(x.shape[2] - self.freq_dim, self.stride)[1] != 0:
            raise ValueError(
                "Incompatible value combination in image height/width, freq_dim and stride detected. "
                "Adapt these parameters to fulfill the following conditions: "
                "divmod(image_height - freq_dim, stride)[1] == 0 "
                "and "
                "divmod(image_width - freq_dim, stride)[1] == 0"
            )

        if y is None:
            if self.targeted:
                raise ValueError("Target labels `y` need to be provided for a targeted attack.")
            y_i = np.argmax(y_prob_pred, axis=1)
        else:
            y_i = np.argmax(y, axis=1)

        for i_sample in trange(x.shape[0], desc="SimBA - sample"):

            desired_label = y_i[i_sample]

            current_label = np.argmax(y_prob_pred, axis=1)[i_sample]
            last_prob = y_prob_pred[i_sample].reshape(-1)[desired_label]

            if self.estimator.channels_first:
                nb_channels = x.shape[1]
            else:
                nb_channels = x.shape[3]

            n_dims = np.prod(x[[0]].shape)

            if self.attack == "px":
                if self.order == "diag":
                    indices = self.diagonal_order(x.shape[2], nb_channels)[: self.max_iter]
                elif self.order == "random":
                    indices = np.random.permutation(n_dims)[: self.max_iter]
                indices_size = len(indices)
                while indices_size < self.max_iter:
                    if self.order == "diag":
                        tmp_indices = self.diagonal_order(x.shape[2], nb_channels)
                    elif self.order == "random":
                        tmp_indices = np.random.permutation(n_dims)
                    indices = np.hstack((indices, tmp_indices))[: self.max_iter]
                    indices_size = len(indices)
            elif self.attack == "dct":
                indices = self._block_order(x.shape[2], nb_channels, initial_size=self.freq_dim, stride=self.stride)[
                    : self.max_iter
                ]
                indices_size = len(indices)
                while indices_size < self.max_iter:
                    tmp_indices = self._block_order(
                        x.shape[2], nb_channels, initial_size=self.freq_dim, stride=self.stride
                    )
                    indices = np.hstack((indices, tmp_indices))[: self.max_iter]
                    indices_size = len(indices)

                def trans(var_z):
                    return self._block_idct(var_z, block_size=x.shape[2])

            clip_min = -np.inf
            clip_max = np.inf
            if self.estimator.clip_values is not None:
                clip_min, clip_max = self.estimator.clip_values

            term_flag = 1
            if self.targeted:
                if desired_label != current_label:
                    term_flag = 0
            else:
                if desired_label == current_label:
                    term_flag = 0

            nb_iter = 0
            while term_flag == 0 and nb_iter < self.max_iter:
                diff = np.zeros(n_dims).astype(MY_NUMPY_DTYPE)
                diff[indices[nb_iter]] = self.epsilon

                if self.attack == "dct":
                    left_preds = self.estimator.predict(
                        np.clip(x[[i_sample]] - trans(diff.reshape(x[[i_sample]].shape)), clip_min, clip_max),
                        batch_size=self.batch_size,
                    )
                elif self.attack == "px":
                    left_preds = self.estimator.predict(
                        np.clip(x[[i_sample]] - diff.reshape(x[[i_sample]].shape), clip_min, clip_max),
                        batch_size=self.batch_size,
                    )
                left_prob = left_preds.reshape(-1)[desired_label]

                if self.attack == "dct":
                    right_preds = self.estimator.predict(
                        np.clip(x[[i_sample]] + trans(diff.reshape(x[[i_sample]].shape)), clip_min, clip_max),
                        batch_size=self.batch_size,
                    )
                elif self.attack == "px":
                    right_preds = self.estimator.predict(
                        np.clip(x[[i_sample]] + diff.reshape(x[[i_sample]].shape), clip_min, clip_max),
                        batch_size=self.batch_size,
                    )
                right_prob = right_preds.reshape(-1)[desired_label]

                # Use (2 * int(self.targeted) - 1) to shorten code?
                if self.targeted:
                    if left_prob > last_prob:
                        if left_prob > right_prob:
                            if self.attack == "dct":
                                x[[i_sample]] = np.clip(
                                    x[[i_sample]] - trans(diff.reshape(x[[i_sample]].shape)), clip_min, clip_max
                                )
                            elif self.attack == "px":
                                x[[i_sample]] = np.clip(
                                    x[[i_sample]] - diff.reshape(x[[i_sample]].shape), clip_min, clip_max
                                )
                            last_prob = left_prob
                            current_label = np.argmax(left_preds, axis=1)[0]
                        else:
                            if self.attack == "dct":
                                x[[i_sample]] = np.clip(
                                    x[[i_sample]] + trans(diff.reshape(x[[i_sample]].shape)), clip_min, clip_max
                                )
                            elif self.attack == "px":
                                x[[i_sample]] = np.clip(
                                    x[[i_sample]] + diff.reshape(x[[i_sample]].shape), clip_min, clip_max
                                )
                            last_prob = right_prob
                            current_label = np.argmax(right_preds, axis=1)[0]
                    else:
                        if right_prob > last_prob:
                            if self.attack == "dct":
                                x[[i_sample]] = np.clip(
                                    x[[i_sample]] + trans(diff.reshape(x[[i_sample]].shape)), clip_min, clip_max
                                )
                            elif self.attack == "px":
                                x[[i_sample]] = np.clip(
                                    x[[i_sample]] + diff.reshape(x[[i_sample]].shape), clip_min, clip_max
                                )
                            last_prob = right_prob
                            current_label = np.argmax(right_preds, axis=1)[0]
                else:
                    if left_prob < last_prob:
                        if left_prob < right_prob:
                            if self.attack == "dct":
                                x[[i_sample]] = np.clip(
                                    x[[i_sample]] - trans(diff.reshape(x[[i_sample]].shape)), clip_min, clip_max
                                )
                            elif self.attack == "px":
                                x[[i_sample]] = np.clip(
                                    x[[i_sample]] - diff.reshape(x[[i_sample]].shape), clip_min, clip_max
                                )
                            last_prob = left_prob
                            current_label = np.argmax(left_preds, axis=1)[0]
                        else:
                            if self.attack == "dct":
                                x[[i_sample]] = np.clip(
                                    x[[i_sample]] + trans(diff.reshape(x[[i_sample]].shape)), clip_min, clip_max
                                )
                            elif self.attack == "px":
                                x[[i_sample]] = np.clip(
                                    x[[i_sample]] + diff.reshape(x[[i_sample]].shape), clip_min, clip_max
                                )
                            last_prob = right_prob
                            current_label = np.argmax(right_preds, axis=1)[0]
                    else:
                        if right_prob < last_prob:
                            if self.attack == "dct":
                                x[[i_sample]] = np.clip(
                                    x[[i_sample]] + trans(diff.reshape(x[[i_sample]].shape)), clip_min, clip_max
                                )
                            elif self.attack == "px":
                                x[[i_sample]] = np.clip(
                                    x[[i_sample]] + diff.reshape(x[[i_sample]].shape), clip_min, clip_max
                                )
                            last_prob = right_prob
                            current_label = np.argmax(right_preds, axis=1)[0]

                if self.targeted:
                    if desired_label == current_label:
                        term_flag = 1
                else:
                    if desired_label != current_label:
                        term_flag = 1

                nb_iter = nb_iter + 1

            if nb_iter < self.max_iter:
                pass
            else:
                pass
            x_adv[i_sample] = x[i_sample]

        return x_adv

    def _block_order(self, img_size, channels, initial_size=2, stride=1):
        order = np.zeros((channels, img_size, img_size)).astype(MY_NUMPY_DTYPE)
        total_elems = channels * initial_size * initial_size
        perm = np.random.permutation(total_elems)
        order[:, :initial_size, :initial_size] = perm.reshape((channels, initial_size, initial_size))
        for i in range(initial_size, img_size, stride):
            num_elems = channels * (2 * stride * i + stride * stride)
            perm = np.random.permutation(num_elems) + total_elems
            num_first = channels * stride * (stride + i)
            order[:, : (i + stride), i : (i + stride)] = perm[:num_first].reshape((channels, -1, stride))
            order[:, i : (i + stride), :i] = perm[num_first:].reshape((channels, stride, -1))
            total_elems += num_elems

        if self.estimator.channels_first:
            return order.reshape(1, -1).squeeze().argsort()

        return order.transpose((1, 2, 0)).reshape(1, -1).squeeze().argsort()

    def _block_idct(self, x, block_size=8, masked=False, ratio=0.5):
        if not self.estimator.channels_first:
            x = x.transpose(0, 3, 1, 2)
        var_z = np.zeros(x.shape).astype(MY_NUMPY_DTYPE)
        num_blocks = int(x.shape[2] / block_size)
        mask = np.zeros((x.shape[0], x.shape[1], block_size, block_size))
        if not isinstance(ratio, float):
            for i in range(x.shape[0]):
                mask[i, :, : int(block_size * ratio[i]), : int(block_size * ratio[i])] = 1
        else:
            mask[:, :, : int(block_size * ratio), : int(block_size * ratio)] = 1
        for i in range(num_blocks):
            for j in range(num_blocks):
                submat = x[:, :, (i * block_size) : ((i + 1) * block_size), (j * block_size) : ((j + 1) * block_size)]
                if masked:
                    submat = submat * mask
                var_z[
                    :, :, (i * block_size) : ((i + 1) * block_size), (j * block_size) : ((j + 1) * block_size)
                ] = idct(idct(submat, axis=3, norm="ortho"), axis=2, norm="ortho")

        if self.estimator.channels_first:
            return var_z

        return var_z.transpose((0, 2, 3, 1))

    def diagonal_order(self, image_size, channels):
        x = np.arange(0, image_size).cumsum()
        order = np.zeros((image_size, image_size)).astype(MY_NUMPY_DTYPE)
        for i in range(image_size):
            order[i, : (image_size - i)] = i + x[i:]
        for i in range(1, image_size):
            reverse = order[image_size - i - 1].take([i for i in range(i - 1, -1, -1)])  # pylint: disable=R1721
            order[i, (image_size - i) :] = image_size * image_size - 1 - reverse
        if channels > 1:
            order_2d = order
            order = np.zeros((channels, image_size, image_size))
            for i in range(channels):
                order[i, :, :] = 3 * order_2d + i
        elif channels == 1:
            order = np.expand_dims(order, axis=0)

        if self.estimator.channels_first:
            return order.reshape(1, -1).squeeze().argsort()

        return order.transpose((1, 2, 0)).reshape(1, -1).squeeze().argsort()
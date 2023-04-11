from __future__ import absolute_import, division, print_function, unicode_literals
# import logging
from typing import Optional, Union, TYPE_CHECKING
import numpy as np
from tqdm.auto import trange
from attacks.attack import EvasionAttack
from attacks.config import MY_NUMPY_DTYPE
from estimators.estimator import BaseEstimator
from estimators.classification.classifier import ClassGradientsMixin
from attacks.utils import check_and_transform_label_format
if TYPE_CHECKING:
    from attacks.utils import CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE
# logger = logging.getLogger(__name__)

# Saliency_map Attack 该攻击默认为有目标攻击，需要提前设置目标类，否则算法将启动随机目标
# 该攻击在论文中并没有使用范数来作为约束
# gamma限制了允许修改的像素点的比例，可以看作是零范数下的约束
# theta值限制了每一次对每个像素特征值可以修改的大小
class SaliencyMapMethod(EvasionAttack):
    attack_params = EvasionAttack.attack_params + ["theta", "gamma", "batch_size"]
    _estimator_requirements = (BaseEstimator, ClassGradientsMixin)

    def __init__(
        self,
        classifier: "CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE",
        theta: float = 0.1,
        gamma: float = 1.0,
        batch_size: int = 128,
        # verbose: bool = True,
    ) -> None:
        super().__init__(estimator=classifier)
        self.theta = theta
        self.gamma = gamma
        self.batch_size = batch_size
        # self.verbose = verbose
        # self._check_params()

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        if y is not None:
            y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes)

        ## 初始化相关变量
        # 得到样本维度
        dims = list(x.shape[1:])
        # 得到特征数目，特征即样本的每一个像素点
        self._nb_features = np.product(dims)
        # 将样本x按照特征数的方式展开
        x_adv = np.reshape(x.astype(MY_NUMPY_DTYPE), (-1, self._nb_features))
        # 第一次得到预测结果
        preds = np.argmax(self.estimator.predict(x, batch_size=self.batch_size), axis=1)

        # 设定目标类
        if y is None:
            # Randomly choose target from the incorrect classes for each sample
            from attacks.utils import random_targets

            targets = np.argmax(random_targets(preds, self.estimator.nb_classes), axis=1)
        else:
            # if self.estimator.nb_classes == 2 and y.shape[1] == 1:  # pragma: no cover
            #     raise ValueError(
            #         "This attack has not yet been tested for binary classification with a single output classifier."
            #     )

            targets = np.argmax(y, axis=1)

        # 对每一个batch的样本进行对抗扰动
        for batch_id in trange(
            int(np.ceil(x_adv.shape[0] / float(self.batch_size))), desc="JSMA"
        ):
            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            batch = x_adv[batch_index_1:batch_index_2]

            # Main algorithm for each batch
            # Initialize the search space; optimize to remove features that can't be changed
            if self.estimator.clip_values is not None:
                search_space = np.zeros(batch.shape)
                clip_min, clip_max = self.estimator.clip_values
                if self.theta > 0:
                    search_space[batch < clip_max] = 1
                else:  # pragma: no cover
                    search_space[batch > clip_min] = 1
            else:
                search_space = np.ones(batch.shape)

            # 得到当前的预测结果
            current_pred = preds[batch_index_1:batch_index_2]
            target = targets[batch_index_1:batch_index_2]
            # 仅仅对没有达到目标的样本进行处理
            active_indices = np.where(current_pred != target)[0]
            all_feat = np.zeros_like(batch)

            while active_indices.size != 0:
                # 计算特征图
                feat_ind = self._saliency_map(
                    np.reshape(batch, [batch.shape[0]] + dims)[active_indices],
                    target[active_indices],
                    search_space[active_indices],
                )

                # 更新被使用的特征
                all_feat[active_indices, feat_ind[:, 0]] = 1
                all_feat[active_indices, feat_ind[:, 1]] = 1

                # Apply attack with clipping
                if self.estimator.clip_values is not None:
                    # Prepare update depending of theta
                    if self.theta > 0:
                        clip_func, clip_value = np.minimum, clip_max  # type: ignore
                    else:  # pragma: no cover
                        clip_func, clip_value = np.maximum, clip_min  # type: ignore

                    # 使用theta计算对抗扰动
                    tmp_batch = batch[active_indices]
                    tmp_batch[np.arange(len(active_indices)), feat_ind[:, 0]] = clip_func(
                        clip_value,
                        tmp_batch[np.arange(len(active_indices)), feat_ind[:, 0]] + self.theta,
                    )
                    tmp_batch[np.arange(len(active_indices)), feat_ind[:, 1]] = clip_func(
                        clip_value,
                        tmp_batch[np.arange(len(active_indices)), feat_ind[:, 1]] + self.theta,
                    )
                    batch[active_indices] = tmp_batch
                    search_space[batch == clip_value] = 0
                else:
                    tmp_batch = batch[active_indices]
                    tmp_batch[np.arange(len(active_indices)), feat_ind[:, 0]] += self.theta
                    tmp_batch[np.arange(len(active_indices)), feat_ind[:, 1]] += self.theta
                    batch[active_indices] = tmp_batch

                # Recompute model prediction
                current_pred = np.argmax(
                    self.estimator.predict(np.reshape(batch, [batch.shape[0]] + dims)),
                    axis=1,
                )

                # Update active_indices
                active_indices = np.where((current_pred != target)*(np.sum(all_feat, axis=1) / self._nb_features <= self.gamma)*(np.sum(search_space, axis=1) > 0))[0]

            x_adv[batch_index_1:batch_index_2] = batch

        x_adv = np.reshape(x_adv, x.shape)

        return x_adv
    
    # 计算特征图
    def _saliency_map(self, x: np.ndarray, target: Union[np.ndarray, int], search_space: np.ndarray) -> np.ndarray:
        # 注意，使用的是class_gradient,该梯度是模型的输出与输入的导数，并不是模型损失函数和输入的导数
        # class_gradient使用了torch.autograd.backward函数，计算模型输出与输入的导数，同时对于每个样本点其导数值通过模型输出的比例进行了叠加
        # class_gradient是较为耗时的操作
        grads = self.estimator.class_gradient(x, label=target)
        grads = np.reshape(grads, (-1, self._nb_features)) # 将梯度以特征维度的方式进行展开

        # Remove gradients for already used features
        used_features = 1 - search_space
        # theta参数的第一个作用，控制是在样本点上加
        coeff = 2 * int(self.theta > 0) - 1
        grads[used_features == 1] = -np.inf * coeff

        if self.theta > 0:
            ind = np.argpartition(grads, -2, axis=1)[:, -2:]
        else:  # pragma: no cover
            ind = np.argpartition(-grads, -2, axis=1)[:, -2:]
        # 每个样本取影响最大的两个特征点
        return ind

    # def _check_params(self) -> None:
    #     if self.gamma <= 0 or self.gamma > 1:
    #         raise ValueError("The total perturbation percentage `gamma` must be between 0 and 1.")

    #     if self.batch_size <= 0:
    #         raise ValueError("The batch size `batch_size` has to be positive.")

    #     if not isinstance(self.verbose, bool):
    #         raise ValueError("The argument `verbose` has to be of type bool.")
import torch
from torch.utils.data import DataLoader

from typing import Union, List, Tuple

from .tl_trainer import TransferLearningTrainer
from ....src.networks import SupportedAllModuleType
from ....src.utils import logger


class BNTransferLearningTrainer(TransferLearningTrainer):

    def __init__(self, k: int, teacher_model_path: str,
                 model: SupportedAllModuleType, train_loader: DataLoader,
                 test_loader: DataLoader, freeze_bn: Union[bool, List[int], Tuple[int, int]] = False,
                 reuse_statistic: bool = False,
                 reuse_teacher_statistic: bool = False,
                 checkpoint_path: str = None):
        """we obey following ideas in `transform learning trainer`

        Ideas:
            1. first reshape fully connect layer of teacher state_dict
            2. load reshaped state_dict
            3. set `requires_grad = False` for all parameters in model
            4. set `requires_grad = True` for parameters in last `k` blocks
            5. reset parameters of last `k` blocks
        """
        super().__init__(k, teacher_model_path, model, train_loader, test_loader, checkpoint_path)

        if checkpoint_path:
            import os
            self._checkpoint_path = checkpoint_path
            if os.path.exists(checkpoint_path):
                logger.warning("We load checkpoint here")
                self._load_from_checkpoint(checkpoint_path)

        self._fine_tuned_block_cnt = k
        # freeze BatchNorm layers to be tuned
        self._freeze_bn = freeze_bn
        self._reuse_statistic = reuse_statistic
        if freeze_bn:
            logger.info(f"freezing batch norm layers and paramerter are {freeze_bn}, {reuse_statistic}")
            if isinstance(freeze_bn, bool):
                # assume 'freeze_bn' === True
                self._freeze_bn = None
            self.freeze_bn_layer(verbose=True, freezing_range=self._freeze_bn, reuse_statistic=self._reuse_statistic)

        # freeze BatchNorm layers of the teacher
        self._reuse_teacher_statistic = reuse_teacher_statistic
        if self._fine_tuned_block_cnt >= 1:
            _total_blocks = self._blocks.get_total_blocks()
            self.freeze_bn_layer(verbose=True, freezing_range=(1, _total_blocks - self._fine_tuned_block_cnt),
                                 reuse_statistic=self._reuse_teacher_statistic)

        logger.debug(f"*************** trainable param: ")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                logger.debug(f"{name}")

        logger.debug(f"*************** BN in training mode: ")
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d) and module.training:
                logger.debug(f"{name}")

    def step_batch(self, inputs: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float]:
        if self._freeze_bn and self._reuse_statistic:
            self.freeze_bn_layer(verbose=False, freezing_range=self._freeze_bn,
                                 reuse_statistic=self._reuse_statistic)
        if self._reuse_teacher_statistic and self._fine_tuned_block_cnt >= 1:
            _total_blocks = self._blocks.get_total_blocks()
            self.freeze_bn_layer(verbose=False, freezing_range=(1, _total_blocks - self._fine_tuned_block_cnt),
                                 reuse_statistic=self._reuse_teacher_statistic)

        return super().step_batch(inputs, labels)

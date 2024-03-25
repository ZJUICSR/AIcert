import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from typing import List, Tuple, Union

from .tl_trainer import TransferLearningTrainer
from ....src.networks import SupportedAllModuleType
from ....src.utils import logger

# Instead of using spectral_norm in torch.nn.utils, here we use a custommed 
# function, since we reform the norm as
#               W = W / sigma * norm_beta.
# If ignore 'norm_beta', spectral_norm is identical to the implementation of PyTorch.
from ....src.utils.spectral_norm import spectral_norm, remove_spectral_norm


# SpectralNorm without considerring 'norm_beta'
class SpectralNormTransferLearningTrainer(TransferLearningTrainer):
    def __init__(self, k: int, teacher_model_path: str,
                 model: SupportedAllModuleType, train_loader: DataLoader,
                 test_loader: DataLoader, power_iter:int=1, norm_beta:float=1.0, 
                freeze_bn:Union[bool, List[int], Tuple[int, int]]=False, reuse_statistic: bool=False, 
                reuse_teacher_statistic:bool=False,
                checkpoint_path: str = None):

        # Ugly Hack
        # For adapt 'BaseTrainer', since it loads checkpoint during init before layers add spectral norm
        self.spectral_norm_initialized = False

        super().__init__(k, teacher_model_path, model, train_loader, test_loader, checkpoint_path)
        self._fine_tuned_block_cnt = k
        self._power_iter = power_iter
        self._norm_beta = norm_beta
        self._apply_spectral_norm(self._fine_tuned_block_cnt)

        self.spectral_norm_initialized = True
        # Here we actually try to load checkpoint.
        if checkpoint_path:
            import os
            self._checkpoint_path = checkpoint_path
            if os.path.exists(checkpoint_path):
                logger.warning("We load checkpoint here")
                self._load_from_checkpoint(checkpoint_path)
        
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
            self.freeze_bn_layer(verbose=True, freezing_range=(1, _total_blocks - self._fine_tuned_block_cnt), reuse_statistic=self._reuse_teacher_statistic)
        
        logger.debug(f"*************** trainable param: ")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                logger.debug(f"{name}")
        
        logger.debug(f"*************** BN in training mode: ")
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d) and module.training:
                logger.debug(f"{name}")


    def step_batch(self, inputs: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float]:
        # If we want to reuse statistic of batch norm layer, we have to refreeze them, since 
        # after model validation, model goes back to 'eval' mode.
        if self._freeze_bn and self._reuse_statistic:
            self.freeze_bn_layer(verbose=False, freezing_range=self._freeze_bn, reuse_statistic=self._reuse_statistic)
        if self._reuse_teacher_statistic and self._fine_tuned_block_cnt >= 1:
            _total_blocks = self._blocks.get_total_blocks()
            self.freeze_bn_layer(verbose=False, freezing_range=(1, _total_blocks - self._fine_tuned_block_cnt), reuse_statistic=self._reuse_teacher_statistic)
        
        return super().step_batch(inputs, labels)

    
    def _apply_spectral_norm(self, k):
        total_blocks = self._blocks.get_total_blocks()

        for i in range(total_blocks, total_blocks-k, -1):
            block = getattr(self._blocks, f"block{i}") # type:nn.Sequential
            for key in block._modules.keys():
                layer = getattr(block, key)
                if isinstance(layer, (nn.Conv2d, nn.Linear)):
                    setattr(block, key, spectral_norm(layer, n_power_iterations=self._power_iter, norm_beta=self._norm_beta))
                    logger.debug(f"replace '{key}' of 'block{i}' by SN version, \
                                    with 'n_power_iterations'={self._power_iter}, 'norm_beta'={self._norm_beta}")

    def _remove_spectral_norm(self, k):
        total_blocks = self._blocks.get_total_blocks()

        for i in range(total_blocks, total_blocks-k, -1):
            block = getattr(self._blocks, f"block{i}") # type:nn.Sequential
            for key in block._modules.keys():
                layer = getattr(block, key)
                if isinstance(layer, (nn.Conv2d, nn.Linear)):
                    setattr(block, key, remove_spectral_norm(layer))
                    logger.debug(f"recover '{key}' of 'block{i}' to normal version")
    
    def _save_last_model(self, save_path: str) -> None:
        self._remove_spectral_norm(self._fine_tuned_block_cnt)
        super()._save_last_model(save_path)
    
    def _save_checkpoint(self, current_epoch, best_acc):
        return super()._save_checkpoint(current_epoch, best_acc)
    
    def _load_from_checkpoint(self, checkpoint_path: str) -> None:
        if not self.spectral_norm_initialized:
            logger.warning("We don't load checkpoint at this moment")
            # here we give some fake data
            self.start_epoch = 1
            self.best_acc = 0
        else:
            super()._load_from_checkpoint(checkpoint_path)



if __name__ == "__main__":
    from src.networks import wrn34_10
    from src.utils import get_cifar_test_dataloader, get_cifar_train_dataloader

    trainer = SpectralNormTransferLearningTrainer(
        k=8,
        teacher_model_path="./trained_models/cartl_wrn34_cifar100_8_0.01-best_robust",
        model=wrn34_10(num_classes=10),
        train_loader=get_cifar_train_dataloader("cifar10"),
        test_loader=get_cifar_test_dataloader("cifar10"),
        checkpoint_path=f"checkpoint/sntl_wrn34_cifar10_4_cartl_wrn34_cifar100_4_0.01-best_robust.pth"
    )

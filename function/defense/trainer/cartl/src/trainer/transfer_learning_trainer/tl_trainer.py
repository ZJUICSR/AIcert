import torch
from torch.utils.data import DataLoader

import os

from .mixins import ReshapeTeacherFCLayerMixin
from ..mixins import InitializeTensorboardMixin
from ..normal_trainer import NormalTrainer
from ..retrain_trainer import ResetBlockMixin, FreezeModelMixin
from ....src.networks import make_blocks
from ....src.networks import SupportedAllModuleType
from ....src.utils import logger


class TransferLearningTrainer(NormalTrainer, ResetBlockMixin, FreezeModelMixin,
                              ReshapeTeacherFCLayerMixin, InitializeTensorboardMixin):

    def __init__(self, k: int, teacher_model_path: str,
                 model: SupportedAllModuleType, train_loader: DataLoader,
                 test_loader: DataLoader, checkpoint_path: str = None, freeze_bn: bool = False):
        """we obey following ideas in `transform learning trainer`

        Ideas:
            1. first reshape fully connect layer of teacher state_dict
            2. load reshaped state_dict
            3. set `requires_grad = False` for all parameters in model
            4. set `requires_grad = True` for parameters in last `k` blocks
            5. reset parameters of last `k` blocks
        """
        super().__init__(model, train_loader, test_loader, checkpoint_path)

        if not checkpoint_path or not os.path.exists(checkpoint_path):
            teacher_state_dict = torch.load(teacher_model_path, map_location=self._device)
            self.reshape_teacher_fc_layer(teacher_state_dict)
            logger.info(f"load from teacher model: \n {teacher_model_path}")
            self.model.load_state_dict(teacher_state_dict)
        else:
            logger.info("load from old checkpoint, no need for teacher model!")

        self._blocks = make_blocks(model)

        self.freeze_model()
        # fixme
        # if re-initialize influence?
        self.unfreeze_last_k_blocks(k)
        if freeze_bn:
            self.freeze_bn_layer()

        logger.debug("trainable layers")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                logger.debug(f"name: {name}, size: {param.size()}")

        self.summary_writer = self.init_writer()


if __name__ == '__main__':
    from src.networks import wrn34_10
    from src.utils import get_cifar_test_dataloader, get_cifar_train_dataloader

    trainer = TransferLearningTrainer(
        k=6,
        teacher_model_path="./trained_models/cifar100_pgd7_train-best",
        model=wrn34_10(num_classes=10),
        train_loader=get_cifar_train_dataloader("cifar10"),
        test_loader=get_cifar_test_dataloader("cifar10"),
    )
from torch.utils.data import DataLoader

from ....src.trainer import NormalTrainer
from .mixins import ResetBlockMixin, FreezeModelMixin
from ....src.networks import WRNBlocks
from ....src.networks import SupportedWideResnetType


class RetrainTrainer(NormalTrainer, ResetBlockMixin, FreezeModelMixin):

    def __init__(self, k: int, model: SupportedWideResnetType, train_loader: DataLoader,
                 test_loader: DataLoader, checkpoint_path: str = None):
        """initialize retrain test_trainer

        Args:
            k: the last k blocks which will be retrained
        """
        super().__init__(model, train_loader, test_loader, checkpoint_path)
        self._blocks = WRNBlocks(model)
        self.freeze_model()
        self.reset_and_unfreeze_last_k_blocks(k)


if __name__ == '__main__':
    from src.networks import wrn34_10

    model = WRNBlocks(wrn34_10())
    print(model.block15)
    # print(model)

"""`learning without forgetting` in transfer learning

training steps:
    1. PGD-7 adversarial training on teacher model(e.g. cifar100)
    2. initialize student model from robust teacher model(with reshaped fc layer)
    3. calculate feature representations of **student dataset**(e.g. cifar10 dataset) with initialized student model
    4. store feature representations in memory
        - custom defined Dataloader could be used
    5. use loss: f(x, y_hat) +
                 λ * torch.mean(torch.norm(stored feature representations - running feature representations, p=1, dim=1))
       to train student model with benign student dataset(e.g. cifar10 dataset)
        - in warm-start step only train fully connect(last) layer
        - after warm-start step, train whole model

Notations:
    1. We use the same hyperparameters as `https://github.com/ashafahi/RobustTransferLWF`
"""


from torch.utils.data import Dataset, DataLoader
import torch
from torch import optim
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from typing import Tuple
import time
import os
import json

from ....src.networks import SupportedAllModuleType
from ....src.utils import evaluate_accuracy
from .mixins import ReshapeTeacherFCLayerMixin
from ..mixins import InitializeTensorboardMixin
from ..retrain_trainer import ResetBlockMixin, FreezeModelMixin
from ....src.networks import make_blocks
from ....src.utils import logger
from ....src import settings


# hyperparameters
_WEIGHT_DECAY = 0.0002
# approximately equals 20000 steps
_TRAIN_EPOCHS = 52
_WARM_START_EPOCHS = 26
_LEARNING_RATE = 0.001
_MOMENTUM = 0.9


class LWFTransferLearningTrainer(ReshapeTeacherFCLayerMixin, ResetBlockMixin,
                                 FreezeModelMixin, InitializeTensorboardMixin):

    def __init__(self, _lambda: float, teacher_model_path: str,
                 model: SupportedAllModuleType, train_loader: DataLoader,
                 test_loader: DataLoader, checkpoint_path: str = None):
        """`learning without forgetting` in transfer learning

        Args:
            _lambda: feature representation similarity penalty
        """
        logger.info("initialize trainer")
        # can not change the order
        self._init_hyperparameters()
        self._init_model(model)

        # load state_dict from robust teacher model
        # this process must above `_init_dataloader` method
        teacher_state_dict = torch.load(teacher_model_path, map_location=self._device)
        self.reshape_teacher_fc_layer(teacher_state_dict)
        logger.info(f"load from teacher model: \n {teacher_model_path}")
        self.model.load_state_dict(teacher_state_dict)

        self._init_dataloader(train_loader, test_loader)
        self._init_optimizer()
        self._init_criterion()

        if checkpoint_path:
            self._checkpoint_path = checkpoint_path
            if os.path.exists(checkpoint_path):
                self._load_from_checkpoint(checkpoint_path)
            else:
                self.start_epoch = 1
                # best accuracy of current model
                self.best_acc = 0

        self._blocks = make_blocks(self.model)
        # reset fc layer
        self.reset_last_k_blocks(1)

        self._lambda = _lambda

        self.summary_writer = self.init_writer()

        logger.info("initialize finished")
        self.print_parameters()

    def step_batch(self, inputs: Tuple[Tensor, Tensor], labels: torch.Tensor, optimizer) -> Tuple[float, float]:
        inputs, robust_feature_representations = inputs[0].to(self._device), inputs[1].to(self._device)
        labels = labels.to(self._device)

        optimizer.zero_grad()

        outputs = self.model(inputs)
        running_feature_representations = self.model.get_feature_representations()

        feature_representations_distance = self._lambda * \
               torch.mean(torch.norm(robust_feature_representations - running_feature_representations, p=1, dim=1))

        loss = self.criterion(outputs, labels) + feature_representations_distance

        loss.backward()
        optimizer.step()

        batch_training_acc = (outputs.argmax(dim=1) == labels).float().mean().item()
        batch_running_loss = loss.item()

        return batch_running_loss, batch_training_acc


    def _init_optimizer(self) -> None:
        """override `_init_optimizer` of super class

        we provide two optimizers, `fc_optimizer` and `all_optimizer`
            - `fc_optimizer`: only train fc layer, use for warm-start
            - `all_optimizer`: train all layers
        """

        self.fc_optimizer = optim.SGD(
            self.model.fc.parameters(),
            lr=self._lr,
            momentum=_MOMENTUM,
            weight_decay=_WEIGHT_DECAY
        )

        self.all_optimizer = optim.SGD(
            self.model.parameters(),
            lr=self._lr,
            momentum=_MOMENTUM,
            weight_decay=_WEIGHT_DECAY
        )

    def train(self, save_path):
        """override `train` of super class

        if current epochs < warm-start epochs, we should use optimizer `fc_optimizer`,
        otherwise, we should use optimizer `all_optimizer`
        """
        batch_number = len(self._train_loader)
        best_acc = self.best_acc
        start_epoch = self.start_epoch

        logger.info(f"starting epoch: {start_epoch}")
        logger.info(f"lr: {self._lr}")
        logger.info(f"best accuracy: {best_acc}")

        only_fc_unfreezed_flag = False
        for ep in range(start_epoch, self._train_epochs + 1):

            if ep < self._warm_start_epochs:
                if not only_fc_unfreezed_flag:
                    # freeze all layers except fc layer
                    self.freeze_model()
                    self.unfreeze_last_k_blocks(1)
                    only_fc_unfreezed_flag = True
                optimizer = self.fc_optimizer
            else:
                if only_fc_unfreezed_flag:
                    # unfreeze all layers
                    self.unfreeze_model()
                    only_fc_unfreezed_flag = False
                optimizer = self.all_optimizer

            # show current learning rate
            logger.debug(f"lr: {self._lr}")

            training_acc, running_loss = 0, .0
            start_time = time.perf_counter()

            for index, data in enumerate(self._train_loader):
                batch_running_loss, batch_training_acc = self.step_batch(data[0], data[1], optimizer)

                training_acc += batch_training_acc
                running_loss += batch_running_loss

                if index % batch_number == batch_number - 1:
                    end_time = time.perf_counter()

                    acc = self.test()
                    average_train_loss = (running_loss / batch_number)
                    average_train_accuracy = training_acc / batch_number
                    epoch_cost_time = end_time - start_time

                    # write loss, time, test_acc, train_acc to tensorboard
                    if hasattr(self, "summary_writer"):
                        self.summary_writer: SummaryWriter
                        self.summary_writer.add_scalar("train loss", average_train_loss, ep)
                        self.summary_writer.add_scalar("train accuracy", average_train_accuracy, ep)
                        self.summary_writer.add_scalar("test accuracy", acc, ep)
                        self.summary_writer.add_scalar("time per epoch", epoch_cost_time, ep)

                    logger.info(
                        f"epoch: {ep}   loss: {average_train_loss:.6f}   train accuracy: {average_train_accuracy}   "
                        f"test accuracy: {acc}   time: {epoch_cost_time:.2f}s")

                    if best_acc < acc:
                        best_acc = acc
                        self._save_best_model(save_path, ep, acc)

            self._save_checkpoint(ep, best_acc)

        logger.info("finished training")
        logger.info(f"best accuracy on test set: {best_acc}")

    def test(self):
        return evaluate_accuracy(self.model, self._test_loader, self._device)

    def _save_checkpoint(self, current_epoch, best_acc):
        model_weights = self.model.state_dict()

        torch.save({
            "model_weights": model_weights,
            "fc_optimizer": self.fc_optimizer.state_dict(),
            "all_optimizer": self.all_optimizer.state_dict(),
            "current_epoch": current_epoch,
            "best_acc": best_acc
        }, f"{self._checkpoint_path}")

        # Added by imTyrant
        # For saving 'numpy' and 'torch' random state.
        if hasattr(settings, "save_rand_state") and settings.save_rand_state:
            from src.utils import RandStateSnapshooter
            RandStateSnapshooter.lazy_take(f"{self._checkpoint_path}.rand")
            logger.debug(f"random state is saved to '{self._checkpoint_path}.rand'")

    def _save_best_model(self, save_path, current_epochs, accuracy):
        """save best model with current info"""
        info = {
            "current_epochs": current_epochs,
            "total_epochs": self._train_epochs,
            "best_accuracy": accuracy
        }
        suffix = save_path.split("/")[-1]
        with open(os.path.join(os.path.dirname(save_path), f"{suffix}_info.json"), "w", encoding="utf8") as f:
            json.dump(info, f)
        torch.save(self.model.state_dict(), f"{save_path}-best")

    def _load_from_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint.get("model_weights"))
        self.fc_optimizer.load_state_dict(checkpoint.get("fc_optimizer"))
        self.all_optimizer.load_state_dict(checkpoint.get("all_optimizer"))
        start_epoch = checkpoint.get("current_epoch") + 1
        best_acc = checkpoint.get("best_acc")

        self.start_epoch = start_epoch
        self.best_acc = best_acc

        # Added by imTyrant
        # For loading and setting random state.
        if hasattr(settings, "save_rand_state") and settings.save_rand_state:
            from src.utils import RandStateSnapshooter
            
            if not os.path.exists(f"{self._checkpoint_path}.rand"):
                # Since no deterministically resuming is not consierred, previously, 
                # '.rand' file may not exist.
                return

            RandStateSnapshooter.lazy_set(f"{self._checkpoint_path}.rand")
            # imTyrant: High logging level is for notification.
            logger.warning("loaded random state from '{self._checkpoint_path}.rand'")

    def _init_hyperparameters(self):
        self._lr = _LEARNING_RATE
        self._batch_size = settings.batch_size
        self._train_epochs = _TRAIN_EPOCHS
        self._warm_start_epochs = _WARM_START_EPOCHS
        self._device = torch.device(settings.device if torch.cuda.is_available() else "cpu")

    def _init_criterion(self):
        self.criterion = getattr(torch.nn, settings.criterion)()

    def _init_model(self, model):
        model.to(self._device)
        self.model = model

    def _init_dataloader(self, train_loader: DataLoader, test_loader: DataLoader):
        # precalculate robustness feature representations
        dataset_with_rft = DatasetWithRobustFeatureRepresentations(train_loader, self.model, self._device)
        self._train_loader = DataLoader(dataset_with_rft, batch_size=settings.batch_size,
                                        num_workers=settings.num_worker, shuffle=True)
        self._test_loader = test_loader

    def print_parameters(self):
        params = {
            "network": type(self.model).__name__,
            "device": str(self._device),
            "train_epochs": str(self._train_epochs),
            "warm_start_epochs": str(self._warm_start_epochs),
            "batch_size": str(self._batch_size),
            "fc_optimizer": str(self.fc_optimizer),
            "all_optimizer": str(self.all_optimizer),
            "criterion": str(self.criterion),
            "lambda": str(self._lambda)
        }
        params_str = "\n".join([": ".join(item) for item in params.items()])

        logger.info(f"training hyperparameters: \n{params_str}")


class DatasetWithRobustFeatureRepresentations(Dataset):

    def __init__(self, origin_train_loader: DataLoader, model: SupportedAllModuleType, device: torch.device):
        """extend origin dataset with robust feature representations

        Args:
            origin_train_loader: origin train loader
            model: untrained robust model
        """
        logger.info("precalculate robust feature representations")

        self._dataset_len = len(origin_train_loader.dataset)

        inputs_tensor_list = []
        feature_representations_tensor_list = []
        labels_tensor_list = []

        model.eval()
        for inputs, labels in origin_train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs_tensor_list.append(inputs.detach().cpu())
            model(inputs)
            robust_feature_representations = model.get_feature_representations()
            # todo
            # cost lots of gpu
            feature_representations_tensor_list.append(robust_feature_representations.detach().cpu())
            labels_tensor_list.append(labels.detach().cpu())

        self._inputs = torch.cat(inputs_tensor_list, dim=0)
        self._feature_representations = torch.cat(feature_representations_tensor_list, dim=0)
        self._labels = torch.cat(labels_tensor_list, dim=0)

        logger.debug(f"dataset inputs shape: {self._inputs.shape}")
        logger.debug(f"dataset feature representations shape: {self._feature_representations.shape}")
        logger.debug(f"dataset labels shape: {self._labels.shape}")

        model.train()

        logger.info("calculate done")

    def __getitem__(self, idx) -> Tuple[Tuple[Tensor, Tensor], Tensor]:
        inputs = self._inputs[idx]
        feature_representations = self._feature_representations[idx]
        labels = self._labels[idx]

        return (inputs, feature_representations), labels

    def __len__(self):
        return self._dataset_len


if __name__ == '__main__':
    from src.networks import resnet18
    from src.utils import (get_mnist_train_dataloader, get_mnist_test_dataloader,
                           get_subset_cifar_train_dataloader)
    from src import settings

    _lambda = 0.1

    save_path = f"normalization_svhn_lwf_tl_pgd7_lambda{_lambda}"
    logger.change_log_file(settings.log_dir / f"{save_path}.log")

    model = resnet18(num_classes=10)
    teacher_model_path = str(settings.root_dir / "trained_models/svhn_pgd7_train-best")
    trainer = LWFTransferLearningTrainer(
        _lambda=_lambda,
        teacher_model_path=teacher_model_path,
        model=model,
        train_loader=get_mnist_train_dataloader(),
        test_loader=get_mnist_test_dataloader(),
        checkpoint_path=str(settings.root_dir / "checkpoint" / f"{save_path}.pth")
    )

    trainer.train(str(settings.root_dir / "trained_models" / save_path))
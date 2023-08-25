from typing import Dict
import time

import torch
from torch.utils.data import DataLoader

from .base_trainer import BaseTrainer
from ...src.networks import SupportedAllModuleType
from ...src.utils import logger


class BaseADVTrainer(BaseTrainer):
    def __init__(self, model: SupportedAllModuleType, train_loader: DataLoader,
                 test_loader: DataLoader, attacker, params: Dict,
                 checkpoint_path: str = None):
        super().__init__(model, train_loader, test_loader, checkpoint_path)
        self.attacker = self._init_attacker(attacker, params)

    def _init_attacker(self, attacker, params):
        raise NotImplementedError("must overwrite method `init_attacker`")

    def train(self, save_path):
        batch_number = len(self._train_loader)
        best_robustness = self.best_acc
        start_epoch = self.start_epoch

        logger.info(f"starting epoch: {start_epoch}")
        logger.info(f"start lr: {self.current_lr}")
        logger.info(f"best robustness: {best_robustness}")

        for ep in range(start_epoch, self._train_epochs + 1):

            self._adjust_lr(ep)

            # show current learning rate
            logger.debug(f"lr: {self.current_lr}")

            training_acc, running_loss = 0, .0
            start_time = time.perf_counter()

            for index, data in enumerate(self._train_loader):
                batch_running_loss, batch_training_acc = self.step_batch(data[0], data[1])

                training_acc += batch_training_acc
                running_loss += batch_running_loss

                # warm up learning rate
                if ep <= self._warm_up_epochs:
                    self.warm_up_scheduler.step()

                if index % batch_number == batch_number - 1:
                    end_time = time.perf_counter()

                    acc = self.test()
                    average_train_loss = (running_loss / batch_number)
                    average_train_accuracy = training_acc / batch_number
                    epoch_cost_time = end_time - start_time

                    logger.info(
                        f"epoch: {ep}   loss: {average_train_loss:.6f}   train accuracy: {average_train_accuracy}   "
                        f"test accuracy: {acc}   time: {epoch_cost_time:.2f}s")

                    if best_robustness < average_train_accuracy:
                        best_robustness = average_train_accuracy
                        logger.info(f"better robustness: {best_robustness}")
                        logger.info(f"corresponding accuracy on test set: {acc}")
                        self._save_model(f"{save_path}-best_robust")

            self._save_checkpoint(ep, best_robustness)

        logger.info("finished training")
        logger.info(f"best robustness on test set: {best_robustness}")

        self._save_last_model(f"{save_path}-last") # imTyrant added it for saving last model.

class ADVTrainer(BaseADVTrainer):
    def __init__(self, model: SupportedAllModuleType, train_loader: DataLoader,
                 test_loader: DataLoader, attacker, params: Dict,
                 checkpoint_path: str = None):
        super().__init__(model, train_loader, test_loader, attacker, params, checkpoint_path)

    def _init_attacker(self, attacker, params):
        attacker = attacker(self.model, **params)
        attacker.print_parameters()

        return attacker

    def step_batch(self, inputs: torch.Tensor, labels: torch.Tensor):
        inputs, labels = inputs.to(self._device), labels.to(self._device)

        self._freeze_all_layers()
        adv_inputs = self._gen_adv(inputs, labels)
        self._unfreeze_all_layers()
        
        outputs = self.model(adv_inputs)
        loss = self.criterion(outputs, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        batch_training_acc = (outputs.argmax(dim=1) == labels).float().mean().item()
        batch_running_loss = loss.item()

        return batch_running_loss, batch_training_acc

    def _gen_adv(self, inputs: torch.Tensor, labels: torch.Tensor):
        self.model.eval()

        adv_inputs = self.attacker.calc_perturbation(inputs, labels)
        adv_inputs = adv_inputs.to(self._device)

        self.model.train()

        return adv_inputs
    
    def _unfreeze_all_layers(self):
        for p in self.model.parameters():
            p.requires_grad = True

        # logger.debug(f"all parameters of model are unfreezed")

    def _freeze_all_layers(self):
        for p in self.model.parameters():
            p.requires_grad = False

        # logger.debug(f"all parameters of model are freezed")

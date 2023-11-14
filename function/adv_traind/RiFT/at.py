from .gen_adv import get_at_dataloader
from tqdm import tqdm
import numpy as np
from torch import nn, optim
import torch
from os.path import join, dirname
from torch.utils.data import DataLoader
from .dataloader import *
from .model import create_model
from copy import deepcopy


class AdversarialTraining(object):
    def __init__(self, model, dataloader, method='pgd', eps=8, device='cuda', batch_size=128,
                 train_epoch=10, at_epoch=5, save_path='', save_name=''):
        self.method = method
        self.eps = eps
        self.model = deepcopy(model)
        self.dataloader = dataloader
        self.device = device
        self.train_epoch = train_epoch
        self.batch_size = batch_size
        self.at_epoch = at_epoch
        self.save_path = save_path
        self.save_name = save_name

    def train_single_epoch(self,
                           dataloader: DataLoader,
                           lr_scheduler,
                           loss_func,
                           optimizer,
                           epoch: int):
        self.model.train()
        criterion = loss_func
        train_loss = list()
        train_acc = list()
        loop = tqdm(enumerate(dataloader), ncols=100, desc=f'Train epoch {epoch}', total=len(dataloader),
                    colour='blue')
        for batch_idx, data in loop:
            images, labels = data
            images, labels = images.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            mini_out = self.model(images)
            mini_loss = criterion(mini_out, labels.long())
            mini_loss.backward()
            optimizer.step()

            _, pred = torch.max(mini_out, 1)
            acc = (pred.data == labels).float().mean()
            train_loss.append(float(mini_loss))
            train_acc.append(float(acc))

            loop.set_postfix({"Loss": f'{np.array(train_loss).mean():.6f}',
                              "Acc": f'{np.array(train_acc).mean():.6f}'})

        torch.cuda.empty_cache()
        lr_scheduler.step(epoch=epoch)
        return np.array(train_acc).mean(), np.array(train_loss).mean()

    def train_at_epoch(self, at_epoch):
        dataloader, _, acc = get_at_dataloader(method=self.method,
                                          model=self.model,
                                          dataloader=self.dataloader,
                                          eps=self.eps,
                                          device=self.device,
                                          batch_size=self.batch_size)
        if acc > 0.95:
            return True
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adamax(self.model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.005)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [30, 70, 150], gamma=0.1)
        for epoch in tqdm(range(self.train_epoch), ncols=100, desc=f'train at_epoch: {at_epoch}'):
            acc, _ = self.train_single_epoch(dataloader=dataloader,
                                             lr_scheduler=lr_scheduler,
                                             loss_func=criterion,
                                             optimizer=optimizer,
                                             epoch=epoch)
            if acc > 0.99:
                break
        # return self.model
        return False

    def train(self):
        for epoch in tqdm(range(self.at_epoch), ncols=100):
            if self.train_at_epoch(epoch):
                break
        print('==> Saving checkpoints...')
        state = {
            'model': self.model.state_dict(),
            'acc': '',
            'epoch': epoch,
        }
        torch.save(state, join(self.save_path, self.save_name))
        return self.model
    

if __name__ == '__main__':
    trainloader, testloader = create_dataloader(128, dataset='cifar10')
    save_path = join(dirname(__file__), 'results')
    model = create_model(model_name='ResNet18', num_classes=10, device='cuda', resume=None)
    at = AdversarialTraining(model=model, dataloader=testloader,  method='fgsm', eps=8, device='cuda',
                             batch_size=128, train_epoch=1, at_epoch=1, save_path=save_path)
    results = at.train()
    print(results)



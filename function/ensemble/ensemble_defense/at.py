from function.ensemble.attack.gen_adv import get_at_dataloader
from tqdm import tqdm
import numpy as np
import os
import json
from torch import nn, optim
import torchvision,torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


class AdversarialTraining(object):
    def __init__(self, method, attackparam, model, dataloader, device='cuda', batch_size=128, train_epoch=10, at_epoch=5, dataset='mnist'):
        self.method = method
        self.attackparam = attackparam
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.train_epoch = train_epoch
        self.batch_size = batch_size
        self.at_epoch = at_epoch
        self.dataset = dataset

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
                                          attackparam=self.attackparam,
                                          device=self.device,
                                          batch_size=self.batch_size,
                                          dataset = self.dataset)
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
        return self.model
    

if __name__ == '__main__':
    pass



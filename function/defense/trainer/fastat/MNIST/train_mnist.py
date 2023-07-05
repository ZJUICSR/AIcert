
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys
sys.path.append('../../../../..')
from function.defense.models import SmallCNN
def main():
    train_config = {'epochs': 1, 'alpha': 0.01, 'epsilon': 0.3, 'attack_iters': 40, 'lr_max': 0.0001, 'lr_type': 'flat'}
    train_config = {'epochs': 1, 'alpha': 0.01, 'epsilon': 8. / 255., 'attack_iters': 7, 'lr_max': 0.2, 'lr_type': 'flat'}
    mnist_train = datasets.MNIST("./dataset", train=True, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(mnist_train, batch_size=100, shuffle=True)
    transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
    trainset = torchvision.datasets.CIFAR10(root='./dataset/CIFAR10', train=True, download=True, transform=transform_train)
    kwargs = {'num_workers': 1, 'pin_memory': True} if True else {}
    train_loader = DataLoader(trainset, batch_size=64, shuffle=True, **kwargs)
    

    model = SmallCNN().cuda()
    model.train()

    lr_steps = train_config['epochs'] * len(train_loader)
    if train_config['lr_type'] == 'cyclic' and dataset == 'CIFAR10':
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=0, max_lr=train_config['lr_max'],
            step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif train_config['lr_type'] == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)
    elif train_config['lr_type'] == 'cyclic' and dataset = 'MNIST': 
        lr_schedule = lambda t: np.interp([t], [0, train_config['epochs'] * 2//5, train_config['epochs']], [0, train_config['lr_max'], 0])[0]
    elif train_config['lr_type'] == 'flat': 
        lr_schedule = lambda t: train_config['lr_max']
    else:
        raise ValueError('Unknown lr_type')
    opt = torch.optim.SGD(model.parameters(), lr=train_config['lr_max'], momentum=0.9, weight_decay=5e-4)

    criterion = nn.CrossEntropyLoss()
    for epoch in range(train_config['epochs']):
        print(epoch)
        for i, (X, y) in enumerate(train_loader):
            X, y = X.cuda(), y.cuda()
            lr = lr_schedule(epoch + (i+1)/len(train_loader))
            opt.param_groups[0].update(lr=lr)
            delta = torch.zeros_like(X).uniform_(-train_config['epsilon'], train_config['epsilon'])
            delta.data = torch.max(torch.min(1-X, delta.data), 0-X)
            for _ in range(train_config['attack_iters']):
                delta.requires_grad = True
                output = model(X + delta)
                loss = criterion(output, y)
                opt.zero_grad()
                loss.backward()
                grad = delta.grad.detach()
                I = output.max(1)[1] == y
                delta.data[I] = torch.clamp(delta + train_config['alpha'] * torch.sign(grad), -train_config['epsilon'], train_config['epsilon'])[I]
                delta.data[I] = torch.max(torch.min(1-X, delta.data), 0-X)[I]
            delta = delta.detach()
            output = model(torch.clamp(X + delta, 0, 1))
            loss = criterion(output, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

if __name__ == "__main__":
    main()

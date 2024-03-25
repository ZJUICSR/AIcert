'''DenseNet in PyTorch.
https://github.com/kuangliu/pytorch-cifar
'''

import os
import math
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from function.formal_verify.auto_LiRPA_verifiy.vision.data import get_cifar_data


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4 * growth_rate, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        # out = self.conv1(F.relu(x))
        # out = self.conv2(F.relu(out))
        out = torch.cat([out, x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=True)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=True)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2] * growth_rate
        # out_planes = int(math.floor(num_planes*reduction))
        # self.trans3 = Transition(num_planes, out_planes)
        # num_planes = out_planes

        # self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        # num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear1 = nn.Linear(14336, 512)
        self.linear2 = nn.Linear(512, num_classes)


    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        # out = self.dense4(out)
        out = F.relu(self.bn(out))
        out = out.view(out.size(0), -1)
        out = F.relu(self.linear1(out))
        out = self.linear2(out)

        return out


def Densenet_cifar_32(in_ch=3, in_dim=32):
    return DenseNet(Bottleneck, [2, 4, 4], growth_rate=32)


def train(model, train_loader, val_loader, epoches, save_name, device='cuda'):
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
    criterion = nn.CrossEntropyLoss()
    train_loss = list()
    train_acc = list()
    val_loss = list()
    val_acc = list()
    for epoch in range(epoches):
        model.train()
        loop = tqdm(train_loader, desc=f'train modal @epoch {epoch + 1}', ncols=150)
        for data, target in loop:
            data, target = Variable(data), Variable(target)
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            max_index = output.max(dim=1)[1]
            correct = (max_index == target).cpu().float().mean()
            train_loss.append(float(loss.cpu()))
            train_acc.append(correct)

            loop.set_postfix({'TrainLoss': f'{np.array(train_loss).mean():.6f}',
                              'TrainAcc': f'{np.array(train_acc).mean():.6f}'})
        # print(fr'Training set: Average loss: {(training_loss / len(train_loader.dataset)):.4f}, Accuracy: {correct}/{len(train_loader.dataset)} ({(100. * correct / len(train_loader.dataset)):.0f})%')

        if val_loader is None:
            save_model(model, epoch, save_name)
            continue

        # 评估模型
        model.eval()
        validation_loss = 0
        correct = 0
        loop_val = tqdm(val_loader, desc=f'var modal @epoch {epoch + 1}', ncols=80)
        for data, target in loop_val:
            with torch.no_grad():
                data, target = Variable(data), Variable(target)
                data, target = data.to(device), target.to(device)
                output = model(data)
                validation_loss = F.nll_loss(output, target, size_average=False).data.item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct = pred.eq(target.data.view_as(pred)).cpu().float().mean()
                val_loss.append(validation_loss)
                val_acc.append(correct)
            loop_val.set_postfix({'ValLoss': f'{np.array(val_loss).mean():.6f}',
                                  'ValAcc': f'{np.array(val_acc).mean():.6f}'})

        validation_loss /= len(val_loader.dataset)
        scheduler.step()
        save_model(model, epoch, save_name)


def save_model(model, epoch, path):
    check_point = {"epochs": epoch, "model": model.state_dict()}
    torch.save(check_point, path)


def load_model(path):
    model = Densenet_cifar_32()
    epochs = 0
    if os.path.exists(path):
        check_point = torch.load(path)
        model.load_state_dict(check_point['model'])
        epochs = check_point['epochs']
    return model, epochs


def main(device='cuda'):
    dataloader, _ = get_cifar_data(number=-1, batch_size=128, device=device)
    model = Densenet_cifar_32()
    model.to(device)
    train(model=model, train_loader=dataloader, val_loader=None, save_name=f'model_densenet_cifar.pth', epoches=60)




if __name__ == "__main__":
    main()

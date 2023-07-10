import math
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.nn import Module
from torch.utils.data import DataLoader
from torchvision.models import vgg16
from typing import List, Optional, Union
from art.defences.trainer import *
from function.defense.trainer.trades import * #
from function.defense.trainer.freeat import freeat_train, freeat_adjust_learning_rate
from function.defense.trainer.mart import * #
from function.defense.trainer.madry import * #
from function.defense.trainer.cartl.cartl import * #
from function.defense.models import *
from function.defense.utils.generate_aes import generate_adv_examples

class at(object):
    def __init__(self, model:Module,
                mean:List[float]=[0.485, 0.456, 0.406],
                std:List[float]=[0.229, 0.224, 0.225],
                adv_examples=None,
                adv_method: Optional[str] = 'PGD',
                adv_dataset: Optional[str] = 'CIFAR10',
                adv_nums: Optional[int] = 1,#
                device:Union[str, torch.device]='cuda',
                ):

        super(at, self).__init__()

        self.model = model
        self.norm = transforms.Normalize(mean, std)
        if adv_examples is None and (adv_method is None or adv_nums is None):
            raise Exception('Attack method and adversarial example nums need to be specified when the adversarial sample is not specified!')
        self.adv_examples = adv_examples
        self.adv_method = adv_method
        self.adv_dataset = adv_dataset
        self.adv_nums = adv_nums
        self.device = device
        # self.un_norm = UnNorm(mean, std)
        self.total_num = 0
        self.detect_num = 0
        self.detect_rate = 0
        self.no_defense_accuracy = 0

    def generate_adv_examples(self):
        return generate_adv_examples(self.model, self.adv_method, self.adv_dataset, self.adv_nums, self.device)

    def load_adv_examples(self):
        data = torch.load(self.adv_examples)
        print('successfully load adversarial examples!')
        return data['adv_img'], data['cln_img'], data['y']

    def dataset(self):
        print("Step 1: Load the {} dataset".format(self.adv_dataset))
        kwargs = {'num_workers': 1, 'pin_memory': True} if True else {}
        if self.adv_dataset == 'MNIST':
            if self.model.__class__.__name__ == 'VGG':
                transform_train = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                ])
            else:
                transform_train = transforms.Compose([
                    transforms.ToTensor(),
                ])
            trainset = torchvision.datasets.MNIST(root='/mnt/data2/yxl/AI-platform/dataset', train=True, download=True, transform=transform_train)
            train_loader = DataLoader(trainset, batch_size=128, shuffle=True, **kwargs)
        elif self.adv_dataset == 'CIFAR10':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
            trainset = torchvision.datasets.CIFAR10(root='/mnt/data2/yxl/AI-platform/dataset/CIFAR10', train=True, download=True, transform=transform_train)
            train_loader = DataLoader(trainset, batch_size=128, shuffle=True, **kwargs)

        return train_loader

    def train(self, at_method):
        if self.adv_examples is None:
            adv_imgs, cln_imgs, true_labels = self.generate_adv_examples()
        else:
            adv_imgs, cln_imgs, true_labels = self.load_adv_examples() 
        with torch.no_grad():
            adv_predictions = self.model(adv_imgs)
        no_defense_accuracy = torch.sum(torch.argmax(adv_predictions, dim = 1) == true_labels) / float(len(adv_imgs))
        self.no_defense_accuracy = no_defense_accuracy.cpu().numpy()
        train_loader = self.dataset()
        print("Step 2: Create the model")
        if self.adv_dataset == 'CIFAR10':
            if self.model.__class__.__name__ == 'ResNet':
                model = ResNet18()
            elif self.model.__class__.__name__ == 'VGG':
                model = vgg16()
                model.classifier[6] = nn.Linear(4096, 10)
            else:
                raise Exception('CIFAR10 can only use ResNet18 and VGG16!')
            model = model.to(self.device)
            train_config = {
                'epsilon': 0.031,
                'num_steps': 10,
                'step_size': 0.007,
            }
        elif self.adv_dataset == 'MNIST':
            if self.model.__class__.__name__ == 'SmallCNN':
                model = SmallCNN()
            elif self.model.__class__.__name__ == 'VGG':
                model = vgg16()
                model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
                model.classifier[6] = nn.Linear(4096, 10)
            else:
                raise Exception('MNIST can only use SmallCNN and VGG16!')
            model = model.to(self.device)
            train_config = {
                'epsilon': 0.3,
                'num_steps': 40,
                'step_size': 0.01,
            }

        print("Step 2a: Define the optimizer")
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        print("Step 4: Train the {} classifier".format(at_method))
        train_method = None
        if at_method == 'TRADES':
            train_method = trades_train
        elif at_method == 'Madry':
            train_method = madry_train
        elif at_method == 'MART':
            train_method = mart_train
        for epoch in range(1, 2):
            adjust_learning_rate(optimizer, epoch)
            train_method(model, self.device, train_loader, optimizer, epoch, train_config)

        with torch.no_grad():
            model.eval()
            predictions = model(cln_imgs)
            predictions_adv = model(adv_imgs)
        acc = torch.sum(torch.argmax(predictions, dim = 1) == true_labels) / float(len(adv_imgs))
        print('acc: ', float(acc.cpu()))
        detect_rate = torch.sum(torch.argmax(predictions_adv, dim = 1) == true_labels) / float(len(adv_imgs))
        self.detect_rate = float(detect_rate.cpu())
        if self.adv_examples is None:
            attack_method = 'from user'
        else:
            attack_method = self.adv_method

        return attack_method, self.detect_num, self.detect_rate, self.no_defense_accuracy

    def print_res(self):
        print('detect rate: ', self.detect_rate)

class Madry(at):
    def __init__(self, model, mean, std, adv_examples, adv_method, adv_dataset, adv_nums, device):
        super().__init__(model = model, mean = mean, std = std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)

    def detect(self):
        return self.train('Madry')

class Trades(at):
    def __init__(self, model, mean, std, adv_examples, adv_method, adv_dataset, adv_nums, device):
        super().__init__(model = model, mean = mean, std = std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)

    def detect(self):
        return self.train('TRADES')

class FreeAT(at):
    def __init__(self, model, mean, std, adv_examples, adv_method, adv_dataset, adv_nums, device):
        super().__init__(model = model, mean = mean, std = std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)

    def detect(self):
        if self.adv_examples is None:
            adv_imgs, cln_imgs, true_labels = self.generate_adv_examples()
        else:
            adv_imgs, cln_imgs, true_labels = self.load_adv_examples() 
        with torch.no_grad():
            adv_predictions = self.model(adv_imgs)
        no_defense_accuracy = torch.sum(torch.argmax(adv_predictions, dim = 1) == true_labels) / float(len(adv_imgs))
        self.no_defense_accuracy = no_defense_accuracy.cpu().numpy()
        train_loader = self.dataset()

        print("Step 2: Create the model")
        if self.adv_dataset == 'CIFAR10':
            if self.model.__class__.__name__ == 'ResNet':
                model = ResNet18()
            elif self.model.__class__.__name__ == 'VGG':
                model = vgg16()
                model.classifier[6] = nn.Linear(4096, 10)
            else:
                raise Exception('CIFAR10 can only use ResNet18 and VGG16!')
            model = model.to(self.device)
        elif self.adv_dataset == 'MNIST':
            if self.model.__class__.__name__ == 'SmallCNN':
                model = SmallCNN()
            elif self.model.__class__.__name__ == 'VGG':
                model = vgg16()
                model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
                model.classifier[6] = nn.Linear(4096, 10)
            else:
                raise Exception('MNIST can only use SmallCNN and VGG16!')
            model = model.to(self.device)

        print("Step 2a: Define the optimizer")
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        print("Step 4: Train the FreeAT classifier")
        args_epochs = 8
        n_repeats = 4
        epochs = int(math.ceil(args_epochs / n_repeats))
        criterion = nn.CrossEntropyLoss().cuda()
        if self.adv_dataset == 'CIFAR10':
            global_noise_data = torch.zeros([train_loader.batch_size, 3, 32, 32]).cuda()
        elif self.adv_dataset == 'MNIST':
            if self.model.__class__.__name__ == 'VGG':
                global_noise_data = torch.zeros([train_loader.batch_size, 1, 32, 32]).cuda()
            else:
                global_noise_data = torch.zeros([train_loader.batch_size, 1, 28, 28]).cuda()
        for epoch in range(0, epochs):
            freeat_adjust_learning_rate(0.1, optimizer, epoch, n_repeats)
            freeat_train(train_loader, model, criterion, optimizer, epoch, global_noise_data)

        with torch.no_grad():
            model.eval()
            predictions = model(cln_imgs)
            predictions_adv = model(adv_imgs)
        acc = torch.sum(torch.argmax(predictions, dim = 1) == true_labels) / float(len(adv_imgs))
        # print('acc: ', float(acc.cpu()))
        detect_rate = torch.sum(torch.argmax(predictions_adv, dim = 1) == true_labels) / float(len(adv_imgs))
        self.detect_rate = float(detect_rate.cpu())
        if self.adv_examples is None:
            attack_method = 'from user'
        else:
            attack_method = self.adv_method

        return attack_method, self.detect_num, self.detect_rate, self.no_defense_accuracy

class Mart(at):
    def __init__(self, model, mean, std, adv_examples, adv_method, adv_dataset, adv_nums, device):
        super().__init__(model = model, mean = mean, std = std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)

    def detect(self):
        return self.train('MART')
    
class FastAT(at):
    def __init__(self, model, mean, std, adv_examples, adv_method, adv_dataset, adv_nums, device):
        super().__init__(model = model, mean = mean, std = std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)

    def detect(self):
        if self.adv_examples is None:
            adv_imgs, cln_imgs, true_labels = self.generate_adv_examples()
        else:
            adv_imgs, cln_imgs, true_labels = self.load_adv_examples() 
        with torch.no_grad():
            adv_predictions = self.model(adv_imgs)
        no_defense_accuracy = torch.sum(torch.argmax(adv_predictions, dim = 1) == true_labels) / float(len(adv_imgs))
        self.no_defense_accuracy = no_defense_accuracy.cpu().numpy()
        train_loader = self.dataset()
        if self.adv_dataset == 'CIFAR10':
            if self.model.__class__.__name__ == 'ResNet':
                model = ResNet18()
            elif self.model.__class__.__name__ == 'VGG':
                model = vgg16()
                model.classifier[6] = nn.Linear(4096, 10)
            else:
                raise Exception('CIFAR10 can only use ResNet18 and VGG16!')
            model = model.to(self.device)
            train_config = {'epochs': 1, 'alpha': 0.01, 'epsilon': 0.3, 'attack_iters': 40, 'lr_max': 0.0001, 'lr_type': 'flat'}
        elif self.adv_dataset == 'MNIST':
            if self.model.__class__.__name__ == 'SmallCNN':
                model = SmallCNN()
            elif self.model.__class__.__name__ == 'VGG':
                model = vgg16()
                model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
                model.classifier[6] = nn.Linear(4096, 10)
            else:
                raise Exception('MNIST can only use SmallCNN and VGG16!')
            model = model.to(self.device)
            train_config = {'epochs': 1, 'alpha': 2, 'epsilon': 8. / 255., 'attack_iters': 7, 'lr_max': 0.2, 'lr_type': 'cyclic'}

        model.train()

        if train_config['lr_type'] == 'cyclic': 
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
                print(i)
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
        with torch.no_grad():
            model.eval()
            predictions = model(cln_imgs)
            predictions_adv = model(adv_imgs)
        acc = torch.sum(torch.argmax(predictions, dim = 1) == true_labels) / float(len(adv_imgs))
        print('acc: ', float(acc.cpu()))
        detect_rate = torch.sum(torch.argmax(predictions_adv, dim = 1) == true_labels) / float(len(adv_imgs))
        self.detect_rate = float(detect_rate.cpu())
        if self.adv_examples is None:
            attack_method = 'from user'
        else:
            attack_method = self.adv_method

        return attack_method, self.detect_num, self.detect_rate, self.no_defense_accuracy
    
class Cartl(at):
    def __init__(self, model, mean, std, adv_examples, adv_method, adv_dataset, adv_nums, device):
        super().__init__(model = model, mean = mean, std = std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)

    def generate_adv_examples(self):
        return generate_adv_examples(self.model, self.adv_method, self.adv_dataset, self.adv_nums, self.device, normalize=True)

    def detect(self):
        if self.adv_examples is None:
            adv_imgs, cln_imgs, true_labels = self.generate_adv_examples()
        else:
            adv_imgs, cln_imgs, true_labels = self.load_adv_examples() 
        with torch.no_grad():
            adv_predictions = self.model(adv_imgs)
        no_defense_accuracy = torch.sum(torch.argmax(adv_predictions, dim = 1) == true_labels) / float(len(adv_imgs))
        self.no_defense_accuracy = no_defense_accuracy.cpu().numpy()
        source_dataset = 'cifar100'
        source_num_classes = 100
        if self.adv_dataset == 'CIFAR10':
            target_dataset = 'cifar10'
            target_num_classes = 10
        elif self.adv_dataset == 'MNIST':
            target_dataset = 'mnist'
            target_num_classes = 10
            adv_imgs = adv_imgs.repeat(1, 3, 1, 1)
            cln_imgs = cln_imgs.repeat(1, 3, 1, 1)
        model = cartl(source_dataset, source_num_classes, target_dataset, target_num_classes)

        with torch.no_grad():
            model.eval()
            predictions = model(cln_imgs)
            predictions_adv = model(adv_imgs)
        acc = torch.sum(torch.argmax(predictions, dim = 1) == true_labels) / float(len(adv_imgs))
        print('acc: ', float(acc.cpu()))
        detect_rate = torch.sum(torch.argmax(predictions_adv, dim = 1) == true_labels) / float(len(adv_imgs))
        self.detect_rate = float(detect_rate.cpu())
        if self.adv_examples is None:
            attack_method = 'from user'
        else:
            attack_method = self.adv_method

        return attack_method, self.detect_num, self.detect_rate, self.no_defense_accuracy
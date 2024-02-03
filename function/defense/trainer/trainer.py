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
                    transforms.Normalize((0.1307, ), (0.3081, ))
                ])
            else:
                transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307, ), (0.3081, ))
                ])
            trainset = torchvision.datasets.MNIST(root='./dataset', train=True, download=True, transform=transform_train)
            train_loader = DataLoader(trainset, batch_size=128, shuffle=True, **kwargs)
        elif self.adv_dataset == 'CIFAR10':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010],)
            ])
            trainset = torchvision.datasets.CIFAR10(root='./dataset/CIFAR10', train=True, download=True, transform=transform_train)
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
        print('no defense accuracy: {:.4f}'.format(self.no_defense_accuracy))
        train_loader = self.dataset()
        print("Step 2: Create the model")
        if self.adv_dataset == 'CIFAR10':
            weight_decay = 2e-4
            if self.model.__class__.__name__ == 'ResNet':
                lr = 0.1
                model = ResNet18()
            elif self.model.__class__.__name__ == 'VGG':
                lr = 0.01
                model = vgg16()
                model.classifier[6] = nn.Linear(4096, 10)
            else:
                raise Exception('CIFAR10 can only use ResNet18 and VGG16!')
            model = model.to(self.device)
            train_config = {
                'epsilon': 0.031,
                'num_steps': 10,
                'step_size': 0.007,
                'beta': 6.0,
            }
        elif self.adv_dataset == 'MNIST':
            lr = 0.01
            weight_decay = 0
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
                'beta': 1.0,
            }

        print("Step 2a: Define the optimizer")
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        print("Step 4: Train the {} classifier".format(at_method))
        train_method = None
        if at_method == 'TRADES':
            train_method = trades_train
        elif at_method == 'Madry':
            train_method = madry_train
        elif at_method == 'MART':
            train_method = mart_train
        for epoch in range(1, 2):
            adjust_learning_rate(optimizer, epoch, lr)
            train_method(model, self.device, train_loader, optimizer, epoch, train_config)

        with torch.no_grad():
            model.eval()
            # predictions = model(cln_imgs)
            predictions_adv = model(adv_imgs)
        # acc = torch.sum(torch.argmax(predictions, dim = 1) == true_labels) / float(len(adv_imgs))
        # print('acc: ', float(acc.cpu()))
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
            cifar10_mean = (0.4914, 0.4822, 0.4465)
            cifar10_std = (0.2471, 0.2435, 0.2616)
            mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
            std = torch.tensor(cifar10_std).view(3,1,1).cuda()
            upper_limit = ((1 - mu)/ std)
            lower_limit = ((0 - mu)/ std)
            if self.model.__class__.__name__ == 'ResNet':
                model = ResNet18()
            elif self.model.__class__.__name__ == 'VGG':
                model = vgg16()
                model.classifier[6] = nn.Linear(4096, 10)
            else:
                raise Exception('CIFAR10 can only use ResNet18 and VGG16!')
            model = model.to(self.device)
            model.train()
            train_config = {'epochs': 1, 'alpha': 2 / 255. / std, 'epsilon': 8 / 255. / std, \
                            'attack_iters': 7, 'lr_max': 0.2, 'lr_min': 0., 'weight_decay': 5e-4}
            opt = torch.optim.SGD(model.parameters(), lr=train_config['lr_max'], momentum=0.9, weight_decay=train_config['weight_decay'])
            amp_args = dict(opt_level='O2', loss_scale='1.0', verbosity=False)
            amp_args['master_weights'] = None
            import apex.amp as amp
            model, opt = amp.initialize(model, opt, **amp_args)
            criterion = nn.CrossEntropyLoss()
            lr_steps = train_config['epochs'] * len(train_loader)
            scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=train_config['lr_min'], max_lr=train_config['lr_max'],
            step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
            epsilon = train_config['epsilon']
            alpha = train_config['alpha']
            def clamp(X, lower_limit, upper_limit):
                return torch.max(torch.min(X, upper_limit), lower_limit)
            for epoch in range(train_config['epochs']):
                print('epoch', epoch)
                for i, (X, y) in enumerate(train_loader):
                    if i % 100 == 0:
                        print(i)
                    X, y = X.cuda(), y.cuda()
                    delta = torch.zeros_like(X).cuda()
                    for i in range(len(epsilon)):
                        delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
                    delta.data = clamp(delta, lower_limit - X, upper_limit - X)
                    delta.requires_grad = True
                    for _ in range(train_config['attack_iters']):
                        output = model(X + delta)
                        loss = criterion(output, y)
                        with amp.scale_loss(loss, opt) as scaled_loss:
                            scaled_loss.backward()
                        grad = delta.grad.detach()
                        delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
                        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
                        delta.grad.zero_()
                    delta = delta.detach()
                    output = model(X + delta)
                    loss = criterion(output, y)
                    opt.zero_grad()
                    with amp.scale_loss(loss, opt) as scaled_loss:
                        scaled_loss.backward()
                    opt.step()
                    scheduler.step()
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
            model.train()
            train_config = {'epochs': 1, 'alpha': 0.375, 'epsilon': 0.3, 'attack_iters': 40, \
                            'lr_max': 5e-3, 'lr_type': 'cyclic'}
            opt = torch.optim.Adam(model.parameters(), lr=train_config['lr_max'])
            lr_schedule = lambda t: np.interp([t], [0, train_config['epochs'] * 2//5, train_config['epochs']], \
                                              [0, train_config['lr_max'], 0])[0]
            criterion = nn.CrossEntropyLoss()
            epsilon = train_config['epsilon']
            alpha = train_config['alpha']
            for epoch in range(train_config['epochs']):
                print(epoch)
                for i, (X, y) in enumerate(train_loader):
                    print(i)
                    X, y = X.cuda(), y.cuda()
                    lr = lr_schedule(epoch + (i+1)/len(train_loader))
                    opt.param_groups[0].update(lr=lr)

                    delta = torch.zeros_like(X).uniform_(-epsilon, epsilon).cuda()
                    delta.requires_grad = True
                    output = model(X + delta)
                    loss = F.cross_entropy(output, y)
                    loss.backward()
                    grad = delta.grad.detach()
                    delta.data = torch.clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
                    delta.data = torch.max(torch.min(1-X, delta.data), 0-X)
                    delta = delta.detach()
                    output = model(torch.clamp(X + delta, 0, 1))
                    loss = criterion(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
        with torch.no_grad():
            model.eval()
            # predictions = model(cln_imgs)
            predictions_adv = model(adv_imgs)
        # acc = torch.sum(torch.argmax(predictions, dim = 1) == true_labels) / float(len(adv_imgs))
        # print('acc: ', float(acc.cpu()))
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
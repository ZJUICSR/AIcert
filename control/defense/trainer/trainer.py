import math
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from typing import List, Optional, Union
from art.estimators.classification import PyTorchClassifier
from art.utils import load_cifar10, load_mnist
from art.defences.trainer import *
from art.attacks.evasion import ProjectedGradientDescent
from art.data_generators import PyTorchDataGenerator
from control.defense.trainer.trades import * #
from control.defense.trainer.freeat import freeat_train, freeat_adjust_learning_rate
from control.defense.trainer.mart import * #
from control.defense.trainer.madry import * #
from control.defense.models import *

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

    def dataset(self):
        print("Step 1: Load the {} dataset".format(self.adv_dataset))
        kwargs = {'num_workers': 1, 'pin_memory': True} if True else {}
        if self.adv_dataset == 'MNIST':
            transform_train = transforms.Compose([
                transforms.RandomCrop(28, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
            ])
            trainset = torchvision.datasets.MNIST(root='/mnt/data2/yxl/AI-platform/data/MNIST', train=True, download=True, transform=transform_train)
            train_loader = DataLoader(trainset, batch_size=64, shuffle=True, **kwargs)
            testset = torchvision.datasets.MNIST(root='/mnt/data2/yxl/AI-platform/data/MNIST', train=False, download=True, transform=transform_test)
            test_loader = DataLoader(testset, batch_size=64, shuffle=False, **kwargs)
        elif self.adv_dataset == 'CIFAR10':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
            ])
            trainset = torchvision.datasets.CIFAR10(root='/mnt/data2/yxl/AI-platform/data/CIFAR10', train=True, download=True, transform=transform_train)
            train_loader = DataLoader(trainset, batch_size=64, shuffle=True, **kwargs)
            testset = torchvision.datasets.CIFAR10(root='/mnt/data2/yxl/AI-platform/data/CIFAR10', train=False, download=True, transform=transform_test)
            test_loader = DataLoader(testset, batch_size=64, shuffle=False, **kwargs)
        l = [x for (x, y) in test_loader]
        x_test = torch.cat(l, 0)
        l = [y for (x, y) in test_loader]
        y_test = torch.cat(l, 0)
        x_test = x_test[:self.adv_nums]
        y_test = y_test[:self.adv_nums]
        return x_test, y_test, train_loader

    def train(self, at_method):
        x_test, y_test, train_loader = self.dataset()

        print("Step 2: Create the model")
        if self.adv_dataset == 'CIFAR10':
            model = ResNet18().to(self.device)
            train_config = {
                'epsilon': 0.031,
                'num_steps': 10,
                'step_size': 0.007,
            }
            test_config = {
                'epsilon': 0.031,
                'num_steps': 20,
                'step_size': 0.003,
            }
        elif self.adv_dataset == 'MNIST':
            model = SmallCNN().to(self.device)
            train_config = {
                'epsilon': 0.3,
                'num_steps': 40,
                'step_size': 0.01,
            }
            test_config = train_config

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

        print("Step 5: Evaluate the ART classifier on benign test examples")
        if self.adv_method == 'PGD':
            accuracy, robustness = eval_adv_test_whitebox(model, self.device, x_test, y_test, test_config)
            robustness = robustness.cpu()
        elif self.adv_method == 'FGSM':
            _, accuracy = eval_clean(model, x_test, y_test)
            _, robustness = eval_robust(model, x_test, y_test, perturb_steps=1, epsilon=test_config['epsilon'])
        print("Accuracy on benign test examples: {}%".format(accuracy * 100))
        print("Step 7: Evaluate the ART classifier on adversarial test examples")
        print("Accuracy on adversarial test examples: {}%".format(robustness * 100))
        self.detect_rate = float(robustness)
        if self.adv_examples is None:
            attack_method = 'from user'
        else:
            attack_method = self.adv_method

        return attack_method, self.detect_num, self.detect_rate

    def print_res(self):
        print('detect rate: ', self.detect_rate)

class Madry(at):
    def __init__(self, model, mean, std, adv_method, adv_dataset, adv_nums, device):
        super().__init__(model = model, mean = mean, std = std, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)

    def detect(self):
        return self.train('Madry')

class CIFAR10_dataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform

    def __getitem__(self, index):
        x = Image.fromarray(((self.data[index] * 255).round()).astype(np.uint8).transpose(1, 2, 0))
        x = self.transform(x)
        y = self.targets[index]
        return x, y

    def __len__(self):
        return len(self.data)

def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
        module.weight.data.normal_(0, math.sqrt(2.0 / n))
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()

class FastAT(at):
    def __init__(self, model, mean, std, adv_method, adv_dataset, adv_nums, device):
        super().__init__(model = model, mean = mean, std = std, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)

    def detect(self):
        print("Step 1: Load the {} dataset".format(self.adv_dataset))
        if self.adv_dataset == 'CIFAR10':
            (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_cifar10()
            dataset_mu = np.ones((3, 32, 32))
            dataset_mu[0, :, :] = 0.4914
            dataset_mu[1, :, :] = 0.4822
            dataset_mu[2, :, :] = 0.4465

            # (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

            dataset_std = np.ones((3, 32, 32))
            dataset_std[0, :, :] = 0.2471
            dataset_std[1, :, :] = 0.2435
            dataset_std[2, :, :] = 0.2616

            x_train = x_train.transpose(0, 3, 1, 2).astype("float32")
            x_test = x_test.transpose(0, 3, 1, 2).astype("float32")

            transform = transforms.Compose(
                [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()]
            )

            dataset = CIFAR10_dataset(x_train, y_train, transform=transform)
            dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
        elif self.adv_dataset == 'MNIST':
            (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()
            dataset_mu = np.ones((1, 28, 28))
            dataset_mu[0, :, :] = 0.1307

            # (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

            dataset_std = np.ones((1, 28, 28))
            dataset_std[0, :, :] = 0.3081

            x_train = x_train.transpose(0, 3, 1, 2).astype("float32")
            x_test = x_test.transpose(0, 3, 1, 2).astype("float32")

            transform = transforms.Compose(
                [transforms.RandomCrop(28, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()]
            )

            dataset = CIFAR10_dataset(x_train, y_train, transform=transform)
            dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
        
        print("Step 2: create the PyTorch model")
        if self.adv_dataset == 'CIFAR10':
            model = ResNet18()
        elif self.adv_dataset == 'MNIST':
            model = SmallCNN()
        # For running on GPU replace the model with the
        # model = PreActResNet18().cuda()

        model.apply(initialize_weights)
        model.train()

        opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

        # if you have apex installed, the following line should be uncommented for faster processing
        # import apex.amp as amp
        # model, opt = amp.initialize(model, opt, opt_level="O2", loss_scale=1.0, master_weights=False)

        criterion = nn.CrossEntropyLoss()
        print("Step 3: Create the ART classifier")
        if self.adv_dataset == 'CIFAR10':
            input_shape = (3, 32, 32)
        elif self.adv_dataset == 'MNIST':
            input_shape = (1, 28, 28)
        classifier = PyTorchClassifier(
            model=model,
            clip_values=(0.0, 1.0),
            preprocessing=(dataset_mu, dataset_std),
            loss=criterion,
            optimizer=opt,
            input_shape=input_shape,
            nb_classes=10,
        )

        attack = ProjectedGradientDescent(
            classifier,
            norm=np.inf,
            eps=8.0 / 255.0,
            eps_step=2.0 / 255.0,
            max_iter=40,
            targeted=False,
            num_random_init=5,
            batch_size=32,
        )

        print("Step 4: Create the trainer object - AdversarialTrainerFBFPyTorch")
        # if you have apex installed, change use_amp to True
        epsilon = 8.0 / 255.0
        trainer = AdversarialTrainerFBFPyTorch(classifier, eps=epsilon, use_amp=False)

        # Build a Keras image augmentation object and wrap it in ART
        art_datagen = PyTorchDataGenerator(iterator=dataloader, size=x_train.shape[0], batch_size=128)

        print("Step 5: fit the trainer")
        trainer.fit_generator(art_datagen, nb_epochs=1) #

        x_test_pred = np.argmax(classifier.predict(x_test), axis=1)
        print(
            "Accuracy on benign test samples after adversarial training: %.2f%%"
            % (np.sum(x_test_pred == np.argmax(y_test, axis=1)) / x_test.shape[0] * 100)
        )

        x_test_attack = attack.generate(x_test)
        x_test_attack_pred = np.argmax(classifier.predict(x_test_attack), axis=1)
        self.detect_rate = np.sum(x_test_attack_pred == np.argmax(y_test, axis=1)) / x_test.shape[0]
        print(
            "Accuracy on original PGD adversarial samples after adversarial training: %.2f%%"
            % (self.detect_rate * 100)
        )        
        if self.adv_examples is None:
            attack_method = 'from user'
        else:
            attack_method = self.adv_method

        return attack_method, self.detect_num, self.detect_rate

class Trades(at):
    def __init__(self, model, mean, std, adv_method, adv_dataset, adv_nums, device):
        super().__init__(model = model, mean = mean, std = std, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)

    def detect(self):
        return self.train('TRADES')

class FreeAT(at):
    def __init__(self, model, mean, std, adv_method, adv_dataset, adv_nums, device):
        super().__init__(model = model, mean = mean, std = std, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)

    def detect(self):
        x_test, y_test, train_loader = self.dataset()

        print("Step 2: Create the model")
        if self.adv_dataset == 'CIFAR10':
            model = ResNet18().to(self.device)
        elif self.adv_dataset == 'MNIST':
            model = SmallCNN().to(self.device)

        print("Step 2a: Define the optimizer")
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        print("Step 4: Train the FreeAT classifier")
        args_epochs = 8
        n_repeats = 4
        epochs = int(math.ceil(args_epochs / n_repeats))
        criterion = nn.CrossEntropyLoss().cuda()
        if self.adv_dataset == 'CIFAR10':
            global_noise_data = torch.zeros([64, 3, 32, 32]).cuda()
        elif self.adv_dataset == 'MNIST':
            global_noise_data = torch.zeros([64, 1, 28, 28]).cuda()
        for epoch in range(0, epochs):
            freeat_adjust_learning_rate(0.1, optimizer, epoch, n_repeats)
            freeat_train(train_loader, model, criterion, optimizer, epoch, global_noise_data)

        print("Step 5: Evaluate the ART classifier on benign test examples")
        if self.adv_method == 'PGD':
            accuracy, robustness = eval_adv_test_whitebox(model, self.device, x_test, y_test)
            robustness = robustness.cpu()
        elif self.adv_method == 'FGSM':
            _, accuracy = eval_clean(model, x_test, y_test)
            _, robustness = eval_robust(model, x_test, y_test, perturb_steps=1, epsilon=0.3)
        print("Accuracy on benign test examples: {}%".format(accuracy * 100))
        print("Step 7: Evaluate the ART classifier on adversarial test examples")
        print("Accuracy on adversarial test examples: {}%".format(robustness * 100))
        self.detect_rate = float(robustness)
        if self.adv_examples is None:
            attack_method = 'from user'
        else:
            attack_method = self.adv_method

        return attack_method, self.detect_num, self.detect_rate

class Mart(at):
    def __init__(self, model, mean, std, adv_method, adv_dataset, adv_nums, device):
        super().__init__(model = model, mean = mean, std = std, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)

    def detect(self):
        return self.train('MART')
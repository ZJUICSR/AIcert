import os
import numpy as np
from typing import List, Optional
from typing import Union
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import torch
import torchvision
from torch.nn import Module
from torch.utils.data import DataLoader
from torchvision.models import vgg16
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from art.defences.preprocessor import *
from art.defences.trainer import * 
from art.estimators.classification import PyTorchClassifier
from art.estimators.encoding.tensorflow import TensorFlowEncoder
from art.estimators.generation.tensorflow import TensorFlowGenerator
from function.defense.preprocessor.inverse_gan import InverseGAN, DefenseGAN
from function.defense.models import *
from function.defense.utils.generate_aes import generate_adv_examples

class Prepro(object):
    def __init__(self, model:Module, 
                mean:List[float]=[0.485, 0.456, 0.406], 
                std:List[float]=[0.229, 0.224, 0.225],
                adv_examples=None,
                adv_method: Optional[str] = 'PGD',
                adv_dataset: Optional[str] = 'CIFAR10',
                adv_nums: Optional[int] = 10,#
                device:Union[str, torch.device]='cuda',
                ):

        super(Prepro, self).__init__()

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

    def detect_base(self, preprocess_method:Preprocessor):
        if self.adv_examples is None:
            adv_imgs, cln_imgs, true_labels = self.generate_adv_examples()
        else:
            adv_imgs, cln_imgs, true_labels = self.load_adv_examples() 
        with torch.no_grad():
            adv_predictions = self.model(adv_imgs)
        no_defense_accuracy = torch.sum(torch.argmax(adv_predictions, dim = 1) == true_labels) / float(len(adv_imgs))
        self.no_defense_accuracy = no_defense_accuracy.cpu().numpy()
        with torch.no_grad():
            # predictions = self.model(cln_imgs)
            predictions_adv = self.model(adv_imgs)
        # acc = torch.sum(torch.argmax(predictions, dim = 1) == true_labels) / float(len(adv_imgs))
        # print('acc: ', float(acc.cpu()))
        # rob = torch.sum(torch.argmax(predictions_adv, dim = 1) == true_labels) / float(len(adv_imgs))
        # print('rob: ', float(rob.cpu()))
        if preprocess_method.__name__ == 'InverseGAN' or preprocess_method.__name__ == 'DefenseGAN':
            sess = tf.Session()
            if self.adv_dataset == 'CIFAR10':
                gen_tf, enc_tf, z_ph, image_to_enc_ph = load_model(sess, "model-dcgan", "./model/gan_model/cifar10")
            elif self.adv_dataset == 'MNIST':
                gen_tf, enc_tf, z_ph, image_to_enc_ph = load_model(sess, "model-dcgan", "./model/gan_model/mnist")
            gan = TensorFlowGenerator(input_ph=z_ph, model=gen_tf, sess=sess,)
            inverse_gan = TensorFlowEncoder(input_ph=image_to_enc_ph, model=enc_tf, sess=sess,)
            if preprocess_method.__name__ == 'DefenseGAN':
                inverse_gan = None
            preproc = InverseGAN(sess=sess, gan=gan, inverse_gan=inverse_gan)
            adv_imgs = np.transpose(adv_imgs.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
            if self.model.__class__.__name__ == 'VGG' and self.adv_dataset == 'MNIST':
                adv_imgs = adv_imgs[:, 2:-2, 2:-2, :]
            _, adv_imgs_ss = preproc(adv_imgs, maxiter=10) #20
            if self.model.__class__.__name__ == 'VGG' and self.adv_dataset == 'MNIST':
                adv_imgs_ss = np.pad(adv_imgs_ss, ((0, 0), (2, 2), (2, 2), (0, 0)))
            adv_imgs_ss = np.transpose(adv_imgs_ss, (0, 3, 1, 2)).astype(np.float32)
        else:
            if preprocess_method.__name__ == 'SpatialSmoothing':
                preprocess = preprocess_method(clip_values=(0,1), window_size=6)
            else:
                preprocess = preprocess_method(clip_values=(0,1))
            adv_imgs_ss, _ = preprocess(adv_imgs.cpu().numpy())
        with torch.no_grad():
            predictions_ss = self.model(torch.from_numpy(adv_imgs_ss).to(self.device))
        if preprocess_method.__name__ == 'InverseGAN' or preprocess_method.__name__ == 'DefenseGAN':
            detect_rate = torch.sum(torch.argmax(predictions_ss, dim = 1) == true_labels) / float(len(adv_imgs))
        else:
            detect_rate = torch.sum(torch.argmax(predictions_adv, dim = 1) != torch.argmax(predictions_ss, dim = 1)) / float(len(adv_imgs))
        self.detect_rate = float(detect_rate.cpu())
        if self.adv_examples is None:
            attack_method = 'from user'
        else:
            attack_method = self.adv_method

        return attack_method, self.detect_num, self.detect_rate, self.no_defense_accuracy

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
            transform_test = transform_train
            trainset = torchvision.datasets.MNIST(root='./dataset/', train=True, download=True, transform=transform_train)
            train_loader = DataLoader(trainset, batch_size=128, shuffle=True, **kwargs)
            testset = torchvision.datasets.MNIST(root='./dataset/', train=False, download=True, transform=transform_test)
            test_loader = DataLoader(testset, batch_size=128, shuffle=False, **kwargs)
        elif self.adv_dataset == 'CIFAR10':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
            ])
            trainset = torchvision.datasets.CIFAR10(root='./dataset/CIFAR10', train=True, download=True, transform=transform_train)
            train_loader = DataLoader(trainset, batch_size=128, shuffle=True, **kwargs)
            testset = torchvision.datasets.CIFAR10(root='./dataset/CIFAR10', train=False, download=True, transform=transform_test)
            test_loader = DataLoader(testset, batch_size=128, shuffle=False, **kwargs)
        l = [x for (x, y) in test_loader]
        x_test = torch.cat(l, 0)
        l = [y for (x, y) in test_loader]
        y_test = torch.cat(l, 0)
        x_test = x_test[:self.adv_nums]
        y_test = y_test[:self.adv_nums]
        return x_test, y_test, train_loader

    def train(self, preprocess_method:Preprocessor):
        if self.adv_examples is None:
            adv_imgs, cln_imgs, true_labels = self.generate_adv_examples()
        else:
            adv_imgs, cln_imgs, true_labels = self.load_adv_examples() 
        with torch.no_grad():
            adv_predictions = self.model(adv_imgs)
        no_defense_accuracy = torch.sum(torch.argmax(adv_predictions, dim = 1) == true_labels) / float(len(adv_imgs))
        self.no_defense_accuracy = no_defense_accuracy.cpu().numpy()
        print("Step 1: Load the dataset")
        _, _, train_loader = self.dataset()

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
        print("Step 3: Define the optimizer")
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        print("Step 4: Train the {} classifier".format(preprocess_method.__name__))
        preprocess_raw = preprocess_method 
        preprocess = Preprocessor
        if preprocess_raw == LabelSmoothing:
            preprocess = preprocess_raw()
        else:
            preprocess = preprocess_raw(0,1)
        for epoch in range(1, 2):
            adjust_learning_rate(optimizer, epoch)
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                if preprocess_raw == GaussianAugmentation:
                    data = np.transpose(data, (0, 2, 3, 1))
                    data, target = preprocess(data.numpy(), target.numpy())
                    data = np.transpose(data, (0, 3, 1, 2))
                    data, target = torch.from_numpy(data), torch.from_numpy(target)
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                # calculate robust loss
                logits = model(data)
                if preprocess_raw == LabelSmoothing:
                    loss = SmoothCrossEntropyLoss(0.1)(logits, target)
                else:
                    loss = F.cross_entropy(logits, target)
                loss.backward()
                optimizer.step()
                # print progress
                if batch_idx % 100 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                            100. * batch_idx / len(train_loader), loss.item()))
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

# class Feature_squeezing(Prepro):
#     def __init__(self, model, mean, std, adv_examples, adv_method, adv_dataset, adv_nums, device):
#         super().__init__(model = model, mean = mean, std = std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
        
#     def detect(self):
#         return self.detect_base(FeatureSqueezing)

# class Jpeg_compression(Prepro):
#     def __init__(self, model, mean, std, adv_examples, adv_method, adv_dataset, adv_nums, device):
#         super().__init__(model = model, mean = mean, std = std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
        
#     def detect(self):
#         return self.detect_base(JpegCompression)

class Label_smoothing(Prepro):
    def __init__(self, model, mean, std, adv_examples, adv_method, adv_dataset, adv_nums, device):
        super().__init__(model = model, mean = mean, std = std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
        
    def detect(self):
        return self.train(LabelSmoothing)

class Spatial_smoothing(Prepro):
    def __init__(self, model, mean, std, adv_examples, adv_method, adv_dataset, adv_nums, device):
        super().__init__(model = model, mean = mean, std = std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
        
    def detect(self):
        return self.detect_base(SpatialSmoothing)

class Gaussian_augmentation(Prepro):
    def __init__(self, model, mean, std, adv_examples, adv_method, adv_dataset, adv_nums, device):
        super().__init__(model = model, mean = mean, std = std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
        
    def detect(self):
        return self.train(GaussianAugmentation)

class Total_var_min(Prepro):
    def __init__(self, model, mean, std, adv_examples, adv_method, adv_dataset, adv_nums, device):
        super().__init__(model = model, mean = mean, std = std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
        
    def detect(self):
        return self.detect_base(TotalVarMin)

class ModelImageMNIST(nn.Module):
    def __init__(self):
        super(ModelImageMNIST, self).__init__()
        self.fc = nn.Linear(28 * 28 * 1, 28 * 28 * 1 * 256)
    def forward(self, x):
        x = x.view(-1, 28 * 28 * 1)
        logit_output = self.fc(x)
        logit_output = logit_output.view(-1, 28, 28, 1, 256)
        return logit_output
    
class ModelImageMNISTVGG(nn.Module):
    def __init__(self):
        super(ModelImageMNISTVGG, self).__init__()
        self.fc = nn.Linear(32 * 32 * 1, 32 * 32 * 1 * 256)
    def forward(self, x):
        x = x.view(-1, 32 * 32 * 1)
        logit_output = self.fc(x)
        logit_output = logit_output.view(-1, 32, 32, 1, 256)
        return logit_output

class ModelImageCIFAR10(nn.Module):
    def __init__(self):
        super(ModelImageCIFAR10, self).__init__()
        self.fc = nn.Linear(32 * 32 * 1, 32 * 32 * 1 * 256)
    def forward(self, x):
        x = x.view(-1, 32 * 32 * 1)
        logit_output = self.fc(x)
        logit_output = logit_output.view(-1, 32, 32, 1, 256)
        return logit_output

class BPDAattack(object):
    def __init__(self, model, defense, device, epsilon, step_size, num_steps):
        self.model = model
        self.epsilon = epsilon
        self.defense = defense
        self.step_size = step_size
        self.num_steps = num_steps
        self.device = device

    def generate(self, X, y):
        from torch.autograd import Variable
        X_pgd = Variable(X.data, requires_grad=True)
        if True:
            random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-self.epsilon, self.epsilon).to(self.device)
            X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

        for i in range(self.num_steps):
            print(i)
            adv_purified, _ = self.defense(X_pgd.detach().cpu().numpy()) #pixdefend处理
            adv_purified = torch.from_numpy(adv_purified).to(self.device) #转成tensor
            opt = optim.SGD([X_pgd], lr=1e-3)
            opt.zero_grad()

            with torch.enable_grad():
                loss = nn.CrossEntropyLoss()(self.model(X_pgd), y)
            loss.backward()
            eta = self.step_size * X_pgd.grad.data.sign()
            X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
            eta = torch.clamp(X_pgd.data - X.data, -self.epsilon, self.epsilon)
            X_pgd = Variable(X.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
        return X_pgd

class Pixel_defend(Prepro):
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
        if self.adv_dataset == 'CIFAR10':
            input_shape = (3, 32, 32)
            model = ModelImageCIFAR10()
        elif self.adv_dataset == 'MNIST':
            if self.model.__class__.__name__ == 'VGG':
                input_shape = (1, 32, 32)
                model = ModelImageMNISTVGG()
            else:
                input_shape = (1, 28, 28)
                model = ModelImageMNIST()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        pixel_cnn = PyTorchClassifier(
            model=model, loss=loss_fn, optimizer=optimizer, input_shape=input_shape, nb_classes=10, clip_values=(0, 1)
        )
        preprocess = PixelDefend(eps=64, pixel_cnn=pixel_cnn)
        if self.adv_method == 'BPDA':
            if self.adv_dataset == 'CIFAR10':
                epsilon = 0.031
                step_size = 0.003
                num_steps = 20
            elif self.adv_dataset == 'MNIST':
                epsilon = 0.3
                step_size = 0.01
                num_steps = 20
            adversary = BPDAattack(model=self.model, defense=preprocess, device=self.device, epsilon=epsilon, step_size=step_size, num_steps=num_steps)
            bpda_examples = adversary.generate(cln_imgs, true_labels)
            with torch.no_grad():
                predictions_bpda = self.model(bpda_examples)
            bpda_robustness = torch.sum(torch.argmax(predictions_bpda, dim = 1) == true_labels) / float(len(adv_imgs))
            self.detect_rate = float(bpda_robustness.cpu())
            # print('bpda_robustness:', float(bpda_robustness.cpu()))
        else:
            adv_imgs_ss, _ = preprocess(adv_imgs.cpu().numpy()) #
            with torch.no_grad():
                # predictions = self.model(cln_imgs)
                # predictions_adv = self.model(adv_imgs)
                predictions_ss = self.model(torch.from_numpy(adv_imgs_ss).to(self.device))
            # accuracy = torch.sum(torch.argmax(predictions, dim = 1) == true_labels) / float(len(adv_imgs)) #0.9360000491142273 
            # robustness = torch.sum(torch.argmax(predictions_adv, dim = 1) == true_labels) / float(len(adv_imgs)) #0.017000000923871994
            # print('accuracy:', float(accuracy.cpu()), 'robustness:', float(robustness.cpu()))
            detect_rate = torch.sum(torch.argmax(predictions_ss, dim = 1) == true_labels) / float(len(adv_imgs)) #0.029000001028180122
            self.detect_rate = float(detect_rate.cpu())
        if self.adv_examples is None:
            attack_method = 'from user'
        else:
            attack_method = self.adv_method

        return attack_method, self.detect_num, self.detect_rate, self.no_defense_accuracy
    
class Pixel_defend_enhanced(Prepro):
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
        if self.adv_dataset == 'CIFAR10':
            input_shape = (3, 32, 32)
            model = ModelImageCIFAR10()
        elif self.adv_dataset == 'MNIST':
            if self.model.__class__.__name__ == 'VGG':
                input_shape = (1, 32, 32)
                model = ModelImageMNISTVGG()
            else:
                input_shape = (1, 28, 28)
                model = ModelImageMNIST()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        pixel_cnn = PyTorchClassifier(
            model=model, loss=loss_fn, optimizer=optimizer, input_shape=input_shape, nb_classes=10, clip_values=(0, 1)
        )
        preprocess = PixelDefend(eps=64, pixel_cnn=pixel_cnn)
        if self.adv_method == 'BPDA':
            if self.adv_dataset == 'CIFAR10':
                epsilon = 0.031
                step_size = 0.003
                num_steps = 20
                if self.model.__class__.__name__ == 'ResNet':
                    bpda_model = ResNet18()
                    checkpoint = torch.load('model/model-cifar-resnet18/model-res-epoch90.pt')
                elif self.model.__class__.__name__ == 'VGG':
                    bpda_model = vgg16()
                    bpda_model.classifier[6] = nn.Linear(4096, 10)
                    checkpoint = torch.load('model/model-cifar-vgg16/model-vgg16-epoch89.pt')
                bpda_model.load_state_dict(checkpoint)
                bpda_model = bpda_model.to(self.device)
            elif self.adv_dataset == 'MNIST':
                epsilon = 0.3
                step_size = 0.01
                num_steps = 20
                if self.model.__class__.__name__ == 'SmallCNN':
                    bpda_model = SmallCNN()
                    checkpoint = torch.load('model/model-mnist-smallCNN/model-nn-epoch99.pt')
                elif self.model.__class__.__name__ == 'VGG':
                    bpda_model = vgg16()
                    bpda_model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
                    bpda_model.classifier[6] = nn.Linear(4096, 10)
                    checkpoint = torch.load('model/model-mnist-vgg16/model-vgg16-epoch57.pt')
                bpda_model.load_state_dict(checkpoint)
                bpda_model = bpda_model.to(self.device).eval()    
            adversary = BPDAattack(model=self.model, defense=preprocess, device=self.device, epsilon=epsilon, step_size=step_size, num_steps=num_steps)
            bpda_examples = adversary.generate(cln_imgs, true_labels)
            with torch.no_grad():
                predictions_bpda = bpda_model(bpda_examples)
            bpda_robustness = torch.sum(torch.argmax(predictions_bpda, dim = 1) == true_labels) / float(len(adv_imgs))
            self.detect_rate = float(bpda_robustness.cpu())
            # print('bpda_robustness:', float(bpda_robustness.cpu()))
        else:
            adv_imgs_ss, _ = preprocess(adv_imgs.cpu().numpy()) #
            with torch.no_grad():
                # predictions = self.model(cln_imgs)
                # predictions_adv = self.model(adv_imgs)
                predictions_ss = self.model(torch.from_numpy(adv_imgs_ss).to(self.device))
            # accuracy = torch.sum(torch.argmax(predictions, dim = 1) == true_labels) / float(len(adv_imgs)) #0.9360000491142273 
            # robustness = torch.sum(torch.argmax(predictions_adv, dim = 1) == true_labels) / float(len(adv_imgs)) #0.017000000923871994
            # print('accuracy:', float(accuracy.cpu()), 'robustness:', float(robustness.cpu()))
            detect_rate = torch.sum(torch.argmax(predictions_ss, dim = 1) == true_labels) / float(len(adv_imgs)) #0.029000001028180122
            self.detect_rate = float(detect_rate.cpu())
        if self.adv_examples is None:
            attack_method = 'from user'
        else:
            attack_method = self.adv_method

        return attack_method, self.detect_num, self.detect_rate, self.no_defense_accuracy

def load_model(sess, model_name, model_path):
    saver = tf.train.import_meta_graph(os.path.join(model_path, model_name + ".meta"))
    saver.restore(sess, os.path.join(model_path, model_name))

    graph = tf.get_default_graph()

    generator_tf = graph.get_tensor_by_name("generator/output:0")
    image_to_encode_ph = graph.get_tensor_by_name("image_to_encode_input:0")
    encoder_tf = graph.get_tensor_by_name("encoder_1/dense_1/Relu:0")
    z_ph = graph.get_tensor_by_name("z_input:0")

    return generator_tf, encoder_tf, z_ph, image_to_encode_ph

def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = 0.1
    if epoch >= 75:
        lr = 0.1 * 0.1
    if epoch >= 90:
        lr = 0.1 * 0.01
    if epoch >= 100:
        lr = 0.1 * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

import torch.nn as nn
import torch.nn.functional as F

class SmoothCrossEntropyLoss(nn.Module):
    def __init__(self, alpha=0.1):
        super(SmoothCrossEntropyLoss, self).__init__()
        self.alpha = alpha

    def forward(self, output, target):
        num_classes = output.size()[-1]
        log_softmax_output = F.log_softmax(output, dim=-1)
        one_hot_target = F.one_hot(target, num_classes=num_classes).float()
        smooth_labels = (1 - self.alpha) * one_hot_target + self.alpha / num_classes
        loss = -1 * (smooth_labels * log_softmax_output).sum(dim=-1).mean()
        return loss


class Inverse_gan(Prepro):
    def __init__(self, model, mean, std, adv_examples, adv_method, adv_dataset, adv_nums, device):
        super().__init__(model = model, mean = mean, std = std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    def detect(self):
        return self.detect_base(InverseGAN)
    
class Defense_gan(Inverse_gan):
    def __init__(self, model, mean, std, adv_examples, adv_method, adv_dataset, adv_nums, device):
        super().__init__(model = model, mean = mean, std = std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    def detect(self):
        return self.detect_base(DefenseGAN)

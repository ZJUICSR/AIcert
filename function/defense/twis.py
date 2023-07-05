import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Module
from typing import List, Optional
from typing import Union
from function.defense.utils.generate_aes import generate_adv_examples

class Twis(object):
    def __init__(self, model:Module, 
                mean:List[float]=[0.485, 0.456, 0.406], 
                std:List[float]=[0.229, 0.224, 0.225],
                noise_radius:float = 0.01,
                targeted_lr:float = 0.005,
                targeted_radius:float = 0.5,
                untargeted_step_threshold:float = 10000,
                untargeted_radius:float = 0.5,
                untargeted_lr:float = 1,
                fpr: float = 0.1,
                adv_examples=None,
                adv_method: Optional[str] = 'PGD',
                adv_dataset: Optional[str] = 'MNIST',
                adv_nums: Optional[int] = 10000,
                device:Union[str, torch.device]='cuda',
                ):

        super(Twis, self).__init__()

        self.model = model
        self.mean = mean
        self.std = std
        if adv_examples is None and (adv_method is None or adv_nums is None):
            raise Exception('Attack method and adversarial example nums need to be specified when the adversarial sample is not specified!')
        self.noise_radius = noise_radius
        self.targeted_lr = targeted_lr
        self.targeted_radius =targeted_radius
        self.untargeted_radius = untargeted_radius
        self.untargeted_lr = untargeted_lr
        self.fpr = fpr
        self.adv_examples = adv_examples
        self.adv_method = adv_method
        self.adv_dataset = adv_dataset
        self.adv_nums = adv_nums
        self.device = device
        self.tpr = 0
        self.criterions = {0.1: (0.009, 119, untargeted_step_threshold), 0.2: (0.0006, 89, untargeted_step_threshold)}
        self.no_defense_accuracy = 0

    def generate_adv_examples(self):
        return generate_adv_examples(self.model, self.adv_method, self.adv_dataset, self.adv_nums, self.device)


    def load_adv_examples(self):
        data = torch.load(self.adv_examples)
        print('successfully load adversarial examples!')
        return data['adv_img'], data['cln_img'], data['y']

        """Normalize the data given the dataset. Only ImageNet and CIFAR-10 are supported"""
    def transform(self, img):
        # Data
        if self.adv_dataset == 'MNIST':
            mean = torch.Tensor([self.mean[0]]).unsqueeze(1).expand_as(img[0, :, :, 0]).unsqueeze(2).expand_as(
                img[0]).unsqueeze(0).expand_as(img).cuda()
            std = torch.Tensor([self.std[0]]).unsqueeze(1).expand_as(img[0, :, :, 0]).unsqueeze(2).expand_as(
                img[0]).unsqueeze(0).expand_as(img).cuda()
        elif self.adv_dataset == 'CIFAR10':
            mean = torch.Tensor(self.mean).unsqueeze(1).expand_as(img[0, :, :, 0]).unsqueeze(2).expand_as(
                img[0]).unsqueeze(0).expand_as(img).cuda()
            std = torch.Tensor(self.std).unsqueeze(1).expand_as(img[0, :, :, 0]).unsqueeze(2).expand_as(
                img[0]).unsqueeze(0).expand_as(img).cuda()
        else:
            raise "dataset is not supported"
        return (img - mean) / std


    """Given [label] and [dataset], return a random label different from [label]"""
    def random_label(self, label):
        if self.adv_dataset == 'MNIST':
            class_num = 10
        elif self.adv_dataset == 'CIFAR10':
            class_num = 10
        else:
            raise "dataset is not supported"
        attack_label = np.random.randint(class_num)
        while label == attack_label:
            attack_label = np.random.randint(class_num)
        return attack_label

    """Given the variance of zero_mean Gaussian [n_radius], return a noisy version of [img]"""

    class Noisy(torch.autograd.Function):
        @staticmethod
        def forward(self, img, noise_radius):
            return img + noise_radius * torch.randn_like(img)

        @staticmethod
        def backward(self, grad_output):
            return grad_output, None

    

    """ Return the value of l1 norm of [img] with noise radius [n_radius]"""
    def l1_detection(self, img):
        noisy = self.Noisy.apply
        return torch.norm(F.softmax(self.model(self.transform(img))) - F.softmax(
            self.model(self.transform(noisy(img, self.noise_radius)))), 1).item()

    """ Return the number of steps to cross boundary using targeted attack on [img]. Iteration stops at 
        [cap] steps """
    def targeted_detection(self, 
                        img, 
                        cap=200,
                        margin=20,
                        use_margin=False):
        self.model.eval()
        x_var = torch.autograd.Variable(img.clone().cuda(), requires_grad=True)
        true_label = self.model(self.transform(x_var.clone())).data.max(1, keepdim=True)[1][0].item()
        optimizer_s = optim.SGD([x_var], lr=self.targeted_lr)
        target_l = torch.LongTensor([self.random_label(true_label)]).cuda()
        counter = 0
        while self.model(self.transform(x_var.clone())).data.max(1, keepdim=True)[1][0].item() == true_label:
            optimizer_s.zero_grad()
            output = self.model(self.transform(x_var))
            if use_margin:
                target_l = target_l[0].item()
                _, top2_1 = output.data.cpu().topk(2)
                argmax11 = top2_1[0][0]
                if argmax11 == target_l:
                    argmax11 = top2_1[0][1]
                loss = (output[0][argmax11] - output[0][target_l] + margin).clamp(min=0)
            else:
                loss = F.cross_entropy(output, target_l)
            loss.backward()

            x_var.data = torch.clamp(x_var - self.targeted_lr * x_var.grad.data, min=0, max=1)
            x_var.data = torch.clamp(x_var - img, min=-self.targeted_radius, max=self.targeted_radius) + img
            counter += 1
            if counter >= cap:
                break
        return counter

    """ Return the number of steps to cross boundary using untargeted attack on [img]. Iteration stops at 
        [cap] steps """
    def untargeted_detection(self, 
                            img, 
                            cap=1000,
                            margin=20,
                            use_margin=False):
        self.model.eval()
        x_var = torch.autograd.Variable(img.clone().cuda(), requires_grad=True)
        true_label = self.model(self.transform(x_var.clone())).data.max(1, keepdim=True)[1][0].item()
        optimizer_s = optim.SGD([x_var], lr=self.untargeted_lr)
        counter = 0
        while self.model(self.transform(x_var.clone())).data.max(1, keepdim=True)[1][0].item() == true_label:
            optimizer_s.zero_grad()
            output = self.model(self.transform(x_var))
            if use_margin:
                _, top2_1 = output.data.cpu().topk(2)
                argmax11 = top2_1[0][0]
                if argmax11 == true_label:
                    argmax11 = top2_1[0][1]
                loss = (output[0][true_label] - output[0][argmax11] + margin).clamp(min=0)
            else:
                loss = -F.cross_entropy(output, torch.LongTensor([true_label]).cuda())
            loss.backward()

            x_var.data = torch.clamp(x_var - self.untargeted_lr * x_var.grad.data, min=0, max=1)
            x_var.data = torch.clamp(x_var - img, min=-self.untargeted_radius, max=self.untargeted_radius) + img
            counter += 1
            if counter >= cap:
                break
        return counter

    """ Return a set of values of l1 norm. 
    """
    def l1_vals(self, adv_total, real_label_total):
        vals = np.zeros(0)
        cout = self.adv_nums
        for i in range(self.adv_nums):
            adv = adv_total[i].unsqueeze(0)
            real_label = real_label_total[i]
            self.model.eval()
            predicted_label = self.model(self.transform(adv.clone())).data.max(1, keepdim=True)[1][0]
            if real_label == predicted_label:
                cout -= 1 #number of successful adversary minus 1
                continue #only load successful adversary
            val = self.l1_detection(adv)
            vals = np.concatenate((vals, [val]))
        print('this is number of success in l1 detection', cout)
        return vals

    """ Return a set of number of steps using targeted detection.
    """
    def targeted_vals(self, adv_total, real_label_total):
        vals = np.zeros(0)
        cout = self.adv_nums
        for i in range(self.adv_nums):
            adv = adv_total[i].unsqueeze(0)
            real_label = real_label_total[i]
            self.model.eval()
            predicted_label = self.model(self.transform(adv.clone())).data.max(1, keepdim=True)[1][0]
            if real_label == predicted_label:
                cout -= 1 #number of successful adversary minus 1
                continue #only load successful adversary
            val = self.targeted_detection(adv)
            vals = np.concatenate((vals, [val]))
        print('this is number of success in targeted detection', cout)
        return vals


    """ Return a set of number of steps using untargeted detection. 
    """
    def untargeted_vals(self, adv_total, real_label_total):
        vals = np.zeros(0)
        cout = self.adv_nums
        for i in range(self.adv_nums):
            adv = adv_total[i].unsqueeze(0)
            real_label = real_label_total[i]
            self.model.eval()
            predicted_label = self.model(self.transform(adv.clone())).data.max(1, keepdim=True)[1][0]
            if real_label == predicted_label:
                cout -= 1 #number of successful adversary minus 1
                continue #only load successful adversary
            val = self.untargeted_detection(adv)
            vals = np.concatenate((vals, [val]))
        print('this is number of success in untargeted detection', cout)
        return vals
        
    def detect(self):
        if self.adv_examples is None:
            adv_imgs, clean_examples, true_labels = self.generate_adv_examples()
        else:
            adv_imgs, clean_examples, true_labels = self.load_adv_examples()
        with torch.no_grad():
            adv_predictions = self.model(adv_imgs)
        no_defense_accuracy = torch.sum(torch.argmax(adv_predictions, dim = 1) == true_labels) / float(len(adv_imgs))
        self.no_defense_accuracy = no_defense_accuracy.cpu().numpy()
        a_target_1 = self.l1_vals(adv_imgs, true_labels)
        a_target_2 = self.targeted_vals(adv_imgs, true_labels)
        a_target_3 = self.untargeted_vals(adv_imgs, true_labels)
        self.tpr = len(a_target_1[np.logical_or(np.logical_or(a_target_1 > self.criterions[self.fpr][0], a_target_2 > \
            self.criterions[self.fpr][1]),a_target_3 > self.criterions[self.fpr][2])]) * 1.0 / len(a_target_1)
        print("corresponding tpr for " + self.adv_method + " of this threshold is", self.tpr)
        return self.noise_radius, self.fpr, self.tpr, self.no_defense_accuracy
                
    def print_res(self):
        print('detect rate: ', self.tpr)
        if self.adv_examples is None:
            attack_method = 'from user'
        else:
            attack_method = self.adv_method

        return attack_method, self.tpr

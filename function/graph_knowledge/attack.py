import torch
from tqdm import tqdm
import os
from torchattacks import *


# SUPPORT_METHODS = {'fgsm', 'bim', 'rfgsm', 'c&w', 'pgd', 'tpgd', 'mi-fgsm', 'autopgd', 'fab', 'square', 'deepfool', 'difgsm'}
SUPPORT_METHODS = {'fgsm', 'bim', 'rfgsm', 'c&w', 'pgd', 'tpgd', 'mi-fgsm', 'autopgd', 'fab', 'square', 'difgsm'}


class ArtAttack(object):
    def __init__(self, model, dataloarder, n_class=10, eps=0.01,
                 device='cpu', min_pixel_value=0, max_pixel_value=255,
                 save_path=None, log_func=None):
        super(ArtAttack, self).__init__()
        self.model = model
        self.dataloader = dataloarder
        self.n_class = n_class
        self.eps = eps
        self.device = device
        self.min_pixel_value = min_pixel_value
        self.max_pixel_value = max_pixel_value
        self.save_path = save_path
        self.model.to(self.device)
        self.log_func = log_func
        self.attacks = {
            'fgsm': FGSM(self.model, eps=self.eps),
            'bim': BIM(self.model, eps=self.eps, alpha=2/255, steps=100),
            'rfgsm': RFGSM(self.model, eps=self.eps, alpha=2/255, steps=100),
            'c&w': CW(self.model, c=1, lr=0.01, steps=100, kappa=0),
            'pgd': PGD(self.model, eps=self.eps, alpha=2/225, steps=100, random_start=True),
            'tpgd': TPGD(self.model, eps=self.eps, alpha=2/255, steps=100),
            'mi-fgsm': MIFGSM(self.model, eps=self.eps, alpha=2/255, steps=100, decay=0.1),
            'autopgd': APGD(self.model, eps=self.eps, steps=100, eot_iter=1, n_restarts=1, loss='ce'),
            'fab': FAB(self.model, eps=self.eps, steps=100, n_classes=self.n_class, n_restarts=1, targeted=False),
            'square': Square(self.model, eps=self.eps, n_queries=5000, n_restarts=1, loss='ce'),
            'deepfool': DeepFool(self.model, steps=20),
            'difgsm': DIFGSM(self.model, eps=self.eps, alpha=2/255, steps=100, diversity_prob=0.5, resize_rate=0.9)
            }
        # if torch.cuda.is_available():
        #     os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        #     os.environ['CUDA_VISIBLE_DEVICES'] = str(device)[-1]

    def write_logs(self, info):
        if self.log_func is None:
            return
        self.log_func(info)

    def calc_adv_acc(self, method, attack):
        correct = 0
        for data, label in tqdm(self.dataloader, ncols=100, desc=f'{method} generate adversarial examples'):
            data, label = data.to(self.device), label.to(self.device)
            adv = attack(data, label)
            outputs = self.model(adv)
            _, pre = torch.max(outputs.data, 1)
            correct += float((pre == label).sum()) / len(label)

        return round(correct / len(self.dataloader), 4)

    def calculate_attack_acc(self, methods):
        acc = dict()
        for method in methods:
            method_lower = method.lower()
            if method_lower not in self.attacks:
                acc[method] = 0
                continue
            attack = self.attacks[method_lower]
            acc[method] = self.calc_adv_acc(method, attack)
            self.write_logs(f'[模型测试阶段] 执行{method}算法，准确率为：{acc[method]}')
            # print(f'{method}={acc[method]}')
        return acc

    def calc_ori_acc(self):
        self.model.to(self.device)
        correct = 0
        for data, label in self.dataloader:
            data = data.to(self.device)
            label = label.to(self.device)
            outputs = self.model(data)
            _, pre = torch.max(outputs.data, 1)
            correct += float((pre == label).sum()) / len(label)
        return round(correct / len(self.dataloader), 4)

    def run(self, methods):
        acc = dict()
        self.write_logs('[模型测试阶段] 计算模型初始准确率')
        acc['ori'] = self.calc_ori_acc()
        self.write_logs('[模型测试阶段] 开始进行对抗攻击')
        adv_acc = self.calculate_attack_acc(methods)
        acc.update(adv_acc)

        self.write_logs('[模型测试阶段] 测试结束')
        return acc


if __name__ == '__main__':
    pass

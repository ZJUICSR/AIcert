import torch
from function.adversarial_test.attack.gen_adv import attacks_dict
import os
from function.adversarial_test.unit import write_json
from tqdm import tqdm
import torchvision,torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os


RESULT_FOLDER = os.path.join(os.path.dirname(__file__), 'results')
DATA_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
for path in [RESULT_FOLDER, DATA_FOLDER]:
    if not os.path.exists(path):
        os.mkdir(path)


class FlowLineRules(object):
    def __init__(self, model, dataset='mnist', eps=4, attack_methods=['FGSM'], device='cuda', batch_size=128, log_func=None,ori_loader=None):
        self.model = model
        self.dataset = dataset.lower()
        self.eps = eps
        self.attack_methods = attack_methods
        self.device = device
        self.batch_size = batch_size
        self.log_func = log_func
        self.attacks = attacks_dict(self.model, self.eps)
        self.ori_loader = ori_loader

    def write_logs(self, info):
        print(info)
        if self.log_func is None:
            return
        self.log_func(info)

    def mnist_dataloader(self, batch_size=128):
        data_dir = os.path.join(DATA_FOLDER, 'mnist')
        train_dataset = torchvision.datasets.MNIST(root=data_dir, download=True, train=True, transform=transforms.ToTensor())
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_dataset = torchvision.datasets.MNIST(root=data_dir, download=True, train=False, transform=transforms.ToTensor())
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        return dataloader, test_dataloader

    def cifar10_dataloader(self, batch_size):
        data_dir = os.path.join(DATA_FOLDER, 'cifar10')
        train_data = torchvision.datasets.CIFAR10(root=data_dir, train=True, transform=torchvision.transforms.ToTensor(),
                                                  download=True)
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
        test_data = torchvision.datasets.CIFAR10(root=data_dir, train=False, transform=torchvision.transforms.ToTensor(),
                                                  download=True)
        test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
        return train_loader, test_loader

    def calc_adv_acc(self, method, attack, dataloader):
        correct = 0
        self.model.eval()
        for data, label in tqdm(dataloader, ncols=100, desc=f'{method} generate adversarial examples'):
            data, label = data.to(self.device), label.to(self.device)
            adv = attack(data, label)
            outputs = self.model(adv)
            _, pre = torch.max(outputs.data, 1)
            correct += float((pre == label).sum()) / len(label)

        return round(correct / len(dataloader), 4) * 100

    def calculate_attack_acc(self, dataloader):
        acc = dict()
        for method in self.attack_methods:
            self.write_logs(f"启动{method}方法攻击测试")
            method_lower = method.lower()
            if method_lower not in self.attacks:
                acc[method] = 0
                continue
            attack = self.attacks[method_lower]
            acc[method] = self.calc_adv_acc(method, attack, dataloader)
            self.write_logs(f'执行{method}算法，准确率为：{acc[method]}')
        return acc

    def calc_ori_acc(self, dataloader):
        self.model.to(self.device)
        correct = 0
        self.model.eval()
        for data, label in dataloader:
            data = data.to(self.device)
            label = label.to(self.device)
            outputs = self.model(data)
            _, pre = torch.max(outputs.data, 1)
            correct += float((pre == label).sum()) / len(label)
        return round(correct / len(dataloader), 4) * 100

    def load_datasets(self):
        data_func = {'mnist': self.mnist_dataloader,
                     'cifar10': self.cifar10_dataloader}
        assert self.dataset in data_func
        return data_func[self.dataset]()

    def run(self):
        results = dict()
        self.write_logs("加载数据集")
        if self.ori_loader:
            dataloader = self.ori_loader
        else:
            dataloader, _ = self.load_datasets()
        self.write_logs("计算原始准确率")
        results['ori_acc'] = self.calc_ori_acc(dataloader)
        self.write_logs("启动攻击")
        results['attack_results'] = self.calculate_attack_acc(dataloader)
        write_json(results, os.path.join(RESULT_FOLDER, 'results.json'))
        return results


if __name__ == '__main__':
    pass


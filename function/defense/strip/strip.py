import numpy as np
import copy
import matplotlib.pyplot as plt
import torch
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from typing import Union, Optional, Tuple
from torch.utils.data import TensorDataset
from torchvision import transforms
import torchvision
from torch.utils.data import Subset


class CustomSubset(Subset):
    '''A custom subset class'''

    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.targets = dataset.targets  # 保留targets属性
        self.classes = dataset.classes  # 保留classes属性
        self.data = dataset.data

    def __getitem__(self, idx):  # 同时支持索引访问操作
        x, y = self.dataset[self.indices[idx]]
        return x, y

    def __len__(self):  # 同时支持取长度操作
        return len(self.indices)


# 投毒数据
# 目前只支持CIFAR10，待优化
class PoisonedDataset(Dataset):
    def __init__(self, dataset, trigger_label, portion=0.1, mode="train",
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), dataname="mnist"):
        self.device = device
        self.dataname = dataname
        self.ori_dataset = dataset
        self.data, self.targets = self.add_trigger(self.reshape(dataset.data, dataname), dataset.targets, trigger_label,
                                                   portion, mode)
        self.channels, self.width, self.height = self.__shape_info__()

    def __getitem__(self, item):
        img = self.data[item]
        label_idx = self.targets[item]
        label = np.zeros(10)
        label[label_idx] = 1  # 把num型的label变成10维列表。
        label = torch.Tensor(label)
        img = img.to(self.device)
        label = label.to(self.device)
        return img, label

    def __len__(self):
        return len(self.data)

    def __shape_info__(self):
        return self.data.shape[1:]

    def reshape(self, data, dataname="CIFAR10"):
        if dataname == "MNIST":
            new_data = data.reshape(len(data), 1, 28, 28)
        elif dataname == "CIFAR10":
            new_data = data.reshape(len(data), 3, 32, 32)
        return np.array(new_data)

    def norm(self, data):
        offset = np.mean(data, 0)
        scale = np.std(data, 0).clip(min=1)
        return (data - offset) / scale

    def add_trigger(self, data, targets, trigger_label, portion, mode):
        print("## generate " + mode + " Bad Imgs")
        new_data = copy.deepcopy(data)
        new_targets = np.array(copy.deepcopy(targets))
        perm = np.random.permutation(len(new_data))[0: int(len(new_data) * portion)]
        self.poison = np.zeros_like(targets)
        self.poison[perm] = 1
        _, width, height = new_data.shape[1:]
        new_targets[perm] = trigger_label
        new_data[perm, :, width - 3, height - 3] = 255
        new_data[perm, :, width - 3, height - 2] = 255
        new_data[perm, :, width - 2, height - 3] = 255
        new_data[perm, :, width - 2, height - 2] = 255
        print("Injecting Over: %d Bad Imgs, %d Clean Imgs (%.2f)" % (len(perm), len(new_data) - len(perm), portion))
        return torch.Tensor(new_data), new_targets


class Strip(object):
    def __init__(self, model: Module,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 adv_examples=None,
                 adv_method: Optional[str] = 'PGD',
                 adv_dataset: Optional[str] = 'CIFAR10',
                 adv_nums: Optional[int] = 10,
                 device: Union[str, torch.device] = 'cuda'):

        super(Strip, self).__init__()

        self.model = copy.deepcopy(model).to(device)
        self.norm = transforms.Normalize(mean, std)
        if adv_examples is None and (adv_method is None or adv_nums is None):
            raise Exception(
                'Attack method and adversarial example nums need to be specified when the adversarial sample is not specified!')
        self.adv_examples = adv_examples
        self.adv_method = adv_method
        self.dataset = adv_dataset
        self.adv_nums = adv_nums
        self.device = device
        self.total_num = 0
        self.detect_num = 0
        self.detect_rate = 0
        self.no_defense_accuracy = 0
        # STRIP参数
        self.entropy_threshold = 0.1
        self.plot_hist = True
        self.benign_num = 100

    """
    重训练模型植入后门
    只支持读取cifar10+resnet18的预训练后门模型，其他模型需在线训练，待优化
    """

    def retrain(self, train_loader):
        if self.dataset == "CIFAR10":
            try:
                self.model.load_state_dict(torch.load('model/model-cifar-resnet18/model-res-backdoor.pt'))
                return
            except:
                print("no pretrained backdoor model")

        learning_rate = 0.01
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=2e-5)
        CE = torch.nn.CrossEntropyLoss()
        epochs = 40
        self.model.train()

        for epoch in range(epochs):
            print("======RETRAINING...EPOCH: {}======".format(epoch))
            for batch_id, (imgs, labels) in enumerate(train_loader):
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                output = self.model(imgs)
                loss = CE(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def test(self, test_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_id, (imgs, labels) in enumerate(test_loader):
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                output = self.model(imgs)
                pred = torch.argmax(output, dim=1)
                targets = torch.argmax(labels, dim=1)
                correct += torch.sum(pred == targets)
                total += len(labels)
        return correct / total

    def strip_single_predict(self, test_input, benign_loader):
        ext_test_imgs = test_input.repeat_interleave(100, dim=0)
        for _, (bgn_imgs, _) in enumerate(benign_loader):
            added_image = 0.5 * ext_test_imgs.to(self.device) + 0.5 * bgn_imgs.to(self.device)
            added_image = torch.clamp(added_image, 0, 255)
            output = self.model(added_image)
        return output

    def explore(self, test_loader, benign_loader) -> Tuple[bool, float, plt.hist]:
        decisions = []
        for batch_id, (test_input, _) in enumerate(test_loader):
            output = self.strip_single_predict(test_input, benign_loader)
            entropy_avg = self.entropy_cal(output)
            if entropy_avg <= self.entropy_threshold:
                decision = 1  # trojan
            else:
                decision = 0  # benign
            decisions.append(decision)

        # entropy = -np.nansum(output * np.log2(output), axis=1)
        # prob, ent = np.histogram(entropy, bins=30, weights=np.ones(len(entropy)) / len(entropy))
        # prob = [i.item() for i in prob]
        # ent = [str(i.item()) for i in ent]
        # hist = dict(zip(ent, prob))
        hist = 0

        return decisions, entropy_avg, hist

    def entropy_cal(self, output) -> float:
        output = output.cpu().detach().numpy()
        entropy_sum = -np.nansum(output * np.log2(output))
        entropy_avg = entropy_sum / self.benign_num
        return entropy_avg

    def detect(self):
        # 加载数据
        if self.dataset == "MNIST":
            train_set = torchvision.datasets.MNIST(root="dataset", train=True, download=False)
            test_set = torchvision.datasets.MNIST(root="dataset", train=False, download=False)

        if self.dataset == "CIFAR10":
            train_set = torchvision.datasets.CIFAR10(root="dataset/CIFAR10", train=True, download=False)
            test_set = torchvision.datasets.CIFAR10(root="dataset/CIFAR10", train=False, download=False)

        # 投毒的训练数据
        train_poi = PoisonedDataset(train_set, 0, portion=0.25, mode="train", dataname=self.dataset)
        train_poi_loader = DataLoader(dataset=train_poi, batch_size=512, shuffle=True)
        # 50%植入后门的1000个测试数据  （修改portion可控制投毒数据的占比）
        small_indices = np.random.choice(range(len(test_set)), size=1000, replace=False)
        small_sampler = torch.utils.data.sampler.SubsetRandomSampler(small_indices)
        test_half_poi = PoisonedDataset(test_set, 0, portion=0.5, mode="test", dataname=self.dataset)
        test_half_poi_loader = DataLoader(dataset=test_half_poi, batch_size=1, shuffle=False, sampler=small_sampler)
        # 全植入后门的测试数据
        test_all_poi = PoisonedDataset(test_set, 0, portion=1, mode="test", dataname=self.dataset)
        test_all_poi_loader = DataLoader(dataset=test_all_poi, batch_size=512, shuffle=False)
        # 用于检测后门的50个干净样本
        tiny_indices = np.random.choice(range(len(train_set)), size=100, replace=False)
        tiny_sampler = torch.utils.data.sampler.SubsetRandomSampler(tiny_indices)
        benign_set = PoisonedDataset(train_set, 0, portion=0, mode="train", dataname=self.dataset)
        benign_loader = DataLoader(dataset=benign_set, batch_size=100, shuffle=False, sampler=tiny_sampler)

        # 植入后门
        self.retrain(train_poi_loader)

        # 测试后门攻击下的模型准确性
        self.no_defense_accuracy = self.test(test_all_poi_loader)

        # 检测成功率
        decisions, entropy_avg, hist = self.explore(test_half_poi_loader, benign_loader)
        self.detect_rate = np.sum(decisions == test_half_poi.poison[small_indices]) / len(small_indices)

        if self.adv_examples is None:
            attack_method = 'from user'
        else:
            attack_method = self.adv_method

        return attack_method, self.detect_num, self.detect_rate, self.no_defense_accuracy

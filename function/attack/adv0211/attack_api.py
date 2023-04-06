import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
print(sys.path)
# 逃逸攻击接口
from art.attacks.evasion import FastGradientMethod, UniversalPerturbation, AutoAttack, BoundaryAttack, ProjectedGradientDescent
from art.attacks.evasion import BasicIterativeMethod, CarliniWagner, DeepFool, SaliencyMapMethod
from art.attacks.evasion import SquareAttack, HopSkipJump, PixelAttack, SimBA, GDUniversarial, ZooAttack, GeoDA, Fastdrop
# 后门攻击接口
from art.attacks.poisoning import PoisoningAttackBackdoor, PoisoningAttackCleanLabelBackdoor, FeatureCollisionAttack, PoisoningAttackAdversarialEmbedding
from art.estimators.classification import PyTorchClassifier
from art.utils import load_mnist, load_cifar10
# from art.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from art.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from art.mnist import Mnist
from art.utils import compute_success, compute_accuracy
import random

class EvasionAttacker():
    def __init__(self, modelnet=Mnist, modelpath: str="./models/mnist-6-mnist_0.9890.pkl", 
    dataset="mnist", datasetpath="./datasets/", nb_classes=10, datanormalize: bool = True, 
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) -> None:
        self.modelnet = eval(modelnet)
        self.modelpath = modelpath
        self.dataset = dataset
        self.datasetpath = datasetpath
        self.device = device
        self.nb_classes = nb_classes
        self.datanormalize = datanormalize
        if dataset == "cifar10":
            self.loaddataset = load_cifar10
        elif dataset == "mnist":
            self.loaddataset = load_mnist
        else:
            raise ValueError("Dataset not supported:",dataset)

    def perturb(self, method: str="FastGradientMethod", sample_num: int=5, **kwargs) -> None:
        # 数据集加载
        input_shape, (x_select, y_select), _, _, min_pixel_value, max_pixel_value = self.loaddataset(self.datasetpath, sample_num, normalize=self.datanormalize)
        # 模型加载
        self.model = self.modelnet()
        checkpoint = torch.load(self.modelpath)
        # self.model.load_state_dict(checkpoint,strict=False)
        try:
            self.model.load_state_dict(checkpoint)
        except:
            self.model.load_state_dict(checkpoint['net'])
        self.cln_examples = x_select.astype(np.float32)
        print(len(self.cln_examples))
        self.real_lables = y_select
        # 在模型结构的基础上进一步构造分类器
        self.min_pixel_value = min_pixel_value
        self.max_pixel_value = max_pixel_value
        self.input_shape = input_shape
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.006)
        self.classifier = PyTorchClassifier (
            model=self.model,
            clip_values=(min_pixel_value, max_pixel_value),
            loss=criterion,
            optimizer=optimizer,
            input_shape=input_shape,
            nb_classes=self.nb_classes,
        )
        setattr(self.classifier, "_device", self.device)
        self.method = method
        self.attack = eval(self.method)(self.classifier)

        # 设置攻击参数
        for key, value in kwargs.items():
            if key not in self.attack.attack_params:
                raise ValueError("{} is not the parameter of {}".format(key, method))
            else:
                setattr(self.attack, key, value)
        for key in self.attack.attack_params:
            try:
                print(key, str(getattr(self.attack, key)))
            except:
                pass
        # 第一次计算干净样本上的分类准确率
        print("first accurate:{}".format(compute_accuracy(self.classifier.predict(self.cln_examples), self.real_lables)))
        self.adv_examples  = self.attack.generate(self.cln_examples)
        # 性能评估
        clean_predictions = self.classifier.predict(self.cln_examples)
        self.psrb= compute_accuracy(clean_predictions, self.real_lables)
        adv_predictions = self.classifier.predict(self.adv_examples)
        self.psra= compute_accuracy(adv_predictions, self.real_lables)
        # 计算真正的攻击成功率，将本来分类错误的样本排除在外
        self.coverrate, self.asr  = compute_success(self.real_lables, clean_predictions, adv_predictions)
    
    def attack_with_eps(self, epslist: list=[]) -> list:
        def attack_with_compute_asr(self):
            self.asr  = compute_success(self.classifier, self.cln_examples, self.cln_lables, self.adv_examples)
        asr_list = []
        for eps in epslist:
            self.attack.generate(eps=eps)
            attack_with_compute_asr()
            asr_list.append(self.asr)
        return asr_list

    def print_res(self) -> None:
        print("Before {} attack, the accuracy on benign test examples: {}%".format(self.method, self.psrb* 100))
        print("After {} attack, the accuracy on adversarial test examples: {}%".format(self.method, self.psra*100))
        print("The final {} attack success rate: {}%, coverrate: {}%".format(self.method, (self.asr)*100, (self.coverrate)*100))


class BackdoorAttacker():
    def __init__(
    self, modelnet=Mnist, modelpath: str="./models/ckpt-mnist-15-mnist_0.9905.pth", 
    dataset="mnist", datasetpath="./datasets/", nb_classes=10, datanormalize: bool = True, 
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) -> None:
        self.modelnet = eval(modelnet)
        self.modelpath = modelpath
        self.dataset = dataset
        self.datasetpath = datasetpath
        self.device = device
        self.nb_classes = nb_classes
        self.datanormalize = datanormalize
        if dataset == "cifar10":
            self.loaddataset = load_cifar10
        elif dataset == "mnist":
            self.loaddataset = load_mnist
        else:
            raise ValueError("Dataset not supported")
    
    # 添加固定样式的后门
    def add_backdoor(x: np.ndarray):
        xt = x.copy()
        xt[:, :, 30:31, 30:31] = 1
        return xt
    
    # 传入整个将用于训练的数据集，按照比例和目标进行自动的投毒
    # 返回投毒后的数据集和一个用来标识投毒样本的列表
    def poison(self, xs: np.ndarray, ys: np.ndarray, target: np.ndarray, pp_poison: float, backdoor_item: np.ndarray = None, add_backdoor_func = add_backdoor, batch_size=256):
        # backdoor_item是为了应对FeatureCollisionAttack中对于攻击实体的选择
        # 默认将最后一层作为特征层
        name = [item[0] for item in self.model._modules.items()]
        if self.method == "PoisoningAttackBackdoor":
            self.attacker = PoisoningAttackBackdoor(add_backdoor_func)
        elif self.method == "PoisoningAttackCleanLabelBackdoor":
            # 这种情况下的投毒比例是对于一个类的样本而言的
            self.attackerbase = PoisoningAttackBackdoor(add_backdoor_func)
            self.attacker = PoisoningAttackCleanLabelBackdoor(target=target, pp_poison=pp_poison, proxy_classifier=self.classifier, batch_size=batch_size, backdoor=self.attackerbase)
        elif self.method == "FeatureCollisionAttack":
            # 本方法中的target是指作为后门实体的一个样本
            self.backdoor_item = backdoor_item
            self.attacker = FeatureCollisionAttack(classifier=self.classifier, target=backdoor_item, feature_layer=name[len(name)-1])
        else:
            # PoisoningAttackAdversarialEmbedding
            self.attackerbase = PoisoningAttackBackdoor(add_backdoor_func)
            self.attacker = PoisoningAttackAdversarialEmbedding(backdoor=self.attackerbase, device=self.device)
        # 对于没有自动选择样本进行投毒操作的，这里通过比例和攻击要求为其选择样本进行投毒
        # 防止修改原样本
        x, y = xs.copy(), ys.copy()
        total_num = len(x)
        if self.method == "PoisoningAttackBackdoor":
            select_list = random.sample(range(0, total_num), int(total_num*pp_poison))
            print(int(total_num*pp_poison))
            x_needp = x.take(select_list, axis=0)
            # y_needp = y.take(select_list, axis=0)
            x_p, y_p = self.attacker.poison(x_needp, target)
            for i, j in enumerate(select_list):
                x[j,:,:,:] = x_p[i,:,:,:]
                y[j,:] = y_p[i,:]
            return x, y, select_list
        elif self.method == "PoisoningAttackAdversarialEmbedding":
            select_list = random.sample(range(0, total_num), int(total_num*pp_poison))
            print(int(total_num*pp_poison))
            x_needp = x.take(select_list, axis=0)
            # y_needp = y.take(select_list, axis=0)
            x_p, y_p = self.attacker.poison(x_needp, target)
            for i, j in enumerate(select_list):
                x[j,:,:,:] = x_p[i,:,:,:]
                y[j,:] = y_p[i,:]
            return x, y, select_list
        elif self.method == "FeatureCollisionAttack":
            data = np.copy(x)
            estimated_labels = np.copy(y)
            all_indices = np.arange(len(data))
            target_indices = all_indices[np.all(estimated_labels == target, axis=1)]
            print(len(target_indices)*pp_poison)
            num_poison = int(pp_poison * len(target_indices))
            selected_indices = np.random.choice(target_indices, num_poison)
            xp = x.take(selected_indices, axis=0)
            xp = self.attacker.poison(xp)
            for i, j in enumerate(selected_indices):
                x[j] = xp[i]
                y[i] = target[0]
            return x, y, selected_indices
        else:
            return self.attacker.poison(x, y)

    def finetune(self, x_train: np.ndarray, y_train: np.ndarray, select_list: list=[], num_epochs: int=40, batch_size: int=700, lr=0.001, alpha=50):
        """
        模型在投毒样本上微调
        """
        if self.method == "PoisoningAttackAdversarialEmbedding":
            self.attacker.fintune(self.model, x_train, y_train, select_list, num_epochs=num_epochs, batch_size=batch_size, lr=0.001, alpha=alpha)
        else:
            num_batch = int(np.ceil(len(x_train) / float(batch_size)))
            total_step = len(x_train)
            for epoch in range(num_epochs):
                train_loss = 0
                for m in range(num_batch):
                    # Batch indexes
                    begin, end = (
                        m * batch_size,
                        min((m + 1) * batch_size, x_train.shape[0]),
                    )
                    # 前向传播
                    model_outputs = self.model(torch.from_numpy(x_train[begin:end]).to(self.device))
                    # 损失计算
                    loss = self.criterion(model_outputs, torch.from_numpy(y_train[begin:end]).to(self.device))
                    self.optimizer.zero_grad()
                    loss.backward()
                    train_loss += loss.item()
                    self.optimizer.step()
                print("Epoch [{}/{}], train_loss: {}".format(epoch+1, num_epochs, train_loss / total_step))
                print(self.evaluate(self.x, self.y, self.plist))
    
    # 两个评价函数
    def compute_accuracy(self, x: np.ndarray, y: np.ndarray, batch_size:int=100):
        return compute_accuracy(self.classifier.predict(x, batch_size=batch_size), y)
    
    def evaluate(self, xs: np.ndarray, ys: np.ndarray, select_list: list, batch_size:int=100):
        """
        评价模型对后门攻击的应对情况
        包括：干净样本上的准确率，后门样本上的攻击成功率
        """
        clean_list = [i for i in range(len(xs)) if i not in select_list]
        x_clean = xs.take(clean_list, axis=0)
        y_clean = ys.take(clean_list, axis=0)
        x_poisoned = xs.take(select_list, axis=0)
        y_poisoned = ys.take(select_list, axis=0)

        # 该方法的评价方式较为特殊
        if self.method == "FeatureCollisionAttack":
            # 总样本上的准确率
            self.accuracy = compute_accuracy(self.classifier.predict(xs, batch_size=batch_size), ys)
            # 干净样本上的准确率
            self.accuracyonb = compute_accuracy(self.classifier.predict(x_clean, batch_size=batch_size), y_clean)
            # 攻击成功率
            self.attack_success_rate = (compute_accuracy(self.classifier.predict(self.backdoor_item, batch_size=batch_size), self.target[np.newaxis, :]))
            if self.attack_success_rate == 1:
                print("攻击成功!")
        else:
            # 总样本上的准确率
            self.accuracy = compute_accuracy(self.classifier.predict(xs, batch_size=batch_size), ys)
            # 干净样本上的准确率
            self.accuracyonb = compute_accuracy(self.classifier.predict(x_clean, batch_size=batch_size), y_clean)
            # 后门样本上的攻击成功率
            self.attack_success_rate = compute_accuracy(self.classifier.predict(x_poisoned, batch_size=batch_size), y_poisoned)
        return self.accuracy, self.accuracyonb, self.attack_success_rate

    def backdoorattack(self, method: str="PoisoningAttackBackdoor", pp_poison: float=0.5, target: int=3, batch_size:int=700, num_epochs=40, lr=0.01, alpha=50, test_sample_num:int=1024):
        """
        为了简化后门攻击接口的使用提出的一个后门攻击样例方法
        该函数按照目标和投毒比例自动完成攻击
        """
        self.support_method = ["PoisoningAttackBackdoor", "PoisoningAttackCleanLabelBackdoor", "FeatureCollisionAttack", "PoisoningAttackAdversarialEmbedding"]
        if method not in self.support_method:
            raise ValueError("This method is not supported")
        self.method = method
        self.model = self.modelnet()
        checkpoint = torch.load(self.modelpath)
        try:
            self.model.load_state_dict(checkpoint)
        except:
            self.model.load_state_dict(checkpoint['net'])
        
        self.datasetpath = self.datasetpath
        self.device = self.device
        self.model.to(self.device)
        self.batch_size = batch_size

        # 训练参数
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # 数据集加载
        input_shape, (x_select, y_select), (x_train, y_train), _, min_pixel_value, max_pixel_value = self.loaddataset(self.datasetpath, test_sample_num)
        # 数据集规范
        x_train_examples = x_train.astype(np.float32)
        x_select_examples = x_select.astype(np.float32)
        # 构造分类器方便使用
        self.classifier = PyTorchClassifier (
            model=self.model,
            clip_values=(min_pixel_value, max_pixel_value),
            loss=self.criterion,
            optimizer=self.optimizer,
            input_shape=input_shape,
            nb_classes=10,
        )
        # 决定目标
        self.target = np.zeros((10))
        self.target[target] = 1

        if self.method=="FeatureCollisionAttack":
            # "FeatureCollisionAttack"
            print(self.compute_accuracy(x_select_examples[0][np.newaxis, :], self.target[np.newaxis, :],batch_size=batch_size))
            x, y, plist= self.poison(x_train_examples, y_train, self.target, pp_poison=pp_poison, backdoor_item=x_select_examples[random.randint(1,len(x_select_examples))][np.newaxis, :])
            self.x = x
            self.y = y
            self.plist = plist
            print(self.evaluate(x, y, plist, batch_size=batch_size))
            self.finetune(x, y, batch_size=batch_size)
            print(self.evaluate(x, y, plist, batch_size=batch_size))
        else:
            # 其他三种攻击
            # 首先完成投毒
            x, y, plist= self.poison(x_train_examples, y_train, self.target, pp_poison=pp_poison, batch_size=self.batch_size)
            self.x = x
            self.y = y
            self.plist = plist
            print(self.evaluate(x, y, plist, batch_size=batch_size))
            self.finetune(x, y, self.plist, batch_size=batch_size, num_epochs=num_epochs, lr=lr)
            print(self.evaluate(x, y, plist, batch_size=batch_size))
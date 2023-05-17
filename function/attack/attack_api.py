import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# 逃逸攻击接口
from .attacks.evasion import FastGradientMethod, ProjectedGradientDescent, BasicIterativeMethod, DeepFool, SaliencyMapMethod, UniversalPerturbation, AutoAttack, GDUAP, CarliniWagner
from .attacks.evasion import PixelAttack, SimBA, ZooAttack, SquareAttack
from .attacks.evasion import BoundaryAttack, HopSkipJump, GeoDA, Fastdrop
# 后门攻击接口
from .attacks.poisoning import PoisoningAttackBackdoor, PoisoningAttackCleanLabelBackdoor, FeatureCollisionAttack, PoisoningAttackAdversarialEmbedding
# 工具函数
from function.attack.estimators.classification import PyTorchClassifier
from .attacks.utils import load_mnist, load_cifar10
# from models.model_net.resnet import ResNet18
from .attacks.utils import compute_predict_accuracy, compute_attack_success, compute_accuracy, compute_success
import random
from typing import Optional
from .attacks.config import MY_NUMPY_DTYPE
import time
from PIL import Image
import shutil
import os
from tqdm import tqdm

class EvasionAttacker():
    def __init__(self, modelnet=None, modelpath: str="ckpt-resnet18-mnist_epoch3_acc0.9898.pth", 
    dataset="mnist", datasetpath="./datasets/", nb_classes=10, datanormalize: bool = False, 
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), sample_num=0) -> None:
        self.modelnet = modelnet
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
        if sample_num > 0:
            self.input_shape, (self.x_select, self.y_select), _, _, self.min_pixel_value, self.max_pixel_value = self.loaddataset(self.datasetpath, sample_num, normalize=self.datanormalize)

    def generate(self, method: str="FastGradientMethod", save_num: int = 0, **kwargs) -> None:
        # 模型加载
        self.model = self.modelnet.to(self.device)
        checkpoint = torch.load(self.modelpath)
        
        try:
            self.model.load_state_dict(checkpoint)
        except:
            self.model.load_state_dict(checkpoint['net'])
        
        cln_examples = self.x_select.astype(MY_NUMPY_DTYPE)
        # print(len(self.cln_examples))
        real_lables = self.y_select
        # 在模型结构的基础上进一步构造分类器
        self.min_pixel_value = self.min_pixel_value
        self.max_pixel_value = self.max_pixel_value
        self.input_shape = self.input_shape
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.006)
        self.classifier = PyTorchClassifier (
            model=self.model,
            clip_values=(self.min_pixel_value, self.max_pixel_value),
            loss=criterion,
            optimizer=optimizer,
            input_shape=self.input_shape,
            nb_classes=self.nb_classes,
            device=self.device
        )
        # setattr(self.classifier, "_device", self.device)
        self.method = method
        self.attack = eval(self.method)(self.classifier)

        # 设置攻击参数
        for key, value in kwargs.items():
            if key not in self.attack.attack_params:
                if key != "save_path":
                    error = "{} is not the parameter of {}".format(key, method)
                    raise ValueError(error)
            else:
                setattr(self.attack, key, value)
        for key in self.attack.attack_params:
            try:
                print(key, str(getattr(self.attack, key)))
            except:
                pass
        
        # 第一次计算干净样本上的分类准确率
        print("first accurate:{}".format(compute_predict_accuracy(self.classifier.predict(cln_examples), real_lables)))
        try:
            starttime = time.clock()
        except:
            starttime = time.perf_counter()
        adv_examples  = self.attack.generate(cln_examples)
        try:
            endtime = time.clock()
        except:
            endtime = time.perf_counter()
        
        self.timespend = (endtime - starttime)/len(adv_examples)
        # 性能评估
        clean_predictions = self.classifier.predict(cln_examples)
        self.psrb= compute_predict_accuracy(clean_predictions, real_lables)
        adv_predictions = self.classifier.predict(adv_examples)
        self.psra= compute_predict_accuracy(adv_predictions, real_lables)
        # 计算真正的攻击成功率，将本来分类错误的样本排除在外
        self.coverrate, self.asr  = compute_attack_success(real_lables, clean_predictions, adv_predictions)
        self.save_examples(save_num, self.nb_classes, real_lables, cln_examples, clean_predictions, adv_examples, adv_predictions, kwargs["save_path"])
        return adv_examples
    
    def save_examples(self, save_num, class_num, label: np.ndarray, clean_example: np.ndarray, clean_prediction: np.ndarray, adv_example: np.ndarray, adv_prediction: np.ndarray, path:str="./results/"):
        # 尽量均匀到类
        if save_num <= 0:
            return
        
        try:
            os.makedirs(os.path.join(path,self.method))
        except Exception as e:
            shutil.rmtree(os.path.join(path,self.method))
            os.makedirs(os.path.join(path,self.method))
        
        l = np.argmax(label,1)
        s = np.argmax(clean_prediction,1)
        d = np.argmax(adv_prediction, 1)
        used_list = []
        for i in range(save_num):
            for k in range(len(s)):
                if s[k] != d[k] and l[k] == i%class_num and s[k] == l[k]:
                    if k in used_list:
                        break
                    used_list.append(k)
                    if clean_example[k].shape[0] == 1:
                        tmpc = clean_example[k][0]
                        tmpa = adv_example[k][0]
                    else:
                        tmpc = clean_example[k].transpose(1,2,0)
                        tmpa = adv_example[k].transpose(1,2,0)
                    clean = Image.fromarray(np.uint8(255*tmpc))
                    clean.save(path+"/"+self.method+"/"+"index{}_clean_l{}_t{}.jpeg".format(i, l[k], d[k]))
                    adv = Image.fromarray(np.uint8(255*tmpa))
                    adv.save(path+"/"+self.method+"/"+"index{}_adv_l{}_t{}.jpeg".format(i, l[k], d[k]))
                    per = np.uint8(255*tmpc) - np.uint8(255*tmpa)
                    per = per + np.abs(np.min(per))
                    per = Image.fromarray(np.uint8(255*per))
                    per.save(path+"/"+self.method+"/"+"index{}_per_l{}_t{}.jpeg".format(i, l[k], d[k]))
                    break

    def print_res(self) -> None:
        print("Before {} attack, the accuracy on benign test examples: {}%".format(self.method, self.psrb* 100))
        print("After {} attack, the accuracy on adversarial test examples: {}%".format(self.method, self.psra*100))
        print("The final {} attack success rate: {}%, coverrate: {}%".format(self.method, (self.asr)*100, (self.coverrate)*100))
        print("Time spent per sample: {}s".format(self.timespend))
        return {"before_acc":self.psrb* 100,"after_acc":self.psra*100, "asr":(self.asr)*100, "coverrate":(self.coverrate)*100, "time":self.timespend}

class BackdoorAttacker():
    def __init__(
    self, modelnet=None, modelpath: str="./models/model_ckpt/ckpt-resnet18-mnist_epoch3_acc0.9898.pth", 
    dataset="mnist", datasetpath="./datasets/", nb_classes=10, datanormalize: bool = False, 
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) -> None:
        self.modelnet = modelnet
        self.modelpath = modelpath
        self.dataset = dataset
        self.datasetpath = datasetpath
        self.device = device
        self.nb_classes = nb_classes
        self.datanormalize = datanormalize
        if dataset == "cifar10":
            self.loaddataset = load_cifar10
            self.backdoor_color = 1
        elif dataset == "mnist":
            self.loaddataset = load_mnist
            self.backdoor_color = 2
        else:
            raise ValueError("Dataset not supported")
    
    # 添加固定样式的后门
    def add_backdoor(x: np.ndarray, px=25, py=25, l=2, value=2):
        if px+l >= x.shape[2] or py+l >= x.shape[3]:
            raise ValueError("Invalid px or py!")
        xt = x.copy()
        xt[:, :, px:px+l, py:py+l] = value
        return xt
    
    # 传入整个将用于训练的数据集，按照比例和目标进行自动的投毒
    # 返回投毒后的数据集和一个用来标识投毒样本的列表
    def poison(self, xs: np.ndarray, ys: np.ndarray, target: np.ndarray, pp_poison: float, add_backdoor_func = add_backdoor, batch_size=256):
        
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
            all_indices = np.arange(len(xs))
            source_indices = all_indices[np.all(ys == self.source, axis=1)]
            selected_indices = np.random.choice(len(source_indices), 1)
            backdoor_item = xs[selected_indices[0]:selected_indices[0]+1,:]
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
            print(int(len(target_indices)*pp_poison))
            num_poison = int(pp_poison * len(target_indices))
            selected_indices = np.random.choice(len(target_indices), num_poison)
            xp = x.take(selected_indices, axis=0)
            xp = self.attacker.poison(xp)
            for i, j in enumerate(selected_indices):
                x[j] = xp[i]
                y[i] = target[0]
            return x, y, selected_indices
        else:
            return self.attacker.poison(x, y)

    def finetune(self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, select_list: list=[], test_list: list=[], num_epochs: int=40, batch_size: int=700, lr=0.001, alpha=500):
        """
        模型在投毒样本上重新训练和微调
        """
        print("(总体样本准确率，干净样本准确率，投毒样本准确率)")
        if self.method == "PoisoningAttackAdversarialEmbedding":
            self.attacker.fintune(self.model, x_train, y_train, select_list, num_epochs=num_epochs, batch_size=batch_size, lr=0.001, alpha=alpha)
            all, clean, backdoor = self.evaluate(x_test, y_test, test_list)
            print((all, clean, backdoor))
            self.save_model(desc="all{}_clean{}_backdoor{}".format(all, clean, backdoor))
        else:
            num_batch = int(np.ceil(len(x_train) / float(batch_size)))
            for i in tqdm(range(num_epochs), desc="Training"):
                self.classifier.fit(x_train, y_train, nb_epochs=1, batch_size=num_batch, show=False)
                all, clean, backdoor = self.evaluate(x_test, y_test, test_list)
                print((all, clean, backdoor))
                self.save_model(desc="all{}_clean{}_backdoor{}".format(all, clean, backdoor), epoch=i)
    
    # 两个评价函数
    def compute_accuracy(self, x: np.ndarray, y: np.ndarray, batch_size:int=100):
        return compute_accuracy(self.classifier.predict(x, batch_size=batch_size), y)
    
    def evaluate(self, xs: np.ndarray, ys: np.ndarray, select_list: list, batch_size:int=128):
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
            self.accuracy, _ = compute_accuracy(self.classifier.predict(xs, batch_size=batch_size), ys)
            # 干净样本上的准确率
            self.accuracyonbm = self.accuracy
            # 攻击成功率
            # 对于冲突碰撞攻击，当一个样本的真实标签为原标签且预测标签为目标标签的时候认为攻击成功
            self.attack_success_rate = np.sum(np.logical_and(np.argmax(ys, axis=1) == np.argmax(self.source, axis=0), np.argmax(self.classifier.predict(xs, batch_size=batch_size), axis=1) == np.argmax(self.target, axis=0)))/ np.sum(np.argmax(ys, axis=1) == np.argmax(self.source, axis=0))
        else:
            # 总样本上的准确率
            self.accuracy, _ = compute_accuracy(self.classifier.predict(xs, batch_size=batch_size), ys)
            # 干净样本上的准确率
            self.accuracyonb, _ = compute_accuracy(self.classifier.predict(x_clean, batch_size=batch_size), y_clean)
            # 后门样本上的攻击成功率
            self.attack_success_rate, _ = compute_accuracy(self.classifier.predict(x_poisoned, batch_size=batch_size), y_poisoned)
        return self.accuracy, self.accuracyonb, self.attack_success_rate

    def backdoorattack(self, method: str="PoisoningAttackBackdoor", pp_poison: float=0.5, source: int = 0, target: int=3, batch_size:int=700, num_epochs=40, lr=0.01, alpha=50, test_sample_num:int=2048, save_num: int=32, save_path=None):
        """
        为了简化后门攻击接口的使用提出的一个后门攻击样例方法
        该函数按照目标和投毒比例自动完成攻击
        """
        self.support_method = ["PoisoningAttackBackdoor", "PoisoningAttackCleanLabelBackdoor", "FeatureCollisionAttack", "PoisoningAttackAdversarialEmbedding"]
        if method not in self.support_method:
            raise ValueError("This method is not supported")
        self.method = method

        self.model = self.modelnet
        checkpoint = torch.load(self.modelpath)
        try:
            self.model.load_state_dict(checkpoint)
        except:
            self.model.load_state_dict(checkpoint['net'])
        
        self.model.to(self.device)
        self.datasetpath = self.datasetpath
        self.batch_size = batch_size

        # 训练参数
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # 数据集加载
        input_shape, (x_select, y_select), (x_train, y_train), _, min_pixel_value, max_pixel_value = self.loaddataset(self.datasetpath, test_sample_num)
        # 数据集规范
        x_train_tmp = x_train.astype(MY_NUMPY_DTYPE)
        y_train_tmp = y_train.astype(MY_NUMPY_DTYPE)
        x_select_tmp = x_select.astype(MY_NUMPY_DTYPE)
        y_select_tmp = y_select.astype(MY_NUMPY_DTYPE)
        # 构造分类器方便使用
        self.classifier = PyTorchClassifier (
            model=self.model,
            clip_values=(min_pixel_value, max_pixel_value),
            loss=self.criterion,
            optimizer=self.optimizer,
            input_shape=input_shape,
            nb_classes=10,
            device=self.device,
        )

        # 对于特征碰撞攻击，需要设置原目标类
        self.source = np.zeros((10))
        self.source[source] = 1
        # 决定目标
        self.target = np.zeros((10))
        self.target[target] = 1

        # 测试集用来测试攻击效果，测试集中一般的样本被投毒
        # 主要是为了区分开测试集和训练集
        # 对于特征碰撞攻击，测试数据集是不需要投毒的
        if self.method == "FeatureCollisionAttack":
            x_test = x_select_tmp
            y_test = y_select_tmp
            test_plist = [0]
        else:
            x_test, y_test, test_plist = self.poison(x_select_tmp, y_select_tmp, self.target, pp_poison=0.5)
        
        # 根据比例进行训练集投毒
        x_train, y_train, plist= self.poison(x_train_tmp, y_train_tmp, self.target, pp_poison=pp_poison)
        self.poisoned_num = len(plist)
        print("投毒样本数目:{}".format(self.poisoned_num))

        # 保存部分投毒样本
        if save_path != None:
            self.save_examples(save_num=32, class_num=self.classifier.nb_classes, label=y_train_tmp[plist], clean_example=x_train_tmp[plist], poisoned_example=x_train[plist], path=save_path)
        
        # 测试投毒攻击效果的时候都在测试样本上进行
        # 对于特征冲突攻击，不需要考虑测试数据集的投毒
        print(self.evaluate(x_test, y_test, test_plist, batch_size=batch_size))
        self.finetune(x_train, y_train, x_test, y_test, plist, test_plist, batch_size=batch_size, num_epochs=num_epochs)
        print(self.evaluate(x_test, y_test, test_plist, batch_size=batch_size))
        return self.evaluate(x_test, y_test, test_plist, batch_size=batch_size)

    def save_model(self, path:str="./results/", desc:str="backdoor_model", epoch:int=0):
        path = path+self.method+"/backdoor_model/"
        if epoch == 0:
            try:
                os.makedirs(path)
            except Exception as e:
                shutil.rmtree(path)
                os.makedirs(path)
        path = path + desc + ".pth"
        torch.save(self.classifier.model.state_dict(), path)
    
    def save_examples(self, save_num, class_num, label: np.ndarray, clean_example: np.ndarray, poisoned_example: np.ndarray, path:str="./results/"):
        # 尽量均匀到类
        if save_num <= 0:
            return
        path = os.path.join(path,self.method,"example")
        try:
            os.makedirs(path)
        except Exception as e:
            shutil.rmtree(path)
            os.makedirs(path)
        
        l = np.argmax(label,1)
        used_list = []

        if self.method == "PoisoningAttackCleanLabelBackdoor":
            for i in range(save_num):
                if clean_example[i].shape[0] == 1:
                    tmpc = clean_example[i][0]
                    tmpa = poisoned_example[i][0]
                else:
                    tmpc = clean_example[i].transpose(1,2,0)
                    tmpa = poisoned_example[i].transpose(1,2,0)
                clean = Image.fromarray(np.uint8(255*tmpc))
                clean.save(path+"index{}_clean.jpeg".format(i, l[i]))
                adv = Image.fromarray(np.uint8(255*tmpa))
                adv.save(path+"index{}_poisoned.jpeg".format(i, l[i]))
                backdoor = np.uint8(255*tmpc) - np.uint8(255*tmpa)
                backdoor = backdoor + np.abs(np.min(backdoor))
                backdoor = Image.fromarray(np.uint8(255*backdoor))
                backdoor.save(path+"index{}_backdoor.jpeg".format(i, l[i]))
        else:
            for i in range(save_num):
                for k in range(len(clean_example)):
                    if l[k] == i%class_num:
                        if k in used_list:
                            break
                        used_list.append(k)
                        if clean_example[k].shape[0] == 1:
                            tmpc = clean_example[k][0]
                            tmpa = poisoned_example[k][0]
                        else:
                            tmpc = clean_example[k].transpose(1,2,0)
                            tmpa = poisoned_example[k].transpose(1,2,0)
                        clean = Image.fromarray(np.uint8(255*tmpc))
                        clean.save(path+"index{}_clean.jpeg".format(i, l[k]))
                        adv = Image.fromarray(np.uint8(255*tmpa))
                        adv.save(path+"index{}_poisoned.jpeg".format(i, l[k]))
                        backdoor = np.uint8(255*tmpc) - np.uint8(255*tmpa)
                        backdoor = backdoor + np.abs(np.min(backdoor))
                        backdoor = Image.fromarray(np.uint8(255*backdoor))
                        backdoor.save(path+"index{}_backdoor.jpeg".format(i, l[k]))
                        break
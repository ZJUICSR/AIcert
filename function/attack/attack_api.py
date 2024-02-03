import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# 逃逸攻击接口
from .attacks.evasion import *
from .attacks.evasion import FastGradientMethod, ProjectedGradientDescent, BasicIterativeMethod, DeepFool, SaliencyMapMethod, UniversalPerturbation, AutoAttack, GDUAP, CarliniWagner
from .attacks.evasion import PixelAttack, SimBA, ZooAttack, SquareAttack
from .attacks.evasion import BoundaryAttack, HopSkipJump, GeoDA, Fastdrop
from .attacks.torchattack import *
# 后门攻击接口
from .attacks.poisoning import PoisoningAttackBackdoor, PoisoningAttackCleanLabelBackdoor, FeatureCollisionAttack, PoisoningAttackAdversarialEmbedding
# 工具函数
from function.attack.estimators.classification import PyTorchClassifier
from .attacks.utils import load_mnist, load_cifar10, to_categorical
# from models.model_net.resnet import ResNet18
from .attacks.utils import compute_predict_accuracy, compute_attack_success, compute_accuracy, compute_success
import random
from typing import List
from .attacks.config import MY_NUMPY_DTYPE
import time
from PIL import Image
import shutil
import os
from tqdm import tqdm

# norm [1,2,"inf",np.inf]

class EvasionAttacker():
    # def __init__(self, model, 
    # dataset="mnist", datasetpath="./datasets/", nb_classes=10, datanormalize: bool = False, 
    # device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), sample_num=128) -> None:
    def __init__(self, modelnet=None, modelpath: str="ckpt-resnet18-mnist_epoch3_acc0.9898.pth", 
    dataset="mnist", datasetpath="./datasets/", nb_classes=10, datanormalize: bool = False, 
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), sample_num=128, model=None,) -> None:
        self.modelnet = modelnet
        
        self.model = model
        self.device = device
        # 模型加载
        if self.model == None:
            self.modelpath = modelpath
            self.model = self.modelnet.to(self.device)
            checkpoint = torch.load(self.modelpath)
            
            try:
                self.model.load_state_dict(checkpoint)
            except:
                self.model.load_state_dict(checkpoint['net'])
        self.dataset = dataset
        self.datasetpath = datasetpath
        
        self.nb_classes = nb_classes
        self.datanormalize = datanormalize
        self.norm_param = ""
        if dataset == "cifar10":
            self.loaddataset = load_cifar10
            self.input_shape = (3, 32, 32)
        elif dataset == "mnist":
            self.loaddataset = load_mnist
            self.input_shape = (1, 28, 28)
        else:
            raise ValueError("Dataset not supported")
        if sample_num > 0:
            self.input_shape, (self.x_select, self.y_select), _, _, self.min_pixel_value, self.max_pixel_value, _ = self.loaddataset(self.datasetpath, sample_num, normalize=self.datanormalize)
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
            
    def generate(self, method: str="FastGradientMethod", save_num: int = 1, **kwargs) -> None:
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
        print(self.method)
        for key, value in kwargs.items():
            if key not in self.attack.attack_params:
                if key != "save_path":
                    error = "{} is not the parameter of {}".format(key, method)
                    raise ValueError(error)
            else:
                if key == "norm":
                    self.norm_param = str(value)
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
        # adv_examples  = self.attack.generate(cln_examples, real_lables)
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
        if 'save_path' in kwargs:
            piclist = self.save_examples(save_num, real_lables, cln_examples, clean_predictions, adv_examples, adv_predictions, kwargs["save_path"])
        else:
            piclist=None
        return adv_examples, real_lables, piclist
    
    def get_adv_data(self, dataloader, method: str="FastGradientMethod",  **kwargs) -> None:
        x_select,y_select = None, None
        for step,(x, y) in enumerate(dataloader):
            if x_select is None:
                x_select = x.detach().to('cpu')
                y_select = y.detach().to('cpu')
                continue
            x_select = torch.cat((x_select, x.detach().to('cpu')), dim=0)
            y_select = torch.cat((y_select, y.detach().to('cpu')), dim=0)
        x_select = x_select.numpy()
        
        y_test = y_select.numpy()
        y_select = to_categorical(y_test, 10)
        cln_examples = x_select.astype(MY_NUMPY_DTYPE)
        # print(len(self.cln_examples))
        real_lables = y_select
        min_value = np.amin(cln_examples) if np.amin(cln_examples) < self.min_pixel_value else self.min_pixel_value
        max_value = np.amax(cln_examples) if np.amax(cln_examples) > self.max_pixel_value else self.max_pixel_value
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.006)
        print("self.input_shape:",self.input_shape)
        classifier = PyTorchClassifier (
            model=self.model,
            clip_values=(min_value, max_value),
            loss=criterion,
            optimizer=optimizer,
            input_shape=self.input_shape,
            nb_classes=self.nb_classes,
            device=self.device
        )
        self.method = method
        self.attack = eval(self.method)(classifier)

        # 设置攻击参数
        print(self.method)
        for key, value in kwargs.items():
            if key not in self.attack.attack_params:
                if key != "save_path":
                    error = "{} is not the parameter of {}".format(key, method)
                    raise ValueError(error)
            else:
                if key == "norm":
                    self.norm_param = str(value)
                setattr(self.attack, key, value)
        for key in self.attack.attack_params:
            try:
                print(key, str(getattr(self.attack, key)))
            except:
                pass
        # 第一次计算干净样本上的分类准确率
        print("first accurate:{}".format(compute_predict_accuracy(classifier.predict(cln_examples), real_lables)))
        # adv_examples  = self.attack.generate(cln_examples, real_lables)
        adv_examples  = self.attack.generate(cln_examples)
        # 性能评估

        adv_predictions = classifier.predict(adv_examples)
        self.psra= compute_predict_accuracy(adv_predictions, real_lables)
        print("self.psra:",self.psra)
        return adv_examples, cln_examples, y_test, y_test, self.psra
    
    def get_attack(self, method: str="FastGradientMethod", **kwargs) -> None:
        
        self.method = method
        self.attack = eval(self.method)(self.classifier)

        # 设置攻击参数
        print(self.method)
        for key, value in kwargs.items():
            if key not in self.attack.attack_params:
                if key not in ["save_path",'eps']:
                    error = "{} is not the parameter of {}".format(key, method)
                    raise ValueError(error)
            else:
                if key == "norm":
                    self.norm_param = str(value)
                setattr(self.attack, key, value)
        for key in self.attack.attack_params:
            try:
                print(key, str(getattr(self.attack, key)))
            except:
                pass
        return self.attack.generate
    
    def save_examples(self, save_num, label: np.ndarray, clean_example: np.ndarray, clean_prediction: np.ndarray, adv_example: np.ndarray, adv_prediction: np.ndarray, path:str="output/cache/results/"):
        
        path = os.path.join(path, self.method+self.norm_param)
        try:
            os.makedirs(path)
        except Exception as e:
            shutil.rmtree(path)
            os.makedirs(path)
        piclist = []
        if save_num <= 0:
            return piclist
        else:
            l = np.argmax(label,1)
            s = np.argmax(clean_prediction,1)
            d = np.argmax(adv_prediction, 1)
            real_adv_examples = adv_example.take(np.where(s != d), axis=0)[0]

            random_index = random.sample(range(0, len(real_adv_examples)), len(real_adv_examples))
            l = np.argmax(label,1)
            for i in range(len(real_adv_examples)):
                if clean_example[random_index[i]].shape[0] == 1:
                    tmpc = clean_example[random_index[i]][0]
                    tmpa = adv_example[random_index[i]][0]
                else:
                    tmpc = clean_example[random_index[i]].transpose(1,2,0)
                    tmpa = adv_example[random_index[i]].transpose(1,2,0)
                clean = Image.fromarray(np.uint8(255*tmpc))
                cleanname = os.path.join(path, "index{}_clean_l{}_t{}.jpeg".format(i, l[random_index[i]], d[random_index[i]]))
                clean.save(cleanname)
                adv = Image.fromarray(np.uint8(255*tmpa))
                advname = os.path.join(path, "index{}_adv_l{}_t{}.jpeg".format(i, l[random_index[i]], d[random_index[i]] ))
                # advname = os.path.join(path, "index{}_adv_l{}_t{}_{}.jpeg".format(i, l[random_index[i]], d[random_index[i]], time.time()))
                adv.save(advname)
                per = np.uint8(255*tmpc) - np.uint8(255*tmpa)
                per = per + np.abs(np.min(per))
                per = Image.fromarray(np.uint8(255*per))
                pername = os.path.join(path, "index{}_per_l{}_t{}.jpeg".format(i, l[random_index[i]], d[random_index[i]])) 
                per.save(pername)
                piclist = [cleanname,advname,pername]
                if i >= save_num - 1:
                    break
        return piclist
    def print_res(self) -> None:
        print("Before {} attack, the accuracy on benign test examples: {}%".format(self.method, self.psrb* 100))
        print("After {} attack, the accuracy on adversarial test examples: {}%".format(self.method, self.psra*100))
        print("The final {} attack success rate: {}%, coverrate: {}%".format(self.method, (self.asr)*100, (self.coverrate)*100))
        print("Time spent per sample: {}s".format(self.timespend))
        return {"before_acc":round(self.psrb, 4) * 100,"after_acc":round(self.psra, 4) * 100, "asr":round(self.asr, 4) * 100, "coverrate":round(self.coverrate, 4) * 100, "time":round(self.timespend, 4)}

class BackdoorAttacker():
    def __init__(
    self, modelnet=None, modelpath: str="./models/model_ckpt/ckpt-resnet18-mnist_epoch3_acc0.9898.pth", 
    dataset="mnist", datasetpath="./datasets/", nb_classes=10, datanormalize: bool = False, 
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),test_sample_num=1000,train_sample_num=1000) -> None:
        self.modelnet = modelnet
        self.modelpath = modelpath
        self.dataset = dataset
        self.datasetpath = datasetpath
        self.device = device
        self.nb_classes = nb_classes
        self.datanormalize = datanormalize
        self.norm_param = ""
        if dataset == "cifar10":
            self.loaddataset = load_cifar10
        elif dataset == "mnist":
            self.loaddataset = load_mnist
        else:
            raise ValueError("Dataset not supported")
        # 数据集加载
        self.input_shape, (self.x_select, self.y_select), (self.x_train, self.y_train), _, self.min_pixel_value, self.max_pixel_value, (self.train_x_select, self.train_y_select )= self.loaddataset(self.datasetpath, samplenum=test_sample_num, train_sample_num = train_sample_num)
        
    # 添加固定样式的后门
    def add_backdoor(self, x: np.ndarray):
        if self.px+self.l >= x.shape[2] or self.py+self.l >= x.shape[3]:
            raise ValueError("Invalid px or py!")
        xt = x.copy()
        xt[:, :, self.px:self.px+self.l, self.py:self.py+self.l] = self.value
        return xt
    
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
            self.accuracyonb, _ = compute_accuracy(self.classifier.predict(x_clean, batch_size=batch_size), y_clean)
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

    def save_model(self, path:str="./output/cache/", desc:str="backdoor_model", epoch:int=0):
        path = os.path.join(path, self.method)
        if not os.path.exists(path):
            try:
                os.makedirs(path)
            except Exception as e:
                shutil.rmtree(path)
                os.makedirs(path)
        print(f"path:{path}")
        print(f"exist:{os.path.exists(path)}")
        path =os.path.join( path , desc + ".pth")
        torch.save(self.classifier.model.state_dict(), path)

    def save_examples(self, save_num, label: np.ndarray, clean_example: np.ndarray, poisoned_example: np.ndarray, path:str="output/cache/results/"):
        if self.norm_param != '':
            path = os.path.join(path, self.method+'L'+self.norm_param)
        else:
            path = os.path.join(path, self.method)
        if save_num <= 0:
            return
        try:
            os.makedirs(path)
        except Exception as e:
            shutil.rmtree(path)
            os.makedirs(path)

        else:
            random_index = random.sample(range(0, len(clean_example)), len(clean_example))
            l = np.argmax(label,1)
            for i in range(len(clean_example)):
                if clean_example[random_index[i]].shape[0] == 1:
                    tmpc = clean_example[random_index[i]][0]
                    tmpa = poisoned_example[random_index[i]][0]
                else:
                    tmpc = clean_example[random_index[i]].transpose(1,2,0)
                    tmpa = poisoned_example[random_index[i]].transpose(1,2,0)
                clean = Image.fromarray(np.uint8(255*tmpc))
                clean.save(os.path.join(path,"index{}_clean.jpeg".format(i, l[i])))
                adv = Image.fromarray(np.uint8(255*tmpa))
                adv.save(os.path.join(path, "index{}_poisoned.jpeg".format(i, l[i])))
                backdoor = np.uint8(255*tmpc) - np.uint8(255*tmpa)
                backdoor = backdoor + np.abs(np.min(backdoor))
                backdoor = Image.fromarray(np.uint8(255*backdoor))
                backdoor.save(os.path.join(path, "index{}_backdoor.jpeg".format(i, l[i])))
                if i >= save_num - 1:
                    break
    
    # 训练集投毒函数
    # **kwargs中是除了poision函数参数以外的其他投毒参数
    def poision(self, method: str="PoisoningAttackBackdoor", pp_poison: float=0.001, save_num: int=32, target: int=3, trigger: List=[25, 25, 2, 1], **kwargs):
        
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
        # self.datasetpath = self.datasetpath

        # 训练参数
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        # 数据集加载
        # self.input_shape, (self.x_select, self.y_select), (self.x_train, self.y_train), _, self.min_pixel_value, self.max_pixel_value, (self.train_x_select, self.train_y_select )= self.loaddataset(self.datasetpath, samplenum=100, train_sample_num=1000)
        
        # 数据集规范
        # x_train_tmp = self.train_x_select.astype(MY_NUMPY_DTYPE)
        # y_train_tmp = self.train_y_select.astype(MY_NUMPY_DTYPE)
        x_train_tmp = self.train_x_select.astype(MY_NUMPY_DTYPE)
        y_train_tmp = self.train_y_select.astype(MY_NUMPY_DTYPE)
        x_select_tmp = self.x_select.astype(MY_NUMPY_DTYPE)
        y_select_tmp = self.y_select.astype(MY_NUMPY_DTYPE)
        # 构造分类器方便使用
        self.classifier = PyTorchClassifier (
            model=self.model,
            clip_values=(self.min_pixel_value, self.max_pixel_value),
            loss=self.criterion,
            optimizer=self.optimizer,
            input_shape=self.input_shape,
            nb_classes=10,
            device=self.device,
        )

        # 设置后门样式
        self.px=trigger[0]
        self.py=trigger[1]
        self.l=trigger[2]
        self.value=trigger[3]
        
        # 对于特征碰撞攻击，需要设置原目标类
        self.source = np.zeros((10))
        self.source[(target+1)%self.classifier.nb_classes] = 1
        self.target = np.zeros((10))
        self.target[target] = 1

        # 首先根据方法设置默认参数
        attackbase = PoisoningAttackBackdoor(self.add_backdoor)
        if self.method == "PoisoningAttackBackdoor":
            self.attack = eval(self.method)(self.add_backdoor)
        elif self.method == "PoisoningAttackCleanLabelBackdoor":
            self.attack = eval(self.method)(target=self.target, pp_poison=pp_poison, proxy_classifier=self.classifier, backdoor=attackbase)
        elif self.method == "FeatureCollisionAttack":
            # 默认将最后一层作为特征层
            name = [item[0] for item in self.model._modules.items()]
            fl = name[len(name)-1]
            backdoor_item = x_select_tmp.take([0], axis=0)
            self.attack = eval(self.method)(classifier=self.classifier, target=backdoor_item, feature_layer=fl)
        else:
            # PoisoningAttackAdversarialEmbedding
            self.attack = eval(self.method)(attackbase, self.device)
        
        # 然后根据用户输入设置攻击参数
        print(self.method)
        for key, value in kwargs.items():
            if key in ["save_path","test_sample_num","train_sample_num"]:
                continue
            if key not in self.attack.attack_params:
                error = "{} is not the parameter of {}".format(key, self.method)
                raise ValueError(error)
            else:
                if key == "norm":
                    self.norm_param = str(value)
                setattr(self.attack, key, value)
        for key in self.attack.attack_params:
            try:
                print(key, str(getattr(self.attack, key)))
            except:
                pass
        
        # 根据比例进行数据集投毒
        # 首先完成数据的copy，防止被篡改
        # 测试集默认投毒50%
        xtrp, ytrp = x_train_tmp.copy(), y_train_tmp.copy()
        train_total_num = len(xtrp)
        xtep, ytep = x_select_tmp.copy(), y_select_tmp.copy()
        test_total_num = len(xtep)
        if self.method in ["PoisoningAttackBackdoor", "PoisoningAttackAdversarialEmbedding"]:
            self.train_poisoned_num = int(pp_poison*len(x_train_tmp))
            self.test_poisoned_num = int(0.5*len(x_select_tmp))
        else:
            self.train_poisoned_num = int(pp_poison*np.sum(np.argmax(y_train_tmp, axis=1) == np.argmax(self.target, axis=0)))
            self.test_poisoned_num = int(0.5*np.sum(np.argmax(y_select_tmp, axis=1) == np.argmax(self.target, axis=0)))
        print("投毒样本数目:{}".format(self.train_poisoned_num))
        if self.train_poisoned_num == 0:
            return -1
        

        if self.method == "PoisoningAttackBackdoor":
            # 训练集投毒
            self.train_list = random.sample(range(0, train_total_num), self.train_poisoned_num)
            x_needp = xtrp.take(self.train_list, axis=0)
            x_p, y_p = self.attack.poison(x_needp, self.target)
            for i, j in enumerate(self.train_list):
                xtrp[j,:,:,:] = x_p[i,:,:,:]
                ytrp[j,:] = y_p[i,:]
            self.x_train_poisoned = xtrp
            self.y_train_poisoned = ytrp
            # 测试集投毒
            self.test_list = random.sample(range(0, test_total_num), self.test_poisoned_num)
            x_needp = xtep.take(self.test_list, axis=0)
            x_p, y_p = self.attack.poison(x_needp, self.target)
            for i, j in enumerate(self.test_list):
                xtep[j,:,:,:] = x_p[i,:,:,:]
                ytep[j,:] = y_p[i,:]
            self.x_test_poisoned = xtep
            self.y_test_poisoned = ytep
        elif self.method == "PoisoningAttackCleanLabelBackdoor":
            # self.attack.poison(x, y)
            # 训练集投毒
            setattr(self.attack, "pp_poison", pp_poison)
            self.x_train_poisoned, self.y_train_poisoned, self.train_list = self.attack.poison(xtrp, ytrp)
            # 测试集投毒
            setattr(self.attack, "pp_poison", 0.5)
            self.x_test_poisoned, self.y_test_poisoned, self.test_list = self.attack.poison(xtep, ytep)
        elif self.method == "FeatureCollisionAttack":
            # 训练集投毒
            # 默认将最后一层作为特征层
            all_indices = np.arange(len(xtrp))
            target_indices = all_indices[np.all(ytrp == self.target, axis=1)]
            target_selected_indices = np.random.choice(target_indices, self.train_poisoned_num)
            xp = xtrp.take(target_selected_indices, axis=0)

            source_indices = all_indices[np.all(ytrp == self.source, axis=1)]
            source_selected_indices = np.random.choice(source_indices, self.train_poisoned_num)
            backdoor_item = xtrp.take(source_selected_indices, axis=0)

            for index in tqdm(range(len(xp)), desc="Feature collision"):
                setattr(self.attack, "target", np.expand_dims(backdoor_item[index], axis=0))
                xp[index] = self.attack.poison(np.expand_dims(xp[index], axis=0))[0]
            
            for i, j in enumerate(target_selected_indices):
                xtrp[j,:,:,:] = xp[i,:,:,:]
            
            self.x_train_poisoned = xtrp
            self.y_train_poisoned = ytrp
            self.train_list = target_selected_indices

            # 测试集投毒
            self.x_test_poisoned = x_select_tmp
            self.y_test_poisoned = y_select_tmp
            self.test_list = np.arange(len(x_select_tmp))[np.all(y_select_tmp == self.target, axis=1)]
        else:
            # PoisoningAttackAdversarialEmbedding
            # 训练集投毒
            self.train_list = random.sample(range(0, train_total_num), self.train_poisoned_num)
            x_needp = xtrp.take(self.train_list, axis=0)
            x_p, y_p = self.attack.poison(x_needp, self.target)
            for i, j in enumerate(self.train_list):
                xtrp[j,:,:,:] = x_p[i,:,:,:]
                ytrp[j,:] = y_p[i,:]
            self.x_train_poisoned = xtrp
            self.y_train_poisoned = ytrp
            # 测试集投毒
            self.test_list = random.sample(range(0, test_total_num), self.test_poisoned_num)
            x_needp = xtep.take(self.test_list, axis=0)
            x_p, y_p = self.attack.poison(x_needp, self.target)
            for i, j in enumerate(self.test_list):
                xtep[j,:,:,:] = x_p[i,:,:,:]
                ytep[j,:] = y_p[i,:]
            self.x_test_poisoned = xtep
            self.y_test_poisoned = ytep
        
        # 保存部分投毒样本
        if  "save_path" in kwargs.keys():
            self.save_examples(save_num=save_num, label=y_train_tmp[self.train_list], clean_example=x_train_tmp[self.train_list], poisoned_example=self.x_train_poisoned[self.train_list], path=kwargs['save_path'])
        else:
            self.save_examples(save_num=save_num, label=y_train_tmp[self.train_list], clean_example=x_train_tmp[self.train_list], poisoned_example=self.x_train_poisoned[self.train_list])
        return 1
    # 在投毒数据集上进行训练
    def train(self, num_epochs: int=20, batch_size: int=128, save_model: bool = True):
        self.batch_size = batch_size

        num_batch = int(np.ceil(len(self.x_train_poisoned) / float(batch_size)))
        tmp_model = None
        for i in tqdm(range(num_epochs), desc="Training"):
            if self.method != "PoisoningAttackAdversarialEmbedding":
                self.classifier.fit(self.x_train_poisoned, self.y_train_poisoned, nb_epochs=1, batch_size=num_batch, show=False)
                all, clean, backdoor = self.evaluate(self.x_test_poisoned, self.y_test_poisoned, self.test_list)
            else:
                tmp_model,_,_ = self.attack.fintune(self.model, i, self.x_train_poisoned, self.y_train_poisoned, self.train_list, batch_size, tmp_model)
                all, clean, backdoor = self.evaluate(self.x_test_poisoned, self.y_test_poisoned, self.test_list)
            print((all, clean, backdoor))
        if save_model:
            self.save_model(desc="all{}_clean{}_backdoor{}".format(all, clean, backdoor), epoch=i)

        return all, clean, backdoor

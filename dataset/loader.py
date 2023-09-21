__author__ = 'homeway'
__copyright__ = 'Copyright © 2021/07/01, ZJUICSR'


import os,sys
import numpy as np
import os.path as osp
from copy import deepcopy
from torchvision import datasets, transforms
from dataset import BaseDataLoader, ArgpDataLoader
from torchvision.datasets.mnist import MNIST
from torchvision.datasets.vision import VisionDataset
import warnings
from PIL import Image
import codecs
import string
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.error import URLError
from torchvision.datasets.utils import download_and_extract_archive, extract_archive, verify_str_arg, check_integrity
import shutil
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision.datasets import ImageFolder

class mydataset(MNIST):
    flag = False
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if self.flag:
            img = Image.fromarray(img,mode = 'L')    
        else:
            img = Image.fromarray(img.numpy(),mode = 'L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CustomDataset():
    def __init__(self,dataset):
        self.data_root = osp.join(osp.dirname(osp.abspath(__file__)),"data/Custom")
        self.size = [32, 32]
        # train_dataset = ImageFolder(root=self.data_root, transform=None)
        # self.mean,self.std = getStat(train_dataset)
    
    @staticmethod
    @classmethod
    def upload_dataset_entity(cls,dataset_id,zippath):
        '''
        :parms dataset_id: str, unique index of a dataset
        '''
        
        datasetInfo=IOtool.load_json(osp.join(cls.data_root,'datasetInfo.json'))
        dataset_name=datasetInfo['customize'][dataset_id]

        outputPath=osp.join(cls.data_root,dataset_id)

        iniInfo=IOtool.load_json(osp.join(outputPath,'info.json'))

        num=iniInfo['class_num']
        labelmap=iniInfo['label_map']

        label={}
        for idlabel in labelmap.keys():
            label[labelmap[idlabel]]=int(idlabel)

        if num!=0:
            print("********************")
            x=np.load(osp.join(outputPath,'x.npy'),allow_pickle=True)
            y=np.load(osp.join(outputPath,'y.npy'),allow_pickle=True)
            x=list(x)
            y=list(y)
        else:
            x=[]
            y=[]

        with zipfile.ZipFile(zippath, mode='r') as zfile: # 只读方式打开压缩包
            nWaitTime = 1
            for name in zfile.namelist():  # 获取zip文档内所有文件的名称列表
                if '.jpg' not in name:# 仅读取.jpg图片，过滤掉文件夹，及其他非.jpg后缀文件
                    continue
                
                with zfile.open(name,mode='r') as image_file:
                    content = image_file.read() # 一次性读入整张图片信息
                    # image = np.asarray(bytearray(content), dtype='uint8')
                    # image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                
                    image = Image.open(BytesIO(content))
                    InitialSize=image.size
                    maxSize=max(InitialSize)
                    if maxSize>224:
                        image=image.resize((32,32))
                        # print(image.size)

                    img=np.array(image).astype(np.uint8)
                    
                    reallabel=(name.split('_'))[1]
                    reallabel=(reallabel.split('.'))[0]
                    x.append(img)
                    if reallabel not in label:
                        label[reallabel]=num
                        num+=1
                    y.append(label[reallabel])
    
    # @classmethod
    # def get_npy(cls,dataset_id):
    #     x = np.load(osp.join(cls.data_root,dataset_id,"x.npy"))
    #     transforms = cls.get_transforms()
    
class LoadCustom(Dataset):
    """
    """
    def __init__(self, dataset_id,transform = None):
        self.x_data =np.load( osp.join(osp.dirname(osp.abspath(__file__)),"data/Custom",dataset_id,"x.npy"))
        self.y_data = np.load(osp.join(osp.dirname(osp.abspath(__file__)),"data/Custom",dataset_id,"y.npy"))
        if transform == None:
            self.transforms = transforms.Compose([
                        transforms.ToTensor(),
                    ])
        else:
            self.transforms = transform
        self.size =[32,32]
    def __getitem__(self, index):
        hdct= self.x_data[index, :, :, :]  # 读取每一个npy的数据
        ldct = self.y_data[index]
        hdct=Image.fromarray(np.uint8(hdct)) #转成image的形式   
        hdct= self.transforms(hdct)  #转为tensor形式
        return hdct, ldct #返回数据还有标签
    

    def __len__(self):
        return self.x_data.shape[0] #返回数据的总个数


# This class provide default dataset config
class ArgpLoader(BaseDataLoader):
    def __init__(self, data_root, dataset="CIFAR10", **kwargs):
        self.support = ["MNIST", "CIFAR10", "CIFAR100", "Imagenet1k", "CUBS200", "SVHN"]
        if dataset not in self.support:
            raise ValueError(f"-> System don't support {dataset}!!!")
        super(ArgpLoader, self).__init__(data_root, dataset, **kwargs)

        self.dataset = dataset
        self.data_path = osp.join(data_root, dataset)
        if not osp.exists(self.data_path):
            os.makedirs(self.data_path)

        self._train = True
        self.train_loader = None
        self.test_loader = None
        self.train_transforms = None
        self.test_transforms = None
        self.params = self.__check_params__()
        self.get_transforms()

    @staticmethod
    def __config__(dataset):
        channel = 3
        if dataset.lower() == "mnist":
            channel = 1
        params = {}
        params["name"] = dataset
        params["batch_size"] = 256
        params["size"] = ArgpLoader.get_size(dataset)
        params["shape"] = (channel, params["size"][0], params["size"][1])
        params["mean"], params["std"] = ArgpLoader.get_mean_std(dataset)
        params["bounds"] = ArgpLoader.get_bounds(dataset)
        params["num_classes"] = ArgpLoader.get_num_classes(dataset)
        params["path"] = osp.join("dataset/data", dataset)
        params["labels"] = ArgpLoader.get_labels(dataset)

        return params

    def __call__(self):
        if self._train:
            if self.train_loader is not None:
                for x, y in self.train_loader:
                    yield x, y
        else:
            if self.test_loader is not None:
                for x, y in self.test_loader:
                    yield x, y

    @staticmethod
    def get_num_classes(dataset):
        dnum = {
            "CIFAR10": 10,
            "MNIST": 10,
            "CIFAR100": 100,
            "Imagenet1k": 1000,
            "SVHN": 10,
            "CUBS200": 200
        }
        return dnum[dataset]

    @staticmethod
    def get_mean_std(dataset):
        attribute = {
            "MNIST": [(0.1307), (0.3081)],
            "CIFAR": [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)],
            "Imagenet1k": [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)],
            "CUBS200": [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)],
            "SVHN": [(0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)]
        }
        attribute["CIFAR10"] = deepcopy(attribute["CIFAR"])
        attribute["CIFAR100"] = deepcopy(attribute["CIFAR"])
        return attribute[dataset]

    @staticmethod
    def get_size(dataset):
        attribute = {
            "MNIST": [32, 32],
            "CIFAR": [32, 32],
            "Imagenet1k": [299, 299],
            "CUBS200": [256, 256],
            "SVHN": [128, 128]
        }
        attribute["CIFAR10"] = deepcopy(attribute["CIFAR"])
        attribute["CIFAR100"] = deepcopy(attribute["CIFAR"])
        return attribute[dataset]

    @staticmethod
    def get_bounds(dataset):
        mean, std = ArgpLoader.get_mean_std(dataset)
        bounds = [-1, 1]
        if type(mean) == type(()):
            c = len(mean)
            _min = (np.zeros([c]) - np.array(mean)) / np.array([std])
            _max = (np.ones([c]) - np.array(mean)) / np.array([std])
            bounds = [np.min(_min).item(), np.max(_max).item()]
        elif type(mean) == float:
            bounds = [(0.0 - mean) / std, (1.0 - mean) / std]
        return bounds

    def is_train(self, train=True):
        self._train = bool(train)

    def set_transforms(self, train_transforms, test_transforms):
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms

    def get_transforms(self):
        if (self.train_transforms is not None) \
                and (self.test_transforms is not None):
            return self.train_transforms, self.test_transforms
        attribute = {
            "MNIST": [
                transforms.Compose([
                    transforms.Resize(self.size),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)
                ]), transforms.Compose([
                    transforms.Resize(self.size),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)
                ])
            ],
            "CIFAR": [
                transforms.Compose([
                    transforms.RandomCrop(self.size, padding=2),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ]),
                transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ])
            ],
            "Imagenet1k": [
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ]),
                transforms.Compose([
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ])
            ],
            "CUBS200": [
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ]),
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ])
            ],
            "SVHN": [
                transforms.Compose([
                    transforms.RandomResizedCrop(self.size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ]),
                transforms.Compose([
                    transforms.CenterCrop(self.size),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ])
            ]
        }
        attribute["CIFAR10"] = deepcopy(attribute["CIFAR"])
        attribute["CIFAR100"] = deepcopy(attribute["CIFAR"])
        self.train_transforms, self.test_transforms = attribute[self.dataset]
        return attribute[self.dataset]

    @staticmethod
    def get_labels(dataset):
        dst = dataset.upper()
        results = {
            "CIFAR10": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
            "CIFAR100": ["apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur", "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear", "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone", "television", "tiger", "tractor", "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"],
            "MNIST": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        }
        return results[dst]

    def get_loader(self, **kwargs):
        train_transforms, test_transforms = self.get_transforms()
        if self.dataset == "MNIST":
            self.train_loader, self.test_loader = [
                ArgpDataLoader(
                    dataset=datasets.MNIST(self.data_path, train=True, download=True, transform=train_transforms),
                    # dataset=mydataset(self.data_path, train=True, download=True, transform=train_transforms),
                    batch_size=self.batch_size,
                    shuffle=True, num_workers=2,
                    **kwargs
                ),
                ArgpDataLoader(
                    dataset=datasets.MNIST(self.data_path, train=False, download=True, transform=test_transforms),
                    # dataset=mydataset(self.data_path, train=False, download=True, transform=test_transforms),
                    batch_size=self.batch_size,
                    shuffle=False, num_workers=2,
                    **kwargs
                )]
        elif self.dataset == "CIFAR10":
            self.train_loader, self.test_loader = [
                ArgpDataLoader(
                    dataset=datasets.CIFAR10(self.data_path, train=True, download=True, transform=train_transforms),
                    batch_size=self.batch_size,
                    shuffle=True, num_workers=2,
                    **kwargs
                ),
                ArgpDataLoader(
                    dataset=datasets.CIFAR10(self.data_path, train=False, download=True, transform=test_transforms),
                    batch_size=self.batch_size,
                    shuffle=False, num_workers=2,
                    **kwargs
                )
            ]
        elif self.dataset == "CIFAR100":
            self.train_loader, self.test_loader = [
                ArgpDataLoader(
                    dataset=datasets.CIFAR100(self.data_path, train=True, download=True, transform=train_transforms),
                    batch_size=self.batch_size,
                    shuffle=True, num_workers=2,
                    **kwargs
                ),
                ArgpDataLoader(
                    dataset=datasets.CIFAR100(self.data_path, train=False, download=True, transform=test_transforms),
                    batch_size=self.batch_size,
                    shuffle=False, num_workers=2,
                    **kwargs
                )
            ]
        elif self.dataset == "SVHN":
            self.train_loader, self.test_loader = [
                ArgpDataLoader(
                    dataset=datasets.SVHN(self.data_path, split='train', download=True, transform=train_transforms),
                    batch_size=self.batch_size,
                    shuffle=True, num_workers=2,
                    **kwargs
                ),
                ArgpDataLoader(
                    dataset=datasets.SVHN(self.data_path, split='test', download=True, transform=test_transforms),
                    batch_size=self.batch_size,
                    shuffle=False, num_workers=2,
                    **kwargs
                )
            ]
        
        else:
            raise NotImplementedError(f"-> Can't find {self.dataset} implementation!!")

        self.train_loader.set_params(self.params)
        self.train_loader.set_params({
            "transforms": self.train_transforms
        })
        self.test_loader.set_params(self.params)
        self.test_loader.set_params({
            "transforms": self.test_transforms
        })
        
        return self.train_loader, self.test_loader


if __name__ == '__main__':
    dataset = LoadCustom("1009_test")
    print(dataset[2])
    print(dataset[2][0].shape)
    data_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)
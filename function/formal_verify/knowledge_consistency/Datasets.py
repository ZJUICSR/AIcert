import os
import numpy as np
import torch
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data
import torchvision.transforms.functional as F
from PIL import Image
import random
import pickle

class DiscreteDataset(torch.utils.data.Dataset):
    def __init__(self, dataset='CUB200', layer=40, suffix='', t_or_v='train', path=''):
        super(type(self), self).__init__()
        self.dir = path
        self.len = os.listdir(self.dir).__len__()

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        meta = torch.load(self.dir+"{:06d}.pkl".format(idx))
        return meta


class NewVOCDataset(torch.utils.data.Dataset):
    def __init__(self, img_list, trans=None, path='../VOCdevkit/VOC2012/JPEGImages/'):
        super(type(self), self).__init__()
        with open(img_list, 'rb+') as f:
            self.data_list = pickle.load(f)
            self.path = path
            self.trans = trans


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        xmin = int(self.data_list[idx]["bndbox"]["xmin"])
        xmax = int(self.data_list[idx]["bndbox"]["xmax"])
        ymin = int(self.data_list[idx]["bndbox"]["ymin"])
        ymax = int(self.data_list[idx]["bndbox"]["ymax"])
        img_dir = self.path + self.data_list[idx]["source"] + '.jpg'
        img = Image.open(img_dir)
#         obj = transforms.ToTensor()(img)
        obj = transforms.functional.crop(img, ymin, xmin, ymax-ymin, xmax-xmin)
        if self.trans:
            obj = self.trans(obj)
        tar = self.data_list[idx]["class"]
        return (obj,tar)


def Generate_Dataloader(dataset, batch_size, workers=4, suffix='',sample_num = '', train_path='', val_path='',onlyone=None):
    if dataset == 'ilsvrc':
        # Data loading code
        traindir = train_path + 'ILSVRC2012_sampling_{}'.format(sample_rate) # Deprecated!
        valdir = val_path

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
            num_workers=workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True)

    elif dataset == 'CUB200' or dataset=='mix320-cub':
        # Data loading code
        traindir = "../CUB_200_2011/crop/train/"
        valdir = "../CUB_200_2011/crop/test/"

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_sampler = None

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                normalize,
            ]))

        if sample_num!='':
            if os.path.exists("sub_sampler_{}_{}.pt".format(dataset,sample_num)):
                print("Find sub_sampler_{}_{}.pt!".format(dataset,sample_num))
                sub_idx = torch.load("sub_sampler_{}_{}.pt".format(dataset,sample_num))
            else:
                return

            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size,
                num_workers=workers, pin_memory=True, sampler=SubsetRandomSampler(sub_idx))
        
        else:

            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                num_workers=workers, pin_memory=True, sampler=None)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True)
    
    elif dataset == 'mix320':
        # Data loading code
        traindir = "../mixedData/train/"
        valdir = "../mixedData/test/"

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_sampler = None

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                normalize,
            ]))

        if sample_num!='':
            if os.path.exists("sub_sampler_{}_{}.npy".format(dataset,suffix)):
                print("Find sub_sampler_{}_{}.npy!".format(dataset,suffix))
                sub_idx = np.load("sub_sampler_{}_{}.npy".format(dataset,suffix)).tolist()
            else:
                return

            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size,
                num_workers=workers, pin_memory=True, sampler=SubsetRandomSampler(sub_idx))
        
        else:

            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                num_workers=workers, pin_memory=True, sampler=None)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True)

    
    elif dataset == 'DOG120' or dataset == 'mix320-dog':
        # Data loading code
        traindir = "../DOG120/crop/train/"
        valdir = "../DOG120/crop/val/"

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_sampler = None

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                normalize,
            ]))

        if sample_num!='':
            if os.path.exists("sub_sampler_{}_{}.npy".format(dataset,sample_num)):
                print("Find sub_sampler_{}_{}.npy!".format(dataset,sample_num))
                sub_idx = np.load("sub_sampler_{}_{}.npy".format(dataset,sample_num)).tolist()
            else:
                portion = int(sample_num)/100
                class_map = [[] for i in range(120)]
                for i in range(len(train_dataset)):
                    class_map[train_dataset[i][1]].append(i)
                sub_idx = []
                for c in class_map:
                    sub_idx += c[:int(len(c)*portion)]

                np.save('sub_sampler_{}_{}.npy'.format(dataset, sample_num),sub_idx)
                print("create a sub_sampler!!")
                return 
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size,
                num_workers=workers, pin_memory=True, sampler=SubsetRandomSampler(sub_idx))       
        else:

            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                num_workers=workers, pin_memory=True, sampler=None)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True)
            
    elif dataset == 'VOC2012_crop' or dataset == 'VOC2012_animal':
        if dataset == 'VOC2012_crop':
            traindata = 'VOC_traindata.pkl'
            valdata = 'VOC_valdata.pkl'
        if dataset == 'VOC2012_animal':
            traindata = 'VOC_Animal_traindata.pkl'
            valdata = 'VOC_Animal_valdata.pkl'        

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        train_dataset = NewVOCDataset(traindata,
                        transforms.Compose([
                            transforms.Resize((224,224)),
                            transforms.ToTensor(),
                            normalize,
                        ]))
        val_dataset = NewVOCDataset(valdata,
                       transforms.Compose([
                           transforms.Resize((224,224)),
                           transforms.ToTensor(),
                           normalize,
                       ]))
        if sample_num!='':
            if os.path.exists("sub_sampler_{}_{}.npy".format(dataset,sample_num)):
                print("Find sub_sampler_{}_{}.npy!".format(dataset,sample_num))
                sub_idx = np.load("sub_sampler_{}_{}.npy".format(dataset,sample_num)).tolist()
            else:
                return

            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size,
                num_workers=workers, pin_memory=True, sampler=SubsetRandomSampler(sub_idx))
        else:
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                num_workers=workers, pin_memory=True, sampler=None)

        val_loader = torch.utils.data.DataLoader(val_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True)
    elif dataset == 'cifar10':
        import torchvision
        test_data = torchvision.datasets.CIFAR10(
        "/mnt/data/AISec/backend/data", train=False, download=True, 
        transform=torchvision.transforms.Compose([torchvision.transforms.Resize((224,224)),torchvision.transforms.ToTensor(), 
                                  torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])]))
        train_data = torchvision.datasets.CIFAR10(
        "/mnt/data/AISec/backend/data", train=False, download=True, 
        transform=torchvision.transforms.Compose([torchvision.transforms.Resize((224,224)),torchvision.transforms.ToTensor(), 
                                  torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])]))
        train_loader= torch.utils.data.DataLoader(train_data, batch_size=batch_size,shuffle=True, num_workers=workers)
        if onlyone:
            data=[]
            for item in test_data:
                if item[1]==onlyone:
                    data.append(item)
            val_loader= torch.utils.data.DataLoader(data, batch_size=batch_size,shuffle=True, num_workers=workers)
        else:
            val_loader= torch.utils.data.DataLoader(test_data, batch_size=batch_size,shuffle=True, num_workers=workers)
    elif dataset == 'mnist':
        import torchvision
        test_data = torchvision.datasets.MNIST(
        "/mnt/data/AISec/backend/data", train=False, download=True, 
        transform=torchvision.transforms.Compose([torchvision.transforms.Resize((224,224)),torchvision.transforms.Grayscale(num_output_channels=3),torchvision.transforms.ToTensor(), 
                                  torchvision.transforms.Normalize((0.1307,), (0.3081,))]))
        train_data = torchvision.datasets.CIFAR10(
        "/mnt/data/AISec/backend/data", train=False, download=True, 
        transform=torchvision.transforms.Compose([torchvision.transforms.Resize((224,224)),torchvision.transforms.Grayscale(num_output_channels=3),torchvision.transforms.ToTensor(), 
                                  torchvision.transforms.Normalize((0.1307,), (0.3081,))]))
        train_loader= torch.utils.data.DataLoader(train_data, batch_size=batch_size,shuffle=True, num_workers=workers)
        if onlyone!=None:
            data=[]
            for item in test_data:
                if item[1]==onlyone:
                    data.append(item)
            val_loader= torch.utils.data.DataLoader(data, batch_size=batch_size,shuffle=True, num_workers=workers)
        else:
            val_loader= torch.utils.data.DataLoader(test_data, batch_size=batch_size,shuffle=True, num_workers=workers)
    return train_loader, val_loader

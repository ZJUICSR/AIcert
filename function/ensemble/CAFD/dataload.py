import torch
from torch.utils.data import Dataset 
import re
import pickle
import numpy as np
from PIL import Image
import os
import random


def sort_key(s):
    #sort_strings_with_embedded_numbers
    re_digits = re.compile(r'(\d+)')
    pieces = re_digits.split(s)  
    pieces[1::2] = map(int, pieces[1::2])  
    return pieces


def load_variavle(filename):
    f=open(filename,'rb')
    r=pickle.load(f)
    f.close()
    return r


class DatasetIMG(Dataset): 
    """
    Clean dataset.
    Args:
        img_dirs: dir list to clean images from.
    """

    def __init__(self, img_dirs, label_dirs, transform=None):
        self.img_dirs = img_dirs
        self.label_dirs = label_dirs
        self.img_names = self.__get_imgnames__()
        self.label = self.__get_label__()
        self.transform = transform

    def __get_imgnames__(self):
        tmp = []
        images_name = os.listdir(self.img_dirs)
        images_name.sort(key=sort_key)
        for name in images_name:
            tmp.append(os.path.join(self.img_dirs, name))
        return tmp

    def __get_label__(self):
        label = load_variavle(self.label_dirs)
        label = np.array(label)
        label = torch.from_numpy(label)
        return label

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        image_path   = self.img_names[idx]
        image = Image.open(image_path).convert('RGB')

        label = self.label[idx]

        if self.transform:
            image = self.transform(image)
        return image, label


class DatasetIMG2D(Dataset):  
    """
    Clean dataset.
    Args:
        img_dirs: dir list to clean images from.
    """

    def __init__(self, img_dirs, label_dirs, transform=None):
        self.img_dirs = img_dirs
        self.label_dirs = label_dirs
        self.img_names = self.__get_imgnames__()
        self.label = self.__get_label__()
        self.transform = transform

    def __get_imgnames__(self):
        tmp = []
        images_name = os.listdir(self.img_dirs)
        images_name.sort(key=sort_key)
        for name in images_name:
            tmp.append(os.path.join(self.img_dirs, name))
        return tmp

    def __get_label__(self):
        label = load_variavle(self.label_dirs)
        label = np.array(label)
        label = torch.from_numpy(label)
        return label

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        image_path   = self.img_names[idx]
        image        = Image.open(image_path).convert('L')

        label = self.label[idx]

        if self.transform:
            image = self.transform(image)
        return image, label


class DatasetNPY(Dataset): 
    """
    Clean dataset.
    Args:
        img_dirs: dir list to clean images from.
    """

    def __init__(self, img_dirs, label_dirs, transform = None):  
        self.img_dirs = img_dirs
        self.img_names = self.__get_imgnames__()
        self.label_dirs = label_dirs
        self.label = self.__get_label__()
        self.transform = transform

    def __get_imgnames__(self):
        tmp = []
        images_name = os.listdir(self.img_dirs)
        images_name.sort(key=sort_key)
        for name in images_name:
            tmp.append(os.path.join(self.img_dirs, name))
        return tmp

    def __get_label__(self):
        label = load_variavle(self.label_dirs)
        label = np.array(label)
        label = torch.from_numpy(label)
        return label

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        image_path   = self.img_names[idx]
        image        = np.load(image_path)
        image = image.astype(np.float32)

        label = self.label[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class DatasetIMG_Dual(Dataset):  
    """
    Clean dataset.
    Args:
        img_dirs: dir list to clean images from.
    """

    def __init__(self, imgcln_dirs, imgadv_dirs, label_dirs, transform=None):
        self.imgcln_dirs = imgcln_dirs
        self.imgadv_dirs = imgadv_dirs
        self.label_dirs = label_dirs
        self.img_names = self.__get_imgnames__()
        self.label = self.__get_label__()
        self.transform = transform

    def __get_imgnames__(self):
        tmp = []
        images_name = os.listdir(self.imgcln_dirs)
        images_name.sort(key=sort_key)
        for name in images_name:
            tmp.append(os.path.join(self.imgcln_dirs, name))
        return tmp

    def __get_label__(self):
        label = load_variavle(self.label_dirs)
        label = np.array(label)
        label = torch.from_numpy(label)
        return label

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        imagecln_path = self.img_names[idx]
        imagecln = Image.open(imagecln_path).convert('RGB')
        imageadv_path = imagecln_path.replace(self.imgcln_dirs, self.imgadv_dirs)
        imageadv = Image.open(imageadv_path).convert('RGB')

        label = self.label[idx]

        if self.transform:
            imagecln = self.transform(imagecln)
            imageadv = self.transform(imageadv)
        return imagecln, imageadv, label


class DatasetNPY_Dual(Dataset):  
    """
    Clean dataset.
    Args:
        img_dirs: dir list to clean images from.
    """

    def __init__(self, imgcln_dirs, imgadv_dirs, label_dirs, transform = None):  # shuffle:随机排列
        self.imgcln_dirs = imgcln_dirs
        self.imgadv_dirs = imgadv_dirs
        self.img_names = self.__get_imgnames__()
        self.label_dirs = label_dirs
        self.label = self.__get_label__()
        self.transform = transform

    def __get_imgnames__(self):
        tmp = []
        images_name = os.listdir(self.imgcln_dirs)
        images_name.sort(key=sort_key)
        for name in images_name:
            tmp.append(os.path.join(self.imgcln_dirs, name))
        return tmp

    def __get_label__(self):
        label = load_variavle(self.label_dirs)
        label = np.array(label)
        label = torch.from_numpy(label)
        return label

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        imagecln_path = self.img_names[idx]
        imagecln = np.load(imagecln_path)
        imagecln = imagecln.astype(np.float32)

        imageadv_path   = imagecln_path.replace(self.imgcln_dirs, self.imgadv_dirs)
        imageadv = np.load(imageadv_path)
        imageadv = imageadv.astype(np.float32)

        label = self.label[idx]

        if self.transform:
            imagecln = self.transform(imagecln)
            imageadv = self.transform(imageadv)

        return imagecln, imageadv, label

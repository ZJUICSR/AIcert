import glob
from PIL import Image
import matplotlib.pyplot as plt
from collections import OrderedDict
import scipy
import os
from typing import List
import numpy as np
import cv2
from pywt import dwt2, idwt2

import torch
import torchvision.transforms as transforms

#from models.resnet50 import ResNet50


# 这个是直接修改tensor自身，但我需要的是tensor不改变，返回一个改变后的值，因此定义一个类UnNorm
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

# 这个类传来的图片只能是(3,h,w)，不能是(1,3,h,w)，因为1会导致下面的zip只循环一次，也就是只用了
# 一个mean和norm，那么后面两个通道的结果就不对了
class UnNorm(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        len_4_flag = False
        if len(tensor.size()) == 4:
            len_4_flag = True
            tensor_temp = tensor.squeeze().clone()
        else:
            tensor_temp = tensor.clone()
        for t, m, s in zip(tensor_temp, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        if len_4_flag:
            tensor_temp = tensor_temp.unsqueeze(0)
        return tensor_temp



# DWT
def dwt_lpf(img:np.ndarray, mode:int=0)->np.ndarray:
    img_return = img.copy()
    height = int(img.shape[0]/2)
    zero_matrix = np.zeros((height, height))
    for i in range(3):
        img_temp = img_return[:, :, i]
        cA, (cH, cV, cD) = dwt2(img_temp, 'haar')
        if mode==0:
            img_temp = idwt2((cA, (zero_matrix, zero_matrix, zero_matrix)), 'haar')
        elif mode==1:
            img_temp = idwt2((cA, (cH, zero_matrix, zero_matrix)), 'haar')
        elif mode==2:
            img_temp = idwt2((cA, (zero_matrix, cV, zero_matrix)), 'haar')
        elif mode==3:
            img_temp = idwt2((cA, (zero_matrix, zero_matrix, cD)), 'haar')
        elif mode==4:
            img_temp = idwt2((cA, (zero_matrix, cV, cD)), 'haar')
        elif mode==5:
            img_temp = idwt2((cA, (cH, zero_matrix, cD)), 'haar')
        else:
            img_temp = idwt2((cA, (cH, cV, zero_matrix)), 'haar')
        img_return[:, :, i] = img_temp

    return img_return



# DFT
def dft_lpf(img:np.ndarray, magnitude:float=8.0)->np.ndarray:
    h, w = img.shape[:2]

    # 低通滤波器
    lpf = np.zeros((h, w, 3))
    R = (h + w) // magnitude  # 控制滤波器的大小
    for x in range(w):
        for y in range(h):
            if ((x - (w - 1) / 2) ** 2 + (y - (h - 1) / 2) ** 2) < (R ** 2):
                lpf[y, x, :] = 1

    freq = np.fft.fft2(img, axes=(0, 1))
    freq = np.fft.fftshift(freq)
    lf = freq * lpf
    lf = np.fft.ifftshift(lf)
    img_l = np.abs(np.fft.ifft2(lf, axes=(0, 1)))
    img_l = np.clip(img_l, 0, 255)  # 会产生一些过大值需要截断
    img_l = img_l.astype('uint8')

    return img_l


def get_noise(l2_norm:float)->torch.tensor:
    divisor = 387 / l2_norm
    noise_l2 = 999999
    noise = None
    while noise_l2 > l2_norm:
        noise = torch.randn(3, 224, 224) / divisor
        noise_l2 = torch.norm(noise, p=2)

    return noise

# 两数组中数值不同的元素的index
def get_index(pred:torch.tensor, result:torch.tensor, num:int=100)->List:
    index = []
    temp = torch.ne(pred, result)
    for i in range(num):
        if temp[i]:
            index.append(i)

    return index
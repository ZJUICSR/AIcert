from . import render
import numpy as np
import os
import json
import torchvision
import cv2
import copy
import os.path as osp
from PIL import Image

import torch
from torch.utils.data import DataLoader, TensorDataset

def get_normalize_para(dataset):
    dataset = dataset.lower()
    mean, std = None, None
    if dataset == "imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif dataset == "cifar10":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif dataset == "mnist":
        mean = [0.1307]
        std = [0.3081]
    else:
        raise NotImplementedError("not support for other data set!")
    return mean, std

def format_np_output(np_arr):
    """
        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
    Args:
        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr*255).astype(np.uint8)
    return np_arr

def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
    im.save(path)
    return path.split("/")[-5] + "/" + path.split("/")[-4] + "/" + path.split("/")[-3] + "/" + path.split("/")[-2] + "/" + path.split("/")[-1]

def save_gradient_images(gradient, file_name):
    """
        Exports the original gradient image
    Args:
        gradient (np arr): Numpy array of the gradient with shape (3, 224, 224)
        file_name (str): File name to be exported
    """
    # Normalize
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    # Save image
    path_to_file = os.path.join(file_name)
    return save_image(gradient, path_to_file)

'''测试存储json文件'''
def save_process_result(root, results, filename):
    path = osp.join(root, filename)

    with open(path, "w") as fp:
        json.dump(results, fp, indent=2)
        fp.close()

'''获取缓存的样本数据'''
def get_loader(url,batchsize=8):
    """
    对于ImageNet数据集，batchsize默认为8,
    当使用lrp在vgg系列上时，batchsize可以设置为16
    当使用lrp解释Resnet50时，batchsize最大为8; resnet152时，batch为4.
    """
    data = torch.load(url)
    adv_dst = TensorDataset(torch.cat(data["x"],dim=0).float().cpu() ,torch.cat(data["y"],dim=0).long().cpu())
    adv_loader = DataLoader(
        adv_dst,
        batch_size=batchsize,
        shuffle=False,
        num_workers=2
    )
    return adv_loader

'''将三通道图片转成单通道图片'''
def convert_to_grayscale(im_as_arr):
    """
        Converts 3d image to grayscale

    Args:
        im_as_arr (numpy arr): RGB image with shape (D,W,H)

    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.max(grayscale_im)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im

'''获取分类数据集的每个类名称（英文）'''
def get_class_list(dataset, root):
    if dataset == 'imagenet':
        imagenet_path = osp.join(root, "ex_methods/cache/imagenet_class_index.json")
        with open(os.path.abspath(imagenet_path), 'r') as read_file:
            class_idx = json.load(read_file)
            idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
            return idx2label
    elif dataset == 'mnist':
        return ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    elif dataset == 'cifar10':
        return ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    else:
        return "暂不支持其他数据集"

'''获取任务数据集的分类数量'''
def get_target_num(dataset):
    if dataset == "imagenet":
        return 1000
    else:
        return 10

'''将正则化后的图片（tensor）做逆变换成图片'''
def recreate_image(im_as_var):
    """
        Recreates images from a torch variable, sort of reverse preprocessing
    Args:
        im_as_var (torch variable): Image to recreate
    returns:
        recreated_im (numpy arr): Recreated image in array
    """
    # reverse_mean = []
    # reverse_std = []
    # if dataset == "cifar10":
    #     reverse_mean = [-0.43768206, -0.44376972, -0.47280434]
    #     reverse_std = [1 / 0.19803014, 1 / 0.20101564, 1 / 0.19703615]
    # elif dataset == "mnist":
    #     reverse_mean = [(-0.1307)]
    #     reverse_std = [(1 / 0.3081)]
    # elif dataset == "imagenet":
    #     reverse_mean = [-0.485, -0.456, -0.406]
    #     reverse_std = [1 / 0.229, 1 / 0.224, 1 / 0.225]
    recreated_im = copy.copy(im_as_var.data.cpu().numpy())
    # for c in range(3):
    #     recreated_im[c] /= reverse_std[c]
    #     recreated_im[c] -= reverse_mean[c]
    # recreated_im[recreated_im > 1] = 1
    # recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    return recreated_im

# 测试存储json文件
def save_process_result(root, results, filename):
    path = os.path.join(root, filename)

    with open(path, "w") as fp:
        json.dump(results, fp, indent=2)
        fp.close()


# def recreate_image(im_as_var, dataset):
#     """
#         Recreates images from a torch variable, sort of reverse preprocessing
#     Args:
#         im_as_var (torch variable): Image to recreate
#     returns:
#         recreated_im (numpy arr): Recreated image in array
#     """
#     reverse_mean = []
#     reverse_std = []
#     if dataset == "cifar10":
#         reverse_mean = [-0.43768206, -0.44376972, -0.47280434]
#         reverse_std = [1 / 0.19803014, 1 / 0.20101564, 1 / 0.19703615]
#     elif dataset == "mnist":
#         reverse_mean = [(-0.1307)]
#         reverse_std = [(1 / 0.3081)]
#     elif dataset == "imagenet":
#         reverse_mean = [-0.485, -0.456, -0.406]
#         reverse_std = [1 / 0.2023, 1 / 0.1994, 1 / 0.2010]
#     recreated_im = copy.copy(im_as_var.data.cpu().numpy())
#     for c in range(3):
#         recreated_im[c] /= reverse_std[c]
#         recreated_im[c] -= reverse_mean[c]
#     recreated_im[recreated_im > 1] = 1
#     recreated_im[recreated_im < 0] = 0
#     recreated_im = np.round(recreated_im * 255)

#     recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
#     return recreated_im

def preprocess_transform(image, dataset):
    if dataset == "imagenet":
        data_mean = np.array([0.485, 0.456, 0.406])
        data_std = np.array([0.229, 0.224, 0.225])
        img_size = 224
    elif dataset == "cifar10":
        data_mean = np.array([0.43768206, 0.44376972, 0.47280434])
        data_std = np.array([0.19803014, 0.20101564, 0.19703615])
        img_size = 32
    else:
        data_mean = 0.1307
        data_std = 0.3081
        img_size = 28

    normalize = torchvision.transforms.Normalize(mean=data_mean, std=data_std)
    transf = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(img_size),
        # torchvision.transforms.CenterCrop(img_size),
        normalize
    ])
    return transf(image)


def load_image(device, image, dataset):
    x = preprocess_transform(image, dataset)
    x = x.unsqueeze(0).to(device)
    return x

'''存储解释图'''
def save_ex_img(root, img, img_name):
    path = osp.join(root, "adv_explain", img_name)
    if not os.path.exists(root+'/adv_explain/'):
        os.makedirs(root+'/adv_explain/')
    img.save(path)
    return path.split("/")[-3] + "/" + path.split("/")[-2] + "/" + path.split("/")[-1]

'''将tensor存储的样本逆转换成image格式的list'''
def loader2imagelist(dataloader, dataset, size):
    image_list = []
    for i in range(size):
        img, label = dataloader.dataset[i]
        if dataset == "cifar10":
            img_numpy = recreate_image(img)
            img_x = Image.fromarray(img_numpy)
            image_list.append(img_x)
        elif dataset == "imagenet":
            img_x = torchvision.transforms.ToPILImage()(img)
            image_list.append(img_x)
        elif dataset == "mnist":
            img_x = torchvision.transforms.ToPILImage()(img)
            image_list.append(img_x)
        else:
            raise ValueError("not supported data set!")
    return image_list


def target_layer(model,dataset):
    target_layer_ = None
    model = model.lower()
    dataset = dataset.lower()
    if dataset == "imagenet":
        if model == 'vgg11':
            target_layer_ = '19'
        elif model == 'vgg13':
            target_layer_ = '23'
        elif model == 'vgg16':
            target_layer_ = '29'
        elif model == 'vgg19':
            target_layer_ = '35'
        elif model == 'resnet18':
            target_layer_ = '11'
        elif model == 'resnet34':
            target_layer_ = '19'
        elif model == 'resnet50':
            target_layer_ = '19'
        elif model == 'resnet101':
            target_layer_ = '36'
        elif model == 'resnet152':
            target_layer_ = '53'
        elif model == 'densenet121':
            target_layer_ = '64'
        elif model == 'densenet161':
            target_layer_ = '84'
        elif model == 'densenet169':
            target_layer_ = '88'
        elif model == 'densenet201':
            target_layer_ = '104'
        else:
            print("not find model!")
    else:
        if model == 'vgg11':
            target_layer_ = '9'
        elif model == 'vgg13':
            target_layer_ = '13'
        elif model == 'vgg16':
            target_layer_ = '15'
        elif model == 'vgg19':
            target_layer_ = '17'
        elif model == 'resnet18':
            target_layer_ = '7'
        elif model == 'resnet34':
            target_layer_ = '10'
        elif model == 'resnet50':
            target_layer_ = '10'
        elif model == 'densenet121':
            target_layer_ = '64'
        elif model == 'cnn':
            target_layer_ = '4'
        else:
            raise NotImplementedError("selected model is not supported!")
    return target_layer_


def lrp_visualize(R_lrp, gamma=0.9):
    heatmaps_lrp = []
    for h, heat in enumerate(R_lrp):
        heat = heat.permute(1, 2, 0).detach().cpu().numpy()
        maps = render.heatmap(heat, reduce_axis=-1, gamma_=gamma)
        heatmaps_lrp.append(maps)
    return heatmaps_lrp

def grad_visualize(R_grad, image):
    img_h,img_w = image.shape[1], image.shape[2]
    R_grad = R_grad.squeeze(1).permute(1, 2, 0)
    R_grad = R_grad.cpu().detach().numpy()
    R_grad = cv2.resize(R_grad, (img_h, img_w))
    R_grad.reshape(img_h, img_w, image.shape[0])
    heatmaps_grad = []
    for i in range(image.shape[0]):
        heatmap = np.float32(cv2.applyColorMap(
            np.uint8((1 - R_grad[:, :, i]) * 255), cv2.COLORMAP_JET)) / 255
        cam = heatmap + np.float32(image[i])
        cam = cam / np.max(cam)
        heatmaps_grad.append(cam)
    return heatmaps_grad


import logging
from logging import handlers

class Logger:
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }
    def __init__(self,filename,
                 level='info',
                 when='D',
                 backCount=3,
                 fmt='%(asctime)s [%(levelname)s] %(message)s'
                 #fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                 ):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)#设置日志格式
        self.logger.setLevel(self.level_relations.get(level))#设置日志级别
        sh = logging.StreamHandler()#往屏幕上输出
        sh.setFormatter(format_str) #设置屏幕上显示的格式
        th = handlers.TimedRotatingFileHandler(filename=filename, when=when, backupCount=backCount, encoding='utf-8')
        #实例化TimedRotatingFileHandler
        #interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
        # S 秒
        # M 分
        # H 小时、
        # D 天、
        # W 每星期（interval==0时代表星期一）
        # midnight 每天凌晨
        th.setFormatter(format_str)#设置文件里写入的格式
        self.logger.addHandler(sh) #把对象加到logger里
        self.logger.addHandler(th)

    def info(self, msg):
        return self.logger.info(msg)


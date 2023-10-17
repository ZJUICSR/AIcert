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
from torchvision.transforms import transforms

def predict(net, x):
    activation_output = net.forward(x)
    _, prediction = torch.max(activation_output, 1)
    return prediction, activation_output


def apply_trigger(data, mask, trigger):
    # Only return poisoned samples

    X = copy.deepcopy(data)
    for i in range(32):
        for j in range(32):
            if mask[i, j] > 0.95:
                X[:, i, j] = X[:, i, j] * (1 - mask[i, j]) + trigger[:, i, j] * mask[i, j]

    return X

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
    return path.split("/")[-3] + "/" + path.split("/")[-2] + "/" + path.split("/")[-1]

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

'''获取数据集标准化时的均值和方差'''
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


'''加载样本数据'''
def get_loader(data, batchsize=8):
    """
    对于ImageNet数据集，batchsize默认为8,
    当使用lrp在vgg系列上时，batchsize可以设置为16
    当使用lrp解释Resnet50时，batchsize最大为8; resnet152时，batch为4.
    """
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
        imagenet_path = osp.join(root, "function/ex_methods/cache/imagenet_class_index.json")
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
def recreate_image(im_as_var, dataset="imagenet"):
    """
        Recreates images from a torch variable, sort of reverse preprocessing
    Args:
        im_as_var (torch variable): Image to recreate
    returns:
        recreated_im (numpy arr): Recreated image in array
    """
    reverse_mean = []
    reverse_std = []
    if dataset == "cifar10":
        reverse_mean = [-0.485, -0.456, -0.406]
        reverse_std = [1 / 0.229, 1 / 0.224, 1 / 0.225]
    elif dataset == "mnist":
        reverse_mean = [(-0.1307)]
        reverse_std = [(1 / 0.3081)]
    elif dataset == "imagenet":
        reverse_mean = [-0.485, -0.456, -0.406]
        reverse_std = [1 / 0.229, 1 / 0.224, 1 / 0.225]
    recreated_im = copy.copy(im_as_var.squeeze().data.cpu().detach().numpy())
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)
    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    return recreated_im


# 测试存储json文件
def save_process_result(root, results, filename):
    path = os.path.join(root, filename)

    with open(path, "w") as fp:
        json.dump(results, fp, indent=2)
        fp.close()


def preprocess_transform(image, dataset):
    if dataset == "imagenet":
        data_mean = [0.485, 0.456, 0.406]
        data_std = [0.229, 0.224, 0.225]
        img_size = 224
    elif dataset == "cifar10":
        data_mean = [0.43768206, 0.44376972, 0.47280434]
        data_std = [0.19803014, 0.20101564, 0.19703615]
        img_size = 32
    else:
        data_mean = 0.1307
        data_std = 0.3081
        img_size = 28

    normalize = torchvision.transforms.Normalize(mean=data_mean, std=data_std)
    transf = torchvision.transforms.Compose([
        torchvision.transforms.Resize(img_size),
        # torchvision.transforms.CenterCrop(img_size),
        torchvision.transforms.ToTensor(),
        normalize
    ])
    return transf(image)

'''将Image数据转成tensor，并做normalization处理，并加入device中'''
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
    return path.split("/")[-2] + "/" + path.split("/")[-1]

'''将tensor存储的样本逆转换成image格式的list'''
def loader2imagelist(dataloader, dataset, size):
    image_list = []
    for i in range(size):
        img, label = dataloader.dataset[i]
        if dataset in ["cifar10","imagenet","mnist" ]:
            img_x = torchvision.transforms.ToPILImage()(img)
            image_list.append(img_x)
        else:
            raise ValueError("not supported data set!")
    return image_list

def get_batchsize(model_name, dataset):
    model = model_name.lower()
    dataset = dataset.lower()
    if dataset == "imagenet":
        if model in ['vgg11','vgg13','vgg16','vgg19','resnet18','resnet34']:
            return 16 
        elif model in ['resnet50','densenet121']:
            return 8
        else: 
            return 4
    else:
        return 32

def target_layer(model_name, dataset):
    target_layer_ = None
    model = model_name.lower()
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
            target_layer_ = '22'
        elif model == 'densenet161':
            target_layer_ = '84'
        elif model == 'densenet169':
            target_layer_ = '88'
        elif model == 'densenet201':
            target_layer_ = '104'
        elif model == 'lenet':
            target_layer_ = '4'
        else:
            raise NotImplementedError("selected model is not supported!")
    return target_layer_

def get_conv_layer(model_name):
    cnn_layers = None
    model = model_name.lower()
    if model == 'vgg11':
        cnn_layers = ['1','4','7','9','12','14','17','19']
    elif model == 'vgg13':
        cnn_layers = ['1','3','6','8','11','13','16','18','21','23']
    elif model == 'vgg16':
        cnn_layers = ['1','3','6','8','11','13','15','18','20','22','25','27','29']
    elif model == 'vgg19':
        cnn_layers = ["1", "3", "6", "8", "11", "13", "15", "17", "20", "22", "24", "26", "29", "31", "33", "35"]
    elif model == 'resnet18':
        cnn_layers = ['4','5','6','7','8','9','10','11']
    elif model == 'resnet34':
        cnn_layers = [str(layer) for layer in range(4,19,2)]
    elif model == 'resnet50':
        cnn_layers = [str(layer) for layer in range(4,19,2)]
    elif model == 'resnet101':
        cnn_layers = [str(layer) for layer in range(4,36,3)]
    elif model == 'resnet152':
        cnn_layers = [str(layer) for layer in range(4,53,5)]
    elif model == 'densenet121':
        cnn_layers = [str(layer) for layer in range(4,64,6)]
    elif model == 'densenet161':
        cnn_layers = [str(layer) for layer in range(4,84,8)]
    elif model == 'densenet169':
        cnn_layers = [str(layer) for layer in range(4,88,8)]
    elif model == 'densenet201':
        cnn_layers = [str(layer) for layer in range(4,104,10)]
    else:
        raise NotImplementedError("selected model is not supported!")
    return cnn_layers

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
from logging.handlers import RotatingFileHandler
class Logger:
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }
    def __init__(self):
        self.__loggers = {}
    
    def add_logger(self, stid,
        filename,
        level='info',
        when='D',
        backCount=3,
        # fmt='%(asctime)s [%(levelname)s] %(filename)-12s %(funcName)-24s Line: %(lineno)-6s Msg: %(message)s'
        fmt='%(asctime)s [%(levelname)s]  Msg: %(message)s'):
        if stid not in self.__loggers.keys():
            logger = logging.getLogger(stid)
            format_str = logging.Formatter(fmt)#设置日志格式
            sh = logging.StreamHandler()#往屏幕上输出
            sh.setFormatter(format_str) #设置屏幕上显示的格式
            th = handlers.TimedRotatingFileHandler(filename=filename, when=when, backupCount=backCount, encoding='utf-8')
            th.setFormatter(format_str)#设置文件里写入的格式
            logger.setLevel(self.level_relations.get(level))#设置日志级别
            logger.addHandler(sh)
            logger.addHandler(th)
            self.__loggers.update({stid:logger})
        else:
            logger = self.__loggers[stid]
        return logger
    
    def get_sub_logger(self, stid):
        if stid not in self.__loggers.keys():
            return -1
        return self.__loggers[stid]
    
    def del_logger(self, stid):
        del self.__loggers[stid]
        return 1
        

    def info(self, stid, msg):
        return self.__loggers[stid].info(msg)
        # return self.logger.info(msg)




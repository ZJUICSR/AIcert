import ex_methods.module.render as render
import numpy as np
import os
import torchvision
from PIL import Image
import pandas as pd
import cv2
import torch


def preprocess_transform(image, dataset):
    if dataset == "Imagenet":
        data_mean = np.array([0.485, 0.456, 0.406])
        data_std = np.array([0.229, 0.224, 0.225])
        img_size = 224
    elif dataset == "cifar10":
        data_mean = np.array([0.5, 0.5, 0.5])
        data_std = np.array([0.5, 0.5, 0.5])
        img_size = 32
    else:
        data_mean = 0.1307
        data_std = 0.3081
        img_size = 28

    normalize = torchvision.transforms.Normalize(mean=data_mean, std=data_std)
    transf = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(img_size),
        torchvision.transforms.CenterCrop(img_size),
        normalize
    ])
    return transf(image)


def load_image(device, image, dataset):
    x = preprocess_transform(image, dataset)
    x = x.unsqueeze(0).to(device)
    return x


def norm_mse(R, R_):
    n = R.shape[0]
    R = R.reshape(n, -1)
    R_ = R_.reshape(n, -1)
    return ((R - R_) ** 2).mean(axis=1)


def target_quanti(interpreter, R_ori, R):
    rcorr_sign = []
    rcorr_rank = []

    if len(R.shape) == 4:
        R = R.sum(axis=1)
        R_ori = R_ori.sum(axis=1)

    rank_corr_sign = quanti_metric(R_ori, R, interpreter, keep_batch=True)
    mse = norm_mse(R, R_ori)

    return rank_corr_sign, mse


def quanti_metric(R_ori, R, interpreter, device, keep_batch=False):
    R_shape = R.shape
    l = R.shape[2]
    R_ori_f = torch.tensor(R_ori.reshape(R.shape[0], -1), dtype=torch.float32).to(device)
    R_f = torch.tensor(R.reshape(R.shape[0], -1), dtype=torch.float32).to(device)
    corr_list = []

    for i in range(R.shape[0]):
        corr = 0
        mask_ori = ((abs(R_ori_f[i]) - abs(R_ori_f[i]).mean()) / abs(R_ori_f[i]).std()) > 1
        R_ori_f_sign = torch.tensor(mask_ori, dtype=torch.float32).to(device) * torch.sign(R_ori_f[i])

        mask_ = ((abs(R_f[i]) - abs(R_f[i]).mean()) / abs(R_f[i]).std()) > 1
        R_f_sign = torch.tensor(mask_, dtype=torch.float32).to(device) * torch.sign(R_f[i])
        corr = np.corrcoef(R_ori_f_sign.detach().cpu().numpy(), R_f_sign.detach().cpu().numpy())[0][1]

        corr_list.append(corr)

    if keep_batch is True:
        return corr_list
    return np.mean(corr_list)


def get_accuracy(output, labels, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = labels.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            #                 res.append(correct_k.mul_(100.0 / batch_size))
            res.append(correct_k)
        return res


def target_layer(model="VGG19"):
    # Set target layer
    global target_layer_
    if model == 'VGG11':
        target_layer_ = '18'
    elif model == 'VGG13':
        target_layer_ = '22'
    elif model == 'VGG16':
        target_layer_ = '28'
    elif model == 'VGG19':
        target_layer_ = '34'
    elif model == 'Resnet18':
        target_layer_ = '11'
    elif model == 'Resnet34':
        target_layer_ = '19'
    elif model == 'Resnet50':
        target_layer_ = '19'
    elif model == 'Densenet121':
        target_layer_ = '64'
    elif model == 'mnist':
        target_layer_ = '6'
    elif model == 'cifar10':
        target_layer_ = '6'
    else:
        print("not find model!")
    return target_layer_


def lrp_visualize(R_lrp, gamma=0.9):
    heatmaps_lrp = []
    for h, heat in enumerate(R_lrp):
        heat = heat.permute(1, 2, 0).detach().cpu().numpy()
        maps = render.heatmap(heat, reduce_axis=-1, gamma_=gamma)
        heatmaps_lrp.append(maps)
    return heatmaps_lrp


def grad_visualize(R_grad, image):
    img_size = image.shape[1]
    R_grad = R_grad.squeeze(1).permute(1, 2, 0)
    R_grad = R_grad.cpu().detach().numpy()
    R_grad = cv2.resize(R_grad, (img_size, img_size))
    R_grad.reshape(img_size, img_size, image.shape[0])
    heatmap = np.float32(cv2.applyColorMap(np.uint8((1 - R_grad[:, :]) * 255), cv2.COLORMAP_JET)) / 255
    cam = heatmap + np.float32(image)
    cam = cam / np.max(cam)
    return cam


class logger(object):
    def __init__(self, file_name='mnist_result', resume=False, path="./results_ImageNet/", data_format='csv'):

        self.data_name = os.path.join(path, file_name)
        self.data_path = '{}.csv'.format(self.data_name)
        self.log = None
        if os.path.isfile(self.data_path):
            if resume:
                self.load(self.data_path)
            else:
                os.remove(self.data_path)
                self.log = pd.DataFrame()
        else:
            self.log = pd.DataFrame()

        self.data_format = data_format

    def add(self, **kwargs):
        """Add a new row to the dataframe
        example:
            resultsLog.add(epoch=epoch_num, train_loss=loss,
                           test_loss=test_loss)
        """
        df = pd.DataFrame([kwargs.values()], columns=kwargs.keys())
        self.log = self.log.append(df, ignore_index=True)

    def save(self):
        self.log.to_csv(self.data_path, index=False, index_label=False)

    def load(self, path=None):
        path = path or self.data_path
        if os.path.isfile(path):
            self.log.read_csv(path)
        else:
            raise ValueError('{} isn''t a file'.format(path))

from . import render
import numpy as np
import os
import json
import torchvision
from PIL import Image
import pandas as pd
import cv2
import copy
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from torchvision import models
import matplotlib.cm as mpl_color_map

# 测试存储json文件


def save_process_result(root, results, filename):
    path = os.path.join(root, filename)

    with open(path, "w") as fp:
        json.dump(results, fp, indent=2)
        fp.close()


def recreate_image(im_as_var, dataset):
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
        reverse_mean = [-0.43768206, -0.44376972, -0.47280434]
        reverse_std = [1 / 0.19803014, 1 / 0.20101564, 1 / 0.19703615]
    elif dataset == "mnist":
        reverse_mean = [(-0.1307)]
        reverse_std = [(1 / 0.3081)]
    elif dataset == "imagenet":
        reverse_mean = [-0.485, -0.456, -0.406]
        reverse_std = [1 / 0.2023, 1 / 0.1994, 1 / 0.2010]
    recreated_im = copy.copy(im_as_var.data.cpu().numpy())
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    return recreated_im


def save_ex_img(root, img, img_name):
    path = os.path.join(root, "keti1", img_name)
    img.save(path)
    return path


def get_loader(url):
    data = torch.load(url)
    # 不再是"x"和"y"
    adv_dst = TensorDataset(data[0].float().cpu(), data[1].long().cpu())
    adv_loader = DataLoader(
        adv_dst,
        batch_size=1,
        shuffle=False,
        num_workers=2
    )
    return adv_loader


def loader2imagelist(dataloader, dataset):
    image_list = []
    for i in range(30):
        img, _ = dataloader.dataset[i]
        if dataset == "cifar10":
            img_numpy = recreate_image(img, dataset)
            img_x = Image.fromarray(img_numpy)
            image_list.append(img_x)
        elif dataset == "imagenet":
            img_x = torchvision.transforms.ToPILImage()(img)
            image_list.append(img_x)
    return image_list


def convert_to_grayscale(im_as_arr):
    """
        Converts 3d image to grayscale
    Args:
        im_as_arr (numpy arr): RGB image with shape (D,W,H)
    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im


def save_class_activation_images(org_img, activation_map, file_name):
    """
        Saves cam activation map and activation map on the original image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        file_name (str): File name of the exported image
    """
    # Grayscale activation map
    heatmap, heatmap_on_image = apply_colormap_on_image(
        org_img, activation_map, 'hsv')
    # Save colored heatmap
    path_to_file = os.path.join('./results', file_name+'_Cam_Heatmap.png')
    save_image(heatmap, path_to_file)
    # Save heatmap on iamge
    path_to_file = os.path.join('../results', file_name+'_Cam_On_Image.png')
    save_image(heatmap_on_image, path_to_file)
    # SAve grayscale heatmap
    path_to_file = os.path.join('./results', file_name+'_Cam_Grayscale.png')
    save_image(activation_map, path_to_file)


def apply_colormap_on_image(org_im, activation, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

    # Apply heatmap on iamge
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(
        heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image


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


def preprocess_image(pil_im, resize_im=True):
    """
        Processes image for CNNs
    Args:
        PIL_img (PIL_img): PIL Image or numpy array to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (torch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # ensure or transform incoming image to PIL image
    if type(pil_im) != Image.Image:
        try:
            pil_im = Image.fromarray(pil_im)
        except Exception as e:
            print(
                "could not transform PIL_img to a PIL Image object. Please check input.")

    # Resize image
    if resize_im:
        pil_im = pil_im.resize((224, 224), Image.ANTIALIAS)

    im_as_arr = np.float32(pil_im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var


def get_positive_negative_saliency(gradient):
    """
        Generates positive and negative saliency maps based on the gradient
    Args:
        gradient (numpy arr): Gradient of the operation to visualize
    returns:
        pos_saliency ( )
    """
    pos_saliency = (np.maximum(0, gradient) / gradient.max())
    neg_saliency = (np.maximum(0, -gradient) / -gradient.min())
    return pos_saliency, neg_saliency


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
    R_ori_f = torch.tensor(R_ori.reshape(
        R.shape[0], -1), dtype=torch.float32).to(device)
    R_f = torch.tensor(
        R.reshape(R.shape[0], -1), dtype=torch.float32).to(device)
    corr_list = []

    for i in range(R.shape[0]):
        corr = 0
        mask_ori = (
            (abs(R_ori_f[i]) - abs(R_ori_f[i]).mean()) / abs(R_ori_f[i]).std()) > 1
        R_ori_f_sign = torch.tensor(mask_ori, dtype=torch.float32).to(
            device) * torch.sign(R_ori_f[i])

        mask_ = ((abs(R_f[i]) - abs(R_f[i]).mean()) / abs(R_f[i]).std()) > 1
        R_f_sign = torch.tensor(mask_, dtype=torch.float32).to(
            device) * torch.sign(R_f[i])
        corr = np.corrcoef(R_ori_f_sign.detach().cpu().numpy(),
                           R_f_sign.detach().cpu().numpy())[0][1]

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
            correct_k = correct[:k].contiguous(
            ).view(-1).float().sum(0, keepdim=True)
            #                 res.append(correct_k.mul_(100.0 / batch_size))
            res.append(correct_k)
        return res


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
        elif model == 'densenet121':
            target_layer_ = '64'
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
    heatmap = np.float32(cv2.applyColorMap(
        np.uint8((1 - R_grad[:, :]) * 255), cv2.COLORMAP_JET)) / 255
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

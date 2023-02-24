# -*- coding: utf-8 -*-
import sys
import os 
sys.path.append(os.getcwd(),"function/formal_verify")
# sys.path.append(".")
import torchvision
from auto_LiRPA.auto_LiRPA.perturbations import *
from torch.utils.data import TensorDataset, DataLoader
from auto_LiRPA_verifiy.vision.verify import verify
from auto_LiRPA_verifiy.vision.cnn import mnist_model
from auto_LiRPA_verifiy.vision.data import get_mnist_data, get_cifar_data, get_gtsrb_data, get_MTFL_data
from auto_LiRPA_verifiy.vision.model import get_mnist_cnn_model, get_cifar_resnet18, get_gtsrb_resnet18, get_MTFL_resnet18


if __name__ == '__main__':
    mn_model = get_mnist_cnn_model(activate=True)
    print(mn_model)
    ver_data, n_class = get_mnist_data()
    input_param = {'input_param': {'model': mn_model,
                                   'dataset': ver_data,
                                   'n_class': n_class,
                                   'up_eps': 0.1,
                                   'down_eps': 0.01,
                                   'steps': 5,
                                   'device': 'cpu',
                                   'output_path': 'output'}}

    result = verify(input_param)
    print(result)


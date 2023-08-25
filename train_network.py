from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import torch
from function.attack.attacks.utils import load_mnist, load_cifar10
from model.model_net.lenet import Lenet
from model.model_net.resnet_attack import *
import torch.optim as optim
from function.attack.estimators.classification import PyTorchClassifier
# from models.model_net.resnet import ResNet18
from model.model_net.mnist import Mnist
from function.attack.attacks.utils import compute_success, compute_accuracy

# Resnet-Mnist
def train_resnet_mnist(modelname, modelpath='', logging = None, device = None):
    if device:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if logging:
        logging.info(f"[模型训练]:加载与训练模型{modelname}")
    model = eval(modelname)(1).to(device)
    if logging:
        logging.info(f"[模型训练]:加载数据集MNIST")
    input_shape, (_, _), (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist("/root/fairness/AI-Platform/dataset/MNIST", 1)
    batch_size = 256
    num_epochs = 5
    num_batch = int(np.ceil(len(x_train) / float(batch_size)))
    total_step = len(x_train)
    optimizer = optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        print("epoch:", epoch)
        train_loss = 0
        for m in range(num_batch):
            # Batch indexes
            begin, end = (
                m * batch_size,
                min((m + 1) * batch_size, x_train.shape[0]),
            )
            # 前向传播
            model_outputs = model(torch.from_numpy(x_train[begin:end]).to(device))
            # 损失计算
            loss = criterion(model_outputs, torch.from_numpy(y_train[begin:end]).to(device))
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        
        classifier = PyTorchClassifier (
            model=model,
            clip_values=(min_pixel_value, max_pixel_value),
            loss=criterion,
            optimizer=optimizer,
            input_shape=input_shape,
            nb_classes=10,
            device = device
        )
        print("[模型训练]:Epoch [{}/{}], train_loss: {:.4f}".format(epoch+1, num_epochs, train_loss / total_step))
        if logging:
            logging.info("[模型训练]:Epoch [{}/{}], train_loss: {:.4f}".format(epoch+1, num_epochs, train_loss / total_step))
    if modelpath == '':
        modelpath = "./model/ckpt/MNIST_{}.pth".format(modelname.lower())
    torch.save(model.state_dict(), modelpath)

# Resnet-cifar10
def train_resnet_cifar10(modelname, modelpath='', logging=None, device=None):
    if device:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if logging:
        logging.info(f"[模型训练]:加载与训练模型{modelname}") 
    model = eval(modelname)(3).to(device)
    if logging:
        logging.info(f"[模型训练]:加载数据集CIFAR10") 
    input_shape, (_, _), (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_cifar10("/root/fairness/AI-Platform/dataset/CIFAR10", 1)

    batch_size = 128
    num_epochs = 20
    num_batch = int(np.ceil(len(x_train) / float(batch_size)))
    total_step = len(x_train)
    optimizer = optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        print("epoch:", epoch)
        train_loss = 0
        for m in range(num_batch):
            # Batch indexes
            begin, end = (
                m * batch_size,
                min((m + 1) * batch_size, x_train.shape[0]),
            )
            # 前向传播
            model_outputs = model(torch.from_numpy(x_train[begin:end]).to(device))
            # 损失计算
            loss = criterion(model_outputs, torch.from_numpy(y_train[begin:end]).to(device))
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        
        classifier = PyTorchClassifier (
            model=model,
            clip_values=(min_pixel_value, max_pixel_value),
            loss=criterion,
            optimizer=optimizer,
            input_shape=input_shape,
            nb_classes=10,
            device = device
        )
        print("[模型训练]:Epoch [{}/{}], train_loss: {:.4f}".format(epoch+1, num_epochs, train_loss / total_step))
        if logging:
            logging.info("[模型训练]:Epoch [{}/{}], train_loss: {:.4f}".format(epoch+1, num_epochs, train_loss / total_step))
    if modelpath == '':
        modelpath = "./model/ckpt/CIFAR10_{}.pth".format(modelname.lower())
    torch.save(model.state_dict(), modelpath)

if __name__ == "__main__":
    # train_resnet_mnist()
    train_resnet_cifar10(modelname='ResNet101')
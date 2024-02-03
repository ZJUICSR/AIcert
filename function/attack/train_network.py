from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import torch
from function.attack.attacks.utils import load_mnist, load_cifar10
# from models.model_net.lenet import Lenet
# from models.model_net.resnet import ResNet18
import torch.optim as optim
from function.attack.estimators.classification import PyTorchClassifier
# from models.model_net.resnet import ResNet18
# from models.model_net.mnist import Mnist
from function.attack.attacks.utils import compute_success, compute_accuracy

# Resnet-Mnist
def train_resnet_mnist():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNet18(1).to(device)
    input_shape, (_, _), (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist("./dataset/", 1)
    batch_size = 256
    num_epochs = 5
    num_batch = int(np.ceil(len(x_train) / float(batch_size)))
    total_step = len(x_train)
    optimizer = optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
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
        )
        print("Epoch [{}/{}], train_loss: {:.4f}".format(epoch+1, num_epochs, train_loss / total_step))
        path = "./model/model_ckpt/ckpt-resnet18-mnist_epoch{}_acc{:.4f}.pth".format(epoch+1, compute_accuracy(classifier.predict(x_test, training_mode=True), y_test))
        torch.save(model.state_dict(), path)

# Resnet-cifar10
def train_resnet_cifar10():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNet18().to(device)
    input_shape, (_, _), (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_cifar10("./dataset/", 1)
    batch_size = 128
    num_epochs = 20
    num_batch = int(np.ceil(len(x_train) / float(batch_size)))
    total_step = len(x_train)
    optimizer = optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
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
        )
        print("Epoch [{}/{}], train_loss: {:.4f}".format(epoch+1, num_epochs, train_loss / total_step))
        path = "./model/model_ckpt/ckpt-resnet18-cifar10_epoch{}_acc{:.4f}.pth".format(epoch+1, compute_accuracy(classifier.predict(x_test, training_mode=True), y_test))
        torch.save(model.state_dict(), path)

if __name__ == "__main__":
    # train_resnet_mnist()
    train_resnet_cifar10()
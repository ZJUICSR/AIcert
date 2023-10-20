from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import torch, copy, random
from function.attack.attacks.utils import load_mnist, load_cifar10
from model.model_net.lenet import Lenet
from model.model_net.resnet_attack import *
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


def robust_train(model, train_loader, test_loader, adv_loader, attack, device, epochs=40, method=None, adv_param=None, rate=0.25, **kwargs):
    """
    用于系统内定的对抗算法做鲁棒训练
    :param model:
    :param train_loader:
    :param test_loader:
    :param adv_loader:
    :param epochs:
    :param atk:
    :param epoch_fn:
    :param rate:
    :param kwargs:
    :return:
    """
    # 训练超参数
    lr = 0.1  # 学习率
    momentum = 0.9  # 动量参数，用于优化算法

    assert "atk_method" in kwargs.keys()
    assert "def_method" in kwargs.keys()
    train_res = {}

    import torchattacks as attacks
    copy_model1 = copy.deepcopy(model)
    
    model = model.to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=2e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[100, 150], last_epoch=-1)
    best_acc = 0.0
    best_loss = 0.0
    best_epoch = 0
    train_list = []
    test_list = []
    criterion = torch.nn.CrossEntropyLoss()
    print(f"-> 开始鲁棒训练，服务运行在显卡:{device}")
    _eps = copy.deepcopy(adv_param['eps'])
    from IOtool import IOtool
    for epoch in range(1, epochs + 1):
        print("-> Method {:s} for epoch:{:d} adv training on device: {:s}".format(method, epoch, str(device)))
        model.train()
        model = model.to(device)
        num_step = len(train_loader)
        total, sum_correct, sum_loss = 0, 0, 0.0
        for step, (x, y) in enumerate(train_loader):
            if method is not None:
                size = int(rate * len(x))
                idx = np.random.choice(len(x), size=size, replace=False)
                adv_param['eps'] = _eps * (random.randint(80, 180) * 0.01)
                x = x.detach().to('cpu').numpy()
                y = y.detach().to('cpu').numpy()
                # x, y = x.to(device), y.to(device)
                x[idx] = attack(copy.deepcopy(x[idx]), copy.deepcopy(y[idx]))
            x, y = torch.tensor(x).to(device), torch.tensor(y).to(device)
            # 向模型中输入数据
            out = model.forward(x)
            # 计算损失值
            loss = criterion(out, y)
            # 清理当前优化器中梯度信息
            optimizer.zero_grad()
            # 根据损失值计算梯度
            loss.backward()
            # 根据梯度信息进行模型优化
            optimizer.step()
            # 统计损失信息
            sum_loss += loss.item()
            
            total += y.size(0)
   
            _, pred = out.max(1)
            # sum_correct = (pred == y).sum().item()
            sum_correct += pred.eq(y.view_as(pred)).sum().item()
            info = "[Train] Epoch:{:d}/{:d} Attack:{:s}_{:.4f} Defense:{:s} Loss: {:.6f} Acc:{:.3f}%".format(
                epoch,
                epochs + 1,
                kwargs["atk_method"],
                adv_param['eps'],
                kwargs["def_method"],
                sum_loss / total,
                100.0 * (sum_correct / total)
            )
            IOtool.progress_bar(step, num_step, info)
            
        adv_param['eps'] = _eps
        train_acc, train_loss = 100.0 * (sum_correct / total), sum_loss / total
        test_acc = eval_test(model, test_loader, device)
        adv_test_acc = eval_test(model, adv_loader, device)

        if best_acc < test_acc:
            best_acc = test_acc
            best_epoch = epoch

        train_list.append(train_acc)
        test_list.append(test_acc)
        epoch_result = {
            "epoch": epoch,
            "best_acc": best_acc,
            "best_loss": best_loss,
            "best_epoch": best_epoch,
            "train": train_acc,
            "test": test_acc,
            "train_list": train_list,
            "test_list": test_list
        }
        # lr_scheduler.step()
    print(epoch_result)
    model = model.cpu()
    return model

def test_batch(model, test_loader, device=None, **kwargs):
    # print(device)
    model = model.to(device)
    with torch.no_grad():
        x, y = iter(test_loader).next()
        x = x.to(device)
        y = y.to(device)
        output = model(x).detach()
        loss = F.cross_entropy(output, y).detach().cpu()
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(y.view_as(pred)).sum().item()
        acc = 100.0 * float(correct / len(x))
        output = output.cpu()
        return output, round(float(acc), 3), round(float(loss), 5)

def eval_test(model, test_loader, device, criterion=None):
    # 进行模型评估
    eval_loss = 0
    eval_acc = 0
    # copy_model = copy.deepcopy(model).eval().to(device)
    copy_model = model.eval().to(device)
    if criterion== None:
        criterion = torch.nn.CrossEntropyLoss()
    for img, label in test_loader:
        img = img.to(device)
        label = label.to(device)

        out = copy_model.forward(img)
        loss = criterion(out, label)

        # 记录误差
        eval_loss += loss.item()

        # 记录准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]
        eval_acc += acc

    return  round(float(100*(eval_acc/ len(test_loader))), 3)
if __name__ == "__main__":
    # train_resnet_mnist()
    train_resnet_cifar10(modelname='ResNet101')
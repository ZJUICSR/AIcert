import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, mnist
from torch.utils.data import DataLoader
import torch.optim as optim
import tqdm, copy, random
import numpy as np
import torch.nn.functional as F

def adjust_learning_rate(optimizer, epoch, lr_):
    """decrease the learning rate"""
    lr = lr_
    if epoch >= 40:
        lr = lr * 0.1
    if epoch >= 60:
        lr = lr * 0.01

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def Train(model_name, model, dataset, device):
    # 训练超参数
    train_batchsize = 128  # 训练批大小
    test_batchsize = 128  # 测试批大小
    num_epoches = 0  # 训练轮次
    lr = 0.1  # 学习率
    momentum = 0.9  # 动量参数，用于优化算法

    #获取训练数据
    if dataset == "mnist":
        transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), transforms.Normalize([0.1307], [0.3081])])
        train_dataset = mnist.MNIST('./dataset/data/MNIST', train=True, transform=transform, download=True)
        test_dataset = mnist.MNIST('./dataset/data/MNIST', train=False, transform=transform, download=False)
        num_epoches = 20
    elif dataset == "cifar10":
        transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        train_dataset = CIFAR10('./dataset/data/CIFAR10', train=True, transform=transform, download=True)
        test_dataset = CIFAR10('./dataset/data/CIFAR10', train=False, transform=transform, download=False)
        num_epoches = 80

    #datalodar用于加载训练数据
    train_loader = DataLoader(train_dataset, batch_size=train_batchsize, shuffle=True,num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=test_batchsize,shuffle=False)

    model = model.to(device)
    # model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    # cudnn.benchmark = True
    # print(model)

    # 损失函数
    criterion = torch.nn.CrossEntropyLoss()

    # 优化器
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=2e-4)
    #optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[100, 150], last_epoch=-1)


    model.train()
    for epoch in range(num_epoches):
        # 统计损失值和精确度
        train_loss = 0
        train_acc = 0
        train_iter = tqdm.tqdm(enumerate(train_loader),
                           desc=f"{dataset}_{model_name} Epoch:{epoch}",
                           total=len(train_loader),
                           bar_format="{l_bar}{r_bar}")
        # 读取数据
        for i, (img, label) in train_iter:
            # 将数据放入设备中
            img = img.to(device)
            label = label.to(device)

            # 向模型中输入数据
            out = model.forward(img)
            # 计算损失值
            loss = criterion(out, label)
            # 清理当前优化器中梯度信息
            optimizer.zero_grad()
            # 根据损失值计算梯度
            loss.backward()
            # 根据梯度信息进行模型优化
            optimizer.step()

            # 统计损失信息
            train_loss += loss.item()

            # 得到预测值
            _, pred = out.max(1)

            # 判断预测正确个数，计算精度
            num_correct = (pred == label).sum().item()
            acc = num_correct / img.shape[0]
            train_acc += acc
    
        # 打印学习情况
        print('epoch: {}, Train Loss: {:.4f}, Train Acc: {:.4f}'.format(epoch, train_loss / len(train_loader), train_acc / len(train_loader)))
        
        lr_scheduler.step()

    test_acc = eval_test(model, test_loader, device, criterion)

    return test_acc, model
   
def robust_train(model, train_loader, test_loader, adv_loader, device, epochs=40, method=None, adv_param=None, rate=0.25, **kwargs):
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
        print("-> Method{:s} for epoch:{:d} adv training on device: {:s}".format(method, epoch, str(device)))
        model.train()
        model = model.to(device)
        num_step = len(train_loader)
        total, sum_correct, sum_loss = 0, 0, 0.0
        for step, (x, y) in enumerate(train_loader):
            if method is not None:
                size = int(rate * len(x))
                idx = np.random.choice(len(x), size=size, replace=False)
                adv_param['eps'] = _eps * (random.randint(80, 180) * 0.01)
                x, y = x.to(device), y.to(device)
                attackObj = eval("attacks.{:s}".format(method))(copy_model1.to(device), **adv_param)
                x[idx] = attackObj(copy.deepcopy(x[idx]), copy.deepcopy(y[idx]))

            x, y = x.to(device), y.to(device)
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
        model.to(device)
        with torch.no_grad():
            x, y = iter(test_loader).next()
            x = x.to(device)
            output = model(x).detach().cpu()
            loss = F.cross_entropy(output, y).detach().cpu()
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(y.view_as(pred)).sum().item()
            acc = 100.0 * float(correct / len(x))
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

    return 100*(eval_acc/ len(test_loader))
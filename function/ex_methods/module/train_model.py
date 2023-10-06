import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, mnist
from torch.utils.data import DataLoader
import torch.optim as optim
import tqdm

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
                                                        milestones=[50, 70], last_epoch=-1)


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
    

def eval_test(model, test_loader, device, criterion):
    # 进行模型评估
    eval_loss = 0
    eval_acc = 0
    model.eval()

    for img, label in test_loader:
        img = img.to(device)
        label = label.to(device)

        out = model.forward(img)
        loss = criterion(out, label)

        # 记录误差
        eval_loss += loss.item()

        # 记录准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]
        eval_acc += acc

    return eval_acc/ len(test_loader)

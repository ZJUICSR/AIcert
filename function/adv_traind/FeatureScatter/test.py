import AttackModel
import json
from AttackModel import RobustTest
from models.resnet import ResNet
from LoadModel import RobustEnhance

args_dict1 = {
        'model_dir': '/data/user/WZT/models/feasca_cifar10_letnet/',  # 模型训练输出路径及加载路径
        'init_model_pass': 'checkpoint-0',  # 默认'-1'， 加,路径为model_dir。(-1: from scratch; K: checkpoint-K; latest = latest)
        'resume': True, # 加载init_model_pass 继续训练
        'lr': 0.1,  # 学习率
        'batch_size_train': 128,  # 每次train的样本数
        'max_epoch': 200, # 训练批次大小
        'decay_epoch1': 60,
        'decay_epoch2': 90,
        'decay_rate': 0.1,
        'dataset': 'cifar10',  # 数据集选择（cifar10，cifar100，svhn）
        'num_classes': 10,  # 图片种类（cifar10 = 10，cifar100 =100，svhn =10）
        'image_size': 32, # 数据集样本规格大小
        'model1': LeNet(10)  # 输入要进行鲁棒增强的网路架构，ResNet(50, 10), LeNet(10), VGG(16, 10), WideResNet(depth=28, num_classes=10, widen_factor=10)
    }   
args_dict2 = {
        'attack': True,
        'model2': LeNet(10),  # 输入要进行鲁棒增强的网路架构，ResNet(50, 10), LeNet(10), VGG(16, 10), WideResNet(depth=28, num_classes=10, widen_factor=10)
        'model_dir': '/data/user/WZT/models/feasca_cifar10_letnet/',  # 模型路径
        'init_model_pass': 'latest',  # 加载文件名latest,路径为model_dir，
        'attack_method': 'pgd',  # adv_mode : natural, pdg or cw
        'attack_method_list': 'natural-fgsm-pgd-cw',
        'dataset': 'cifar10',  # 数据集cifar10, cifar100,svhn
        'image_size': 32,  # cifar10, cifar100,svhn, = 32; minist = 28
        'num_classes': 10, # 图片种类
        'batch_size_test': 100  # 每次测试的样本数
    }
# 鲁棒性训练接口，输出鲁棒增强后的模型和增强后的模型分类准确率
RobustEnhance(args_dict1)
# 鲁棒性检验接口，
# 3、输入鲁棒增强后的模型，输出attack之后的分类准确率
RobustTest(args_dict2)






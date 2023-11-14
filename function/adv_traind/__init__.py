import json,torch, os, shutil
from .FeatureScatter.models.resnet import ResNet
from .FeatureScatter.models.lenet import LeNet
from .FeatureScatter.models.wideresnet import WideResNet
from .FeatureScatter.models.vggnet import VGG
from .FeatureScatter.LoadModel import RobustEnhance
from .FeatureScatter.AttackModel import RobustTest

def run_featurescatter(dataset, modelname, lr, batch_size, max_epoch, decay_epoch, decay_rate, weight_decay, out_path, logging=None):
    # 数据集和模型加载
    logging.info('Dataset preparing......')
    if dataset == 'cifar100':
        n_class = 100
    else:
        n_class = 10
    image_size = 32
    if modelname == 'resnet':
        model = ResNet(50, n_class)
    elif modelname == 'lenet':
        model = LeNet(n_class)
    elif modelname == 'wideresnet':
        model = WideResNet(depth=28, num_classes=n_class, widen_factor=10)
    elif modelname == 'vgg':
        model = VGG(16, n_class)
    logging.info('Dataset prepared......')
    
    # 判断是会否有预存模型
    if os.path.exists(os.path.join('./output/cache/FeaSca/',str(dataset)+'_'+str(modelname),'normal','latest')):
        normal_resume = True
        normal_init_model_pass = 'latest'
        normal_path = os.path.join('./output/cache/FeaSca/',str(dataset)+'_'+str(modelname),'normal')
    else:
        normal_resume = False
        normal_init_model_pass = '-1'
        normal_path = os.path.join('./output/cache/FeaSca/',str(dataset)+'_'+str(modelname),'normal')
        if os.path.exists(normal_path):
            shutil.rmtree(normal_path) 
        os.makedirs(normal_path)
    
    if os.path.exists(os.path.join('./output/cache/FeaSca/',str(dataset)+'_'+str(modelname),'enhance','latest')):
        enhance_resume = True
        enhance_init_model_pass = 'latest'
        enhance_path = os.path.join('./output/cache/FeaSca/',str(dataset)+'_'+str(modelname),'enhance')
    else:
        enhance_resume = False
        enhance_init_model_pass = '-1'
        enhance_path = os.path.join('./output/cache/FeaSca/',str(dataset)+'_'+str(modelname),'enhance')
        if os.path.exists(enhance_path):
            shutil.rmtree(enhance_path) 
        os.makedirs(enhance_path)
    
    # Normal模型训练
    logging.info('Model normal training......')
    args_dict_normal = {
        'model_dir': normal_path,  # 模型训练输出路径及加载路径
        'adv_mode': 'source', #  训练初始模型source 训练鲁棒性增强模型用 feature_scatter,
        'init_model_pass': normal_init_model_pass,  # 默认'-1'， 加,路径为model_dir。(-1: from scratch; K: checkpoint-K; latest = latest)
        'resume': normal_resume, # 加载init_model_pass 继续训练
        'lr': lr,  # 学习率
        'batch_size_train': batch_size,  # 每次train的样本数
        'max_epoch': max_epoch, # 训练批次大小 默认200
        'decay_epoch1': decay_epoch,
        'decay_epoch2': decay_epoch,
        'decay_rate': decay_rate,
        'dataset': dataset,  # 数据集选择（cifar10，cifar100，svhn）
        'num_classes': n_class,  # 图片种类（cifar10 = 10，cifar100 =100，svhn =10）
        'image_size': image_size, # 数据集样本规格大小
        'model1': model  # 输入要进行鲁棒增强的网路架构，ResNet(50, 10), LeNet(10), VGG(16, 10), WideResNet(depth=28, num_classes=10, widen_factor=10)
    }
    normal_model, _ = RobustEnhance(args_dict_normal)
    shutil.copytree(normal_path, os.path.join(out_path, 'normal'))
    
    # Normal模型测试
    logging.info('Model robustness testing......')
    args_dict = {
        'attack': True,
        'model2': model, # 输入要进行鲁棒增强的网路架构，ResNet(50, 10), LeNet(10), VGG(16, 10), WideResNet(depth=28, num_classes=10, widen_factor=10)
        'model_dir': normal_path,  # 模型路径
        'init_model_pass': 'latest',  # 加载文件名latest,路径为model_dir，
        'attack_method_list': 'Natural-PGD-CW-FGSM',# 无任何攻击下的acc和三种攻击下的acc
        'dataset': dataset,  # 数据集cifar10, cifar100,svhn
        'image_size': image_size,  # cifar10, cifar100,svhn, = 32; minist = 28
        'num_classes': n_class,  # 图片种类
        'batch_size_test': 100  # 每次测试的样本数
    }
    ori_acc = RobustTest(args_dict)
    logging.info('Model robustness already tested......')
    

    # Enhance模型训练
    logging.info('Model enhance training......')
    args_dict_enhance = {
        'model_dir': enhance_path,  # 模型训练输出路径及加载路径
        'adv_mode': 'feature_scatter', #  训练初始模型source 训练鲁棒性增强模型用 feature_scatter,
        'init_model_pass': enhance_init_model_pass,  # 默认'-1'， 加,路径为model_dir。(-1: from scratch; K: checkpoint-K; latest = latest)
        'resume': enhance_resume, # 加载init_model_pass 继续训练
        'lr': lr,  # 学习率
        'batch_size_train': batch_size,  # 每次train的样本数
        'max_epoch': max_epoch, # 训练批次大小 默认200
        'decay_epoch1': decay_epoch,
        'decay_epoch2': decay_epoch,
        'decay_rate': decay_rate,
        'dataset': dataset,  # 数据集选择（cifar10，cifar100，svhn）
        'num_classes': n_class,  # 图片种类（cifar10 = 10，cifar100 =100，svhn =10）
        'image_size': image_size, # 数据集样本规格大小
        'model1': model  # 输入要进行鲁棒增强的网路架构，ResNet(50, 10), LeNet(10), VGG(16, 10), WideResNet(depth=28, num_classes=10, widen_factor=10)
    }
    enhance_model, _ = RobustEnhance(args_dict_enhance)
    shutil.copytree(enhance_path, os.path.join(out_path, 'enhance'))
    logging.info('Model already enhanced......')
    
    # Enhance模型测试
    logging.info('Model robustness testing......')
    args_dict2 = {
        'attack': True,
        'model2': model, # 输入要进行鲁棒增强的网路架构，ResNet(50, 10), LeNet(10), VGG(16, 10), WideResNet(depth=28, num_classes=10, widen_factor=10)
        'model_dir': enhance_path,  # 模型路径
        'init_model_pass': 'latest',  # 加载文件名latest,路径为model_dir，
        'attack_method_list': 'Natural-PGD-CW-FGSM',# 无任何攻击下的acc和三种攻击下的acc  natural-PGD-CW-fgsm
        'dataset': dataset,  # 数据集cifar10, cifar100,svhn
        'image_size': 32,  # cifar10, cifar100,svhn, = 32; minist = 28
        'num_classes': n_class,  # 图片种类
        'batch_size_test': 100  # 每次测试的样本数
    }
    enh_acc = RobustTest(args_dict2)
    logging.info('Enhanced model robustness already tested......')
    
    return {
        "ori_acc": ori_acc,
        "enh_acc": enh_acc 
    }
        
from .SEAT.seat import seat

def run_seat(modelname, lr, n_class, max_epoch, out_path, logging=None):
    logging.info('Start SEAT running......')
    args_dict = {'epochs': max_epoch,
             'arch': modelname,
             'num_classes': n_class,
             'lr': lr,
             'loss_fn': 'cent',
             'log_step': 7,
             'epsilon': 0.031,
             'num_steps': 10,
             'step_size': 0.007,
             'resume': False,
             'out_dir': out_path,
             'ablation': ''}
    nat_acc, pgd20_acc, ema_nat_acc, ema_pgd20_acc = seat(args_dict)
    logging.info('Finish SEAT Enhancing......')
    return {
        "nat_acc": nat_acc,
        "pgd20_acc": pgd20_acc,
        "ema_nat_acc": ema_nat_acc,
        "ema_pgd20_acc": ema_pgd20_acc
    }
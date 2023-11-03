import AttackModel1
import json
from AttackModel1 import RobustTest
from models.resnet import ResNet
from LoadModel1 import RobustEnhance

# LeNet(10),Wide_ResNet(28, 10, 0.3, 10),VGG(16, 10)
model = ResNet(50, 10)
args_dict1 = {
    'dataset': 'cifar10',
    'init_model_pass': -1, # -1
    'model_dir': "./result",
    'resume': False,
    'lr': 0.1,
    'adv_mode': 'feature_scatter',
    'max_epoch': 5, #200
    'save_epochs': 100,
    'decay_epoch1': 60,
    'decay_epoch2': 90,
    'decay_rate': 0.1,
    'batch_size_train': 128,
    'momentum': 0.9,
    'weight_decay': 2e-4,
    'log_step': 7,
    'num_classes': 10,
    'image_size': 32,
}
args_dict2 = {
    'attack': True,
    'attack_method': 'pgd',
    # 'attack_method_list': "pgd",
    # 'batch_size_test': 100,
    'dataset': 'cifar10',
    'image_size': 32,
    'init_model_pass': 'latest',
    'log_step': 7,
    'model_dir': "/data/user/WZT/models/feature_scatter_cifar10/resnet",
    'num_classes': 10,
    'resume': False
}


# 鲁棒性增强接口，输出鲁棒增强后的模型和增强后的模型分类准确率
model2, acc = RobustEnhance(args_dict1)

# 鲁棒性检验接口，
# 3、输入鲁棒增强后的模型，输出attack之后的分类准确率
attack_acc = RobustTest(model, args_dict2)

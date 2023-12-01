import AttackModel
from AttackModel import RobustTest
from models.resnet import ResNet
from LoadModel import RobustEnhance
import os
from normal_train import train_normal_model

# LeNet(10),Wide_ResNet(28, 10, 0.3, 10),VGG(16, 10)
model = ResNet(50, 10)
args_dict1 = {
    'dataset': 'cifar10',
    'init_model_pass': 'latest',
    'model_dir': "checkpoint/",
    'resume': True,
    'lr': 0.1,
    'adv_mode': 'feature_scatter',
    'attack_method_list': 'natural-FGSM-PGD-CW',
    'max_epoch': 1,
    'save_epochs': 20,
    'decay_epoch1': 60,
    'decay_epoch2': 90,
    'decay_rate': 0.1,
    'batch_size_train': 128,
    'momentum': 0.9,
    'weight_decay': 2e-4,
    'log_step': 50,
    'num_classes': 10,
    'image_size': 32,
}
args_dict2 = {
    'attack': True,
    'attack_method': 'pgd',
    'attack_method_list': 'natural-FGSM-PGD-CW',
    'batch_size_test': 128,
    'dataset': 'cifar10',
    'image_size': 32,
    'init_model_pass': 'latest',
    'log_step': 50,
    'model_dir': "checkpoint/",
    'num_classes': 10,
    'resume': True
}
results = dict()
results['normal_train'] = train_normal_model(model=model, args_dict=args_dict1)

# 鲁棒性增强接口，输出鲁棒增强后的模型和增强后的模型分类准确率
print(f'start feature scatter train')
RobustEnhance(model, args_dict1)

# 鲁棒性检验接口，
# 3、输入鲁棒增强后的模型，输出attack之后的分类准确率
print(f'start test')
results['feature_scatter'] = RobustTest(model, args_dict2)

results_file = os.path.join(os.path.dirname(__file__), 'results')
if not os.path.exists(results_file):
    os.mkdir(results_file)

import json
def write_json(info, file_name=''):
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=4, ensure_ascii=False)
    return

write_json(results, os.path.join(results_file, 'result.json'))

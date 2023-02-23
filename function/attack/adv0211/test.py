from attack_api import EvasionAttacker, BackdoorAttacker
import torch
from art.resnet import ResNet34
from art.mnist import Mnist

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# cifar10 逃逸攻击测试
a = EvasionAttacker(modelnet=ResNet34, modelpath="./models/ckpt-cifar10-32-resnet18_0.9513.pth", dataset="cifar10", device=device, datanormalize=True)
# 参数直接添加到perturb()函数中
a.perturb(method="ProjectedGradientDescent", sample_num=1024, batch_size=256, eps_step=0.03, eps=0.3)
a.print_res()

# # mnist 逃逸攻击测试
# a = EvasionAttacker(modelnet=Mnist, modelpath="./models/ckpt-mnist-15-mnist_0.9905.pth", dataset="mnist", datanormalize=False, device=device)
# a.perturb(method="SquareAttack", sample_num=20, batch_size=1024)
# a.print_res()

# # mnist 后门攻击测试
# # 初始化硬件,数据集,模型等，选定方法
# b = BackdoorAttacker(modelnet=Mnist, datanormalize=False)
# # 设置batchsize,投毒比例以及目标
# b.backdoorattack(method="PoisoningAttackAdversarialEmbedding",batch_size=1024, pp_poison=0.5, target=1, test_sample_num=1024)

# # cifar10 后门攻击测试
# # 初始化硬件,数据集,模型等，选定方法
# b = BackdoorAttacker(modelnet=ResNet18, modelpath="models/ckpt-cifar10-32-resnet18_0.9513.pth", dataset="cifar10", datanormalize=True)
# # 设置batchsize,投毒比例以及目标
# b.backdoorattack(method="PoisoningAttackBackdoor", batch_size=128, pp_poison=0.01, target=1, test_sample_num=1024)
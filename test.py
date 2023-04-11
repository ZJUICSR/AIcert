# 测试用例
from attack_api import EvasionAttacker, BackdoorAttacker
import torch
from models.model_net.lenet import Lenet
from models.model_net.resnet import ResNet18
from attacks.utils import load_mnist, load_cifar10

# 对抗攻击测试
## MNIST
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
a = EvasionAttacker(modelnet=ResNet18(1), modelpath="./models/model_ckpt/ckpt-resnet18-mnist_epoch3_acc0.9898.pth", 
    dataset="mnist", datasetpath="./datasets/", nb_classes=10, datanormalize = False, 
    device=device, sample_num=128)
### white box attack
# a.generate(method="FastGradientMethod", norm="inf", save_num=32)
# a.print_res()
# a.generate(method="ProjectedGradientDescent", norm="inf", save_num=32)
# a.print_res()
# a.generate(method="BasicIterativeMethod", norm="inf", save_num=32)
# a.print_res()
# a.generate(method="DeepFool", save_num=32)
# a.print_res()
# a.generate(method="SaliencyMapMethod", save_num=32)
# a.print_res()
# a.generate(method="UniversalPerturbation", save_num=32)
# a.print_res()
# a.generate(method="AutoAttack", save_num=32)
# a.print_res()
# a.generate(method="GDUAP", save_num=32)
# a.print_res()
# a.generate(method="CarliniWagner", save_num=32)
# a.print_res()
### BlackBox ScoreBased Attak
a.generate(method="PixelAttack", save_num=32)
a.print_res()
a.generate(method="SimBA", save_num=32)
a.print_res()
a.generate(method="ZooAttack", save_num=32)
a.print_res()
a.generate(method="SquareAttack", save_num=32)
a.print_res()
### BlackBox DecesionBased Attak

# ## CIFAR10
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# a = EvasionAttacker(modelnet=ResNet18(3), modelpath="./models/model_ckpt/ckpt-resnet18-cifar10_epoch19_acc0.8335.pth", 
#     dataset="cifar10", datasetpath="./datasets/", nb_classes=10, datanormalize = False, 
#     device=device, sample_num=128)
# ### white box attack
# a.generate(method="FastGradientMethod", norm="inf", save_num=32)
# a.print_res()
# a.generate(method="ProjectedGradientDescent", norm="inf", save_num=32)
# a.print_res()
# a.generate(method="BasicIterativeMethod", norm="inf", save_num=32)
# a.print_res()
# a.generate(method="DeepFool", save_num=32)
# a.print_res()
# a.generate(method="SaliencyMapMethod", save_num=32)
# a.print_res()
# a.generate(method="UniversalPerturbation", save_num=32)
# a.print_res()
# a.generate(method="AutoAttack", save_num=32)
# a.print_res()
# a.generate(method="GDUAP", save_num=32)
# a.print_res()
# a.generate(method="CarliniWagner", save_num=32)
# a.print_res()


# 后门攻击测试
## MNIST
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# b = BackdoorAttacker(modelnet=ResNet18(1), modelpath="./models/model_ckpt/ckpt-resnet18-mnist_epoch3_acc0.9898.pth", 
#     dataset="mnist", datasetpath="./datasets/", nb_classes=10, datanormalize = False, 
#     device=device)
# ### PoisoningAttackBackdoor
# b.backdoorattack(method="PoisoningAttackBackdoor",batch_size=128, pp_poison=0.01, target=1, test_sample_num=1024)

## CIFAR10
# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# b = BackdoorAttacker(modelnet=ResNet18(3), modelpath="./models/model_ckpt/ckpt-resnet18-cifar10_epoch19_acc0.8335.pth", 
#     dataset="cifar10", datasetpath="./datasets/", nb_classes=10, datanormalize = False, 
#     device=device)
### PoisoningAttackBackdoor
# b.backdoorattack(method="PoisoningAttackBackdoor",batch_size=128, pp_poison=0.01, target=1, test_sample_num=1024)
### PoisoningAttackCleanLabelBackdoor
# b.backdoorattack(method="PoisoningAttackCleanLabelBackdoor", batch_size=128, pp_poison=0.01, target=1, test_sample_num=1024)
### PoisoningAttackAdversarialEmbedding
# b.backdoorattack(method="PoisoningAttackAdversarialEmbedding", batch_size=128, pp_poison=0.01, target=1, test_sample_num=1024)
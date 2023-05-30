# 测试用例
from function.attack.attack_api import EvasionAttacker, BackdoorAttacker
import torch
from model.model_net.lenet import Lenet
from model.model_net.resnet import ResNet18
from function.attack.attacks.utils import load_mnist, load_cifar10

# 对抗攻击测试
## MNIST
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# a = EvasionAttacker(modelnet=ResNet18(1), modelpath="./model/model_ckpt/ckpt-resnet18-mnist_epoch3_acc0.9898.pth", 
#     dataset="mnist", datasetpath="./datasets/", nb_classes=10, datanormalize = False, 
#     device=device, sample_num=2)
## white box attack
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
# a.generate(method="PixelAttack", save_num=32)
# a.print_res()
# a.generate(method="SimBA", save_num=32)
# a.print_res()
a.generate(method="ZooAttack", save_num=32)
a.print_res()
# a.generate(method="SquareAttack", save_num=32)
# a.print_res()
### BlackBox DecesionBased Attak

# ## CIFAR10
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# a = EvasionAttacker(modelnet=ResNet18(3), modelpath="./model/model_ckpt/ckpt-resnet18-cifar10_epoch19_acc0.8335.pth", 
#     dataset="cifar10", datasetpath="./datasets/", nb_classes=10, datanormalize = False, 
#     device=device, sample_num=5)
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
# a.generate(method="SquareAttack", norm="inf", save_num=32, eps=0.156, p_init=0.03, loss_type="margin", n_restarts=3)
# a.print_res()


# 后门攻击测试
## MNIST
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# b = BackdoorAttacker(modelnet=ResNet18(1), modelpath="./model/model_ckpt/resnet_mnist_random.pth", 
#     dataset="mnist", datasetpath="./datasets/", nb_classes=10, datanormalize = False, 
#     device=device)
### PoisoningAttackBackdoor
# b.poision(method="PoisoningAttackAdversarialEmbedding", trigger=[16,16,3,1], pp_poison=0.01)
# b.train()
### PoisoningAttackCleanLabelBackdoor
# b.poision(method="PoisoningAttackAdversarialEmbedding", trigger=[16,16,3,1], pp_poison=0.01)
# b.train()
### FeatureCollisionAttack
# b.poision(method="PoisoningAttackAdversarialEmbedding", trigger=[16,16,3,1], pp_poison=0.01)
# b.train()
### PoisoningAttackAdversarialEmbedding
# b.poision(method="PoisoningAttackAdversarialEmbedding", trigger=[16,16,3,1], pp_poison=0.01)
# b.train()

## CIFAR10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
b = BackdoorAttacker(modelnet=ResNet18(3), modelpath="./model/model_ckpt/resnet_cifar10_random.pth", 
    dataset="cifar10", datasetpath="./datasets/", nb_classes=10, datanormalize = False, 
    device=device)
### PoisoningAttackBackdoor
# b.poision(method="PoisoningAttackBackdoor", trigger=[16,16,3,1], pp_poison=0.01, save_num=32, test_sample_num=4096, source=0, target=3)
# b.train(num_epochs=20, batch_size=128, save_model=True)
### PoisoningAttackCleanLabelBackdoor
# b.poision(method="PoisoningAttackCleanLabelBackdoor", trigger=[16,16,3,1], pp_poison=0.01)
# b.train()
### FeatureCollisionAttack
# b.poision(method="FeatureCollisionAttack", trigger=[16,16,3,1], pp_poison=0.01)
# b.train()
### PoisoningAttackAdversarialEmbedding
b.poision(method="PoisoningAttackAdversarialEmbedding", trigger=[16,16,3,1], pp_poison=0.01)
b.train()
from function.attack.attacks import *
from function.attack.estimators import *
from function.attack import attack_api, train_network
from model.model_net.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from function.attack.attack_api import EvasionAttacker, BackdoorAttacker
import torch

def run_adversarial(model, modelpath, dataname, method, attackparam, device):
    if dataname.lower() == "mnist":
        channel = 1
    else:
        channel = 3
    
    a = EvasionAttacker(modelnet=eval(model)(channel), modelpath=modelpath, dataset=dataname.lower(), device=device, datanormalize=True, sample_num=128)
        
    # 对应关系list
    methoddict={
        "FGSM":"FastGradientMethod",
        "BIM":"BasicIterativeMethod",
        "PGD":"ProjectedGradientDescent",
        "C&W":"CarliniWagner",
        "DeepFool":"DeepFool",
        "JacobianSaliencyMap":"SaliencyMapMethod",
        "Brendel&BethgeAttack":"BoundaryAttack",
        "UniversalPerturbation":"UniversalPerturbation",
        "AutoAttack":"AutoAttack",
        "GD-UAP":"GDUAP",
        "SquareAttack":"SquareAttack",
        "HSJA":"HopSkipJump",
        "PixelAttack":"PixelAttack",
        "SimBA":"SimBA",
        "ZOO":"ZooAttack",
        "GeoDA":"GeoDA",
        "Fastdrop":"Fastdrop"
    }
    res = a.generate(methoddict[method], 32, **attackparam)
    return a.print_res()

def run_backdoor(model, modelpath, dataname, method, pp_poison, target, device, save_path):
    if dataname.lower() == "mnist":
        channel = 1
    else:
        channel = 3
    b = BackdoorAttacker(modelnet=eval(model)(channel), modelpath=modelpath, dataset=dataname.lower(),  datanormalize=True, device=torch.device(device))
    # 对应关系list
    methoddict={
        "BackdoorAttack":"PoisoningAttackBackdoor",
        "Clean-LabelBackdoorAttack":"PoisoningAttackCleanLabelBackdoor",
        "CleanLabelFeatureCollisionAttack":"FeatureCollisionAttack",
        "AdversarialBackdoorEmbedding":"PoisoningAttackAdversarialEmbedding",
    }
    res = {"asr":[], "pp_poison":[]}
    res["accuracy"], res["accuracyonb"], res["attack_success_rate"]=b.backdoorattack(method=methoddict[method], batch_size=128, pp_poison=pp_poison, target=target, test_sample_num=1024,save_path=save_path)
    # for i in range(10):
    #     temp_pp_poison = 0.001 + 0.01 * i
    #     accuracy, accuracyonb, attack_success_rate=b.backdoorattack(method=methoddict[method], batch_size=128, pp_poison=temp_pp_poison, target=target, test_sample_num=1024,save_path=None)
    #     res["asr"].append(attack_success_rate)
    #     res["pp_poison"].append(temp_pp_poison)
    return res
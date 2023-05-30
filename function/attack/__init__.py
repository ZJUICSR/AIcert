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
    
    a = EvasionAttacker(modelnet=eval(model)(channel), modelpath=modelpath, dataset=dataname.lower(), device=device, datanormalize=False, sample_num=128)
        
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

def run_backdoor(model, modelpath, dataname, method, pp_poison, source, target, trigger, device, save_path, datasetpath="./datasets/", nb_classes=10):
    if dataname.lower() == "mnist":
        channel = 1
    else:
        channel = 3
    b = BackdoorAttacker(modelnet=eval(model)(channel), modelpath=modelpath, dataset=dataname.lower(), datasetpath=datasetpath, nb_classes=nb_classes, datanormalize=False, device=torch.device(device))
    # 对应关系list
    methoddict={
        "BackdoorAttack":"PoisoningAttackBackdoor",
        "Clean-LabelBackdoorAttack":"PoisoningAttackCleanLabelBackdoor",
        "CleanLabelFeatureCollisionAttack":"FeatureCollisionAttack",
        "AdversarialBackdoorEmbedding":"PoisoningAttackAdversarialEmbedding",
    }
    res = {"asr":[], "pp_poison":[]}
    b.poision(method=methoddict[method], pp_poison=pp_poison, source=source, target=target, trigger=trigger, test_sample_num=1024,save_path=save_path)
    res["accuracy"], res["accuracyonb"], res["attack_success_rate"] = b.train()
    for i in range(10):
        temp_pp_poison = 0.001 + 0.01 * i
        b.poision(method=methoddict[method], pp_poison=temp_pp_poison, source=source, target=target, trigger=trigger, test_sample_num=1024)
        accuracy, accuracyonb, attack_success_rate = b.train()
        res["asr"].append(attack_success_rate)
        res["pp_poison"].append(temp_pp_poison)
    return res
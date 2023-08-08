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
        "PGDL1":"ProjectedGradientDescent",
        "PGDL2":"ProjectedGradientDescent",
        "PGDLinf":"ProjectedGradientDescent",
        "C&W":"CarliniWagner",
        "DeepFool":"DeepFool",
        "JacobianSaliencyMap":"SaliencyMapMethod",
        "Brendel&BethgeAttack":"BoundaryAttack",
        "UniversalPerturbationL1":"UniversalPerturbation",
        "UniversalPerturbationL2":"UniversalPerturbation",
        "UniversalPerturbationLinf":"UniversalPerturbation",
        "AutoAttackL1":"AutoAttack",
        "AutoAttackL2":"AutoAttack",
        "AutoAttackLinf":"AutoAttack",
        "GD-UAP":"GDUAP",
        "SquareAttackL1":"SquareAttack",
        "SquareAttackL2":"SquareAttack",
        "SquareAttackLinf":"SquareAttack",
        "HSJA":"HopSkipJump",
        "PixelAttack":"PixelAttack",
        "SimBA":"SimBA",
        "ZOO":"ZooAttack",
        "GeoDAL1":"GeoDA",
        "GeoDAL2":"GeoDA",
        "GeoDALinf":"GeoDA",
        "Fastdrop":"Fastdrop",
        "DIFGSM":"DIFGSM",
        "FAB":"FAB",
        "FABL1":"FABL1",
        "FABL2":"FABL2",
        "EOTPGD":"EOTPGD",
        "FFGSM":"FFGSM",
        "Jitter":"Jitter",
        "MIFGSM":"MIFGSM",
        "NIFGSM":"NIFGSM",
        "PGDRS":"PGDRS",
        "PGDRSL2":"PGDRSL2",
        "RFGSM":"RFGSM",
        "SINIFGSM":"SINIFGSM",
        "SparseFool":"SparseFool",
        "SPSA":"SPSA",
        "TIFGSM":"TIFGSM",
        "TPGD":"TPGD",
        "VMIFGSM":"VMIFGSM",
        "VNIFGSM":"VNIFGSM"
    }
    if method not in ["FABL1","FABL2","PGDRSL2"]:
        if "L1" in method :
            attackparam["norm"] = 1
        elif "L2" in method :
            attackparam["norm"] = 2
        elif "Linf" in method :
            attackparam["norm"] = "inf"
    res, piclist = a.generate(methoddict[method], **attackparam)
    return a.print_res(),piclist

def run_backdoor(model, modelpath, dataname, method, pp_poison, save_num, test_sample_num, target, trigger, device, nb_classes=10, method_param=None):
    datasetpath="./datasets/"
    channel = 1 if dataname.lower() == "mnist" else 3
    # if dataname.lower() == "mnist":
    #     channel = 1
    # else:
    #     channel = 3
    b = BackdoorAttacker(modelnet=eval(model)(channel), modelpath=modelpath, dataset=dataname.lower(), datasetpath=datasetpath, nb_classes=nb_classes, datanormalize=False, device=torch.device(device))
    # 对应关系list
    methoddict={
        "PoisoningAttackBackdoor":"PoisoningAttackBackdoor",
        "PoisoningAttackCleanLabelBackdoorL1":"PoisoningAttackCleanLabelBackdoor",
        "PoisoningAttackCleanLabelBackdoorL2":"PoisoningAttackCleanLabelBackdoor",
        "PoisoningAttackCleanLabelBackdoorLinf":"PoisoningAttackCleanLabelBackdoor",
        "FeatureCollisionAttack":"FeatureCollisionAttack",
        "PoisoningAttackAdversarialEmbedding":"PoisoningAttackAdversarialEmbedding",
    }
    if method == "PoisoningAttackCleanLabelBackdoorL1":
        method_param["norm"] = 1
    elif method == "PoisoningAttackCleanLabelBackdoorL2":
        method_param["norm"] = 2
    elif method == "PoisoningAttackCleanLabelBackdoorLinf":
        method_param["norm"] = "inf"
    res = {"asr":[], "pp_poison":[]}
    
    print(f"param:{method_param}")
    # b.poision(method=methoddict[method], pp_poison=pp_poison, save_num=save_num, test_sample_num=test_sample_num, 
    #           target=target, trigger=trigger, **method_param)
    b.poision(method=methoddict[method], **method_param)
    res["accuracy"], res["accuracyonb"], res["attack_success_rate"] = b.train()
    res["accuracy"] = round(res["accuracy"],1)
    res["accuracyonb"] = round(res["accuracyonb"],1)
    res["attack_success_rate"] = round(res["attack_success_rate"],1)
    for i in range(10):
        temp_pp_poison = 0.001 + 0.1 * i
        # b.poision(method=methoddict[method], pp_poison=temp_pp_poison, test_sample_num=test_sample_num, target=target, trigger=trigger, **method_param)
        b.poision(method=methoddict[method], **method_param)
        accuracy, accuracyonb, attack_success_rate = b.train()
        res["asr"].append(round(attack_success_rate*100,1))
        res["pp_poison"].append(temp_pp_poison)
    return res


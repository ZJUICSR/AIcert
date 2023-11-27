from function.attack.attacks import *
from function.attack.estimators import *
from function.attack import attack_api, train_network
from model.model_net.resnet_attack import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from model.model_net.resnet_attack_feature import ResNet18 as ResNet18_f
from model.model_net.resnet_attack_feature import ResNet34 as ResNet34_f
from model.model_net.resnet_attack_feature import ResNet50 as ResNet50_f
from model.model_net.resnet_attack_feature import ResNet101 as ResNet101_f
from model.model_net.resnet_attack_feature import ResNet152 as ResNet152_f
from function.attack.attack_api import EvasionAttacker, BackdoorAttacker
import torch, os, time
import os.path as osp

def get_method(method, attackparam):
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
        "VNIFGSM":"VNIFGSM",
        "AutoPGDL1":"AutoProjectedGradientDescent",
        "AutoPGDL2":"AutoProjectedGradientDescent",
        "AutoPGDLinf":"AutoProjectedGradientDescent",
        "Auto-CGL1":"AutoConjugateGradient",
        "Auto-CGL2":"AutoConjugateGradient",
        "Auto-CGLinf":"AutoConjugateGradient",
        "ElasticNetL1":"ElasticNet",
        "ElasticNetL2":"ElasticNet",
        "ElasticNet-EN":"ElasticNet",
        "FeatureAdversaries":"FeatureAdversariesPyTorch",
        "NewtonFool":"NewtonFool",
        "SpatialTransformation":"SpatialTransformation",
        "TargetedUniversalPerturbationL2":"TargetedUniversalPerturbation",
        "TargetedUniversalPerturbationLinf":"TargetedUniversalPerturbation",
        "VirtualAdversarialMethod":"VirtualAdversarialMethod",
        "SignOPTAttack":"SignOPTAttack",
    }
    if methoddict[method] == "ElasticNet":
        if "L1" in method :
            attackparam["decision_rule"] = "L1"
        elif "L2" in method :
            attackparam["decision_rule"] = "L2"
        elif "-EN" in method :
            attackparam["decision_rule"] = "EN"
    elif method not in ["FABL1","FABL2","PGDRSL2"]:
        if "L1" in method :
            attackparam["norm"] = 1
        elif "L2" in method :
            attackparam["norm"] = 2
        elif "Linf" in method :
            attackparam["norm"] = "inf"
    if method == "AdversarialPatch":
        attackparam["patch_shape"] = tuple(attackparam["patch_shape"])
        print("attackparam:",attackparam)
    return methoddict[method], attackparam

def run_adversarial(model, modelpath='./model/ckpt/mnist_resnet18.pth', dataname='mnist', method='FGSM', attackparam={}, device='cuda', sample_num=128):
    if dataname.lower() == "mnist":
        channel = 1
    else:
        channel = 3
    if modelpath:
        a = EvasionAttacker(modelnet=eval(model)(channel), modelpath=modelpath, dataset=dataname.lower(), device=device, datanormalize=False, sample_num=sample_num)
    else:
        a = EvasionAttacker(dataset=dataname.lower(), device=device, datanormalize=False, sample_num=sample_num, model=model)
    method_in_alg, attackparam = get_method(method, attackparam)
   
    adv, real_lables, piclist = a.generate(method_in_alg, **attackparam)
    if modelpath:
        data = {
            "x":adv,
            "y":real_lables
        }
        if "eps" not in attackparam.keys():
            eps = 0.1
        else:
            eps = attackparam["eps"]

        if "steps" not in attackparam.keys():
            steps = 1
        else:
            steps = attackparam["steps"]
        try:
            model_name = modelpath.split("_")[-1].split('.')[0]
        except:
            model_name = modelpath.split("/")[-1].split('.')[0]
        save_root = "dataset/adv_data"
        if not osp.exists(save_root):
            os.makedirs(save_root)
        path = osp.join(save_root, "adv_attack_{:s}_{:s}_{:s}_s{:04d}_{:.5f}.pt".format(
                                method, model_name, dataname, sample_num, eps))
        torch.save(data, path)
        del data
    else:
        path=None

    return a.print_res(), piclist, path, len(real_lables)


def run_adversarial_graph(model, dataname='mnist', methods=['FGSM'], attackparam={}, device='cuda', sample_num=128,log_func=None):
    if dataname.lower() == "mnist":
        channel = 1
    else:
        channel = 3
    a = EvasionAttacker(dataset=dataname.lower(), device=device, datanormalize=False, sample_num=sample_num, model=model)
    result = {}
    for method in methods:
        if log_func:
            log_func('[模型测试阶段] 选择算法：{:s}进行测试'.format(str(method)))
        method_in_alg, attackparam1 = get_method(method, attackparam[method])
        adv, real_lables, piclist = a.generate(method_in_alg, **attackparam1)
        result[method] = a.print_res()
        if log_func:
            log_func('[模型测试阶段] {:s}攻击后的准确率为{:.4f}'.format(str(method),result[method]['after_acc']))
    return result

def run_get_adv_data(dataset_name, model, dataloader, device='cuda', method='FGSM', attackparam={}):
    print("dataset_name.lower():",dataset_name.lower())
    AttackObj = EvasionAttacker(dataset=dataset_name.lower(), device=device, datanormalize=False, sample_num=10,model=model)
    method_in_alg, attackparam = get_method(method, attackparam)
    attack_info = AttackObj.get_adv_data(dataloader=dataloader, method=method_in_alg, **attackparam)
    del AttackObj
    return attack_info

def run_get_adv_attack(dataset_name, model, dataloader, device='cuda', method='FGSM', attackparam={}):
    AttackObj = EvasionAttacker(dataset=dataset_name.lower(), device=device, datanormalize=False, sample_num=10,model=model)
    method_in_alg, attackparam = get_method(method, attackparam)
    attack = AttackObj.get_attack(method=method_in_alg, **attackparam)
    del AttackObj
    return attack

class Attack_Obj(object):
    def __init__(self, dataset_name, model, dataloader, device='cuda', method='FGSM', attackparam={}):
        self.AttackObj=EvasionAttacker(dataset=dataset_name.lower(), device=device, datanormalize=False, sample_num=10,model=model)
        self.method_in_alg, self.attackparam = get_method(method, attackparam)
        self.dataloader = dataloader
    
    def attack(self, x, **kwargs):
        for key, value in kwargs.items():
            self.attackparam[key] = value
        attack = self.AttackObj.get_attack(method=self.method_in_alg, **self.attackparam)
        return attack(x)
    
    def get_adv_attack(self):
        attack_info = self.AttackObj.get_adv_data(dataloader=self.dataloader, method=self.method_in_alg, **self.attackparam)
        return attack_info
    
def run_backdoor(model, modelpath, dataname, method, pp_poison, save_num,  target, trigger, device, 
                 test_sample_num=1000, train_sample_num=1000, nb_classes=10, method_param=None, feature=False):
    model = model+'_f' if feature else model
    datasetpath="./datasets/"
    channel = 1 if dataname.lower() == "mnist" else 3
    b = BackdoorAttacker(modelnet=eval(model)(channel), modelpath=modelpath, dataset=dataname.lower(), datasetpath=datasetpath, 
                         nb_classes=nb_classes, datanormalize=False, device=torch.device(device),
                         test_sample_num=test_sample_num, train_sample_num=train_sample_num)
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
    res["accuracy"] = round(res["accuracy"],3)
    res["accuracyonb"] = round(res["accuracyonb"],3)
    res["attack_success_rate"] = round(res["attack_success_rate"],3)
    pp_ = method_param['pp_poison']
    save_num_ = method_param['save_num']
    method_param['save_num'] = 0
    for i in range(5):
        if i == 0:
            temp_pp_poison = 0.001
        else:
            temp_pp_poison = 0.01 * i
        method_param['pp_poison'] = temp_pp_poison
        
        # b.poision(method=methoddict[method], pp_poison=temp_pp_poison, test_sample_num=test_sample_num, target=target, trigger=trigger, **method_param)
        code = b.poision(method=methoddict[method], **method_param)
        if code == -1:
            res["asr"].append(0)
        else:
            accuracy, accuracyonb, attack_success_rate = b.train()
            res["asr"].append(round(attack_success_rate*100,1))
        res["pp_poison"].append(temp_pp_poison)
    method_param['save_num'] = save_num_
    method_param['pp_poison'] = pp_
    return res


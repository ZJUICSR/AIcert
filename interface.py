#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'chunlai'
__copyright__ = 'Copyright © 2021/11/01, ZJUICSR'

from function.attack.old.backdoor.badnets.badnets_v2 import Badnets
from model import ModelLoader
from model.trainer import Trainer
from dataset import ArgpLoader
from dataset import LoadCustom
from function.attack.config import Conf
from function.attack.old.adv import Attack
import os, json, copy
import os.path as osp
from torch.utils.data import Dataset,DataLoader
from IOtool import IOtool, Logger, Callback
from torchvision import  transforms
import torch
from function.formal_verify import *
from function.attack.adv0211 import EvasionAttacker, BackdoorAttacker

from function.fairness import api
from function import concolic, env_test


ROOT = osp.dirname(osp.abspath(__file__))
def run_model_debias(tid,AAtid,dataname,modelname,algorithmname):
    """模型公平性提升
    :params tid:主任务ID
    :params AAtid:子任务id
    :params dataname:数据集名称
    :params modelname:模型名称
    :params algorithmname:优化算法名称
    """
    taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
    res = api.model_debias(dataname,modelname,algorithmname)
    conslist = []
    for cons in res["Consistency"]:
        conslist.append(float(cons))
    res["Consistency"] = conslist
    res["stop"] = 1
    IOtool.write_json(res,osp.join(ROOT,"output", tid, AAtid+"_result.json"))
    taskinfo[tid]["function"][AAtid]["state"]=2
    taskinfo[tid]["state"]=2
    IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))

def run_data_debias(tid,AAtid,dataname,datamethod):
    """数据集公平性提升
    :params tid:主任务ID
    :params AAtid:子任务id
    :params dataname:数据集名称
    :params datamethod:优化算法名称
    """
    taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
    res = api.dataset_debias(dataname,datamethod)
    res["stop"] = 1
    IOtool.write_json(res,osp.join(ROOT,"output", tid, AAtid+"_result.json"))
    taskinfo[tid]["function"][AAtid]["state"]=2
    taskinfo[tid]["state"]=2
    IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))

def get_badnets_dataloader(dataloader):
    '''
    获取投毒数据
    :params dataloader:数据集
    '''
    badnet = Badnets(data_loader=dataloader)
    poison_loader = badnet.poison()
    return poison_loader

def get_data_loader(data_path, data_name, params, transform =None):
    """
    加载数据集
    :param data_path:数据集路径
    :param data_name:数据集名称 
    :param params:数据集属性:num_classes path batch_size bounds mean std
    """
    # print(data_path, data_name, params)
    if "Custom" in data_path:
        dataset = LoadCustom(data_name,transform=transform)
        test_loader = DataLoader(dataset=dataset, batch_size=params["batch_size"], shuffle=True)
        return test_loader
    else:
        dataloader = ArgpLoader(data_root=data_path, dataset=data_name, **params)
        train_loader, test_loader = dataloader.get_loader()
        
        params = dataloader.__config__(dataset=data_name)
        return test_loader,train_loader,params

# def get_model_loader(data_path, data_name, model_name, num_classes):
def get_model_loader(dataset, modelparam, logging, train_loader=None, test_loader=None, taskparam=None):
    """
    加载模型,模型训练
    :param dataset:数据集信息
    :param data_name:数据集名称 
    :param logging:日志对象
    """
    data_path = dataset["path"]
    data_name = dataset["name"]
    num_classes = dataset["num_classes"]
    model_name = modelparam["name"]
    model_loader = ModelLoader(data_path=data_path, arch=model_name, task=data_name)
    model = model_loader.get_model(num_classes=num_classes)
    if train_loader == None:
        train_loader = copy.deepcopy(test_loader)
    logging.info("[模型训练阶段] 正在分析AI模型（网络结构、参数和大小等）")
    batch_x = list(train_loader)[0][0]
    summary = IOtool.summary_dict(model.cpu(), input_size=batch_x.shape[1:], batch_size=batch_x.shape[0])
    trainerParam = {
        "lr": 0.05,
        "optim": "SGD",
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "robustness": [],
        "epochs": 100
    }
    try:
        taskparam.set_res_value(key = "summary",value = summary)
        taskparam.set_params_value(key="trainer",value=trainerParam)
    except:
        pass
    trainer = Trainer(**trainerParam)
    trainer.update_config(arch=model_name, task=data_name)
    flag = False
    if modelparam["pretrained"]:
        model_path = modelparam["path"]
        if modelparam["path"] and osp.exists(model_path):
            # load from path
            weights = torch.load(model_path)
            if weights is not None:
                flag = True
                logging.info("[模型训练阶段] 从指定文件:'{:s}'中加载预训练模型".format(str(model_path)))
                model.load_state_dict(weights)
        else:
            # load from default saved
            weights = model_loader.load(model_name, model_loader)
            if weights is not None:
                flag = True
                logging.info("[模型训练阶段] 从默认缓存中加载预训练模型")
                model.load_state_dict(weights)
    if (not modelparam["pretrained"]) or not flag:
        logging.info("[模型训练阶段] 开始模型训练...")
        model = trainer.train(model, train_loader, test_loader, epochs=trainerParam["epochs"],
                              epoch_fn=Callback.callback_train, logging=logging,results=taskparam)
        logging.info("[模型训练阶段] 模型训练完毕，保存预训练模型")
        model_loader.save(model, arch=model_name, task=data_name)
    test_acc, test_loss = trainer.test(model, test_loader=test_loader)
    adv_attack_res = taskparam.get_res_value(key="AdvAttack")
    adv_attack_res = {"test_acc":test_acc,"test_loss":test_loss}
    taskparam.set_res_value(key="AdvAttack",value=adv_attack_res)
    logging.info("[模型训练阶段] 模型训练完成，测试准确率：{:.3f}% 测试损失：{:.5f}".format(test_acc, test_loss))
    return model, trainer, summary

def run_verify(tid, AAtid, param):
    taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
    N = param['size']
    device = 'cpu'
    verify = vision_verify
    if param['dataset'] == 'mnist':
        mn_model = get_mnist_cnn_model()
        test_data, n_class = get_mnist_data(number=N, batch_size=10)
    if param['dataset'] == 'cifar':
        mn_model = get_cifar_resnet18()
        test_data, n_class = get_cifar_data(number=N, batch_size=10)
    if param['dataset'] == 'gtsrb':
        mn_model = get_gtsrb_resnet18()
        test_data, n_class = get_gtsrb_data(number=N, batch_size=10)
    if param['dataset'] == 'mtfl':
        mn_model = get_MTFL_resnet18()
        test_data, n_class = get_MTFL_data(number=N, batch_size=10)
    if param['dataset'] == 'sst':
        mn_model = get_lstm_demo_model()
        test_data, _ = get_sst_data(ver_num=N)
        n_class = 2
        verify = language_verify

    global LiRPA_LOGS
    input_param = {'interface': 'Verification',
                    'node': "中间结果可视化",
                    'input_param': {'model': mn_model,
                                    'dataset': test_data,
                                    'n_class': n_class,
                                    'up_eps': param['up_eps'],
                                    'down_eps': param['down_eps'],
                                    'steps': param['steps'],
                                    'device': device,
                                    'output_path': osp.join('output',tid,AAtid,"formal_img"),
                                    'task_id': f"{param['task_id']}"}}
    global result
    result = verify(input_param)
    result["stop"] = 1
    IOtool.write_json(result,osp.join(ROOT,"output", tid, AAtid+"_result.json"))
    taskinfo[tid]["function"][AAtid]["state"]=2
    taskinfo[tid]["state"]=2
    IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
    
    
def run_concolic(tid, AAtid, dataname, modelname, norm):
    """测试样本自动生成
    :params tid:主任务ID
    :params AAtid:子任务id
    :params dataname:数据集名称
    :params modelname:模型名称
    :params norm:范数约束
    """
    taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
    res = concolic.run_concolic(dataname, modelname, norm)   
    IOtool.write_json(res,osp.join(ROOT,"output", tid, AAtid+"_result.json"))
    taskinfo[tid]["function"][AAtid]["state"]=2
    taskinfo[tid]["state"]=2
    IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
    
    
def run_envtest(tid,AAtid,matchmethod,frameworkname,frameversion):
    """系统环境分析
    :params tid:主任务ID
    :params AAtid:子任务id
    :params matchmethod:环境分析匹配机制
    :params frameworkname:适配框架名称
    :params frameversion:框架版本
    """
    taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
    res = env_test.run_env_frame(matchmethod,frameworkname,frameversion)
    # res = concolic.run_concolic(dataname, modelname, norm)   
    IOtool.write_json(res,osp.join(ROOT,"output", tid, AAtid+"_result.json"))
    taskinfo[tid]["function"][AAtid]["state"]=2
    taskinfo[tid]["state"]=2
    IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))


def run_adv_attack(tid, stid, dataname, model, methods, inputParam):
    """对抗攻击评估
    :params tid:主任务ID
    :params stid:子任务id
    :params dataname:数据集名称
    :params model:模型名称
    :params methods:list，对抗攻击方法
    :params inputParam:输入参数
    """
    # 开始执行标记任务状态
    taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
    taskinfo[tid]["function"][stid]["state"]=1
    taskinfo[tid]["state"]=1
    IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
    modelpath = osp.join("./model/ckpt",dataname.upper() + "_" + model.lower()+".pt")
    device = torch.device(inputParam['device'])
    a = EvasionAttacker(modelnet=model.lower(), modelpath=modelpath, dataset=dataname.lower(), device=device, datanormalize=True)
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
        "GD-UAP":"GDUniversarial",
        "SquareAttack":"SquareAttack",
        "HSJA":"HopSkipJump",
        "PixelAttack":"PixelAttack",
        "SimBA":"SimBA",
        "ZOO":"ZooAttack",
        "GeoDA":"GeoDA",
        "Fastdrop":"Fastdrop"
    }
    resultlist={}
    for method in methods:
        attackparam = inputParam[method]
        print("methoddict[method]--------------",methoddict[method])
        a.perturb(methoddict[method], 1024, **attackparam)
        print("********************method**********:",method)
        a.print_res()
        resultlist[method]=a.attack_with_eps(epslist=[0.00001, 0.01])
    taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
    taskinfo[tid]["function"][stid]["state"] = 2
    IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
    IOtool.change_task_success_v2(tid)
    
    
def run_backdoor_attack(tid, stid, dataname, model, methods, inputParam):
    """后门攻击评估
    :params tid:主任务ID
    :params stid:子任务id
    :params dataname:数据集名称
    :params model:模型名称
    :params methods:list，后门攻击方法
    :params inputParam:输入参数
    """
    # 开始执行标记任务状态
    taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
    taskinfo[tid]["function"][stid]["state"]=1
    taskinfo[tid]["state"]=1
    IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
    modelpath = osp.join("./model/ckpt",dataname.upper() + "_" + model.lower()+".pt")
    b = BackdoorAttacker(modelnet=model.lower(), modelpath=modelpath, dataset=dataname.lower(),  datanormalize=True, device=torch.device(inputParam["device"]))
    # 对应关系list
    methoddict={
        "BackdoorAttack":"PoisoningAttackBackdoor",
        "Clean-LabelBackdoorAttack":"PoisoningAttackCleanLabelBackdoor",
        "CleanLabelFeatureCollisionAttack":"FeatureCollisionAttack",
        "AdversarialBackdoorEmbedding":"PoisoningAttackAdversarialEmbedding",
    }
    for method in methods:
        attackparam = inputParam[method]
        print("methoddict[method]--------------",methoddict[method])
        b.backdoorattack(method=methoddict[method], batch_size=128, pp_poison=0.01, target=1, test_sample_num=1024)
        
        print("********************method**********:",method)
        
    taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
    taskinfo[tid]["function"][stid]["state"] = 2
    IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
    IOtool.change_task_success_v2(tid)

from function.ex_methods.module.func import get_loader, Logger
from function.ex_methods.module.generate_adv import get_adv_loader
from function.ex_methods.module.load_model import load_model
from function.ex_methods import attribution_maps, layer_explain, dim_reduciton_visualize
from function.ex_methods.module.model_Lenet import lenet

def run_ex_method(tid, stid, datasetparam, modelparam, ex_methods, adv_methods, device):
    params = {
        "dataset": datasetparam,
        "model": modelparam,
        "out_path": "./output",
        "device": torch.device("cuda:4"),
        "ex_methods":{"methods":ex_methods},
        "adv_methods":{"methods":adv_methods},
        "root":ROOT
    }
    logging = Logger(filename=osp.join(ROOT,"output", tid, stid +"_log.txt"))

    root = ROOT
    dataset = datasetparam["name"]
    nor_data = torch.load(osp.join(root, f"dataset/{dataset}/data/{dataset}_NOR.pt"))
    nor_loader = get_loader(nor_data, batchsize=16)
    logging.info("[数据集获取]：获取{:s}数据集正常样本已完成.".format(dataset))

    model_name = modelparam["name"]
    model = modelparam["ckpt"]
    logging.info("[加载被解释模型]：准备加载被解释模型{:s}".format(model_name))
    net = load_model(model_name, dataset, device, root, reference_model=model, logging=logging)
    # net = torchvision.models.inception_v3(num_classes=10)
    net = net.eval().to(device)
    logging.info("[加载被解释模型]：被解释模型{:s}已加载完成".format(model_name))

    adv_loader = {}
    for adv_method in adv_methods:
        adv_loader[adv_method] = get_adv_loader(net, nor_loader, adv_method, params, batchsize=16, logging=logging)
    logging.info("[数据集获取]：获取{:s}对抗样本已完成".format(dataset))

    save_path = osp.join(ROOT,"output", tid, stid)
    if not osp.exists(save_path):
        os.mkdir(save_path)
    vis_type_list = ['pca', 'ss', 'tsne', 'svm', 'mean_diff']
    dim_reduciton_visualize(vis_type_list, nor_loader, adv_loader["FGSM"], net, model_name, dataset, device, save_path)
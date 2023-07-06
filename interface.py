#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'chunlai'
__copyright__ = 'Copyright © 2021/11/01, ZJUICSR'

from model import ModelLoader
from model.trainer import Trainer
from dataset import ArgpLoader
from dataset import LoadCustom
from function.attack.config import Conf
import os, json, copy
import os.path as osp
from torch.utils.data import Dataset,DataLoader
from IOtool import IOtool, Callback
from torchvision import  transforms
import torch
from function.formal_verify import *
from PIL import Image
from model.model_net.lenet import Lenet
from model.model_net.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from function.attack import run_adversarial, run_backdoor
import cv2 
from function.fairness import run_dataset_debias, run_model_debias
from function import concolic, env_test, deepsst, dataclean
from function.ex_methods.module.func import get_loader, Logger, recreate_image
from function.ex_methods.module.generate_adv import get_adv_loader, sample_untargeted_attack
from function.ex_methods.module.load_model import load_model
from function.ex_methods import attribution_maps, layer_explain, dim_reduciton_visualize
from function.ex_methods.module.model_Lenet import lenet
from function.ex_methods.lime import lime_image_ex

from function.formal_verify.auto_verify import auto_verify_img
from function.formal_verify.knowledge_consistency import load_checkpoint,get_feature
from function.formal_verify.knowledge_consistency import Model_zoo as models
import matplotlib.pyplot as plt
from function.formal_verify.veritex import Net as reachNet
from function.formal_verify.veritex.networks.cnn import Method as ReachMethod
from function.formal_verify.veritex.utils.plot_poly import plot_polytope2d

from function.defense.jpeg import Jpeg
from function.defense.twis import Twis
from function.defense.region_based import RegionBased
from function.defense.pixel_deflection import Pixel_Deflection
from function.defense.feature_squeeze import feature_squeeze
from function.defense.preprocessor.preprocessor import *
from function.defense.trainer.trainer import *
from function.defense.detector.poison.detect_poison import *
from function.defense.transformer.poisoning.transformer_poison import *
from function.defense.sage.sage import *
from function.defense.models import *
from torchvision.models import vgg16
from function.side import *

ROOT = osp.dirname(osp.abspath(__file__))
def run_model_debias_api(tid, stid, dataname, modelname, algorithmname, metrics, sensattrs, targetattr, staAttrList):
    """模型公平性提升
    :params tid:主任务ID
    :params AAtid:子任务id
    :params dataname:数据集名称
    :params modelname:模型名称
    :params algorithmname:优化算法名称
    """
    logging = Logger(filename=osp.join(ROOT,"output", tid, stid +"_log.txt"))
    taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
    res = run_model_debias(dataname, modelname, algorithmname, metrics, sensattrs, targetattr, staAttrList, logging=logging)
    res["stop"] = 1
    IOtool.write_json(res, osp.join(ROOT,"output", tid, stid+"_result.json"))
    taskinfo[tid]["function"][stid]["state"]=2
    taskinfo[tid]["state"]=2
    IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
    

def run_data_debias_api(tid, stid, dataname, datamethod, senAttrList, tarAttrList, staAttrList):
    """数据集公平性提升
    :params tid:主任务ID
    :params AAtid:子任务id
    :params dataname:数据集名称
    :params datamethod:优化算法名称
    """
    logging = Logger(filename=osp.join(ROOT,"output", tid, stid +"_log.txt"))
    taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
    res = run_dataset_debias(dataname, datamethod, senAttrList, tarAttrList, staAttrList, logging=logging)
    res["stop"] = 1
    IOtool.write_json(res,osp.join(ROOT,"output", tid, stid+"_result.json"))
    taskinfo[tid]["function"][stid]["state"]=2
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
    logging = Logger(filename=osp.join(ROOT,"output", tid, AAtid +"_log.txt"))
    taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
    N = param['size']
    device = 'cpu'
    verify = vision_verify
    if param['dataset'] == 'mnist':
        mn_model = get_mnist_cnn_model()
        test_data, n_class = get_mnist_data(number=N, batch_size=10)
    if param['dataset'] == 'cifar10':
        mn_model = get_cifar_resnet18() if param['model']!= 'densenet' else get_cifar_densenet_model()
        test_data, n_class = get_cifar_data(number=N, batch_size=10)
    if param['dataset'] == 'gtsrb':
        mn_model = get_gtsrb_resnet18()
        test_data, n_class = get_gtsrb_data(number=N, batch_size=10)
    if param['dataset'] == 'mtfl':
        mn_model = get_MTFL_resnet18()
        test_data, n_class = get_MTFL_data(number=N, batch_size=10)
    if param['dataset'] == 'sst2':
        mn_model = get_lstm_demo_model() if param['model'] != 'transformer' else get_transformer_model()
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
                                    "log_func":logging,
                                    'output_path': osp.join('output',tid,AAtid,"formal_img"),
                                    'task_id': f"{param['task_id']}"}}
    global result
    result = verify(input_param)
    result["stop"] = 1
    IOtool.write_json(result,osp.join(ROOT,"output", tid, AAtid+"_result.json"))
    taskinfo[tid]["function"][AAtid]["state"]=2
    taskinfo[tid]["state"]=2
    IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
    
    
def run_concolic(tid, AAtid, dataname, modelname, norm, times):
    """测试样本自动生成
    :params tid:主任务ID
    :params AAtid:子任务id
    :params dataname:数据集名称
    :params modelname:模型名称
    :params norm:范数约束
    """
    logging = Logger(filename=osp.join(ROOT,"output", tid, AAtid +"_log.txt"))
    taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
    res = concolic.run_concolic(dataname.lower(), modelname.lower(), norm.lower(), int(times), osp.join(ROOT,"output", tid, AAtid), logging)  
    res["stop"]=1 
    IOtool.write_json(res,osp.join(ROOT,"output", tid, AAtid+"_result.json"))
    taskinfo[tid]["function"][AAtid]["state"]=2
    taskinfo[tid]["state"]=2
    IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
    
def run_dataclean(tid, AAtid, dataname):
    """异常数据检测
    :params tid:主任务ID
    :params AAtid:子任务id
    :params dataname:数据集名称
    :output res:需保存到子任务json中的返回结果/路径
    """
    taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
    res = dataclean.run_dataclean(dataname)   
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
    :output res:需保存到子任务json中的返回结果/路径
    """
    logging = Logger(filename=osp.join(ROOT,"output", tid, AAtid +"_log.txt"))
    
    taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
    res = env_test.run_env_frame(matchmethod,frameworkname,frameversion, osp.join(ROOT,"output", tid, AAtid), logging)
    # res = concolic.run_concolic(dataname, modelname, norm)  
    res["detection_result"]=IOtool.load_json(res["env_test"]["detection_result"]) 
    res["stop"] = 1
    IOtool.write_json(res,osp.join(ROOT,"output", tid, AAtid+"_result.json"))
    taskinfo[tid]["function"][AAtid]["state"]=2
    taskinfo[tid]["state"]=2
    IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))

def run_coverage(tid,AAtid,dataset,modelname):
    pass
  
def run_deepsst(tid,AAtid,dataset,modelname,pertube,m_dir):
    """敏感神经元测试准则
    :params tid:主任务ID
    :params AAtid:子任务id
    :params dataset: 数据集名称
    :params modelname: 模型名称
    :params pertube: 敏感神经元扰动比例
    :params m_dir: 敏感度值文件位置
    :output res:需保存到子任务json中的返回结果/路径
    """
    logging = Logger(filename=osp.join(ROOT,"output", tid, AAtid +"_log.txt"))
    taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
    res = deepsst.run_deepsst(dataset.lower(), modelname, float(pertube.strip("%"))/100, m_dir, osp.join(ROOT,"output", tid, AAtid), logging)  
    res["stop"] = 1
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
    logging = Logger(filename=osp.join(ROOT,"output", tid, stid +"_log.txt"))
    # 开始执行标记任务状态
    taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
    taskinfo[tid]["function"][stid]["state"]=1
    taskinfo[tid]["state"]=1
    IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
    modelpath = osp.join("./model/ckpt",dataname.upper() + "_" + model.lower()+".pth")
    device = torch.device(inputParam['device'])
    if (not osp.exists(modelpath)):
            logging.info("[模型获取]:服务器上模型不存在")
            result["stop"] = 1
            IOtool.write_json(result,osp.join(ROOT,"output", tid, stid+"_result.json"))
            taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
            taskinfo[tid]["function"][stid]["state"] = 3
            IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
            IOtool.change_task_success_v2(tid)
            return 0
    if not osp.exists(osp.join(ROOT,"output", tid, stid)):
        os.mkdir(osp.join(ROOT,"output", tid, stid))
    resultlist={}
    for method in methods:
        logging.info("[执行对抗攻击]:正在执行{:s}对抗攻击".format(method))
        attackparam = inputParam[method]
        attackparam["save_path"] = osp.join(ROOT,"output", tid, stid)
        if "norm" in attackparam.keys() and attackparam["norm"]=="np.inf":
            attackparam["norm"]=np.inf
        print(attackparam)
        resultlist[method] = run_adversarial(model, modelpath, dataname, method, attackparam, device)
        logging.info("[执行对抗攻击中]:{:s}对抗攻击结束，攻击成功率为{}%".format(method,resultlist[method]["asr"]))
    logging.info("[执行对抗攻击]:对抗攻击执行完成，数据保存中")
    resultlist["stop"] = 1
    IOtool.write_json(resultlist,osp.join(ROOT,"output", tid, stid+"_result.json"))
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
    logging = Logger(filename=osp.join(ROOT,"output", tid, stid +"_log.txt"))
    taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
    taskinfo[tid]["function"][stid]["state"]=1
    taskinfo[tid]["state"]=1
    IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
    modelpath = osp.join("./model/ckpt",dataname.upper() + "_" + model.lower()+".pth")
    if (not osp.exists(modelpath)):
            logging.info("[模型获取]:服务器上模型不存在")
            result["stop"] = 1
            IOtool.write_json(result,osp.join(ROOT,"output", tid, stid+"_result.json"))
            taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
            taskinfo[tid]["function"][stid]["state"] = 3
            IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
            IOtool.change_task_success_v2(tid)
            return 0
    res = {}
    logging.info("[执行后门攻击]:开始后门攻击")
    for method in methods:
        logging.info("[执行后门攻击]:正在执行{:s}后门攻击".format(method))
        res[method]={}
        attackparam = inputParam[method]
        save_path = osp.join(ROOT,"output", tid, stid)
        if not osp.exists(save_path):
            os.makedirs(save_path)
        res[method]= run_backdoor(model, modelpath, dataname, method, pp_poison=inputParam[method]["pp_poison"], source=inputParam[method]["source"], target=inputParam[method]["target"],trigger=inputParam[method]["trigger"], device=inputParam["device"], save_path=save_path)
        logging.info("[执行后门攻击]:{:s}后门攻击运行结束，投毒率为{}时，攻击成功率为{}%".format(method, inputParam[method]["pp_poison"], res[method]["attack_success_rate"]))
    res["stop"] = 1
    IOtool.write_json(res, osp.join(ROOT,"output", tid, stid+"_result.json"))
    taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
    taskinfo[tid]["function"][stid]["state"] = 2
    IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
    IOtool.change_task_success_v2(tid)

def run_dim_reduct(tid, stid, datasetparam, modelparam, vis_methods, adv_methods, device):
    """降维可视化
    :params tid:主任务ID
    :params stid:子任务id
    :params datasetparam:数据集参数
    :params modelparam:模型参数
    :params vis_methods:list，降维方法
    :params adv_methods:list，对抗攻击方法
    :params device:GPU
    """
    taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
    taskinfo[tid]["function"][stid]["state"]=1
    taskinfo[tid]["state"]=1
    IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
    params = {
        "dataset": datasetparam,
        "model": modelparam,
        "out_path": osp.join(ROOT,"output", tid),
        "device": torch.device(device),
        "adv_methods":{"methods":adv_methods},
        "root":ROOT
    }
    logging = Logger(filename=osp.join(ROOT,"output", tid, stid +"_log.txt"))

    root = ROOT
    dataset = datasetparam["name"]
    nor_data = torch.load(osp.join(root, f"dataset/data/{dataset}_NOR.pt"))
    nor_loader = get_loader(nor_data, batchsize=16)
    logging.info("[数据集获取]：获取{:s}数据集正常样本已完成.".format(dataset))

    model_name = modelparam["name"]
    if modelparam["ckpt"] != "None":
        model = torch.load(modelparam["ckpt"])
    else:
        modelparam["ckpt"] = None
        model = modelparam["ckpt"]
    logging.info("[加载被解释模型]：准备加载被解释模型{:s}".format(model_name))
    net = load_model(model_name, dataset, device, root, reference_model=model, logging=logging)
    net = net.eval().to(device)
    logging.info("[加载被解释模型]：被解释模型{:s}已加载完成".format(model_name))

    adv_loader = {}
    for adv_method in adv_methods:
        adv_loader[adv_method] = get_adv_loader(net, nor_loader, adv_method, params, batchsize=16, logging=logging)
    logging.info("[数据集获取]：获取{:s}对抗样本已完成".format(dataset))

    save_path = osp.join(ROOT,"output", tid, stid)
    if not osp.exists(save_path):
        os.mkdir(save_path)
    res = {}
    for adv_method in adv_methods:
        temp = dim_reduciton_visualize(vis_methods, nor_loader, adv_loader[adv_method], net, model_name, dataset, device, save_path)
        res[adv_method] = temp
    
    IOtool.write_json(res,osp.join(ROOT,"output", tid, stid+"_result.json")) 
    taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
    taskinfo[tid]["function"][stid]["state"] = 2
    IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
    IOtool.change_task_success_v2(tid)
    
def run_attrbution_analysis(tid, stid, datasetparam, modelparam, ex_methods, adv_methods, device):
    """对抗攻击归因解释
    :params tid:主任务ID
    :params stid:子任务id
    :params datasetparam:数据集参数
    :params modelparam:模型参数
    :params ex_methods:list，攻击解释方法
    :params adv_methods:list，对抗攻击方法
    :params device:GPU
    """
    taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
    taskinfo[tid]["function"][stid]["state"]=1
    taskinfo[tid]["state"]=1
    IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
    params = {
        "dataset": datasetparam,
        "model": modelparam,
        "out_path": osp.join(ROOT,"output", tid, stid),
        "device": torch.device(device),
        "ex_methods":{"methods":ex_methods},
        "adv_methods":{"methods":adv_methods},
        "root":ROOT,
        "stid":stid
    }
    logging = Logger(filename=osp.join(ROOT,"output", tid, stid +"_log.txt"))

    root = ROOT
    dataset = datasetparam["name"]
    nor_data = torch.load(osp.join(root, f"dataset/data/{dataset}_NOR.pt"))
    nor_loader = get_loader(nor_data, batchsize=16)
    logging.info("[数据集获取]：获取{:s}数据集正常样本已完成.".format(dataset))

    model_name = modelparam["name"]
    if modelparam["ckpt"] != "None":
        model = torch.load(modelparam["ckpt"])
    else:
        modelparam["ckpt"] = None
        model = modelparam["ckpt"]
    logging.info("[加载被解释模型]：准备加载被解释模型{:s}".format(model_name))
    net = load_model(model_name, dataset, device, root, reference_model=model, logging=logging)
    net = net.eval().to(device)
    logging.info("[加载被解释模型]：被解释模型{:s}已加载完成".format(model_name))

    adv_loader = {}
    for adv_method in adv_methods:
        adv_loader[adv_method] = get_adv_loader(net, nor_loader, adv_method, params, batchsize=16, logging=logging)
    logging.info("[数据集获取]：获取{:s}对抗样本已完成".format(dataset))

    save_path = osp.join(ROOT,"output", tid, stid)
    if not osp.exists(save_path):
        os.mkdir(save_path)
    
    logging.info("[注意力分布图计算]：选择了{:s}解释算法".format(", ".join(ex_methods)))
    ex_images = attribution_maps(net, nor_loader, adv_loader, ex_methods, params, 20, logging)
    IOtool.write_json(ex_images,osp.join(ROOT,"output", tid, stid+"_result.json")) 
    taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
    taskinfo[tid]["function"][stid]["state"] = 2
    IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
    IOtool.change_task_success_v2(tid)


def run_layer_explain(tid, stid, datasetparam, modelparam, ex_methods, adv_methods, device):
    """模型内部解释
    :params tid:主任务ID
    :params stid:子任务id
    :params datasetparam:数据集参数
    :params modelparam:模型参数
    :params adv_methods:list，对抗攻击方法
    :params device:GPU
    """
    taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
    taskinfo[tid]["function"][stid]["state"]=1
    taskinfo[tid]["state"]=1
    IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
    params = {
        "dataset": datasetparam,
        "model": modelparam,
        "out_path": osp.join(ROOT,"output", tid),
        "device": torch.device(device),
        "adv_methods":{"methods":adv_methods},
        "ex_methods":ex_methods,
        "root":ROOT,
        "stid":stid
    }
    logging = Logger(filename=osp.join(ROOT,"output", tid, stid +"_log.txt"))

    root = ROOT
    dataset = datasetparam["name"]
    nor_data = torch.load(osp.join(root, f"dataset/data/{dataset}_NOR.pt"))
    nor_loader = get_loader(nor_data, batchsize=16)
    logging.info("[数据集获取]：获取{:s}数据集正常样本已完成.".format(dataset))
    
    model_name = modelparam["name"]
    if modelparam["ckpt"] != "None":
        model = torch.load(modelparam["ckpt"])
    else:
        modelparam["ckpt"] = None
        model = modelparam["ckpt"]
    logging.info("[加载被解释模型]：准备加载被解释模型{:s}".format(model_name))
    net = load_model(model_name, dataset, device, root, reference_model=model, logging=logging)
    net = net.eval().to(device)
    logging.info("[加载被解释模型]：被解释模型{:s}已加载完成".format(model_name))
    
    adv_loader = {}
    res = {}
    for adv_method in adv_methods:
        logging.info("[数据集获取]：获取{:s}对抗样本".format(adv_method))
        adv_loader[adv_method] = get_adv_loader(net, nor_loader, adv_method, params, batchsize=16, logging=logging)
        
        logging.info("[特征层可视化]:对{:s}模型逐层提取特征并进行可视化分析".format(model_name))
        result = layer_explain(model_name, nor_loader, adv_loader[adv_method], params)
        res[adv_method]=result
        
        print(result)
    IOtool.write_json(res, osp.join(ROOT,"output", tid, stid+"_result.json")) 
    taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
    taskinfo[tid]["function"][stid]["state"] = 2
    IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
    IOtool.change_task_success_v2(tid)

def run_lime(tid, stid, datasetparam, modelparam, adv_methods, device):
    """多模态解释
    :params tid:主任务ID
    :params stid:子任务id
    :params datasetparam:数据集参数
    :params modelparam:模型参数
    :params adv_methods:list，对抗攻击方法
    :params device:GPU
    """
    taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
    taskinfo[tid]["function"][stid]["state"]=1
    taskinfo[tid]["state"]=1
    IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
    params = {
        "dataset": datasetparam,
        "model": modelparam,
        "out_path": osp.join(ROOT,"output", tid),
        "device": torch.device(device),
        "adv_methods":{"methods":adv_methods},
        # "ex_methods":ex_methods,
        "root":ROOT,
        "stid":stid
    }
    logging = Logger(filename=osp.join(ROOT,"output", tid, stid +"_log.txt"))

    root = ROOT
    dataset = datasetparam["name"]
    nor_data = torch.load(osp.join(root, f"dataset/data/{dataset}_NOR.pt"))
    # nor_loader = get_loader(nor_data, batchsize=16)
    nor_img_x = nor_data["x"][2]
    label = nor_data['y'][2]
    img = recreate_image(nor_img_x.squeeze())
    logging.info("[数据集获取]：获取{:s}数据集正常样本已完成.".format(dataset))
    
    model_name = modelparam["name"]
    if modelparam["ckpt"] != "None":
        model = torch.load(modelparam["ckpt"])
    else:
        modelparam["ckpt"] = None
        model = modelparam["ckpt"]
    logging.info("[加载被解释模型]：准备加载被解释模型{:s}".format(model_name))
    net = load_model(model_name, dataset, device, root, reference_model=model, logging=logging)
    net = net.eval().to(device)
    logging.info("[加载被解释模型]：被解释模型{:s}已加载完成".format(model_name))
    
    adv_loader = {}
    res = {}
    for adv_method in adv_methods:
        logging.info("[数据集获取]：获取{:s}对抗样本".format(adv_method))
        adv_img_x = sample_untargeted_attack(dataset, adv_methods[0], net, nor_img_x, label, device, root)
        logging.info("[数据集获取]：获取{:s}对抗样本已完成".format(adv_method))

        save_path = params["out_path"]
        result = lime_image_ex(img, net, model_name, dataset, device, root, save_path)
        
        res[adv_method]=result
    IOtool.write_json(res, osp.join(ROOT,"output", tid, stid+"_result.json")) 
    taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
    taskinfo[tid]["function"][stid]["state"] = 2
    IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
    IOtool.change_task_success_v2(tid)

def verify_img(tid, stid, net, dataset, eps, pic_path):
    b1=[]
    b2=[]
    model=''
    if net=='cnn_7layer_bn':
        model='cnn_7layer_bn'
    elif net=='Densenet' and dataset=='CIFAR':
        model='Densenet_cifar_32'
    elif net=='Resnet' and dataset=='MNIST':
        model='resnet'
    elif net=='Resnet' and dataset=='CIFAR':
        model='ResNeXt_cifar'
    elif net=='Wide Resnet' and dataset=='CIFAR':
        model='wide_resnet_cifar_bn_wo_pooling'
    print(model,net)
    lb1,ub1,lb2,ub2,predicted,score_IBP,score_CROWN=auto_verify_img(net, dataset, eps, pic_path)
    categories=[]
    for i in range(len(lb1)):
        b1.append([round(lb1[i],4),round(ub1[i],4)])
        b2.append([round(lb2[i],4),round(ub2[i],4)])
        categories.append(f'f_{i}')
    resp={'boundary1':b1,'boundary2':b2,'categories':categories,'predicted':predicted,
          'score_IBP':score_IBP,'score_CROWN':score_CROWN}
    IOtool.write_json(resp, osp.join(ROOT,"output", tid, stid+"_result.json")) 
    taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
    taskinfo[tid]["function"][stid]["state"] = 2
    IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
    IOtool.change_task_success_v2(tid)
    return resp

def knowledge_consistency(tid, stid, arch,dataset,img_path,layer):
    base=os.path.join(os.getcwd(),"model","ckpt")
    base_path=os.path.join(base,'knowledge_consistency_checkpoints')
    if arch=='vgg16_bn' and dataset=='mnist':
        conv_layer=30
        model_path1=os.path.join(base_path,'checkpoint_mnist_vgg16_bn_lr-2_sd0.pth.tar')
        model_path2=os.path.join(base_path,'checkpoint_mnist_vgg16_bn_lr-2_sd5.pth.tar')
        resume=os.path.join(base_path,f'{dataset}_{arch}','checkpoint_L30__a0.1_lr-4.pth.tar')
    elif arch=='vgg16_bn' and dataset=='cifar10':
        conv_layer=30
        model_path1=os.path.join(base_path,'checkpoint_cifar10_vgg16_bn_lr-2_sd0.pth.tar')
        model_path2=os.path.join(base_path,'checkpoint_cifar10_vgg16_bn_lr-2_sd5.pth.tar')
        resume=os.path.join(base_path,f'{dataset}_{arch}','checkpoint_L30__a0.1_lr-4.pth.tar')
    elif arch=='resnet18' and dataset=='mnist':
        conv_layer=3
        model_path1=os.path.join(base_path,'checkpoint_mnist_resnet18_lr-2_sd0.pth.tar')
        model_path2=os.path.join(base_path,'checkpoint_mnist_resnet18_lr-2_sd5.pth.tar')
        resume=os.path.join(base_path,f'{dataset}_{arch}','checkpoint_L3__a0.1_lr-4.pth.tar')
    elif arch=='resnet18' and dataset=='cifar10':
        conv_layer=3
        model_path1=os.path.join(base_path,'checkpoint_cifar10_resnet18_lr-2_sd0.pth.tar')
        model_path2=os.path.join(base_path,'checkpoint_cifar10_resnet18_lr-2_sd5.pth.tar')
        resume=os.path.join(base_path,f'{dataset}_{arch}','checkpoint_L3__a0.1_lr-3.pth.tar')
    else:
        print(arch,dataset)
        return None
    if dataset=='mnist':
        transform=transforms.Compose([transforms.Resize((224,224)),transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(), 
            transforms.Normalize((0.1307,), (0.3081,))])
    elif dataset=='cifar10':
        transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(), 
                                  transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])
    net1 = models.__dict__[arch](num_classes=10)
    load_checkpoint(model_path1, net1)
    net2 = models.__dict__[arch](num_classes=10)
    load_checkpoint(model_path2, net2)
    img = Image.open(img_path)
    img = img.convert('RGB')

    x_ori = transform(img)
    y = net1(x_ori.unsqueeze(0))
    y = np.argmax(y.cpu().detach().numpy())
    x = get_feature(x_ori, net1,arch,conv_layer)
    input_size = x.shape
    output_size = x.shape
    model = models.LinearTester(input_size,output_size, affine=False, bn = False, instance_bn=True).cuda()
    checkpoint = torch.load(resume, map_location=torch.device(0))
    model.load_state_dict(checkpoint['state_dict'])
    del checkpoint
    input = get_feature(x_ori,net1,arch,conv_layer)
    target = get_feature(x_ori,net2,arch,conv_layer)
    model.eval()
    output, output_n, output_contrib, res = model.val_linearity(input.unsqueeze(0).cuda())
    t=0
    img_name=os.path.basename(img_path).split('.')[0]
    pic_dir=os.path.dirname(img_path)
    delta = target - output
    # for t in range(len(output)):
    t=layer
    plt.figure(frameon=False)
    plt.axis('off')
    plt.imshow(output[t], cmap='jet', norm=None, vmin=output.min(), vmax=output.max())
    plt.savefig(os.path.join(pic_dir,stid+f'_output_{t}.png'),bbox_inches='tight')
    plt.close()
    plt.figure(frameon=False)
    plt.axis('off')
    plt.imshow(input[t], cmap='jet', norm=None, vmin=input.min(), vmax=input.max())
    plt.savefig(os.path.join(pic_dir,stid+f'_input_{t}.png'),bbox_inches='tight')
    plt.close()
    plt.figure(frameon=False)
    plt.axis('off')
    plt.imshow(target[t], cmap='jet', norm=None, vmin=target.min(), vmax=target.max())
    plt.savefig(os.path.join(pic_dir,stid+f'_target_{t}.png'),bbox_inches='tight')
    plt.close()
    plt.figure(frameon=False)
    plt.axis('off')
    plt.imshow(delta[t], cmap='jet', norm=None, vmin=delta.min(), vmax=delta.max())
    plt.savefig(os.path.join(pic_dir,stid+f'_delta_{t}.png'),bbox_inches='tight')
    plt.close()
    return torch.norm(delta, p=2).numpy().tolist(),len(target)
def reach(tid,stid,dataset,pic_path,label,target_label):
    base=os.path.join(os.getcwd(),"model","ckpt")
    base_path=os.path.join(base,'reach_checkpoints')
    model = reachNet()
    if dataset=='CIFAR10':  
        model.load_state_dict(torch.load(os.path.join(base_path,'cifar_torch_net.pth'), map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(os.path.join(base_path,'mnist_torch_net.pth'), map_location=torch.device('cpu')))
    model.eval()
    transf = transforms.ToTensor()
    img = cv2.imread(pic_path)
    image=transf(img)
    image=torch.unsqueeze(image, 0)
    label=torch.tensor(int(label))
    target_label=torch.tensor(int(target_label))
    # [image, label, target_label, _] = torch.load("/mnt/data/ai/veritex/examples/CIFAR10/data/images/0.pt")

    attack_block = (1,1)
    epsilon = 0.02
    relaxation = 0.01
    reach_model = ReachMethod(model, image, label, 'logs',
                         attack_block=attack_block,
                         epsilon=epsilon,
                         relaxation=relaxation)
    output_sets = reach_model.reach()

    sims = reach_model.simulate(num=100)

    # Plot output reachable sets and simulations
    dim0, dim1 = label.numpy(), target_label.numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for item in output_sets:
        out_vertices = item[0]
        plot_polytope2d(out_vertices[:, [dim0, dim1]], ax, color='b', alpha=1.0, edgecolor='k', linewidth=0.0,zorder=2)
    ax.plot(sims[:,dim0], sims[:,dim1],'k.',zorder=1)
    ax.autoscale()
    plt.tight_layout()
    plt.show()
    pt_dir=os.path.dirname(pic_path)
    plt.savefig(os.path.join(pt_dir,stid+'.png'))
    plt.close()
    resp={
        
    }
    return resp

def detect(adv_dataset, adv_model, adv_method, adv_nums, defense_methods, adv_examples=None, logging=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
    if torch.cuda.is_available():
        print("got GPU")
    logging.info("加载模型{:s}".format(adv_model))
    if adv_dataset == 'CIFAR10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        if adv_model == 'ResNet18':
            model = ResNet18()
            checkpoint = torch.load(osp.join(ROOT,'model/model-cifar-resnet18/model-res-epoch85.pt'))
        elif adv_model == 'VGG16':
            model = vgg16()
            model.classifier[6] = nn.Linear(4096, 10)
            checkpoint = torch.load(osp.join(ROOT,'model/model-cifar-vgg16/model-vgg16-epoch85.pt'))
        else:
            raise Exception('CIFAR10 can only use ResNet18 and VGG16!')
        model.load_state_dict(checkpoint)
        model = model.to(device)
    elif adv_dataset == 'MNIST':
        mean = (0.1307,)
        std = (0.3081,)
        if adv_model == 'SmallCNN':
            model = SmallCNN()
            checkpoint = torch.load(osp.join(ROOT,'model/model-mnist-smallCNN/model-nn-epoch61.pt'))
        elif adv_model == 'VGG16':
            model = vgg16()
            model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
            model.classifier[6] = nn.Linear(4096, 10)
            checkpoint = torch.load(osp.join(ROOT,'model/model-mnist-vgg16/model-vgg16-epoch32.pt'))
        else:
            raise Exception('MNIST can only use SmallCNN and VGG16!')
        model.load_state_dict(checkpoint)
        model = model.to(device).eval()    
    logging.info("{:s}模型加载结束".format(adv_model))
    if defense_methods == 'JPEG':
        detector =  Jpeg(model, mean, std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'Feature Squeeze':
        detector =  feature_squeeze(model, mean, std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'Twis':
        detector =  Twis(model, mean, std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'Rgioned-based':
        detector =  RegionBased(model, mean, std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'Pixel Deflection':
        detector =  Pixel_Deflection(model, mean, std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'Label Smoothing':
        detector =  Label_smoothing(model, mean, std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'Spatial Smoothing':
        detector =  Spatial_smoothing(model, mean, std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'Gaussian Data Augmentation':
        detector =  Gaussian_augmentation(model, mean, std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'Total Variance Minimization':
        detector =  Total_var_min(model, mean, std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'Pixel Defend':
        detector =  Pixel_defend(model, mean, std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'InverseGAN':
        detector =  Inverse_gan(model, mean, std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'DefenseGAN':
        detector =  Defense_gan(model, mean, std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'Madry':
        detector =  Madry(model, mean, std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'FastAT':
        detector =  FastAT(model, mean, std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'TRADES':
        detector =  Trades(model, mean, std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'FreeAT':
        detector =  FreeAT(model, mean, std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'MART':
        detector =  Mart(model, mean, std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'CARTL':
        detector =  Cartl(model, mean, std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'Activation':
        detector =  Activation_defence(model, mean, std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'Spectral Signature':
        detector =  Spectral_signature(model, mean, std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'Provenance':
        detector =  Provenance_defense(model, mean, std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'Neural Cleanse L1':
        detector =  Neural_cleanse_l1(model, mean, std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'Neural Cleanse L2':
        detector =  Neural_cleanse_l2(model, mean, std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'Neural Cleanse Linf':
        detector =  Neural_cleanse_linf(model, mean, std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)
    elif defense_methods == 'SAGE':
        detector =  Sage(model, mean, std, adv_examples=adv_examples, adv_method=adv_method, adv_dataset=adv_dataset, adv_nums=adv_nums, device=device)

    _, _, detect_rate, no_defense_accuracy = detector.detect()
    return detect_rate, no_defense_accuracy


def run_side_api(trs_file, methods, tid, stid):
    
    taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
    taskinfo[tid]["function"][stid]["state"]=1
    taskinfo[tid]["state"]=1
    IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
    logging = Logger(filename=osp.join(ROOT,"output", tid, stid +"_log.txt"))
    logging.info("开始执行侧信道分析")
    res={}
    for method in methods:
        logging.info("当前分析文件为{:s}，分析方法为{:s}，分析时间较久，约需50分钟".format(trs_file, method))
        outpath = osp.join(ROOT,"output", tid,stid + "_" + method+"_out.txt")
        trs_file_path = osp.join(ROOT,"dataset/Trs/samples",trs_file)
        use_time = run_side(trs_file_path, method, outpath)
        if method in ["cpa","dpa"]:
            for line in open(outpath, 'r'):
                res[method] = [float(s) for s in line.split()]
        else:
            # 其他方法结果处理
            pass 
        logging.info("分析方法{:s}执行结束，耗时{:s}s".format(method,str(round(use_time,1))))
    logging.info("侧信道分析执行结束！")
    IOtool.write_json(res, osp.join(ROOT,"output", tid, stid+"_result.json")) 
    taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
    taskinfo[tid]["function"][stid]["state"] = 2
    IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
    IOtool.change_task_success_v2(tid)
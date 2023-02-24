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
import gol
from function.formal_verify import *
 
from function.fairness import api

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
    taskparam.set_res_value(key = "summary",value = summary)
    # adv_result["summary"] = summary
    trainerParam = {
        "lr": 0.05,
        "optim": "SGD",
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "robustness": [],
        "epochs": 100
    }
    taskparam.set_params_value(key="trainer",value=trainerParam)
    trainer = Trainer(**trainerParam)
    trainer.update_config(arch=model_name, task=data_name)
    adv_result = taskparam.get_result()
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
    return model,trainer

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
                                    'output_path': 'static/output',
                                    'task_id': f"{param['task_id']}"}}
    global result
    result = verify(input_param)
    result["stop"] = 1
    IOtool.write_json(result,osp.join(ROOT,"output", tid, AAtid+"_result.json"))
    taskinfo[tid]["function"][AAtid]["state"]=2
    taskinfo[tid]["state"]=2
    IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
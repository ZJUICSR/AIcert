import os
import os.path as osp
import torch
import torchvision

from function.ex_methods.module.func import get_loader, Logger, recreate_image, get_batchsize
from function.ex_methods.module.generate_adv import get_adv_loader, sample_untargeted_attack
from function.ex_methods.module.load_model import load_model
from function.ex_methods import attribution_maps, layer_explain, dim_reduciton_visualize
from function.ex_methods.module.model_Lenet import lenet
from function.ex_methods.lime import lime_image_ex

from IOtool import IOtool, Callback
ROOT = osp.dirname(osp.abspath(__file__))

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
    model_name = modelparam["name"]

    batchsize = get_batchsize(model_name,dataset)
    nor_data = torch.load(osp.join(root, f"dataset/data/{dataset}_NOR.pt"))
    nor_loader = get_loader(nor_data, batchsize=batchsize)
    logging.info("[数据集获取]：获取{:s}数据集正常样本已完成.".format(dataset))

    model = modelparam["ckpt"]
    logging.info("[加载被解释模型]：准备加载被解释模型{:s}".format(model_name))
    net = load_model(model_name, dataset, device, root, reference_model=model, logging=logging)
    net = net.eval().to(device)
    logging.info("[加载被解释模型]：被解释模型{:s}已加载完成".format(model_name))

    adv_loader = {}
    for adv_method in adv_methods:
        adv_loader[adv_method] = get_adv_loader(net, nor_loader, adv_method, params, batchsize=batchsize, logging=logging)
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
    
def run_attrbution_analysis(tid, stid, datasetparam, modelparam, ex_methods, adv_methods, device, use_layer_explain):
    """对抗攻击归因解释
    :params tid:主任务ID
    :params stid:子任务id
    :params datasetparam:数据集参数
    :params modelparam:模型参数
    :params ex_methods:list，攻击解释方法
    :params adv_methods:list，对抗攻击方法
    :params device:GPU
    :params use_layer_explain: bool, 是否使用层间解释分析方法
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
    result = {}
    img_num = 20
    dataset = datasetparam["name"]
    model_name = modelparam["name"]

    batchsize = get_batchsize(model_name, dataset)
    nor_data = torch.load(osp.join(root, f"dataset/data/{dataset}_NOR.pt"))
    nor_loader = get_loader(nor_data, batchsize=batchsize)
    logging.info("[数据集获取]：获取{:s}数据集正常样本已完成.".format(dataset))

    # ckpt参数 直接存储模型object，不存储模型路径；可以直接带入load_model函数中，该函数会自动根据输入作相应处理
    model = modelparam["ckpt"]
    logging.info("[加载被解释模型]：准备加载被解释模型{:s}".format(model_name))
    net = load_model(model_name, dataset, device, root, reference_model=model, logging=logging)
    net = net.eval().to(device)
    logging.info("[加载被解释模型]：被解释模型{:s}已加载完成".format(model_name))

    adv_loader = {}
    for adv_method in adv_methods:
        adv_loader[adv_method] = get_adv_loader(net, nor_loader, adv_method, params, batchsize=batchsize, logging=logging)
    logging.info("[数据集获取]：获取{:s}对抗样本已完成".format(dataset))

    save_path = osp.join(ROOT,"output", tid, stid)
    if not osp.exists(save_path):
        os.mkdir(save_path)
    
    logging.info("[注意力分布图计算]：选择了{:s}解释算法".format(", ".join(ex_methods)))
    ex_images = attribution_maps(net, nor_loader, adv_loader, ex_methods, params, img_num, logging)
    result.update({"adv_ex":ex_images})

    if use_layer_explain == True:
        logging.info("[已选择执行模型层间解释]：正在执行...")
        layer_ex = layer_explain(net, model_name, nor_loader, adv_loader, dataset, params["out_path"], device, img_num, logging)
        result.update({"layer_ex": layer_ex})
        logging.info("[已选择执行模型层间解释]：层间解释执行完成")
    else:
        logging.info("[未选择执行模型层间解释]：将不执行模型层间解释分析方法")

    IOtool.write_json(result,osp.join(ROOT,"output", tid, stid+"_result.json")) 
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
        
        print(result)
    IOtool.write_json(res, osp.join(ROOT,"output", tid, stid+"_result.json")) 
    taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
    taskinfo[tid]["function"][stid]["state"] = 2
    IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
    IOtool.change_task_success_v2(tid)

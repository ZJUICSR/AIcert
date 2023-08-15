import os, base64
import os.path as osp
import torch
import torchvision
import time
from function.ex_methods.module.func import get_loader, Logger, get_batchsize,load_image, predict, get_class_list
from function.ex_methods.module.generate_adv import get_adv_loader, sample_untargeted_attack
from function.ex_methods.module.load_model import load_model, load_torch_model
from function.ex_methods import attribution_maps, layer_explain, dim_reduciton_visualize
from function.ex_methods.module.model_Lenet import lenet
from function.ex_methods.lime import lime_image_ex
from PIL import Image
from IOtool import IOtool, Callback
ROOT = osp.dirname(osp.abspath(__file__))

def run_dim_reduct(tid, stid, datasetparam, modelparam, vis_methods, adv_methods):
    """降维可视化
    :params tid:主任务ID
    :params stid:子任务id
    :params datasetparam:数据集参数
    :params modelparam:模型参数
    :params vis_methods:list，降维方法
    :params adv_methods:list，对抗攻击方法
    :params device:GPU
    """
    IOtool.change_subtask_state(tid, stid, 1)
    IOtool.change_task_state(tid, 1)
    IOtool.set_task_starttime(tid, stid, time.time())
    logging = IOtool.get_logger(stid)
    device = IOtool.get_device()
    params = {
        "dataset": datasetparam,
        "model": modelparam,
        "out_path": osp.join(ROOT,"output", tid),
        "device": torch.device(device),
        "adv_methods":{"methods":adv_methods},
        "root":ROOT
    }
    root = ROOT
    dataset = datasetparam["name"]
    model_name = modelparam["name"]

    batchsize = get_batchsize(model_name,dataset)
    nor_data = torch.load(osp.join(root, f"dataset/data/{dataset}_NOR.pt"))
    nor_loader = get_loader(nor_data, batchsize=batchsize)
    logging.info( "[数据集获取]：获取{:s}数据集正常样本已完成.".format(dataset))

    model = modelparam["ckpt"]
    logging.info( "[加载被解释模型]：准备加载被解释模型{:s}".format(model_name))
    net = load_model(model_name, dataset, device, root, reference_model=model, logging=logging)
    net = net.eval().to(device)
    logging.info( "[加载被解释模型]：被解释模型{:s}已加载完成".format(model_name))

    adv_loader = {}
    for adv_method in adv_methods:
        adv_loader[adv_method] = get_adv_loader(net, nor_loader, adv_method, params, batchsize=batchsize, logging=logging)
    logging.info( "[数据集获取]：获取{:s}对抗样本已完成".format(dataset))

    save_path = osp.join(ROOT,"output", tid, stid)
    if not osp.exists(save_path):
        os.mkdir(save_path)
    res = {}
    for adv_method in adv_methods:
        temp = dim_reduciton_visualize(vis_methods, nor_loader, adv_loader[adv_method], net, model_name, dataset, device, save_path)
        res[adv_method] = temp
        logging.info( "[数据分布降维解释]：{:s}对抗样本数据分布降维解释已完成".format(adv_method))
    res["stop"] = 1
    IOtool.write_json(res, osp.join(ROOT,"output", tid, stid+"_result.json")) 
    print("interfase modify sub task state:",tid, stid)
    IOtool.change_subtask_state(tid, stid, 2)
    IOtool.change_task_success_v2(tid)
    
def run_attrbution_analysis(tid, stid, datasetparam, modelparam, ex_methods, adv_methods, use_layer_explain):
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
    IOtool.change_subtask_state(tid, stid, 1)
    IOtool.change_task_state(tid, 1)
    IOtool.set_task_starttime(tid, stid, time.time())
    device = IOtool.get_device()
    logging = IOtool.get_logger(stid)
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
    logging.info( "[加载被解释模型]：准备加载被解释模型{:s}".format(model_name))
    net = load_model(model_name, dataset, device, root, reference_model=model, logging=logging)
    net = net.eval().to(device)
    logging.info( "[加载被解释模型]：被解释模型{:s}已加载完成".format(model_name))

    adv_loader = {}
    for adv_method in adv_methods:
        adv_loader[adv_method] = get_adv_loader(net, nor_loader, adv_method, params, batchsize=batchsize, logging=logging)
    logging.info( "[数据集获取]：获取{:s}对抗样本已完成".format(dataset))

    save_path = osp.join(ROOT,"output", tid, stid)
    if not osp.exists(save_path):
        os.mkdir(save_path)
    
    logging.info( "[注意力分布图计算]：选择了{:s}解释算法".format(", ".join(ex_methods)))
    ex_images = attribution_maps(net, nor_loader, adv_loader, ex_methods, params, img_num, logging)
    result.update({"adv_ex":ex_images})

    if use_layer_explain == True:
        logging.info( "[已选择执行模型层间解释]：正在执行...")
        layer_ex = layer_explain(net, model_name, nor_loader, adv_loader, dataset, params["out_path"], device, img_num, logging)
        result.update({"layer_ex": layer_ex})
        logging.info( "[已选择执行模型层间解释]：层间解释执行完成")
    else:
        logging.info( "[未选择执行模型层间解释]：将不执行模型层间解释分析方法")
    result["stop"] = 1
    IOtool.write_json(result, osp.join(ROOT,"output", tid, stid+"_result.json")) 
    print("interfase modify sub task state:",tid, stid)
    IOtool.change_subtask_state(tid, stid, 2)
    IOtool.change_task_success_v2(tid)

def run_adversarial_analysis(tid, stid, datasetparam, modelparam, ex_methods, vis_methods, adv_methods, use_layer_explain):
    """对抗攻击特征归因和降维解释集成接口
    :params tid:主任务ID
    :params stid:子任务id
    :params datasetparam:数据集参数
    :params modelparam:模型参数
    :params ex_methods:list, 特征归因解释方法
    :params vis_methods: list, 特征降维解释方法
    :params adv_methods:list, 对抗攻击方法
    :params device:GPU
    :params use_layer_explain: bool, 是否使用层间解释分析方法
    """
    IOtool.change_subtask_state(tid, stid, 1)
    IOtool.change_task_state(tid, 1)
    IOtool.set_task_starttime(tid, stid, time.time())
    device = IOtool.get_device()
    logging = IOtool.get_logger(stid)
    params = {
        "dataset": datasetparam,
        "model": modelparam,
        "out_path": osp.join(ROOT,"output", tid, stid),
        "device": torch.device(device),
        "ex_methods":{"methods":ex_methods},
        "vis_methods":{"methods":vis_methods},
        "adv_methods":{"methods":adv_methods},
        "root":ROOT,
        "stid":stid
    }

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
    logging.info( "[加载被解释模型]：准备加载被解释模型{:s}".format(model_name))
    net = load_model(model_name, dataset, device, root, reference_model=model, logging=logging)
    net = net.eval().to(device)
    logging.info( "[加载被解释模型]：被解释模型{:s}已加载完成".format(model_name))

    adv_loader = {}
    for adv_method in adv_methods:
        adv_loader[adv_method] = get_adv_loader(net, nor_loader, adv_method, params, batchsize=batchsize, logging=logging)
    logging.info( "[数据集获取]：获取{:s}对抗样本已完成".format(dataset))

    save_path = osp.join(ROOT,"output", tid, stid)
    if not osp.exists(save_path):
        os.mkdir(save_path)
    
    # 如果选择了归因方法
    if len(ex_methods) != 0:
        logging.info("已选择了特征归因解释算法，即将执行...")
        logging.info( "[注意力分布图计算]：选择了{:s}解释算法".format(", ".join(ex_methods)))
        ex_images = attribution_maps(net, nor_loader, adv_loader, ex_methods, params, img_num, logging)
        result.update({"adv_ex":ex_images})

        if use_layer_explain == True:
            logging.info( "[已选择执行模型层间解释]：正在执行...")
            layer_ex = layer_explain(net, model_name, nor_loader, adv_loader, dataset, params["out_path"], device, img_num, logging)
            result.update({"layer_ex": layer_ex})
            logging.info( "[已选择执行模型层间解释]：层间解释执行完成")
        else:
            logging.info( "[未选择执行模型层间解释]：将不执行模型层间解释分析方法")
    else:
        logging.info("当前未选择特征归因解释算法，跳过执行...")
    
    # 如果选择了降维方法
    if len(vis_methods) != 0:
        logging.info("已选择了特征降维解释算法，即将执行...")
        result.update({"dim_ex":{}})
        for adv_method in adv_methods:
            temp = dim_reduciton_visualize(vis_methods, nor_loader, adv_loader[adv_method], net, model_name, dataset, device, save_path)
            result["dim_ex"][adv_method] = temp
            logging.info( "[数据分布降维解释]：{:s}对抗样本数据分布降维解释已完成".format(adv_method))
    else:
        logging.info("当前未选择特征降维解释算法，跳过执行...")

    logging.info("[执行完成] 当前攻击机理分析功能已全部完成！")
    result["stop"] = 1
    IOtool.write_json(result, osp.join(ROOT,"output", tid, stid+"_result.json")) 
    print("interfase modify sub task state:",tid, stid)
    IOtool.change_subtask_state(tid, stid, 2)
    IOtool.change_task_success_v2(tid)


def run_lime(tid, stid, datasetparam, modelparam, adv_methods, mode):
    """多模态解释
    :params tid:主任务ID
    :params stid:子任务id
    :params datasetparam:数据集参数
    :params modelparam:模型参数
    :params adv_methods:list, 对抗攻击方法
    :params mode:数据模型信息:str, "text" or "image"
    """
    IOtool.change_subtask_state(tid, stid, 1)
    IOtool.change_task_state(tid, 1)
    IOtool.set_task_starttime(tid, stid, time.time())
    device = IOtool.get_device()
    logging = IOtool.get_logger(stid)
    params = {
        "dataset": datasetparam,
        "model": modelparam,
        "out_path": osp.join(ROOT,"output", tid),
        "device": torch.device(device),
        "adv_methods":{"methods":adv_methods},
        "mode":mode,
        "root":ROOT,
        "stid":stid
    }

    logging = Logger(filename=osp.join(ROOT,"output", tid, stid +"_log.txt"))
    root = ROOT
    dataset = datasetparam["name"]

    if mode == "image":
        logging.info("[数据集获取]：目前数据模态为图片类型")
        image = datasetparam['data']
        img_dir=os.path.join(os.getcwd(),"web/static/img/tmp_imgs")
        if not os.path.exists(img_dir):
            os.mkdir(img_dir)
        pic_path=os.path.join(img_dir,tid,stid+'.png')
        try:
            os.mkdir(os.path.join(img_dir,tid))
        except:
            pass
        if "image/jpeg;" in image:
            with open( pic_path, 'wb') as f:
                f.write(base64.b64decode(image.replace('data:image/jpeg;base64,','')))
                f.close()
        else:
            with open( pic_path, 'wb') as f:
                f.write(base64.b64decode(image.replace('data:image/png;base64,','')))
                f.close()
        img = Image.open(pic_path).convert('RGB')
    else:
        logging.info("[数据集获取]：目前数据模态为文本类型")


    logging.info("[数据集获取]：获取{:s}数据集正常样本已完成.".format(dataset))
    
    model_name = modelparam["name"]
    if modelparam["ckpt"] != "None":
        model = torch.load(modelparam["ckpt"])
    else:
        modelparam["ckpt"] = None
        model = modelparam["ckpt"]
    logging.info("[加载被解释模型]：准备加载被解释模型{:s}".format(model_name))
    net = load_torch_model(model_name)
    net = net.eval().to(device)
    img_x = load_image(device, img, dataset)
    label, _ = predict(net, img_x)
    logging.info("[加载被解释模型]：被解释模型{:s}已加载完成".format(model_name))
    
    adv_loader = {}
    res = {}
    class_list = get_class_list(dataset, root)
    class_name = 

    for adv_method in adv_methods:
        logging.info("[数据集获取]：获取图像{:s}对抗样本".format(adv_method))
        adv_img = sample_untargeted_attack(dataset, adv_method, net, img_x, label, device, root)
        logging.info("[数据集获取]：获取图像{:s}对抗样本已完成".format(adv_method))

        save_path = params["out_path"]
        result = lime_image_ex(img, net, model_name, dataset, device, root, save_path)
        
        res[adv_method]=result
        
        print(result)
    IOtool.write_json(res, osp.join(ROOT,"output", tid, stid+"_result.json")) 
    taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
    taskinfo[tid]["function"][stid]["state"] = 2
    IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
    IOtool.change_task_success_v2(tid)

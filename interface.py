import os
import os.path as osp
import numpy as np
import torch
import torchvision
import time
import textattack
from function.ex_methods.module.func import get_loader, get_batchsize,load_image, predict, get_class_list, get_normalize_para
from function.ex_methods.module.generate_adv import get_adv_loader, sample_untargeted_attack, text_attack
from function.ex_methods.module.load_model import load_model, load_torch_model, load_text_model
from function.ex_methods import attribution_maps, layer_explain, dim_reduciton_visualize
from function.ex_methods.lime import lime_image_ex, lime_text_ex
from scipy.stats import kendalltau
from PIL import Image
from IOtool import IOtool, Callback
ROOT = osp.dirname(osp.abspath(__file__))

"""
因为3.2.4版本的torchattack会在每个对抗攻击函数的最后将图片归一化到（0,1）之间，
这意味着torchattack默认的输入图片是不经过Normalization的，
但该版本也没有设置过内部的normalization过程，因此需要在模型的开头加入一个自定义的Normal layer（继承Module类）
所以在构造对抗样本时输入的图片是没有经过Normalization的，否则图片会失真
"""

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
        "out_path": osp.join(ROOT,"output", tid, stid),
        "device": torch.device(device),
        "adv_methods":{"methods":adv_methods},
        "mode":mode,
        "root":ROOT,
        "stid":stid
    }

    root = ROOT
    dataset = datasetparam["name"]
    model_name = modelparam["name"]
    model = modelparam["ckpt"]
    res = {}

    if mode == "image":
        logging.info("[数据集获取]：目前数据模态为图片类型")
    
        save_dir = os.path.join('./dataset/data/ckpt/upload_lime.jpg')
        if os.path.exists(save_dir):
            image = Image.open(save_dir) 

            if dataset == "mnist":
                img = image.convert("L")
            img = image.convert('RGB')
        logging.info("[数据集获取]：获取{:s}数据样本已完成.".format(dataset))

        logging.info("[加载被解释模型]：准备加载被解释模型{:s}".format(model_name))
        net = load_torch_model(model_name)
        net = net.eval().to(device)

        img_x = load_image(device, img, dataset)
        label, _ = predict(net, img_x)
        logging.info("[加载被解释模型]：被解释模型{:s}已加载完成".format(model_name))

        # 将图片tensor去标准化，之后再输入torchattack中
        mean, std = get_normalize_para(dataset)
        re_trans = torchvision.transforms.Normalize(
                mean=tuple(-m / s for m, s in zip(mean, std)),
                std=tuple(1.0 / s for s in std),
            )
        img_x = re_trans(img_x)

        class_list = get_class_list(dataset, root)
        class_name = class_list[label.item()]

        save_path = params["out_path"]
        logging.info("[LIME图像解释]：正在对正常图片数据进行解释")
        res["adv_methods"] = adv_methods
        nor_img_lime, img_url, img_lime_url = lime_image_ex(img, net, model_name, dataset, device, save_path)
        res["nor"] = {"image":img_url,"ex_image":img_lime_url,"class_name":class_name,"kendalltau":"不适用"}
        logging.info("[LIME图像解释]：正常图片数据解释已完成")
        for adv_method in adv_methods:
            logging.info("[数据集获取]：正在生成图像{:s}对抗样本".format(adv_method))
            adv_img, adv_label = sample_untargeted_attack(dataset, adv_method, net, img_x, label, device, root)
            logging.info("[数据集获取]：图像{:s}对抗样本已生成".format(adv_method))
            class_name = class_list[adv_label.item()]
            save_path = params["out_path"]
            logging.info("[LIME图像解释]：正在对{:s}图片数据进行解释".format(adv_method))
            adv_img_lime, img_url, img_lime_url = lime_image_ex(adv_img, net, model_name, dataset, device, save_path, imagetype=adv_method)
            kendall_value, _ = kendalltau(np.array(nor_img_lime,dtype="float64"),np.array(adv_img_lime,dtype="float64"))
            kendall_value = round(kendall_value, 4)
            res[adv_method] = {"image":img_url, "ex_image":img_lime_url,"class_name":class_name,"kendalltau":kendall_value}
            logging.info("[LIME图像解释]：{:s}图片数据解释已完成".format(adv_method))
        logging.info("[LIME图像解释]：LIME解释已全部完成")
    else:
        # 数据为文本模态
        logging.info("[数据集获取]：目前数据模态为文本类型")
        text = datasetparam['data']
        res["adv_methods"] = adv_methods
        logging.info("[数据集获取]：获取{:s}数据样本已完成.".format(dataset))

        logging.info("[加载模型中]：正在加载{:s}模型.".format(model_name))
        model = load_text_model(model_name)
        tokenizer = model.tokenizer
        model_wrapper = textattack.models.wrappers.PyTorchModelWrapper(model, tokenizer)
        class_names = ["negitive", "positive"]
        logging.info("[加载模型中]：{:s}模型已加载完成.".format(model_name))

        logging.info("[LIME文本解释]：正在对正常文本数据进行解释.")
        ex_nor = lime_text_ex(model_wrapper, text, class_names=class_names)
        logging.info("[LIME文本解释]：正常文本数据解释分析完成.")
        res['nor'] = {"class_names": ex_nor[0], 
            "predictions":ex_nor[1], 
            "regression_result":ex_nor[2], 
            "explain_res":ex_nor[3], 
            "raw_js":ex_nor[4]
            }
        logging.info("[LIME文本解释]：正在对对抗文本数据进行解释.")
        for adv_method in adv_methods:
            logging.info(f"[LIME文本对抗攻击]：采用{adv_method}对抗攻击方法生成对抗样本中...")
            adv_text = text_attack(model_wrapper, adv_method, text)
            ex_adv = lime_text_ex(model_wrapper, adv_text, class_names=class_names)
            logging.info(f"[LIME文本解释]：{adv_method}对抗文本攻击数据解释分析完成.")
            res[adv_method] = {"class_names": ex_adv[0], 
                "predictions":ex_adv[1], 
                "regression_result":ex_adv[2], 
                "explain_res":ex_adv[3], 
                "raw_js":ex_adv[4]
                }
        logging.info("[LIME文本解释]：LIME解释已全部完成")
    res["stop"] = 1  
    IOtool.write_json(res, osp.join(ROOT,"output", tid, stid+"_result.json")) 
    print("interfase modify sub task state:", tid, stid)
    IOtool.change_subtask_state(tid, stid, 2)
    IOtool.change_task_success_v2(tid)

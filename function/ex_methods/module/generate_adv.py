import warnings
warnings.filterwarnings('ignore')
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
import torchattacks as attacks
from function.ex_methods.module.func import get_loader, get_normalize_para, recreate_image
import os.path as osp
import os
import json

def get_targeted_loader(data):
    x = torch.cat(data["x"],dim=0).float().cpu()
    adv_dst = TensorDataset(x ,torch.full([x.size(0)], data["y"]).long().cpu())
    adv_loader = DataLoader(
        adv_dst,
        batch_size=16,
        shuffle=False,
        num_workers=2
    )
    return adv_loader


def check_params(params, keys):
    """
    Filter useless kwargs.
    Args:
        params: dict
        keys: list

    Returns:
        params: dict
    """
    _params = {}
    for k, v in params.items():
        if k in keys:
            _params[k] = v
    return _params


def get_attack(dataset, method, model, params):
    adv_params = check_params(params[dataset][method], params[dataset][method].keys())
    atk = eval("attacks.{:s}".format(method))(model, **adv_params)
    return atk
        
def get_accuracy(dataloader,model,device,mean,std):
    num = 0
    for i, (x,y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)
        activation = model.forward(transforms.Normalize(mean=mean,std=std)(x))
        _, y_ = torch.max(activation,1)
        num_correct = (y_== y).sum().item()
        num += num_correct
    return num/len(dataloader.dataset)

def sample_untargeted_attack(dataset, method, model, img_x, label, device, root):
    with open(osp.join(root, "function/ex_methods/cache/adv_params.json")) as fp:
        def_params = json.load(fp)
    attack = get_attack(dataset, method, model, def_params)
    img_x = img_x.to(device)
    label = label.to(device)
    adv_img_x = attack(img_x, label)
    adv_img = recreate_image(adv_img_x, dataset)
    return adv_img

def targeted_attack(dataset, method, model, data_loader, targeted_label, device, params, mean, std):
    adv_x = []
    ben_x = []
    ben_y = []
    attack = get_attack(dataset, method, model, params)
    attack.set_normalization_used(mean=mean, std=std)
    attack.set_mode_targeted_by_function(target_map_function=lambda x, y: torch.full(y.size(),targeted_label).to(y.device))
    print("当前执行{:s}对抗方法生成对抗样本...".format(method))
    for step, (x, y) in enumerate(data_loader):
        x = x.to(device)
        y = y.to(device)
        z = attack(x, y)
        ben_x.append(x.cpu())
        adv_x.append(z.cpu())
        ben_y.append(y.cpu())

    data = {
        "x": adv_x,
        "y": targeted_label
    }

    adv_loader = get_targeted_loader(data)
    adv_acc = get_accuracy(adv_loader,model,device,mean,std)
    print(f"{method}对抗样本targeted-class:{targeted_label}的攻击成功率为：{adv_acc}")
    return data


def untargeted_attack(dataset, method, model, data_loader, device, params, mean, std, logging):
    adv_x = []
    ben_x = []
    ben_y = []
    attack = get_attack(dataset, method, model, params)
    # attack.set_normalization_used(mean=mean, std=std)
    logging.info("[生成对抗样本]：当前执行{:s}对抗方法生成对抗样本...".format(method))
    for step, (x, y) in enumerate(data_loader):
        x = x.to(device)
        y = y.to(device)
        z = attack(x, y)
        ben_x.append(x.cpu())
        adv_x.append(z.cpu())
        ben_y.append(y.cpu())

    data = {
        "x": adv_x,
        "y": ben_y
    }

    adv_loader = get_loader(data)
    adv_acc = get_accuracy(adv_loader, model, device, mean, std)
    logging.info(f"[生成对抗样本]：{method}对抗样本untargeted的攻击成功率为：{1-adv_acc}")
    return data


'''加载对抗样本'''
def get_adv_loader(model, dataloader, method, param, batchsize, logging):
    dataset = param["dataset"]["name"]
    model_name = param["model"]["name"]
    device = param["device"]
    root = param["root"]
    
    model.eval().to(device)
    with open(osp.join(root, "function/ex_methods/cache/adv_params.json")) as fp:
        def_params = json.load(fp)
    
    if "eps" not in def_params[dataset][method].keys():
        eps = 0.1
    else:
        eps = def_params[dataset][method]["eps"]

    if "steps" not in def_params[dataset][method].keys():
        steps = 1
    else:
        steps = def_params[dataset][method]["steps"]

    save_root = osp.join(root,f"dataset/adv_data")
    if not osp.exists(save_root):
        os.makedirs(save_root)
    path = osp.join(save_root, "adv_{:s}_{:s}_{:s}_{:04d}_{:.5f}.pt".format(
                            method, model_name, dataset, steps, eps))

    mean, std = get_normalize_para(dataset)

    if not osp.exists(path):
        logging.info("[加载数据集]：未检测到缓存的{:s}对抗样本，将重新执行攻击算法并缓存".format(method))
        data = untargeted_attack(dataset, method, model, dataloader, device, def_params, mean, std, logging)
        torch.save(data, path)
    else:
        logging.info("[加载数据集]：检测到缓存{:s}对抗样本，直接加载缓存文件".format(method))
        data = torch.load(path)

    adv_loader = get_loader(data, batchsize)
    
    return adv_loader
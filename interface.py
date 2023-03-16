import os.path as osp
import torch
import torchvision

from function.ex_methods.module.func import get_loader, Logger
from function.ex_methods.module.generate_adv import get_adv_loader
from function.ex_methods.module.load_model import load_model
from function.ex_methods import attribution_maps, layer_explain, dim_reduciton_visualize
from function.ex_methods.module.model_Lenet import lenet

ROOT = osp.dirname(osp.abspath(__file__))

"""对抗样本加载计算，绘制解释图接口"""
# if __name__ == '__main__':
#     # m = lenet(None,1)
#     # m.load_state_dict(torch.load("./model/ckpt/mnist_lenet.pt"))

#     params = {
#         "dataset": {"name": "imagenet"},
#         "model": {"name": "densenet121","ckpt":None},
#         "out_path": "./output",
#         "device": torch.device("cuda:1"),
#         "ex_methods":{"methods":["lrp", "gradcam", "integrated_grad"]},
#         "adv_methods":{"methods":["FGSM"]},
#         "root":ROOT
#     }

#     logging = Logger(filename=osp.join(params["out_path"], "kt1_logs.txt"))

#     root = params["root"]
#     dataset = params["dataset"]["name"]
#     nor_data = torch.load(osp.join(root, f"dataset/{dataset}/data/{dataset}_NOR.pt"))
#     nor_loader = get_loader(nor_data, batchsize=16)
#     logging.info("[数据集获取]：获取{:s}数据集正常样本已完成.".format(dataset))

#     model_name = params["model"]["name"]
#     device = params["device"]
#     model = params["model"]["ckpt"]
#     logging.info("[加载被解释模型]：准备加载被解释模型{:s}".format(model_name))
#     net = load_model(model_name, dataset, device, root, reference_model=model, logging=logging)
#     net = net.eval().to(device)
#     logging.info("[加载被解释模型]：被解释模型{:s}已加载完成".format(model_name))

#     adv_loader = {}
#     adv_methods = params["adv_methods"]["methods"]
#     for adv_method in adv_methods:
#         adv_loader[adv_method] = get_adv_loader(net, nor_loader, adv_method, params, batchsize=16, logging=logging)
#     logging.info("[数据集获取]：获取{:s}对抗样本已完成".format(dataset))

#     ex_methods = params["ex_methods"]["methods"]
#     logging.info("[注意力分布图计算]：选择了{:s}解释算法".format(", ".join(ex_methods)))
#     ex_images = attribution_maps(net, nor_loader, adv_loader, ex_methods, params, 20, logging)

"""模型每层特征图可视化，支持vgg和alexnet"""
# if __name__ == '__main__':
#     params = {
#         "dataset": {"name": "imagenet"},
#         "model": {"name": "vgg19"},
#         "out_path": "./output",
#         "device": torch.device("cuda:6"),
#         "ex_methods":{"dataset":"ImageNet"},
#         "root": ROOT
#     }
#     logging = Logger(filename=osp.join(params["out_path"], "kt1_logs.txt"))

#     dataset = params["dataset"]["name"]

#     nor_loader = get_loader(osp.join(ROOT,"dataset",dataset,"data"))
#     logging.info("[数据集获取]：获取正常样本已完成.")

#     adv_loader = {}
#     adv_dataloader_path = osp.join(ROOT,"imagenet_adv_data")
#     for data in os.listdir(adv_dataloader_path):
#         adv_method = data.split("_")[1]
#         adv_loader[adv_method] = get_loader(osp.join(adv_dataloader_path, data))
#     logging.info("[获取数据集]：获取对抗样本已完成")

#     # 执行卷积层解释运算
#     model_name = params["model"]["name"]
#     logging.info("[特征层可视化]:对{:s}模型逐层提取特征并进行可视化分析".format(model_name))
#     layer_explain(model_name, nor_loader, adv_loader["BIM"], params)


if __name__ == "__main__":

    params = {
        "dataset": {"name": "imagenet"},
        "model": {"name": "densenet121","ckpt":None},
        "out_path": "./output",
        "device": torch.device("cuda:4"),
        "ex_methods":{"methods":["lrp", "gradcam", "integrated_grad"]},
        "adv_methods":{"methods":["FGSM"]},
        "root":ROOT
    }

    logging = Logger(filename=osp.join(params["out_path"], "kt1_logs.txt"))

    root = params["root"]
    dataset = params["dataset"]["name"]
    nor_data = torch.load(osp.join(root, f"dataset/{dataset}/data/{dataset}_NOR.pt"))
    nor_loader = get_loader(nor_data, batchsize=16)
    logging.info("[数据集获取]：获取{:s}数据集正常样本已完成.".format(dataset))

    model_name = params["model"]["name"]
    device = params["device"]
    model = params["model"]["ckpt"]
    logging.info("[加载被解释模型]：准备加载被解释模型{:s}".format(model_name))
    net = load_model(model_name, dataset, device, root, reference_model=model, logging=logging)
    # net = torchvision.models.inception_v3(num_classes=10)
    net = net.eval().to(device)
    logging.info("[加载被解释模型]：被解释模型{:s}已加载完成".format(model_name))

    adv_loader = {}
    adv_methods = params["adv_methods"]["methods"]
    for adv_method in adv_methods:
        adv_loader[adv_method] = get_adv_loader(net, nor_loader, adv_method, params, batchsize=16, logging=logging)
    logging.info("[数据集获取]：获取{:s}对抗样本已完成".format(dataset))

    save_path = params["out_path"]
    vis_type_list = ['pca', 'ss', 'tsne', 'svm', 'mean_diff']
    dim_reduciton_visualize(vis_type_list, nor_loader, adv_loader["FGSM"], net, model_name, dataset, device, save_path)
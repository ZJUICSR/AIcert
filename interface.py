import os
import os.path as osp
import torch
import torchvision.models as models
from function.ex_methods.module.func import get_loader, Logger
from function.ex_methods.module.generate_adv import get_adv_loader
from function.ex_methods import attribution_maps, layer_explain

ROOT = osp.dirname(osp.abspath(__file__))

"""mnist对抗样本加载计算，绘制解释图接口"""
if __name__ == '__main__':
    params = {
        "dataset": {"name": "cifar10"},
        "model": {"name": "vgg16","model_ckpt":models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)},
        "out_path": "./output",
        "device": torch.device("cuda:1"),
        "ex_methods":{"methods":["lrp", "gradcam", "integrated_grad"]},
        "adv_methods":{"methods":["FGSM","PGD"]},
        "root":ROOT
    }

    logging = Logger(filename=osp.join(params["out_path"], "kt1_logs.txt"))

    root = params["root"]
    dataset = params["dataset"]["name"]
    nor_data = torch.load(osp.join(root, f"dataset/{dataset}/data/{dataset}_NOR.pt"))
    nor_loader = get_loader(nor_data, batchsize=16)
    logging.info("[数据集获取]：获取{:s}数据集正常样本已完成.".format(dataset))

    adv_loader = {}
    adv_methods = params["adv_methods"]["methods"]
    for adv_method in adv_methods:
        adv_loader[adv_method] = get_adv_loader(nor_loader, adv_method, params, batchsize=16, logging=logging)
    logging.info("[数据集获取]：获取{:s}对抗样本已完成".format(dataset))

    ex_methods = params["ex_methods"]["methods"]
    logging.info("[注意力分布图计算]：选择了{:s}解释算法".format(", ".join(ex_methods)))

    reference_model = params["model"]["model_ckpt"]
    ex_images = attribution_maps(reference_model, nor_loader, adv_loader, ex_methods, params, 20, logging)


"""cifar10对抗样本加载计算，绘制解释图接口"""
# if __name__ == '__main__':
#     params = {
#         "dataset": {"name": "cifar10"},
#         "model": {"name": "resnet18"},
#         "out_path": "./output",
#         "device": torch.device("cuda:6"),
#         "ex_methods":{"dataset":"cifar10"},
#         "root":ROOT
#     }
#     logging = Logger(filename=osp.join(params["out_path"], "kt1_logs.txt"))
#     nor_loader = get_loader(osp.join(ROOT,"cifar10_NOR.pt"), batchsize=128)
#     logging.info("[数据集获取]：获取正常样本已完成.")

#     adv_loader = {}
#     adv_dataloader_path = osp.join(ROOT,"cifar10_adv_data")
#     for data in os.listdir(adv_dataloader_path):
#         adv_method = data.split(".")[0].split("_")[-1]
#         adv_loader[adv_method] = get_loader(osp.join(adv_dataloader_path, data), batchsize=128)
#     logging.info("[获取数据集]：获取对抗样本已完成")

#     ex_methods = ["gradcam", "integrated_grad"]
#     logging.info("[选择的解释算法]:{:s}".format(", ".join(ex_methods)))

#     model_path = osp.join(ROOT,"models","cifar10_resnet18.pt")
#     reference_model = torch.load(model_path,map_location=params['device'])
#     ex_images = run(reference_model, nor_loader, adv_loader, ex_methods, params, 20, logging)

"""ImageNet对抗样本加载计算，绘制解释图接口"""
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

#     nor_loader = get_loader(osp.join(ROOT,"imagenet_NOR.pt"))
#     logging.info("[数据集获取]：获取正常样本已完成.")

#     adv_loader = {}
#     adv_dataloader_path = osp.join(ROOT,"imagenet_adv_data")
#     for data in os.listdir(adv_dataloader_path):
#         adv_method = data.split("_")[1]
#         adv_loader[adv_method] = get_loader(osp.join(adv_dataloader_path, data))
#     logging.info("[获取数据集]：获取对抗样本已完成")

#     ex_methods = ["lrp", "gradcam", "integrated_grad"]
#     logging.info("[选择的解释算法]:{:s}".format(", ".join(ex_methods)))

#     reference_model = None
#     # ex_images = run(reference_model, nor_loader, adv_loader, ex_methods, params, 20, logging)
#     # 执行卷积层解释运算
#     model_name = params["model"]["name"]
#     layer_explain(model_name, nor_loader, adv_loader["BIM"], params)

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
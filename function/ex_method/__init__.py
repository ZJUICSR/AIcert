from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fileinput import filename

import json
import os
import os.path as osp
import copy
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
from skimage.segmentation import mark_boundaries
from skimage.color import rgb2gray

from .lime import lime_image
from .module.load_model import load_model
from .module.func import grad_visualize, lrp_visualize, load_image, target_layer, preprocess_transform
ROOT = osp.dirname(osp.abspath(__file__))


def get_class_list(dataset):
    if dataset == 'imagenet':
        imagenet_path = osp.join(ROOT, "imagenet_class_index.json")
        with open(os.path.abspath(imagenet_path), 'r') as read_file:
            class_idx = json.load(read_file)
            idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
            return idx2label
    elif dataset == 'mnist':
        return ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    elif dataset == 'cifar10':
        return ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    else:
        return "暂不支持其他数据集"


def get_target_num(dataset):
    if dataset == "imagenet":
        return 1000
    else:
        return 10


def predict(x, net, device):
    net = net.eval().to(device)
    
    activation_output = net.forward(x)
    _, prediction = torch.max(activation_output, 1)

    return prediction, activation_output


def draw_grad(prediction, x, activation_output, net, dataset, model, device):
    result_lrp = net.interpretation(activation_output,
                                    interpreter="integrated_grad",
                                    labels=prediction,
                                    num_target=get_target_num(dataset),
                                    device=device,
                                    target_layer=target_layer(model,"imagenet"),
                                    inputs=x)
    result_cam = net.interpretation(activation_output,
                                    interpreter="grad_cam",
                                    labels=prediction,
                                    num_target=get_target_num(dataset),
                                    device=device,
                                    target_layer=target_layer(model,dataset),
                                    inputs=x)
    x = x.permute(0, 2, 3, 1).cpu().detach().numpy()
    x = x - x.min(axis=(1, 2, 3), keepdims=True)
    x = x / x.max(axis=(1, 2, 3), keepdims=True)

    img_l = lrp_visualize(result_lrp, 0.9)[0]

    img_h = grad_visualize(result_cam, x)[0]

    img_l = Image.fromarray((img_l * 255).astype(np.uint8))
    img_h = Image.fromarray((img_h * 255).astype(np.uint8))
    return img_l, img_h


def batch_predict(images, model, device, dataset):
    """
    lime 中对随机取样图像进行预测
    :param images: np.array
    :param model:
    :param device:
    :return:
    """
    if dataset == "mnist":
        images = rgb2gray(images)
    batch = torch.stack(tuple(preprocess_transform(i, dataset) for i in images), dim=0)
    batch = batch.to(device).type(dtype=torch.float32)
    probs = model.forward(batch)
    return probs.detach().cpu().numpy()


def draw_lime(img, net, device, dataset):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(np.array(img),
                                             net,
                                             device,
                                             dataset,
                                             batch_predict,  # classification function
                                             top_labels=5,
                                             hide_color=0,
                                             num_samples=1000)  # number of images that will be sent to classification function

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, negative_only=False,
                                                num_features=5, hide_rest=False)
    img_boundry = mark_boundaries(temp / 255.0, mask)
    img_boundry = Image.fromarray((img_boundry * 255).astype(np.uint8))
    return img_boundry


def get_explain(imgs, net, dataset, device, model):
    class_list = get_class_list(dataset)
    ex_images = {}
    i = 0
    for img in imgs:
        if dataset == "mnist":
            img = img.convert("L")
        x = load_image(device, img, dataset)
        prediction, activation_output = predict(x, net, device)
        class_name = class_list[prediction.item()]
        img_l, img_h = draw_grad(prediction, x, activation_output, net, dataset, model, device)
        img_lime = draw_lime(img, net, device, dataset)
        ex_images[f"image_{i}"] = {"class_name": class_name, "ex_imgs": [img_l, img_h, img_lime]}
        i += 1
    return ex_images


def recreate_image(im_as_var, dataset):
    """
        Recreates images from a torch variable, sort of reverse preprocessing
    Args:
        im_as_var (torch variable): Image to recreate
    returns:
        recreated_im (numpy arr): Recreated image in array
    """
    reverse_mean = []
    reverse_std = []
    if dataset == "cifar10":
        reverse_mean = [-0.43768206, -0.44376972, -0.47280434]
        reverse_std = [1 / 0.19803014, 1 / 0.20101564, 1 / 0.19703615]
    elif dataset == "mnist":
        reverse_mean = [(-0.1307)]
        reverse_std = [(1 / 0.3081)]
    elif dataset == "imagenet":
        reverse_mean = [-0.485, -0.456, -0.406]
        reverse_std = [1 / 0.2023, 1 / 0.1994, 1 / 0.2010]
    recreated_im = copy.copy(im_as_var.data.cpu().numpy())
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    return recreated_im


def save_ex_img(root, img, img_name):
    path = osp.join(root, "adv_explain", img_name)
    if not os.path.exists(root+'/adv_explain/'):
        os.makedirs(root+'/adv_explain/')
    img.save(path)
    return path.split("/")[-3]+ "/" +path.split("/")[-2] + "/" + path.split("/")[-1]


def loader2imagelist(dataloader, dataset,size):
    image_list = []
    for i in range(size):
        img, label = dataloader.dataset[i]
        if dataset == "cifar10":
            img_numpy = recreate_image(img, dataset)
            img_x = Image.fromarray(img_numpy)
            image_list.append(img_x)
        elif dataset == "imagenet":
            img_x = torchvision.transforms.ToPILImage()(img)
            image_list.append(img_x)
    return image_list


from torch.utils.data import TensorDataset
def get_imagenet_cache(methods):
    adv_dataloaders = {}
    for method in methods:
        path = osp.join(ROOT, "cache/{:s}.pt".format(method))
        adv_dataloaders[method] = get_loader(path)
    path = osp.join(ROOT, "cache/NOR.pt")
    test_loader = get_loader(path)
    return test_loader, adv_dataloaders


def run(model, test_loader, adv_dataloader, params, log_func, size=30):
    """
    :param model: pytorch模型类型
    :param test_loader: 正常样本测试集
    :param adv_dataloader: 对抗样本测试集: type:dict
    :param params: 其他参数
    :return: 总接口
    """
    model_name = params["model"]["name"].lower()
    log_func("[模型测试阶段]【指标1.1】课题一当前运行模型为：{:s}".format(str(model_name)))
    dataset = params["ex_methods"]["dataset"].lower()
    use_upload  = params["ex_methods"]["use_upload"].lower()
    nor_dataloader_path = params["ex_methods"]["nor_path"]
    adv_dataloader_path = params["ex_methods"]["adv_path"]
    att_method = params["ex_methods"]["att_method"]
    if use_upload == "true":
        test_loader = get_loader(nor_dataloader_path)
        adv_dataloader = {}
        adv_dataloader[att_method] = get_loader(adv_dataloader_path)
        size = len(test_loader) 
        if dataset == "imagenet":
            model = None
    elif dataset == "imagenet":
        # 系统不跑imagenet，读取已经跑好的adv_dataloaders缓存
        model = None
        test_loader, adv_dataloader = get_imagenet_cache(methods=list(adv_dataloader.keys()))
        
    root = osp.join(params["out_path"],"keti1")
    results = {}
    ex_results = {"methods": []}
    device = params["device"]
    
    log_func("[模型测试阶段]【指标1.1】正在当前执行模型内部重构与解析")
    net = load_model(model_name, pretrained=True, reference_model= model, num_classes=get_target_num(dataset))
    
    nor_img_list = loader2imagelist(test_loader, dataset, size)
    log_func("[模型测试阶段]【指标1.1】正在当前执行正常样本解释")
    ex_images = get_explain(imgs=nor_img_list, net=net, model=model_name, dataset=dataset, device=device)
    nor_ex_list = []
    for i in range(size):
        class_name = ex_images[f"image_{i}"]["class_name"]
        ex_images_ = ex_images[f"image_{i}"]["ex_imgs"]
        nor_ex_list.append(ex_images_)
        nor_f = save_ex_img(root, nor_img_list[i], f"nor_{i}.png")
        nor_l_f = save_ex_img(root, ex_images_[0], f"nor_lrp_{i}.png")
        nor_h_f = save_ex_img(root, ex_images_[1], f"nor_heatmap_{i}.png")
        nor_lime_f = save_ex_img(root, ex_images_[2], f"nor_lime_{i}.png")
        ex_results[f'nor_{i}'] = {
            "nor_classname": class_name,
            "nor_img": nor_f,
            "heatmap_url": nor_h_f,
            "lrp_url": nor_l_f,
            "lime_url": nor_lime_f
        }
    adv_ex_dict = {}
    for method, adv_loader in adv_dataloader.items():
        log_func("[模型测试阶段]【指标1.1】正在当前执行:{:s}对抗样本攻击机理分析".format(str(method)))
        ex_results["methods"].append(method)
        adv_img_list = loader2imagelist(adv_loader, dataset, size)
        ex_images = get_explain(imgs=adv_img_list, net=net, model=model_name, dataset=dataset, device=device)
        ex_results[method] = {}
        adv_ex_dict[method] = []
        # 存储对抗样本解释图
        for i in range(size):
            class_name = ex_images[f"image_{i}"]["class_name"]
            ex_images_ = ex_images[f"image_{i}"]["ex_imgs"]
            adv_ex_dict[method].append(ex_images_)
            adv_f = save_ex_img(root, adv_img_list[i], f"{method}_adv_{i}.png")
            adv_l_f = save_ex_img(root, ex_images_[0], f"{method}_adv_lrp_{i}.png")
            adv_h_f = save_ex_img(root, ex_images_[1], f"{method}_adv_heatmap_{i}.png")
            adv_lime_f = save_ex_img(root, ex_images_[2], f"{method}_adv_lime_{i}.png")
            ex_results[method].update({
                f'adv_{i}': {
                    "adv_classname": class_name,
                    "adv_img": adv_f,
                    "heatmap_url": adv_h_f,
                    "lrp_url": adv_l_f,
                    "lime_url": adv_lime_f
                }
            })
        
    # 计算肯德尔系数
    log_func("[模型测试阶段]【指标1.1】解释分析算法结束，正在进行解释结果评估")
    kendall = get_kendalltau(nor_ex_list,adv_ex_dict,root)
    log_func("[模型测试阶段]【指标1.1】解释结果评估完成")
    
    #执行卷积层解释运算
    log_func("[模型测试阶段] 课题一：正在执行“模型层间解释算法”")
    l_ex = layer_explain(root,test_loader,adv_dataloader,params)
    log_func("[模型测试阶段]课题一：“模型层间解释算法”完成")
    
    results["adv_explain"] = ex_results
    results["kendalltau"] = kendall
    results["layer_explain"] = l_ex
    results["model"] = model_name
    results["dataset"] = dataset
    # json存储课题一所有结果
    save_process_result(root, results, filename="keti1.json")
    return results


from .module.layer_activation_with_guided_backprop import get_all_layer_analysis
def layer_explain(root,loader,adv_loader,params):
    adv_loader = list(adv_loader.values())[0]
    model = torchvision.models.vgg19(pretrained=True)
    result = get_all_layer_analysis(model=model, nor_loader=loader, adv_loader=adv_loader, params=params,root=root)
    return result

# 测试存储json文件
def save_process_result(root, results, filename):
    path = osp.join(root, filename)

    with open(path, "w") as fp:
        json.dump(results, fp, indent=2)
        fp.close()


def get_loader(url):
    data = torch.load(url)
    adv_dst = TensorDataset(data["x"].float().cpu(), data["y"].long().cpu())
    adv_loader = DataLoader(
        adv_dst,
        batch_size=1,
        shuffle=False,
        num_workers=2
    )
    return adv_loader

from scipy.stats import kendalltau
def get_kendalltau(nor_list,adv_dict,root):
    ex_methods = ["lrp","gcam","lime"]
    json_result = {}
    methods = list(adv_dict.keys())
    root = osp.join(root,'kendalltau')
    #三种解释算法
    for i in range(3):
        result = {"name":methods,"data":[]}
        l = []
        for method in methods:
            tmp = []
            for nor_img, adv_img in zip(nor_list,adv_dict[method]):
                value, _ = kendalltau(np.array(nor_img[i],dtype="float64"),np.array(adv_img[i],dtype="float64"))
                tmp.append(value)
            l.append(tmp)
        result["data"] = l
        json_result[ex_methods[i]] = result
    return json_result


"""
如果是cifar输入数据集，调用接口都不变，以下为本地测试时做的测试样子
"""
# if __name__ == '__main__':
#     param = {
#         "dataset": {"name": "cifar10"},
#         "model": {"name": "vgg13"},
#         "out_path": "../data/cifar10/output",
#         "device": torch.device("cuda:0")
#     }
#     loader = get_loader()
#     adv_loader = {
#         "FGSM": loader,
#         "PGD": loader
#     }
#     model_state_dic = torch.load("./CIFAR10_vgg13.pt")
#     net = torchvision.models.vgg13(pretrained=False, num_classes=10)
#     net.load_state_dict(model_state_dic)
#     result = run(model=net, test_loader=loader, adv_dataloader=adv_loader, params=param)
#     save_process_result(results=result, root=param["out_path"])


"""
兼容了以前的之前的接口，因为新的对抗样本也是dataloader。但因为还是用预训练的模型，就还是不需要传模型，只用传param就行
"""
# if __name__ == '__main__':
#     param = {
#         "dataset": {"name": "imagenet"},
#         "model": {"name": "vgg19"},
#         "out_path": "../data/imagenet/output",
#         "device": torch.device("cuda:0")
#     }
#     loader = get_loader("./argp/third_party/evaluator/ex_methods/cache/NOR.pt")
#     adv_loader = {}
#     adv_loader["FGSM"] = get_loader("./fgsm.pt")
#     adv_loader["BIM"] = get_loader("./bim.pt")
#     adv_loader["DIFGSM"] = get_loader("./difgsm.pt")
#     adv_loader["EOTPGD"] = get_loader("./eotpgd.pt")
#     adv_loader["FFGSM"] = get_loader("./ffgsm.pt")
#     adv_loader["MIFGSM"] = get_loader("./mifgsm.pt")
#     adv_loader["PGD"] = get_loader("./pgd.pt")
#     adv_loader["PGDL2"] = get_loader("./pgdl2.pt")
#     adv_loader["RFGSM"] = get_loader("./rfgsm.pt")
#     adv_loader["FAB"] = get_loader("./fab.pt")

#     result = run(model=None, test_loader=loader, adv_dataloader=adv_loader, params=param)
#     save_process_result(results=result, root="../data/imagenet/test")
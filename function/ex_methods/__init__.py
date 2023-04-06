import os.path as osp
import os
import numpy as np
import torch
import torchvision
from PIL import Image

from .module.load_model import load_model
from .module.func import grad_visualize, lrp_visualize, target_layer
from .module.layer_activation_with_guided_backprop import get_all_layer_analysis
from .module.func import get_class_list, get_target_num, grad_visualize, convert_to_grayscale, \
    loader2imagelist, save_ex_img, save_process_result, get_normalize_para, preprocess_transform, predict
from scipy.stats import kendalltau

from .lime import lime_image
from skimage.color import rgb2gray
from skimage.segmentation import mark_boundaries


'''获取基于梯度/反向传播的解释方法'''


def get_gradbase_explain(img_x, net, target_layer, dataset, ex_methods, device, class_list):
    prediction, activation_output = predict(net, img_x)
    class_name = [class_list[i] for i in prediction]

    lrp_list, gradcam_list, ig_list = [], [], []
    # 和pytorch自动求导计算得到的梯度值基本一致，但存在浮点后几位细微区别
    for method in ex_methods:
        if method == "lrp":
            lrp_list = explain_lrp(net, img_x, dataset,
                                   device, prediction, activation_output)
        elif method == "gradcam":
            gradcam_list = explain_gradcam(
                net, img_x, target_layer, dataset, device, prediction, activation_output)
        elif method == "integrated_grad":
            ig_list = explain_ig(net, img_x, dataset, device,
                                 prediction, activation_output)

    return lrp_list, gradcam_list, ig_list, class_name


'''lrp解释算法接口'''


def explain_lrp(net, x, dataset, device, prediction, activation_output):
    result_lrp = net.interpretation(activation_output,
                                    interpreter="lrp",
                                    labels=prediction,
                                    num_target=get_target_num(dataset),
                                    device=device,
                                    target_layer=None,
                                    inputs=x)
    imgs_lrp = lrp_visualize(result_lrp, 0.9)
    img_list = []
    for img in imgs_lrp:
        img_list.append(Image.fromarray((img * 255).astype(np.uint8)))
    return img_list


'''grad-cam解释算法接口'''


def explain_gradcam(net, x, target_layer, dataset, device, prediction, activation_output):
    result_cam = net.interpretation(activation_output,
                                    interpreter="grad_cam",
                                    labels=prediction,
                                    num_target=get_target_num(dataset),
                                    device=device,
                                    target_layer=target_layer,
                                    inputs=x)
    x = x.permute(0, 2, 3, 1).cpu().detach().numpy()
    x = x - x.min(axis=(1, 2, 3), keepdims=True)
    x = x / x.max(axis=(1, 2, 3), keepdims=True)
    imgs_gradcam = grad_visualize(result_cam, x)
    img_list = []
    for img in imgs_gradcam:
        img_list.append(Image.fromarray((img * 255).astype(np.uint8)))
    return img_list


'''积分梯度解释算法接口'''


def explain_ig(net, x, dataset, device, prediction, activation_output):
    result_ig = net.interpretation(activation_output,
                                   interpreter="integrated_grad",
                                   labels=prediction,
                                   num_target=get_target_num(dataset),
                                   device=device,
                                   target_layer=None,
                                   inputs=x)
    img_list = []
    for index in range(len(result_ig)):
        # Convert to grayscale
        result = result_ig[index].cpu().detach().numpy()
        grayscale_integrated_grad = convert_to_grayscale(
            result)[0]  # [0] to remove the first dim
        img_list.append(Image.fromarray(
            (grayscale_integrated_grad * 255).astype(np.uint8)))
    return img_list


'''课题一总接口'''


def attribution_maps(net, nor_loader, adv_dataloader, ex_methods, params, size, logging):
    """
    :param model: pytorch模型类型
    :param test_loader: 正常样本测试集
    :param adv_dataloader: 对抗样本测试集: type:dict
    :param params: 其他参数
    :return: 总接口
    """
    dataset = params["dataset"]["name"].lower()
    model_name = params["model"]["name"].lower()
    # model_name = "resnet34"

    save_path = params["out_path"]
    root = params["root"]
    results = {"nor": {}, "adv_methods": []}
    device = params["device"]

    mean, std = get_normalize_para(dataset)
    trans = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(
            mean=mean, std=std)
    ])

    class_list = get_class_list(dataset, root)

    t_layer = target_layer(model_name, dataset)
    nor_img_list = loader2imagelist(nor_loader, dataset, size)

    logging.info("[执行解释算法]：系统已准备完成，开始运行所选解释算法")
    logging.info("[正常样本分析阶段]：正在对正常样本进行分析")
    lrp_result_list, gradcam_result_list, ig_result_list, class_name_list = [], [], [], []
    for imgs_x, label in nor_loader:
        imgs_x = trans(imgs_x)
        label = label.to(device)
        imgs_x = imgs_x.to(device)
        lrp_list, gradcam_list, ig_list, class_name = get_gradbase_explain(
            imgs_x, net, t_layer, dataset, ex_methods, device, class_list)
        lrp_result_list.extend(lrp_list)
        gradcam_result_list.extend(gradcam_list)
        ig_result_list.extend(ig_list)
        class_name_list.extend(class_name)
    nor_ex_dict = {"lrp": lrp_result_list,
                   "gradcam": gradcam_result_list,
                   "ig": ig_result_list}
    logging.info("[正常样本分析阶段]：正常样本分析阶段已完成")

    # 存储指定size数量的正常图片
    nor_f = []
    for i in range(size):
        nor_f.append(save_ex_img(save_path, nor_img_list[i], f"nor_{i}.png"))
    for ex_name, nor_ex in nor_ex_dict.items():
        nor_ex_f = []
        if len(nor_ex) == 0:
            continue
        for i in range(size):
            nor_ex_f.append(save_ex_img(
                save_path, nor_ex[i], f"nor_{ex_name}_{i}.png"))
        results["nor"].update({
            f'{ex_name}': nor_ex_f
        })
    results["nor"].update({
        "nor_imgs": nor_f,
        "class_name": class_name_list[:size]
    })

    if len(adv_dataloader) != 0:
        adv_ex_dict = {}
        for method, adv_loader in adv_dataloader.items():
            logging.info("[对抗样本分析阶段]：当前执行{:s}对抗样本攻击机理分析".format(str(method)))
            results["adv_methods"].append(method)
            adv_img_list = loader2imagelist(adv_loader, dataset, size)
            lrp_result_list, gradcam_result_list, ig_result_list, class_name_list = [], [], [], []
            for imgs_x, label in adv_loader:
                imgs_x = imgs_x.to(device)
                label = label.to(device)
                lrp_list, gradcam_list, ig_list, class_name = get_gradbase_explain(
                    imgs_x, net, t_layer, dataset, ex_methods, device, class_list)
                lrp_result_list.extend(lrp_list)
                gradcam_result_list.extend(gradcam_list)
                ig_result_list.extend(ig_list)
                class_name_list.extend(class_name)
                results[method] = {}
            adv_ex_dict[method] = {"lrp": lrp_result_list,
                                   "gradcam": gradcam_result_list, "ig": ig_result_list}

            # 存储指定size数量的对抗样本解释图
            adv_f = []
            for i in range(size):
                adv_f.append(save_ex_img(
                    save_path, adv_img_list[i], f"{method}_adv_{i}.png"))
            for ex_name, adv_ex in adv_ex_dict[method].items():
                adv_ex_f = []
                if len(adv_ex) == 0:
                    continue
                for i in range(size):
                    adv_ex_f.append(save_ex_img(
                        save_path, adv_ex[i], f"{method}_adv_{ex_name}_{i}.png"))
                results[method].update({
                    f"{ex_name}": adv_ex_f
                })
            results[method].update({
                "adv_imgs": adv_f,
                "class_name": class_name_list[:size]
            })
        logging.info("[解释结果量化分析阶段]：采用肯德尔相关系数对解释结果进行分析")
        kendalltau_result = get_kendalltau(nor_ex_dict, adv_ex_dict, save_path)
        results["kendalltau"] = kendalltau_result
        logging.info("[解释结果量化分析阶段]：解释结果量化分析已完成")

    logging.info("[算法完成]：注意力分布图绘制完成")
    save_process_result(save_path, results, filename="keti1.json")
    return results


'''获取模型每层内的特征并做可视化'''


def layer_explain(model_name, loader, adv_loader, params):
    dataset = params["ex_methods"]["dataset"].lower()
    save_path = osp.join(params["out_path"], "layer_explain")
    result = get_all_layer_analysis(
        model_name=model_name, nor_loader=loader, adv_loader=adv_loader, dataset=dataset, save_path=save_path)
    save_process_result(root=save_path, results=result,
                        filename="layer_vis_result.json")


"""计算正常样本和对抗样本解释图的肯德尔相关系数"""


def get_kendalltau(nor_dict, adv_dict, save_path):
    ex_methods = list(nor_dict.keys())
    adv_methods = list(adv_dict.keys())
    save_path = osp.join(save_path, 'kendalltau')
    result = {}
    for adv_method in adv_methods:
        values = {}
        for nor_ex, adv_ex, method in zip(nor_dict.values(), adv_dict[adv_method].values(), ex_methods):
            tmp = []
            for nor_ex_img, adv_ex_img in zip(nor_ex, adv_ex):
                value, _ = kendalltau(np.array(nor_ex_img, dtype="float"), np.array(
                    adv_ex_img, dtype="float"))
                tmp.append(value)
            values[f"{method}"] = tmp
        result[f"{adv_method}"] = values
    return result


from function.ex_methods.dim_reduction.ctl import Controller
from function.ex_methods.dim_reduction.draw import draw_contrast

def dim_reduciton_visualize(vis_type_list, nor_loader, adv_loader, model, model_name, dataset, device, save_path):
    # (1) determine your wanted visualization methods and instantiate one controller
    my_pca_ctl = Controller(vis_type_list, device)
    # (2) set model(torch.nn.Module type)
    #         target_layer('' is the default one-the last feature extraction layer)
    #         data_loader(both the benign one and the poisoned one)
    my_pca_ctl.set_model(model)
    my_pca_ctl.set_layer(target_layer(model_name,dataset))
    my_pca_ctl.set_data_loaders(nor_loader, adv_loader)
    # (3) call the 'prepare_feats' function to collect original feature
    my_pca_ctl.prepare_feats()
    # (4) call the get_reduced_feats function to obtain dimensionality-reduced feature and visualize it through 'draw_contrast'
    save_path = osp.join(save_path ,'dim_reduction')
    if not osp.exists(save_path):
        os.makedirs(save_path)
    for vis_type in vis_type_list:
        clean_feat, bad_feat = my_pca_ctl.get_reduced_feats(vis_type)
        draw_contrast(clean_feat, bad_feat, save_path= osp.join(save_path,f'{dataset}_{model_name}_{vis_type}.jpg'), vis_type=vis_type)
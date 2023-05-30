import os.path as osp
import os
import numpy as np
import torch

from PIL import Image

from function.ex_methods.module.func import get_class_list, preprocess_transform, load_image, predict

from function.ex_methods.lime import lime_image
from skimage.color import rgb2gray
from skimage.segmentation import mark_boundaries



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
    batch = torch.stack(tuple(preprocess_transform(i, dataset)
                        for i in images), dim=0)
    batch = batch.to(device).type(dtype=torch.float32)
    probs = model.forward(batch)
    return probs.detach().cpu().numpy()


def explain_lime_image(img, net, device, dataset):
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


def lime_image_ex(img, model, model_name, dataset, device, root, save_path):
    class_list = get_class_list(dataset, root)
    if dataset == "mnist":
        img = img.convert("L")
    x = load_image(device, img, dataset)
    prediction, _ = predict(model, x)
    class_name = class_list[prediction.item()]
    img_lime = explain_lime_image(img, model, device, dataset)
    path = osp.join(save_path, "lime_result")
    if not os.path.exists(path):
        os.makedirs(path)
    save_path = osp.join(path, f"{model_name}_lime.png") 
    img_lime.save(save_path)
    return class_name, img_lime



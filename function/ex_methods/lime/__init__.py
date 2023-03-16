import os.path as osp
import os
import numpy as np
import torch

from PIL import Image

from function.ex_methods.module.func import get_class_list, preprocess_transform
from scipy.stats import kendalltau

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


def lime_image_ex(img, net, dataset, device, model):
    class_list = get_class_list(dataset)
    ex_images = {}
    i = 0
    for img in imgs:
        if dataset == "mnist":
            img = img.convert("L")
        x = load_image(device, img, dataset)
        prediction, activation_output = predict(x, net, device)
        class_name = class_list[prediction.item()]
        img_lime = draw_lime(img, net, device, dataset)
        ex_images[f"image_{i}"] = {"class_name": class_name,
                                   "ex_imgs": [img_l, img_h, img_lime]}
        i += 1
    return ex_images
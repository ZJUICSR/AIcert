import json
import os
from pyexpat import model
from flask import request
from flask import Blueprint, render_template, send_file
from control.mechanism.train import Train
from control.mechanism.explain import get_explain, get_kendalltau
from control.mechanism.fool import fool
from control.mechanism.draw_result import plot_overview
from utils.functional import save_ex_img, clean_ex_img, save_adv_img
import numpy as np
from interface import detect
from werkzeug.utils import secure_filename

api = Blueprint("api", __name__, template_folder="../templates", static_folder="../static")


@api.route("/", methods=["GET", "POST"])
def home():
    return render_template("demo.html")


@api.route("/train", methods=["POST"])
def train():
    if request.method == "POST":
        model_name = request.form.get("model_name")
        conv_k = request.form.get("conv_k")
        conv_stride = request.form.get("conv_stride")
        pool_k = request.form.get("pool_k")
        pool_stride = request.form.get("pool_stride")
        linear_bais = request.form.get("linear_bais")
        loss_func = request.form.get("loss_func")
        optimizer = request.form.get("optimizer")
        learning_rate = request.form.get("learning_rate")
        dataset = request.form.get("dataset")
        conv_k = int(conv_k[0])
        conv_stride = int(conv_stride[0])
        pool_stride = int(pool_stride[0])
        pool_k = int(pool_k[0])
        linear_bais = 1 if linear_bais[0] == "是" else 0
        loss_func = str(loss_func)
        optimizer = str(optimizer)
        learning_rate = float(learning_rate)
        dataset = str(dataset)

        result = Train(model_name=model_name, conv_k=conv_k, conv_stride=conv_stride,
                       pool_stride=pool_stride, pool_k=pool_k, 
                       linear_bais=linear_bais,  loss_func=loss_func,
                       optimizer=optimizer, learning_rate=learning_rate, dataset=dataset)
        print("-> request forward", result)
        return json.dumps({"data":result})


@api.route("/explain", methods=["POST"])
def explain():
    nor_img_f = request.files.get("nor_img")
    adv_img_f = request.files.get("adv_img")
    nor_img = nor_img_f.stream
    adv_img = adv_img_f.stream

    dataset = request.form.get("dataset").lower()
    model = request.form.get("model").lower()
    
    # 正常样本
    nor_classname, ex_nor_imgs = get_explain(nor_img, model, dataset)
    # 对抗样本
    adv_classname, ex_adv_imgs = get_explain(adv_img, model, dataset)

    nor_l_ex_img, nor_h_ex_img, nor_lime_ex_img = ex_nor_imgs[0], ex_nor_imgs[1], ex_nor_imgs[2]
    adv_l_ex_img, adv_h_ex_img, adv_lime_ex_img = ex_adv_imgs[0], ex_adv_imgs[1], ex_adv_imgs[2]

    clean_ex_img()
    # 存储正常样本解释图
    nor_l_f = save_ex_img(nor_l_ex_img, "nor_lrp.png")
    nor_h_f = save_ex_img(nor_h_ex_img, "nor_heatmap.png")
    nor_lime_f = save_ex_img(nor_lime_ex_img, "nor_lime.png")
    # 存储对抗样本解释图
    adv_l_f = save_ex_img(adv_l_ex_img, "adv_lrp.png")
    adv_h_f = save_ex_img(adv_h_ex_img, "adv_heatmap.png")
    adv_lime_f = save_ex_img(adv_lime_ex_img, "adv_lime.png")
    
    kendall = get_kendalltau(ex_nor_imgs,ex_adv_imgs)


    return json.dumps({"nor_result": {"nor_classname": nor_classname,
                                      "heatmap_url": nor_h_f,
                                      "lrp_url": nor_l_f,
                                      "lime_url": nor_lime_f},
                       "adv_result": {"adv_classname": adv_classname,
                                      "heatmap_url": adv_h_f,
                                      "lrp_url": adv_l_f,
                                      "lime_url": adv_lime_f
                                      },
                       "kendalltau":kendall})


@api.route("/download_result", methods=["GET"])
def download_result():
    url = plot_overview()
    return api.send_static_file(url)

@api.route("/download_model", methods=["POST"])
def download_model():
    model_name = request.form.get("model_name")
    url = "models/MNIST"+ model_name + ".pkl"
    return send_file(url)

@api.route("/adverse", methods=["POST"])
def generage_adversary():
    nor_img_f = request.files.get("nor_img")
    nor_img = nor_img_f.stream
    model = request.form.get("model")
    method = request.form.get("method")
    dataset = request.form.get("dataset")
    adv_img = fool(nor_img, model, method, dataset)
    adv_img_f = save_adv_img(adv_img, "adv_img.png")
    return api.send_static_file(adv_img_f)


api = Blueprint("api", __name__, template_folder="../templates", static_folder="../static")


@api.route("/", methods=["GET", "POST"])
def home():
    return render_template("Adv_detect.html")

@api.route("/detect", methods=["POST"])
def Detect():
    adv_dataset = request.form.get("adv_dataset")
    adv_model = request.form.get("adv_model")
    adv_method = request.form.get("adv_method")
    adv_nums = request.form.get("adv_nums")
    # defense_methods = request.form.getlist("defense_methods[]")
    defense_methods_str = request.form.get("defense_methods")
    defense_methods = json.loads(defense_methods_str)
    adv_nums = int(adv_nums)
    if 'adv_examples' in request.files:
        adv_examples = request.files['adv_examples']
        # 获取文件名
        file_name = secure_filename(adv_examples.filename)
        
        # 生成唯一的文件路径
        adv_file_path = "/mnt/data2/yxl/AI-platform/dataset/adv_examples/" + file_name
        # 将对抗样本文件保存到服务器上的指定位置
        adv_examples.save(adv_file_path)
    else:
        adv_file_path = None
    
    detect_rate_dict = {}
    for defense_method in defense_methods:
        detect_rate, no_defense_accuracy = detect(adv_dataset, adv_model, adv_method, adv_nums, defense_method, adv_file_path)
        detect_rate_dict[defense_method] = round(detect_rate, 4)
    no_defense_accuracy_list = no_defense_accuracy.tolist() if isinstance(no_defense_accuracy, np.ndarray) else no_defense_accuracy
    response_data = {
        "detect_rates": detect_rate_dict,
        "no_defense_accuracy": no_defense_accuracy_list
    }
    return json.dumps(response_data)

import json
import numpy as np
from flask import request
from flask import Blueprint, render_template
from interface import detect
from werkzeug.utils import secure_filename

api = Blueprint("api", __name__, template_folder="../templates", static_folder="../static")


@api.route("/", methods=["GET", "POST"])
def home():
    return render_template("Adv_detect.html")

@api.route("/detect", methods=["POST"])
def Detect():
    adv_dataset = request.form.get("adv_dataset")
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
    _, no_defense_accuracy = detect(adv_dataset, adv_method, adv_nums, defense_methods[0], adv_file_path)
    no_defense_accuracy_list = no_defense_accuracy.tolist() if isinstance(no_defense_accuracy, np.ndarray) else no_defense_accuracy
    detect_rate_dict = {}
    for defense_method in defense_methods:
        detect_rate, _ = detect(adv_dataset, adv_method, adv_nums, defense_method, adv_file_path)
        detect_rate_dict[defense_method] = round(detect_rate, 4)
    response_data = {
        "detect_rates": detect_rate_dict,
        "no_defense_accuracy": no_defense_accuracy_list
    }
    return json.dumps(response_data)

import json
from flask import request
from flask import Blueprint, render_template
from control.detect import detect

api = Blueprint("api", __name__, template_folder="../templates", static_folder="../static")


@api.route("/", methods=["GET", "POST"])
def home():
    return render_template("Adv_detect.html")

@api.route("/detect", methods=["POST"])
def Detect():
    adv_dataset = request.form.get("adv_dataset")
    adv_method = request.form.get("adv_method")
    adv_nums = request.form.get("adv_nums")
    defense_methods = request.form.getlist("defense_methods[]")
    adv_nums = int(adv_nums)
    detect_rate_dict = {}
    for defense_method in defense_methods:
        detect_rate = detect(adv_dataset, adv_method, adv_nums, defense_method)
        detect_rate_dict[defense_method] = round(detect_rate, 4)
    return json.dumps({"detect_rates": detect_rate_dict})

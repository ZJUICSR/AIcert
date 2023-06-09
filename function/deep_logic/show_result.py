import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
import json

dict = {
    "deeplogic": "red",
    "deepgini": "orange",
    "random": "black",
    "lsc_ctm": "bisque",
    "dsc": "CornflowerBlue",
    "entropy": "royalblue",
    "tknc_cam": "Gray",
    "mcp": "#e7298a",
    "nac": "#1b9e77",
    "nbc": "#66a61e",
    "snac_ctm": 'royalblue',
    "snac_cam": 'Mediumblue',
    "kmnc_cam": "skyblue"
}

"""
apfd图像绘制
"""


# 获得图片标签,颜色,渐变,图线
def get_label(p):
    p = os.path.splitext(p)[0]
    l = p
    ls = "-"
    al = 0.7
    c = "black"
    m = "v"

    if p.find("deeplogic") > -1:
        l = "DEEPLOGIC"
        al = 1
        c = dict["deeplogic"]
        m = 'o'
    else:
        temp_arr = p.split("_")
        cover_method = temp_arr[0]
        dataset_name = temp_arr[1]
        if temp_arr[2] == "adv":
            is_adv = True
            idx = 1
        else:
            is_adv = False
            idx = 0
        metrics = temp_arr[2 + idx]
        if p.find("LSC") > -1:
            args = "(1000,100)"
        elif p.find("DSC") > -1:
            args = "(1000,2)"
        else:
            args = "({})".format(temp_arr[-1])
        l = "{}".format(metrics.upper())
        if l == "RANDOM":
            ls = "--"
            m = '+'
        else:
            ls = "-"
        c = dict["{}".format(metrics.lower())]
    return l, ls, al, c, m


# 图像绘制
def plot_apfd(data_path, outputdir, use_adv, metrics_arr=None, idx=0,
              cover="all", ):
    def metrics_filter_func(path: str):
        flag = False
        for met in metrics_arr:
            if path.find(met) > -1:
                flag = True
                break
        return flag

    jsondata = {}
    list_file = os.listdir(data_path)
    #print(list_file)
    for i, p in enumerate(metrics_arr):
        for l_file in list_file:
            if l_file.find(p) > -1:
                file_name = l_file
        l, ls, al, c, m = get_label(file_name)
        df_res = pd.read_csv("{}{}".format(data_path, file_name))
        col = df_res.iloc[:, [0]]
        max_num = col.max().values[0]
        col = col.apply(lambda x: x / max_num * 100)
        col2 = np.array(list(range(len(df_res)))) / (len(df_res) - 1) * 100

        step = len(col2) // 11
        x = col2[::step]
        y = col[::step]
        x_smooth = np.linspace(col2.min(), col2.max(), 300)
        y_smooth = make_interp_spline(x, y)(x_smooth)

        plt.plot(x_smooth, y_smooth, label=l, alpha=al, color=c, linestyle=ls)
        step_s = len(x_smooth) // 11
        plt.scatter(x_smooth[::step_s], y_smooth[::step_s], marker=m, color=c)

        jsondata[l+'_test_case_executed']=[float(x) for x in x_smooth[::step_s]]
        jsondata[l + '_fault_detected']=[float(y) for y in y_smooth[::step_s]]
    plt.xlabel("Percentage of test case executed")
    plt.ylabel("Percentage of fault detected")
    plt.legend()
    plt.grid(linestyle='--')
    res_path = "{}{}.pdf".format(outputdir, idx)
    plt.savefig(res_path, bbox_inches='tight')
    res_path = "{}{}.png".format(outputdir, idx)
    plt.savefig(res_path, bbox_inches='tight')
    plt.close()

    with open("{}{}.data".format(outputdir, idx),'w') as f:
        json.dump(jsondata,f)


def show(dataset, modelname, out_path,apfd, logging=None):
    # 参数配置
    params = [
        {
            "use_adv": True,  # 是否使用adv
            "metrics_arr": ["deeplogic"],  # 选择绘制的图线
            "cover": "cam",  # 选择覆盖的方法
            "idx": 1,  # 图片编号
        },

    ]
    dir_list = ["cifar","fashionminist"]#数据集

    input_base_path = out_path+"/apfd_figure_csv"
    output_base_path = out_path+"/fig"
    for dataset_name in dir_list:
        lst = os.listdir(input_base_path + '/' + dataset_name)
        for model_name in lst:  # 遍历每个模型
            inputdir = input_base_path + '/' + dataset_name + "/" + model_name + "/"
            outputdir = output_base_path + "/" + dataset_name + "/" + model_name + "/"
            if not os.path.exists(outputdir):
                os.makedirs(outputdir)
            for param in params:
                use_adv, metrics_arr, cover, idx = param["use_adv"], param["metrics_arr"], param["cover"], param["idx"]
                #print("plot {} {} {}".format(dataset_name, model_name, idx))
                plot_apfd(inputdir, outputdir, use_adv, metrics_arr=metrics_arr, idx=idx, cover=cover)

    json_data = {}
    if modelname=='vgg16':
        json_data['vgg16']=out_path+'/fig/cifar/vgg16/1.png'
        json_data['data'] = out_path + '/fig/cifar/vgg16/1.data'
    elif modelname=='resnet34':
        json_data['resnet34']=out_path+'/fig/cifar/resnet34/1.png'
        json_data['data'] = out_path + '/fig/cifar/resnet34/1.data'
    else:
        json_data['resnet18']=out_path+'/fig/fashionminist/resnet18/1.png'
        json_data['data'] = out_path + '/fig/fashionminist/resnet18/1.data'
    json_data['apfd']=apfd
    print(apfd)
    return json_data

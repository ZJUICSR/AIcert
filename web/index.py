#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os.path as osp
# from function.attack.old.attack import AdvAttack
import interface
import os, json, datetime, pickle, io, ast, time
import pytz,shutil
from function.attack.config import Conf
import argparse
from IOtool import IOtool
from flask import render_template, redirect, url_for, Flask, request, jsonify, send_from_directory
from flask import current_app as abort
from multiprocessing import Process
from gol import Taskparam
from function.fairness import api
from flask_cors import *
import threading
ROOT = os.getcwd()
app = Flask(__name__)
CORS(app, supports_credentials=True)
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(ROOT, 'static'), 'favicon.ico')

@app.route('/', methods=['GET'])
def index():
    if request.method == "GET":
        return render_template("index.html")

@app.route('/index_function_introduction', methods=['GET'])
def index_function_introduction():
    if request.method == "GET":
        return render_template("index_function_introduction.html")

@app.route('/index_task_center', methods=['GET'])
def index_task_center():
    if request.method == "GET":
        return render_template("index_task_center.html")

@app.route('/index_params_1', methods=['GET'])
def index_params_1():
    if request.method == "GET":
        return render_template("index_params_1.html")

@app.route('/index_params_2', methods=['GET'])
def index_params_2():
    if request.method == "GET":
        return render_template("index_params_2.html")


# 鲁棒性增强平台
@app.route('/ModelRobust', methods=['GET'])
def ModelRobust():
    if (request.method == "GET"):
        return render_template("model_robust.html")
    else:
        abort(403)

# 公平性平台
@app.route('/Fairness', methods=['GET'])
def Fairness():
    if (request.method == "GET"):
        return render_template("fairness.html")
    else:
        abort(403)

# 数据集公平性网页
@app.route('/FairnessEva', methods=['GET'])
def DataFairness():
    if (request.method == "GET"):
        return render_template("fairness_eva.html")
    else:
        abort(403)
@app.route('/FairnessDebias', methods=['GET'])
def ModelFairness():
    if (request.method == "GET"):
        return render_template("ModelFairness.html")
    else:
        abort(403)
# ---------------模板：数据集公平性评估---------
@app.route('/DataFairnessEvaluate', methods=['POST'])
def DataFairnessEvaluate():
    """
    数据集公平性评估
    输入：tid：主任务ID
    dataname：数据集名称
    """
    if (request.method == "POST"):
        inputParam = json.loads(request.data)
        dataname = inputParam["dataname"]
        # 获取主任务ID
        tid = inputParam["tid"]
<<<<<<< HEAD
        format_time = str(datetime.datetime.now().strftime("%Y%m%d%H%M"))
=======
>>>>>>> gitee/feature_chunlai
        # 生成子任务ID
        format_time = str(datetime.datetime.now().strftime("%Y%m%d%H%M"))
        stid = "S"+IOtool.get_task_id(str(format_time))
        # 获取任务列表
        taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
        # 添加任务信息到taskinfo
        taskinfo[tid]["function"].update({stid:{
            # 任务类型,注意任务类型不能重复，用于结果返回的key值索引
            "type":"date_evaluate",
            # 任务状态：0 未执行；1 正在执行；2 执行成功；3 执行失败
            "state":0,
            # 方法名称：如对抗攻击中的fgsm，ffgsm等，呈现在结果界面
            "name":["date_evaluate"],
            # 数据集信息，呈现在结果界面，若干有选择模型还需增加模型字段：model
            "dataset":dataname,
        }})
        taskinfo[tid]["dataset"]=dataname
        senAttrList=json.loads(inputParam["senAttrList"])
        tarAttrList=json.loads(inputParam["tarAttrList"])
        staAttrList=json.loads(inputParam["staAttrList"])
        
        # 执行任务，运行时间超过3分钟的请使用多线程，参考DataFairnessDebias函数的执行部分
        res = api.dataset_evaluate(dataname, sensattrs=senAttrList, targetattrs=tarAttrList)
        res1 = api.dataset_analysis(dataname, attrs=staAttrList, targetattrs=staAttrList)
        res['Corelation coefficients'] = res1['Corelation coefficients']
        for key in res1['Proportion'].keys():
            if key not in res['Proportion'].keys():
                res['Proportion'][key] = res1['Proportion'][key]
        print(res)
        # 执行完成，结果中的stop置为1，表示结束
        
        res["stop"] = 1
        # 保存结果
        IOtool.write_json(res,osp.join(ROOT,"output", tid, stid+"_result.json"))
        # 将taskinfo中的状态置为2 代表子任务结果执行成功，此步骤为每个子任务必要步骤，请勿省略
        taskinfo[tid]["function"][stid]["state"]=2
        # 任务信息写回任务文件
        IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
        # 调用主任务状态修改函数，此步骤为每个子任务必要步骤，请勿省略
        IOtool.change_task_success_v2(tid=tid)
        return jsonify(res)
    else:
        abort(403)
# 数据集公平性提升
@app.route('/DataFairnessDebias', methods=['POST'])
def DataFairnessDebias():
    """
    数据集公平性提升
    输入：tid：主任务ID
    dataname：数据集名称
    datamethod：数据集优化算法名称
    """
    global LiRPA_LOGS
    if (request.method == "POST"):
        dataname = request.form.get("dataname")
        datamethod = request.form.get("datamethod")
        tid = request.form.get("tid")
        format_time = str(datetime.datetime.now().strftime("%Y%m%d%H%M"))
        AAtid = "S"+IOtool.get_task_id(str(format_time))
        taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
        taskinfo[tid]["function"].update({AAtid:{
            "type":"data_debias",
            "state":0,
            "name":["data_debias"],
            "dataset":dataname,
            "datamethod":datamethod,
        }})
        taskinfo[tid]["dataset"]=dataname
        IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
        # 执行任务
        t2 = threading.Thread(target=interface.run_data_debias,args=(tid,AAtid,dataname,datamethod))
        t2.setDaemon(True)
        t2.start()
        res = {
            "tid":tid,
            "AAtid":AAtid
        }
        return jsonify(res)
    else:
        abort(403)
# 模型公平性评估
@app.route('/ModelFairnessEvaluate', methods=['POST'])
def ModelFairnessEvaluate():
    """
    模型公平性评估
    输入：tid：主任务ID
    dataname：数据集名称
    modelname：模型名称
    """
    global LiRPA_LOGS
    if (request.method == "POST"):
        dataname = request.form.get("dataname")
        modelname = request.form.get("modelname")
        tid = request.form.get("tid")
        format_time = str(datetime.datetime.now().strftime("%Y%m%d%H%M"))
        AAtid = "S"+IOtool.get_task_id(str(format_time))
        taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
        taskinfo[tid]["function"].update({AAtid:{
            "type":"model_evaluate",
            "state":0,
            "name":["model_evaluate"],
            "dataset":dataname,
            "model":modelname,
        }})
        taskinfo[tid]["dataset"]=dataname
        taskinfo[tid]["model"]=modelname
        IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
        res = api.model_evaluate(dataname,modelname)
        res["Consistency"] = float(res["Consistency"])
        res["stop"] = 1
        IOtool.write_json(res,osp.join(ROOT,"output", tid, AAtid+"_result.json"))
        taskinfo[tid]["function"][AAtid]["state"]=2
        taskinfo[tid]["state"]=2
        IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
        return jsonify(res)
    else:
        abort(403)
# 模型公平性提升

@app.route('/ModelFairnessDebias', methods=['POST'])
def ModelFairnessDebias():
    """
    模型公平性提升
    输入：tid：主任务ID
    dataname：数据集名称
    modelname：模型名称
    algorithmname：模型优化算法名称
    """
    global LiRPA_LOGS
    if (request.method == "POST"):
        dataname = request.form.get("dataname")
        modelname = request.form.get("modelname")
        algorithmname = request.form.get("algorithmname")
        tid = request.form.get("tid")
        format_time = str(datetime.datetime.now().strftime("%Y%m%d%H%M"))
        AAtid = "S"+IOtool.get_task_id(str(format_time))
        taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
        taskinfo[tid]["function"].update({AAtid:{
            "type":"model_debias",
            "state":0,
            "name":["model_debias"],
            "dataset":dataname,
            "model":modelname,
            "algorithmname":algorithmname
        }})
        taskinfo[tid]["dataset"]=dataname
        taskinfo[tid]["model"]=modelname
        IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
        t2 = threading.Thread(target=interface.run_model_debias,args=(tid,AAtid,dataname,modelname,algorithmname))
        t2.setDaemon(True)
        t2.start()
        res = {
            "tid":tid,
            "AAtid":AAtid
        }
        return jsonify(res)
    else:
        abort(403)

# 任务列表查询
@app.route('/Task/QueryTask', methods=['GET'])
def query_task():
    '''任务列表查询
    输入：
    record:待查询的首条记录，从哪条开始查询，不填从最新一条查起
    count：数量，查几条，不填从record开始查所有
    输出：
    TaskList：任务列表
    Number：查到的任务数量
    '''
    record = -1 
    count = -1
    if request.form.get("record"):
        record = IOtool.atoi(request.form.get("record"))
    if request.form.get("count"):
        count = IOtool.atoi(request.form.get("count"))
    taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
    
    start_num = 0
    end_num = 0
    taskresult = {}
    with open(osp.join(ROOT,"output","task_list.txt"),"r") as fp:
        tasklist = []
        for line in fp.readlines():
            tasklist.append(line.strip())
    task_num = len(tasklist)
    if count < 0 and record < 0:
        body = {"code":1,"msg":"success","TaskList":taskinfo,"Number":len(taskinfo)}
        return jsonify(body)
    elif count >= 0 and record < 0:
        start_num = 0
        if count <task_num:
            end_num = count
        else:
            end_num = task_num
    elif count < 0 and record >= 0:
        if record > task_num:
            body = {"code":1001,"msg":"fail,parameter error","TaskList":{}}
            return jsonify(body)
        else:
            start_num = record
            end_num = task_num
    else:
        start_num = record
        end_num = record+count
        if start_num > task_num:
            body = {"code":1001,"msg":"fail,parameter error","TaskList":{}}
            return jsonify(body)
        elif end_num > task_num:
            end_num = task_num
    for i in range(start_num,end_num):
        taskresult.update(taskinfo[tasklist[i]])
    body = {"code":1,"msg":"success","TaskList":taskresult,"Number":len(taskresult)}
    return jsonify(body)

# 单个任务查询
@app.route('/Task/QuerySingleTask', methods=['GET'])
def query_single_task():
    '''单个任务查询
    输入：
    Taskid:主任务id
    '''
    if request.method == "GET":
        tid = request.args.get("Taskid")
        taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
        if tid not in taskinfo.keys():
            return jsonify({"code":1002,"msg":"fail,taskid not found!"})
        return jsonify({"code":1,"msg":"success","result":taskinfo[tid]})
    else:
        abort(403)
    
# 主任务创建
@app.route('/Task/CreateTask', methods=['POST'])
def creat_task():
    '''创建主任务
    输入：
    AttackAndDefenseTask:是否创建对抗攻击任务
    输出：
    Taskid：总任务ID
    '''
    if request.method == "POST":
        format_time = str(datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y%m%d%H%M"))
        tid = IOtool.get_task_id(str(format_time))
        outpath = osp.join(ROOT,"output",tid)
        cachepath = osp.join(ROOT,"output/cache")
        if not osp.exists(outpath):
            os.makedirs(outpath)
        if not osp.exists(cachepath):
            os.makedirs(cachepath)
        curinfo = {
            "state":0,
            "createtime":format_time,
            "dataset":"",
            "model":"",
            "function":{}
            }
        if not osp.exists(osp.join(ROOT,"output","task_info.json")):
            file = open(osp.join(ROOT,"output","task_info.json"),"w")
            file.close()
            data = {}
            IOtool.write_json(data,osp.join(ROOT,"output","task_info.json"))
        taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
        taskinfo[tid] = curinfo
        IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
        with open(osp.join(ROOT,"output","task_list.txt"),"a+") as fp:
            fp.write(tid)
            fp.write("\n")
        data = {"Taskid": tid}
        return jsonify(data)

# 日志查询
@app.route('/Task/QueryLog', methods=['GET'])
def query_log():
    '''任务日志查询
    输入：
    Taskid:主任务id
    输出：
    Log：返回日志信息
    '''
    Log = {}
    stid_list = []
    if not request.args.get("Taskid"):
        body = {"code":1001,"msg":"fail,parameter error"}
        return jsonify(body)
    tid = request.args.get("Taskid")
    taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
    stid_list =taskinfo[tid]["function"].keys()
    for stid in stid_list:
        if osp.exists(osp.join(ROOT, "output", tid, stid+"_log.txt")):
            with open(osp.join(ROOT, "output", tid, stid+"_log.txt"), "r") as fp:
                Log[stid] = fp.readlines()
    if len(stid_list)==0:
        body = {"code":1003,"msg":"fail,log is NULL","Log":Log}
    else:
        body = {"code":1,"msg":"success","Log":Log}
    return jsonify(body)
# 删除任务
@app.route('/Task/DeleteTask', methods=['DELETE'])
def delete_task():
    '''删除任务
    输入：
    Taskid:主任务id
    '''
    if not request.form.get("Taskid"):
        body = {"code":1001,"msg":"fail,parameter error"}
        return jsonify(body)

    tid = request.form.get("Taskid")
    with open(osp.join(ROOT,"output","task_list.txt"),"r") as fp:
        tasklist = []
        for line in fp.readlines():
            tasklist.append(line.strip())
    
    if tid not in tasklist:
        taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
        del taskinfo[tid]
        IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
        body = {"code":1002,"msg":"fail,task not found"}
        return jsonify(body)
    
    taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
    del taskinfo[tid]
    tasklist.remove(tid)
    IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))

    with open(osp.join(ROOT,"output","task_list.txt"),"w+") as fp:
        for temp in tasklist:
            fp.write(temp)
            fp.write("\n")
    outpath = osp.join(ROOT,"output",tid)
    if osp.exists(outpath):
        shutil.rmtree(outpath)
    body = {"code":1,"msg":"success"}
    return jsonify(body)
# 输出界面
@app.route("/index_results", methods=["GET", "POST"])
def index_results():
    if request.method == "GET":
        tid = request.args.get('tid')
        print(tid)
        return render_template("index_results.html",tid=tid)
    else:
        abort(403)

# 执行界面
@app.route("/index_evaluate", methods=["GET", "POST"])
def index_evaluate():
    if request.method == "GET":
        return render_template("index_evaluate.html")
    else:
        abort(403)

# 对抗攻击评估和鲁棒性增强
@app.route('/Attack/AdvAttack_old', methods=['POST'])
def adv_attack():
    if request.method == "POST":
        advInputDatastr = list(request.form.to_dict())[0]
        advInputData = json.loads(advInputDatastr)
        tid = advInputData["Taskid"]
        format_time = str(datetime.datetime.now().strftime("%Y%m%d%H%M"))
        AAtid = "S"+IOtool.get_task_id(str(format_time))
        from function.attack.old import adv_attack
        adv_attack.adv_attack(advInputData=advInputData,AAtid=AAtid)
        return json.dumps({"code":1,"msg":"success","Taskid":tid,"AdvAttackid":AAtid})
    else:
        abort(403)


# 结果输出
@app.route("/output/Resultdata", methods=["GET"])
def get_result():
    if request.method == "GET":
        stidlist = []
        # 使用postman获取参数
        try:
            inputdata = json.loads(request.data)
            print(inputdata)
            tid = inputdata["Taskid"]
            stidlist = inputdata["sid"]
        except:
            pass
        # 从web上传下来的参数
        if request.args.get("Taskid") != None:
            tid = request.args.get("Taskid")
        taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
        if stidlist== []:
            stidlist = taskinfo[tid]["function"].keys()
        # 如果能获取到子任务列表就使用获取，否则读取主任务下的所有子任务
        
        if request.args.get("stid") != None:
            stidlist = request.args.get("stid")
        print(stidlist)
        result = {}
        for stid in stidlist:
            attack_type = taskinfo[tid]["function"][stid]["type"]
            # 如果子任务状态不是执行成功，则返回子任务结果为空
            if taskinfo[tid]["function"][stid]["state"] != 2:
                result[attack_type]= {}
            # 如果子任务状态结果文件不存在，则返回子任务结果为空
            elif not osp.exists(osp.join(ROOT,"output",tid,stid+"_result.json")):
                result[attack_type]= {}
            else:
                result[attack_type] = (IOtool.load_json(osp.join(ROOT,"output",tid,stid+"_result.json")))
        stopflag = 1
        for temp in  result.keys():
            if "stop" not in result[temp].keys():
                stopflag = 0
            elif  result[temp]["stop"] != 1:
                stopflag = 0
        return jsonify({"code":1,"msg":"success","result":result,"stop":stopflag})

# ----------------- 课题4 形式化验证 -----------------

@app.route('/FormalVerification', methods=['GET',"POST"])
def FormalVerification():
    
    if (request.method == "GET"):
        return render_template("former_verification.html")
    else:
        param = {
            "dataset": request.form.get("dataset"),
            "model": request.form.get("model"),
            "size": int(request.form.get("size")),
            "up_eps": float(request.form.get("up_eps")),
            "down_eps": float(request.form.get("down_eps")),
            "steps": int(request.form.get("steps")),
            "task_id": request.form.get("tid"),
        }
        tid = request.form.get("tid")
        format_time = str(datetime.datetime.now().strftime("%Y%m%d%H%M"))
        AAtid = "S"+IOtool.get_task_id(str(format_time))
        taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
        print("*************************************add stid******************")
        taskinfo[tid]["function"].update({AAtid:{
            "type":"formal_verification",
            "state":0,
            "name":["formal_verification"],
            "dataset":request.form.get("dataset"),
            "model":request.form.get("model")
        }})
        taskinfo[tid]["dataset"]=request.form.get("dataset")
        taskinfo[tid]["model"]=request.form.get("model")
        IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
        t2 = threading.Thread(target=interface.run_verify, args=(tid, AAtid, param))
        t2.setDaemon(True)
        t2.start()
        res = {
            "tid":tid,
            "AAtid":AAtid
        }
        return jsonify(res)

# ----------------- 课题1 对抗攻击评估 -----------------
<<<<<<< HEAD
# from function.attack.adv0211 import *
=======
>>>>>>> gitee/feature_chunlai
@app.route('/Attack/AdvAttack', methods=['POST'])
def AdvAttack():
    """
    对抗攻击评估
    输入：tid：主任务ID
    Dataset：数据集名称
    Model：模型名称
    Method:list 对抗攻击算法名称
    
    """
    global LiRPA_LOGS
    if (request.method == "POST"):
        inputParam = json.loads(request.data)
<<<<<<< HEAD
        print(request.data)
        tid = inputParam["Taskid"]
=======
        tid = inputParam["Taskid"]
        inputParam["device"] = "cuda:0"
>>>>>>> gitee/feature_chunlai
        dataname = inputParam["Dataset"]
        model = inputParam["Model"]
        adv_method = inputParam["Method"]
        format_time = str(datetime.datetime.now().strftime("%Y%m%d%H%M"))
        stid = "S"+IOtool.get_task_id(str(format_time))
        taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
        taskinfo[tid]["function"].update({stid:{
            "type":"adv_attack",
            "state":0,
            "name":["adv_attack"],
            "dataset":dataname,
            "method":adv_method,
            "model":model,
        }})
        taskinfo[tid]["dataset"] = dataname
        taskinfo[tid]["model"] = model
        IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
        # 执行任务
        
        t2 = threading.Thread(target=interface.run_adv_attack,args=(tid, stid, dataname, model, adv_method, inputParam))
        t2.setDaemon(True)
        t2.start()
        res = {
            "tid":tid,
<<<<<<< HEAD
            "AAtid":stid
=======
            "stid":stid
        }
        return jsonify(res)
    else:
        abort(403)

# ----------------- 课题1 后门攻击评估 -----------------
@app.route('/Attack/BackdoorAttack', methods=['POST'])
def BackdoorAttack():
    """
    后门攻击评估
    输入：tid：主任务ID
    Dataset：数据集名称
    Model：模型名称
    Method:list 对抗攻击算法名称
    
    """
    global LiRPA_LOGS
    if (request.method == "POST"):
        inputParam = json.loads(request.data)
        print(request.data)
        tid = inputParam["Taskid"]
        inputParam["device"] = "cuda:0"
        dataname = inputParam["Dataset"]
        model = inputParam["Model"]
        adv_method = inputParam["Method"]
        format_time = str(datetime.datetime.now().strftime("%Y%m%d%H%M"))
        stid = "S"+IOtool.get_task_id(str(format_time))
        taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
        taskinfo[tid]["function"].update({stid:{
            "type":"backdoor_attack",
            "state":0,
            "name":["adv_attack"],
            "dataset":dataname,
            "method":adv_method,
            "model":model,
        }})
        taskinfo[tid]["dataset"] = dataname
        taskinfo[tid]["model"] = model
        IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
        # 执行任务
        
        t2 = threading.Thread(target=interface.run_backdoor_attack,args=(tid, stid, dataname, model, adv_method, inputParam))
        t2.setDaemon(True)
        t2.start()
        res = {
            "code":1,
            "msg":"success",
            "tid":tid,
            "stid":stid
        }
        return jsonify(res)
    else:
        abort(403)

# ----------------- 课题1 攻击机理分析 -----------------
@app.route('/Attack/AttackDimReduciton', methods=['POST'])
def AttackDimReduciton():
    """
    数据降维分布解释
    输入：tid：主任务ID
    Dataset：数据集名称
    Model：模型名称
    AdvMethods:list 对抗攻击算法名称
    
    """
    global LiRPA_LOGS
    if (request.method == "POST"):
        inputParam = json.loads(request.data)
        print(request.data)
        tid = inputParam["Taskid"]
        datasetparam = inputParam["DatasetParam"]
        modelparam = inputParam["ModelParam"]
        adv_methods = inputParam["AdvMethods"]
        vis_methods = inputParam["VisMethods"]
        device = "cuda:0"
        format_time = str(datetime.datetime.now().strftime("%Y%m%d%H%M"))
        stid = "S"+IOtool.get_task_id(str(format_time))
        taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
        taskinfo[tid]["function"].update({stid:{
            "type":"attack_dim_reduciton",
            "state":0,
            "name":["adv_attack"],
            "dataset":datasetparam["name"],
            "method":adv_methods,
            "model":modelparam["name"],
        }})
        taskinfo[tid]["dataset"] = datasetparam["name"]
        taskinfo[tid]["model"] = modelparam["name"]
        IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
        # 执行任务
        datasetparam["name"] = datasetparam["name"].lower()
        modelparam["name"] = modelparam["name"].lower()
        t2 = threading.Thread(target=interface.run_dim_reduct,args=(tid, stid, datasetparam, modelparam, vis_methods, adv_methods, device))
        t2.setDaemon(True)
        t2.start()
        res = {
            "code":1,
            "msg":"success",
            "tid":tid,
            "stid":stid
        }
        return jsonify(res)
    else:
        abort(403)

@app.route('/Attack/AttackAttrbutionAnalysis', methods=['POST'])
def AttackAttrbutionAnalysis():
    """
    对抗图像归因解释
    输入：tid：主任务ID
    Dataset：数据集名称
    Model：模型名称
    AdvMethods:list 对抗攻击算法名称
    ExMethods:攻击机理解释方法名称
    """
    global LiRPA_LOGS
    if (request.method == "POST"):
        inputParam = json.loads(request.data)
        print(request.data)
        tid = inputParam["Taskid"]
        datasetparam = inputParam["DatasetParam"]
        modelparam = inputParam["ModelParam"]
        adv_methods = inputParam["AdvMethods"]
        ex_methods = inputParam["ExMethods"]
        device = "cuda:0"
        format_time = str(datetime.datetime.now().strftime("%Y%m%d%H%M"))
        stid = "S"+IOtool.get_task_id(str(format_time))
        taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
        taskinfo[tid]["function"].update({stid:{
            "type":"attack_attrbution_analysis",
            "state":0,
            "name":["adv_attack"],
            "dataset":datasetparam["name"],
            "method":adv_methods,
            "model":modelparam["name"],
        }})
        taskinfo[tid]["dataset"] = datasetparam["name"]
        taskinfo[tid]["model"] = modelparam["name"]
        IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
        # 执行任务
        datasetparam["name"] = datasetparam["name"].lower()
        modelparam["name"] = modelparam["name"].lower()
        t2 = threading.Thread(target=interface.run_attrbution_analysis,args=(tid, stid, datasetparam, modelparam, ex_methods, adv_methods, device))
        t2.setDaemon(True)
        t2.start()
        res = {
            "code":1,
            "msg":"success",
            "tid":tid,
            "stid":stid
        }
        return jsonify(res)
    else:
        abort(403)

@app.route('/Attack/AttackLayerExplain', methods=['POST'])
def AttackLayerExplain():
    """
    模型内部分析解释
    输入：tid：主任务ID
    Dataset：数据集名称
    Model：模型名称
    AdvMethods:list 对抗攻击算法名称
    ExMethods:？？
    """
    global LiRPA_LOGS
    if (request.method == "POST"):
        inputParam = json.loads(request.data)
        print(request.data)
        tid = inputParam["Taskid"]
        datasetparam = inputParam["DatasetParam"]
        modelparam = inputParam["ModelParam"]
        adv_methods = inputParam["AdvMethods"]
        ex_methods = inputParam["ExMethods"]
        device = "cuda:0"
        format_time = str(datetime.datetime.now().strftime("%Y%m%d%H%M"))
        stid = "S"+IOtool.get_task_id(str(format_time))
        taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
        taskinfo[tid]["function"].update({stid:{
            "type":"attack_layer_explain",
            "state":0,
            "name":["adv_attack"],
            "dataset":datasetparam["name"],
            "method":adv_methods,
            "model":modelparam["name"],
        }})
        taskinfo[tid]["dataset"] = datasetparam["name"]
        taskinfo[tid]["model"] = modelparam["name"]
        IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
        # 执行任务
        datasetparam["name"] = datasetparam["name"].lower()
        modelparam["name"] = modelparam["name"].lower()
        t2 = threading.Thread(target=interface.run_layer_explain,args=(tid, stid, datasetparam, modelparam, ex_methods, adv_methods, device))
        t2.setDaemon(True)
        t2.start()
        res = {
            "code":1,
            "msg":"success",
            "tid":tid,
            "stid":stid
        }
        return jsonify(res)
    else:
        abort(403)

@app.route('/Attack/AttackLime', methods=['POST'])
def AttackLime():
    """
    多模态黑盒解释
    输入：tid：主任务ID
    Dataset：数据集名称
    Model：模型名称
    AdvMethods:list 对抗攻击算法名称
    ExMethods:？？
    """
    global LiRPA_LOGS
    if (request.method == "POST"):
        inputParam = json.loads(request.data)
        print(request.data)
        tid = inputParam["Taskid"]
        datasetparam = inputParam["DatasetParam"]
        modelparam = inputParam["ModelParam"]
        adv_methods = inputParam["AdvMethods"]
        # ex_methods = inputParam["ExMethods"]
        device = "cuda:0"
        format_time = str(datetime.datetime.now().strftime("%Y%m%d%H%M"))
        stid = "S"+IOtool.get_task_id(str(format_time))
        taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
        taskinfo[tid]["function"].update({stid:{
            "type":"attack_lime",
            "state":0,
            "name":["adv_attack"],
            "dataset":datasetparam["name"],
            "method":adv_methods,
            "model":modelparam["name"],
        }})
        taskinfo[tid]["dataset"] = datasetparam["name"]
        taskinfo[tid]["model"] = modelparam["name"]
        IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
        # 执行任务
        datasetparam["name"] = datasetparam["name"].lower()
        modelparam["name"] = modelparam["name"].lower()
        t2 = threading.Thread(target=interface.run_lime,args=(tid, stid, datasetparam, modelparam, adv_methods, device))
        t2.setDaemon(True)
        t2.start()
        res = {
            "code":1,
            "msg":"success",
            "tid":tid,
            "stid":stid
>>>>>>> gitee/feature_chunlai
        }
        return jsonify(res)
    else:
        abort(403)
<<<<<<< HEAD
=======

>>>>>>> gitee/feature_chunlai
# ----------------- 课题2 测试样本自动生成 -----------------
@app.route('/Concolic/SamGenParamGet', methods=['GET','POST'])
def Concolic():
    if (request.method == "GET"):
        return render_template("")
    elif (request.method == "POST"):
        concolic_dataset = request.form.get("dataname")
        concolic_model = request.form.get("modelname")
        norm = request.form.get("norm")
        
        tid = request.form.get("tid")
        format_time = str(datetime.datetime.now().strftime("%Y%m%d%H%M"))
        AAtid = "S"+IOtool.get_task_id(str(format_time))
        taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
        taskinfo[tid]["function"].update({AAtid:{
            "type":"Concolic",
            "state":0,
            "name":["Concolic"],
            "dataset":concolic_dataset,
            "model": concolic_model,
            "norm": norm
        }})
        taskinfo[tid]["dataset"]=concolic_dataset
        taskinfo[tid]["model"]=concolic_model
        IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
        t2 = threading.Thread(target=interface.run_concolic,args=(tid,AAtid,concolic_dataset,concolic_model,norm))
        t2.setDaemon(True)
        t2.start()
        res = {"code":1,"msg":"success","Taskid":tid,"Concolicid":AAtid}
        return jsonify(res)
    else:
        abort(403)

# ----------------- 课题2 系统环境分析与框架适配 -----------------
@app.route('/EnvTest/ETParamSet', methods=['GET','POST'])
def EnvTest():
    if (request.method == "GET"):
        return render_template("")
    elif (request.method == "POST"):
        matchmethod = request.form.get("matchmethod")
        frameworkname = request.form.get("frameworkname")
        frameversion = request.form.get("frameversion")
        tid = request.form.get("tid")
        format_time = str(datetime.datetime.now().strftime("%Y%m%d%H%M"))
        AAtid = "S"+IOtool.get_task_id(str(format_time))
        taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
        taskinfo[tid]["function"].update({AAtid:{
            "type":"EnvTest",
            "state":0,
            "name":["EnvTest"],
            "dataset": "",
            "model": "",
            "matchmethod": matchmethod,
            "framework": frameworkname+frameversion
        }})
        taskinfo[tid]["dataset"]=""
        taskinfo[tid]["model"]=""
        IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
        t2 = threading.Thread(target=interface.run_envtest,args=(tid,AAtid,matchmethod, frameworkname,frameversion))
        t2.setDaemon(True)
        t2.start()
        res = {"code":1,"msg":"success","Taskid":tid,"EnvTestid":AAtid}
        return jsonify(res)
    else:
        abort(403)
        
# ----------------- 课题2 标准化单元测试-- -----------------
@app.route('/UnitTest/DeepSstParamSet', methods=['GET','POST'])
def DeepSstParamSet():
    if (request.method == "GET"):
        return render_template("")
    elif (request.method == "POST"):
        dataset = request.form.get("dataset")
        modelname = request.form.get("modelname")
        pertube = request.form.get("pertube")
        m_dir = request.form.get("m_dir")
        tid = request.form.get("tid")
        format_time = str(datetime.datetime.now().strftime("%Y%m%d%H%M"))
        AAtid = "S"+IOtool.get_task_id(str(format_time))
        taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
        taskinfo[tid]["function"].update({AAtid:{
            "type":"DeepSst",
            "state":0,
            "name":["DeepSst"],
            "dataset": dataset,
            "model": modelname,
            "pertube": pertube,
            "m_dir": m_dir
        }})
        taskinfo[tid]["dataset"]=dataset
        taskinfo[tid]["model"]=modelname
        IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
        t2 = threading.Thread(target=interface.run_deepsst,args=(tid, AAtid, dataset, modelname, pertube, m_dir))
        t2.setDaemon(True)
        t2.start()
        res = {"code":1,"msg":"success","Taskid":tid,"DeepSstid":AAtid}
        return jsonify(res)
    else:
        abort(403)

# ----------------- 课题2 异常数据检测 -----------------
@app.route('/DataClean/DataCleanParamSet', methods=['GET','POST'])
def DataClean():
    '''
    输入：
        tid：主任务ID
        
    '''
    if (request.method == "GET"):
        return render_template("")
    elif (request.method == "POST"):
        dataset = request.form.get("dataset")
        tid = request.form.get("tid")
        format_time = str(datetime.datetime.now().strftime("%Y%m%d%H%M"))
        AAtid = "S"+IOtool.get_task_id(str(format_time))
        taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
        taskinfo[tid]["function"].update({AAtid:{
            "type":"DataClean",
            "state":0,
            "name":["DataClean"],
            "dataset": dataset,
            "model": "",
        }})
        taskinfo[tid]["dataset"]=dataset
        taskinfo[tid]["model"]=""
        IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
        t2 = threading.Thread(target=interface.run_dataclean,args=(tid,AAtid,dataset))
        t2.setDaemon(True)
        t2.start()
        res = {"code":1,"msg":"success","Taskid":tid,"DataCleanId":AAtid}
        return jsonify(res)
    else:
        abort(403)        


def app_run(args):
    web_config={'host':args.host,'port':args.port,'debug':args.debug}
    app.run(**web_config)

#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os.path as osp
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
from function.fairness import run_model_evaluate
from function.ex_methods.module.func import Logger
from flask_cors import *
import threading
import hashlib,base64
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
        try:
            senAttrList=json.loads(inputParam["senAttrList"])
            tarAttrList=json.loads(inputParam["tarAttrList"])
            staAttrList=json.loads(inputParam["staAttrList"])
        except:
            senAttrList=inputParam["senAttrList"]
            tarAttrList=inputParam["tarAttrList"]
            staAttrList=inputParam["staAttrList"]
        
        logging = Logger(filename=osp.join(ROOT,"output", tid, stid +"_log.txt"))
        # 执行任务，运行时间超过3分钟的请使用多线程，参考DataFairnessDebias函数的执行部分
        from function.fairness import run_dataset_evaluate
        res = run_dataset_evaluate(dataname, sensattrs=senAttrList, targetattrs=tarAttrList, staAttrList=staAttrList, logging=logging)
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
        inputParam = json.loads(request.data)
        dataname = inputParam["dataname"]
        datamethod = inputParam["datamethod"]
        tid = inputParam["tid"]
        format_time = str(datetime.datetime.now().strftime("%Y%m%d%H%M"))
        stid = "S"+IOtool.get_task_id(str(format_time))
        taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
        taskinfo[tid]["function"].update({stid:{
            "type":"data_debias",
            "state":0,
            "name":["data_debias"],
            "dataset":dataname,
            "datamethod":datamethod,
        }})
        taskinfo[tid]["dataset"]=dataname
        IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
        try:
            senAttrList=json.loads(inputParam["senAttrList"])
            tarAttrList=json.loads(inputParam["tarAttrList"])
            staAttrList=json.loads(inputParam["staAttrList"])
        except:
            senAttrList=inputParam["senAttrList"]
            tarAttrList=inputParam["tarAttrList"]
            staAttrList=inputParam["staAttrList"]
        # 执行任务
        t2 = threading.Thread(target=interface.run_data_debias_api,args=(tid, stid, dataname, datamethod, senAttrList, tarAttrList, staAttrList))
        t2.setDaemon(True)
        t2.start()
        res = {
            "tid":tid,
            "stid":stid
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
        inputParam = json.loads(request.data)
        dataname = inputParam["dataname"]
        tid = inputParam["tid"]
        modelname = inputParam["modelname"]
        format_time = str(datetime.datetime.now().strftime("%Y%m%d%H%M"))
        stid = "S"+IOtool.get_task_id(str(format_time))
        taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
        taskinfo[tid]["function"].update({stid:{
            "type":"model_evaluate",
            "state":0,
            "name":["model_evaluate"],
            "dataset":dataname,
            "model":modelname,
        }})
        taskinfo[tid]["dataset"]=dataname
        taskinfo[tid]["model"]=modelname
        IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
        
        try:
            metrics = json.loads(inputParam["metrics"])
            senAttrList=json.loads(inputParam["senAttrList"])
            tarAttrList=inputParam["tarAttrList"]
            staAttrList=json.loads(inputParam["staAttrList"])
        except:
            metrics = inputParam["metrics"]
            senAttrList=inputParam["senAttrList"]
            tarAttrList=inputParam["tarAttrList"]
            staAttrList=inputParam["staAttrList"]
        logging = Logger(filename=osp.join(ROOT,"output", tid, stid +"_log.txt"))
        res = run_model_evaluate(dataname, modelname, metrics, senAttrList, tarAttrList, staAttrList, logging=logging)
        res["Consistency"] = float(res["Consistency"])
        res["stop"] = 1
        IOtool.write_json(res,osp.join(ROOT,"output", tid, stid+"_result.json"))
        taskinfo[tid]["function"][stid]["state"]=2
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
        inputParam = json.loads(request.data)
        dataname = inputParam["dataname"]
        modelname = inputParam["modelname"]
        tid = inputParam["tid"]
        algorithmname = inputParam["algorithmname"]
       
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
        try:
            metrics = json.loads(inputParam["metrics"])
            senAttrList=json.loads(inputParam["senAttrList"])
            tarAttrList=inputParam["tarAttrList"]
            staAttrList=json.loads(inputParam["staAttrList"])
        except:
            metrics = inputParam["metrics"]
            senAttrList=inputParam["senAttrList"]
            tarAttrList=inputParam["tarAttrList"]
            staAttrList=inputParam["staAttrList"]
        t2 = threading.Thread(target=interface.run_model_debias_api,args=(tid, AAtid, dataname, modelname, algorithmname, metrics, senAttrList, tarAttrList, staAttrList))
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
        data_path = osp.join(ROOT, "dataset/data")
        # advInputData=json.loads(request.data)
        advInputDatastr = list(request.form.to_dict())[0]
        advInputData = json.loads(advInputDatastr)
        for method in advInputData["Method"]:
            for method_key in advInputData[method].keys():
                if method_key in ["steps","inf_batch","popsize", "pixels", "sampling", "n_restarts","n_classes","eot_iter", "n_queries"]:
                    advInputData[method][method_key]=int(advInputData[method][method_key])
                elif method_key == "attacks":
                    advInputData[method][method_key]=ast.literal_eval(advInputData[method][method_key])
                elif method_key not in ["random_start", "norm", "loss", "version"]:
                    advInputData[method][method_key]=float(advInputData[method][method_key])
        tid = advInputData["Taskid"]
        data_name = advInputData["Dataset"]["name"]
        model_name = advInputData["Model"]["name"].lower()
        pretrained = advInputData["Model"]["pretrained"]
        methods = advInputData["Method"]
        format_time = str(datetime.datetime.now().strftime("%Y%m%d%H%M"))
        AAtid = "S"+IOtool.get_task_id(str(format_time))
        outpath = osp.join(ROOT,"output",tid)
        cachepath = osp.join(ROOT,"output/cache")
        
        attack_params={
            'out_path': outpath,
            'cache_path': cachepath,
            "device": 0,
            "tid":tid,
            "stid":AAtid,
            'model': {
                'name': model_name,
                'path': osp.join("models/ckpt",f"{data_name}_{model_name}.pt"),
                'pretrained': pretrained
                }
        }
        for method in methods:
            attack_params[method]=advInputData[method]
        print("******************root:",ROOT)
        result = {
            "summary":{},
            "type":"AdvAttack",
            "stop": 0
        }
        func = []
        if advInputData["IsAdvAttack"]:
            func.append("AdvAttack")
            result["AdvAttack"] = {}
        if advInputData["IsAdvTrain"]:
            func.append("AdvTrain")
            result["AdvTrain"] = {}
        if advInputData["IsEnsembleDefense"]:
            func.append("EnsembleDefense")
            result["EnsembleDefense"] = {}
        if advInputData["IsPACADetect"]:
            func.append("PACA")
            result["PACA"] = {}
        taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
        taskinfo[tid]["dataset"]=data_name
        taskinfo[tid]["model"]=model_name
        taskinfo[tid]["function"].update({AAtid:{
            "type":"AdvAttack",
            "state":0,
            "name":func,
            "attackmethod":methods
        }})
        IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
        # IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
        # IOtool.write_json(result,osp.join(ROOT,"output",tid,AAtid+"_result.json"))
        taskparam = Taskparam(taskinfo[tid],attack_params,result)
        print("***************IsAdvAttack:")
        print(advInputData["IsAdvAttack"])
        print(method)
        print("读取结果中……")
        methodsall = ["FGSM","FFGSM","RFGSM","MIFGSM","BIM","PGD","PGDL2","DI2FGSM","EOTPGD"]
        result = IOtool.load_json(osp.join(ROOT,"output","S20230103_1556_929f12a_result.json"))
        for key in methodsall:
            if key not in methods:
                del result["AdvAttack"]["atk_acc"][key]
                del result["AdvAttack"]["atk_asr"][key]
                del result["AdvAttack"][key]
                del result["AdvTrain"]["def_acc"][key]
                del result["AdvTrain"]["def_asr"][key]
                del result["EnsembleDefense"]["ens_asr"][key]
                del result["EnsembleDefense"]["ens_acc"][key]
                del result["PACA"][key]
        if advInputData["IsAdvTrain"] != 1:
            del result["AdvTrain"]
        if advInputData["IsEnsembleDefense"] != 1:
            del result["EnsembleDefense"]
        if advInputData["IsPACADetect"] != 1:
            del result["PACA"]
        
        taskinfo[tid]["function"][AAtid]["state"]=1
        taskinfo[tid]["state"]=1
        IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
        time.sleep(60*len(methods))
        result["stop"]=1
        IOtool.write_json(result,osp.join(ROOT,"output",tid,AAtid+"_result.json"))
        taskinfo[tid]["function"][AAtid]["state"]=2
        taskinfo[tid]["state"]=2
        IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
        return json.dumps({"code":1,"msg":"success","Taskid":tid,"AdvAttackid":AAtid})
    else:
        abort(403)
        # advInputDatastr = list(request.form.to_dict())[0]
        # advInputData = json.loads(advInputDatastr)
        # tid = advInputData["Taskid"]
        # format_time = str(datetime.datetime.now().strftime("%Y%m%d%H%M"))
        # AAtid = "S"+IOtool.get_task_id(str(format_time))
        # from function.attack.old import adv_attack
        # adv_attack.adv_attack(advInputData=advInputData,AAtid=AAtid)
        # return json.dumps({"code":1,"msg":"success","Taskid":tid,"AdvAttackid":AAtid})
   

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
        print("tid",tid)
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
        print(result)
        return jsonify({"code":1,"msg":"success","result":result,"stop":stopflag})

# ----------------- 课题4 形式化验证 -----------------

@app.route('/FormalVerification', methods=['GET',"POST"])
def FormalVerification():
    
    if (request.method == "GET"):
        return render_template("former_verification.html")
    else:
        res = {
            "tid":"20230224_1106_d5ab4b1",
            "stid":"S20230224_1106_368e295"
        }
        return jsonify(res)
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
            "stid":AAtid
        }
        return jsonify(res)

# ----------------- 课题1 对抗攻击评估 -----------------
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
        print(request.data)
        inputParam = json.loads(request.data)
        tid = inputParam["Taskid"]
        inputParam["device"] = "cuda:0"
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

@app.route('/reach',methods=["GET","POST"])
def model_reach():
    if request.method=='POST':
        inputParam = json.loads(request.data)
        dataset=inputParam['dataset']
        pic=inputParam['pic']
        label=inputParam['label']
        target=inputParam['target']
        tid = inputParam["tid"]
        format_time = str(datetime.datetime.now().strftime("%Y%m%d%H%M"))
        stid = "S"+IOtool.get_task_id(str(format_time))
        img_dir=os.path.join(os.getcwd(),"web/static/imgs/tmp_imgs")
        try:
            os.mkdir(os.path.join(img_dir,tid))
        except:
            pass
        pic_path=os.path.join(img_dir,tid,stid+'.pt')
        with open(pic_path, 'wb') as f:
            f.write(base64.b64decode(pic.replace('data:image/png;base64,','')))
            f.close()
        taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
        taskinfo[tid]["function"].update({stid:{
            "type":"attack_dim_reduciton",
            "state":0,
            "dataset":dataset,
            "model":'CNN',
            'label':label,
            'target':target,
            'pic':pic_path
        }})
        taskinfo[tid]["dataset"] = dataset
        taskinfo[tid]["model"] = 'CNN'
        IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
        
        resp=interface.reach(tid,stid,dataset,pic_path,label,target)
        resp['input']=f'static/imgs/tmp_imgs/{tid}/{stid}.png'
        IOtool.write_json(resp, osp.join(ROOT,"output", tid, stid+"_result.json")) 
        taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
        taskinfo[tid]["function"][stid]["state"] = 2
        IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
        IOtool.change_task_success_v2(tid)
        return resp
    return render_template('reach.html')
@app.route('/knowledge_consistency',methods=["GET","POST"])
def model_consistency():
    if request.method=='POST':
        inputParam = json.loads(request.data)
        tid = inputParam["tid"]
        net=inputParam['net']
        layer=inputParam['layer']
        dataset=inputParam['dataset']
        pic=inputParam['pic']
        format_time = str(datetime.datetime.now().strftime("%Y%m%d%H%M"))
        stid = "S"+IOtool.get_task_id(str(format_time))
        img_dir=os.path.join(os.getcwd(),"web/static/imgs/tmp_imgs")
        try:
            os.mkdir(os.path.join(img_dir,tid))
        except:
            pass
        taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
        taskinfo[tid]["function"].update({stid:{
            "type":"attack_dim_reduciton",
            "state":0,
            "dataset":dataset,
            "layer":layer,
            "model":net,
        }})
        taskinfo[tid]["dataset"] = dataset
        taskinfo[tid]["model"] = net
        IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
        
        img_path=os.path.join(img_dir,tid,stid+'.png')
        with open(img_path, 'wb') as f:
            f.write(base64.b64decode(pic.replace('data:image/png;base64,','')))
            f.close()
        
        l2,layers=interface.knowledge_consistency(tid, stid, net,dataset,img_path,layer)
        resp={'l2':l2,'input':f'static/imgs/tmp_imgs/{tid}/{stid}.png',
                'output':f'static/imgs/tmp_imgs/{tid}/{stid}_output_{layer}.png',
                'target':f'static/imgs/tmp_imgs/{tid}/{stid}_target_{layer}.png',
                'delta':f'static/imgs/tmp_imgs/{tid}/{stid}_delta_{layer}.png',
            }
        IOtool.write_json(resp, osp.join(ROOT,"output", tid, stid+"_result.json")) 
        taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
        taskinfo[tid]["function"][stid]["state"] = 2
        IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
        IOtool.change_task_success_v2(tid)
        return json.dumps(resp,ensure_ascii=False)
    return render_template('knowledge_consistency.html')
@app.route('/auto_verify_img',methods=["GET","POST"])
def auto_verify_img():
    if request.method=='POST':
        inputParam = json.loads(request.data)
        tid = inputParam["tid"]
        format_time = str(datetime.datetime.now().strftime("%Y%m%d%H%M"))
        stid = "S"+IOtool.get_task_id(str(format_time))
        net=inputParam['net']
        
        if 'eps' in inputParam:
            eps=float(inputParam['eps'])
        else:
            eps=0.1
        
        pic=inputParam['pic']
        dataset=inputParam['dataset']
        taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
        taskinfo[tid]["function"].update({stid:{
            "type":"attack_dim_reduciton",
            "state":0,
            "dataset":dataset,
            "model":net,
            "eps":eps
        }})
        taskinfo[tid]["dataset"] = dataset
        taskinfo[tid]["model"] = net
        IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
        img_dir=os.path.join(os.getcwd(),"web/static/imgs/tmp_imgs")
        try:
            os.mkdir(os.path.join(img_dir,tid))
        except:
            pass
        
        pic_path=os.path.join(img_dir,tid,stid+'.png')
        taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
        taskinfo[tid]["function"].update({stid:{
            "type":"formal_verify_1",
            "state":0,
            "dataset":dataset,
            "pic_path":pic_path,
            'eps':eps,
            'model':net
        }})
        taskinfo[tid]["dataset"] = dataset
        with open( pic_path, 'wb') as f:
            f.write(base64.b64decode(pic.replace('data:image/png;base64,','')))
            f.close()
        resp=interface.verify_img(tid, stid, net, dataset, eps, pic_path)
        IOtool.write_json(resp, osp.join(ROOT,"output", tid, stid+"_result.json")) 
        taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
        taskinfo[tid]["function"][stid]["state"] = 2
        IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
        IOtool.change_task_success_v2(tid)
        return json.dumps(resp,ensure_ascii=False)

    return render_template('index_auto_verify.html')
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
        }
        return jsonify(res)
    else:
        abort(403)
# ----------------- 课题2 测试样本自动生成 -----------------
@app.route('/Concolic/SamGenParamGet', methods=['GET','POST'])
def Concolic():
    if (request.method == "GET"):
        return render_template("")
    elif (request.method == "POST"):
        # concolic_dataset = request.form.get("dataname")
        # concolic_model = request.form.get("modelname")
        # norm = request.form.get("norm")
        # times = request.form.get("times")
        # tid = request.form.get("tid")
        inputdata = json.loads(request.data)
        print(inputdata)
        concolic_dataset = inputdata["dataname"]
        concolic_model = inputdata["modelname"]
        norm = inputdata["norm"]
        times = inputdata["times"]
        tid = inputdata["tid"]
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
        t2 = threading.Thread(target=interface.run_concolic,args=(tid,AAtid,concolic_dataset,concolic_model,norm, times))
        t2.setDaemon(True)
        t2.start()
        res = {"code":1,"msg":"success","Taskid":tid,"stid":AAtid}
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
        try:
            input_param = json.loads(request.data)
            tid = input_param["tid"]
            matchmethod = input_param["matchmethod"]
            frameworkname = input_param["frameworkname"]
            frameversion = input_param["frameversion"]
        except:
            pass
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
        res = {"code":1,"msg":"success","Taskid":tid,"stid":AAtid}
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
        uoload_flag = request.form.get("flag")
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
        res = {"code":1,"msg":"success","Taskid":tid,"stid":AAtid}
        return jsonify(res)
    else:
        abort(403) 


# ----------------- 课题2 标准化单元测试-- -----------------
@app.route('/UnitTest/DeepSstParamSet', methods=['GET','POST']) # 敏感神经元测试准则
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
        res = {"code":1,"msg":"success","Taskid":tid,"stid":AAtid}
        return jsonify(res)
    else:
        abort(403)

       


def app_run(args):
    web_config={'host':args.host,'port':args.port,'debug':args.debug}
    app.run(**web_config)

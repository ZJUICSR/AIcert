#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os.path as osp
import interface
import os, json, datetime, time, base64, threading
import pytz,shutil
from IOtool import IOtool
from flask import render_template, redirect, url_for, Flask, request, jsonify, send_from_directory, send_file, make_response
from flask import current_app as abort
from flask_cors import *

ROOT = os.getcwd()


poollist = []
#  {tid:{stid:feature}}
task_list = {}
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
        # return render_template("index_function_introduction.html")
        return render_template("task_center.html")

@app.route('/ex/uploadModel', methods=['POST'])
def ExUploadModel():
    fileinfo = request.files.get("ex_upload_model")
    filepath = "model/ckpt/ex_upload_model.pt"
    if osp.exists(filepath):
        os.remove(filepath)
    fileinfo.save(filepath)
    res={
        "code":10000,
        "msg":"success"
    }
    return jsonify(res)
    # 
@app.route('/login', methods=['POST'])
def login():
    if request.method == 'POST' and request.form.get('username') and request.form.get('password'):
        datax = request.form.to_dict()
        usernamx = datax.get("username")
        passwordx = datax.get("password")
        print(usernamx,passwordx)
        if osp.exists(os.path.join(ROOT,"output","user_info.json")):
            userinfo = IOtool.load_json(osp.join(ROOT,"output","user_info.json"))
            if usernamx in userinfo.keys() and userinfo[usernamx] == passwordx:
                resp = {
                    "code":"1",
                    "msg":"Login Success"
                }
            else:
                resp = {
                    "code":"-1",
                    "msg":"Login Fail"
                }
        else:
            resp = {
                "code":"-1",
                "msg":"Login Fail"
            }
        return jsonify(resp)
    else:
        abort(403)

@app.route('/register', methods=['POST'])
def register():
    if request.method == 'POST' and request.form.get('username') and request.form.get('password'):
        datax = request.form.to_dict()
        usernamx = datax.get("username")
        passwordx = datax.get("password")
        if osp.exists(os.path.join(ROOT,"output","user_info.json")):
            userinfo = IOtool.load_json(osp.join(ROOT,"output","user_info.json"))
        else:
            userinfo = {}
        if usernamx in userinfo.keys() or usernamx=='unknown':
            resp = {"code":-1,"msg":"You input the user name already exists, please re-entry."}
        else:
            userinfo.update({usernamx:passwordx})
            IOtool.write_json(userinfo, osp.join(ROOT,"output","user_info.json"))
            resp = {
                "code":1,
                "msg":"success"
            }
        return jsonify(resp)
    else:
        return "no"

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
    
        # 添加任务信息到taskinfo
        value = {
            # 任务类型,注意任务类型不能重复，用于结果返回的key值索引
            "type":"date_evaluate",
            # 任务状态：0 未执行；1 正在执行；2 执行成功；3 执行失败
            "state":0,
            # 方法名称：如对抗攻击中的fgsm，ffgsm等，呈现在结果界面
            "name":["date_evaluate"],
            # 数据集信息，呈现在结果界面，若干有选择模型还需增加模型字段：model
            "dataset":dataname,
        }
        IOtool.add_subtask_info(tid, stid, value)
        IOtool.change_task_info(tid, "dataset", dataname)
        
        try:
            senAttrList=json.loads(inputParam["senAttrList"])
            tarAttrList=json.loads(inputParam["tarAttrList"])
            staAttrList=json.loads(inputParam["staAttrList"])
        except:
            senAttrList=inputParam["senAttrList"]
            tarAttrList=inputParam["tarAttrList"]
            staAttrList=inputParam["staAttrList"]
        
        logging = IOtool.get_logger(stid, tid)
        # 执行任务，运行时间超过3分钟的请使用多线程，参考DataFairnessDebias函数的执行部分
        from function.fairness import run_dataset_evaluate
        pool = IOtool.get_pool(tid)
        t2 = pool.submit(run_dataset_evaluate, dataname, sensattrs=senAttrList, targetattrs=tarAttrList, staAttrList=staAttrList, logging=logging)
        IOtool.add_task_queue(tid, stid, t2, 300)
        res = t2.result()
        # res = run_dataset_evaluate(dataname, sensattrs=senAttrList, targetattrs=tarAttrList, staAttrList=staAttrList, logging=logging)
        # 执行完成，结果中的stop置为1，表示结束
        res["stop"] = 1
        # 保存结果
        IOtool.write_json(res, osp.join(ROOT,"output", tid, stid+"_result.json"))
        # 将taskinfo中的状态置为2 代表子任务结果执行成功，此步骤为每个子任务必要步骤，请勿省略
        IOtool.change_subtask_state(tid, stid, 2)
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
        value = {
            "type":"data_debias",
            "state":0,
            "name":["data_debias"],
            "dataset":dataname,
            "datamethod":datamethod,
        }
        IOtool.add_subtask_info(tid, stid, value)
        IOtool.change_task_info(tid, "dataset", dataname)
        
        try:
            senAttrList=json.loads(inputParam["senAttrList"])
            tarAttrList=json.loads(inputParam["tarAttrList"])
            staAttrList=json.loads(inputParam["staAttrList"])
        except:
            senAttrList=inputParam["senAttrList"]
            tarAttrList=inputParam["tarAttrList"]
            staAttrList=inputParam["staAttrList"]
        # 执行任务
        pool = IOtool.get_pool(tid)
        t2 = pool.submit(interface.run_data_debias_api, tid, stid, dataname, datamethod, senAttrList, tarAttrList, staAttrList)
        IOtool.add_task_queue(tid, stid, t2, 300)
        # t2 = threading.Thread(target=interface.run_data_debias_api,args=(tid, stid, dataname, datamethod, senAttrList, tarAttrList, staAttrList))
        # t2.setDaemon(True)
        # t2.start()
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
        value = {
            "type":"model_evaluate",
            "state":0,
            "name":["model_evaluate"],
            "dataset":dataname,
            "model":modelname,
        }
        IOtool.add_subtask_info(tid, stid, value)
        IOtool.change_task_info(tid, "dataset", dataname)
        IOtool.change_task_info(tid, "model", modelname)
        
        pool = IOtool.get_pool(tid)
        if dataname in ["Compas", "Adult", "German"]:
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
                
            t2 = pool.submit(interface.run_model_eva_api, tid, stid, dataname, modelname, metrics = metrics, senAttrList = senAttrList, tarAttrList = tarAttrList, staAttrList = staAttrList)
        else:
            dataname = dataname.lower()
            if dataname == "cifar10-s":
                dataname = "cifar-s" 
            try:
                metrics = json.loads(inputParam["metrics"])
                
                print(1,type(metrics))
            except:
                metrics = inputParam["metrics"]
            print(metrics)
            test_mode = inputParam["test_mode"]
            t2 = pool.submit(interface.run_model_eva_api, tid, stid, dataname, modelname, metrics = metrics, test_mode = test_mode)
            
        IOtool.add_task_queue(tid, stid, t2, 300)
        # interface.run_model_eva_api( tid, stid, dataname, modelname, metrics = metrics, test_mode = test_mode)
        res = {
            "tid":tid,
            "stid":stid
        }
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
        
        value = {
            "type":"model_debias",
            "state":0,
            "name":["model_debias"],
            "dataset":dataname,
            "model":modelname,
            "algorithmname":algorithmname
        }
        IOtool.add_subtask_info(tid, AAtid, value)
        IOtool.change_task_info(tid, "dataset", dataname)
        IOtool.change_task_info(tid, "model", modelname)
        pool = IOtool.get_pool(tid)
        
        if dataname in ["Compas", "Adult", "German"]:
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
            t2 = pool.submit(interface.run_model_debias_api, tid, AAtid, dataname, modelname, algorithmname, metrics, sensattrs = senAttrList, targetattr = tarAttrList, staAttrList = staAttrList)
        else:
            dataname = dataname.lower()
            # time.sleep(20)
            if dataname == "cifar10-s":
                dataname = "cifar-s" 
            try:
                metrics = json.loads(inputParam["metrics"])
                
            except:
                metrics = inputParam["metrics"]
            test_mode = inputParam["test_mode"]
            t2 = pool.submit(interface.run_model_debias_api, tid, AAtid, dataname, modelname, algorithmname, metrics, test_mode = test_mode)
            # time.sleep(20)
            
        IOtool.add_task_queue(tid, AAtid, t2, 300)
        # interface.run_model_debias_api(tid, AAtid, dataname, modelname, algorithmname, metrics, test_mode = test_mode)
        # interface.run_model_debias_api(tid, AAtid, dataname, modelname, algorithmname, metrics, senAttrList, tarAttrList, staAttrList)
        
        res = {
            "tid":tid,
            "stid":AAtid
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
    
    taskinfo = IOtool.get_task_info()
    
    start_num = 0
    end_num = 0
    taskresult = {}
    username = request.headers.get('user')  if request.headers.get('user') else 'unknown'
    if osp.exists(osp.join(ROOT,"output","task_list.json")):
        task_user_info = IOtool.load_json(osp.join(ROOT,"output","task_list.json"))
        if username not in task_user_info.keys():
            body = {"code":1,"msg":"success","TaskList":{},"Number":0}
            return jsonify(body)
        tasklist = task_user_info[username]
    else:
        body = {"code":1001,"msg":"fail,parameter error","TaskList":{}}
        return jsonify(body)
    task_num = len(tasklist)
    if count < 0 and record < 0:
        for temp in tasklist:
            taskresult.update({temp:taskinfo[temp]})
        body = {"code":1,"msg":"success","TaskList":taskresult,"Number":len(taskresult)}
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
        taskresult.update({tasklist[i]:taskinfo[tasklist[i]]})
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
        inputdata = json.loads(request.data)
        tid = inputdata["Taskid"]
        taskinfo = IOtool.get_task_info()
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
        username = request.headers.get('user')  if request.headers.get('user') else 'unknown'
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
            IOtool.reset_task_info(data)
        
        IOtool.add_task_info(tid, curinfo)
        if osp.exists(osp.join(ROOT,"output","task_list.json")):
            task_user_info = IOtool.load_json(osp.join(ROOT,"output","task_list.json"))
        else:
            task_user_info = {"unknown":[]}
            file = open(osp.join(ROOT,"output","task_list.json"),"w")
            file.close()
        if username in task_user_info.keys():
            task_user_info[username].append(tid)
        else:
            task_user_info.update({username:[tid]})
        IOtool.write_json(task_user_info, osp.join(ROOT,"output","task_list.json"))
        IOtool.add_pool(tid)
        # if len(IOtool.query_task_queue()) == 0:
        #     IOtool.task_queue_init(tid)
        #     t2 = threading.Thread(target = IOtool.check_sub_task_threading)
        #     t2.setDaemon(True)
        #     t2.start()
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
    tid = request.args.get("Taskid")
    if not tid:
        inputdata = json.loads(request.data)
        tid = inputdata["Taskid"]
    if not tid:
        body = {"code":1001,"msg":"fail,parameter error"}
        return jsonify(body)
    taskinfo = IOtool.get_task_info()
    
    stid_list =taskinfo[tid]["function"].keys()
    for stid in stid_list:
        if osp.exists(osp.join(ROOT, "output", tid, stid+"_log.txt")):
            with open(osp.join(ROOT, "output", tid, stid+"_log.txt"), "r") as fp:
                Log[stid] = fp.readlines()
    if len(stid_list)==0:
        body = {"code":1003,"msg":"fail,log is NULL","Log":Log}
    else:
        body = {"code":1,"msg":"success","Log":Log}
    # print(body)
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
    username = request.headers.get('user')  if request.headers.get('user') else 'unknown'
    if osp.exists(osp.join(ROOT,"output","task_list.json")):
        task_user_info = IOtool.load_json(osp.join(ROOT,"output","task_list.json"))
    else:
        taskinfo = IOtool.del_task_info(tid)
        body = {"code":1002,"msg":"fail,task not found"}
        return jsonify(body)
    
    if username not in task_user_info.keys():
        return jsonify({"code":1002,"msg":"fail,task not found"})
    else:
        if tid not in task_user_info[username]:
            return jsonify({"code":1002,"msg":"fail,task not found"})
        else:
            task_user_info[username].remove(tid)
    taskinfo = IOtool.del_task_info(tid) 
    IOtool.write_json(task_user_info, osp.join(ROOT,"output","task_list.json"))
    outpath = osp.join(ROOT,"output",tid)
    if osp.exists(outpath):
        shutil.rmtree(outpath)
    body = {"code":1,"msg":"success"}
    return jsonify(body)
# 输出界面
# @app.route("/index_results", methods=["GET", "POST"])
# def index_results():
#     if request.method == "GET":
#         tid = request.args.get('tid')
#         print(tid)
#         return render_template("index_results.html",tid=tid)
#     else:
#         abort(403)

# 执行界面
# @app.route("/index_evaluate", methods=["GET", "POST"])
# def index_evaluate():
#     if request.method == "GET":
#         return render_template("index_evaluate.html")
#     else:
#         abort(403)

# 结果输出
@app.route("/output/Resultdata", methods=["GET"])
def get_result():
    if request.method == "GET":
        stidlist = []
        # 使用postman获取参数
        try:
            inputdata = json.loads(request.data)
            tid = inputdata["Taskid"]
            stidlist = inputdata["sid"]
        except:
            pass
        # 从web上传下来的参数
        if request.args.get("Taskid") != None:
            tid = request.args.get("Taskid")
        
        taskinfo = IOtool.get_task_info()
        
        if stidlist== []:
            stidlist = taskinfo[tid]["function"].keys()
        # 如果能获取到子任务列表就使用获取，否则读取主任务下的所有子任务
        
        if request.args.get("stid") != None:
            stidlist = request.args.get("stid")
        result = {}
        for stid in stidlist:
            attack_type = taskinfo[tid]["function"][stid]["type"]
            # 如果子任务状态不是执行成功，则返回子任务结果为空
            if taskinfo[tid]["function"][stid]["state"] < 2 :
                result[attack_type]= {}
            # 如果子任务状态结果文件不存在，则返回子任务结果为空
            elif not osp.exists(osp.join(ROOT,"output",tid,stid+"_result.json")):
                result[attack_type]= {}
            else:
                result[attack_type] = (IOtool.load_json(osp.join(ROOT,"output",tid,stid+"_result.json")))
        stopflag = 1
        for temp in  result.keys():
            # print("result[temp]:", temp,result)
            if "stop" not in result[temp].keys():
                stopflag = 0
            elif  result[temp]["stop"] == 0:
                stopflag = 0
            elif result[temp]["stop"] == 2:
                stopflag = 2
        # print(result)
        # print("stopflag", stopflag)
        return jsonify({"code":1,"msg":"success","result":result,"stop":stopflag})

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
        inputParam = json.loads(request.data)
        tid = inputParam["Taskid"]
        # inputParam["device"] = "cuda:0"
        dataname = inputParam["Dataset"]
        model = inputParam["Model"]
        sample_num = inputParam["sample_num"]
        try:
            adv_method = json.loads(inputParam["Method"])
        except:
            adv_method = inputParam["Method"]
        format_time = str(datetime.datetime.now().strftime("%Y%m%d%H%M"))
        stid = "S"+IOtool.get_task_id(str(format_time))
        value = {
            "type":"adv_attack",
            "state":0,
            "name":["adv_attack"],
            "dataset":dataname,
            "method":adv_method,
            "model":model,
        }
        IOtool.add_subtask_info(tid, stid, value)
        IOtool.change_task_info(tid, "dataset", dataname)
        IOtool.change_task_info(tid, "model", model)
        # 执行任务
        pool = IOtool.get_pool(tid)
        t2 = pool.submit(interface.run_adv_attack, tid, stid, dataname, model, adv_method, inputParam, sample_num)
        print(inputParam)
        IOtool.add_task_queue(tid, stid, t2, 72000*len(adv_method))
        # interface.run_adv_attack(tid, stid, dataname, model, adv_method, inputParam)
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
        # inputParam["device"] = "cuda:0"
        dataname = inputParam["Dataset"]
        model = inputParam["Model"]
        try:
            adv_method = json.loads(inputParam["Method"])
        except:
            adv_method = inputParam["Method"]
        format_time = str(datetime.datetime.now().strftime("%Y%m%d%H%M"))
        stid = "S"+IOtool.get_task_id(str(format_time))
        
        value = {
            "type":"backdoor_attack",
            "state":0,
            "name":["adv_attack"],
            "dataset":dataname,
            "method":adv_method,
            "model":model,
        }
        IOtool.add_subtask_info(tid, stid, value)
        IOtool.change_task_info(tid, "dataset", dataname)
        IOtool.change_task_info(tid, "model", model)
        # 执行任务
        pool = IOtool.get_pool(tid)
        t2 = pool.submit(interface.run_backdoor_attack, tid, stid, dataname, model, adv_method, inputParam)
        IOtool.add_task_queue(tid, stid, t2, 72000*len(adv_method))
        # t2 = threading.Thread(target=interface.run_backdoor_attack,args=(tid, stid, dataname, model, adv_method, inputParam))
        # t2.setDaemon(True)
        # t2.start()
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
        tid = inputParam["Taskid"]
        print("tid:",tid)
        datasetparam = inputParam["DatasetParam"]
        modelparam = inputParam["ModelParam"]
        adv_methods = inputParam["AdvMethods"]
        vis_methods = inputParam["VisMethods"]
        format_time = str(datetime.datetime.now().strftime("%Y%m%d%H%M"))
        stid = "S"+IOtool.get_task_id(str(format_time))
        value = {
            "type":"attack_dim_reduciton",
            "state":0,
            "name":["adv_attack"],
            "dataset":datasetparam["name"],
            "method":adv_methods,
            "model":modelparam["name"],
            "vis_method":vis_methods
        }
        IOtool.add_subtask_info(tid, stid, value)
        IOtool.change_task_info(tid, "dataset", datasetparam["name"])
        IOtool.change_task_info(tid, "model", modelparam["name"])
        # 执行任务
        datasetparam["name"] = datasetparam["name"].lower()
        modelparam["name"] = modelparam["name"].lower()
        pool = IOtool.get_pool(tid)
        
        t2 = pool.submit(interface.run_dim_reduct, tid, stid, datasetparam, modelparam, vis_methods, adv_methods)
        
        IOtool.add_task_queue(tid, stid, t2, 4000 * len(vis_methods) + 3000*len(adv_methods))
        # interface.run_dim_reduct(tid, stid, datasetparam, modelparam, vis_methods, adv_methods)
        # t2 = threading.Thread(target=interface.run_dim_reduct,args=(tid, stid, datasetparam, modelparam, vis_methods, adv_methods, device))
        # t2.setDaemon(True)
        # t2.start()
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
        img_dir=os.path.join(os.getcwd(),"output",tid, stid)
        if not os.path.exists(img_dir):
            os.mkdir(img_dir)
        pic_path=os.path.join(img_dir,'input.png')
        if "image/jpeg;" in pic:
            
            with open( pic_path, 'wb') as f:
                f.write(base64.b64decode(pic.replace('data:image/jpeg;base64,','')))
                f.close()
        else:
            
            with open( pic_path, 'wb') as f:
                f.write(base64.b64decode(pic.replace('data:image/png;base64,','')))
                f.close()
        value = {
            "type":"model_reach",
            "state":0,
            "dataset":dataset,
            "model":'CNN',
            'label':label,
            'target':target,
            'pic':pic_path
        }
        IOtool.add_subtask_info(tid, stid, value)
        IOtool.change_task_info(tid, "dataset", dataset)
        IOtool.change_task_info(tid, "model", 'CNN')
        pool = IOtool.get_pool(tid)
        t2 = pool.submit(interface.reach, tid,stid,dataset.upper(),pic_path,label,target)
        
        IOtool.add_task_queue(tid, stid, t2, 300)
        # interface.reach(tid, stid, dataset.upper(), pic_path, label, target)
        resp = t2.result()
        print(resp)
        return jsonify(resp)
    
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
        img_dir=os.path.join(os.getcwd(),"web/static/img/tmp_imgs")
        if not os.path.exists(img_dir):
            os.mkdir(img_dir)
        pic_path=os.path.join(img_dir,tid,stid+'.png')
        try:
            os.mkdir(os.path.join(img_dir,tid))
        except:
            pass
        if "image/jpeg;" in pic:
            with open( pic_path, 'wb') as f:
                f.write(base64.b64decode(pic.replace('data:image/jpeg;base64,','')))
                f.close()
        else:
            
            with open( pic_path, 'wb') as f:
                f.write(base64.b64decode(pic.replace('data:image/png;base64,','')))
                f.close()
        value = {
            "type":"model_consistency",
            "state":0,
            "dataset":dataset,
            "layer":layer,
            "model":net,
        }
        IOtool.add_subtask_info(tid, stid, value)
        IOtool.change_task_info(tid, "dataset", dataset)
        IOtool.change_task_info(tid, "model", net)
        pool = IOtool.get_pool(tid)
        t2 = pool.submit(interface.knowledge_consistency, tid, stid, net, dataset, pic_path,layer)
        
        IOtool.add_task_queue(tid, stid, t2, 300)
        # interface.knowledge_consistency(tid, stid, net, dataset, pic_path,layer)
        resp = t2.result()
        # resp=interface.knowledge_consistency(tid, stid, net,dataset,pic_path,layer)
        return json.dumps(resp,ensure_ascii=False)
    
    # return render_template('knowledge_consistency.html')

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
        img_dir=os.path.join(os.getcwd(),"web/static/img/tmp_imgs")
        if not os.path.exists(img_dir):
            os.mkdir(img_dir)
        pic_path=os.path.join(img_dir,tid,stid+'.png')
        try:
            os.mkdir(os.path.join(img_dir,tid))
        except:
            pass
        if "image/jpeg;" in pic:
            
            with open( pic_path, 'wb') as f:
                f.write(base64.b64decode(pic.replace('data:image/jpeg;base64,','')))
                f.close()
        else:
            
            with open( pic_path, 'wb') as f:
                f.write(base64.b64decode(pic.replace('data:image/png;base64,','')))
                f.close()
        value = {
            "type":"auto_verify",
            "state":0,
            "dataset":dataset,
            "pic_path":pic_path,
            'eps':eps,
            'model':net
        }
        IOtool.add_subtask_info(tid, stid, value)
        IOtool.change_task_info(tid, "dataset", dataset)
        IOtool.change_task_info(tid, "model", net)
        if "cifar" in dataset.lower():
            dataset="CIFAR"
        pool = IOtool.get_pool(tid)
        t2 = pool.submit(interface.verify_img, tid, stid, net, dataset.upper(), eps, pic_path)
        IOtool.add_task_queue(tid, stid, t2, 300)
        resp = t2.result()
        # resp=interface.verify_img(tid, stid, net, dataset.upper(), eps, pic_path)
        
        return json.dumps(resp,ensure_ascii=False)
    # return render_template('index_auto_verify.html')

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
        tid = inputParam["Taskid"]
        datasetparam = inputParam["DatasetParam"]
        modelparam = inputParam["ModelParam"]
        adv_methods = inputParam["AdvMethods"]
        ex_methods = inputParam["ExMethods"]
        use_layer_explain = inputParam["Use_layer_explain"]
        format_time = str(datetime.datetime.now().strftime("%Y%m%d%H%M"))
        stid = "S"+IOtool.get_task_id(str(format_time))
        value = {
            "type":"attack_attrbution_analysis",
            "state":0,
            "name":["model"],
            "dataset":datasetparam["name"],
            "method":adv_methods,
            "model":modelparam["name"],
            "exmethod":ex_methods
        }
        IOtool.add_subtask_info(tid, stid, value)
        IOtool.change_task_info(tid, "dataset", datasetparam["name"])
        IOtool.change_task_info(tid, "model", modelparam["name"])
        # 执行任务
        datasetparam["name"] = datasetparam["name"].lower()
        modelparam["name"] = modelparam["name"].lower()
        
        pool = IOtool.get_pool(tid)
        t2 = pool.submit(interface.run_attrbution_analysis, tid, stid, datasetparam, modelparam, ex_methods, adv_methods, use_layer_explain)
        IOtool.add_task_queue(tid, stid, t2, 40000 * len(ex_methods) + 30000*len(adv_methods))
        # print(t2.done())
        # print(t2.result())
        # t2 = threading.Thread(target=interface.run_attrbution_analysis,args=(tid, stid, datasetparam, modelparam, ex_methods, adv_methods, device, use_layer_explain))
        # t2.setDaemon(True)
        # t2.start()
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
        format_time = str(datetime.datetime.now().strftime("%Y%m%d%H%M"))
        stid = "S"+IOtool.get_task_id(str(format_time))
        value = {
            "type":"attack_lime",
            "state":0,
            "name":["model"],
            "dataset":datasetparam["name"],
            "method":adv_methods,
            "model":modelparam["name"],
        }
        IOtool.add_subtask_info(tid, stid, value)
        IOtool.change_task_info(tid, "dataset", datasetparam["name"])
        IOtool.change_task_info(tid, "model", modelparam["name"])
        # 执行任务
        datasetparam["name"] = datasetparam["name"].lower()
        modelparam["name"] = modelparam["name"].lower()
        
        pool = IOtool.get_pool(tid)
        t2 = pool.submit(interface.run_lime, tid, stid, datasetparam, modelparam, adv_methods)
        IOtool.add_task_queue(tid, stid, t2, 300)
        
        # t2 = threading.Thread(target=interface.run_lime,args=(tid, stid, datasetparam, modelparam, adv_methods, device))
        # t2.setDaemon(True)
        # t2.start()
        res = {
            "code":1,
            "msg":"success",
            "tid":tid,
            "stid":stid
        }
        return jsonify(res)
    else:
        abort(403)
from werkzeug.utils import secure_filename
# ----------------- 课题1 防御 -----------------
@app.route("/defense", methods=["GET", "POST"])
def home():
    return render_template("Adv_detect.html")

@app.route("/detect", methods=["POST"])
def Detect():
    
    inputParam = json.loads(request.data)
    adv_dataset = inputParam["adv_dataset"]
    adv_model = inputParam["adv_model"]
    adv_method = inputParam["adv_method"]
    adv_nums = inputParam["adv_nums"]
    try:
        defense_methods = json.loads(inputParam["defense_methods"])
    except:
        defense_methods = inputParam["defense_methods"]
    
    tid = inputParam["tid"]
    format_time = str(datetime.datetime.now().strftime("%Y%m%d%H%M"))
    stid = "S"+IOtool.get_task_id(str(format_time))
    
    
    adv_nums = int(adv_nums)
    value = {
        "type":"attack_defense",
        "state":0,
        "name":["attack_defense"],
        "dataset":adv_dataset,
        "method":defense_methods,
        "model":adv_method,
    }
    IOtool.add_subtask_info(tid, stid, value)
    IOtool.change_task_info(tid, "dataset", adv_dataset)
    IOtool.change_task_info(tid, "model", adv_model)
    
    if 'adv_examples' in request.files:
        adv_examples = request.files['adv_examples']
        # 获取文件名
        file_name = secure_filename(adv_examples.filename)
        
        # 生成唯一的文件路径
        adv_file_path = "./dataset/adv_examples/" + file_name
        # 将对抗样本文件保存到服务器上的指定位置
        adv_examples.save(adv_file_path)
    else:
        adv_file_path = None
    
    pool = IOtool.get_pool(tid)
    t2 = pool.submit(interface.run_detect, tid, stid, defense_methods, adv_dataset, adv_model, adv_method, adv_nums, adv_file_path)
    
    IOtool.add_task_queue(tid, stid, t2, 30000*len(defense_methods))
    response_data = t2.result()
    
    # response_data = interface.run_detect(tid, stid, defense_methods, adv_dataset, adv_model, adv_method, adv_nums, adv_file_path)

    return json.dumps(response_data)
# ----------------- 课题2 测试样本自动生成 -----------------
@app.route('/Concolic/SamGenParamSet', methods=['GET','POST'])
def Concolic():
    if (request.method == "GET"):
        return render_template("")
    elif (request.method == "POST"):
        concolic_dataset = request.form.get("dataname")
        concolic_model = request.form.get("modelname")
        norm = request.form.get("norm")
        times = request.form.get("times")
        tid = request.form.get("tid")
        try:
            inputdata = json.loads(request.data)
            print(inputdata)
            concolic_dataset = inputdata["dataname"]
            concolic_model = inputdata["modelname"]
            norm = inputdata["norm"]
            times = inputdata["times"]
            tid = inputdata["tid"]
        except:
            pass
        format_time = str(datetime.datetime.now().strftime("%Y%m%d%H%M"))
        AAtid = "S"+IOtool.get_task_id(str(format_time))
        value = {
            "type":"Concolic",
            "state":0,
            "name":["Concolic"],
            "dataset":concolic_dataset,
            "model": concolic_model,
            "norm": norm
        }
        
        IOtool.add_subtask_info(tid, AAtid, value)
        IOtool.change_task_info(tid, "dataset", concolic_dataset)
        IOtool.change_task_info(tid, "model", concolic_model)
        
        pool = IOtool.get_pool(tid)
        t2 = pool.submit(interface.run_concolic, tid, AAtid, concolic_dataset, concolic_model, norm, times)
        IOtool.add_task_queue(tid, AAtid, t2, 300)
        # t2 = threading.Thread(target=interface.run_concolic,args=(tid,AAtid,concolic_dataset,concolic_model,norm, times))
        # t2.setDaemon(True)
        # t2.start()
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
        value = {
            "type":"EnvTest",
            "state":0,
            "name":["EnvTest"],
            "dataset": "",
            "model": "",
            "matchmethod": matchmethod,
            "framework": frameworkname+frameversion
        }
        IOtool.add_subtask_info(tid, AAtid, value)
        
        pool = IOtool.get_pool(tid)
        t2 = pool.submit(interface.run_envtest, tid, AAtid, matchmethod, frameworkname, frameversion)
        IOtool.add_task_queue(tid, AAtid, t2, 300)
        # t2 = threading.Thread(target=interface.run_envtest,args=(tid,AAtid,matchmethod, frameworkname,frameversion))
        # t2.setDaemon(True)
        # t2.start()
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
        inputdata = json.loads(request.data)
        dataset = inputdata["dataset"]
        upload_flag = inputdata["upload_flag"]
        upload_path=''
        if upload_flag != 0:
            upload_path = inputdata["upload_path"]
        tid = inputdata["tid"]
        format_time = str(datetime.datetime.now().strftime("%Y%m%d%H%M"))
        AAtid = "S"+IOtool.get_task_id(str(format_time))
       
        value = {
            "type":"DataClean",
            "state":0,
            "name":["DataClean"],
            "dataset": dataset,
            "uoload": upload_flag,
            "model": "",
        }
        IOtool.add_subtask_info(tid, AAtid, value)
        IOtool.change_task_info(tid, "dataset", dataset)
        pool = IOtool.get_pool(tid)
        t2 = pool.submit(interface.run_dataclean, tid, AAtid, dataset, upload_flag, upload_path)
        IOtool.add_task_queue(tid, AAtid, t2, 300)
        # t2 = threading.Thread(target=interface.run_dataclean,args=(tid,AAtid,dataset))
        # t2.setDaemon(True)
        # t2.start()
        res = {"code":1,"msg":"success","Taskid":tid,"stid":AAtid}
        return jsonify(res)
    else:
        abort(403) 


# ----------------- 课题2 标准化单元测试-- -----------------
@app.route('/UnitTest/CoverageNeuralParamSet', methods=['GET','POST']) # 单神经元覆盖测试准则
def CoverageNeuralParamSet():
    if (request.method == "GET"):
        return render_template("")
    elif (request.method == "POST"):
        inputdata = json.loads(request.data)
        dataset = inputdata["dataset"]
        model = inputdata["model"]
        k = inputdata["k"]
        N = inputdata["N"]
        tid = inputdata["tid"]
        format_time = str(datetime.datetime.now().strftime("%Y%m%d%H%M"))
        AAtid = "S"+IOtool.get_task_id(str(format_time))
        value = {
            "type":"CoverageNeural",
            "state":0,
            "name":["CoverageNeural"],
            "dataset": dataset,
            "model": model,
            "threshold": k,
            "number_of_image": N
        }
        IOtool.add_subtask_info(tid, AAtid, value)
        IOtool.change_task_info(tid, "dataset", dataset)
        IOtool.change_task_info(tid, "model", model)
        pool = IOtool.get_pool(tid)
        t2 = pool.submit(interface.run_coverage_neural, tid, AAtid, dataset, model, k, N)
        IOtool.add_task_queue(tid, AAtid, t2, 300)
        
        res = {"code":1,"msg":"success","Taskid":tid,"stid":AAtid}
        return jsonify(res)
    else:
        abort(403)

@app.route('/UnitTest/CoverageLayerParamSet', methods=['GET','POST']) # 神经层覆盖测试准则
def CoverageLayerParamSet():
    if (request.method == "GET"):
        return render_template("")
    elif (request.method == "POST"):
        inputdata = json.loads(request.data)
        dataset = inputdata["dataset"]
        model = inputdata["model"]
        k = inputdata["k"]
        N = inputdata["N"]
        tid = inputdata["tid"]
        format_time = str(datetime.datetime.now().strftime("%Y%m%d%H%M"))
        AAtid = "S"+IOtool.get_task_id(str(format_time))
        value = {
            "type":"CoverageLayer",
            "state":0,
            "name":["CoverageLayer"],
            "dataset": dataset,
            "model": model,
            "threshold": k,
            "number_of_image": N
        }
        IOtool.add_subtask_info(tid, AAtid, value)
        IOtool.change_task_info(tid, "dataset", dataset)
        IOtool.change_task_info(tid, "model", model)
        pool = IOtool.get_pool(tid)
        t2 = pool.submit(interface.run_coverage_layer, tid, AAtid, dataset, model, k, N)
        IOtool.add_task_queue(tid, AAtid, t2, 300)
        res = {"code":1,"msg":"success","Taskid":tid,"stid":AAtid}
        return jsonify(res)
    else:
        abort(403)
        
@app.route('/UnitTest/CoverageImportanceParamSet', methods=['GET','POST']) # 重要神经元覆盖测试准则
def CoverageImportanceParamSet():
    if (request.method == "GET"):
        return render_template("")
    elif (request.method == "POST"):
        inputdata = json.loads(request.data)
        dataset = inputdata["dataset"]
        model = inputdata["model"]
        n_imp = inputdata["n_imp"]
        clus = inputdata["clus"]
        tid = inputdata["tid"]
        format_time = str(datetime.datetime.now().strftime("%Y%m%d%H%M"))
        # AAtid = "S20230704_1557_6aa2239"
        AAtid = "S"+IOtool.get_task_id(str(format_time))
        value = {
            "type":"CoverageImportance",
            "state":0,
            "name":["CoverageImportance"],
            "dataset": dataset,
            "model": model,
            "number_of_importance": n_imp,
            "clus": clus
        }
        IOtool.add_subtask_info(tid, AAtid, value)
        IOtool.change_task_info(tid, "dataset", dataset)
        pool = IOtool.get_pool(tid)
        t2 = pool.submit(interface.run_coverage_importance, tid, AAtid, dataset, model, n_imp, clus)
        IOtool.add_task_queue(tid, AAtid, t2, 300)
        res = {"code":1,"msg":"success","Taskid":tid,"stid":AAtid}
        return jsonify(res)
    else:
        abort(403)

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
        try:
            inputdata = json.loads(request.data)
            dataset = inputdata["dataset"]
            modelname = inputdata["model"]
            pertube = inputdata["pertube"]
            m_dir = inputdata["m_dir"]
            tid = inputdata["tid"]
        except:
            pass
        format_time = str(datetime.datetime.now().strftime("%Y%m%d%H%M"))
        AAtid = "S"+IOtool.get_task_id(str(format_time))
        value = {
            "type":"DeepSst",
            "state":0,
            "name":["DeepSst"],
            "dataset": dataset,
            "model": modelname,
            "pertube": pertube,
            "m_dir": m_dir
        }
        IOtool.add_subtask_info(tid, AAtid, value)
        IOtool.change_task_info(tid, "dataset", dataset)
        IOtool.change_task_info(tid, "model", modelname)
        pool = IOtool.get_pool(tid)
        t2 = pool.submit(interface.run_deepsst, tid, AAtid, dataset, modelname, pertube, m_dir)
        IOtool.add_task_queue(tid, AAtid, t2, 300)

        # t2 = threading.Thread(target=interface.run_deepsst,args=(tid, AAtid, dataset, modelname, pertube, m_dir))
        # t2.setDaemon(True)
        # t2.start()
        res = {"code":1,"msg":"success","Taskid":tid,"stid":AAtid}
        return jsonify(res)
    else:
        abort(403)

@app.route('/UnitTest/DeepLogicParamSet', methods=['GET','POST']) # 逻辑神经元测试准则
def DeepLogicParamSet():
    if (request.method == "GET"):
        return render_template("")
    elif (request.method == "POST"):
        dataset = request.form.get("dataset")
        modelname = request.form.get("model")
        tid = request.form.get("tid")
        try:
            inputdata = json.loads(request.data)
            dataset = inputdata["dataset"]
            modelname = inputdata["model"]
            tid = inputdata["tid"]
        except:
            pass
        format_time = str(datetime.datetime.now().strftime("%Y%m%d%H%M"))
        AAtid = "S"+IOtool.get_task_id(str(format_time))
        value = {
            "type":"DeepLogic",
            "state":0,
            "name":["DeepLogic"],
            "dataset": dataset,
            "model": modelname,
        }
        IOtool.add_subtask_info(tid, AAtid, value)
        IOtool.change_task_info(tid, "dataset", dataset)
        IOtool.change_task_info(tid, "model", modelname)
        pool = IOtool.get_pool(tid)
        t2 = pool.submit(interface.run_deeplogic, tid, AAtid, dataset, modelname)
        IOtool.add_task_queue(tid, AAtid, t2, 300)

        res = {"code":1,"msg":"success","Taskid":tid,"stid":AAtid}
        return jsonify(res)
    else:
        abort(403)       

# ----------------- 课题2 开发框架安全结构度量 -------------------
@app.route('/FWTest/FrameworkTestParamSet', methods=['GET','POST']) 
def FrameworkTestParamSet():
    if (request.method == "GET"):
        return render_template("")
    elif (request.method == "POST"):
        inputdata = json.loads(request.data)
        framework = inputdata["framework"]
        model = inputdata["model"]
        tid = inputdata["tid"]
        format_time = str(datetime.datetime.now().strftime("%Y%m%d%H%M"))
        AAtid = "S"+IOtool.get_task_id(str(format_time))
        value = {
            "type":"FrameworkTest",
            "state":0,
            "name":["FrameworkTest"],
            "dataset": "",
            "model": model,
            "framework": framework,
        }
        IOtool.add_subtask_info(tid, AAtid, value)
        IOtool.change_task_info(tid, "model", model)
        pool = IOtool.get_pool(tid)

        t2 = pool.submit( interface.run_frameworktest, tid, AAtid, model, framework)
        IOtool.add_task_queue(tid, AAtid, t2, 300)
        res = {"code":1,"msg":"success","Taskid":tid,"stid":AAtid}
        return jsonify(res)
    else:
        abort(403)

# ----------------- 课题2 模型安全度量 -------------------
@app.route('/MSTest/ModelMeasureParamSet', methods=['GET','POST']) 
def ModelMeasureParamSet():
    if (request.method == "GET"):
        return render_template("")
    elif (request.method == "POST"):
        inputdata = json.loads(request.data)
        dataset = inputdata["dataset"]
        model = inputdata["model"]
        naturemethod = inputdata["naturemethod"]
        natureargs = inputdata["natureargs"]
        advmethod = inputdata["advmethod"]
        advargs = inputdata["advargs"]
        measuremethod = inputdata["measuremethod"]
        tid = inputdata["tid"]
        format_time = str(datetime.datetime.now().strftime("%Y%m%d%H%M"))
        AAtid = "S"+IOtool.get_task_id(str(format_time))
        value = {
            "type":"ModelMeasure",
            "state":0,
            "name":["ModelMeasure"],
            "dataset": dataset,
            "model": model
        }
        IOtool.add_subtask_info(tid, AAtid, value)
        IOtool.change_task_info(tid, "dataset", dataset)
        IOtool.change_task_info(tid, "model", model)
        pool = IOtool.get_pool(tid)
        t2 = pool.submit( interface.run_modelmeasure, tid, AAtid, dataset, model, naturemethod, natureargs, advmethod, advargs, measuremethod)
        IOtool.add_task_queue(tid, AAtid, t2, 300)
        res = {"code":1,"msg":"success","Taskid":tid,"stid":AAtid}
        return jsonify(res)
    else:
        abort(403)

# ----------------- 课题2 模型模块化开发 -------------------
@app.route('/MDTest/ModularDevelopParamSet', methods=['GET','POST']) 
def ModularDevelopParamSet():
    if (request.method == "GET"):
        return render_template("")
    elif (request.method == "POST"):
        inputdata = json.loads(request.data)
        dataset = inputdata["dataset"]
        model = inputdata["model"]
        tuner = inputdata["tuner"]
        init = inputdata["init"]
        epoch = inputdata["epoch"]
        iternum = inputdata["iternum"]
        tid = inputdata["tid"]
        format_time = str(datetime.datetime.now().strftime("%Y%m%d%H%M"))
        AAtid = "S"+IOtool.get_task_id(str(format_time))
        value = {
            "type":"ModularDevelop",
            "state":0,
            "name":["ModularDevelop"],
            "dataset": dataset,
            "model": model
        }
        IOtool.add_subtask_info(tid, AAtid, value)
        IOtool.change_task_info(tid, "dataset", dataset)
        IOtool.change_task_info(tid, "model", model)
        pool = IOtool.get_pool(tid)
        t2 = pool.submit(interface.run_modulardevelop, tid, AAtid, dataset, model, tuner, init, epoch, iternum)
        IOtool.add_task_queue(tid, AAtid, t2, 3000)
        res = {"code":1,"msg":"success","Taskid":tid,"stid":AAtid}
        return jsonify(res)
    else:
        abort(403)
# ----------------- 课题3 侧信道分析 -----------------
@app.route('/SideAnalysis', methods=["POST"])
def SideAnalysis():
    if (request.method == "POST"):
        inputParam = json.loads(request.data)
        trs_file = inputParam["trs_file"]
        methods = inputParam["methods"]
        tid = inputParam["tid"]
        format_time = str(datetime.datetime.now().strftime("%Y%m%d%H%M"))
        stid = "S"+IOtool.get_task_id(str(format_time))
        value = {
            "type":"side",
            "state":0,
            "name":["side"],
            "dataset":trs_file,
            "method":methods,
            "model":"",
        }
        IOtool.add_subtask_info(tid, stid, value)
        IOtool.change_task_info(tid, "dataset", trs_file)
        pool = IOtool.get_pool(tid)
        t2 = pool.submit(interface.run_side_api, trs_file, methods, tid, stid)
        IOtool.add_task_queue(tid, stid, t2, 300)
        # interface.run_side_api(trs_file, methods, tid, stid)
        # t2 = threading.Thread(target=interface.run_side_api,args=(trs_file, methods, tid, stid))
        # t2.setDaemon(True)
        # t2.start()
        res = {"code":1,"msg":"success","Taskid":tid,"stid":stid}
        return jsonify(res)
    else:
        abort(403)
       
# ----------------- 课题4 形式化验证 -----------------

@app.route('/FormalVerification', methods=['GET',"POST"])
def FormalVerification():
    
    if (request.method == "GET"):
        return render_template("former_verification.html")
    else:
        # res = {
        #     "tid":"20230224_1106_d5ab4b1",
        #     "stid":"S20230224_1106_368e295"
        # }
        # return jsonify(res)
        inputParam = json.loads(request.data)
        param = {
            "dataset": inputParam["dataset"],
            "model": inputParam['model'],
            "size": int(inputParam["size"]),
            "up_eps": float(inputParam["up_eps"]),
            "down_eps": float(inputParam["down_eps"]),
            "steps": int(inputParam["steps"]),
            "task_id": inputParam["task_id"],
        }
        tid = inputParam["task_id"]
        format_time = str(datetime.datetime.now().strftime("%Y%m%d%H%M"))
        stid = "S"+IOtool.get_task_id(str(format_time))
        value = {
            "type":"formal_verification",
            "state":0,
            "name":["formal_verification"],
            "dataset":inputParam["dataset"],
            "model":inputParam['model']
        }
        IOtool.add_subtask_info(tid, stid, value)
        IOtool.change_task_info(tid, "dataset", inputParam["dataset"])
        IOtool.change_task_info(tid, "model", inputParam["model"])
        pool = IOtool.get_pool(tid)
        # t2 = pool.submit(interface.run_modulardevelop, tid, stid, param)
        t2 = pool.submit(interface.submitAandB, tid, stid, 1, 2)
        # t2 = pool.submit(interface.run_verify, tid, stid, param)
        IOtool.add_task_queue(tid, stid, t2, 30000)
        interface.run_verify(tid, stid, param)
        # t2 = threading.Thread(target=interface.run_verify, args=(tid, stid, param))
        # t2.setDaemon(True)
        # t2.start()
        res = {
            "tid":tid,
            "stid":stid
        }
        return jsonify(res)

@app.route('/Defense/Ensemble', methods=['POST'])
def ensemble():
    """
    群智化防御
    输入：tid：主任务ID
    Dataset：数据集名称
    Model：模型名称
    AdvMethods:list 对抗攻击算法名称
    ExMethods:攻击机理解释方法名称
    """
    global LiRPA_LOGS
    if (request.method == "POST"):
        inputParam = json.loads(request.data)
        tid = inputParam["Taskid"]
        datasetparam = inputParam["DatasetParam"]
        modelparam = inputParam["ModelParam"]
        adv_methods = inputParam["AdvMethods"]
        defense_methods = inputParam["DefMethod"]
        format_time = str(datetime.datetime.now().strftime("%Y%m%d%H%M"))
        stid = "S"+IOtool.get_task_id(str(format_time))
        value = {
            "type":"Defense_Ensemble",
            "state":0,
            "name":["Defense_Ensemble"],
            "dataset":datasetparam["name"],
            "method":adv_methods,
            "model":modelparam["name"],
            "def_method":defense_methods
        }
        IOtool.add_subtask_info(tid, stid, value)
        IOtool.change_task_info(tid, "dataset", datasetparam["name"])
        IOtool.change_task_info(tid, "model", modelparam["name"])
        # 执行任务
        datasetparam["name"] = datasetparam["name"]
        modelparam["name"] = modelparam["name"]
        adv_param = {}
        for temp in adv_methods:
            adv_param.update({temp:inputParam[temp]})
        pool = IOtool.get_pool(tid)
        # t2 = pool.submit(interface.run_ensemble_defense, tid, stid, datasetparam, modelparam, adv_methods, adv_param, defense_methods)
        t2 = pool.submit(interface.run_group_defense, tid, stid, datasetparam, modelparam, adv_methods, adv_param, defense_methods)
        # t2 = pool.submit(interface.submitAandB, tid, stid, 10, 20)
        IOtool.add_task_queue(tid, stid, t2, 72000*len(adv_methods))
        # interface.run_group_defense(tid, stid, datasetparam, modelparam, adv_methods, adv_param, defense_methods)
        res = {
            "code":1,
            "msg":"success",
            "tid":tid,
            "stid":stid
        }
        return jsonify(res)
    else:
        abort(403)

@app.route('/GraphKnowledge', methods=['POST'])
def graph_knowledge():
    """
    知识图谱
    输入：tid：主任务ID
    Dataset：数据集名称
    Model：模型名称
    AdvMethods:list 对抗攻击算法名称
    ExMethods:攻击机理解释方法名称
    """
    global LiRPA_LOGS
    if (request.method == "POST"):
        inputParam = json.loads(request.data)
        tid = inputParam["Taskid"]
        datasetparam = inputParam["DatasetParam"]
        modelparam = inputParam["ModelParam"]
        attack_mode = inputParam["attack_mode"]
        attack_type = inputParam["attack_type"]
        data_type = inputParam['data_type']
        defend_algorithm = inputParam['defend_algorithm']
        format_time = str(datetime.datetime.now().strftime("%Y%m%d%H%M"))
        stid = "S"+IOtool.get_task_id(str(format_time))
        value = {
            "type":"Graph_Knowledge",
            "state":0,
            "name":["Graph_Knowledge"],
            "dataset":datasetparam["name"],
            "method":attack_mode,
            "model":modelparam["name"],
            "def_method":defend_algorithm
        }
        IOtool.add_subtask_info(tid, stid, value)
        IOtool.change_task_info(tid, "dataset", datasetparam["name"])
        IOtool.change_task_info(tid, "model", modelparam["name"])
        # 执行任务
        datasetparam["name"] = datasetparam["name"]
        modelparam["name"] = modelparam["name"]
        pool = IOtool.get_pool(tid)
        # t2 = pool.submit(interface.run_group_defense, tid, stid, datasetparam, modelparam, adv_methods, adv_param, defense_methods)
        t2 = pool.submit(interface.submitAandB, tid, stid, 10, 20)
        IOtool.add_task_queue(tid, stid, t2, 72000)
        interface.run_graph_knowledge(tid, stid, datasetparam, modelparam, attack_mode, attack_type, data_type, defend_algorithm)
        res = {
            "code":1,
            "msg":"success",
            "tid":tid,
            "stid":stid
        }
        return jsonify(res)
    else:
        abort(403)

@app.route('/Task/UploadData', methods=['POST'])
def UploadData():
     if request.method == "POST":
         file = request.files.get('file')
         data_type = request.form.get('type')
         file_name = file.filename
         suffix = os.path.splitext(file_name)[-1]
         basePath = os.path.abspath(os.path.dirname(__file__)).rsplit('/', 1)
         save_dir = os.path.join(basePath[0], 'dataset/data/ckpt', data_type+'_data'+suffix)
         # file_name = request.form.get()
        #  save_dir = os.path.join(basePath, 'upload_data'+suffix)
         file.save(save_dir)
         return jsonify({'save_dir': save_dir})

@app.route('/Task/DownloadData', methods=['POST'])
def DownloadData():
     if request.method == "POST":
            download_path = request.form.get('file')
            input_path,input_name=os.path.split(download_path)
            if os.path.isdir(download_path): 
                print("it's a directory")
                outFullName = input_path +'/'+ input_name+'.zip'
                # if not os.path.exists(outFullName):
                interface.zipDir(download_path, outFullName)
                return send_from_directory(input_path,
                                    input_name+'.zip', as_attachment=True)
            elif os.path.isfile(download_path):
                print("it's a normal file")
                if ROOT not in input_path:
                    input_path = osp.join(ROOT, input_path)
                import zipfile
                outFullName = osp.join(input_path,"download.zip")
                if osp.exists(outFullName):
                    os.remove(outFullName)
                zip = zipfile.ZipFile(outFullName, "w", zipfile.ZIP_DEFLATED)
                zip.write(os.path.join(input_path, input_name),input_name)
                zip.close
                
                # response = make_response()
                # print(response)
                down_path,down_name = os.path.split(outFullName)
                print(down_path)
                print(down_name)
                time.sleep(1)
                return send_from_directory(down_path,
                                    input_name, as_attachment=True)
            else:
                return jsonify(bool=False, msg='No such file, please check file path')

@app.route('/Task/UploadPic', methods=['POST'])
def UploadPic():
    if request.method == "POST":
        file = request.files.get('avatar')
        basePath = os.path.abspath(os.path.dirname(__file__)).rsplit('/', 1)
        save_dir = os.path.join(basePath[0], 'dataset/data/ckpt',"upload.jpg")
        file.save(save_dir)
        return jsonify({'save_dir': save_dir})

@app.route('/Attack/LLM_attack', methods=['POST'])
def LLM_attack():
    if (request.method == "POST"):
        inputParam = json.loads(request.data)
        tid = inputParam["Taskid"]
        modelname = inputParam['model']
        goal = inputParam['goal']
        target = inputParam['target']
        format_time = str(datetime.datetime.now().strftime("%Y%m%d%H%M"))
        stid = "S"+IOtool.get_task_id(str(format_time))
        value = {
            "type":"LLM_attack",
            "state":0,
            "name":["LLM_attack"],
            "dataset":goal,
            "method":"LLM attack",
            "model":modelname,
        }
        print(goal)
        print(target)
        IOtool.add_subtask_info(tid, stid, value)
        pool = IOtool.get_pool(tid)
        # t2 = pool.submit(interface.submitAandB, tid, stid, 1, 2)
        t2 = pool.submit(interface.llm_attack, tid, stid, goal, target)
        IOtool.add_task_queue(tid, stid, t2, 30000)
        # interface.llm_attack(tid, stid, goal, target)
        res = {
            "code":1,
            "msg":"success",
            "tid":tid,
            "stid":stid
        }
        return jsonify(res)
def app_run(args):
    web_config={'host':args.host,'port':args.port,'debug':args.debug}
    app.run(**web_config)
    

    

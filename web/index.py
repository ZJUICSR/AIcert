#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os.path as osp
import interface
import os, json, datetime, pickle, io, ast, time
import pytz,shutil

from IOtool import IOtool
from flask import render_template, redirect, url_for, Flask, request, jsonify, send_from_directory
from flask import current_app as abort
from multiprocessing import Process


from function.ex_methods.module.func import Logger
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

@app.route('/Task/UploadPic', methods=['POST'])
def UploadPic():
    if request.method == "POST":
        file = request.files.get('avatar')
        basePath = os.path.abspath(os.path.dirname(__file__)).rsplit('/', 1)
        print(basePath)
        save_dir = os.path.join(basePath[0], 'dataset/data/ckpt',"upload_lime.jpg")
        file.save(save_dir)
        return jsonify({'save_dir': save_dir})


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
    VisMethods:list 降维方法名称
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
        
        IOtool.add_task_queue(tid, stid, t2, 400 * len(vis_methods) + 300*len(adv_methods))
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
        IOtool.add_task_queue(tid, stid, t2, 400 * len(ex_methods) + 300*len(adv_methods))
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

@app.route('/Attack/AdversarialAnalysis', methods=['POST'])
def AdversarialAnalysis():
    """
    对抗图像归因与降维解释集成api
    输入：tid：主任务ID
    Dataset：数据集名称
    Model：模型名称
    AdvMethods:list 对抗攻击算法名称
    ExMethods:攻击机理解释方法名称
    vis_methods: list 降维方法名称
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
        vis_methods = inputParam["VisMethods"]
        format_time = str(datetime.datetime.now().strftime("%Y%m%d%H%M"))
        stid = "S"+IOtool.get_task_id(str(format_time))
        value = {
            "type":"adversarial_analysis",
            "state":0,
            "name":["model"],
            "dataset":datasetparam["name"],
            "method":adv_methods,
            "model":modelparam["name"],
            "exmethod":ex_methods,
            "vis_methods":vis_methods
        }
        IOtool.add_subtask_info(tid, stid, value)
        IOtool.change_task_info(tid, "dataset", datasetparam["name"])
        IOtool.change_task_info(tid, "model", modelparam["name"])
        # 执行任务
        datasetparam["name"] = datasetparam["name"].lower()
        modelparam["name"] = modelparam["name"].lower()
        
        pool = IOtool.get_pool(tid)
        t2 = pool.submit(interface.run_adversarial_analysis, tid, stid, datasetparam, modelparam, ex_methods, vis_methods, adv_methods, use_layer_explain)
        IOtool.add_task_queue(tid, stid, t2, 300 * len(vis_methods) + 400 * len(ex_methods) + 300*len(adv_methods))
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
        tid = inputParam["Taskid"]
        datasetparam = inputParam["DatasetParam"]
        modelparam = inputParam["ModelParam"]
        adv_methods = inputParam["AdvMethods"]
        data_mode = inputParam["mode"]
        format_time = str(datetime.datetime.now().strftime("%Y%m%d%H%M"))
        stid = "S"+IOtool.get_task_id(str(format_time))
        value =  {
            "type":"attack_lime",
            "state":0,
            "name":["adv_attack"],
            "dataset":datasetparam["name"],
            "method":adv_methods,
            "model":modelparam["name"],
            "mode":data_mode
        }
        IOtool.add_subtask_info(tid, stid, value)
        IOtool.change_task_info(tid, "dataset", datasetparam["name"])
        IOtool.change_task_info(tid, "model", modelparam["name"])

        # 执行任务
        datasetparam["name"] = datasetparam["name"].lower()
        modelparam["name"] = modelparam["name"].lower()

        pool = IOtool.get_pool(tid)
        t2 = pool.submit(interface.run_lime, tid, stid, datasetparam, modelparam, adv_methods, data_mode)
        IOtool.add_task_queue(tid, stid, t2,  400*len(adv_methods))
        res = {
            "code":1,
            "msg":"success",
            "tid":tid,
            "stid":stid
        }
        return jsonify(res)
    else:
        abort(403)


def app_run(args):
    web_config={'host':args.host,'port':args.port,'debug':args.debug}
    app.run(**web_config)

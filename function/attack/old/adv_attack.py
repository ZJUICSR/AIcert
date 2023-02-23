import os
import os.path as osp
import ast, datetime, time, copy
from IOtool import IOtool, Logger, Callback
from functions.attack.config import Conf
from gol import Taskparam
from multiprocessing import Process
from interface import get_data_loader, get_model_loader
from torchvision import  transforms
from functions.attack.old.adv import Attack
from models.trainer.adv_robust import RobustTrainer

ROOT =os.getcwd()


def run_adv_attack_defense(methods,  taskparam):
    """
    执行对抗攻击
    :param methods: 攻击方法名称,list
    :param AAtid:对抗攻击子任务ID
    :param taskparam:任务全局变量参数,包含任务信息info,参数信息params,输出result
    其中params的结构为
    {
        'out_path':结果保存路径,
        'cache_path': 中间数据保存路径，如图片,
        "device": 指定GPU卡号,
        "IsAdvAttack":是否进行对抗攻击,
        "IsAdvTrain":是否进行对抗训练,
        "IsEnsembleDefense":是否进行群智化防御,
        "IsPACADetect":是否进行PACA检测,
        'model': {
            'name': 模型名称,
            'path': 模型路径，预置参数,
            'pretrained': 是否预训练
            },
        'dataset': {
            'name': 数据集名称
            'num_classes': 分类数
            'path': 数据集路径，预置参数
            'batch_size': 批量处理数
            'bounds': [-1, 1],界限
            'mean': [0.4914, 0.4822, 0.4465],各个通道的均值
            'std': [0.2023, 0.1994, 0.201]},各个通道的标准差
    }
    """
    try:
        taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
        # 修改任务状态
        params = taskparam.get_params()
        print(params)
        AAtid = params["stid"]
        # if taskinfo[params["tid"]]["state"] != 1:
        taskparam.set_info_value(key="state",value=1)
        info = taskparam.get_info_value(key="function")
        info[params["stid"]]["state"]=1
        taskparam.set_info_value(key="function",value=info)
        taskinfo[params["tid"]]=copy.deepcopy(taskparam.get_info())
        IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
        
        # 日志模块
        logging = Logger(filename=osp.join(params["out_path"], AAtid + "_log.txt"))
        logging.info('[初始化系统]: 系统运行在{:s}设备上'.format(str(params["device"])))

        # 加载数据
        taskparam.set_res_value("out_path",params["out_path"])
        taskparam.set_res_value("AAtid",AAtid)
        defenselist = taskinfo[params["tid"]]["function"][AAtid]["name"]
        logging.info('[加载数据]: 加载{:s}数据集'.format(str(params["dataset"]["name"])))
        
        if params["dataset"]["upload_flag"] == 1:
            test_loader = get_data_loader(params["dataset"]["path"],  params["dataset"]["name"], params["dataset"])
            mean, std = IOtool.get_mean_std(test_loader)
            params["dataset"]["mean"] = mean.tolist()
            params["dataset"]["std"] = std.tolist()
        transform =transforms.Compose([
                        transforms.RandomCrop([32,32], padding=2),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize( params["dataset"]["mean"], params["dataset"]["std"]),
                    ])
        logging.info("[数据处理阶段] 正在进行数据处理")
        # 内置数据集
        if "Custom" not in params["dataset"]["path"]:
            test_loader,train_loader,params["dataset"] = get_data_loader(params["dataset"]["path"],  params["dataset"]["name"], params["dataset"],transform=transform)
            
            # 模型加载
            model,trainer = get_model_loader(params["dataset"], params["model"], logging, train_loader , test_loader,taskparam)
        else:
            # 上传数据集
            test_loader =  get_data_loader(params["dataset"]["path"],  params["dataset"]["name"], params["dataset"],transform=transform)
            model,trainer = get_model_loader(dataset =params["dataset"], model=params["model"], logging=logging, test_loader=test_loader,taskparam=taskparam)
        # 对抗训练
        logging.info('[模型测试阶段] 即将进行的对抗样本测试算法有：{:s}'.format(str(str(methods))))
        num_atk = len(methods)
        advres=taskparam.get_res_value(key="AdvAttack")
        advres["atk_acc"]={}# 加固前对抗攻击成功率
        advres["atk_asr"]={}# 加固前对抗攻击失败率
        adv_dataloader={}#对抗样本集合
        for i,method in enumerate(methods):
            # params[method] = atk_conf.get_method_param(method)
            msg = "[模型测试阶段] 正在运行对抗样本攻击:{:s}...[{:d}/{:d}]".format(method, i + 1, num_atk)
            atk = Attack(method=method, model=model.cpu(), params=params)
            result = atk.run(test_loader)
            # 对抗样本
            adv_loader = copy.deepcopy(result["adv_loader"])
            # 对抗攻击结果，加固前
            adv_test_acc, adv_test_loss = trainer.test(model, test_loader=adv_loader)
            result["atk_acc"] = adv_test_acc
            result["atk_asr"] = 100.0 - adv_test_acc
            result["atk_loss"] = adv_test_loss
            advres["atk_acc"][method] = adv_test_acc
            advres["atk_asr"][method] = 100.0 - adv_test_acc
            adv_dataloader[method] = adv_loader
            del result["adv_loader"]
            del result["x_ben"]
            del result["x_adv"]
            del result["y_ben"]
            del result["y_adv"]
            del result["prob_ben"]
            del result["prob_adv"]
            del atk
            print("-> Attack ASR={:s}".format(str(result['var_asr'])))
            advres[method] = result
            logging.info("[模型测试阶段] 完成对抗样本测试算法{:s}，预训练模型准确率：{:.3f}% 测试对抗样本攻击的准确率（鲁棒性）为：{:.3f}%".format(
                method, advres["test_acc"], adv_test_acc)
            )
        
        logging.info('[对抗攻击阶段]完成对抗攻击测试')
        # 如果子任务只有对抗攻击
        if defenselist == ["AdvAttack"]:
            # 保存结果数据
            taskparam.set_res_value("AdvAttack",advres)
            # 修改任务状态
            IOtool.change_subtask_state(taskparam=taskparam,tid=params["tid"],stid=AAtid,state=2)
            # 写入结果
            taskres = taskparam.get_result()
            try:
                del taskres["AdvTrain"]
                del taskres["EnsembleDefense"]
                del taskres["PACA"]
            except:
                pass
            IOtool.change_task_success(taskparam=taskparam,tid=params["tid"])
            state = taskparam.get_info_value(key="state")
            if state not in [0,1]:
                taskres["stop"]=1
            IOtool.write_json(taskres,osp.join(ROOT,"output",params["tid"],AAtid+"_result.json"))
            return 1
        # 对抗训练
        trainerParam = taskparam.get_params_value("trainer")
        print("****************trainerParam************************************\n",trainerParam)
        adv_trainer = RobustTrainer(**trainerParam)
        robust_models = {}
        model = model.cpu()
        trainres = taskparam.get_res_value("AdvTrain")
        trainres["adv_train_model_path"] = {}
        trainres["def_acc"]={}
        trainres["def_asr"]={}
        logging.info('[对抗训练阶段]即将执行对抗训练算法')
        for i, method in enumerate(methods):
            logging.info('[对抗训练阶段] 测试任务编号：{:s}'.format(AAtid))
            logging.info('[对抗训练阶段]使用对抗样本算法{:s}生成的样本作为鲁棒训练数据，训练鲁棒性强的模型'.format(method))
            """专门针对上传的对抗样本设计，仅做测试不做鲁棒训练"""
            # _eps = params[method]["eps"]
            # for rate in [0.9, 1.0, 1.1]:
            #     params[method]["eps"] = _eps * rate
            copy_model = copy.deepcopy(model)
            logging.info('[对抗训练阶段]使用对抗样本算法{:s}生成的样本作为鲁棒训练数据，eps参数为：{:.3f}'.format(method, float(
                params[method]["eps"])))
            def_method = "{:s}_{:.5f}".format(method, float(params[method]["eps"]))
            atk = Attack(method=method, model=copy_model, params=params)
            cahce_weights = IOtool.load(arch=params["model"]["name"], task=params["dataset"]["name"], tag=def_method)
            if cahce_weights is None:
                logging.info('[对抗训练阶段]缓存模型不存在，开始模型鲁棒训练（这步骤耗时较长）')
                logging.info('[对抗训练阶段] 测试任务编号：{:s}'.format(AAtid))
                # copy_model=copy_model.to(device)
                rst_model = adv_trainer.robust_train(copy_model, train_loader, test_loader=test_loader,
                                                        adv_loader=adv_dataloader[method], atk=atk, epochs=50,
                                                        atk_method=method, def_method=def_method
                                                        )
                IOtool.save(model=rst_model, arch=params["model"]["name"], task=params["dataset"]["name"], tag=def_method)
            else:
                logging.info('[对抗训练阶段]从默认文件夹载入缓存模型')
                copy_model.load_state_dict(cahce_weights)
                rst_model = copy.deepcopy(copy_model).cpu()

            # add robust prob scatter
            print(f"-> method:{method} eps:{params[method]['eps']}")
            ben_prob, _, _ = trainer.test_batch(copy_model, test_loader)
            normal_adv_prob, _, _ = trainer.test_batch(copy_model, adv_dataloader[method])
            robust_adv_prob, _, _ = trainer.test_batch(rst_model, adv_dataloader[method])
            normal_prob = atk.prob_scatter(ben_prob, normal_adv_prob)
            robust_prob = atk.prob_scatter(ben_prob, robust_adv_prob)
            advres[method]["normal_scatter"] = normal_prob.tolist()
            advres[method]["robust_scatter"] = robust_prob.tolist()
            robust_models[def_method] = rst_model.cpu()
            test_acc, test_loss = trainer.test(rst_model, test_loader=test_loader)
            adv_test_acc, adv_test_loss = trainer.test(rst_model, test_loader=adv_dataloader[method])
            logging.info(
                "[对抗训练阶段]鲁棒训练方法'{:s}'结束，模型准确率为：{:.3f}%，模型鲁棒性为：{:.3f}%".format(def_method, test_acc,
                                                                                        adv_test_acc))
            
            trainres["def_acc"][str(method)] = adv_test_acc
            trainres["def_asr"][str(method)] = 100.0 - adv_test_acc
            advres[method]["def_asr"] = 100.0 - adv_test_acc

            """记录下鲁棒训练缓存模型，提供测试人员下载用"""
            dataname = params["dataset"]["name"]
            modelname = params["model"]["name"]
            tmp_path = osp.join(f"/models/ckpt/{dataname}_{modelname}_{def_method}.pt")
            # results["cycle"][9]["defense"]["rbst_model_path"] = {}
            trainres["adv_train_model_path"][def_method] = tmp_path
            del atk
            
        
        #对抗训练结果写入
        taskparam.set_res_value(key="AdvAttack",value=advres )
        taskparam.set_res_value(key="AdvTrain",value=trainres )

        # 群智化防御
        if "EnsembleDefense" not in defenselist:
            taskres = taskparam.get_result()
            try:
                del taskres["EnsembleDefense"]
                taskparam.change_res(taskres)
            except:
                pass
        else:
            from evaluator import ensemble_defense
            logging.info('[群智化防御阶段]即将执行群智化防御算法，利用多智能体（每个鲁棒训练即是一种智能体）集成输出')
            ens_dataloader = adv_dataloader.copy()
            ens_model_list = list(robust_models.values())
            ens_model = ensemble_defense.run(model_list=ens_model_list)
            ensres = taskparam.get_res_value(key="EnsembleDefense")
            ensres["ens_asr"] = {}
            ensres["ens_acc"] = {}
            # ensres["ens_model"] = copy.deepcopy(ens_model)
            for method, adv_loader in ens_dataloader.items():
                test_acc, test_loss = trainer.test(ens_model, test_loader=adv_loader, device=torch.device("cpu"))
                ensres["ens_acc"][method] = float(test_acc)
                ensres["ens_asr"][method] = 100.0 - float(test_acc)
            taskparam.set_res_value(key="EnsembleDefense",value=ensres )
            logging.info("[群智化防御阶段]群智化防御方法算法运行结束")
        
        # PACA
        if "PACA" not in defenselist:
            taskres = taskparam.get_result()
            try:
                del taskres["PACA"]
                taskparam.change_res(taskres)
            except:
                pass
        else:
            from evaluator import paca_detect
            logging.info("[自动化攻击监测阶段] 即将运行自动化攻击监测、算法加固算法：paca_detect")
            paca_detect.run(model=copy_model, test_loader=test_loader, adv_dataloader=adv_dataloader, params=params,taskparam=taskparam,
                            log_func=logging.info)
            logging.info("[自动化攻击监测阶段]自动化攻击监测、算法加固方法：paca_detect运行结束")
            logging.info('[攻防测试] 测试任务编号：{:s},测试任务结束'.format(AAtid))
        taskres = taskparam.get_result()
        # print(taskres)
        IOtool.change_subtask_state(taskparam=taskparam,tid=params["tid"],stid=AAtid,state=2)
        IOtool.change_task_success(taskparam=taskparam,tid=params["tid"])
        state = taskparam.get_info_value(key="state")
        if state not in [0,1]:
            taskres["stop"]=1
        IOtool.write_json(taskres,osp.join(ROOT,"output",params["tid"],AAtid+"_result.json"))
    except:
        taskres = taskparam.get_result()
        IOtool.change_subtask_state(taskparam=taskparam,tid=params["tid"],stid=AAtid,state=3)
        taskparam.set_info_value(key="state",value=3)
        taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
        taskinfo[params["tid"]]=copy.deepcopy(taskparam.get_info())
        IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
        taskres["stop"]=1

        return 1

def adv_attack(advInputData):
    data_path = osp.join(ROOT, "dataset/data")
    # 界面输入使用
    # advInputDatastr = list(request.form.to_dict())[0]
    # advInputData = json.loads(advInputDatastr)
    # postman输入使用
    # advInputData=json.loads(request.data)
    # 参数校验
    for method in advInputData["Method"]:
        for method_key in advInputData[method].keys():
            if method_key in ["steps","inf_batch","popsize", "pixels", "sampling", "n_restarts","n_classes","eot_iter", "n_queries"]:
                advInputData[method][method_key]=int(advInputData[method][method_key])
            elif method_key == "attacks":
                advInputData[method][method_key]=ast.literal_eval(advInputData[method][method_key])
            elif method_key not in ["random_start", "norm", "loss", "version"]:
                advInputData[method][method_key]=float(advInputData[method][method_key])
    # print(advInputData)
    tid = advInputData["Taskid"]
    data_name = advInputData["Dataset"]["name"]
    model_name = advInputData["Model"]["name"].lower()
    pretrained = advInputData["Model"]["pretrained"]
    uploaddata_flag = advInputData["Dataset"]["upload_flag"]
    # methods = ast.literal_eval(advInputData["Method"])
    methods = advInputData["Method"]
    format_time = str(datetime.datetime.now().strftime("%Y%m%d%H%M"))
    AAtid = "S"+IOtool.get_task_id(str(format_time))
    outpath = osp.join(ROOT,"output",tid)
    cachepath = osp.join(ROOT,"output/cache")
    # 使用new做对抗攻击
    # attack_Eof = run_adv_attack(advInputData["Method"], advInputData["Dataset"]["name"].lower(), tid)
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
    if uploaddata_flag == "1":
        
        attack_params["dataset"]={"name":data_name}
        attack_params["dataset"]["path"] = osp.join(data_path,"Custom")
        iniInfo=IOtool.load_json(osp.join(osp.join(data_path,"Custom",data_name),'info.json'))
        attack_params["dataset"]["num_classes"] = iniInfo['class_num']
        attack_params["dataset"]["batch_size"] = 256
        attack_params["dataset"]["bounds"]=[-1, 1]
    else:
        atk_conf = Conf()
        attack_params["dataset"]=atk_conf.get_dataset_param(data_name)
        attack_params["dataset"]["path"] = data_path
    attack_params["dataset"]["upload_flag"] = uploaddata_flag
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
    taskparam = Taskparam(taskinfo[tid],attack_params,result)
    # ------------------读取缓存结果------------------------
    if advInputData["IsAdvAttack"] == 2:
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
        # ------------------读取结果结束------------------------
    else:
        print("进入执行环节……")
        p = Process(target=run_adv_attack_defense, args=(methods, taskparam))
        p.start()
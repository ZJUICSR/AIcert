import json, datetime, pytz, hashlib, time, logging
import os,sys,copy, inspect, ctypes
import os.path as osp
import cv2
import numpy as np
import torch, gpustat, random
from tqdm import tqdm
from collections import OrderedDict
import threading
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from function.ex_methods.module.func import Logger

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)
TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
ROOT = osp.dirname(osp.abspath(__file__))
POOLNUM = 4 # 线程池4个
THREADNUM = 4 # 每个线程池中的线程数4个

class IOtool:
    __taskinfoclock = threading.Lock()
    # {tid:{stid1:{name:feature,starttime:time,outtime:time}}}
    __task_queue = {}
    __task_queue_lock = threading.Lock()
    __logger = Logger()
    __poollist = []
    __curpoollist = {}

    @staticmethod
    def add_pool(tid):
        IOtool.__task_queue_lock.acquire()
        if len(IOtool.__curpoollist) < POOLNUM:

            pool = ThreadPoolExecutor(max_workers=THREADNUM, thread_name_prefix=tid)
            IOtool.__curpoollist.update({tid:{"pool":pool}})
        else:

            IOtool.__poollist.append(tid)
        IOtool.__task_queue_lock.release()
    
    @staticmethod
    def get_pool(tid):
        # 判断tid是否在任务队列中，不在就增加到队列中
        if tid not in IOtool.__curpoollist.keys() and tid not in IOtool.__poollist:
            IOtool.add_pool(tid)
        
        # 判断tid是否在当前执行的任务队列中，不在就等待，在就添加返回pool
        while tid not in IOtool.__curpoollist.keys():
            time.sleep(10)

        return IOtool.__curpoollist[tid]["pool"]
        
    # @staticmethod
    # def task_queue_init(tid):
    #     IOtool.__task_queue_lock.acquire()
    #     IOtool.__task_queue[tid] = {}
    #     IOtool.__task_queue_lock.release()
    

    @staticmethod
    def add_task_queue(tid, stid, feature, outtime):
        IOtool.__task_queue_lock.acquire()
        IOtool.__logger.add_logger(stid=stid, filename=osp.join(ROOT,"output", tid, stid +"_log.txt"))
        if tid in IOtool.__curpoollist.keys():
            IOtool.__curpoollist[tid].update({stid: {"name":feature,"outtime":outtime}})
            msg = 'add success'
        else:
            msg = 'add fail'
        IOtool.__task_queue_lock.release()
        return msg
    
    @staticmethod
    def get_logger(stid=None, tid=None):
        if stid:
            if IOtool.__logger.get_sub_logger(stid) == -1 and tid:
                print("---------111----")
                IOtool.__logger.add_logger(stid=stid, filename=osp.join(ROOT,"output", tid, stid +"_log.txt"))
                print("IOtool.__logger.get_sub_logger(stid):",IOtool.__logger.get_sub_logger(stid))
            return IOtool.__logger.get_sub_logger(stid)
        else:
            return IOtool.__logger
    
    @staticmethod
    def set_task_starttime(tid, stid, timevalue):
        IOtool.__task_queue_lock.acquire()
        IOtool.__curpoollist[tid][stid]["starttime"] = timevalue
        IOtool.__task_queue_lock.release()
    
    # @staticmethod
    # def query_task_queue():
    #     IOtool.__task_queue_lock.acquire()
    #     task = IOtool.__task_queue.keys()
    #     IOtool.__task_queue_lock.release()
    #     return IOtool.__task_queue
     
    @staticmethod
    def check_feature_state(task_queue):
        """
        feature 正常执行且未超时，检查下个tid
        feature 全部执行成功，删除__curpoollist中的tid
        feature 超时任务数小于4等于stidlist，删除__curpoollist中的tid
        feature 超时任务数大于THREADNUM，删除__curpoollist中的tid
        """ 
        tidlist = list(task_queue.keys())

        for tid in tidlist:
            stidlist = list(task_queue[tid].keys())
            stidlist.remove("pool")

            timeout_num = 0 # 记录超时数
            for stid in stidlist:
                future = task_queue[tid][stid]["name"]
                if future.done():
                    try:
                        data = future.result()
                    except Exception as e:
                        #异常结束 写日志，改子/主状态，删进程
                        IOtool.__logger.info(stid, f"[任务异常结束]：异常信息 {e}")
                        if osp.exists(osp.join(ROOT,"output", tid, stid+"_result.json")):
                            resp=IOtool.load_json(osp.join(ROOT,"output", tid, stid+"_result.json"))
                        else:
                            resp={}
                        resp["stop"] = 2
                        IOtool.write_json(resp, osp.join(ROOT,"output", tid, stid+"_result.json"))
                        IOtool.change_subtask_state(tid, stid, 3)
                        IOtool.change_task_success_v2(tid=tid)
                    del task_queue[tid][stid]
                    IOtool.__logger.del_logger(stid)
                # 判断超时
                else:
                    if "starttime" in  task_queue[tid][stid].keys():
                        runtime = time.time()-task_queue[tid][stid]["starttime"]
                        if runtime >task_queue[tid][stid]["outtime"] and not IOtool.__taskinfoclock.locked():
                            timeout_num += 1
                            #任务超时：写日志，改子/主状态，删进程
                            taskinfo = IOtool.get_task_info()
                            if taskinfo[tid]['function'][stid]['state'] != 4:
                                if osp.exists(osp.join(ROOT,"output", tid, stid+"_result.json")):
                                    resp=IOtool.load_json(osp.join(ROOT,"output", tid, stid+"_result.json"))
                                else:
                                    resp={}
                                resp["stop"] = 2
                                IOtool.write_json(resp, osp.join(ROOT,"output", tid, stid+"_result.json"))
                                IOtool.__logger.info(stid, f"[任务异常结束]：{stid}任务超时")
                                IOtool.change_subtask_state(tid, stid, 4)
                                IOtool.change_task_success_v2(tid=tid)
            taskinfo = IOtool.get_task_info()

            try:
                if len(task_queue[tid]) == 1 and taskinfo[tid]["state"] != 0:
                    task_queue[tid]["pool"].shutdown()
                    del task_queue[tid]
                    print("子任务全部执行成功")
                if timeout_num == THREADNUM :
                    print("timeout shutdown1")
                    print(task_queue[tid]["pool"].shutdown(wait=False))
                    del task_queue[tid]
                    print("任务失败，线程池中的任务全部超时")
                if timeout_num == len(stidlist):
                    print("timeout shutdown2")
                    print(task_queue[tid]["pool"].shutdown(wait=False))
                    del task_queue[tid]
                    print("任务失败，未执行成功部分子任务全部超时")
            except Exception as e :
                print(e)
        return task_queue
        
     
        
    @staticmethod
    def check_sub_task_threading():
        """
        1、检查当前执行tid和未执行tid是否有元素
        __curpoollist 无，__poollist无，等待
        __curpoollist 无，__poollist有，则创建pool，添加到cur，poolist中删除对应tid
        __curpoollist 有且小于 POOLNUM，__poollist无，检查__curpoollist中各个tid对应的feature状态
            feature 正常执行且未超时，检查下个tid
            feature 全部执行成功，删除__curpoollist中的tid
            feature 未执行成功部分全部超时，删除__curpoollist中的tid
            feature 超时任务数大于THREADNUM，删除__curpoollist中的tid
        __curpoollist 有且小于 POOLNUM，__poollist有
            检查__curpoollist中各个tid对应的feature状态，
            则创建pool，添加到cur，poolist中删除对应tid
        __curpoollist 有大于 POOLNUM，检查__curpoollist中各个tid对应的feature状态
        
        """
        while True:
            try:
                time.sleep(10)
                IOtool.__task_queue_lock.acquire()

                if len(IOtool.__curpoollist) == 0:
                    if len(IOtool.__poollist) == 0:
                        pass
                    else:

                        pool = ThreadPoolExecutor(max_workers=THREADNUM, thread_name_prefix=tid)
                        IOtool.__curpoollist.update({tid:{"pool":pool}})
                        IOtool.__poollist.remove(tid)
                else:
                    IOtool.__curpoollist = IOtool.check_feature_state(IOtool.__curpoollist)

                    if len(IOtool.__curpoollist) < POOLNUM and len(IOtool.__poollist) > 0:

                        tid = IOtool.__poollist[0]
                        pool = ThreadPoolExecutor(max_workers=THREADNUM, thread_name_prefix=tid)
                        IOtool.__curpoollist.update({tid:{"pool":pool}})
                        IOtool.__poollist.remove(tid)
                IOtool.__task_queue_lock.release()
                time.sleep(20)
            except Exception as e:
                print(e)
        # return 0
            
        
    @staticmethod
    def get_device(bestGPU=None):
        device = torch.device("cpu")
        if torch.cuda.is_available():
            if bestGPU is None:
                stats = gpustat.GPUStatCollection.new_query()
                ids = map(lambda gpu: int(gpu.entry["index"]), stats)
                ratios = map(lambda gpu: float(gpu.entry["memory.used"]) / float(gpu.entry["memory.total"]), stats)
                pairs = list(zip(ids, ratios))
                random.shuffle(pairs)
                bestGPU = min(pairs, key=lambda x: x[1])[0]
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            os.environ['CUDA_VISIBLE_DEVICES'] = str(bestGPU)
            device = torch.device(f"cuda:{bestGPU}")
        return device

    @staticmethod
    def atoi(s):
        s = s[::-1]
        num = 0
        for i, v in enumerate(s):
            t = '%s * 1' % v
            n = eval(t)
            num += n * (10 ** i)
        return num
    
    @staticmethod
    def save(model, arch, task, tag, pre_path=None):
        if pre_path is None:
            pre_path = osp.join(ROOT, "models/ckpt")
        weights = model.cpu().state_dict()

        path = osp.join(pre_path, f"{task}_{arch}_{tag}.pt")
        print(f"-> save check point: {path}")
        torch.save(weights, path)

    @staticmethod
    def load(arch, task, tag, pre_path=None):
        if pre_path is None:
            pre_path = osp.join(ROOT, "models/ckpt")
        path = osp.join(pre_path, f"{task}_{arch}_{tag}.pt")
        if osp.exists(path):
            print(f"-> load check point: {path}")
            return torch.load(path)
        return None
    
    @staticmethod
    def add_subtask_info(tid, stid, value):
        """增加子任务信息"""
        IOtool.__taskinfoclock.acquire()
        taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
        taskinfo[tid]["function"].update({stid:value})
        IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
        IOtool.__taskinfoclock.release()
    
    @staticmethod
    def change_task_info(tid, key, value):
        """修改主任务信息"""
        IOtool.__taskinfoclock.acquire()
        taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
        taskinfo[tid][key]=value
        IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
        IOtool.__taskinfoclock.release()  

    @staticmethod
    def get_task_info():
        """获取主任务信息"""
        IOtool.__taskinfoclock.acquire()
        taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
        IOtool.__taskinfoclock.release()
        return taskinfo
    
    @staticmethod
    def add_task_info(tid, info):
        """增加主任务"""
        IOtool.__taskinfoclock.acquire()
        taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
        taskinfo[tid] = info
        IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
        IOtool.__taskinfoclock.release()
    
    @staticmethod
    def del_task_info(tid):
        """删除主任务"""
        IOtool.__taskinfoclock.acquire()
        taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
        del taskinfo[tid]
        IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
        IOtool.__taskinfoclock.release()
    
    @staticmethod
    def reset_task_info(data):
        IOtool.__taskinfoclock.acquire()
        IOtool.write_json(data, osp.join(ROOT,"output","task_info.json"))
        IOtool.__taskinfoclock.release()
        
    @staticmethod
    def change_subtask_state(tid, stid, state):
        """修改子任务状态"""
        print("修改子任务状态1:",tid,stid, IOtool.__curpoollist,IOtool.__poollist, os.getpid())
        IOtool.__taskinfoclock.acquire()
        print("修改子任务状态2:",tid,stid,state)
        taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
        taskinfo[tid]["function"][stid]["state"] = state
        IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
        IOtool.__taskinfoclock.release()
    
    @staticmethod
    def change_task_state(tid, state):
        """修改主任务状态"""
        IOtool.__taskinfoclock.acquire()
        taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
        taskinfo[tid]["state"] = state
        IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
        IOtool.__taskinfoclock.release()

    @staticmethod   
    def change_task_success_v2(tid):
        """判断主任务是否执行成功,不存在全局变量时使用"""
        IOtool.__taskinfoclock.acquire()
        taskinfo = IOtool.load_json(osp.join(ROOT,"output","task_info.json"))
        info = taskinfo[tid]["function"]
        sub_task_list = info.keys()
        state = 2
        sub_state_list = [] # 子任务状态列表
        for stid in sub_task_list:
            sub_state_list.append(info[stid]["state"])
        if 0 in sub_state_list: # 如果子任务状态列表中存在0 未执行状态，则主任务为执行中
            state = 1
        elif 1 in sub_state_list: # 如果子任务状态列表中存在1 执行中状态，则主任务为执行中
            state = 1
        elif 3 in sub_state_list or 4 in sub_state_list: # 如果子任务状态列表中存在3则代表有子任务执行失败，则主任务为执行失败
            state = 3
        else: # 如果子任务状态列表中不存在0，1，3则代表所有子任务执行成功，则主任务为执行成功
            state = 2
        taskinfo[tid]["state"] = state
        IOtool.write_json(taskinfo,osp.join(ROOT,"output","task_info.json"))
        IOtool.__taskinfoclock.release()

    @staticmethod
    def load_json(path):
        """
        :param path:
        :return res: a dictionary of .json file
        """
        res=[]
        with open(path,mode='r',encoding='utf-8') as f:
            dicts = json.load(f)
            res=dicts
        return res

    @staticmethod
    def write_json(data,path):
        """
        :param data: a dictionary
        :param path:
        """
        with open(path,'w',encoding='utf-8') as f:
            json.dump(data,f,indent=4)
    
    
    @staticmethod
    def save_img(img,path):

        img=img*255
        img=np.int8(img)

        if len(img.shape)==2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(path, img)
        elif len(img.shape)==3:
            r, g, b = cv2.split(img)
            img = cv2.merge([b, g, r])
            cv2.imwrite(path, img)
        
    @staticmethod
    def read_log(path):
        res=""
        with open(path,"r") as f:
            for line in f:
                res=res+'\n'+line
        return res

    @staticmethod
    def get_task_id(salt):
        curr_time = str(datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y%m%d_%H%M"))
        m = hashlib.md5()
        m.update(f"argp_{time.time()}_{str(salt)}".encode('utf-8'))
        tid = f"{curr_time}_{m.hexdigest()}"
        return tid[:21]
    
    @staticmethod
    def get_mean_std(loader):
        # var[X] = E[X**2] - E[X]**2
        channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

        for data, _ in tqdm(loader):
            channels_sum += torch.mean(data, dim=[0, 2, 3])
            channels_sqrd_sum += torch.mean(data ** 2, dim=[0, 2, 3])
            num_batches += 1

        mean = channels_sum / num_batches
        std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

        return mean, std
    @staticmethod
    def format_time(seconds):
        days = int(seconds / 3600 / 24)
        seconds = seconds - days * 3600 * 24
        hours = int(seconds / 3600)
        seconds = seconds - hours * 3600
        minutes = int(seconds / 60)
        seconds = seconds - minutes * 60
        secondsf = int(seconds)
        seconds = seconds - secondsf
        millis = int(seconds * 1000)
        f = ''
        i = 1
        if days > 0:
            f += str(days) + 'D'
            i += 1
        if hours > 0 and i <= 2:
            f += str(hours) + 'h'
            i += 1
        if minutes > 0 and i <= 2:
            f += str(minutes) + 'm'
            i += 1
        if secondsf > 0 and i <= 2:
            f += str(secondsf) + 's'
            i += 1
        if millis > 0 and i <= 2:
            f += str(millis) + 'ms'
            i += 1
        if f == '':
            f = '0ms'
        return f


    @staticmethod
    def progress_bar(current, total, msg=None):
        global last_time, begin_time
        if current == 0:
            begin_time = time.time()  # Reset for new bar.
        cur_len = int(TOTAL_BAR_LENGTH * current / total)
        rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

        sys.stdout.write(' [')
        for i in range(cur_len):
            sys.stdout.write('=')
        sys.stdout.write('>')
        for i in range(rest_len):
            sys.stdout.write('.')
        sys.stdout.write(']')

        cur_time = time.time()
        step_time = cur_time - last_time
        last_time = cur_time
        tot_time = cur_time - begin_time

        L = []
        L.append('  Step: %s' % IOtool.format_time(step_time))
        L.append(' | Tot: %s' % IOtool.format_time(tot_time))
        if msg:
            L.append(' | ' + msg)
        msg = ''.join(L)
        sys.stdout.write(msg)
        for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
            sys.stdout.write(' ')
        for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
            sys.stdout.write('\b')
        sys.stdout.write(' %d/%d ' % (current + 1, total))
        if current < total - 1:
            sys.stdout.write('\r')
        else:
            sys.stdout.write('\n')
        sys.stdout.flush()


    @staticmethod
    def summary_dict(model, input_size, batch_size=-1, device=torch.device('cpu'), dtypes=None):
        if dtypes == None:
            dtypes = [torch.FloatTensor]*len(input_size)

        summary_str = ''
        model = model.cpu()

        def register_hook(module):
            def hook(module, input, output):
                class_name = str(module.__class__).split(".")[-1].split("'")[0]
                module_idx = len(summary)

                m_key = "%s-%i" % (class_name, module_idx + 1)
                summary[m_key] = OrderedDict()
                summary[m_key]["input_shape"] = list(input[0].size())
                summary[m_key]["input_shape"][0] = batch_size
                if isinstance(output, (list, tuple)):
                    summary[m_key]["output_shape"] = [
                        [-1] + list(o.size())[1:] for o in output
                    ]
                else:
                    summary[m_key]["output_shape"] = list(output.size())
                    summary[m_key]["output_shape"][0] = batch_size

                params = 0
                if hasattr(module, "weight") and hasattr(module.weight, "size"):
                    params += torch.prod(torch.LongTensor(list(module.weight.size())))
                    summary[m_key]["trainable"] = module.weight.requires_grad
                if hasattr(module, "bias") and hasattr(module.bias, "size"):
                    params += torch.prod(torch.LongTensor(list(module.bias.size())))
                summary[m_key]["nb_params"] = int(params)

            if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
            ):
                hooks.append(module.register_forward_hook(hook))

        # multiple inputs to the network
        if isinstance(input_size, tuple):
            input_size = [input_size]

        # batch_size of 2 for batchnorm
        x = [torch.rand(2, *in_size).type(dtype).to(device=device)
            for in_size, dtype in zip(input_size, dtypes)]

        # create properties
        summary = OrderedDict()
        hooks = []

        # register hook
        model.apply(register_hook)

        # make a forward pass
        # print(x.shape)
        model(*x)

        # remove these hooks
        for h in hooks:
            h.remove()

        total_params = 0
        total_output = 0
        trainable_params = 0

        layers_info = []
        for layer in summary:
            layers_info.append([
                layer,
                summary[layer]["output_shape"],
                int(summary[layer]["nb_params"])
            ])
            total_params += summary[layer]["nb_params"]
            total_output += np.prod(summary[layer]["output_shape"])
            if "trainable" in summary[layer]:
                if summary[layer]["trainable"] == True:
                    trainable_params += summary[layer]["nb_params"]

        # assume 4 bytes/number (float on cuda).
        total_input_size = abs(np.prod(sum(input_size, ()))
                            * batch_size * 4. / (1024 ** 2.))
        total_output_size = abs(2. * total_output * 4. /
                                (1024 ** 2.))  # x2 for gradients
        total_params_size = abs(total_params * 4. / (1024 ** 2.))
        total_size = total_params_size + total_output_size + total_input_size
        summary_info = {
            "total_params": int(total_params),
            "trainable_params": int(trainable_params),
            "total_input_size": total_input_size,
            "total_output_size": total_output_size,
            "total_params_size": float(total_params_size),
            "total_size": float(total_size),
            #"layers_info": layers_info
        }
        return summary_info

# class Logger:
#     level_relations = {
#         'debug':logging.DEBUG,
#         'info':logging.INFO,
#         'warning':logging.WARNING,
#         'error':logging.ERROR,
#         'crit':logging.CRITICAL
#     }
#     def __init__(self,filename,
#                  level='info',
#                  when='D',
#                  backCount=3,
#                  fmt='%(asctime)s [%(levelname)s] %(message)s'
#                  #fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
#                  ):
#         self.logger = logging.getLogger(filename)
#         format_str = logging.Formatter(fmt)#设置日志格式
#         self.logger.setLevel(self.level_relations.get(level))#设置日志级别
#         sh = logging.StreamHandler()#往屏幕上输出
#         sh.setFormatter(format_str) #设置屏幕上显示的格式
#         th = handlers.TimedRotatingFileHandler(filename=filename, when=when, backupCount=backCount, encoding='utf-8')
#         #实例化TimedRotatingFileHandler
#         #interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
#         # S 秒
#         # M 分
#         # H 小时、
#         # D 天、
#         # W 每星期（interval==0时代表星期一）
#         # midnight 每天凌晨
#         th.setFormatter(format_str)#设置文件里写入的格式
#         self.logger.addHandler(sh) #把对象加到logger里
#         self.logger.addHandler(th)

#     def info(self, msg):
#         return self.logger.info(msg)

class Callback:
    @staticmethod
    def callback_train(model, epoch_result, results=None, step=4,logging=None):
        """
        callback example for train function
        :param model:
        :param train_result:
        :param kwargs:
        :return:
        """
        

        if results is not None:
            # logging.basicConfig(filename=osp.join(results.get_res_value("out_path"), results.get_res_value("AAtid")+"_log.txt"),
            #                     filemode="a", level=logging.INFO,
            #                     format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
            #                     )
            results.set_res_value(key="trainer",value=epoch_result)
            print(f"-> save result for epoch:{epoch_result['epoch']}...")
            logging.info("[模型训练阶段] 正在进行第{:d}轮训练, 训练准确率：{:.3f}%，测试准确率：{:.3f}%".format(epoch_result["epoch"],
                                                                                       epoch_result["test"][0],
                                                                                       epoch_result["train"][0]))
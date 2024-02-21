
import pickle
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
import autokeras as ak

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys 
sys.path.append('./utils')
from operation_test_utils import modify_model
import argparse
from tensorflow.keras.datasets import mnist,cifar10
import time
import gc
import numpy as np
import copy
import uuid
import csv
import string
import kerastuner
import tensorflow as tf
from kerastuner.engine import hypermodel as hm_module
import pandas as pd

import matplotlib.pyplot as plt  
import autokeras as ak

def get_arch(hyperparameters):
    for key in hyperparameters.values.keys():
        if 'block_type' in key:
            arch=hyperparameters.values[key]
            return arch
        else:
            for key in hyperparameters.values.keys():
                if 'res_net_block' in key:
                    return 'resnet'
                if 'vgg_block' in key:
                    return 'vgg'
                if 'xception_block' in key:
                    return 'xception'
                if 'conv_block' in key:
                    return 'vanilla'
                if 'efficient' in key:
                    return 'efficient'
    return None

def ol_judge(history,threshold,rate):
    acc=history['accuracy']
    maximum=[]
    minimum=[]
    count=0
    for i in range(len(acc)):
        if i==0 or i ==len(acc)-1:
            continue
        if acc[i]-acc[i-1]>=0 and acc[i]-acc[i+1]>=0:
            maximum.append(acc[i])
        if acc[i]-acc[i-1]<0 and acc[i]-acc[i+1]<0:
            minimum.append(acc[i])
    for i in range(min(len(maximum),len(minimum))):
        if maximum[i]-minimum[i]>=threshold:
            count+=1
    if count>=rate*len(acc):
        return True
    else:
        return False

def has_NaN(output):
    output=np.array(output)
    try:
        result=(np.isnan(output).any() or np.isinf(output).any())
    except:
        result=None
    return result

def max_delta_acc(acc_list):
    if len(acc_list)<=3:
        return 10
    max_delta=0
    for i in range(len(acc_list)-1):
        if acc_list[i+1]-acc_list[i]>max_delta:
            max_delta=acc_list[i+1]-acc_list[i]
    return max_delta

def get_loss(history,unstable_threshold=0.03,unstable_rate=0.2,sc_threshold=0.01):

    train_loss=history['loss']
    train_acc=history['accuracy']
    test_loss=history['val_loss']
    test_acc=history['val_accuracy']
    count=0

    if train_loss!=[]:
        if has_NaN(test_loss) or has_NaN(train_loss) or test_loss[-1]>=1e+5:
            return 'slow_converge'

        if ol_judge(history,unstable_threshold,unstable_rate):  
            return 'oscillating'
        elif max_delta_acc(test_acc)<sc_threshold and max_delta_acc(train_acc)<sc_threshold:
            return 'slow_converge'
        else:
            return 'normal'

def get_modification(input_dict):
    dict_length=len(input_dict.keys())
    output_list=[]
    for i in range(dict_length-1):
        diff_list=[]
        pre_list=input_dict[str(i)]
        next_list=input_dict[str(i+1)]
        for j in range(len(next_list)):
            diff_list.append(next_list[j]-pre_list[j])
        output_list.append(diff_list)
    return output_list

def gradient_norm(gradient_list):
    # assert len(gradient_list)%2==0
    norm_kernel_list=[]
    norm_bias_list=[]
    for i in range(int(len(gradient_list)/2)):
        norm_kernel_list.append(np.linalg.norm(np.array(gradient_list[2*i])))
    return norm_kernel_list#,norm_bias_list

def gradient_zero_radio(gradient_list):
    kernel=[]
    bias=[]
    total_zero=0
    total_size=0
    for i in range(len(gradient_list)):    
        zeros=np.sum(gradient_list[i]==0)
        total_zero+=zeros
        total_size+=gradient_list[i].size
    total=float(total_zero)/float(total_size)
    return total

def gradient_message_summary(gradient_list):
    total_ratio= gradient_zero_radio(gradient_list)


    norm_kernel= gradient_norm(gradient_list)#, norm_bias 
    if norm_kernel[-1]==0:
        gra_rate=1
    else:
        gra_rate = (norm_kernel[0] / norm_kernel[-1])
    return [norm_kernel,gra_rate], [total_ratio]#, norm_bias

def gradient_issue(gradient_list,threshold_low=1e-3,threshold_low_1=1e-4,threshold_high=70,threshold_die_1=0.7):

    [norm_kernel,gra_rate],[total_ratio]=gradient_message_summary(gradient_list)#avg_bias,
    #[total_ratio,kernel_ratio,bias_ratio,max_zero]\
                    
    for i in range(len(gradient_list)):
        if has_NaN(gradient_list[i]):
            return 'explode'

    if total_ratio>=threshold_die_1:# or max_zero>=threshold_die_2
        return 'dying'
    elif gra_rate<threshold_low and norm_kernel[0]<threshold_low_1:
        return 'vanish'
    elif gra_rate>threshold_high:
        return 'explode'

    # else:
    #     feature_dic['died_relu']=0
    return 'normal'

def get_gradient(gw):
    weight_dict=gw['weight']
    gradient_dict=gw['gradient']

    wgt='normal'
    grad='normal'

    for epoch in weight_dict.keys():
        for i in range(len(weight_dict[epoch])):
            if has_NaN(weight_dict[epoch][i]):
                wgt='nan_weight'
                break
        if wgt=='nan_weight':
            break
    for epoch in gradient_dict.keys():
        grad_result=gradient_issue(gradient_dict[epoch])
        if grad_result!='normal':
            grad=grad_result
            break
    return grad,wgt
    
    

    # weight_modi=get_modification(weight_dict)
    # gradient_modi=get_modification(gradient_dict)
    return grad,wgt
    

def judge_dirs(target_dir):
    params_path=os.path.join(target_dir,'param.pkl')
    gw_path=os.path.join(target_dir,'gradient_weight.pkl')
    his_path=os.path.join(target_dir,'history.pkl')

    with open(params_path, 'rb') as f:#input,bug type,params
        hyperparameters = pickle.load(f)
    with open(his_path, 'rb') as f:#input,bug type,params
        history = pickle.load(f)
    with open(gw_path, 'rb') as f:#input,bug type,params
        gw = pickle.load(f)

    arch=get_arch(hyperparameters)
    loss=get_loss(history)
    grad,wgt=get_gradient(gw)
    

    return arch,loss,grad,wgt

# TODO: back
def load_evaluation(algw,evaluation_pkl='./utils/priority_all.pkl'):
    with open(evaluation_pkl, 'rb') as f:#input,bug type,params
        evaluation = pickle.load(f)
    if algw not in evaluation.keys():
        # return None,None
        algw=algw.split('-')[0]+"-normal-normal-normal"
    if 'vgg' in algw:
        algw=algw.replace('vgg','resnet')
    result_dict=evaluation[algw]
    opt_list=list(result_dict.keys())
    for opt in opt_list:
        if result_dict[opt]=='/':
            del result_dict[opt]
    sorted_result = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)
    operation_list=[r[0] for r in sorted_result]
    
    # with open(os.path.abspath('./utils/priority_pure.pkl'), 'rb') as f:#input,bug type,params
    #     tmp = pickle.load(f)
    # operation_list1=tmp[algw]
    return result_dict,operation_list # key: operation+value; value: weight

# def load_evaluation(algw,evaluation_pkl='./utils/priority_all.pkl',save_dir=None):
#     with open(evaluation_pkl, 'rb') as f:#input,bug type,params
#         evaluation = pickle.load(f)
#     if algw not in evaluation.keys():
#         # return None,None
#         algw=algw.split('-')[0]+"-normal-normal-normal"
#     result_dict=evaluation[algw]
#     opt_list=list(result_dict.keys())
#     for opt in opt_list:
#         if result_dict[opt]=='/':
#             del result_dict[opt]
#     sorted_result = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)
#     operation_list=[r[0] for r in sorted_result]

#     if save_dir!=None:
#         with open(os.path.join(save_dir,'tmp_priority.pkl'), 'rb') as f:#input,bug type,params
#             tmp = pickle.load(f)
#         operation_list=tmp[algw]
#     return result_dict,operation_list# operation_list # key: operation+value; value: weight

def read_history_score(trial_dir_list,read_trials=15):
    his_score_list=[]
    for cur_trial in range(read_trials):
        for trial_dir in trial_dir_list:
            if os.path.basename(trial_dir).startswith('{}-'.format(cur_trial)):
                his_pkl=os.path.join(trial_dir,'history.pkl')
                with open(his_pkl, 'rb') as f:#input,bug type,params
                    history = pickle.load(f)
                his_score_list.append(max(history['val_accuracy']))
                break
    return his_score_list

def read_history_whole(trial_dir_list,read_trials=15):
    his_score_list=[]
    log_dict={}
    log_dict['best_score']=0
    log_dict['best_trial']=0
    log_dict['best_time']=None
    for cur_trial in range(read_trials):
        for trial_dir in trial_dir_list:
            if os.path.basename(trial_dir).startswith('{}-'.format(cur_trial)):
                his_pkl=os.path.join(trial_dir,'history.pkl')
                with open(his_pkl, 'rb') as f:#input,bug type,params
                    history = pickle.load(f)
                his_score_list.append(max(history['val_accuracy']))
                if max(history['val_accuracy'])>log_dict['best_score']:
                    log_dict['best_score']=max(history['val_accuracy'])
                    # log_dict['best_trial']=cur_trial
                    # log_dict['best_time']=None
                log_dict[cur_trial]={}
                log_dict[cur_trial]['history']=history
                log_dict[cur_trial]['time']=None
                log_dict[cur_trial]['score']=max(history['val_accuracy'])
                break
    
    # log_dict['best_trial']
    return log_dict

def to_percent(temp, position):
    return '%1.0f'%(100*temp) + '%'

def plot_line_chart(y_list,label_list,title='Test',x_label='x',y_label='y',save_path='./line_chart.pdf'):
    plt.figure(figsize=(10, 5.5))
    
    x=np.arange(1,len(y_list[0])+1)
    for y in range(len(y_list)):
        l1=plt.plot(x,y_list[y],label=label_list[y])
     
    plt.title(title,fontsize=20)
    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label, fontsize=20)
    # x_stick=np.arange(1,16,2)
    # plt.xticks(x_stick,fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim(0,0.6)
    # plt.xlim(1,15)
    plt.legend(fontsize=16)#,loc=2
    # use % instead of float
    import matplotlib.ticker as ticker 
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(to_percent))

    plt.savefig(save_path,dpi=300)

def traversalDir_FirstDir(path):
    tmplist = []
    if (os.path.exists(path)):
        files = os.listdir(path)
        for file1 in files:
            m = os.path.join(path,file1)
            if (os.path.isdir(m)):
                tmplist.append(m)
                # tmplist1.append(file1)
    return tmplist

def check_move(save_dir):
    if os.path.exists(save_dir):
        dir_list=traversalDir_FirstDir(save_dir)
        if dir_list==[]:
            return None
        num=0
        new_save_dir=None
        for d in dir_list:
            try:
                tmp_num=int(os.path.basename(d).split('-')[0])
                if tmp_num>=num:
                    new_save_dir=d
                    num=tmp_num
            except:
                pass
    return new_save_dir

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    return False

def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        pass
 
    return False

def get_true_value(value):
    if value=='True':
        return True
    elif value=='False':
        return False
    elif not is_number(value):
        return value
    elif is_int(value):
        return int(value)
    else:
        return float(value)

def special_action(action):
    if action in ['activation','initial']:
        return True
    return False

def write_opt(action,value,write_path='./Test_dir/tmp/tmp_action_value.pkl'):
    opt_dict={}
    opt_dict['action']=action
    opt_dict['value']=value
    with open(os.path.abspath(write_path), 'wb') as f:
        pickle.dump(opt_dict, f)

def read_opt(model,read_path='./Test_dir/tmp/tmp_action_value.pkl'):
    read_path=os.path.abspath(read_path)
    if os.path.exists(read_path):
        with open(read_path, 'rb') as f:#input,bug type,params
            opt_dict = pickle.load(f)
        
        opt=model.optimizer
        for key in model.loss.keys():
            loss=model.loss[key].name
            break

        
        if opt_dict['action']=='activation':
            model=modify_model(model,acti=opt_dict['value'],init=None,method='acti')
        elif opt_dict['action']=='initial':
            model=modify_model(model,acti=None,init=opt_dict['value'],method='init')
        elif isinstance(opt_dict['action'],dict):
            for i in opt_dict['action']:
                if 'activation' in i:
                    model=modify_model(model,acti=opt_dict['value'][i],init=None,method='acti')
                if 'initial' in i:
                    model=modify_model(model,acti=None,init=opt_dict['value'][i],method='init')
        else:
            print('Type Error')
            os._exit(0)
        model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
    return model

def sort_opt_wgt_dict(opt_wgt_dict,opt_list,threshold=0):
    # only output the opts whose weight is over threshold
    new_opt_list=[]
    used_action=[]
    for opt in opt_list:
        if opt_wgt_dict[opt]<0:
            return new_opt_list
        action=opt.split('-')[0]
        if action in used_action:
            continue
        new_opt_list.append(opt)
        used_action.append(action)
        
    return new_opt_list

def update_candidate(opt_wgt_dict,opt_list,new_save_dir,candidate_dict_path,beam_size):
    # TODO:add number for the used values, modify tuner too
    # candidate dict is list type, each element is a tuple contains model path,
    # action name and expect score.
    model_path=os.path.abspath(os.path.join(new_save_dir,'model.h5'))
    history_path=os.path.join(new_save_dir,'history.pkl')
    if os.path.exists(candidate_dict_path):
        with open(candidate_dict_path, 'rb') as f:#input,bug type,params
            candidate_dict = pickle.load(f)
    else:
        candidate_dict=[]

    with open(history_path, 'rb') as f:#input,bug type,params
        history = pickle.load(f)
        score=max(history['val_accuracy'])
    
    candidate_actions=sorted(opt_wgt_dict.items(),key=lambda item:item[1])[-beam_size:]
    candidate_actions.reverse()
    #TODO: insert the good accuracy first
    for cond in candidate_actions:
        candidate_dict.append((model_path,cond[0],opt_list,score+cond[1]))
    
    with open(candidate_dict_path, 'wb') as f:
        pickle.dump(candidate_dict, f)

def select_action(candidate_dict_path,beam_size=3):
    if not os.path.exists(candidate_dict_path):
        return None
    with open(candidate_dict_path, 'rb') as f:#input,bug type,params
        candidate_dict = pickle.load(f)
    action_list=sorted(candidate_dict, key=lambda x:x[-1])
    action_list.reverse()
    action_list=action_list[-beam_size:]
    while len(action_list)<beam_size:
        action_list.append(None)# None will be turn to random model in generation
    with open(candidate_dict_path, 'wb') as f:
        pickle.dump(action_list, f)
    return action_list

def write_algw(root_dir):
    import subprocess
    command="/home/Wenjie/anaconda3/envs/autotrain/bin/python ./utils/get_write_algw.py -d {}" #TODO:need to set your your python interpreter path

    out_path=os.path.join(root_dir,'algw_out')
    out_file = open(out_path, 'w')
    out_file.write('logs\n')
    run_cmd=command.format(root_dir)
    subprocess.Popen(run_cmd, shell=True, stdout=out_file, stderr=out_file)

def verify_hp(new_hp,origin_hp):
    for key in origin_hp.keys():
        if key not in new_hp.keys():
            continue
        if new_hp[key]!=origin_hp[key]:
            return False
    return True

def choose_random_select(history_dir_list):
    score_list=[]
    for hdir in history_dir_list:
        score_list.append(float(os.path.basename(hdir).split('-')[1]))
    max_score=max(score_list)
    if max_score<0.5:
        return 0.01
    elif max_score<=0.75:
        return 0.1
    elif max_score>0.75:
        return 0.2


def long_time_task(name,path):
    from tensorflow.keras.models import load_model
    import autokeras
    import tensorflow as tf
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(name)
    print('Run task %s (%s)...' % (name, os.getpid()))
    start = time.time()
    # print(path)
    model=load_model(path)
    # print(tf.config.experimental.list_physical_devices('GPU'))
    end = time.time()
    time.sleep(random.random() * 30)
    print('Task %s runs %0.2f seconds.' % (name, (end - start)))

def evaluate_layer_trainable(model):
    for l in model.layers:
        try:
            sub_layer_list=l.layers
            for sl in sub_layer_list:
                print('SubLayer "{}" :{}'.format(sl.name,sl.trainable))
        except Exception as e:
            print('Layer "{}" :{}'.format(l.name,l.trainable))

def modify_hp_value(ak_params,hp_list_path,num_path):
    if not os.path.exists(num_path):
        num=0
        with open(num_path, 'wb') as f:
            pickle.dump(num, f)

    with open(num_path, 'rb') as f:#input,bug type,params
        num = pickle.load(f)
        
    with open(hp_list_path, 'rb') as f:#input,bug type,params
        hp_list = pickle.load(f)
    ak_params.values=hp_list[num]
        
    num+=1
    with open(num_path, 'wb') as f:
        pickle.dump(num, f)
    return ak_params


def multi_action_search(values,best_hp_name,best_hp_value):
    if best_hp_name in values.keys() and (values[best_hp_name]=='xception' or values[best_hp_name]=='resnet' or values[best_hp_name]=='efficient'):
        with open(os.path.abspath('./utils/hp_relation.pkl'), 'rb') as f:#input,bug type,params
            hp_relation = pickle.load(f)
        for hpr in hp_relation.keys():
            if len(hp_relation[hpr]['parents'])!=1:
                continue
            if best_hp_name+"--"+best_hp_value in hp_relation[hpr]['parents']:
                if '/pretrained' in hpr:
                    values[hpr]=True
                    trainable_setting=hpr.replace('/pretrained','/trainable')
                    values[trainable_setting]=True
            if values[best_hp_name]=='efficient':
                if 'dropout' in hpr:
                    values[hpr]=0.5 # Dropout
                if best_hp_name+"--"+best_hp_value in hp_relation[hpr]['parents'] and '/imagenet_size' in hpr:
                    values[hpr]=True # Imagenet size
                if best_hp_name+"--"+best_hp_value in hp_relation[hpr]['parents']:
                    if '/imagenet_size' in hpr:
                        values[hpr]=True # Imagenet size
                    if '/version' in hpr:
                        values[hpr]='b7' # version
        values['learning_rate']=0.0001
    if best_hp_name == 'multi_step':
        for vkey in values.keys():
            if 'trainable' in vkey:
                values[vkey]=True
                values[best_hp_name]=best_hp_value
    return values

def append_list(best_hps_hash,best_hash_dict,opt,best_hash_path):
    if best_hps_hash not in best_hash_dict.keys():
        best_hash_dict[best_hps_hash]=[]

    best_hash_dict[best_hps_hash].append(opt)
    with open(best_hash_path, 'wb') as f:
        pickle.dump(best_hash_dict, f)

def get_opti_value(log_dict):
    import random
    if log_dict['cur_trial']==6:
        opti=True
        for key in log_dict.keys():
            try:
                if float(key.split('-')[1])>0.81:
                    opti=False
            except:
                print(key)
        if opti:
            optimal_list=['./utils/tmp3.pkl','./utils/tmp2.pkl','./utils/tmp1.pkl','./utils/tmp0.pkl']
            tmp_path=os.path.abspath(random.choice(optimal_list))
            if not os.path.exists(tmp_path):
                print(tmp_path)
                return False,None
            with open(tmp_path, 'rb') as f:#input,bug type,params
                tmp = pickle.load(f)
            print('===============Use optimal Structure!!===============\n')
            return True,tmp
    return False,None
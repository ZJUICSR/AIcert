import os
import shutil
import time
import copy
import numpy as np
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist,cifar10
from tensorflow.keras.models import load_model,Model
from tensorflow.keras import layers 
import tensorflow.keras as keras
import argparse
import autokeras as ak
import pickle
import tensorflow.keras as keras
import sys
import json
sys.path.append("..")



def save_data(x_train,x_test,y_train,y_test,save_path):
    dataset={}
    if isinstance(x_test,np.ndarray):
        dataset['x_train']=x_train
        dataset['x_test']=x_test
    else:
        dataset['x_train']=x_train.numpy() # convert from tensor
        dataset['x_test']=x_test.numpy()
    dataset['y_train']=y_train
    dataset['y_test']=y_test
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)

def normalize_save_data(image,normalize_model,new_data_save_path):
    save_dir=os.path.dirname(new_data_save_path)
    if normalize_model==None:
        write_path=os.path.join(save_dir,'normalize')
        if not os.path.exists(write_path):
            f = open(write_path, 'w')
            f.write('empty normalization !')
            f.close()
        return image
    image=normalize_model(image)
    
    # initialize the normalize model after process the data
    normalize_path=os.path.join(save_dir,'normalize.h5')
    if not os.path.exists(normalize_path):
        normalize_model.save(normalize_path)
    return image

def resize_image(image,normalize_model,new_data_save_path,input_shape):
    # only consider 3 dimension data
    if len(input_shape)==4:
        IMG_SIZE=[input_shape[1],input_shape[2]]
        concatenate_length=int(input_shape[3]/image.shape[3])
    else:
        print('error, check input data shape')
        os._exit(0)
        
    image_result=None
    for i in range(int(image.shape[0]/1000)):
        normalize_image=normalize_save_data(image[i*1000:(i+1)*1000,...],normalize_model,new_data_save_path)
        new_image=tf.image.resize(normalize_image, IMG_SIZE)
        if image_result==None:
            image_result=new_image
        else:
            image_result=tf.concat([image_result,new_image],axis=0)
    if concatenate_length>1:
        concatenate_list=[]
        for i in range(concatenate_length):
            concatenate_list.append(image_result)
        result=tf.keras.layers.concatenate(concatenate_list,axis=-1)
    else:
        result=image_result
    return result

def mnist_load_data(normalize_model=None,new_model_input=None,new_data_save_path=None):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x_train = x_train.reshape(60000, 784)
    # x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    if new_data_save_path!=None:
        
        if len(new_model_input)==4:
            x_train = x_train.reshape(-1, 28,28,1)
            x_test = x_test.reshape(-1, 28,28,1)
            new_x_test=resize_image(x_test,normalize_model,new_data_save_path,input_shape=new_model_input)
            new_x_train=resize_image(x_train,normalize_model,new_data_save_path,input_shape=new_model_input)
        else:
            new_x_train=x_train
            new_x_test=x_test
        save_data(new_x_train,new_x_test,y_train,y_test,new_data_save_path)
    
    return (x_train, y_train), (x_test, y_test)


def cifar10_load_data(normalize_model=None,new_model_input=None,new_data_save_path=None):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = y_train.ravel()
    y_test = y_test.ravel()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]
        x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    if new_data_save_path!=None:      
        new_x_test=resize_image(x_test,normalize_model,new_data_save_path,input_shape=new_model_input)
        new_x_train=resize_image(x_train,normalize_model,new_data_save_path,input_shape=new_model_input)
        save_data(new_x_train,new_x_test,y_train,y_test,new_data_save_path)
    
    return (x_train, y_train), (x_test, y_test)


def model_generate(
    block_type='resnet',
    search=True,
    data='mnist',
    save_dir='./result',
    epoch=2,
    tuner='greedy',
    trial=1,
    gpu='0',
    init='normal',
    iter_num=4,
    param_path='./param.pkl',
):

        
    root_path=save_dir
    tmp_dir=os.path.join(os.path.dirname(root_path),'tmp')
    if os.path.exists(root_path):
        shutil.rmtree(root_path)
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(root_path)
    os.makedirs(tmp_dir)
    log_path=os.path.join(root_path,'log.pkl')

    if data=='mnist':
        (x_train, y_train), (x_test, y_test) = mnist_load_data()
    elif data=='cifar':
        (x_train, y_train), (x_test, y_test) = cifar10_load_data()
    else:
        (x_train, y_train), (x_test, y_test)=data #TODO: 如果是处理过的数据，需要给出数据路径或者读取方法


    # DEMO:1
    if search:
        
        # initialize the search log
        if not os.path.exists(log_path):
            log_dict={}
            log_dict['cur_trial']=-1
            log_dict['start_time']=time.time()
            log_dict['data']=data
            log_dict['tmp_dir']=tmp_dir

            with open(log_path, 'wb') as f:
                pickle.dump(log_dict, f)

        else:
            with open(log_path, 'rb') as f:
                log_dict = pickle.load(f)
            for key in log_dict.keys():
                if key.startswith('{}-'.format(log_dict['cur_trial'])):
                    log_dict['start_time']=time.time()-log_dict[key]['time']
                    break
            with open(log_path, 'wb') as f:
                pickle.dump(log_dict, f)

        #Dream and other based on AK
        if tuner != 'deepalchemy':
            input_node = ak.ImageInput()
            output_node = ak.ImageBlock(
                # Only search ResNet architectures.
                normalize=True,
                block_type=block_type,
            )(input_node)
            output_node = ak.ClassificationHead()(output_node)
            clf = ak.AutoModel(
                inputs=input_node,
                outputs=output_node,
                overwrite=True,
                max_trials=trial,
                directory=os.path.join(root_path,'image_classifier'),
                tuner=tuner,

            )
            clf.fit(x_train, y_train, epochs=epoch,root_path=root_path)

            model_path=os.path.join(root_path,'best_model.h5')
            if not os.path.exists(model_path):
                model = clf.export_model()
                model.save(model_path)
        else:
            #yzx+ deepalchemy
            from Deepalchemy import deepalchemy as da
            np.save(x_train, '../xtr.npy')
            np.save(y_train, '../ytr.npy')
            np.save(x_test, '../xte.npy')
            np.save(y_test, '../yte.npy')
            trainfunc, nmax = da.gen_train_function(False, gpu, block_type, epoch, [x_train, y_train, x_test, y_test])
            wmin, wmax, dmin, dmax = da.NM_search_min(block_type, trainfunc, nmax, init, iter_num)

            trainfunc, nmax = da.gen_train_function(True, gpu, block_type, epoch, [x_train, y_train, x_test, y_test])
            model_path = os.path.join(root_path, 'best_model.h5')
            valloss = trainfunc(dmin, dmax, wmin, wmax)
            shutil.copyfile("./best.h5",model_path)
            shutil.copyfile("../best_param.pkl", os.path.join(root_path, 'best_param.pkl'))

    else:
        # DEMO 2
        with open('./hypermodel.pkl', 'rb') as f:
            hm = pickle.load(f)
        with open('./hyperparam.pkl', 'rb') as f: #you need to input the parameter of the model here
            model_hyperparameter = pickle.load(f)
        with open(param_path, 'rb') as f: #you need to input the parameter of the model here
            param = pickle.load(f)
        model_hyperparameter.values=param

        model=hm.build(model_hyperparameter) #the model will be build by autokeras with this parameter 
        print(1)
        
        model.save(os.path.join(root_path,'best_model.h5'))
        # model.save(os.path.join(root_path,'best_model'))


    print('finish')

def traversalDir_FirstDir(path):
    tmplist = []
    if (os.path.exists(path)):
        files = os.listdir(path)
        for file1 in files:
            m = os.path.join(path,file1)
            if (os.path.isdir(m) and 'image_classifier' not in m):
                tmplist.append(m)
    return tmplist

def sort_dir_name(dir_list):
    dir_name_list=[os.path.basename(d) for d in dir_list]
    sorted_list=[]
    count=0
    stop=False
    while not stop:
        stop=True
        for dir_name in dir_name_list:
            if dir_name.startswith(str(count)+'-0.'):
                sorted_list.append(dir_name)
                stop=False
                count+=1
                break
        # if no dir name reach requirement, return
    return sorted_list

def update_best_history(best_history,trial_history):
    if best_history['loss']==[]:
        best_history['loss'].append(min(trial_history['loss']))
        best_history['val_loss'].append(min(trial_history['val_loss']))
        best_history['accuracy'].append(max(trial_history['accuracy']))
        best_history['val_accuracy'].append(max(trial_history['val_accuracy']))
    else:
        best_history['loss'].append(min(min(trial_history['loss']),best_history['loss'][-1]))
        best_history['val_loss'].append(min(min(trial_history['val_loss']),best_history['val_loss'][-1]))
        best_history['accuracy'].append(max(max(trial_history['accuracy']),best_history['accuracy'][-1]))
        best_history['val_accuracy'].append(max(max(trial_history['val_accuracy']),best_history['val_accuracy'][-1]))
    return best_history

def summarize_result(json_path,save_dir):
    dir_list=traversalDir_FirstDir(save_dir)
    dir_name_list=sort_dir_name(dir_list)#[os.path.basename(d) for d in dir_list]
    # log_path=os.path.join(save_dir,'log.pkl')
    result={}
    result['best_model']=os.path.abspath(os.path.join(save_dir,'best_model.h5'))
    result['best_param']=os.path.abspath(os.path.join(save_dir,'best_param.pkl'))
    result['trial_history']={}
    best_history={}
    best_history['loss']=[]
    best_history['accuracy']=[]
    best_history['val_loss']=[]
    best_history['val_accuracy']=[]
    
    # with open(log_path, 'rb') as f:
    #     log_dict = pickle.load(f)
    # key_list=list(log_dict.keys())
    if os.path.exists('../history.pkl'):
        with open('../history.pkl', 'rb') as f:
            best_history = pickle.load(f)
    else:
        for i in range(len(dir_name_list)):
            dir_name=dir_name_list[i]
            tmp_dir=os.path.join(save_dir,dir_name)
            with open(os.path.join(tmp_dir,'history.pkl'), 'rb') as f:
                history = pickle.load(f)
            result['trial_history'][dir_name.split('-')[0]]=history
            best_history=update_best_history(best_history,history)
    result['best_history']=best_history
    
    with open(json_path, 'w') as fw:
        fw.write(json.dumps(result,ensure_ascii=False,indent=4,separators=(',',':')))
    print(f'========Finish! Json result is saved in {json_path}========') 
            
    

def extract_main_layers(origin_model_path,save_dir):
    save_path=os.path.join(save_dir,'extract_model.h5')
    normalize_path=os.path.join(save_dir,'normalize.h5')
    non_normalize_path=os.path.join(save_dir,'normalize')
    if os.path.exists(save_path) and (os.path.exists(normalize_path) or os.path.exists(non_normalize_path)):
        model=load_model(save_path)
        normalize_model=None
        if os.path.exists(normalize_path) and not os.path.exists(non_normalize_path):
            normalize_model=load_model(normalize_path)
        return model,normalize_model
    
    model=load_model(origin_model_path,custom_objects=ak.CUSTOM_OBJECTS)
    pass_sign=True
    tmp_weight_list=[]
    
    normalize_model=tf.keras.Sequential()
    
    for i in range(len(model.layers)):
        if 'normaliz' in model.layers[i].name:
            normalize_model.add(model.layers[i])
        if pass_sign and not hasattr(model.layers[i],'layers'):
            continue
        if not pass_sign:
            new_config=copy.deepcopy(model.layers[i].get_config())
            outputs=model.layers[i].__class__(**new_config)(outputs)
            tmp_weight=model.layers[i].get_weights()
            if tmp_weight!=[]:
                tmp_weight_list.append((layer_length,tmp_weight))
            layer_length+=1
            continue
        pass_sign=False
        new_model=model.layers[i]
        new_model.layers.pop(0)
        inputs=new_model.inputs[0]
        outputs=new_model.outputs[0]
        layer_length=len(new_model.layers)
        print(1)
    try:
        new_model_1 = Model(inputs, outputs)
    except:
        new_model_1 = model # for vanilla model

    for tmpw in tmp_weight_list:
        new_model_1.layers[tmpw[0]].set_weights(tmpw[1])
        
    new_model_1.compile(optimizer=model.optimizer,loss=model.loss,metrics=['accuracy'])
    new_model_1.save(save_path)
    if normalize_model.layers==[]:
        normalize_model=None
    
    return new_model_1,normalize_model


def onnx_convert(model_path,save_dir):
    import keras2onnx
    from tensorflow.keras.models import load_model
    import autokeras as ak
    import onnx

    # try:
    #     model=load_model(model_path)
    # except:
    if isinstance(model_path,str):
        model=load_model(model_path,custom_objects=ak.CUSTOM_OBJECTS)
    else:
        model=model_path # the input variable model_path can be the model itself
    onnx_model = keras2onnx.convert_keras(model,'autokeras')
    onnx_path=os.path.join(save_dir,'best_model.onnx')
    onnx.save_model(onnx_model, onnx_path)
    # os._exit(0)# TODO: remove
    return onnx_path

def torch_convert(model_path,save_dir,dataset):
    print('========Extracting Main Model...===========')
    new_model,normalize_model=extract_main_layers(model_path,save_dir)
    data_save_path=os.path.join(save_dir,'normalized_data.pkl')
    print('========Saving Dataset...===========')
    try:
        if dataset=='mnist':
            _,_=mnist_load_data(normalize_model,new_model.input_shape,data_save_path)
        elif dataset=='cifar10':
            _,_=cifar10_load_data(normalize_model,new_model.input_shape,data_save_path)
        else:
            print('not support dataset')
    except:
        print('not normalize data and model now!!!')
    
    print('========Converting ONNX Model...===========')
    onnx_path=onnx_convert(new_model,save_dir)
    torch_path=os.path.join(save_dir,'best_model.pth')
    # from onnx_pytorch import code_gen
    # torch_model_dir=os.path.join(save_dir,'torch_model')
    # os.makedirs(torch_model_dir)
    # code_gen.gen(onnx_path, torch_model_dir)
    print('========Converting Pytorch Model...===========')
    import torch
    from onnx2torch import convert
    torch_model_1 = convert(onnx_path)
    torch.save(torch_model_1, torch_path)
    print('=============Finished Converting===========')
    return torch_path

    
def paddle_convert(model_path,save_dir):
    print('========Converting ONNX Model...===========')
    onnx_path=onnx_convert(model_path,save_dir)
    paddle_model_dir=os.path.join(save_dir,'paddle_model')
    params_command='source activate ak2.3; x2paddle --framework=onnx --model={} --save_dir={}'
    print('==========Converting PaddlePaddle Model...============')
    import subprocess
    out_path=os.path.join(save_dir,'out')
    out_file = open(out_path, 'w')
    out_file.write('logs\n')
    run_cmd=params_command.format(onnx_path,paddle_model_dir)
    p=subprocess.Popen(run_cmd, shell=True, stdout=out_file, stderr=out_file, executable='/bin/bash')
    # try:
    #     os.system(os_command)
    # except:
    #     os._exit(0)
    print('=============Finished Converting===========')
    paddle_model_path=os.path.join(paddle_model_dir,'inference_model/model.pdmodel')
    return paddle_model_path
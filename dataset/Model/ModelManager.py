from Utils.IOtool import IOtool
import numpy as np
import os.path as osp
import os
import shutil
CURR = osp.dirname(osp.abspath(__file__))
from PIL import Image
import zipfile
import cv2
from io import BytesIO
import pickle
import json
# import keras
import autokeras as ak
# from keras.models import Sequential, load_model
from tensorflow.keras.models import Sequential, load_model

class ModelDB():
    def __init__(self):
        self.modelInfo=IOtool.load_json(osp.join(CURR,'modelInfo.json'))

    def get_modelinfo(self):
        '''
        :params id: str
        '''
        res=[]
        for key in self.modelInfo.keys():
            res.append(self.modelInfo[key])
        return res
    
    def getNewId(self):
        for i in range(10000):
            if str(i+1) not in self.modelInfo:
                return str(i+1)

    def add_model(self,info):
        '''
        info: 
        {"name": "模型一",
        "type": "图像分类",
        "state": "等待开始",
        "time": "2022-3-5"}
    
        '''
        info['id']=self.getNewId()
        self.modelInfo[info['id']]=info
        IOtool.write_json(self.modelInfo,osp.join(CURR,'modelInfo.json'))
        os.mkdir(osp.join(CURR,'modelstore',info['id']))
        info['save_dir']=osp.join(CURR,'modelstore',info['id'])
        with open(osp.join(CURR,'modelstore',info['id'],'log.txt'), encoding="utf-8",mode="w") as file:  
            file.write("create a new task for {}".format(info['id'])) 
            file.write("\n"+str(json.dumps(info))+"\n")
        return info


    def change_state(self,idx,typ='train'):
        if typ=='train':
            self.modelInfo[idx]['state']="训练中"
            IOtool.write_json(self.modelInfo,osp.join(CURR,'modelInfo.json'))
        if typ=='finish':
            self.modelInfo[idx]['state']="训练完成"
            IOtool.write_json(self.modelInfo,osp.join(CURR,'modelInfo.json'))
        if typ=='error':
            self.modelInfo[idx]['state']="异常终止"
            IOtool.write_json(self.modelInfo,osp.join(CURR,'modelInfo.json'))

    def get_detailLog(self,idx):
        print(osp.join(CURR,'modelstore',idx,'log.txt'))
        Texts=IOtool.read_log(osp.join(CURR,'modelstore',idx,'log.txt'))
        historypath=osp.join(CURR,'modelstore',idx,'best_history.pkl')
        if os.path.exists(historypath):
            with open(historypath, 'rb') as f:
                log_dict = pickle.load(f)
            Texts=Texts+'\n'+'Training Monitor:\n'
            size=len(log_dict['loss'])
            print(size)
            for i in range(size):
                newcont='epoch {}: loss:{},  accuracy:{},   val_loss:{},   val_accuracy:{}'.\
                        format(i,log_dict['loss'][i],log_dict['accuracy'][i],log_dict['val_loss'][i],log_dict['val_accuracy'][i])
            
                Texts=Texts+newcont+'\n'
        else:
            Texts=Texts+'\n'+'Training Monitor:'+'\n'+'Training first epoch:\n'
        return Texts

    def update_Log(self,idx,text):
        print(text)
        with open(osp.join(CURR,'modelstore',idx,'log.txt'), encoding="utf-8",mode="a") as file:  
            file.write(text+'\n') 
    
    def finish_model(self,idx):
        self.modelInfo[idx]['state']='训练完成'
        IOtool.write_json(self.modelInfo,osp.join(CURR,'modelInfo.json'))
    
    def delete_model(self,idx):
        self.modelInfo.pop(idx)
        IOtool.write_json(self.modelInfo,osp.join(CURR,'modelInfo.json'))
        shutil.rmtree(osp.join(CURR,'modelstore',idx))
    
    def get_info_one_model(self,idx):
        return self.modelInfo[idx]


class ModelOnline():
    def __init__(self):
        self.model=None
        self.label_map=None
        self.dataAugmentationParams=None


    def BindModel(self,model_info,data_info):
        '''
        model_info -> Build The Temp
        '''

        print(model_info)
        print(data_info)

        self.label_map=data_info["label_map"]
        self.dataAugmentationParams=model_info["dataAugmentationParams"]

        model_path = osp.join(model_info["save_dir"],'best_model.h5')
        print("***********************  ",model_path)
        self.model = load_model(model_path,custom_objects=ak.CUSTOM_OBJECTS)
        
        print(self.model.summary())

        return
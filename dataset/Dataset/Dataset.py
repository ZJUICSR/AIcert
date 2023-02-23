from Utils.IOtool import IOtool
import numpy as np
import os.path as osp
import os
import shutil
CURR = osp.dirname(osp.abspath(__file__))
from PIL import Image
import zipfile
# import cv2
from io import BytesIO


class Dataset:
    @staticmethod
    def load_data(id):
        '''
        :params id: str
        '''
        name=None
        belong=None
        datasetInfo=IOtool.load_json(osp.join(CURR,'datasetInfo.json'))
        for group in ['public','customize']:
            if id in datasetInfo[group]:
                name=str(id)+'_'+datasetInfo[group][id]
                belong=group
                break
        
        if name is None:
            return 'Sorry, there is no such a dataset.'
        else:
            x=np.load(osp.join(CURR,belong+'/'+name+'/x.npy'),allow_pickle=True)
            y=np.load(osp.join(CURR,belong+'/'+name+'/y.npy'),allow_pickle=True)
            if len(y.shape)!=1:
                y=y.reshape(y.shape[0],)
            info=IOtool.load_json(osp.join(CURR,belong+'/'+name+'/info.json'))
            print('info is : ',info)
            return x,y,info
    
    @staticmethod
    def get_datainfo(dataGroup):
        '''
        :params dataGroup: str, 'public' or 'customize' or 'all'
        '''
        datasetInfo=IOtool.load_json(osp.join(CURR,'datasetInfo.json'))
        res=[]
        dataGroupSet=[]
        if dataGroup=='all':
            dataGroupSet=['public','customize']
        else:
            dataGroupSet=[dataGroup]
        for dataGroup in dataGroupSet:
            for key in datasetInfo[dataGroup].keys():
                res.append(IOtool.load_json(osp.join(CURR,str(dataGroup)+'/'+str(key)+'_'+datasetInfo[dataGroup][key]+'/info.json')))
        
        return res
    
    @staticmethod
    def show_one_sample(dataset_id,sample_id):
        '''
        :params dataset_id: str
        :params sample_id: int
        '''
        # print(CURR)
 
        res=Dataset.load_data(dataset_id)
        if res is None:
            return 'Sorry, there is no such a dataset.'
        #res=(x,y,info)
        img_=res[0][sample_id-1]
        print()
        print(str(res[1][sample_id-1]))
        label_=res[2]["label_map"][str(res[1][sample_id-1])]

        if os.path.exists(osp.join(CURR,'../../app','static/tmp'))==False:
            os.mkdir(osp.join(CURR,'../../app','static/tmp'))

        historyList=os.listdir(osp.join(CURR,'../../app','static/tmp'))
        for eachfile in historyList:
            if 'for_show' in eachfile:
                os.remove(osp.join(CURR,'../../app','static/tmp',eachfile))

        filePath=osp.join(CURR,'../../app','static/tmp/{}for_show{}.png'.format(dataset_id,sample_id-1))
        #filePath=osp.join(CURR,'../../app','tmpdata/for_show.png')

        # img = Image.fromarray((img_*255).astype('uint8'), mode='L').convert('RGB')
        img = Image.fromarray(img_) # 将array转化成图片
        
        img.save(filePath) # 保存图片

        return 'static/tmp/{}for_show{}.png'.format(dataset_id,sample_id-1),label_
    
    @staticmethod
    def add_customize_dataset(dataset_name,dataset_type):
        '''
        funcion is to allocate a new id for the new dataset, then add it into datasetInfo.json, create the dir and info.json for the new dataset.
        :parms dataset_name: str
        :parms dataset_type: str
        '''
        print("**********************")
        dataset_id=None
        tail=1000  # public dataset use id 0-99
        datasetInfo=IOtool.load_json(osp.join(CURR,'datasetInfo.json'))
        while True:
            if str(tail) in datasetInfo['customize']:
                tail+=1
            else:
                dataset_id=str(tail)
                break


        datasetInfo['customize'][dataset_id]=dataset_name
        newDatasetDir=osp.join(CURR,'customize',dataset_id+'_'+dataset_name)
        IOtool.write_json(datasetInfo,osp.join(CURR,'datasetInfo.json'))

        if os.path.exists(newDatasetDir):
            shutil.remove(newDatasetDir)
        os.mkdir(newDatasetDir)

        print(dataset_name,dataset_type,newDatasetDir)

        newDatasetinfo={'id':dataset_id,'name':dataset_name,'type':dataset_type,'num':0,'class_num':0,'label_map':{}}
        IOtool.write_json(newDatasetinfo,osp.join(newDatasetDir,'info.json'))
    
    @staticmethod
    def delete_customize_dataset(dataset_id):
        '''
        :parms dataset_id: str, unique index of a dataset
        '''
        
        datasetInfo=IOtool.load_json(osp.join(CURR,'datasetInfo.json'))
        dataset_name=datasetInfo['customize'][dataset_id]
        newDatasetDir=osp.join(CURR,'customize',dataset_id+'_'+dataset_name)
        shutil.rmtree(newDatasetDir)

        datasetInfo['customize'].pop(dataset_id)
        IOtool.write_json(datasetInfo,osp.join(CURR,'datasetInfo.json'))
    
    # @staticmethod
    # def upload_dataset_entity(dataset_id,zippath):
    #     '''
    #     :parms dataset_id: str, unique index of a dataset
    #     '''
        
    #     datasetInfo=IOtool.load_json(osp.join(CURR,'datasetInfo.json'))
    #     dataset_name=datasetInfo['customize'][dataset_id]

    #     outputPath=osp.join(CURR,'customize',dataset_id+'_'+dataset_name)

    #     x=[]
    #     y=[]
    #     num=0
    #     label={}

    #     with zipfile.ZipFile(zippath, mode='r') as zfile: # 只读方式打开压缩包
    #         nWaitTime = 1
    #         for name in zfile.namelist():  # 获取zip文档内所有文件的名称列表
    #             if '.jpg' not in name:# 仅读取.jpg图片，过滤掉文件夹，及其他非.jpg后缀文件
    #                 continue
                
    #             with zfile.open(name,mode='r') as image_file:
    #                 content = image_file.read() # 一次性读入整张图片信息
    #                 # image = np.asarray(bytearray(content), dtype='uint8')
    #                 # image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                
    #                 image = Image.open(BytesIO(content))
    #                 InitialSize=image.size
    #                 maxSize=max(InitialSize)
    #                 if maxSize>224:
    #                     image=image.resize((32,32))
    #                     # print(image.size)

    #                 img=np.array(image).astype(np.uint8)
                    
    #                 reallabel=(name.split('_'))[1]
    #                 reallabel=(reallabel.split('.'))[0]
    #                 x.append(img)
    #                 if reallabel not in label:
    #                     label[reallabel]=num
    #                     num+=1
    #                 y.append(label[reallabel])

    #                 #image_file.close()

    #                 # key = cv2.waitKey(nWaitTime)
    #                 # if 27 == key:  # ESC
    #                 #     break
    #                 # elif 32 == key:  # space
    #                 #     nWaitTime = not nWaitTime
                    

    #     x=np.array(x)
    #     y=np.array(y)
    #     labelmap={}
    #     for key in label.keys():
    #         labelmap[label[key]]=str(key)
    #     class_num=num

    #     tmpInfo=IOtool.load_json(osp.join(outputPath,'info.json'))
    #     tmpInfo['num']=x.shape[0]
    #     tmpInfo['class_num']=class_num
    #     tmpInfo['label_map']=labelmap
    #     np.save(osp.join(outputPath,'x.npy'),x)
    #     np.save(osp.join(outputPath,'y.npy'),y)
    #     IOtool.write_json(tmpInfo,osp.join(outputPath,'info.json'))

    @staticmethod
    def upload_dataset_entity(dataset_id,zippath):
        '''
        :parms dataset_id: str, unique index of a dataset
        '''
        
        datasetInfo=IOtool.load_json(osp.join(CURR,'datasetInfo.json'))
        dataset_name=datasetInfo['customize'][dataset_id]

        outputPath=osp.join(CURR,'customize',dataset_id+'_'+dataset_name)

        iniInfo=IOtool.load_json(osp.join(outputPath,'info.json'))

        num=iniInfo['class_num']
        labelmap=iniInfo['label_map']

        label={}
        for idlabel in labelmap.keys():
            label[labelmap[idlabel]]=int(idlabel)

        if num!=0:
            print("********************")
            x=np.load(osp.join(outputPath,'x.npy'),allow_pickle=True)
            y=np.load(osp.join(outputPath,'y.npy'),allow_pickle=True)
            x=list(x)
            y=list(y)
        else:
            x=[]
            y=[]

        with zipfile.ZipFile(zippath, mode='r') as zfile: # 只读方式打开压缩包
            nWaitTime = 1
            for name in zfile.namelist():  # 获取zip文档内所有文件的名称列表
                if '.jpg' not in name:# 仅读取.jpg图片，过滤掉文件夹，及其他非.jpg后缀文件
                    continue
                
                with zfile.open(name,mode='r') as image_file:
                    content = image_file.read() # 一次性读入整张图片信息
                    # image = np.asarray(bytearray(content), dtype='uint8')
                    # image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                
                    image = Image.open(BytesIO(content))
                    InitialSize=image.size
                    maxSize=max(InitialSize)
                    if maxSize>224:
                        image=image.resize((32,32))
                        # print(image.size)

                    img=np.array(image).astype(np.uint8)
                    
                    reallabel=(name.split('_'))[1]
                    reallabel=(reallabel.split('.'))[0]
                    x.append(img)
                    if reallabel not in label:
                        label[reallabel]=num
                        num+=1
                    y.append(label[reallabel])

                    #image_file.close()

                    # key = cv2.waitKey(nWaitTime)
                    # if 27 == key:  # ESC
                    #     break
                    # elif 32 == key:  # space
                    #     nWaitTime = not nWaitTime
                    

        x=np.array(x)
        y=np.array(y)
        labelmap={}
        for key in label.keys():
            labelmap[label[key]]=str(key)
        class_num=num

        tmpInfo=IOtool.load_json(osp.join(outputPath,'info.json'))
        tmpInfo['num']=x.shape[0]
        tmpInfo['class_num']=class_num
        tmpInfo['label_map']=labelmap
        np.save(osp.join(outputPath,'x.npy'),x)
        np.save(osp.join(outputPath,'y.npy'),y)
        IOtool.write_json(tmpInfo,osp.join(outputPath,'info.json'))
    

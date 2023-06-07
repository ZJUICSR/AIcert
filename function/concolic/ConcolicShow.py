import os
import sys
from shutil import copyfile
import time
import numpy
import random
from torch.nn import Module
from torch.utils.data import DataLoader
import zipfile
from threading import Thread
import torch
import torch.nn as nn
from torch.nn import Module
import torchvision
import torch.utils.data as Data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import json
import gc



def count_sum_number_zip(path):
    z = zipfile.ZipFile(path, "r")
    znamelist = z.namelist()
    return int(len(znamelist) / 3)


class ConcolicShow(object):
    def __init__(self):
        super(ConcolicShow, self).__init__()

    # def run(model:Module, data:DataLoader, model_name:str, data_name:str, norm:str='l0')

    def show_results(model, data, model_name: str, data_name: str, norm: str, basepath: str, out_path:str,Times:int=0, logging:str=None):
        '''
        model_name: a string of the name of the model.
        data_name: name of the dataset: 'cifar10','mnist'
        norm: methods to guide generation: 'l0','linf'
        '''

        ### we only provide results for vgg16 in cifar10


        json_data={}


        # if (('vgg' not in model_name) and ('VGG' not in model_name)) or ('cifar10' not in data_name):
            
        #     json_data['allnumber'] = 0
        #     json_data['demopath'] = 'none'

            
        #     return
        

        historyNum = 0
        if os.path.exists(basepath + '/GeneratedCases/' + data_name + "_" + norm + "/" + 'results'):
            historyNum = int(len(os.listdir(basepath + '/GeneratedCases/' + data_name + "_" + norm + "/" + 'results')) / 3)

        # ## dynamic generating sample cases
        if Times!=0:
            # try:
            ConcolicShow.Dynamic_run(model, data, model_name, data_name, norm, basepath, times=Times, logging=logging)
            # except:
                
            #     pass


        ### calculate the number of all cases
        allNumber = 0
        if os.path.exists(basepath + '/GeneratedCases/' + data_name + "_" + norm + "/" + 'results'):
            allNumber = int(len(os.listdir(basepath + '/GeneratedCases/' + data_name + "_" + norm + "/" + 'results')) / 3)
        
        newNumber = allNumber - historyNum

        ### provide demo images for showing.
        relative_path = data_name + "_" + norm + "/"
        outImgPath = os.path.join(out_path, relative_path)
        if not os.path.exists(out_path):
            os.mkdir(out_path)

        # demofilespath = basepath + '/demoimgs/' + data_name + "_" + norm + "/"
        demofilespath = './dataset/data/demoimgs/' + data_name + "_" + norm + "/"

        imglist = os.listdir(demofilespath)

        imglist.sort()
        selected_index = random.sample(range(1, int(len(imglist) / 3) - 1), 5)

        showimgs = {}

        if os.path.exists(outImgPath):
            shutil.rmtree(outImgPath)
            os.mkdir(outImgPath)
        else:
            os.mkdir(outImgPath)

        for i in range(5):
            idx = selected_index[i]
            copyfile(demofilespath + imglist[idx * 3], outImgPath + imglist[idx * 3])
            copyfile(demofilespath + imglist[idx * 3 + 1], outImgPath + imglist[idx * 3 + 1])
            copyfile(demofilespath + imglist[idx * 3 + 2], outImgPath + imglist[idx * 3 + 2])

            # showimgs[str(i) + '_diff'] = relative_path + imglist[idx * 3]
            # showimgs[str(i) + '_init'] = relative_path + imglist[idx * 3 + 1]
            # showimgs[str(i) + '_new'] = relative_path + imglist[idx * 3 + 2]
            showimgs[str(i) + '_diff'] = outImgPath + imglist[idx * 3]
            showimgs[str(i) + '_init'] = outImgPath + imglist[idx * 3 + 1]
            showimgs[str(i) + '_new'] = outImgPath + imglist[idx * 3 + 2]

        
        # print(json_data['TestCaseGeneration'])
        json_data['allnumber'] = allNumber
        json_data['newnumber'] = newNumber
        json_data['demopath'] = showimgs

        return json_data



    def Dynamic_run(model, data, model_name: str, data_name: str, norm: str, basepath: str, times, logging: None):
        '''
        times : [1,inf], performing algorithms one time needs around 30-45s...
        '''
        if data_name == 'cifar10':

            if os.path.exists(basepath + '/GeneratedCases/' + data_name + "_" + norm + "/") == False:
                os.makedirs(basepath + '/GeneratedCases/' + data_name + "_" + norm + "/")

            testTargetLayers=['conv1_1']
            if norm=='l0':
                testTargetLayers.append('conv1_2')

            net, loadr = Cifar10.get_back(basepath)
            for i in range(times):
                try:
                    TestMachine = InvConcolic(logging)
                    TestMachine.run(model=net, seedsLoader=loadr, datasetLoader=loadr, device="cpu",
                                    outdir=basepath + '/GeneratedCases/' + data_name + "_" + norm + "/",
                                    norm=norm, initnumber=8, reset=0, metric='nc', maxiteration=40,
                                    feature_area=[-1.0, 1.0], testTargetLayer=testTargetLayers,
                                    postpre=Cifar10.ReNormalization, RandSeed=None)
                    del TestMachine
                    gc.collect()
                except KeyboardInterrupt:
                    del TestMachine
                    gc.collect()
                    return
                except:
                    del TestMachine
                    gc.collect()

        if data_name == 'mnist':
            net, loadr = MNIST.get_back(basepath)
            for i in range(times):
                try:
                    TestMachine = InvConcolic()
                    TestMachine.run(model=net, seedsLoader=loadr, datasetLoader=loadr, device="cpu",
                                    outdir=basepath + '/GeneratedCases/' + data_name + "_" + norm + "/",
                                    norm=norm, initnumber=6, reset=0, metric='nc', maxiteration=30, testTargetLayer=[],
                                    postpre=MNIST.ReNormalization, RandSeed=None)
                    del TestMachine
                except KeyboardInterrupt:
                    del TestMachine
                    gc.collect()
                    return
                except:
                    del TestMachine
                    gc.collect()


class LeNet(Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.flat1 = nn.Flatten()
        self.fc1 = nn.Linear(256, 120)
        self.dr1 = nn.Dropout(p=0.2)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = self.flat1(y)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y


class VGG16_torch(nn.Module):
    def __init__(self):
        super(VGG16_torch, self).__init__()

        # GROUP 1
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1,
                                 padding=(1, 1))  # output:32*32*64
        self.relu1_1 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                                 padding=(1, 1))  # output:32*32*64
        self.relu1_2 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2)  # After pooling, the length and width are halved output:16*16*64

        # GROUP 2
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1,
                                 padding=(1, 1))  # output:16*16*128
        self.relu2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1,
                                 padding=(1, 1))  # output:16*16*128
        self.relu2_2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2)  # After pooling, the length and width are halved output:8*8*128

        # GROUP 3
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1,
                                 padding=(1, 1))  # output:8*8*256
        self.relu3_1 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1,
                                 padding=(1, 1))  # output:8*8*256
        self.relu3_2 = nn.ReLU()
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1)  # output:8*8*256
        self.relu3_3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(2)  # After pooling, the length and width are halved output:4*4*256

        # GROUP 4
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1,
                                 padding=1)  # output:4*4*512
        self.relu4_1 = nn.ReLU()
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,
                                 padding=1)  # output:4*4*512
        self.relu4_2 = nn.ReLU()
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1)  # output:4*4*512
        self.relu4_3 = nn.ReLU()
        self.maxpool4 = nn.MaxPool2d(2)  # After pooling, the length and width are halved output:2*2*512

        # GROUP 5
        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,
                                 padding=1)  # output:14*14*512
        self.relu5_1 = nn.ReLU()
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,
                                 padding=1)  # output:14*14*512
        self.relu5_2 = nn.ReLU()
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1)  # output:14*14*512
        self.relu5_3 = nn.ReLU()
        self.maxpool5 = nn.MaxPool2d(2)  # After pooling, the length and width are halved output:1*1*512

        self.flatten1 = nn.Flatten()

        self.fc1 = nn.Linear(in_features=512 * 1 * 1, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=10)

        # Define forward propagation

    def forward(self, x):
        input_dimen = x.size(0)

        # GROUP 1
        output = self.conv1_1(x)
        output = self.relu1_1(output)
        output = self.conv1_2(output)
        output = self.relu1_2(output)
        output = F.relu(output)
        output = self.maxpool1(output)

        # GROUP 2
        output = self.conv2_1(output)
        output = self.relu2_1(output)
        output = self.conv2_2(output)
        output = self.relu2_2(output)
        output = self.maxpool2(output)

        # GROUP 3
        output = self.conv3_1(output)
        output = self.relu3_1(output)
        output = self.conv3_2(output)
        output = self.relu3_2(output)
        output = self.conv3_3(output)
        output = self.relu3_3(output)
        output = self.maxpool3(output)

        # GROUP 4
        output = self.conv4_1(output)
        output = self.relu4_1(output)
        output = self.conv4_2(output)
        output = self.relu4_2(output)
        output = self.conv4_3(output)
        output = self.relu4_3(output)
        output = self.maxpool4(output)

        # GROUP 5
        output = self.conv5_1(output)
        output = self.relu5_1(output)
        output = self.conv5_2(output)
        output = self.relu5_2(output)
        output = self.conv5_3(output)
        output = self.relu5_3(output)
        output = self.maxpool5(output)

        output = self.flatten1(output)

        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)

        # Return to output
        return output


class MNIST(object):
    def __init__(self):
        super(MNIST, self).__init__()

    def get_back(basepath):
        # mnist_test = torchvision.datasets.MNIST(root=basepath + '/Utils/Datasets', train=False, download=True,
        #                                         transform=transforms.ToTensor())
        mnist_test = torchvision.datasets.MNIST(root='./dataset/data', train=False, download=True,
                                                transform=transforms.ToTensor())
        batch_size = 128
        if sys.platform.startswith('win'):
            num_workers = 0
        else:
            num_workers = 4

        test_iter = Data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        net = LeNet()
        # net.load_state_dict(torch.load(basepath + '/Utils/Models/MNIST_lenet.pth'))
        net.load_state_dict(torch.load('./model/ckpt/MNIST_lenet.pth'))
        return net, test_iter

    def ReNormalization(inputs):
        '''
        input 是一张numpy的（c,:,:）图
        '''
        return inputs


class Cifar10(object):
    def __init__(self):
        super(Cifar10, self).__init__()

    def get_back(basepath):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        # testset = torchvision.datasets.CIFAR10(root=basepath + '/Utils/Datasets', train=False,
        #                                        download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./dataset/data', train=False,
                                               download=True, transform=transform)
        batch_size = 128
        if sys.platform.startswith('win'):
            num_workers = 0
        else:
            num_workers = 4
        test_iter = Data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        net = VGG16_torch()
        # net.load_state_dict(torch.load(basepath + '/Utils/Models/cifar10_vgg16.pth'))
        net.load_state_dict(torch.load('./model/ckpt/cifar10_vgg16.pth'))
        return net, test_iter

    def ReNormalization(inputs):
        '''
        input 是一张numpy的（c,:,:）图
        '''
        x = inputs.copy()
        x = x * 0.5 + 0.5
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image
import matplotlib.pyplot as plt
# import zip
import numpy as np
from torch.nn import Module
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import sys
from typing import Any, AnyStr, Callable, Dict, Union, List
import random
from datetime import datetime
import copy
from pulp import *
import time
import shutil


class InvConcolic(object):
    '''
    Report：records of print information

    '''
    def __init__(self, logging):
        super(InvConcolic, self).__init__()
        self.Report=[]
        self.logging = logging

    def run(self, model:Module, seedsLoader:DataLoader, datasetLoader:DataLoader, outdir:str,norm:str="l0", device:str='cpu', \
        maxiteration:int=-1, initnumber:int=1, metric='nc',feature_area:list=[0,1],testTargetLayer=[],reset=0,postpre=None,RandSeed=None):


        
        self.Report=[]
        self.outdir=outdir

        if os.path.exists(self.outdir)==False:
            os.mkdir(self.outdir)

        if os.path.exists(self.outdir+'/'+'results'+'/')==False:
            os.mkdir(self.outdir+'/'+'results'+'/')

        if reset==1:
            shutil.rmtree(outdir)
            os.mkdir(outdir)
            # os.mkdir(self.outdir+'/'+'new'+'/')
            # os.mkdir(self.outdir+'/'+'seed'+'/')
            os.mkdir(self.outdir+'/'+'results'+'/')

        self.testTargetLayer=testTargetLayer
        self.Rp('Initializing the testing framework......')
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.randomSeed=RandSeed     # 随机种子，为了实验的可复现性
        # self.Rp("device is "+str(self.device))
        self.feature_area=feature_area
        self.model = model.to(device)
        self.model.eval()
        self.excludeLayer = ['Pool','Flatten','ReLU','Dropout']
        self.allLayers, self.testableLayers = self.layers_detection(model)
        self.metric=metric
        self.norm=norm
        self.beforeSaveImg=postpre
        self.Variable={}
        self.failed_neuron=[]
        self.activation = {} 
        self.compute_magnitude_coefficients(datasetLoader)
        self.testSuite=[]
        self.add_sample(self.randomly_select_seeds(seedsLoader, self.randomSeed, initnumber))
        self.Rp("Randomly selected "+str(initnumber)+" seeds...")

        self.engine(maxiteration)
    
    def add_sample(self,X,Types:str="seed",others:str=""):
        for x in X:
            self.testSuite.append(x.copy())
            # if Types=="seed":
            #     self.Rp(Types+" images, id = "+str(len(self.testSuite)-1)+"!  ")
            # else:
            #     self.Rp(Types+" images, id = "+str(len(self.testSuite)-1)+"!  "+others+"  (Coverage is "+str(self.coverage)+")")

            # if not os.path.exists(self.outdir+'/'+Types+'/'):
            #     os.mkdir(self.outdir+'/'+Types+'/')
            # self.save_img(x, self.outdir+'/'+Types+'/'+str(len(self.testSuite)-1)+'.jpg')
    
    def distance_calculate(self, x, y, norm):
        z=x-y
        if norm=='l0':
            z=np.abs(z)
            z[z>1e-5]=1
            return z.sum()
        if norm=='linf':
            return np.max(np.abs(z))

    def save_img(self,iniimgs,outpath):
        img=self.beforeSaveImg(iniimgs)
        img=img.transpose(1,2,0)
        if len(img.shape)>2 and img.shape[2]==1:
            img=img.reshape((img.shape[0],img.shape[1]))

            img*=255
            img=img.astype(np.uint8)
            
            im=Image.fromarray(img,mode='L')
            im.save(outpath)
            #plt.figure()
            #plt.imshow(img, cmap = 'binary') #将图像黑白显示
            #plt.savefig(outpath)


        if len(img.shape)>2 and img.shape[2]==3:
   
            #img=(img-self.feature_area[0])/(self.feature_area[1]-self.feature_area[0])
            #img=(img*0.5)+0.5
            #img=img.reshape((img.shape[0],img.shape[1],img.shape[2]))
            #print(type(img),"  ",np.min(img),"  ",np.max(img))
            #plt.figure()
            #plt.imshow(img) #将图像黑白显示
            #plt.savefig(outpath)
            img*=255
            img=img.astype(np.uint8)
            im=Image.fromarray(img)
            im.save(outpath)
            

        #pic = Image.fromarray(img)
        
        #pic.save(outpath)
    def Remove_failed_r(self,r):
        self.failed_neuron.append(r)
        r=r.replace('+', ' ')
        r=r.split()
        for i in range(len(r)-1):
            r[i+1]=int(r[i+1])
        if len(r)==4:
            self.uncovered_neuron_origin.pop(r[0]+"+"+str(r[1])+"+"+str(r[2])+"+"+str(r[3]))
            self.uncovered_neuron_values.pop(r[0]+"+"+str(r[1])+"+"+str(r[2])+"+"+str(r[3]))
        else:
            self.uncovered_neuron_origin.pop(r[0]+"+"+str(r[1]))
            self.uncovered_neuron_values.pop(r[0]+"+"+str(r[1]))

    def engine(self, maxiteration):
        '''
            run the test
        '''
        
        self.initialize_coverage(self.metric)
        
        if self.norm=='linf':
            #self.test_conv(copy.deepcopy(self.testSuite[0]))
            self.constraints_preprocess(copy.deepcopy(self.testSuite[0]))
            self.Rp("Based Constraints are over!")
        iters=0
        # print("init is over!!")
        while len(self.uncovered_neuron_values)!=0 and (maxiteration < 0 or iters<maxiteration):
            while True and (maxiteration < 0 or iters<maxiteration):
                iters+=1
                r,t = self.requirement_execuation()
                # self.Rp("Aim to cover :"+r)
            
                t_new= self.symbolic_analysis(r,copy.deepcopy(self.testSuite[t]))
               
                if (self.validity_check(t_new)==True):
                    
                    stramp=datetime.now()
                    date_time = stramp.strftime("%Y-%m-%d-%H-%M-%S")

                    rand_id=random.randint(0,10000)
                    rand_id='-'+str(rand_id)

                    diff=t_new.copy()-self.testSuite[t]
                    
                    diff=np.absolute(diff)

                    
                    diff[diff>(1.0/266.0)]=1

                    fenzi=np.sum(diff)
                    fenmu=diff.shape[0]*diff.shape[1]*diff.shape[2]
                    # print(fenzi/fenmu)

                    if fenzi/fenmu<1:
                        self.save_img(self.testSuite[t],self.outdir+'/results/'+str(date_time)+rand_id+'_init.png')
                        self.save_img(t_new.copy(),self.outdir+'/results/'+str(date_time)+rand_id+'_new.png')
                        self.save_img(diff,self.outdir+'/results/'+str(date_time)+rand_id+'_diff.png')

                    # print(np.max(diff),"  ",np.min(diff),"  ",np.mean(diff))
                    #print(diff)
                    #diff[diff>0]=1.0
                    
                    #print(self.outdir+'/results/'+str(date_time)+'_init.jpg')
                    
                    

                    self.add_sample([t_new.copy()],"new"," origin is  "+str(t)+"  "+"distance is  "+str(self.distance_calculate(self.testSuite[t],t_new,self.norm)))
                    self.update_coverage(t_new,len(self.testSuite)-1)
                    break
                else:
                    # self.Rp("failed!")
                    self.Remove_failed_r(r)
                    break
                   
    def validity_check(self, t):
        '''
        检查生成的图片是否是合法输入
        '''
        if t is None:
            return False
        return True

    def test_conv(self,inputs):
        Tar_model=self.model
        layer_name='conv1'
        exec("hook_"+str(layer_name)+" = "+"Tar_model."+str(layer_name)+".register_forward_hook(self._get_activation('"+str(layer_name)+"'))")
        tp = torch.from_numpy(inputs)
        outputs = Tar_model(torch.unsqueeze(tp,dim=0))
        Weight=np.array(self.Weightdict[layer_name+'.weight'])
        Bias=np.array(self.Weightdict[layer_name+'.bias'])
        layer_output=self.activation[layer_name]
        # print("Bias.shape : ",Bias.shape)
        # print("Weight.shape : ",Weight.shape)
        values = np.array(layer_output)
        values = values[0]
        Shape=values.shape
        kernel_size=(5,5)
        stride=(1,1)
        padding=0
        for i in range(Shape[0]):
            for j in range(Shape[1]):
                for k in range(Shape[2]):

                    idx=j*stride[0]-padding
                    idy=k*stride[1]-padding
                    tmp=0.0
                    for jj in range(kernel_size[0]):
                        for kk in range(kernel_size[1]):
                            for d in range(tp.shape[0]):
                                tmp+=inputs[d][jj+idx][kk+idy]*Weight[i][d][jj][kk]
                                
                    tmp+=Bias[i]
                    # if (tmp!=values[i][j][k]):
                    #     print(i,"  ",j,"  ","  ",k,"   ",tmp,"   ",values[i][j][k])



        exec("hook_"+str(layer_name)+".remove()")    
        self.activation.clear()
    
    def constraints_preprocess(self, inputs):
      
        for i in range(inputs.shape[0]):
            for j in range(inputs.shape[1]):
                for k in range(inputs.shape[2]):
                    self.Variable["INPUT+"+str(i)+"+"+str(j)+"+"+str(k)] = LpVariable("INPUT+"+str(i)+"+"+str(j)+"+"+str(k),lowBound=self.feature_area[0],upBound=self.feature_area[1],cat='Continuous')

        Tar_model = self.model
        for layer_name in self.allLayers:
            exec("hook_"+str(layer_name)+" = "+"Tar_model."+str(layer_name)+".register_forward_hook(self._get_activation('"+str(layer_name)+"'))")

       
        tp = torch.from_numpy(inputs)
        outputs = Tar_model(torch.unsqueeze(tp,dim=0))
        info=[]
        for layer_name in self.allLayers:
            Type_ = self.layerType[layer_name]
            if layer_name in self.no_value_layer:
                layer_shape_=info[-1][1]
                info.append([Type_,layer_shape_])
                continue

            layer_shape_ =(np.array(self.activation[layer_name]).shape)[1:]
            info.append([Type_,layer_shape_]) 

        for layer_name in self.allLayers :
            exec("hook_"+str(layer_name)+".remove()")
        self.activation.clear()

        cnt=0
        self.neuron_pre={}

       
        LpBase=LpProblem(name='Linf_generation',sense=LpMinimize)
  

        self.Variable["dis"]=LpVariable("dis",lowBound=1.0/255,upBound=1,cat='Continuous')
        LpBase+=self.Variable["dis"]
        #LpBase+=LpConstraint(LpAffineExpression([(self.Variable["dis"],1)]),LpConstraintGE,"distance",)
        cnt=0

        self.LpBaseDict={}
        self.LpBaseDict["INPUT"]=LpBase.copy()
        
        for lay in range(len(self.allLayers)):
          
            layer_name = self.allLayers[lay]
 

            cnt=lay
            #self.Rp("Bulit base constraints for "+layer_name+" ...")
            # TODO:padding目前只支持zero-padding
            if "Conv" in info[cnt][0]:
                PreLayerName = ""
                PreShape = 0
                if cnt==0:
                    PreLayerName = "INPUT"
                    PreShape=inputs.shape
                else:
                    PreLayerName = self.allLayers[cnt-1]
                    PreShape=info[cnt-1][1]
                
                Shape=info[cnt][1]
                module=self.layer_module[layer_name]
                kernel_size=self.get_module_information(module,'kernel_size')
                stride=self.get_module_information(module,'stride')
                padding=self.get_module_information(module,'padding')
                Weight=np.array(self.Weightdict[layer_name+'.weight'])
                Bias=np.array(self.Weightdict[layer_name+'.bias'])
                # print(Shape[0]*Shape[1]*Shape[2])
                for i in range(Shape[0]):
                    for j in range(Shape[1]):
                        for k in range(Shape[2]):
                            self.Variable[layer_name+"+"+str(i)+"+"+str(j)+"+"+str(k)] = LpVariable(layer_name+"+"+str(i)+"+"+str(j)+"+"+str(k))

                            idx=j*stride[0]-padding[0]
                            idy=k*stride[1]-padding[1]
                            x = []
                            y = []
                            for jj in range(kernel_size[0]):
                                for kk in range(kernel_size[1]):
                                    for d in range(PreShape[0]):
                                        if jj+idx<0 or kk+idy<0 or jj+idx>=PreShape[1] or kk+idy>=PreShape[2]:
                                            continue
                                        x.append(self.Variable[PreLayerName+"+"+str(d)+"+"+str(jj+idx)+"+"+str(kk+idy)])
                                        y.append(Weight[i][d][jj][kk])
                            x.append(self.Variable[layer_name+"+"+str(i)+"+"+str(j)+"+"+str(k)])
                            y.append(-1.0)
                            c=LpConstraint(LpAffineExpression([(x[i],y[i]) for i in range(len(x))]), LpConstraintEQ, "base"+layer_name+"+"+str(i)+"+"+str(j)+"+"+str(k),-1.0*Bias[i])
                            LpBase+=c
                
            elif "MaxPool" in info[cnt][0]:
                
                PreLayerName = self.allLayers[cnt-1]
                PreShape=info[cnt-1][1]
                Shape=info[cnt][1]
                kernel_size=(2,2)
                stride=(2,2)
                padding=0

                for i in range(Shape[0]):
                    for j in range(Shape[1]):
                        for k in range(Shape[2]):
                            self.Variable[layer_name+"+"+str(i)+"+"+str(j)+"+"+str(k)] = LpVariable(layer_name+"+"+str(i)+"+"+str(j)+"+"+str(k))

                            idx=j*stride[0]-padding
                            idy=k*stride[1]-padding
                            for jj in range(kernel_size[0]):
                                for kk in range(kernel_size[1]):
                                    if jj+idx<0 or kk+idy<0 or jj+idx>=PreShape[1] or kk+idy>=PreShape[2]:
                                        continue
                                    c=LpConstraint(LpAffineExpression([(self.Variable[layer_name+"+"+str(i)+"+"+str(j)+"+"+str(k)],1),(self.Variable[PreLayerName+"+"+str(i)+"+"+str(jj+idx)+"+"+str(kk+idy)],-1)]),\
                                                        LpConstraintGE, "base"+layer_name+"+"+str(i)+"+"+str(j)+"+"+str(k)+str(jj)+"+"+str(kk),0.)
                                    LpBase+=c

            elif "ReLU" in info[cnt][0]:
                Shape=info[cnt][1]
                if len(Shape)==3:
                    for i in range(Shape[0]):
                        for j in range(Shape[1]):
                            for k in range(Shape[2]):
                                self.Variable[layer_name+"+"+str(i)+"+"+str(j)+"+"+str(k)] = LpVariable(layer_name+"+"+str(i)+"+"+str(j)+"+"+str(k),lowBound=0)
                if len(Shape)==2:
                    for i in range(Shape[0]):
                        for j in range(Shape[1]):
                            self.Variable[layer_name+"+"+str(i)+"+"+str(j)] = LpVariable(layer_name+"+"+str(i)+"+"+str(j),lowBound=0)
                if len(Shape)==1:
                    for i in range(Shape[0]):
                        self.Variable[layer_name+"+"+str(i)] = LpVariable(layer_name+"+"+str(i),lowBound=0)

            elif "Flatten" in info[cnt][0]:
                PreLayerName = self.allLayers[cnt-1]
                PreShape=info[cnt-1][1]

                Shape=info[cnt][1]
           

                for i in range(PreShape[0]):
                    for j in range(PreShape[1]):
                        for k in range(PreShape[2]):
                            self.Variable[layer_name+"+"+str(i*(PreShape[1]*PreShape[2])+j*PreShape[2]+k)] = self.Variable[PreLayerName+"+"+str(i)+"+"+str(j)+"+"+str(k)]


            elif "Linear" in info[cnt][0]:        
                PreLayerName = self.allLayers[cnt-1]
                PreShape=info[cnt-1][1]
                Shape=info[cnt][1]
                Weight=np.array(self.Weightdict[layer_name+'.weight'])
                Bias=np.array(self.Weightdict[layer_name+'.bias'])

                for j in range(Shape[0]):

                    self.Variable[layer_name+"+"+str(j)]=LpVariable(layer_name+"+"+str(j))
                    x=[]
                    y=[]
                    
                    for i in range(PreShape[0]):
                        x.append(self.Variable[PreLayerName+"+"+str(i)])
                        y.append(Weight[j][i])
                        
                    x.append(self.Variable[layer_name+"+"+str(j)])
                    y.append(-1)
                    c=LpConstraint(LpAffineExpression([(x[i],y[i]) for i in range(len(x))]), LpConstraintEQ, "base"+layer_name+"+"+str(j),-1.0*Bias[j])
                    LpBase+=c
            elif "Dropout" in info[cnt][0]:
                PreLayerName = self.allLayers[cnt-1]
                PreShape=info[cnt-1][1]
                for i in range(PreShape[0]):
                    self.Variable[layer_name+"+"+str(i)] = self.Variable[PreLayerName+"+"+str(i)]
            else:
                self.Rp("Error: Unknown layer! ")
                return
            
            self.LpBaseDict[layer_name]=LpBase.copy()



        return


    def symbolic_analysis(self, r, t):
        '''
        进行符号分析
        '''
        if self.norm=='l0':
            return self.adhoc_global_optimization(r,t)
        if self.norm=='linf':    
            return self.linear_programming_faster(r,t)

    def linear_programming(self,r,t):
        '''
        针对于Linf范数约束的线性规划符号求解
        '''
        r=r.replace('+', ' ')
        r=r.split()
        for i in range(len(r)-1):
            r[i+1]=int(r[i+1])
        
        target_layer = r[0]
        target_layer_index = self.index_of_layer[target_layer]
        
        # bulit constraints of nerous
        LpBase = self.LpBaseDict[target_layer].copy()


        Tar_model = self.model
        
        for i in range(target_layer_index+1):
            layer_name=self.allLayers[i]
            exec("hook_"+str(layer_name)+"="+"Tar_model."+str(layer_name)+".register_forward_hook(self._get_activation('"+str(layer_name)+"'))")
        
        tp = torch.from_numpy(t)
        outputs = Tar_model(torch.unsqueeze(tp,dim=0))
        for itm in range(target_layer_index+1):
            layer_name=self.allLayers[itm]
            Type_ = self.layerType[layer_name]
            layer_output=self.activation[layer_name]
            values = np.array(layer_output)
            values = values[0]
            Shape=values.shape
            if "ReLU" in Type_:
                PreLayerName=self.allLayers[itm-1]
                if len(Shape)==3:
                    for i in range(Shape[0]):
                        for j in range(Shape[1]):
                            for k in range(Shape[2]):
                                if values[i][j][k]>0:
                                    c=LpConstraint(LpAffineExpression([(self.Variable[layer_name+"+"+str(i)+"+"+str(j)+"+"+str(k)],1),(self.Variable[PreLayerName+"+"+str(i)+"+"+str(j)+"+"+str(k)],-1)]),\
                                                        LpConstraintEQ, "new"+layer_name+"+"+str(i)+"+"+str(j)+"+"+str(k),0.)
                                    LpBase+=c
                                else:
                                    c=LpConstraint(LpAffineExpression([(self.Variable[layer_name+"+"+str(i)+"+"+str(j)+"+"+str(k)],1)]),\
                                                        LpConstraintEQ, "new"+layer_name+"+"+str(i)+"+"+str(j)+"+"+str(k),0.)
                                    LpBase+=c

                if len(Shape)==2:
                    for i in range(Shape[0]):
                        for j in range(Shape[1]):
                            if values[i][j]>0:
                                c=LpConstraint(LpAffineExpression([(self.Variable[layer_name+"+"+str(i)+"+"+str(j)],1),(self.Variable[PreLayerName+"+"+str(i)+"+"+str(j)],-1)]),\
                                                    LpConstraintEQ, "new"+layer_name+"+"+str(i)+"+"+str(j),0.)
                                LpBase+=c
                            else:
                                c=LpConstraint(LpAffineExpression([(self.Variable[layer_name+"+"+str(i)+"+"+str(j)],1)]),\
                                                    LpConstraintEQ, "new"+layer_name+"+"+str(i)+"+"+str(j),0.)
                                LpBase+=c

                if len(Shape)==1:
                    for i in range(Shape[0]):
                        if values[i]>0:
                            c=LpConstraint(LpAffineExpression([(self.Variable[layer_name+"+"+str(i)],1),(self.Variable[PreLayerName+"+"+str(i)],-1)]),\
                                                LpConstraintEQ, "new"+layer_name+"+"+str(i),0.)
                            LpBase+=c
                        else:
                            c=LpConstraint(LpAffineExpression([(self.Variable[layer_name+"+"+str(i)],1)]),\
                                                LpConstraintEQ, "new"+layer_name+"+"+str(i),0.)
                            LpBase+=c

            elif "Conv" in Type_:
                if itm==target_layer_index:
                    i = r[1]
                    j = r[2]
                    k = r[3]
              
                    if values[i][j][k]>0:
                        c=LpConstraint(LpAffineExpression([(self.Variable[layer_name+"+"+str(i)+"+"+str(j)+"+"+str(k)],1)]),\
                                            LpConstraintLE, "new"+layer_name+"+"+str(i)+"+"+str(j)+"+"+str(k),0)
               
                        LpBase+=c
                    else:
                        c=LpConstraint(LpAffineExpression([(self.Variable[layer_name+"+"+str(i)+"+"+str(j)+"+"+str(k)],1)]),\
                                            LpConstraintGE, "new"+layer_name+"+"+str(i)+"+"+str(j)+"+"+str(k),1e-7)
                        LpBase+=c
                    
                    break

                for i in range(Shape[0]):
                    for j in range(Shape[1]):
                        for k in range(Shape[2]):
                            if values[i][j][k]>0:
                                c=LpConstraint(LpAffineExpression([(self.Variable[layer_name+"+"+str(i)+"+"+str(j)+"+"+str(k)],1)]),\
                                                    LpConstraintGE, "new"+layer_name+"+"+str(i)+"+"+str(j)+"+"+str(k),1e-7)
                                LpBase+=c
                            else:
                                c=LpConstraint(LpAffineExpression([(self.Variable[layer_name+"+"+str(i)+"+"+str(j)+"+"+str(k)],1)]),\
                                                    LpConstraintLE, "new"+layer_name+"+"+str(i)+"+"+str(j)+"+"+str(k),0.)
                                LpBase+=c
            elif "Linear" in Type_:
                if itm==target_layer_index:
                    i = r[1]
                
                    if values[i]>0:
                   
                        c=LpConstraint(LpAffineExpression([(self.Variable[layer_name+"+"+str(i)],1)]),\
                                            LpConstraintLE, "new"+layer_name+"+"+str(i),0)
                        LpBase+=c
                    else:
                        c=LpConstraint(LpAffineExpression([(self.Variable[layer_name+"+"+str(i)],1)]),\
                                            LpConstraintGE, "new"+layer_name+"+"+str(i),1e-3)
                        LpBase+=c
                
                    break



                for i in range(Shape[0]):
                    if values[i]>0:
                        c=LpConstraint(LpAffineExpression([(self.Variable[layer_name+"+"+str(i)],1)]),\
                                            LpConstraintGE, "new"+layer_name+"+"+str(i),1e-7)
                        LpBase+=c
                    else:
                        c=LpConstraint(LpAffineExpression([(self.Variable[layer_name+"+"+str(i)],1)]),\
                                            LpConstraintLE, "new"+layer_name+"+"+str(i),0.)
                        LpBase+=c
            elif "MaxPool" in Type_:
                PreLayerOuput=np.array(self.activation[self.allLayers[itm-1]])
                PreLayerOuput=PreLayerOuput[0]

                PreLayerName = self.allLayers[itm-1]
                PreShape=PreLayerOuput.shape
                
                kernel_size=(2,2)
                stride=(2,2)
                padding=0
                for i in range(Shape[0]):
                    for j in range(Shape[1]):
                        for k in range(Shape[2]):
                            idx=j*stride[0]-padding
                            idy=k*stride[1]-padding
                            for jj in range(kernel_size[0]):
                                for kk in range(kernel_size[1]):
                                    if values[i][j][k]==PreLayerOuput[i][j+jj][k+kk]:
                                        c=LpConstraint(LpAffineExpression([(self.Variable[layer_name+"+"+str(i)+"+"+str(j)+"+"+str(k)],1),(self.Variable[PreLayerName+"+"+str(i)+"+"+str(jj+idx)+"+"+str(kk+idy)],-1)]),\
                                                            LpConstraintEQ, "new"+layer_name+"+"+str(i)+"+"+str(j)+"+"+str(k)+str(jj)+"+"+str(kk),0.)
                                        LpBase+=c
                del PreLayerOuput

            elif "Flatten" in Type_:
                pass
            else:
                self.Rp("Error: Unknown layer! ")
                return None

        del outputs
        # bulit constrainst of Linf between t and t_new

    
    
       
        for i in range(t.shape[0]):
            for j in range(t.shape[1]):
                for k in range(t.shape[2]):
                    LpBase+=LpConstraint(LpAffineExpression([(self.Variable["INPUT+"+str(i)+"+"+str(j)+"+"+str(k)],1),(self.Variable["dis"],1)]),LpConstraintGE,"newdis1"+"+"+str(i)+"+"+str(j)+"+"+str(k),t[i][j][k])
                    LpBase+=LpConstraint(LpAffineExpression([(self.Variable["INPUT+"+str(i)+"+"+str(j)+"+"+str(k)],1),(self.Variable["dis"],-1)]),LpConstraintLE,"newdis2"+"+"+str(i)+"+"+str(j)+"+"+str(k),t[i][j][k])

        self.Rp("Solving "+str(len(LpBase.constraints))+" constraints in total...")

        #LpBase.solve(pulp.PULP_CBC_CMD(timeLimit='60'))
      
        solver=getSolver('PULP_CBC_CMD',msg=0)
   
        #return None
        eventlet.monkey_patch()
        with eventlet.Timeout(20,False):
            resl=LpBase.solve(solver)
        status = LpStatus[LpBase.status]
        answer=None

        if status == 'Optimal':
            res={}
            for v in LpBase.variables():
                
                if "INPUT" in v.name:
                    res[v.name]=v.varValue
            t_new=copy.deepcopy(t)
            for i in range(t.shape[0]):
                for j in range(t.shape[1]):
                    for k in range(t.shape[2]):
                        t_new[i][j][k]=res["INPUT"+"_"+str(i)+"_"+str(j)+"_"+str(k)]
        
            return t_new.copy()
            tp = torch.from_numpy(t_new)
            Tar_model(torch.unsqueeze(tp,dim=0))
            layer_output=self.activation[target_layer]
            values = np.array(layer_output)
            values = values[0]
            target_sta=0
            if len(r)==4:
                target_sta=values[r[1]][r[2]][r[3]]
            else:
                target_sta=values[r[1]]
            if target_sta>0:
           
                answer=t_new.copy()

        for i in range(target_layer_index+1):
            layer_name=self.allLayers[i]
        self.activation.clear()

        return answer
        

    def linear_programming_faster(self,r,t):
        '''
        针对于Linf范数约束的线性规划符号求解
        '''
        copyr=copy.deepcopy(r)
        r=r.replace('+', ' ')
        r=r.split()
        for i in range(len(r)-1):
            r[i+1]=int(r[i+1])
        
        target_layer = r[0]
        target_layer_index = self.index_of_layer[target_layer]
        
        # bulit constraints of nerous
        pre_indx="INPUT"
        if target_layer_index>0:
            pre_indx=self.allLayers[target_layer_index-1]
        LpBase = self.LpBaseDict[pre_indx].copy()


        Tar_model = self.model
        
        for i in range(target_layer_index+1):
            layer_name=self.allLayers[i]
            exec("hook_"+str(layer_name)+"="+"Tar_model."+str(layer_name)+".register_forward_hook(self._get_activation('"+str(layer_name)+"'))")

        Shape=t.shape
        tp = torch.from_numpy(t)
        outputs = Tar_model(torch.unsqueeze(tp,dim=0))
        for itm in range(target_layer_index+1):
            PreShape=copy.deepcopy(Shape)
            layer_name=self.allLayers[itm]
            Type_ = self.layerType[layer_name]
            if "Dropout" in Type_:
                continue
            layer_output=self.activation[layer_name]
            values = np.array(layer_output)
            values = values[0]
            Shape=values.shape
            if "ReLU" in Type_:
                PreLayerName=self.allLayers[itm-1]
                if len(Shape)==3:
                    for i in range(Shape[0]):
                        for j in range(Shape[1]):
                            for k in range(Shape[2]):
                                if values[i][j][k]>0:
                                    c=LpConstraint(LpAffineExpression([(self.Variable[layer_name+"+"+str(i)+"+"+str(j)+"+"+str(k)],1),(self.Variable[PreLayerName+"+"+str(i)+"+"+str(j)+"+"+str(k)],-1)]),\
                                                        LpConstraintEQ, "new"+layer_name+"+"+str(i)+"+"+str(j)+"+"+str(k),0.)
                                    LpBase+=c
                                else:
                                    c=LpConstraint(LpAffineExpression([(self.Variable[layer_name+"+"+str(i)+"+"+str(j)+"+"+str(k)],1)]),\
                                                        LpConstraintEQ, "new"+layer_name+"+"+str(i)+"+"+str(j)+"+"+str(k),0.)
                                    LpBase+=c

                if len(Shape)==2:
                    for i in range(Shape[0]):
                        for j in range(Shape[1]):
                            if values[i][j]>0:
                                c=LpConstraint(LpAffineExpression([(self.Variable[layer_name+"+"+str(i)+"+"+str(j)],1),(self.Variable[PreLayerName+"+"+str(i)+"+"+str(j)],-1)]),\
                                                    LpConstraintEQ, "new"+layer_name+"+"+str(i)+"+"+str(j),0.)
                                LpBase+=c
                            else:
                                c=LpConstraint(LpAffineExpression([(self.Variable[layer_name+"+"+str(i)+"+"+str(j)],1)]),\
                                                    LpConstraintEQ, "new"+layer_name+"+"+str(i)+"+"+str(j),0.)
                                LpBase+=c

                if len(Shape)==1:
                    for i in range(Shape[0]):
                        if values[i]>0:
                            c=LpConstraint(LpAffineExpression([(self.Variable[layer_name+"+"+str(i)],1),(self.Variable[PreLayerName+"+"+str(i)],-1)]),\
                                                LpConstraintEQ, "new"+layer_name+"+"+str(i),0.)
                            LpBase+=c
                        else:
                            c=LpConstraint(LpAffineExpression([(self.Variable[layer_name+"+"+str(i)],1)]),\
                                                LpConstraintEQ, "new"+layer_name+"+"+str(i),0.)
                            LpBase+=c

            elif "Conv" in Type_:
                if itm==target_layer_index:
                    module=self.layer_module[layer_name]
                    kernel_size=self.get_module_information(module,'kernel_size')
                    stride=self.get_module_information(module,'stride')
                    padding=self.get_module_information(module,'padding')
                    Weight=np.array(self.Weightdict[layer_name+'.weight'])
                    Bias=np.array(self.Weightdict[layer_name+'.bias'])
                    i = r[1]
                    j = r[2]
                    k = r[3]
                    self.Variable[layer_name+"+"+str(i)+"+"+str(j)+"+"+str(k)] = LpVariable(layer_name+"+"+str(i)+"+"+str(j)+"+"+str(k),cat='Continuous')
                    PreLayerName = ""
                  
                    if itm==0:
                        PreLayerName = "INPUT"
                    else:
                        PreLayerName = self.allLayers[itm-1]
                        
                    idx=j*stride[0]-padding[0]
                    idy=k*stride[1]-padding[1]
                    x = []
                    y = []
                  
                    for jj in range(kernel_size[0]):
                        for kk in range(kernel_size[1]):
                            for d in range(Weight.shape[1]):
                                if jj+idx<0 or kk+idy<0 or jj+idx>=PreShape[1] or kk+idy>=PreShape[2]:
                                    continue
                                x.append(self.Variable[PreLayerName+"+"+str(d)+"+"+str(jj+idx)+"+"+str(kk+idy)])
                                y.append(Weight[i][d][jj][kk])
                    x.append(self.Variable[layer_name+"+"+str(i)+"+"+str(j)+"+"+str(k)])
                    y.append(-1.0)
                    c=LpConstraint(LpAffineExpression([(x[o],y[o]) for o in range(len(x))]), LpConstraintEQ, "base"+layer_name+"+"+str(i)+"+"+str(j)+"+"+str(k),-1.0*Bias[i])
        
                    LpBase+=c

                   
                    if values[i][j][k]>0:
                        
                        c=LpConstraint(LpAffineExpression([(self.Variable[layer_name+"+"+str(i)+"+"+str(j)+"+"+str(k)],1)]),\
                                            LpConstraintLE, "new"+layer_name+"+"+str(i)+"+"+str(j)+"+"+str(k),0)
                 
                        LpBase+=c
                    else:
                        c=LpConstraint(LpAffineExpression([(self.Variable[layer_name+"+"+str(i)+"+"+str(j)+"+"+str(k)],1)]),\
                                            LpConstraintGE, "new"+layer_name+"+"+str(i)+"+"+str(j)+"+"+str(k),1e-5)
                        LpBase+=c
                     
                    
                    break

                for i in range(Shape[0]):
                    for j in range(Shape[1]):
                        for k in range(Shape[2]):
                            if values[i][j][k]>0:
                                c=LpConstraint(LpAffineExpression([(self.Variable[layer_name+"+"+str(i)+"+"+str(j)+"+"+str(k)],1)]),\
                                                    LpConstraintGE, "new"+layer_name+"+"+str(i)+"+"+str(j)+"+"+str(k),1e-5)
                                LpBase+=c
                            else:
                                c=LpConstraint(LpAffineExpression([(self.Variable[layer_name+"+"+str(i)+"+"+str(j)+"+"+str(k)],1)]),\
                                                    LpConstraintLE, "new"+layer_name+"+"+str(i)+"+"+str(j)+"+"+str(k),0.)
                                LpBase+=c

            elif "Linear" in Type_:
                if itm==target_layer_index:
                    i = r[1]
                    PreLayerName = self.allLayers[itm-1]
                    Weight=np.array(self.Weightdict[layer_name+'.weight'])
                    Bias=np.array(self.Weightdict[layer_name+'.bias'])
                    self.Variable[layer_name+"+"+str(i)]=LpVariable(PreLayerName+"+"+str(i))
                    x=[]
                    y=[]
                    
                    for j in range(PreShape[0]):
                        x.append(self.Variable[PreLayerName+"+"+str(j)])
                        y.append(Weight[i][j])
                        
                    x.append(self.Variable[layer_name+"+"+str(i)])
                    y.append(-1)
                    c=LpConstraint(LpAffineExpression([(x[o],y[o]) for o in range(len(x))]), LpConstraintEQ, "base"+layer_name+"+"+str(i),-1.0*Bias[i])
                    LpBase+=c

                    
                    if values[i]>0:
                      
                        c=LpConstraint(LpAffineExpression([(self.Variable[layer_name+"+"+str(i)],1)]),\
                                            LpConstraintLE, "new"+layer_name+"+"+str(i),0.)
                        LpBase+=c
                    else:
                        c=LpConstraint(LpAffineExpression([(self.Variable[layer_name+"+"+str(i)],1)]),\
                                            LpConstraintGE, "new"+layer_name+"+"+str(i),1e-5)
                        LpBase+=c
                
                    break


                for i in range(Shape[0]):
                    if values[i]>0:
                        c=LpConstraint(LpAffineExpression([(self.Variable[layer_name+"+"+str(i)],1)]),\
                                            LpConstraintGE, "new"+layer_name+"+"+str(i),1e-5)
                        LpBase+=c
                    else:
                        c=LpConstraint(LpAffineExpression([(self.Variable[layer_name+"+"+str(i)],1)]),\
                                            LpConstraintLE, "new"+layer_name+"+"+str(i),0.)
                        LpBase+=c

            elif "MaxPool" in Type_:
                PreLayerOuput=np.array(self.activation[self.allLayers[itm-1]])
                PreLayerOuput=PreLayerOuput[0]

                PreLayerName = self.allLayers[itm-1]
                PreShape=PreLayerOuput.shape
                
                module=self.layer_module[layer_name]
                kernel_size=self.get_module_information(module,'kernel_size')
                stride=self.get_module_information(module,'stride')
                padding=self.get_module_information(module,'padding')

                for i in range(Shape[0]):
                    for j in range(Shape[1]):
                        for k in range(Shape[2]):
                            idx=j*stride[0]-padding[0]
                            idy=k*stride[1]-padding[1]
                            for jj in range(kernel_size[0]):
                                for kk in range(kernel_size[1]):
                                    if jj+idx<0 or kk+idy<0 or jj+idx>=PreShape[1] or kk+idy>=PreShape[2]:
                                        continue
                                    if values[i][j][k]==PreLayerOuput[i][j+jj][k+kk]:
                                        c=LpConstraint(LpAffineExpression([(self.Variable[layer_name+"+"+str(i)+"+"+str(j)+"+"+str(k)],1),(self.Variable[PreLayerName+"+"+str(i)+"+"+str(jj+idx)+"+"+str(kk+idy)],-1)]),\
                                                            LpConstraintEQ, "new"+layer_name+"+"+str(i)+"+"+str(j)+"+"+str(k)+str(jj)+"+"+str(kk),0.)
                                        LpBase+=c
                del PreLayerOuput

            elif "Flatten" in Type_ or "Dropout" in Type_:
                pass
            else:
                self.Rp("Error: Unknown layer! ")
                return None

        del outputs
        # bulit constrainst of Linf between t and t_new
    
      
        

        for i in range(t.shape[0]):
            for j in range(t.shape[1]):
                for k in range(t.shape[2]):
                    LpBase+=LpConstraint(LpAffineExpression([(self.Variable["INPUT+"+str(i)+"+"+str(j)+"+"+str(k)],1),(self.Variable["dis"],1)]),LpConstraintGE,"newdis1"+"+"+str(i)+"+"+str(j)+"+"+str(k),t[i][j][k])
                    LpBase+=LpConstraint(LpAffineExpression([(self.Variable["INPUT+"+str(i)+"+"+str(j)+"+"+str(k)],1),(self.Variable["dis"],-1)]),LpConstraintLE,"newdis2"+"+"+str(i)+"+"+str(j)+"+"+str(k),t[i][j][k])
                    
        self.Rp("Solving "+str(len(LpBase.constraints))+" constraints in total...")
        answer=None
        #LpBase.solve(pulp.PULP_CBC_CMD(timeLimit='60'))
        
        #solver=getSolver('GLPK_CMD',msg=0,timeLimit=60)
        #solver=getSolver('PULP_CBC_CMD',msg=0,timeLimit=10)
        #solver=getSolver('CPLEX_PY',msg=0)

    
        try:
            #solver=getSolver('COIN_CMD',msg=0,timeLimit=60)
            solver=getSolver('PULP_CBC_CMD',msg=0,timeLimit=60)
            # print("asdasdasd")
        #return None
        #try:
        #LpBase.writeLP('debug.lp')
            resl=LpBase.solve(solver)
            status = LpStatus[LpBase.status]
        
        except:
            for i in range(target_layer_index+1):
                layer_name=self.allLayers[i]
            self.activation.clear()

            return None
        
        if status == 'Optimal':
            res={}
            for v in LpBase.variables():
                if "INPUT" in v.name:
                    res[v.name]=v.varValue
            t_new=copy.deepcopy(t)
            for i in range(t.shape[0]):
                for j in range(t.shape[1]):
                    for k in range(t.shape[2]):
                        t_new[i][j][k]=res["INPUT"+"_"+str(i)+"_"+str(j)+"_"+str(k)]
        
            #return t_new.copy()
            tp = torch.from_numpy(t_new)
            Tar_model(torch.unsqueeze(tp,dim=0))
            layer_output=self.activation[target_layer]
            values = np.array(layer_output)
            values = values[0]
            target_sta=0
            if len(r)==4:
                target_sta=values[r[1]][r[2]][r[3]]
            else:
                target_sta=values[r[1]]
            if target_sta>0:
                answer=t_new.copy()
                self.Rp("Successfully solved a new input to cover target requirement!")
        #except:
            #LpBase.writeLP('./data/'+r+'.lp')
        #    for i in range(target_layer_index+1):
        #        layer_name=self.allLayers[i]
        #    self.activation.clear()
        #    return None


        
        for i in range(target_layer_index+1):
            layer_name=self.allLayers[i]
        self.activation.clear()

        return answer



    def Rp(self, strs: str):
        self.Report.append(strs)
        # print(strs)
        self.logging.info(strs)
        return None

    def layers_detection(self, model:Module):
        '''
            Check which layers can be used to be tested.
            Layers can not be tested are listed in 'self.excludeLayer'
            Input: A model
            Output: two str list, one include names of all layers, another include names of layers to be tested.
        '''
        modules = model.named_children()
        testable=[]
        allayer=[]
        self.no_value_layer=[]
        self.index_of_layer={}
        self.index_of_testablelayer={}
        self.layerType={}
        self.layer_module={}

        brk="none"
        for name, module in modules:
      
            if name in self.testTargetLayer:
                brk=name
   
        modules = model.named_children()
        for name, module in modules:
         
            allayer.append(name)
            self.index_of_layer[name]=len(allayer)-1
            self.layerType[name]=str(type(module))
            if all(ex not in str(type(module)) for ex in self.excludeLayer):
                if len(self.testTargetLayer)==0 or (name in self.testTargetLayer):
                    testable.append(name)
                    self.index_of_testablelayer[name] = len(testable)-1
            
            if 'Conv' in str(type(module)) or 'MaxPool' in str(type(module)):
                self.layer_module[name]=module
                #print("module : ",module.kernel_size)
            
            if 'Dropout' in str(type(module)):
                self.no_value_layer.append(name)

            # 不关心待测试层后面的层
            if name==brk:
                break

        #print(allayer)
        self.Rp("There are "+str(len(allayer))+" layers :"+str(allayer))
        self.Rp(str(len(testable))+" layers to be tested :"+str(testable))
        self.Weightdict = model.state_dict()
        #print(self.Weightdict)
        # print("Testable! == ",testable)
        return allayer, testable

    def get_module_information(self,module,arg:str):
        if arg=='kernel_size':
            kernel_size=module.kernel_size
            if 'int' in str(type(kernel_size)):
                return (kernel_size,kernel_size)
            else:
                return kernel_size

        if arg=='padding':
            padding=module.padding
            if 'int' in str(type(padding)):
                return (padding,padding)
            else:
                return padding
        
        if arg=='stride':
            stride=module.stride
            if 'int' in str(type(stride)):
                return (stride,stride)
            else:
                return stride

    def _get_activation(self, name:str):
        def hook(model, input, output):
            # if need feature to feedback，delete detach（）
            self.activation[name] = output.detach()
        return hook

    def compute_magnitude_coefficients(self, datasets: DataLoader):
     
        Tar_model = self.model
        coeff=np.zeros(len(self.testableLayers))
        # 使用hook建立模型中间值输出与activation字典的联系，之后每次使用模型预测结果，中间值的输出都会保存到activation字典中
        self.activation.clear()
        for layer_name in self.testableLayers :
            exec("hook_"+str(layer_name)+" ="+"Tar_model."+str(layer_name)+".register_forward_hook(self._get_activation('"+str(layer_name)+"'))")

        for number, (input_data,_) in enumerate(datasets):
            outputs = Tar_model(input_data)
            for layer_name, layer_output in self.activation.items():
               
                values = np.array(layer_output)
                maxabs = np.max(np.abs(values))
                if maxabs> coeff[self.index_of_testablelayer[layer_name]]:
                    coeff[self.index_of_testablelayer[layer_name]] = maxabs
                
            del outputs

        for layer_name in self.testableLayers :
            exec("hook_"+str(layer_name)+".remove()")
        
        self.activation.clear()

        for i in range(coeff.shape[0]):
            coeff[i]=1.0/coeff[i]
        
        self.coeff={}
        for i in range(coeff.shape[0]):
            self.coeff[self.testableLayers[i]]=coeff[i]

        self.Rp("The magnitude coefficients of testable layers are :\n"+str(coeff))

        return 

    def update_coverage(self, inputs, origin):
        '''
        inputs : 新生成的样本
        origin ： 这个样本的编号
        '''
        if self.metric=='nc':
            Tar_model = self.model
            for layer_name in self.testableLayers :
                exec("hook_"+str(layer_name)+"="+"Tar_model."+str(layer_name)+".register_forward_hook(self._get_activation('"+str(layer_name)+"'))")
            
            tp = torch.from_numpy(inputs)
            outputs = Tar_model(torch.unsqueeze(tp,dim=0))
            for layer_name, layer_output in self.activation.items():
                values = np.array(layer_output)

                if len(values.shape)==2:
                    for j in range(values.shape[1]):
                        new_name=layer_name+"+"+str(j)
                        if new_name in self.uncovered_neuron_values:
                            if values[0][j]>0:
                                self.uncovered_neuron_values.pop(new_name)
                                self.uncovered_neuron_origin.pop(new_name)
                                continue
                            if self.uncovered_neuron_values[new_name] < values[0][j]*self.coeff[layer_name]:
                                self.uncovered_neuron_values[new_name] = values[0][j]*self.coeff[layer_name]
                                self.uncovered_neuron_origin[new_name] = origin
            
                elif len(values.shape)==3:
                    for j in range(values.shape[1]):
                        for k in range(values.shape[2]):
                            new_name=layer_name+"+"+str(j)+"+"+str(k)
                            if new_name in self.uncovered_neuron_values:
                                if values[0][j][k]>0:
                                    self.uncovered_neuron_values.pop(new_name)
                                    self.uncovered_neuron_origin.pop(new_name)
                                    continue

                                if self.uncovered_neuron_values[new_name] < values[0][j][k]*self.coeff[layer_name]:
                                    self.uncovered_neuron_values[new_name] = values[0][j][k]*self.coeff[layer_name]
                                    self.uncovered_neuron_origin[new_name] = origin

                elif len(values.shape)==4:
                    for j in range(values.shape[1]):
                        for k in range(values.shape[2]):
                            for l in range(values.shape[3]):
                                new_name=layer_name+"+"+str(j)+"+"+str(k)+"+"+str(l)
                                if new_name in self.uncovered_neuron_values:
                                    if values[0][j][k][l]>0:
                                        self.uncovered_neuron_values.pop(new_name)
                                        self.uncovered_neuron_origin.pop(new_name)
                                        continue

                                    if self.uncovered_neuron_values[new_name] < values[0][j][k][l]*self.coeff[layer_name]:
                                        self.uncovered_neuron_values[new_name] = values[0][j][k][l]*self.coeff[layer_name]
                                        self.uncovered_neuron_origin[new_name] = origin

            del outputs

            for layer_name in self.testableLayers :
                exec("hook_"+str(layer_name)+".remove()")
            self.activation.clear()

            self.coverage = 1.0- float(len(self.uncovered_neuron_values)+len(self.failed_neuron))/self.neuron_number 
    
    def initialize_coverage(self, metric):
        '''
            根据选择的种子，初始化覆盖
        '''
        if metric=='nc':
            self.coverage = 0.0
            self.neuron_number = 0
            self.uncovered_neuron_values={}
            self.uncovered_neuron_origin={}

            Tar_model = self.model
            for layer_name in self.testableLayers :
                exec("hook_"+str(layer_name)+"="+"Tar_model."+str(layer_name)+".register_forward_hook(self._get_activation('"+str(layer_name)+"'))")


            for i in range(len(self.testSuite)):
                tp = torch.from_numpy(self.testSuite[i])
                outputs = Tar_model(torch.unsqueeze(tp,dim=0))
                for layer_name, layer_output in self.activation.items():
                    values = np.array(layer_output)
                    #print(values.shape)

                    if len(values.shape)==2:
                        if i==0:
                            for j in range(values.shape[1]):
                                self.neuron_number+=1
                                if values[0][j]<=0:
                                    new_name=layer_name+"+"+str(j)
                                    self.uncovered_neuron_values[new_name] = values[0][j]*self.coeff[layer_name]
                                    self.uncovered_neuron_origin[new_name] = i
                        else:
                            for j in range(values.shape[1]):
                                new_name=layer_name+"+"+str(j)
                                if new_name in self.uncovered_neuron_values:
                                    if values[0][j]>0:
                                        self.uncovered_neuron_values.pop(new_name)
                                        self.uncovered_neuron_origin.pop(new_name)
                                        continue
                                    if self.uncovered_neuron_values[new_name] < values[0][j]*self.coeff[layer_name]:
                                        self.uncovered_neuron_values[new_name] = values[0][j]*self.coeff[layer_name]
                                        self.uncovered_neuron_origin[new_name] = i
                    
                    elif len(values.shape)==3:
                        if i==0:
                            for j in range(values.shape[1]):
                                for k in range(values.shape[2]):
                                    
                                        self.neuron_number+=1
                                        if values[0][j][k][l]<=0:
                                            new_name=layer_name+"+"+str(j)+"+"+str(k)
                                            self.uncovered_neuron_values[new_name] = values[0][j][k]*self.coeff[layer_name]
                                            self.uncovered_neuron_origin[new_name] = i
                        else:
                            for j in range(values.shape[1]):
                                for k in range(values.shape[2]):
                                    new_name=layer_name+"+"+str(j)+"+"+str(k)
                                    if new_name in self.uncovered_neuron_values:
                                        if values[0][j][k]>0:
                                            self.uncovered_neuron_values.pop(new_name)
                                            self.uncovered_neuron_origin.pop(new_name)
                                            continue

                                        if self.uncovered_neuron_values[new_name] < values[0][j][k]*self.coeff[layer_name]:
                                            self.uncovered_neuron_values[new_name] = values[0][j][k]*self.coeff[layer_name]
                                            self.uncovered_neuron_origin[new_name] = i

                    elif len(values.shape)==4:
                        if i==0:
                            for j in range(values.shape[1]):
                                for k in range(values.shape[2]):
                                    for l in range(values.shape[3]):
                                        self.neuron_number+=1
                                        if values[0][j][k][l]<=0:
                                            new_name=layer_name+"+"+str(j)+"+"+str(k)+"+"+str(l)
                                            self.uncovered_neuron_values[new_name] = values[0][j][k][l]*self.coeff[layer_name]
                                            self.uncovered_neuron_origin[new_name] = i
                        else:
                            for j in range(values.shape[1]):
                                for k in range(values.shape[2]):
                                    for l in range(values.shape[3]):
                                        new_name=layer_name+"+"+str(j)+"+"+str(k)+"+"+str(l)
                                        if new_name in self.uncovered_neuron_values:
                                            if values[0][j][k][l]>0:
                                                self.uncovered_neuron_values.pop(new_name)
                                                self.uncovered_neuron_origin.pop(new_name)
                                                continue

                                            if self.uncovered_neuron_values[new_name] < values[0][j][k][l]*self.coeff[layer_name]:
                                                self.uncovered_neuron_values[new_name] = values[0][j][k][l]*self.coeff[layer_name]
                                                self.uncovered_neuron_origin[new_name] = i

                del outputs
            
            for layer_name in self.testableLayers :
                exec("hook_"+str(layer_name)+".remove()")
            self.activation.clear()

            self.coverage = 1.0- float(len(self.uncovered_neuron_values)+len(self.failed_neuron))/self.neuron_number 

            self.Rp("Initial coverage is "+str(self.coverage))

        else:
            '''
            TODO
            '''    
    
    def requirement_execuation(self):
        L = sorted(self.uncovered_neuron_values.items(),key=lambda item:item[1],reverse=True)
        L = L[0]
        return L[0], self.uncovered_neuron_origin[L[0]]

    def adhoc_global_optimization(self,r,t):
        '''
            针对l0-norm的生成方法，输出找到的满足覆盖需求r的新样本 np.array；
            找不到则输出None
        '''
        trytimes=100
        pixelist=[]
        for i in range(trytimes):
            pixelist.append([random.randint(0,t.shape[1]-1),random.randint(0,t.shape[2]-1)])
        gran=2 # 像素划分数
        valuelist =np.linspace(self.feature_area[0],self.feature_area[1],gran)
        
        new_examples=np.zeros((trytimes*gran,t.shape[0],t.shape[1],t.shape[2]))
        for i in range(trytimes):
            for j in range(gran):
                new_examples[i*gran+j]=t.copy()
                new_examples[i*gran+j,:,pixelist[i][0],pixelist[i][1]]=valuelist[j]
        r=r.replace('+', ' ')
        r=r.split()
        for i in range(len(r)-1):
            r[i+1]=int(r[i+1])
        
        Tar_model = self.model
        layer=r[0]
        new_examples = (torch.from_numpy(new_examples)).to(torch.float32)

        exec("hook_"+str(layer)+"="+"Tar_model."+str(layer)+".register_forward_hook(self._get_activation('"+str(layer)+"'))")
        outputs = Tar_model(new_examples)
        tmp=[]
        for layer_name, layer_output in self.activation.items():
            values = np.array(layer_output)
            #print("values.shape ",values.shape,"  ",r)
            for i in range(new_examples.shape[0]):
                if len(r)==2:
                    tmp.append(values[i][r[1]])
                if len(r)==3:
                    tmp.append(values[i][r[1]][r[2]])
                if len(r)==4:
                    tmp.append(values[i][r[1]][r[2]][r[3]])
            
        maxvalue=max(tmp)
        
        exec("hook_"+str(layer)+".remove()")
        self.activation.clear()
    
        if maxvalue<=0:
            return None
        maxidx=tmp.index(maxvalue)

        return np.array(new_examples[maxidx]).copy()

    def randomly_select_seeds(self, seedsLoader:DataLoader, randomSeed, initnumber):
        cnt=0
        for number, (input_data,_) in enumerate(seedsLoader):
            cnt+=_.shape[0]
        if randomSeed!=None:
            random.seed(randomSeed)

        idxList=random.sample(range(0,cnt), initnumber)
        idxList.sort()
        resList=[]
        pos=0
        cnt=0
        for number, (input_data,_) in enumerate(seedsLoader):
            while cnt+(_.shape[0]) > idxList[pos] and cnt<=idxList[pos]:
                resList.append(np.array(input_data[idxList[pos]-cnt]))
                pos+=1
                if pos==len(idxList):
                    break
            if pos==len(idxList):
                break
            cnt+=(_.shape[0])
        
        return resList
    
    def run_concolic_test(model:Module, seedsLoader:DataLoader, datasetLoader:DataLoader, outdir:str='./outs/', device:str='cpu', norm:str='l0', \
        maxiteration:int=-1, initnumber:int=1, metric:str='nc',feature_area:list=[0,1],postpre=None,testTargetLayer=[],reset=0,testLayers=None,RandSeed=None):

        TestMachine = DNNConcolic(model=model, seedsLoader=seedsLoader, datasetLoader=datasetLoader,device=device,outdir=outdir,norm=norm,maxiteration=maxiteration,\
        initnumber=initnumber,metric=metric,feature_area=feature_area,testTargetLayer=testTargetLayer,reset=reset,postpre=postpre,RandSeed=RandSeed)
        return

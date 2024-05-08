#coding=utf-8
import json
import numpy as np

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

def write_json(data,path):
    """
    :param data: a dictionary
    :param path:
    """
    with open(path,'w',encoding='utf-8') as f:
        json.dump(data,f,indent=4)



# from keras.datasets import mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()


# print(x_train.shape)
# print(y_train.shape)
# print(y_train[0:10])

# tabel={}
# tabel[0]='0'
# tabel[1]='1'
# tabel[2]='2'
# tabel[3]='3'
# tabel[4]='4'
# tabel[5]='5'
# tabel[6]='6'
# tabel[7]='7'
# tabel[8]='8'
# tabel[9]='9'

# Restrian={'id':1,'name':'MNIST_traing','type':'Image','num':x_train.shape[0],'class_num':10,'label_map':tabel}
# Restest={'id':1,'name':'MNIST_test','type':'Text','num':x_test.shape[0],'class_num':10,'label_map':tabel}

# np.save("/data/jwp/Platform/v1/dataBase/Dataset/public/MNIST_train/x.npy",x_train)
# np.save("/data/jwp/Platform/v1/dataBase/Dataset/public/MNIST_train/y.npy",y_train)
# write_json(Restrian,"/data/jwp/Platform/v1/dataBase/Dataset/public/MNIST_train/info.json")


# np.save("/data/jwp/Platform/v1/dataBase/Dataset/public/MNIST_test/x.npy",x_test)
# np.save("/data/jwp/Platform/v1/dataBase/Dataset/public/MNIST_test/y.npy",y_test)
# write_json(Restest,"/data/jwp/Platform/v1/dataBase/Dataset/public/MNIST_test/info.json")




from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape)
print(y_train.shape)
print(y_train[0:10])

tabel={}
tabel["0"]='airplane'
tabel["1"]='automobile'
tabel["2"]='bird'
tabel["3"]='cat'
tabel["4"]='deer'
tabel["5"]='dog'
tabel["6"]='frog'
tabel["7"]='horse'
tabel["8"]='ship'
tabel["9"]='truck'

Restrian={'id':1,'name':'CIFAR10_traing','type':'Image','num':x_train.shape[0],'class_num':10,'label_map':tabel}
Restest={'id':1,'name':'CIFAR10_test','type':'Text','num':x_test.shape[0],'class_num':10,'label_map':tabel}

np.save("/data/jwp/Platform/v1/dataBase/Dataset/public/CIFAR10_train/x.npy",x_train)
np.save("/data/jwp/Platform/v1/dataBase/Dataset/public/CIFAR10_train/y.npy",y_train)
write_json(Restrian,"/data/jwp/Platform/v1/dataBase/Dataset/public/CIFAR10_train/info.json")


np.save("/data/jwp/Platform/v1/dataBase/Dataset/public/CIFAR10_test/x.npy",x_test)
np.save("/data/jwp/Platform/v1/dataBase/Dataset/public/CIFAR10_test/y.npy",y_test)
write_json(Restest,"/data/jwp/Platform/v1/dataBase/Dataset/public/CIFAR10_test/info.json")
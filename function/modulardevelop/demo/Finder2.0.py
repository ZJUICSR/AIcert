import os
import math
list_position=[]

def ReadAndCheck(dir,string1,list=list_position):
    file = open(dir, 'r',encoding='gb18030', errors='ignore')
    text_list=file.readlines()
    string_list=string1.split('\\')
    for line in range(len(text_list)):
        sum=0
        for string in string_list:
            temp_result=text_list[line].find(string)
            temp_result1=text_list[line].find(str.lower(string))
            if temp_result!=-1 or temp_result1 !=-1:
                sum+=1
        if sum>=len(string_list):
            list.append(str(dir)+'-'+str(line+1))
            list.append(text_list[line])
            try:
                list.append(text_list[line+1])
                list.append(text_list[line+2])
            except:
                continue
    file.close()

def file_name(file_dir,target):
    for root,dirs,files in os.walk(file_dir):
        for eachfile in files:
            if eachfile.split('.')[-1]=='py':#):#
            # if (eachfile.split('.')[-1]=='cc' or  eachfile.split('.')[-1]=='cpp' or  eachfile.split('.')[-1]=='cxx' or eachfile.split('.')[-1]=='h'):
                filename=root+'/'+eachfile
                ReadAndCheck(filename,target,list_position)
        #print(root)  # 当前目录路径
        #print(dirs)  # 当前路径下所有子目录
        #print(files)  # 当前路径下所有非目录子文件
    return list_position

#file_name("/data/zxy/pytorch/","pooling_output_shape")
file_name("/data/zxy/anaconda3/envs/ak_2.3/lib/python3.6/site-packages/onnx2torch/","xy")
#file_name("/data/zxy/DL_framework/tensorflow-1.14.0/tensorflow","case Padding::SAME")
#file_name("/data/zxy/anaconda3/envs/py36/lib/python3.6/site-packages/keras","ReLU(")
fo = open("/data/zxy/Projects/2020_ZhongDian/ak_test/zdyf_akkt/position.txt", 'w+',encoding='utf-8')
for positions in list_position:
    fo.write(str(positions))
    fo.write('\n')
fo.close()
print('finish')

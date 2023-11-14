#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 11:01:17 2023

@author: ubuntu
"""
import subprocess
def get_cuda_version():
    command = "nvidia-smi"
    open_process = subprocess.Popen(
        command,
        stdout = subprocess.PIPE,
        shell=True)
    cmd_out = open_process.stdout.read() #读取信息，并返回值
    open_process.stdout.close() #关闭输出通道

    #print("终端输出：",cmd_out.decode(encoding='gbk'))
    index = 0
    for i in range(len(cmd_out.decode(encoding='gbk'))):
        if str(cmd_out.decode(encoding='gbk')[i:i+4]) == "CUDA":
            index = i
    cuda_version = str(cmd_out.decode(encoding='gbk')[index:index+18])
    cuda_version = cuda_version.split(" ")[2]
    return cuda_version
    
def framework_test(frame,version):
    # TODO: use value to replace the string to determine the outputs.
    message = ""
    cuda_version = get_cuda_version()
    if frame == 'pytorch':
         if cuda_version == "7.5":
             if version > "1.0.0":
                 message = f"Pytorch version {version} is not aviliable with cuda version {cuda_version}! Please update your cuda or change pytorch version"
         elif cuda_version == "8.0":
             if version not in ["1.1.0","1.0.0","0.4.1"]:
                 message = f"Pytorch version {version} is not aviliable with cuda version {cuda_version}! Please update your cuda or change pytorch version"
         elif cuda_version == "9.0":
             if version not in ["1.1.0","1.0.0","1.0.1","0.4.1"]:
                 message = f"Pytorch version {version} is not aviliable with cuda version {cuda_version}! Please update your cuda or change pytorch version"
         elif cuda_version == "9.2":
             if version not in ["1.7.1","1.7.0","1.6.0","1.5.1","1.5.0","1.4.0","1.2.0","0.4.1"]:
                 message = f"Pytorch version {version} is not aviliable with cuda version {cuda_version}! Please update your cuda or change pytorch version"
         elif cuda_version == "10.0":
             if version not in ["1.2.0","1.1.0","1.0.1","1.0.0"]:
                 message = f"Pytorch version {version} is not aviliable with cuda version {cuda_version}! Please update your cuda or change pytorch version"
         elif cuda_version =="10.1":
             if version not in ["1.7.1","1.7.0","1.6.0","1.5.1","1.5.0","1.4.0","1.3.0","1.8.1"]:
                 message = f"Pytorch version {version} is not aviliable with cuda version {cuda_version}! Please update your cuda or change pytorch version"
         elif cuda_version == "10.2":
             if version not in ["1.7.1","1.7.0","1.6.0","1.5.1","1.5.0","1.8.0","1.8.1","1.9.0","1.9.1","1.10.0","1.10.1","1.11.0","1.12.0","1.12.1"]:
                 message = f"Pytorch version {version} is not aviliable with cuda version {cuda_version}! Please update your cuda or change pytorch version"
         elif cuda_version == "11.0":
             if version not in ["1.7.1","1.7.0"]:
                 message = f"Pytorch version {version} is not aviliable with cuda version {cuda_version}! Please update your cuda or change pytorch version"
         elif cuda_version == "11.1" or cuda_version == "11.2":
             if version not in ["1.8.0","1.8.1","1.9.0","1.9.1","1.10.0","1.10.1"]:
                 message = f"Pytorch version {version} is not aviliable with cuda version {cuda_version}! Please update your cuda or change pytorch version"
         elif cuda_version == "11.3":
             if version not in ["1.7.1","1.7.0","1.8.0","1.8.1","1.9.0","1.9.1","1.10.0","1.10.1","1.12.1","1.12.0","1.11.0"]:
                 message = f"Pytorch version {version} is not aviliable with cuda version {cuda_version}! Please update your cuda or change pytorch version"
         elif cuda_version in  ["11.6","12.0","11.7","11.8"]:
             if version not in ["1.7.1","1.7.0","1.8.0","1.8.1","1.9.0","1.9.1","1.10.0","1.10.1","1.12.1","1.12.0","1.11.0"]:
                 message = f"Pytorch version {version} is not aviliable with cuda version {cuda_version}! Please update your cuda or change pytorch version"
         else :
             message = "Can't get cuda version! Cuda version risk for pytorch-gpu!"
         if message == "":
             message = f"Pytorch-gpu version {version} is available!"
    elif frame == 'tensorflow':
        if cuda_version == "8.0":
            if version in ["1.0.0","1.1.0","1.2.0","1.3.0","1.4.0"]:
                message = f"Tensorflow version {version} is available!"
        elif cuda_version in ["9.0","9.2"] :
            if version in ["1.6.0","1.7.0","1.8.0","1.9.0","1.10.0","1.11.0","1.12.0"]:
                message = f"Tensorflow version {version} is available!"
        elif cuda_version =="10.0":
            if version in ["1.13.0","1.14.0","1.15.0","2.0.0"]:
                message = f"Tensorflow version {version} is available!"
        elif cuda_version == "10.1":
            if version in ["2.1.0","2.2.0","2.3.0"]:
                message = f"Tensorflow version {version} is available!"
        elif cuda_version == "11.0":
            if version == "2.4.0":
                message = f"Tensorflow version {version} is available!"
        elif cuda_version == "11.2":
            if version  in ["2.5.0","2.6.0","2.7.0","2.8.0","2.9.0","2.10.0","2.11.0"]:
                message = f"Tensorflow version {version} is available!"
        else :
            message = "Can't get cuda version! Cuda version risk for tensorflow-gpu!"
        if message == "":
            message = f"Tensorflow version {version} is not aviliable with cuda version {cuda_version}! Please update your cuda or change tensorflow version"#"risk of tensorflow-gpu version!"
    elif frame == "paddlepaddle":
        if cuda_version == "9.0" or cuda_version == "10.0":
            if version in ["1.4.1","1.5.2","1.6.3","1.7.2","1.8.5","2.0.2"]:
                message = f"PaddlePaddle {version} is available!"
        elif cuda_version == "10.1" or cuda_version == "10.2":
            if version in ["2.0.2","2.1.3","2.2.2","2.3.2"]:
                message = f"PaddlePaddle {version} is available!"
        elif cuda_version == "11.0":
            if version == "2.0.2":
                message = f"PaddlePaddle {version} is available!" 
        elif cuda_version == "11.2":
            if version in ["2.1.3","2.2.2","2.3.2"]:
                message = f"PaddlePaddle {version} is available!"
        elif cuda_version == "11.6":
            if version == "2.3.2":
                message = f"PaddlePaddle {version} is available!"
        else :
            message = "Can't get cuda version! Cuda version risk for paddlepaddle!"
        if message == "":
            message = f"PaddlePaddle version {version} is not aviliable with cuda version {cuda_version}! Please update your cuda or change PaddlePaddle version"#"PaddlePaddle risk!"
    else:
        message = r"Not supported framework! (pytorch\tensorflow\paddlepaddle)"
    return message,cuda_version
    
    
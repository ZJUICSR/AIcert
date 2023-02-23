#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 17:01:30 2023

@author: ubuntu
"""
import json
import os
from gpu_test import framework_test

def run(params):
    json_path = os.path.join(params["out_path"],"frame_test.json")
    frame = params["frame"]
    version = params["version"]
    message,cuda_version = framework_test(frame,version)
    path = os.getcwd()
    json_path = os.path.join(path,json_path)
    if os.path.exists(json_path):
        with open(json_path,'r') as f:
            json_result = json.load(f)
    else:
        json_result={}
    frame_test = {frame:message}
    json_result.update(frame_test)
    
    with open(json_path,'w') as f_new:
        json.dump(json_result,f_new)
    
if __name__ == "__main__":
    params = {}
    params["out_path"] = "frame"
    params["frame"] = "pytorch"
    params["version"] = "1.7.1"
    run(params)
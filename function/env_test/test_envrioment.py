#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 16:02:03 2023

@author: ubuntu
"""
import torch
import os 
import platform 
print(os.environ['PATH'])
print("###%%Z***平")
print(os.getenv('PATH'))
print(platform.python_version())
print(torch.cuda.device_count())
print(torch.cuda.is_available())
print(torch.version.cuda)
print(pip.list)
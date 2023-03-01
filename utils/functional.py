#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2021/03/22, ZJUICSR'

import datetime
import os

import redis
from flask_caching import Cache

from control.task import Task


def init(app):
    app.cache = Cache()
    app.cache.init_app(app)

    target_folder = [app.config["LOG_FOLDER"], app.config["OUTPUT_FOLDER"]]
    for folder in target_folder:
        if not os.path.exists(folder):
            try:
                os.makedirs(folder)
            except Exception as e:
                print(f"Write `{folder}` error for read-only file system!")

    try:
        _redis = redis.Redis(host=app.config["REDIS_HOST"], port=app.config["REDIS_PORT"], db=app.config["REDIS_DB"],
                             password=app.config["REDIS_PASS"])
        app.redis = _redis
    except:
        raise ConnectionError("Can not connect to redis server!")


def decode(redis_dict):
    result = {}
    for k in redis_dict.keys():
        v = redis_dict[k]
        result[k.decode("utf-8")] = v.decode("utf-8")
    return result

def init_utils(app):
    app.task = Task(app)


def clean_ex_img():
    path = "web/static/images/result"
    file_list = os.listdir(path)
    for file in file_list:
        file_path = os.path.join(path,file)
        if os.path.exists(file_path):
            os.remove(file_path)

def save_ex_img(img, img_name):
    path = "web/static/images/result/"
    statci_path = "static/images/result/"

    curr_time = datetime.datetime.now()
    current_time = datetime.datetime.strftime(curr_time, '%Y-%m-%d-%H：%M：%S') #需要使用中文引号
    file_name = path + current_time + "_" + img_name
    f = statci_path + current_time + "_" + img_name

    img.save(file_name)
    return f

def save_adv_img(img, img_name):
    path = "web/static/images/adv_images/"
    statci_path = "static/images/adv_images/"

    curr_time = datetime.datetime.now()
    current_time = datetime.datetime.strftime(curr_time, '%Y-%m-%d-%H：%M：%S') #需要使用中文引号
    file_name = path + current_time + "_" + img_name
    f = statci_path + current_time + "_" + img_name

    img.save(file_name)
    return f
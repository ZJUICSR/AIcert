#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2021/04/13, ZJUICSR'


import time
from multiprocessing import Process
import hashlib
import utils.functional as F


class Task(object):
    def __init__(self, app):
        """
        Task manager class, control AI lifelong phase, including:
            data_collecting 数据收集
            data_cleaning 数据清洗
            data_transformation 数据标准化
            model_training 模型训练
            model_pruning 模型精修
            model_testing 模型测试
            model_deployment 模型部署
            platform_compatible 系统兼容
            platform_testing 系统测试

        :param app: Flask.app object
        """
        self.app = app
        self.conf = app.config

    def state(self, tid):
        """
        Get task state by tid, return some task information from DB.
        :param tid: Int, task id
        :param param: Json, some param
        :return: Json, task information
        """
        result = self.app.redis.hgetall(f"task:result:{tid}")
        return F.decode(result)

    def forward(self, param):
        """
        Run for a AI lifelong cycle, you can control a process pool in this function.
        :param param: parameter from website.
        :return: Json, task state
        """
        m = hashlib.md5()
        m.update(f"secai_{str(param)}_{time.time()}".encode('utf-8'))
        tid = m.hexdigest()

        task_result = {
            "tid": tid,
            "state": 0,
            "name": "some_name",
            "model": "",
            "loss": 9999999999999999999999999999,
            "accuracy": 0.0,
        }
        self.app.redis.hmset(f"task:result:{tid}", task_result)

        # 接口进入位置
        #p = Process(target=P.forward_process, args=(tid, self.app.config, param))
        #p.start()
        return task_result
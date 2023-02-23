# -*- coding: utf-8 -*-
import copy
class Taskparam:
    def __init__(self,info={},params={},res={}):#初始化    
        self.result = res
        self.params = params
        self.info = info
    # @staticmethod
    def change_res(self,result):
        self.result = copy.deepcopy(result)
        return self.result
    # @staticmethod
    def change_params(self, param):
        self.params = copy.deepcopy(param)
        return self.params
    # @staticmethod
    def change_info(self,info):
        self.info = copy.deepcopy(info)
        return self.info
    # @staticmethod
    def set_res_value(self,key,value):
        """ 定义一个全局变量 """
        self.result[key] = value
    # @staticmethod
    def set_info_value(self,key,value):
        """ 定义一个全局变量 """
        self.info[key] = value
    # @staticmethod
    def get_res_value(self,key,defValue=None):
        """ 获得一个全局变量,不存在则返回默认值 """
        try:
            return self.result[key]
        except KeyError:
            return "Not Found"
    # @staticmethod
    def get_info_value(self,key,defValue=None):
        """ 获得一个全局变量,不存在则返回默认值 """
        try:
            return self.info[key]
        except KeyError:
            return "Not Found"
    # @staticmethod
    def set_params_value(self,key,value):
        """ 定义一个全局变量 """
        self.params[key] = value

    # @staticmethod
    def get_params_value(self,key,defValue=None):
        """ 获得一个全局变量,不存在则返回默认值 """
        try:
            return self.params[key]
        except KeyError:
            return "Not Found"
    # @staticmethod
    def get_info(self):
        return self.info
    # @staticmethod
    def get_result(self):
        return self.result
    # @staticmethod
    def get_params(self):
        return self.params
    # @staticmethod
    def del_result(self):
        self.result.clear()
    # @staticmethod
    def del_params(self):
        self.params.clear()
    # @staticmethod
    def del_info(self):
        self.info.clear()
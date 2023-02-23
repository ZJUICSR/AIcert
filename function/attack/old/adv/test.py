#
# import os
# import json
#
# class JarProjectUtil:
#     @staticmethod
#     def project_root_path(project_name=None, print_log=True):
#         """
#         获取当前项目根路径
#         :param project_name: 项目名称
#                                 1、可在调用时指定
#                                 2、[推荐]也可在此方法中直接指定 将'XmindUitl-master'替换为当前项目名称即可（调用时即可直接调用 不用给参数）
#         :param print_log: 是否打印日志信息
#         :return: 指定项目的根路径
#         """
#         p_name = 'cycle-evaluation' if project_name is None else project_name
#         project_path = os.path.abspath(os.path.dirname(__file__))
#         # Windows
#         if project_path.find('\\') != -1: separator = '\\'
#         # Mac、Linux、Unix
#         if project_path.find('/') != -1: separator = '/'
#
#         root_path = project_path[:project_path.find(f'{p_name}{separator}') + len(f'{p_name}{separator}')]
#         if print_log: print(f'当前项目名称：{p_name}\r\n当前项目根路径：{root_path}')
#         return root_path
#
#
#
# root_path = JarProjectUtil.project_root_path()
# root_path_json = root_path+"adv_result2.json"
# print(root_path_json)
# result = {}
# result["a"] = "1"
# with open(root_path_json,"w") as fobj:
#     json.dump(result,fobj)

import attack_test
methods = "['FGSM','RFGSM','FFGSM','MIFGSM','PGD','BIM']"
dataset = "CIFAR10"
tid = "20220506_1033"
result = attack_test.run_attack(methods,dataset,tid)
if result:
    print("hello")
    print(result)
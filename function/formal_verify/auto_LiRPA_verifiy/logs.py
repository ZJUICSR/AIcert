# -*- coding: utf-8 -*-
# @Time    : 2022/2/25 9:46
# @File    : logs.py
import time


class LiRPALogs(object):
    def __init__(self):
        self.log_infos = dict()

    def write_logs(self, info: str, task_id='default', task_finish=False):
        if task_id not in self.log_infos:
            self.log_infos[task_id] = list()
        t = time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time()))
        self.log_infos[task_id].append({'logs': f'{t}: {info}',
                                        'task_finish': task_finish})

    def upload_logs(self, task_id='default'):
        if task_id not in self.log_infos:
            return list()

        cur_logs = self.log_infos[task_id].copy()
        self.log_infos[task_id].clear()
        return cur_logs

    def __str__(self):
        return f'{self.log_infos}'



if __name__ == '__main__':
    pass


# -*- coding: utf-8 -*-
import os
import json


def read_json_file(file_name):
    result = dict()
    if not os.path.exists(file_name):
        print(f'read json file {file_name} not exist!')
        return result
    with open(file_name, 'r', encoding='utf-8') as f:
        result = json.load(f)
    return result


def save_as_json_file(info, file_name):
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=4, ensure_ascii=False)
    return


if __name__ == '__main__':
    pass


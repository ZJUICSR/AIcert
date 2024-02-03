import json
import os


def read_json(file_name=''):
    if not os.path.exists(file_name):
        return dict()
    with open(file_name, 'r', encoding='utf-8') as f:
        result = json.load(f)
    return result


def write_json(info, file_name=''):
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=4, ensure_ascii=False)
    return


if __name__ == '__main__':
    pass


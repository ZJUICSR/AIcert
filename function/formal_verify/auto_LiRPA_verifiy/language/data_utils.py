import random
import json
import os


def load_data_sst():
    data = []
    for split in ['train_all_nodes', 'train', 'dev', 'test']:
        name = os.path.join(os.path.dirname(__file__), 'data/sst')
        with open(f'{name}{os.sep}{split}.json') as file:
            data.append(json.loads(file.read()))
    return data


def clean_data(data):
    return [example for example in data if example['candidates'] is not None]


def get_batches(data, batch_size):
    batches = []
    random.shuffle(data)
    for i in range((len(data) + batch_size - 1) // batch_size):
        batches.append(data[i * batch_size: (i + 1) * batch_size])
    return batches


def get_sst_data(ver_num=100, batch_size=10):
    random.seed(123)
    data_train_all_nodes, data_train, data_dev, data_test = load_data_sst()
    ver_num = ver_num if ver_num < len(data_train_all_nodes) else len(data_train_all_nodes)
    ver_nodes = random.sample(data_train_all_nodes, ver_num)
    batches = get_batches(ver_nodes, batch_size)
    return batches, data_train


if __name__ == '__main__':
    a, b = get_sst_data()
    print(len(a))
    print(a[0])


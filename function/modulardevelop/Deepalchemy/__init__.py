import os
import json
import sys

sys.path.append(os.path.dirname(__file__))


from deepalchemy import run


if __name__ == '__main__':
    test_param_1 = {
        'gpu': 1,
        'modelname': 'resnet',
        'dataset': 'cifar10',
        'epochs': 1,
        'init': 'normal',
        'iternum': 4
    }
    run(test_param_1)
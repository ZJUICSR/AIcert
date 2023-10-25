import os
from os.path import join, dirname

RGCN_ROOT = dirname(__file__)
CONFIG_ROOT = join(RGCN_ROOT, 'config')
RESULT_ROOT = join(RGCN_ROOT, 'result')


def get_datasets():
    files = filter(lambda name: name.endswith('.npz'),
                   os.listdir(join(RGCN_ROOT, 'data')))
    return list(map(lambda name: name[:-4], files))

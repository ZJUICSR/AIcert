import os

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():
    
    config = default_config()
    config.model_paths=['/data/home/Wenjie/Project/AIcert/model/ckpt/vicuna-7b-v1.1']
    config.tokenizer_paths=['/data/home/Wenjie/Project/AIcert/model/ckpt/vicuna-7b-v1.1']
    return config
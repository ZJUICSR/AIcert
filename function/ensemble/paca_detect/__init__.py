from .paca import paca_detect
import os.path as osp
import time
import torch
import os


ROOT = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))), 'web', 'static')


def upload_file_valid(upload_info):
    if 'train' not in upload_info or 'test' not in upload_info:
        return False
    if len(upload_info['train']) != 2 or len(upload_info['test']) != 2:
        return False
    if not all(isinstance(d, torch.utils.data.DataLoader) for d in upload_info['train']):
        return False
    if not all(isinstance(d, torch.utils.data.DataLoader) for d in upload_info['test']):
        return False
    return True


def get_upload_info(params, log_func=None):
    def _log(info, func=log_func):
        if func is not None:
            func(info)

    if "upload" not in params.keys() or "adv_loader_path" not in params["upload"].keys():
        return None

    upload_path = osp.join(ROOT + "/output/cache", params["upload"]["adv_loader_path"][0])
    if not os.path.exists(upload_path):
        return None

    upload_info = torch.load(upload_path)
    if not upload_file_valid(upload_info=upload_info):
        _log('[模型测试阶段] 自动化攻击监测PACA模型训练，上传文件格式校验失败')
        return None
    _log('[模型测试阶段] 自动化攻击监测PACA模型训练，上传文件格式校验成功')
    return upload_info


def run(model, test_loader, adv_dataloader, params, param_hash=str(time.time()), log_func=None, channel=3):
    save_path = params["out_path"]
    if not os.path.exists(save_path):
        os.mkdir(path=save_path)

    attack_info = []
    for key, value in adv_dataloader.items():
        _ = {
            "method": key,
            "dataloader": value
        }
        attack_info.append(_)

    upload_info = get_upload_info(params=params, log_func=log_func)

    result = paca_detect(
        ori_model=model,
        ori_dataloader=test_loader,
        attack_info=attack_info,
        upload_info=upload_info,
        param_hash=param_hash,
        device=params["device"],
        save_path=save_path,
        log_func=log_func,
        channel=channel
    )
    print("__init__ paca:",result)
    return result

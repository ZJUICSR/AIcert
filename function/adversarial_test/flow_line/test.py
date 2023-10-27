import sys
from .flow_line import FlowLine
import os
import json
import warnings
from function.adversarial_test.attack.gen_adv import attacks_dict


warnings.filterwarnings("ignore")


RESULT_PATH = os.path.join(os.path.dirname(__file__), 'results')
if not os.path.exists(RESULT_PATH):
    os.mkdir(RESULT_PATH)


def save_json_info(filename, info):
    result = dict()
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            result = json.load(f)
    result.update(info)

    with open(filename, 'w', encoding='utf-8') as file_obj:
        json.dump(result, file_obj)
    return


def json2dict(file_path) -> dict:
    with open(file_path, 'r', encoding='utf-8') as file:
        result = json.load(file)
    return result


def load_cache(file_name, save_file):
    result = json2dict(file_name)
    save_json_info(save_file, result)
    return result


def run(model,
         dataset='mnist',
         batch_size=128,
         eps=4,
         attack_methods=['fgsm', 'mifgsm', 'pgd'],
         param_hash='',
         save_path='',
         log_func=None,
         device='cuda',
         ori_loader=None, 
         adv_loader=None):

    global RESULT_PATH

    cache_file = 'cache.json'
    result_file = 'flow_line_results.json'
    cache_path = os.path.join(RESULT_PATH, param_hash)
    if os.path.exists(os.path.join(cache_path, cache_file)):  # 直接读取缓存结果
        print('param_hash, 读取缓存结果', param_hash)
        result = load_cache(os.path.join(cache_path, cache_file),
                   os.path.join(save_path, result_file))
        print(result)
        return result

    result = dict()

    if log_func is not None:
        log_func(f'[模型测试阶段] 相关攻击算法含：{str(attack_methods)}')
    if not adv_loader:
        support_methods = attacks_dict(model=model, eps=eps).keys()

        methods = {method for method in attack_methods if method.lower() in support_methods}

        if log_func is not None:
            log_func('[模型测试阶段] 选择算法：{:s}进行测试'.format(str(methods)))

        flow_line = FlowLine(model=model, dataset=dataset, eps=eps, attack_methods=methods,
                            device=device, batch_size=batch_size, log_func=log_func)
    else:
        flow_line = FlowLine(model=model, attack_methods=attack_methods,
                            device=device, log_func=log_func, ori_loader=ori_loader, adv_loader=adv_loader)
    result['result'] = flow_line.run()

    save_json_info(os.path.join(save_path, result_file), result)

    if log_func is not None:
        log_func("[模型测试阶段] 保存存到：{:s}， 流程结束".format(os.path.join(save_path, result_file)))

    # 保存缓存
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)
    save_json_info(os.path.join(cache_path, cache_file), result)
    return result



if __name__ == '__main__':
    # from GroupDefense.datasets.mnist import mnist_dataloader
    # from GroupDefense.models.load_model import load_model

    # methods = ['fgsm', 'bim', 'rfgsm', 'cw', 'pgd', 'tpgd', 'mi-fgsm', 'autopgd', 'square', 'deepfool', 'difgsm']
    # method = ['rfgsm', 'mifgsm']
    device = 'cuda'
    eps = 16
    model = load_model()
    model.to(device)
    _, dataloader = mnist_dataloader()

    defend_info = run(model=model,
                     dataset='mnist',
                     batch_size=128,
                     eps=16,
                     attack_methods=['fgsm', 'mifgsm', 'pgd', 'tpgd'],
                     param_hash='123455',
                     save_path='results',
                     log_func=None,
                     device='cuda')
    print(defend_info)



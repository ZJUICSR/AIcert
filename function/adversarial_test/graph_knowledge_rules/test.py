from .graph_info import AttackKnowledge
from .attack import GraphRulesAttack, SUPPORT_METHODS
import os
import json
import warnings
import os.path as osp


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
    return


def test(model,
         dataloader,
         eps=0.01,
         n_classes=10,
         attack_mode='white_box',
         attack_type='evasion_attack',
         data_type='image',
         defend_algorithm='Adversarial-Training',
         device='cpu',
         acc_value={},
         param_hash='',
         min_pixel_value=0,
         max_pixel_value=255,
         save_path='',
         log_func=None):

    global RESULT_PATH

    assert attack_mode in ['white_box', 'black_box']
    assert attack_type in ['poison_attack', 'evasion_attack']
    assert data_type in ['image', 'text', 'graph']

    key_map = {'white_box': '白盒',
               'black_box': '黑盒',
               'evasion_attack': '逃逸攻击',
               'poison_attack': '毒化攻击',
               'image': '图片',
               'text': '文本',
               'graph': '图'}
    attack_mode = key_map[attack_mode]
    attack_type = key_map[attack_type]
    data_type = key_map[data_type]

    cache_file = 'cache.json'
    result_file = 'result.json'
    cache_path = os.path.join(RESULT_PATH, param_hash)
    if os.path.exists(os.path.join(cache_path, cache_file)):  # 直接读取缓存结果
        print('param_hash, 读取缓存结果', param_hash)
        load_cache(os.path.join(cache_path, cache_file),
                   os.path.join(save_path, result_file))
        return

    result = dict()
    knowledge = AttackKnowledge()
    recmmend_method = knowledge.recom(attack_mode=attack_mode,
                                      attack_type=attack_type,
                                      data_type=data_type,
                                      defend_algorithm=defend_algorithm)
    if log_func is not None:
        log_func('[模型测试阶段] 相关攻击算法含：{:s}'.format(str(recmmend_method)))
    result['recom_algorithm'] = list(recmmend_method)

    attack_methods = {method for method in recmmend_method if method.lower() in SUPPORT_METHODS}

    # if log_func is not None:
    #     log_func('[模型测试阶段] 选择算法：{:s}进行测试'.format(str(attack_methods)))

    get_acc = {method: acc_value[method] for method in set(recmmend_method) & set(acc_value.keys())}

    attack_methods = list(attack_methods - set(acc_value.keys()))

    if log_func is not None:
        log_func('[模型测试阶段] 选择算法：{:s}进行测试'.format(str(attack_methods)))

    attack = GraphRulesAttack(model=model,
                              dataloarder=dataloader,
                              n_class=n_classes,
                              device=device,
                              eps=eps,
                              min_pixel_value=min_pixel_value,
                              max_pixel_value=max_pixel_value,
                              save_path=save_path,
                              log_func=log_func)
    result['acc'] = attack.run(attack_methods)
    result['acc'].update(get_acc)

    save_json_info(os.path.join(save_path, result_file), result)

    if log_func is not None:
        log_func("[模型测试阶段] 保存存到：{:s}， 流程结束".format(os.path.join(save_path, result_file)))

    # 保存缓存
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)
    save_json_info(os.path.join(cache_path, cache_file), result)
    return result


import torch
def run(model, test_loader, num_classes, test_acc, params, param_hash="", log_func=None):

    save_path = params["out_path"]
    result = test(model, dataloader=test_loader, eps=0.1,
         n_classes=num_classes,
         attack_mode=params["attack_mode"],
         attack_type=params["attack_type"],
         data_type=params["data_type"],
         defend_algorithm=params["defend_algorithm"],
         device=torch.device(params["device"]),
         acc_value=test_acc,
         save_path=save_path,
         param_hash=param_hash,
         log_func=log_func
    )
    return result


if __name__ == '__main__':
    from GroupDefense.datasets.mnist import mnist_dataloader
    from GroupDefense.models.load_model import load_model

    device = 'cuda'
    eps = 16
    model = load_model()
    model.to(device)
    _, dataloader = mnist_dataloader()
    params = {'attack_mode': 'white_box',
              'attack_type': 'evasion_attack',
              'data_type': 'image',
              'defend_algorithm': 'Adversarial-Training',
              'device': 'cuda',
              'out_path': 'results'}
    if not os.path.exists(params['out_path']):
        os.mkdir(params['out_path'])

    defend_info = run(model, dataloader, num_classes=10, test_acc={}, param_hash='123', params=params)
    print(defend_info)


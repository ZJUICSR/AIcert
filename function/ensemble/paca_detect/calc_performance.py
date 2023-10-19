# -*- coding: utf-8 -*-
from glob import glob
import os
from paca_detect.resnet_cbam import TwoStreamCBAM
from paca_detect.net import TwoStraeamSrncovet
from paca_detect.net import TwoStraeamSrncovet
import torch
from paca_detect.gen_adv import get_adv_dataloader
from paca_detect.eval_model import ModelScore
from paca_detect.resnet_model import get_resnet50_model, get_cifar_resnet_model
import json
from tqdm import tqdm
import socket


SERVICE_NAME = socket.gethostname()
if SERVICE_NAME == 'ubuntu02':
    BASE_DIR = '/opt/data/user/gss/code/data'
else:
    BASE_DIR = '/data/user/gss/code/results'

RESULTS_FOLDER = f'{BASE_DIR}/adv_detect/detect_models/cifar10_0731'
RESULTS_FILE = os.path.join(RESULTS_FOLDER, 'cifar10_resnet18_cbma_smooth_result_difgsm.json')


def main():
    file_name = glob(os.path.join(RESULTS_FOLDER, '*paca*.pt'))
    # print(file_name)
    results = dict()
    for pt_file in tqdm(file_name, desc='Calc Performance', ncols=100):
        pt_file = pt_file.split(os.sep)[-1].split("_")
        attack_method = pt_file[2]
        eps = int(pt_file[-2])

        # model = TwoStreamCBAM()
        model = TwoStraeamSrncovet()

        state_dict = torch.load(os.path.join(RESULTS_FOLDER, f'cifar10_resnet18_{attack_method}_paca_eps_{eps}_255.pt'))
        model.load_state_dict(state_dict['model'])
        train_dataloader, test_dataloader = get_adv_dataloader(method=attack_method, model='resnet18', eps=eps, batch_size=256)
        victim_model = get_cifar_resnet_model(model_name='resnet18', device='cuda', pretrain=True)
        cal_score = ModelScore(model=model, dataloader=train_dataloader, smooth=False, smooth_value=16, victim_model=victim_model)
        if attack_method not in results:
            results[attack_method] = dict()
        results[attack_method].update({f'{eps}':
                                            {'acc': cal_score.cal_acc(),
                                             'pre': cal_score.cal_precision(),
                                             'recall': cal_score.cal_recall(),
                                             'f1': cal_score.cal_f1(),
                                             'auc': cal_score.cal_auc()}})

        with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)


def calc_cw_results(method='paca', device='cuda', eps=2):
    assert method in ['paca', 'cbma_smooth']
    model = TwoStraeamSrncovet() if method == 'paca' else TwoStreamCBAM()
    state_dict = torch.load(os.path.join(RESULTS_FOLDER, f'cifar10_resnet18_cw_{method}_eps_{eps}_255.pt'))
    model.load_state_dict(state_dict['model'])
    train_dataloader, test_dataloader = get_adv_dataloader(method='cw', model='resnet18', eps=eps, batch_size=256)
    victim_model = get_cifar_resnet_model(model_name='resnet18', device=device, pretrain=True)
    cal_score = ModelScore(model=model, dataloader=train_dataloader, smooth=True, smooth_value=16, victim_model=victim_model)
    results = {'acc': cal_score.cal_acc(),
               'pre': cal_score.cal_precision(),
               'recall': cal_score.cal_recall(),
               'f1': cal_score.cal_f1(),
               'auc': cal_score.cal_auc()}
    print(results)


if __name__ == '__main__':
    main()
    # calc_cw_results(method='cbma_smooth')

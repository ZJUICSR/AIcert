import sys
sys.path.append("..")
sys.path.append("../..")
from function.ensemble.paca_detect.train import train_detect_model
from function.ensemble.CAFD.train_or_test_denoiser import cafd, denoising
from function.ensemble.attack.gen_adv import get_integrate_dataloader, calc_acc
from torch.utils.data import DataLoader, Dataset
from function.ensemble.ensemble_defense.at import AdversarialTraining
from copy import deepcopy


class IntegrateDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.label = y

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.x[idx], self.label[idx]


def integrated_defense(model, dataloader, attack_methods, adv_param, train_epoch, in_channel=1, data_size=28, device='cuda', dataset='mnist'):

    ori_acc = calc_acc(model, dataloader, device=device, desc='计算模型原有准确率')
    
    results = dict()
    results['ori_acc'] = ori_acc

    attack_info = dict()
    print(attack_methods)
    for method in attack_methods:
        results[method] = dict()
        # attackparam = adv_param
        # attack_info[method] = attack(model=model, dataloader=dataloader, method=method, eps=eps, device=device)
        attack_data_info = get_integrate_dataloader(method=method, model=model, dataloader=dataloader, attackparam=adv_param[method],
                                                    device=device, batch_size=128, dataset=dataset)
        results[method]['attack_acc'] = attack_data_info["acc"]
        detect_info = train_detect_model(victim_model=deepcopy(model),
                                         dataloader=attack_data_info['paca'][0],
                                         in_channel=in_channel,
                                         method=method,
                                         attackparam=adv_param[method],
                                         epoches=30, device=device, batch_size=128, gen_adv=False)
        print(f'{method} detect_info={detect_info["detect_acc"]}')
        results[method]['defend_rate'] = detect_info["detect_acc"]
        at_model = AdversarialTraining(method=method,
                                       attackparam=adv_param[method],
                                       model=deepcopy(model),
                                       dataloader=dataloader,
                                       device=device,
                                       batch_size=128,
                                       train_epoch=train_epoch,
                                       at_epoch=train_epoch,
                                       dataset=dataset).train()
        target_model = deepcopy(at_model).to(device)
        cafd_info = cafd(target_model=target_model.to(device), dataloader=dataloader, method=method, adv_param=adv_param[method],
                         channel=in_channel, data_size=data_size, weight_adv=5e-3,
                         weight_act=1e3, weight_weight=1e-3, lr=0.001, itr=train_epoch,
                         batch_size=128, weight_decay=2e-4, print_freq=10, save_freq=2, device=device, gen_adv=True, dataset=dataset)
        print(f'detect_info={cafd_info["prec"]}')
        results[method]['defend_attack_acc'] = cafd_info["prec"]
        results[method]['defend_model'] = {'model': at_model, 'denoiser': cafd_info['denoiser']}
    return results


if __name__ == '__main__':
    from function.ensemble.datasets.mnist import mnist_dataloader
    from function.ensemble.models.load_model import load_model

    # methods = ['fgsm', 'bim', 'rfgsm', 'cw', 'pgd', 'tpgd', 'mifgsm', 'autopgd', 'square', 'deepfool', 'difgsm']
    methods = ['FGSM']
    method = ["difgsm"]
    device = 'cuda:0'
    eps = 16/255
    model = load_model()
    model.to(device)
    _, dataloader = mnist_dataloader()
    # method = ['difgsm']

    defend_info = integrated_defense(model, dataloader, methods, eps, 10, in_channel=1, data_size=28, device='cuda')
    print(defend_info)

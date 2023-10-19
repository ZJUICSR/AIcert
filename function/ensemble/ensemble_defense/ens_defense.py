import sys
sys.path.append("..")
sys.path.append("../..")
from GroupDefense.ensemble_defense.at import AdversarialTraining
from copy import deepcopy
from GroupDefense.ensemble_defense import EnsembleModel
from GroupDefense.attack.gen_adv import get_attack, get_adv_data

def ens_defense(model, dataloader, methods=[], eps=1, train_epoch=10, at_epoch=5, device='cuda'):
    model_list = list()
    attack_results = dict()
    for method in methods:
        print(f'鲁棒性训练：{method}')
        at_model = AdversarialTraining(method=method,
                                       eps=eps,
                                       model=model,
                                       dataloader=dataloader,
                                       device=device,
                                       batch_size=128,
                                       train_epoch=1,
                                       at_epoch=1)
        model_list.append(at_model.train())

    ens_model = EnsembleModel(model_list=model_list)
    for method in methods:
        print(f'对抗攻击：{method}')
        attack = get_attack(method=method, model=model, eps=eps)
        attack_info = get_adv_data(model=ens_model, attack=attack, dataloader=dataloader, device=device,
                                   desc=f'生成{method}对抗样本，eps={eps}')
        attack_results[method] = attack_info[-1]
    return attack_results


if __name__ == '__main__':
    from GroupDefense.datasets.mnist import mnist_dataloader
    from GroupDefense.models.load_model import load_model
    # methods = ['fgsm', 'mifgsm']
    methods = ['mifgsm', 'fgsm', 'bim', 'rfgsm', 'cw', 'pgd', 'tpgd', 'autopgd', 'square', 'difgsm']
    device = 'cuda'
    eps = 1
    model = load_model()
    model.to(device)
    _, dataloader = mnist_dataloader()
    results = ens_defense(model, dataloader, methods=methods, eps=1, device='cuda')
    print(results)
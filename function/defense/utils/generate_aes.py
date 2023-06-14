import torch
import numpy as np
import torch.nn as nn
from function.defense.utils.get_clean_data import get_clean_loader
from advertorch.attacks import GradientSignAttack, MomentumIterativeAttack, SparseL1DescentAttack, CarliniWagnerL2Attack, LinfPGDAttack
def generate_adv_examples(model, adv_method, adv_dataset, adv_nums, device, normalize=False):
    clean_loader, num_classes = get_clean_loader(adv_dataset, normalize)
    if adv_dataset == 'CIFAR10':
        eps = 0.031
        nb_iter = 20
        eps_iter = 0.003
    elif adv_dataset == 'MNIST':
        eps = 0.3
        nb_iter = 40
        eps_iter = 0.01
    if adv_method == 'PGD':
        adversary = LinfPGDAttack(
            model, loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"), eps=eps,
            nb_iter=nb_iter, eps_iter=eps_iter, rand_init=True, clip_min=0.0, clip_max=1.0,
            targeted=False)
    elif adv_method == 'FGSM':
        adversary = GradientSignAttack(
            model, loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"), eps=eps,
            clip_min=0.0, clip_max=1.0, targeted=False)
    elif adv_method == 'CW':
        adversary = CarliniWagnerL2Attack(
            model, num_classes, confidence=0, targeted=False, 
            learning_rate=0.01, binary_search_steps=9, max_iterations=10000, 
            abort_early=True, initial_const=0.001, clip_min=0.0, 
            clip_max=1.0, loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"))
    elif adv_method == 'Deepfool':
        adversary = SparseL1DescentAttack(model, loss_fn=None, eps=eps, 
                    nb_iter=nb_iter, eps_iter=eps_iter, rand_init=False, 
                    clip_min=0.0, clip_max=1.0, l1_sparsity=0.95, targeted=False)
    elif adv_method == 'BIM':
        adversary = MomentumIterativeAttack(model, loss_fn=None, eps=eps, 
                    nb_iter=nb_iter, decay_factor=1.0, eps_iter=eps_iter, 
                    clip_min=0.0, clip_max=1.0, targeted=False, 
                    ord=np.inf)
    elif adv_method == 'BPDA':
        adversary = LinfPGDAttack(
            model, loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"), eps=eps,
            nb_iter=nb_iter, eps_iter=eps_iter, rand_init=True, clip_min=0.0, clip_max=1.0,
            targeted=False)
    l = [x for (x, y) in clean_loader]
    clean_examples = torch.cat(l, 0).to(device)[:adv_nums]
    l = [y for (x, y) in clean_loader]
    true_labels = torch.cat(l, 0).to(device)[:adv_nums]
    flag = True
    i = 0
    for cln_data, true_label in clean_loader:
        cln_data, true_label = cln_data.to(device), true_label.to(device)
        adv_data = adversary.perturb(cln_data, true_label)
        adv_logit = model(adv_data)
        adv_label = torch.argmax(adv_logit, axis=1)
        if flag:
            adv_examples = adv_data
            adv_labels = adv_label
            flag = False
        else:
            adv_examples = torch.cat((adv_examples, adv_data))
            adv_labels = torch.cat((adv_labels, adv_label))
        i += 1
        if i * 128 > adv_nums:
            break
    # adv_examples = adversary.perturb(clean_examples, true_labels)
    adv_examples = adv_examples[:adv_nums]
    # np.save('/mnt/data2/yxl/AI-platform/adv_' + adv_dataset + '_' + adv_method + '.npy', adv_examples.cpu().numpy())
    return adv_examples, clean_examples, true_labels
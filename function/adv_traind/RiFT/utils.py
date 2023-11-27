# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import logging
from torchattacks import *
import torch
import torch.nn as nn
import torch.nn.init as init
from tqdm import tqdm


def interpolation(init_sd, ft_sd, model, dataloader, criterion, save_dir,
                  attack_methods=['fgsm', 'mifgsm', 'pgd'], device='cuda', eval_robustness_func=None):
    # alphas = np.arange(0, 1.1, 0.1)
    alpha = 0.6

    records = dict()

    model_dict = {}
    for name, _ in init_sd.items():
        model_dict[name] = alpha * ft_sd[name] + (1 - alpha) * init_sd[name]

    torch.save(model_dict, save_dir + "/finetune_{:.3f}_params.pth".format(alpha))

    model.load_state_dict(model_dict)
    test_loss, test_acc = evaluate(model, dataloader, criterion)
    test_robust_acc = eval_robustness_func(model, dataloader=dataloader, attack_methods=attack_methods)

    print(f"==> Alpha: {alpha:.2f}, test acc: {test_acc:.2f}%, test robust acc: {test_robust_acc}")
    records['test_acc'] = test_acc
    records['robust_acc'] = test_robust_acc

    return records


def load_sd(model_path):
    sd = torch.load(model_path)
    if "net" in sd.keys():
        sd = sd["net"]
    elif "state_dict" in sd.keys():
        sd = sd["state_dict"]
    elif "model" in sd.keys():
        sd = sd["model"]

    return sd


def attacks_dict(model, eps, method):
    attacks = {
        'fgsm': FGSM(model=model, eps=eps),
        'bim': BIM(model=model, eps=eps, alpha=2 / 255, steps=10),
        'rfgsm': RFGSM(model, eps=eps, alpha=2 / 255, steps=10),
        'ffgsm': FFGSM(model=model, eps=eps, alpha=2 / 255),
        'mifgsm': MIFGSM(model, eps=eps, alpha=2 / 255, steps=10, decay=0.1),
        'difgsm': DIFGSM(model=model, eps=eps, alpha=2 / 255, steps=10, diversity_prob=0.5, resize_rate=0.9),
        'cw': CW(model=model, c=1, lr=0.01, steps=20, kappa=0),
        'upgd': UPGD(model=model, eps=eps, alpha=2 / 255, steps=10),
        'pgd': PGD(model=model, eps=eps, alpha=2 / 225, steps=10, random_start=True),
        'tpgd': TPGD(model=model, eps=eps, alpha=2 / 255, steps=10),
        'autopgd': APGD(model=model, eps=eps, steps=10, eot_iter=1, n_restarts=1, loss='ce'),
        'deepfool': DeepFool(model=model, steps=10),
    }
    return attacks[method]


def evaluate_cifar_robustness(model, dataloader, attack_methods=['fgsm', 'mifgsm', 'pgd'], device='cuda'):
    results = dict()
    for method in attack_methods:
        atk_model = attacks_dict(model=model, eps=8/255, method=method)
        model.eval()
        total = 0
        correct = 0
        for images, labels in tqdm(dataloader, ncols=100, desc=f'craft {method} adv'):

            images = images.to(device)
            labels = labels.to(device)

            adv_images = atk_model(images, labels)
            outputs = model(adv_images)

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        results[method] = correct / total * 100
    return results


def evaluate(model, dataloader, criterion, device='cuda'):
    model.eval()
    loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    return loss, acc


class Normalize(nn.Module):
    def __init__(self, mean, std) :
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean).to("cuda"))
        self.register_buffer('std', torch.Tensor(std).to("cuda"))
        
    def forward(self, input):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std


def create_logger(log_path):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def get_all_trained_model_params(path):
	trained_params_list = []

	for (root, dirs, files) in os.walk(path):
		# print(root, dirs, files)
		if len(files) > 0:
			for my_file in files:
				if my_file.find(".pth") != -1:
					trained_params_list.append(root+"/"+my_file)
	# print(trained_params_list)
	# exit()
	return trained_params_list
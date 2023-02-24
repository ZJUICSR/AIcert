from auto_LiRPA.auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.auto_LiRPA.perturbations import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from art.attacks.evasion import FastGradientMethod, DeepFool, AutoAttack
from art.estimators.classification import PyTorchClassifier
import numpy as np
import torch.nn as nn
import torch.optim as optim
import os
import shutil
from torchvision import transforms


plt.switch_backend('Agg')
lirpa_adv_rates = []



class LanguageAutoLirpaVerify(object):
    def __init__(self, verify_model, data_batches, n_class, up_eps=0.1, down_eps=0.01, steps=5, device='cpu',
                 output_path=None, log_func=None, task_id='default'):
        super(LanguageAutoLirpaVerify, self).__init__()
        self.verify_model = verify_model
        self.data_batches = data_batches
        self.n_class = n_class
        self.up_eps = up_eps
        self.down_eps = down_eps
        self.steps = steps
        self.device = device
        self.output_path = output_path
        self.lirpa_res = list()
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
            os.mkdir(self.output_path)
        if output_path is not None and not os.path.exists(output_path):
            os.makedirs(output_path)
        self.verify_model = self.verify_model.to(self.device)
        self.eps_list = list()
        self.original_pic_dir = None
        self.adv_pic_dir = None
        self.original_pic_dir = os.path.join(self.output_path, 'adv') if output_path is not None else None
        self.adv_pic_dir = os.path.join(self.output_path, 'adv') if output_path is not None else None
        self.fgsm_dir = os.path.join(self.output_path, 'fgsm') if output_path is not None else None
        self.auto_LiRPA_dir = os.path.join(self.output_path, 'auto_LiRPA') if output_path is not None else None
        self.bound_model = None
        self.log_func = log_func
        self.task_id = task_id

    def write_logs(self, info: str, task_finish=False):
        if self.log_func is not None:
            self.log_func(info=info,
                          task_id=self.task_id,
                          task_finish=task_finish)
        print(info)

    def computer_eps_range(self):
        step_size = round((self.up_eps - self.down_eps) / self.steps, 3)
        eps = self.down_eps
        bits = len(str(step_size).split('.')[1])
        eps_list = list()
        while eps <= self.up_eps:
            eps_list.append(eps)
            eps += step_size
            eps = round(eps, bits)
        return eps_list

    @staticmethod
    def computer_noise(lbs, ubs, labels):
        cnt = 0
        for i, (lb, ub) in enumerate(zip(lbs, ubs)):
            label = labels[i]
            ub[label] = -10000000
            max_u = ub[np.array(ub).argmax()]
            if max_u > lb[label]:
                cnt += 1
        return round((cnt / len(lbs)), 2)

    def boundary(self, budget, eps):
        lb_list = list()
        ub_list = list()
        label_list = list()

        ptb = PerturbationSynonym(budget=budget, eps=eps)
        ptb.model = self.verify_model

        for i, data_batch in tqdm(enumerate(self.data_batches)):
            embeddings_unbounded, mask, tokens, _ = self.verify_model.get_input(data_batch)
            if self.bound_model is None:
                bound_opts = {'relu': 'zero-lb', 'exp': 'no-max-input', 'fixed_reducemax_index': True}
                self.bound_model = BoundedModule(self.verify_model.model_from_embeddings.to(self.device),
                                                 (embeddings_unbounded, mask),
                                                 bound_opts=bound_opts)

            input_data = BoundedTensor(embeddings_unbounded, ptb)
            prediction = self.bound_model((input_data, mask))
            with torch.no_grad():
                lb, ub = self.bound_model.compute_bounds(x=(input_data, mask), aux=(tokens, data_batch), method="IBP+backward")
            lb_list.append(lb.tolist())
            ub_list.append(ub.tolist())
            label_list.append(prediction.argmax(dim=1).tolist())
        lb_list = np.array(lb_list).reshape([-1, self.n_class]).tolist()
        ub_list = np.array(ub_list).reshape([-1, self.n_class]).tolist()
        label_list = np.array(label_list).reshape(-1).tolist()
        return lb_list, ub_list, label_list

    def computer_adversarial_exist_rate_by_eps(self, eps_list: list):
        adv_rate_list = list()
        for eps in eps_list:
            self.write_logs(f'扰动大小为：{eps}，开始测试')
            lb, ub, label = self.boundary(10, eps)
            noise = self.computer_noise(lb, ub, label)
            adv_rate_list.append(noise)
        return adv_rate_list

    def to_percent(self, temp, position):
        return '%1.0f' % (100 * temp) + '%'

    def draw_rate_pic(self, name_list, data, label, save_path=None):
        x = list(range(len(data)))
        plt.plot(name_list, data, marker='o', mec='r', mfc='w', label=label)
        plt.xticks(x, name_list)
        plt.legend()
        plt.ylim(0, 1.15)
        plt.gca().yaxis.set_major_formatter(FuncFormatter(self.to_percent))
        plt.grid(axis='y')
        plt.xticks(fontproperties='DejaVu Sans', size=15)
        plt.yticks(fontproperties='DejaVu Sans', size=15)
        if save_path:
            file_folder = os.path.join(self.output_path, save_path)
            if not os.path.exists(file_folder):
                os.makedirs(file_folder)
            path = os.path.join(file_folder, f'{label}.png')
            plt.savefig(path)
        plt.show()

    def auto_lirpa_verify(self):
        self.write_logs('启动分析')
        self.eps_list = self.computer_eps_range()
        self.write_logs(f'分析扰动大小：{self.eps_list}')
        lirpa_adv_rates = self.computer_adversarial_exist_rate_by_eps(self.eps_list)
        self.lirpa_res = lirpa_adv_rates
        self.write_logs(info=f'扰动分析结束', task_finish=True)
        
        return

    def pack_result(self):
        self.eps_list = self.computer_eps_range()
        #gen_adv_rates = self.generate_adversarial_example_by_eps_with_fgsm(self.eps_list)
        gen_adv_rates = []
        result_fgsm = {'name': "FGSM", 'eps': self.eps_list, 'rates': gen_adv_rates}
        result_lirpa = {'name': "LiRPA", 'eps': self.eps_list, 'rates': self.lirpa_res}
        return {'FGSM': result_fgsm, 'LiRPA': result_lirpa}



def verify(param: dict):
    interface = param['interface']
    node = param['node']
    input_param = param['input_param'] if 'input_param' in param else None
    assert interface == "Verification"
    assert node == "中间结果可视化"
    assert input_param is not None
    verify_model = input_param['model'] if 'model' in input_param else None
    dataloader = input_param['dataset'] if 'dataset' in input_param else None
    n_class = input_param['n_class'] if 'n_class' in input_param else 0
    up_eps = input_param['up_eps'] if 'up_eps' in input_param else 0.1
    down_eps = input_param['down_eps'] if 'down_eps' in input_param else 0.01
    steps = input_param['steps'] if 'steps' in input_param else 5
    device = input_param['device'] if 'device' in input_param else 'cpu'
    output_path = input_param['output_path'] if 'output_path' in input_param else 'output'
    log_func = input_param['log_func'] if 'log_func' in input_param else None
    task_id = input_param['task_id'] if 'task_id' in input_param else 'default'

    assert verify_model is not None
    assert dataloader is not None
    assert n_class is not 0
    assert output_path is not None

    autoV = LanguageAutoLirpaVerify(verify_model=verify_model,
                                    data_batches=dataloader,
                                    n_class=n_class,
                                    up_eps=up_eps,
                                    down_eps=down_eps,
                                    steps=steps,
                                    device=device,
                                    output_path=output_path,
                                    log_func=log_func,
                                    task_id=task_id)
    autoV.auto_lirpa_verify()

    result = {'interface': interface,
              'node': node,
              'output_param': autoV.pack_result()}
    return result


if __name__ == '__main__':
    pass


from auto_LiRPA.auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.auto_LiRPA.perturbations import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
<<<<<<< HEAD
from function.attack.adv0211.art.attacks.evasion import FastGradientMethod, DeepFool, AutoAttack
from function.attack.adv0211.art.estimators.classification import PyTorchClassifier
=======
from function.attack.attacks.evasion import FastGradientMethod, DeepFool, AutoAttack
from function.attack.estimators.classification import PyTorchClassifier
>>>>>>> gitee/feature_chunlai
import numpy as np
import torch.nn as nn
import torch.optim as optim
import os
import shutil
from torchvision import transforms


plt.switch_backend('Agg')
lirpa_adv_rates = []
gen_adv_rates = []


class VisionAutoLirpaVerify(object):
    def __init__(self, verify_model, dataloader, n_class, up_eps=0.1, down_eps=0.01, steps=5, device='cpu',
                 output_path=None, log_func=None, task_id='default'):
        super(VisionAutoLirpaVerify, self).__init__()
        self.verify_model = verify_model
        self.dataloader = dataloader
        self.n_class = n_class
        self.up_eps = up_eps
        self.down_eps = down_eps
        self.steps = steps
        self.device = device
        self.output_path = output_path
        self.log_func = log_func
        self.task_id = task_id
        self.lirpa_res = list()
        self.fgsm_res = list()
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
            os.mkdir(self.output_path)

        if output_path is not None and not os.path.exists(output_path):
            os.makedirs(output_path)
        self.verify_model.to(self.device)
        self.eps_list = list()
        self.original_pic_dir = None
        self.adv_pic_dir = None
        self.original_pic_dir = os.path.join(self.output_path, 'adv') if output_path is not None else None
        self.adv_pic_dir = os.path.join(self.output_path, 'adv') if output_path is not None else None
        self.fgsm_dir = os.path.join(self.output_path, 'fgsm') if output_path is not None else None
        self.auto_LiRPA_dir = os.path.join(self.output_path, 'auto_LiRPA') if output_path is not None else None

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

    def boundary(self, norm, eps):
        lb_list = None
        ub_list = None
        label_list = list()
        ver_model = None
        number_param = sum(p.numel() for p in self.verify_model.parameters())
        method = 'backward' if number_param <= 600000 else 'CROWN-IBP'
        for i, (data, label) in tqdm(enumerate(self.dataloader)):
            data = data.to(self.device)
            label = label.to(self.device)
            if ver_model is None:
                ver_model = BoundedModule(self.verify_model.to(self.device),
                                          torch.zeros_like(data),
                                          bound_opts={"conv_mode": "patches"})
            ptb = PerturbationLpNorm(norm=norm, eps=eps)
            input_data = BoundedTensor(data, ptb)
            prediction = ver_model(input_data)
            lb, ub = ver_model.compute_bounds(x=(input_data,), method=method)
            if lb_list is None:
                lb_list = lb
                ub_list = ub
                label_list = prediction.argmax(dim=1)
            else:
                lb_list = torch.cat((lb_list, lb), dim=0)
                ub_list = torch.cat((ub_list, ub), dim=0)
                label_list = torch.cat((label_list, prediction.argmax(dim=1)), dim=0)

            # label_list.append(prediction.argmax(dim=1).tolist())
        lb_list = lb_list.detach().numpy().reshape(-1, self.n_class).tolist()
        ub_list = ub_list.detach().numpy().reshape(-1, self.n_class).tolist()
        label_list = label_list.detach().numpy().reshape(-1).tolist()

        # label_list = np.array(label_list).reshape(-1).tolist()
        return lb_list, ub_list, label_list

    def computer_adversarial_exist_rate_by_eps(self, eps_list: list):
        adv_rate_list = list()
        for eps in eps_list:
            self.write_logs(f'扰动大小为：{eps}，开始测试')
            # train_data = self.dataloader
            # train_data.batch_size = self.batch_size
            lb, ub, label = self.boundary(np.inf, eps)
            noise = self.computer_noise(lb, ub, label)
            adv_rate_list.append(noise)
        return adv_rate_list

    def generate_adversarial_examples(self, eps=0.01):
        lr = 1
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.verify_model.parameters(), lr=lr)
        classifier = PyTorchClassifier(
            model=self.verify_model,
            clip_values=(0, 1),
            loss=criterion,
            optimizer=optimizer,
            input_shape=next(iter(self.dataloader))[0][0].shape,
            nb_classes=self.n_class,
        )
        pic_index = 1
        original_pic_dir = ''
        adv_pic_dir = ''
        if self.output_path:
            original_pic_dir = os.path.join(self.original_pic_dir, f'{eps}', 'ori')
            adv_pic_dir = os.path.join(self.adv_pic_dir, f'{eps}', 'adv')

            if not os.path.exists(original_pic_dir):
                os.makedirs(original_pic_dir)

            if not os.path.exists(adv_pic_dir):
                os.makedirs(adv_pic_dir)

        adv_sucess_num = 0
        accuracy = 0
        data_number = 0
        adv_dataset = self.dataloader  # DataLoader(dataset, batch_size=1, shuffle=True)
        for data, label in tqdm(adv_dataset):
            attack = FastGradientMethod(classifier, eps=eps)
            x_test_adv = attack.generate(x=data.cpu().numpy())
            predictions = classifier.predict(x_test_adv)
            accuracy += np.mean(np.argmax(predictions, axis=1) == label.tolist())
            comp = torch.eq(torch.argmax(torch.tensor(predictions), dim=1), label)
            adv_sucess_num += data.shape[0] - int(comp.sum())
            data_number += data.shape[0]
            if len(data) > comp.sum():
                if self.output_path:
                    for index, index in enumerate(range(len(comp.tolist()))):
                        if comp.tolist()[index]:
                            continue
                        d = transforms.ToPILImage()(data[index])
                        original_pic = os.path.join(original_pic_dir, f'{pic_index}.jpg')
                        d.save(original_pic)
                        adv = transforms.ToPILImage()(torch.from_numpy(x_test_adv[index]))
                        adv_pic = os.path.join(adv_pic_dir, f'{pic_index}.jpg')
                        adv.save(adv_pic)
                        pic_index += 1
        return round(adv_sucess_num / data_number, 2)

    def generate_adversarial_example_by_eps_with_fgsm(self, eps_list):
        adv_rate_list = list()
        for eps in eps_list:
            rate = self.generate_adversarial_examples(eps=eps)
            self.write_logs(f'计算扰动大小为：{eps}，FGSM算法攻击成功率为：{rate}')
            adv_rate_list.append(rate)

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
        plt.xticks(fontproperties='Times New Roman', size=15)
        plt.yticks(fontproperties='Times New Roman', size=15)
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
        self.write_logs(f'启动对抗攻击测试')
        gen_adv_rates = self.generate_adversarial_example_by_eps_with_fgsm(self.eps_list)
        dict = {"eps": self.eps_list, "lirpa_adv_rates": lirpa_adv_rates, "gen_adv_rates": gen_adv_rates}
        dict = json.dumps(dict)
        print(dict)
        self.lirpa_res = lirpa_adv_rates
        self.fgsm_res = gen_adv_rates
        self.write_logs(f'扰动分析结束', task_finish=True)
        '''self.draw_rate_pic([str(s) for s in self.eps_list],
                           lirpa_adv_rates,
                           label='adversarial exist rate(LiRPA)',
                           save_path='auto_LiRPA')
        self.draw_rate_pic([str(s) for s in self.eps_list],
                           gen_adv_rates,
                           label='adversarial exist rate(FGSM)',
                           save_path='FGSM')'''
        return

    def pack_result(self):
        
        self.eps_list = self.computer_eps_range()
        result_fgsm = {'name': "FGSM", 'eps': self.eps_list, 'rates': self.fgsm_res}
        result_lirpa = {'name': "LiRPA", 'eps': self.eps_list, 'rates': self.lirpa_res}
        return {'FGSM': result_fgsm, 'LiRPA': result_lirpa}


def verify(param):
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

    autoV = VisionAutoLirpaVerify(verify_model=verify_model,
                                  dataloader=dataloader,
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


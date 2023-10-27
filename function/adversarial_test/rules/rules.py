
from function.adversarial_test.attack.gen_adv import attacks_rules_dict
from function.adversarial_test.unit import write_json
from tqdm import tqdm
import torch
import os


RESULT_FOLDER = os.path.join(os.path.dirname(__file__), 'results')
DATA_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
for path in [RESULT_FOLDER, DATA_FOLDER]:
    if not os.path.exists(path):
        os.mkdir(path)

# ['fgsm', 'bim', 'rfgsm', 'ffgsm', 'tifgsm', 'nifgsm', 'sinfgsm', 'vmifgsm', 'vnifgsm', 'mifgsm', 'difgsm', 'spsa', 'cw',
#  'upgd', 'pgd', 'tpgd', 'pgdl2', 'pgdrsl2', 'sparsefool', 'autopgd', 'onepixel', 'square', 'pixle', 'fab', 'jitter', 'deepfool']
def get_attack_methods_with_rules(normal, attack_type, task_type):
    key_map = {'white_box': '白盒',
               'black_box': '黑盒',
               'evasion_attack': '逃逸攻击',
               'poison_attack': '毒化攻击',
               'image': '图片',
               'text': '文本',
               'graph': '图'}
    if normal == 'l2':
        return


class Rules(object):
    def __init__(self, model, dataloader, eps, attack_methods, device='cuda', log_func=None):
        self.model = model
        self.dataloader = dataloader
        self.eps = eps
        self.attack_methods = attack_methods
        self.device = device
        self.log_func = log_func
        self.attacks = attacks_rules_dict(self.model, self.eps)

    def write_logs(self, info):
        print(info)
        if self.log_func is None:
            return
        self.log_func(info)

    def calc_adv_acc(self, method, attack, dataloader):
        correct = 0
        self.model = self.model.to(self.device)
        self.model.eval()
        for data, label in tqdm(dataloader, ncols=100, desc=f'{method} generate adversarial examples'):
            data, label = data.to(self.device), label.to(self.device)
            adv = attack(data, label)
            outputs = self.model(adv)
            _, pre = torch.max(outputs.data, 1)
            correct += float((pre == label).sum()) / len(label)

        return round(correct / len(dataloader), 4) * 100

    def calculate_attack_acc(self, dataloader):
        acc = dict()
        for method in self.attack_methods:
            self.write_logs(f"启动{method}方法攻击测试")
            method_lower = method.lower()
            if method_lower not in self.attacks:
                acc[method] = 0
                continue
            attack = self.attacks[method_lower]
            acc[method] = self.calc_adv_acc(method, attack, dataloader)
            self.write_logs(f'执行{method}算法，准确率为：{acc[method]}')
        return acc

    def calc_ori_acc(self, dataloader):
        self.model.to(self.device)
        correct = 0
        self.model.eval()
        for data, label in dataloader:
            data = data.to(self.device)
            label = label.to(self.device)
            outputs = self.model(data)
            _, pre = torch.max(outputs.data, 1)
            correct += float((pre == label).sum()) / len(label)
        return round(correct / len(dataloader), 4) * 100

    def run(self):
        results = dict()
        self.write_logs("计算原始准确率")
        results['ori_acc'] = self.calc_ori_acc(self.dataloader)
        self.write_logs("启动攻击")
        results['attack_results'] = self.calculate_attack_acc(self.dataloader)
        write_json(results, os.path.join(RESULT_FOLDER, 'results.json'))
        return results


if __name__ == '__main__':
    pass


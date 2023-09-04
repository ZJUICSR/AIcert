import numpy as np
from torch._C import wait
import torchvision
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose, Normalize
from torch.utils.data import DataLoader, dataloader
import torch
import torch.nn.functional as F
import torch.nn as nn
import json
from tqdm import tqdm
import os


class NCoverage_new():

    def __init__(self, model, model_type='resnet', exclude_layer=['pool', 'fc', 'flatten'], k=2, topk=3, device='cpu', log_func=None):
        '''
        初始化模型
        :param model: 待测试模型
        :param exclude_layer: 不被考虑的神经网络层
        :param k: 神经元输出区间的个数
        :param topk: 联合分布覆盖率选择的神经元数量
        :param device: 进行运算的设备
        '''
        self.model = model.to(device)
        self.topk = topk
        self.k = k
        self.device = device
        self.log_func = log_func

        print('models loaded')

        '''the layers that are considered in neuron coverage computation'''
        self.layer_to_compute = []

        if 'resnet' in model_type:
            for name, layer in self.model.named_children():
                if 'Sequential' in str(type(layer)):
                    for name2, layer2 in layer.named_children():
                        # print(layer2)
                        for name3, layer3 in layer2.named_children():
                            # print(type(layer3))
                            if 'Conv2d' in str(type(layer3)):
                                self.layer_to_compute.append(name + '[' + name2 + ']' + '.' + name3)
                            if 'Linear' in str(type(layer3)):
                                self.layer_to_compute.append(name + '[' + name2 + ']' + '.' + name3)
            print(self.layer_to_compute)
            self.activation = {}
            for layer_name in self.layer_to_compute:
                eval("self.model." + str(layer_name) + ".register_forward_hook(self._get_activation('" + str(
                    layer_name) + "'))")
        else:
            for name, layer in self.model.named_children():
                if name == 'features':
                    for name2, layer2 in layer.named_children():
                        if 'Conv2d' in str(type(layer2)):
                            self.layer_to_compute.append('features[' + name2 + ']')
                elif name == 'classifier':
                    for name2, layer2 in layer.named_children():
                        if 'Linear' in str(type(layer2)):
                            self.layer_to_compute.append('classifier[' + name2 + ']')
                elif all(ex not in str(type(layer)) for ex in exclude_layer):
                    self.layer_to_compute.append(name)

            self.activation = {}
            for layer_name in self.layer_to_compute:
                eval("self.model." + str(layer_name) + ".register_forward_hook(self._get_activation('" + str(
                    layer_name) + "'))")
        # print('computed layers are: ', self.layer_to_compute)

        '''init coverage table'''
        self.reset_dict()

    def _get_activation(self, name):
        def hook(model, input, output):
            # 如果你想feature的梯度能反向传播，那么去掉 detach（）
            self.activation[name] = output.detach()

        return hook

    def refresh_low_and_high(self, dataloader):
        '''
        datasets: with the shape of (n, d, d, 3)
        '''

        mnist_model = self.model
        # 使用hook建立模型中间值输出与activation字典的联系，之后每次使用模型预测结果，中间值的输出都会保存到activation字典中
        for layer_name in self.layer_to_compute:
            eval("mnist_model." + str(layer_name) + ".register_forward_hook(self._get_activation('" + str(
                layer_name) + "'))")

        for number, (input_data, y) in enumerate(dataloader):
            if self.log_func is not None:
                self.log_func("[模型测试阶段] 即将运行课题二的模型标准化测试准则：coverage，更新进度：[{:d}/{:d}]".format(number, 10))

            if number > 10:
                break
            # if number%500 == 0:
            #     print(number,'has done!')

            outputs = self.model(input_data.to(self.device))

            for layer_name, layer_output in self.activation.items():
                scaled = self.scale(layer_output)
                if 'conv' in layer_name or 'features' in layer_name:
                    for neuron_idx in range(scaled.shape[1]):
                        mean = torch.mean(torch.mean(scaled[:, neuron_idx, :, :], dim=-1), dim=-1)
                        mean_max = torch.max(mean)
                        mean_min = torch.min(mean)

                        if mean_max > self.boundaty_dict[(layer_name, neuron_idx)][1]:
                            self.boundaty_dict[(layer_name, neuron_idx)][1] = mean_max
                        if mean_min < self.boundaty_dict[(layer_name, neuron_idx)][0]:
                            self.boundaty_dict[(layer_name, neuron_idx)][0] = mean_min
                else:
                    for neuron_idx in range(scaled.shape[-1]):
                        mean_max = torch.max(scaled[..., neuron_idx])
                        mean_min = torch.min(scaled[..., neuron_idx])
                        if mean_max > self.boundaty_dict[(layer_name, neuron_idx)][1]:
                            self.boundaty_dict[(layer_name, neuron_idx)][1] = mean_max
                        if mean_min < self.boundaty_dict[(layer_name, neuron_idx)][0]:
                            self.boundaty_dict[(layer_name, neuron_idx)][0] = mean_min
            del outputs

        return self.boundaty_dict

    def coverage_step(self, data):
        '''
        Update coverage for individual data
        '''
        layer_model = self.model
        self.activation = {}
        for layer_name in self.layer_to_compute:
            eval("layer_model." + str(layer_name) + ".register_forward_hook(self._get_activation('" + str(
                layer_name) + "'))")

        if len(data.shape) < 4:
            data = torch.unsqueeze(data, dim=0)
        outputs = self.model(data.to(self.device))

        for layer_name, layer_output in self.activation.items():
            scaled = self.scale(layer_output)

            if 'conv' in layer_name or 'features' in layer_name:
                for neuron_idx in range(scaled.shape[1]):
                    low = self.boundaty_dict[(layer_name, neuron_idx)][0]
                    high = self.boundaty_dict[(layer_name, neuron_idx)][1]
                    interval = (high - low) / self.k
                    mean = torch.mean(torch.mean(scaled[:, neuron_idx, :, :], dim=-1), dim=-1)
                    mean_max = torch.max(mean)
                    mean_min = torch.min(mean)
                    if mean_min < low:
                        self.cov_boundary_dict[(layer_name, neuron_idx)][0] = True
                    if mean_max > high:
                        self.cov_boundary_dict[(layer_name, neuron_idx)][1] = True

                    numbers = torch.ceil((mean - low) / interval - 0.0001)
                    for number in numbers:
                        if number < self.k and number > -1:
                            number = int(number.data)
                            self.k_cov_dict[(layer_name, neuron_idx)][number - 1] = True

                mean = torch.mean(torch.mean(scaled, dim=-1), dim=-1)
                for data in mean:
                    top_k_list = torch.argsort(data, descending=True)[0:self.topk]
                    if str(top_k_list) not in self.top_k_neuron_patterns.keys():
                        self.top_k_neuron_patterns[str(top_k_list)] = True

                    for top_k_neuron in top_k_list:
                        self.top_k_neuron_cov_dict[(layer_name, top_k_neuron)] = True

            else:
                for neuron_idx in range(scaled.shape[-1]):
                    low = self.boundaty_dict[(layer_name, neuron_idx)][0]
                    high = self.boundaty_dict[(layer_name, neuron_idx)][1]
                    interval = (high - low) / self.k
                    mean_max = torch.max(scaled[..., neuron_idx])
                    mean_min = torch.min(scaled[..., neuron_idx])

                    if mean_min < low:
                        self.cov_boundary_dict[(layer_name, neuron_idx)][0] = True
                    if mean_max > high:
                        self.cov_boundary_dict[(layer_name, neuron_idx)][1] = True

                    numbers = torch.ceil((scaled[..., neuron_idx] - low) / interval - 0.0001)
                    for number in numbers:
                        if number < self.k and number > -1:
                            number = int(number.data)
                            self.k_cov_dict[(layer_name, neuron_idx)][number - 1] = True

                for data in scaled:
                    top_k_list = torch.argsort(data, descending=True)[0:self.topk]
                    if str(top_k_list) not in self.top_k_neuron_patterns.keys():
                        self.top_k_neuron_patterns[str(top_k_list)] = True

                    for top_k_neuron in top_k_list:
                        self.top_k_neuron_cov_dict[(layer_name, top_k_neuron)] = True

        del outputs
        del layer_model

    def update_coverage(self, dataloader):
        '''
        Given the input, update the neuron covered in the model by this input.
            This includes mark the neurons covered by this input as "covered"
        :param input_data: the input image, with the shape of (1, d, d, 3)
        :return: the neurons that can be covered by the input
        '''
        layer_model = self.model
        self.activation = {}
        for layer_name in self.layer_to_compute:
            eval("layer_model." + str(layer_name) + ".register_forward_hook(self._get_activation('" + str(
                layer_name) + "'))")

        for n, (input_data, y) in enumerate(dataloader):
            if self.log_func is not None:
                self.log_func("[模型测试阶段] 即将运行课题二的模型标准化测试准则：coverage，测试进度：[{:d}/{:d}]".format(n, 15))
            if n > 15:
                break
            outputs = self.model(input_data.to(self.device))

            for layer_name, layer_output in self.activation.items():
                scaled = self.scale(layer_output)

                if 'conv' in layer_name or 'features' in layer_name:
                    for neuron_idx in range(scaled.shape[1]):
                        low = self.boundaty_dict[(layer_name, neuron_idx)][0]
                        high = self.boundaty_dict[(layer_name, neuron_idx)][1]
                        interval = (high - low) / self.k
                        mean = torch.mean(torch.mean(scaled[:, neuron_idx, :, :], dim=-1), dim=-1)
                        mean_max = torch.max(mean)
                        mean_min = torch.min(mean)
                        if mean_min < low:
                            self.cov_boundary_dict[(layer_name, neuron_idx)][0] = True
                        if mean_max > high:
                            self.cov_boundary_dict[(layer_name, neuron_idx)][1] = True

                        numbers = torch.ceil((mean - low) / interval - 0.0001)
                        for number in numbers:
                            if number < self.k and number > -1:
                                number = int(number.data)
                                self.k_cov_dict[(layer_name, neuron_idx)][number - 1] = True

                    mean = torch.mean(torch.mean(scaled, dim=-1), dim=-1)
                    for data in mean:
                        top_k_list = torch.argsort(data, descending=True)[0:self.topk]
                        if str(top_k_list) not in self.top_k_neuron_patterns.keys():
                            self.top_k_neuron_patterns[str(top_k_list)] = True

                        for top_k_neuron in top_k_list:
                            self.top_k_neuron_cov_dict[(layer_name, top_k_neuron)] = True

                else:
                    for neuron_idx in range(scaled.shape[-1]):
                        low = self.boundaty_dict[(layer_name, neuron_idx)][0]
                        high = self.boundaty_dict[(layer_name, neuron_idx)][1]
                        interval = (high - low) / self.k
                        mean_max = torch.max(scaled[..., neuron_idx])
                        mean_min = torch.min(scaled[..., neuron_idx])

                        if mean_min < low:
                            self.cov_boundary_dict[(layer_name, neuron_idx)][0] = True
                        if mean_max > high:
                            self.cov_boundary_dict[(layer_name, neuron_idx)][1] = True

                        numbers = torch.ceil((scaled[..., neuron_idx] - low) / interval - 0.0001)
                        for number in numbers:
                            if number < self.k and number > -1:
                                number = int(number.data)
                                self.k_cov_dict[(layer_name, neuron_idx)][number - 1] = True

                    for data in scaled:
                        top_k_list = torch.argsort(data, descending=True)[0:self.topk]
                        if str(top_k_list) not in self.top_k_neuron_patterns.keys():
                            self.top_k_neuron_patterns[str(top_k_list)] = True

                        for top_k_neuron in top_k_list:
                            self.top_k_neuron_cov_dict[(layer_name, top_k_neuron)] = True

            # if n%100 == 0:
            #     print(n,'has down!')

        del outputs
        del layer_model

    def scale(self, layer_outputs, rmax=1, rmin=0, scale=False):
        '''
        scale the intermediate layer's output between 0 and 1
        :param layer_outputs: the layer's output tensor
        :param rmax: the upper bound of scale
        :param rmin: the lower bound of scale
        :return:
        '''
        if not scale:
            return layer_outputs
        divider = (layer_outputs.max() - layer_outputs.min())
        if divider == 0:
            return np.zeros(shape=layer_outputs.shape)
        X_std = (layer_outputs - layer_outputs.min()) / divider
        X_scaled = X_std * (rmax - rmin) + rmin
        return X_scaled

    def curr_neuron_cov(self):
        '''
        返回当前的k-multisection Neuron Cov
        '''

        covered_neurons = 0
        for value in self.k_cov_dict.values():
            covered_neurons += len([v for v in value if v])
        total_neurons = len(self.k_cov_dict) * self.k
        return covered_neurons / float(total_neurons)

    def curr_neuron_boundary_cov(self):
        '''
        返回Neuron Boundary Cov和Strong Neuron Activation Cov,以列表的形式[NBC, SNAC]
        '''
        upper_corner_neurons = 0
        lower_corner_neurons = 0
        for value in self.cov_boundary_dict.values():
            if value[0]:
                upper_corner_neurons += 1
            if value[1]:
                lower_corner_neurons += 1
        total_neurons = len(self.k_cov_dict)
        neuron_boundary_coverage = torch.true_divide((upper_corner_neurons + lower_corner_neurons), (2 * total_neurons))
        strong_neuron_coverage = torch.true_divide(upper_corner_neurons, total_neurons)
        return neuron_boundary_coverage, strong_neuron_coverage

    def curr_top_k_neuron_coverage(self):
        '''
        返回当前的Top-k Neuron Cov
        '''
        corvered_neurons = sum(self.top_k_neuron_cov_dict.values())
        total_neurons = len(self.top_k_neuron_cov_dict)
        return corvered_neurons / total_neurons

    def curr_top_k_patterns(self):
        '''
        返回当前的Top-k Neuron Patterns
        '''
        return len(self.top_k_neuron_patterns)

    def get_dataset_coverage(self, dataset):
        '''
        输入为数据集,维度(n, d, d, 3)
        该函数也可以只用来更新数据,不保存返回值
        返回五种覆盖率指标,以列表的形式[KMNC, [NBC, SNAC], TKNC, TKNP]
        '''
        for data in dataset:
            self.update_coverage(torch.unsqueeze(data, dim=0))
        return self.curr_neuron_cov(), self.curr_neuron_boundary_cov(), self.curr_top_k_neuron_coverage(), self.curr_top_k_patterns()

    def reset_dict(self):
        '''
        Reset the coverage table
        :return:
        '''
        # activation自定用来存储模型的中间 输出结果
        self.activation = {}

        # boundary_dict storage the minimum and maximum of each neuron, each element is a list of the form like [min, max].
        self.boundaty_dict = {}

        # For K-Multisection Neuron Coverage, each Neuron output is divided into k segments, and k_cov_dict stores information about whether each segment is covered
        # The dictionary is used to calculate KMNC
        self.k_cov_dict = {}

        # cov_boundary_dict stores information about whether the boundaries of each neuron are covered, in the form of [lower,upper]
        # The dictionary is used to calculate NBC and SNAC
        self.cov_boundary_dict = {}

        # top_k_neuron_cov_dict stores information about whether the output of each neuron has reached top_k
        # The dictionary is used to calculate TKNC
        self.top_k_neuron_cov_dict = {}

        # top_k_neuron_patterns
        self.top_k_neuron_patterns = {}

        # Assigns initial values to all sets
        for layer_name in self.layer_to_compute:
            for index in range(self._get_layer_output_number(layer_name)):
                self.boundaty_dict[(layer_name, index)] = [100000, -100000]  # The number 100 is arbitrary
                self.k_cov_dict[(layer_name, index)] = [False] * self.k  # [min,max]
                self.cov_boundary_dict[(layer_name, index)] = [False, False]  # [upper, lower]
                self.top_k_neuron_cov_dict[(layer_name, index)] = False

    def _get_layer_output_number(self, layer_name):
        if isinstance(eval('self.model.' + str(layer_name)), torch.nn.Sequential):
            if hasattr(eval('self.model.' + str(layer_name))[0], 'out_channels'):
                return eval('self.model.' + str(layer_name))[0].out_channels
            else:
                return eval('self.model.' + str(layer_name))[0].out_features

        if 'conv' not in layer_name and 'features' not in layer_name:
            return eval('self.model.' + str(layer_name)).out_features
        else:
            return eval('self.model.' + str(layer_name)).out_channels


def run_coverage(model, testloader, filename, trainloader=None, device='cpu', criterion='KMNC', k=2,
                 model_type='resnet', hash_tag=None, hash_tag_dir='', log_func=None):
    with open(filename, 'r') as file_obj:
        json_data = json.load(file_obj)

    if os.path.exists(os.path.join(hash_tag_dir, hash_tag)):
        with open(os.path.join(hash_tag_dir, hash_tag), 'r') as f:
            result = f.read()

        with open(filename, 'w') as file_obj:
            if 'coverage_test_yz' not in json_data.keys():
                json_data['coverage_test_yz'] = {}
            json_data['coverage_test_yz']['coverage_rate'] = result
            json.dump(json_data, file_obj)
        return

    if log_func is not None:
        log_func("[模型测试阶段] 即将运行课题二的模型标准化测试准则：coverage，测试方法为：{:s}".format(str(criterion)))


    if criterion == 'KMNC':
        nc = NCoverage_new(model, k=k, device=device, model_type=model_type, log_func=log_func)
        if trainloader:
            nc.refresh_low_and_high(trainloader)
        else:
            nc.refresh_low_and_high(testloader)
        nc.update_coverage(testloader)
        result = nc.curr_neuron_cov()
        result = round(result, 4) * 100
        result_str = {'The KMNC result:': result}



    elif criterion == 'NBC':
        nc = NCoverage_new(model, device=device, model_type=model_type, log_func=log_func)
        assert trainloader
        nc.refresh_low_and_high(trainloader)
        nc.update_coverage(testloader)
        result = nc.curr_neuron_boundary_cov()[0]
        result = round(float(result), 4) * 100
        result_str = {'The NBC result:': result}


    elif criterion == 'SNAC':
        nc = NCoverage_new(model, device=device, model_type=model_type, log_func=log_func)
        assert trainloader
        nc.refresh_low_and_high(trainloader)
        nc.update_coverage(testloader)
        result = nc.curr_neuron_boundary_cov()[1]
        result = round(float(result), 4) * 100
        result_str = {'The SNAC result:': result}


    elif criterion == 'TKNC':
        nc = NCoverage_new(model, device=device, model_type=model_type, log_func=log_func)
        nc.update_coverage(testloader)
        result = nc.curr_top_k_neuron_coverage()
        result = round(result, 4) * 100
        result_str = {'The TKNC result:', result}

    elif criterion == 'TKNP':
        nc = NCoverage_new(model, device=device, model_type=model_type, log_func=log_func)
        nc.update_coverage(testloader)
        result = nc.curr_top_k_patterns()
        result_str = {'The TKNP result:': result}

    with open(filename, 'w') as file_obj:
        if 'coverage_test_yz' not in json_data.keys():
            json_data['coverage_test_yz'] = {}
        json_data['coverage_test_yz']['coverage_rate'] = result
        json.dump(json_data, file_obj)

    with open(os.path.join(hash_tag_dir, hash_tag), 'w') as f:
        return f.write(str(result))

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 6, 5, 1),  # input_size=(1*28*28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # input_size=(6*24*24)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),  # input_size=(6*12*12)
            nn.ReLU(),  # input_size=(16*8*8)
            nn.MaxPool2d(2, 2)  # output_size=(16*4*4)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        
        x = self.conv2(x)

        # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
        
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        
        x = self.fc3(x)
        return torch.log_softmax(x, dim=-1)


import torchvision.datasets as datasets
from torchvision.transforms import ToTensor, Compose, Resize


def get_dataloader_mnist(train, batch_size=16, input_size=(32, 32)):
    transform_fn = Compose([Resize(input_size), ToTensor(), ])
    dataset = datasets.MNIST('./data/mnist', download=True, train=train, transform=transform_fn)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def get_dataloader_cifar(train, batch_size=2, input_size=(32, 32)):
    transform_fn = Compose([Resize(input_size), ToTensor(), ])
    dataset = datasets.CIFAR10('./data/cifar10', download=True, train=train, transform=transform_fn)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


if __name__ == '__main__':
    '''
    文件以hash_tag作为文件名，如果存在该文件，则直接读取文件中内容返回，
    如果不存在则正常计算覆盖率，并将计算结果保存在以hash_tag为文件名的文件中, 
    hash_tag_dir是保存hash_tag的路径

    criterion参数已经取消了, 仍然满足验收要求
    '''
    model = torchvision.models.vgg11(False)
    input_size = (32, 32)
    testloader = get_dataloader_cifar(False, 16, input_size=input_size)
    trainloader = get_dataloader_cifar(True, 16, input_size=input_size)
    #
    run_coverage(model, testloader, 'keti2(2)(1).json', trainloader=trainloader, model_type='vgg11', device='cuda',
                 hash_tag='model:vgg11,dataset:cifar', hash_tag_dir='')



import os.path as osp
def run(model, train_loader, test_loader, params, hash_tag="", log_func=None):
    root = osp.join(params["out_path"], "keti2")
    result_file = osp.join(root, "keti2.json")
    criterions = params["coverage"]["criterions"]
    
    lenet_flag = True if 'lenet' in params["coverage"].keys() and eval(params["coverage"]['lenet']) else False
    hash_tag_dir = params["cache_path"]

    # 目录需要更改
    lenet_path = osp.join(os.path.dirname(__file__),'LeNet/model.pth')
    if lenet_flag:
        model=LeNet5()
        model.load_state_dict(torch.load(lenet_path))

    if log_func is not None:
        log_func("[模型测试阶段] 即将运行课题二的模型标准化测试准则：coverage，其中hash_tag={:s}".format(hash_tag))
    for criterion in criterions:
        run_coverage(model, testloader=test_loader,
                     filename=result_file,
                     trainloader=train_loader,
                     device=params["device"],
                     criterion=criterion,
                     hash_tag=hash_tag, hash_tag_dir=hash_tag_dir,
                     log_func=log_func
        )




"""模型标准化准则测试"""
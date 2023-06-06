from graphviz import Digraph
from numpy import ceil, ndenumerate
import numpy as np
from torchvision.transforms import ToTensor, Compose, Normalize
from torch.utils.data import DataLoader, dataloader
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.transforms import ToTensor, Compose, Resize
import torchvision.datasets as datasets
from tqdm import tqdm
import os
import json
import torchvision


class NCoverage_with_output_of_conv_kernel():
    def __init__(self, model, model_type='resnet', exclude_layer=['pool', 'fc', 'flatten'],
                 k=2, device='cpu', input_size=(1, 28, 28), dataloader=None):
        '''
        Initialize the model to be tested
        :param model_name: ImageNet Model name
        :param exclude_layer: these layers are not considered for neuron coverage
        '''
        self.model = model.to(device)
        self.k = k
        self.device = device
        self.input_size = input_size
        self.dataloader = dataloader

        #print('models loaded')

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
                elif 'fc' in name:
                    self.layer_to_compute.append(name)
                elif 'conv' in name:
                    self.layer_to_compute.append(name)
                
            # print(self.layer_to_compute)
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
                elif name == 'classifier' or name=='fc':
                    for name2, layer2 in layer.named_children():
                        if 'Linear' in str(type(layer2)):
                            self.layer_to_compute.append('classifier[' + name2 + ']')
                elif all(ex not in str(type(layer)) for ex in exclude_layer):
                    self.layer_to_compute.append(name)

            self.activation = {}
            for layer_name in self.layer_to_compute:
                eval("self.model." + str(layer_name) + ".register_forward_hook(self._get_activation('" + str(
                    layer_name) + "'))")
        # exit()
        # print('computed layers are: ', self.layer_to_compute)

        '''init coverage table'''
        self.reset_dict()

    def _get_activation(self, name):
        def hook(model, input, output):
            # 如果你想feature的梯度能反向传播，那么去掉 detach（）
            self.activation[name] = output.detach()

        return hook

    def update_coverage_step(self, inputs):
        '''
        Given the input, update the neuron covered in the model by this input.
            This includes mark the neurons covered by this input as "covered"
        :param input_data: the input image, with the shape of (1, d, d, 3)
        :return: the neurons that can be covered by the input
        '''
        layer_model = self.model
        outputs = self.model(inputs.to(self.device))

        for idx, (layer_name, layer_output) in enumerate(self.activation.items()):
            scaled = self.scale(layer_output)
            if len(scaled.shape)>2:
                scaled = torch.mean(torch.mean(scaled,dim=-1),dim=-1)

            for values in scaled:
                for neuron_idx, value in enumerate(values):
                    if value > self.k:
                        self.cov_dict[layer_name][neuron_idx] = True
            


        

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
        total_neurons = 0
        covered_neurons = 0
        for name, layer in self.cov_dict.items():
            for value in layer:
                covered_neurons += 1 if value else 0
                total_neurons += 1
        return covered_neurons / float(total_neurons)

    def curr_cov_dict(self):
        return self.cov_dict

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

        self.cov_dict = {}
        mid = next(iter(self.dataloader))[0]
        self.model(mid)
        for name, layer in self.activation.items():
            if len(layer.shape)>2:
                layer = torch.mean(torch.mean(layer,dim=-1),dim=-1)

            self.cov_dict[name] = [False] * torch.tensor(layer[0].flatten().shape, dtype=torch.int).item()


class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()  # 继承父类
        self.fc1 = nn.Linear(1 * 28 * 28, 28)  # 添加全连接层
        self.fc2 = nn.Linear(28, 10)

    def forward(self, input):
        x = input.view(-1, 1 * 28 * 28)
        x = self.fc1(x)
        x = F.relu(x)
        out = self.fc2(x)
        return F.log_softmax(out, dim=-1)  # log_softmax 与 nll_loss合用，计算交叉熵


def coverage_visualize(model_dir, dataset, batch_size, input_size):
    model = torch.load(model_dir)
    if dataset == 'mnist':
        dataloader = get_dataloader_mnist(False, batch_size=batch_size, input_size=input_size)
    if dataset == 'cifar':
        dataloader = get_dataloader_cifar(False, batch_size=batch_size, input_size=input_size)

    nc_coverage = NCoverage_with_output_of_conv_kernel(model, threshold=0.3)
    for x, y in dataloader:
        nc_coverage.update_coverage_step(x)
        result = nc_coverage.curr_cov_dict()


def get_dataloader_mnist(train, batch_size=16, input_size=(32, 32)):
    transform_fn = Compose([Resize(input_size), ToTensor(), ])
    dataset = datasets.MNIST('./dataset/data', download=True, train=train, transform=transform_fn)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def get_dataloader_cifar(train, batch_size=2, input_size=(32, 32)):
    transform_fn = Compose([Resize(input_size), ToTensor(), ])
    dataset = datasets.CIFAR10('./dataset/data', download=True, train=train, transform=transform_fn)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def load_dataset(dataset_type):
    if dataset_type == 'mnist':
        return get_dataloader_mnist(False, 1, input_size=(28,28))
    elif dataset_type == 'cifar10':
        return get_dataloader_cifar(False, batch_size=2, input_size=(32,32))
    assert 0

def draw_note(g, name, label='', color='blue', fillcolor='blue'):
    g.node(name, label=label,
           _attributes={'color': color, 'fillcolor': fillcolor, 'style': 'filled', 'shape': "circle", 'width': '0.2',
                        'height': '0.2'})


def draw_edge(g, tail, head, color='gray'):
    g.edge(tail, head,
           _attributes={'color': color, 'fontcolor': 'red', 'splines': 'line', 'arrowsize': '0.15', 'penwidth': '0.2'})


def get_color_from_digit(digit, max_value=10):
    digit = int(digit * 10 / max_value)
    color_dict = {0: 'FFFF', 1: 'DDDD', 2: 'CCCC', 3: 'BBBB', 4: 'AAAA', 5: '9999', 6: '8888', 7: '6666', 8: '4444',
                  9: '2222', 10: '0000'}
    color = '#' + color_dict[digit] + 'FF'
    return color


def afterprocess(net, num_list=[10, 6, 12, 12, 10], net_type='lenet'):
    import copy
    result = {}
    number_per_dot = {}
    if net_type == 'lenet':
        for num_idx, (key, value) in enumerate(net.items()):
            if num_idx > 20:
                break
            result[key] = []
            num = num_list[num_idx]
            step = int(ceil(len(value) / num))
            number_per_dot[key] = []
            for i in range(num):
                if i * step + step < len(value):
                    result[key].append(sum(value[i * step:i * step + step]))
                    number_per_dot[key].append(len(value[i * step:i * step + step]))
                else:
                    if len(value[i * step:]) > 0:
                        result[key].append(sum(value[i * step:]))
                        number_per_dot[key].append(len(value[i * step:]))

    return result, number_per_dot


def DrawNet_overlap(net, outputdir='', imagename='test', format='png', type_net='lenet',
                    number_per_dot=[100, 100, 10, 10, 1]):
    # print(net)
    '''
    :param net:网络结构，形如{'layer1':[True,False, False, True], 'layer2':[True, True]}
    :param outputdir:可视化图像的输出路径
    :param imagename:可视化图像名称
    :param format:可视化图像的格式，可为png、pdf、svg等
    '''
    if type(imagename) == list:
        g = Digraph(outputdir + imagename[0], format=format,
                    graph_attr={'rankdir': 'LR', 'splines': 'line', 'autosize': 'false', 'size': '5.0,8.3',
                                'background': 'black'})
    else:
        g = Digraph(outputdir + imagename[0], format=format,
                    graph_attr={'rankdir': 'LR', 'splines': 'line', 'autosize': 'false', 'size': '5.0,8.3',
                                'background': 'black'})

    for num_idx, key in enumerate(net.keys()):
        for n, neuron in enumerate(net[key]):
            if type(neuron) == int:
                draw_note(g, key + '_' + str(n), '', 'black',
                          get_color_from_digit(neuron, max_value=number_per_dot[key][n]))
            else:
                if neuron:
                    draw_note(g, key + '_' + str(n), '', 'black', 'blue')
                else:
                    draw_note(g, key + '_' + str(n), '', 'black', 'white')

    last_key = None

    if type_net == 'lenet':
        for i, key in enumerate(net.keys()):
            for n, neuron in enumerate(net[key]):
                if last_key:
                    if i < 3:
                        for n_last, neuron_last in enumerate(net[last_key]):
                            draw_edge(g, last_key + '_' + str(n_last), key + '_' + str(n))
                    else:
                        for n_last, neuron_last in enumerate(net[last_key]):
                            draw_edge(g, last_key + '_' + str(n_last), key + '_' + str(n))
            last_key = key

    elif type_net == 'type2':
        for i, key in enumerate(net.keys()):
            for n, neuron in enumerate(net[key]):
                if last_key:
                    if i < 3:
                        draw_edge(g, last_key + '_' + str(n), key + '_' + str(n))
                    elif i == 3:
                        draw_edge(g, last_key + '_' + str(2 * n), key + '_' + str(n))
                        draw_edge(g, last_key + '_' + str(2 * n + 1), key + '_' + str(n))
                    else:
                        for n_last, neuron_last in enumerate(net[last_key]):
                            draw_edge(g, last_key + '_' + str(n_last), key + '_' + str(n))
            last_key = key
    else:
        # print('dadadADXasd')
        for key in net.keys():
            for n, neuron in enumerate(net[key]):
                if last_key:
                    for n_last, neuron_last in enumerate(net[last_key]):
                        draw_edge(g, last_key + '_' + str(n_last), key + '_' + str(n))
            last_key = key

    if type(imagename) == list:
        for name in imagename:
            g.render(name, outputdir)
    else:
        g.render(imagename, outputdir)

    return g

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(1,6,5,1),#input_size=(1*28*28)
            # nn.Conv2d(1,6,5,1),#input_size=(1*28*28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)#input_size=(6*24*24)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),#input_size=(6*12*12)
            nn.ReLU(),      #input_size=(16*8*8)
            nn.MaxPool2d(2, 2)  #output_size=(16*4*4)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)

        # 定义前向传播过程，输入为x
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return torch.log_softmax(x, dim=-1)
    
    
def run_visualize_neural(
        model_path, # 模型路径, 需要可以使用torch.load加载
        dataset,           # 数据集名称, 支持mnist,cifar10
        model_type,         # 模型类型,支持lenet5, resnet18, vgg11,vgg13,vgg19   
        k=0.1,                       # 算法参数,范围[0,＋∞],例如k=0时, 输出值大于0的神经元被看作激活
        outputdir="./output",
        number_of_image=None,          # 总共计算多少张图片,None值时会计算整个数据集
        logging=None):
    # result_file = CURR+'/converage_neural_result.json'
    # outputdir=CURR+'/image_neural'
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
    dataloader = load_dataset(dataset)
    
    # log_func=None
    # model_path = CURR.rsplit('/',2)[0]+"/model/ckpt/lenet5.pth"
    
    if model_type == 'lenet5':
        num_list = [10, 6, 12, 12, 10]
        model = LeNet5()
        model.load_state_dict(torch.load(model_path))
    elif 'vgg' in model_type:
        num = int(model_type[3:])
        num_list = [10, 8] + [12, 8] * (num // 2 - 1) + [10] * (num % 2)
        model = torchvision.models.vgg11(num_classes=10)
    elif 'resnet' in model_type:
        num = int(model_type[6:])
        num_list = [10, 8] + [12, 8] * (num // 2 - 2) + [12, 10]
        model = torchvision.models.resnet18(num_classes=10)
    else:
        num = int(model_type[6:])
        num_list = [10, 8] + [12, 8] * (num // 2 - 1) + [10] * (num % 2)
        model = torch.load(model_path)
    # model.load_state_dict(torch.load(model_path))

    x, y = next(iter(dataloader))
    C, W, H = x.shape[1:]
    NC = NCoverage_with_output_of_conv_kernel(model=model, model_type=model_type, k=k,
                                              input_size=(C, W, H), dataloader=dataloader)
    best_conv = 0
    saves = []
    for i, (x, y) in enumerate(dataloader):
        x = x[0:2]
        result = NC.curr_cov_dict()
        result, number_per_dot = afterprocess(result, num_list=num_list)
        now_conv = NC.curr_neuron_cov()
        # if now_conv > best_conv + 0.05 or i %50==0:
        if now_conv > best_conv + 0.05:
            best_conv = now_conv
            saves.append((i, now_conv))
            g = DrawNet_overlap(result, format='png', type_net=model_type, outputdir=outputdir, imagename=str(i),
                            number_per_dot=number_per_dot)
            print('conv:', now_conv)

        NC.update_coverage_step(x)
        # print('conv:', now_conv)
        # logging.info('conv:', now_conv)

        json_data = {}
        if 'coverage_test_yz' not in json_data.keys():
            json_data['coverage_test_yz'] = {}
        json_data['coverage_test_yz']['coverage_neural'] = []
        for idx, conv in saves:
            json_data['coverage_test_yz']['coverage_neural'].append([conv, f'image_neural/{((idx+1)*2)}.svg']) 

        if i == number_of_image:
            break
        

        # logging.info("[模型测试阶段] 运行课题二的模型标准化测试准则：coverage_visualize [{:d}/{:d}]".format(i, len(dataloader)))
        # if log_func is not None:
        #     log_func(info)
    return json_data
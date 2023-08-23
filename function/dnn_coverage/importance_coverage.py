import numpy as np
import torch.utils.data
from torch.utils.data import DataLoader
from torch.nn import Module
from sklearn import cluster
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from matplotlib import pyplot as plt
import json
import pyecharts.options as opts
from pyecharts.charts import Timeline, Bar, Line
import os
import torchvision
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose, Normalize
from torch.utils.data import DataLoader, dataloader
import torch
import torch.nn.functional as F
import torch.nn as nn

def get_imp_neus(net, n_imp, layer_names, trainloader: DataLoader, result_file, pic_savepath, visualization=True):
    last_layer = net._modules[layer_names[-1]]
    activation = {}
    #net._modules[layer_names[-1]].register_forward_hook(get_activation(activation, layer_names[-1]))
    if isinstance(last_layer, torch.nn.Sequential):
        layer_names = list(last_layer._modules)
        last_layer = last_layer._modules[layer_names[-1]]
    last_layer.register_forward_hook(get_activation(activation, layer_names[-1]))

    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    net.to(device)
    weights = last_layer.weight.data.to(device)
    n_neurons = weights.shape[1]
    rels = torch.zeros(n_neurons).to(device)
    iternum = len(trainloader)
    # print(iternum)
    drawlist = [int(iternum * i / 4) for i in range(4)]
    drawlist.append(iternum - 1)
    # print(drawlist)
    reldraw = {}
    pos = 0
    name_list = [i for i in range(n_neurons)]
    with torch.no_grad():
        for iters, (inputs, labels) in enumerate(trainloader):
            inputs = inputs.to(device)
            output = net(inputs).data.to(device)  # batch_size,layer[-1].neurons
            output2 = activation[layer_names[-1]].to(device)  # batch_size,layer[-2].neurons
            for i in range(weights.shape[0]):
                temp = torch.sum(torch.abs(weights[i] * output2), 1)
                rel = (abs(weights[i] * output2).T * output[:, i]) / temp
                rels += torch.sum(rel, 1)
            if visualization:
                if iters in drawlist:
                    reldraw[pos] = list(torch.sqrt(torch.abs(rels)).detach().cpu().numpy())
                    reldraw[pos] = [round(float(reldraw[pos][j]), 7) for j in range(len(reldraw[pos]))]
                    pos = pos + 1

            del output, output2, temp, rel
            torch.cuda.empty_cache()
    timeline = Timeline(init_opts=opts.InitOpts(width="1600px", height="800px"))
    # print(reldraw)
    # for y in range(5):
    #    bar = (
    #        Bar()
    #        .add_xaxis(name_list)
    #        .add_yaxis(series_name='神经元累计重要性', y_axis=reldraw[y], label_opts=opts.LabelOpts(is_show=False))
    #    )
    #    timeline.add(bar, time_point=str(y))

    # timeline.add_schema(is_auto_play=True, play_interval=1000)
    # timeline.render(os.path.join(pic_savepath,'importance.html'))
    with open(result_file, 'r') as file_obj:
        json_data = json.load(file_obj)

    with open(result_file, 'w') as f:
        json_data['ImportanceNeuronsCoverage']['reldraw'] = {}
        json_data['ImportanceNeuronsCoverage']['reldraw']['0'] = reldraw[0]
        json_data['ImportanceNeuronsCoverage']['reldraw']['1'] = reldraw[1]
        json_data['ImportanceNeuronsCoverage']['reldraw']['2'] = reldraw[2]
        json_data['ImportanceNeuronsCoverage']['reldraw']['3'] = reldraw[3]
        json_data['ImportanceNeuronsCoverage']['reldraw']['4'] = reldraw[4]
        f.write(json.dumps(json_data, ensure_ascii=False, indent=4, separators=(',', ':')))

    _, idx = torch.sort(rels)
    idx = idx[-n_imp:]
    print('Top ' + str(n_imp) + ' Important Neurons: ' + str(idx))
    return rels, idx


def get_activation(activation, name):
    def hook(model, input, output):
        # 如果你想feature的梯度能反向传播，那么去掉 detach（）
        activation[name] = input[0]

    return hook


def get_actvalue_for_inp_neus(net, imp_neus, layer_names, trainloader):
    activation = {}
    #net._modules[layer_names[-1]].register_forward_hook(get_activation(activation, layer_names[-1]))
    last_layer = net._modules[layer_names[-1]]
    if isinstance(last_layer, torch.nn.Sequential):
        layer_names = list(last_layer._modules)
        last_layer = last_layer._modules[layer_names[-1]]
    activation = {}
    last_layer.register_forward_hook(get_activation(activation, layer_names[-1]))
    actvalue = []

    actvalue = []
    for id in range(len(imp_neus)):
        actvalue.append(torch.zeros(0))
    device = torch.device("cpu")
    with torch.no_grad():

        for iters, (inputs, labels) in enumerate(trainloader):
            # (str(iters))
            inputs = inputs.to(device)
            output = net(inputs)
            output2 = activation[layer_names[-1]].T  # layer[-2].neurons,batch_size
            for i, id in enumerate(imp_neus):
                id = id.item()
                actvalue[i] = torch.cat([actvalue[i], output2[id].to("cpu")])
            del output, output2, inputs
            torch.cuda.empty_cache()
    for i in range(len(imp_neus)):
        actvalue[i] = actvalue[i].detach().numpy()
    return actvalue


def quantizeSilhouette(actvalue, imp_neus):
    clusterresult = []
    kmeanmodels = []
    for i in range(len(actvalue)):
        clusterdicts = {}
        for size in range(5, 6):
            print('No.' + str(i + 1) + 'Important Neuron, cluster_size=' + str(size))
            kmeans = cluster.KMeans(n_clusters=size)
            clusterLabels = kmeans.fit_predict(actvalue[i].reshape(-1, 1))
            silhouetteAvg = silhouette_score(actvalue[i].reshape(-1, 1), clusterLabels)
            print('Silhouette Score:' + str(silhouetteAvg))
            CHindex = calinski_harabasz_score(actvalue[i].reshape(-1, 1), clusterLabels)
            print('CHindex:' + str(CHindex))  # Calinski-Harabaz Index
            clusterdicts[silhouetteAvg] = kmeans

        maxSilhouetteScore = max(clusterdicts.keys())
        bestKMean = clusterdicts[maxSilhouetteScore]
        kmeanmodels.append(bestKMean)
        values = bestKMean.cluster_centers_.squeeze()
        clusterresult.append(values)

    return clusterresult, kmeanmodels


def get_clustermodel_directly(actvalue, k):
    clusterresult = []
    kmeanmodels = []
    for i in range(len(actvalue)):
        kmeans = cluster.KMeans(n_clusters=k)
        clusterLabels = kmeans.fit_predict(actvalue[i].reshape(-1, 1))
        kmeanmodels.append(kmeans)
        values = kmeans.cluster_centers_.squeeze()
        clusterresult.append(values)

    return clusterresult, kmeanmodels


def Cover(INCC, net, testloader, imp_neus, layer_names, kmeans):
    acts_test = get_actvalue_for_inp_neus(net, imp_neus, layer_names, testloader)

    for i in range(len(imp_neus)):
        acts_test[i] = kmeans[i].fit_predict(acts_test[i].reshape(-1, 1))

    coverage_data = np.array(acts_test).T  # test_cases,imp_neus
    Coverage_history = np.zeros((acts_test[0].size))
    n_incc = INCC.size
    for i in range(len(coverage_data)):
        INCC[tuple(coverage_data[i])] = 1
        Coverage_history[i] = np.sum(INCC) / n_incc

    return Coverage_history


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

def run_importance_coverage(net: Module, trainloader: DataLoader, testloader: DataLoader, n_imp: int, clus: int, pic_savepath, filename, log_func=None):
    net.eval()
    for m in net.parameters():
        m.requires_grad = False
    layer_names = list(net._modules)
    rels, imp_neus = get_imp_neus(net, n_imp, layer_names, trainloader, filename, pic_savepath)
    actvalue = get_actvalue_for_inp_neus(net, imp_neus, layer_names, trainloader)

    clusterresult, kmeansmodels = get_clustermodel_directly(actvalue, clus)
    dimlist = [neu.size for neu in clusterresult]
    INCC_space = np.zeros(dimlist)

    Coverage = Cover(INCC_space, net, testloader, imp_neus, layer_names, kmeansmodels)
    #print('Coverage = ' + str(Coverage[-1]))
    Coverage = list(Coverage)
    Coverage = [round(float(Coverage[j]), 2) for j in range(len(Coverage))]
    x_data = [x for x in range(len(Coverage))]
    (
        Line()
            .set_global_opts(
            tooltip_opts=opts.TooltipOpts(is_show=False),
            xaxis_opts=opts.AxisOpts(type_="category"),
            yaxis_opts=opts.AxisOpts(
                type_="value",
                axistick_opts=opts.AxisTickOpts(is_show=True),
                splitline_opts=opts.SplitLineOpts(is_show=True),
            ),
        )
            .add_xaxis(xaxis_data=x_data)
            .add_yaxis(
            series_name="",
            y_axis=Coverage,
            is_smooth=True,
            is_symbol_show=False,
            label_opts=opts.LabelOpts(is_show=False),
        )
            .render(os.path.join(pic_savepath, 'ImpCoverage.html'))
    )

    with open(filename, 'r') as file_obj:
        json_data = json.load(file_obj)
    with open(filename, 'w') as f:
        json_data['ImportanceNeuronsCoverage']['Coverage'] = Coverage[-1]
        f.write(json.dumps(json_data, ensure_ascii=False, indent=4, separators=(',', ':')))





import os.path as osp
def run(model, train_loader, test_loader, params, log_func=None):
    root = osp.join(params["out_path"], "keti2")
    result_file = osp.join(root, "keti2.json")
    n_imp = params["coverage"]["n_imp"]
    clus = params["coverage"]["clus"]

    lenet_flag = True if 'lenet' in params["coverage"].keys() and eval(params["coverage"]['lenet']) else False
    # 目录需要更改
    lenet_path = osp.join(os.path.dirname(__file__),'LeNet/model.pth')
    if lenet_flag:
        model=LeNet5()
        model.load_state_dict(torch.load(lenet_path))
        #model = torch.load(lenet_path)
    
    print(lenet_flag)
    run_importance_coverage(model, trainloader=train_loader, testloader=test_loader, n_imp=n_imp, clus=clus, pic_savepath=root, filename=result_file, log_func=log_func)
from deeplift import *
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import vgg  #需要修改！vgg是上个系统中集成的文件，需要合适的地址import
import copy
import numpy as np
from PIL import Image

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
    
    
def load_data(dataset):
    #示例，如果有集成好的load_data函数，此处需要输入数据集类别，返回train_loader test_loader和与单张图片同维度的全零向量ref_input
    if dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),  # 转为Tensor
            transforms.Normalize((0.1307,), (0.3081,)),  # 归一化
            ])
        train_dataset = torchvision.datasets.MNIST(root='./mnist', train=True, transform=transform, download=True)
        #test_dataset = torchvision.datasets.MNIST(root='./mnist', train=True, transform=transform, download=True)
        ref_input = torch.zeros((1, 1, 28, 28))        

    elif dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),  # 转为Tensor
            #transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))  # 归一化
            transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
             ])
        train_dataset = torchvision.datasets.CIFAR10(root='./cf10', train=True, transform=transform, download=True)
        #test_dataset = torchvision.datasets.CIFAR10(root='./cf10', train=False, transform=transform, download=True)
        ref_input = torch.zeros((1, 3, 32, 32))
    
    ref_input = ref_input.to(device)
    batch_size = 1
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    return trainloader, ref_input


def load_model_from_path(path, modelname):
    
    if modelname == 'LeNet':
        net = Lenet5()
        
        
    #需要修改！vgg是上个系统中集成的文件，需要import
    elif modelname == 'VGG11':
        net = vgg.vgg11()
    elif modelname == 'VGG13':
        net = vgg.vgg13()
    elif modelname == 'VGG16':
        net = vgg.vgg16()
    elif modelname == 'VGG19':
        net = vgg.vgg19()        
    else:
        raise ValueError('仅支持LeNet，VGG11，VGG13与VGG19')
        
    net.load_state_dict(torch.load(path))
    net = net.to(device)
    return net


def initialization(net, ref_input):
    
    for layer in list(net._modules):
        layername = net._modules[layer]._get_name()
        if 'Conv' in layername or 'Linear' in layername:
            first_layer = layer
            break
        elif 'Sequential' in layername:
            first_layer = net._modules[layer]._moudles[list(net._modules[layer]._moudles[0])]
            break
    
    testx = torch.randn_like(ref_input)
    activation = {}
    hook_reg(net, activation, ref_input)
    testy,_ = sample_Contribute(net,activation,testx)
    mpp = torch.zeros_like(activation[first_layer + '_mpp']).cpu()
    mnn = torch.zeros_like(activation[first_layer + '_mnn']).cpu()
    activation.clear()
    net.eval()
    return activation, net, mpp, mnn, first_layer


def get_sensitive_neurons(net, activation, mpp, mnn, first_layer, trainloader, save_dir):
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(trainloader):
            activation.clear()
            gc.collect()
            sample_Contribute(net,activation,inputs.to(device))
            mpp += activation[first_layer + '_mpp'].cpu()
            mnn += activation[first_layer + '_mpp'].cpu()
            torch.cuda.empty_cache()

#             if i%10 == 9:
#                 print('now: sample '+str(i+1)+', '+str((time.time()-tm)/10)+'s per sample.')
#                 tm = time.time()
    if not os.path.exist(save_dir):
        os.mkdir(save_dir)
    np.save(os.path.join(save_dir,'mpp.npy'), mpp.detach().numpy())
    np.save(os.path.join(save_dir,'mnn.npy'), mnn.detach().numpy())
    return mpp, mnn


def getl(y):
    return np.argmax(y.cpu().detach().numpy())


def fuzz(net_unhook, dataset, trainloader, conv1mpp, conv1mnn, pertube, save_path):
    import time
    time = str(int(time.time()))
    rand_is_adv = 0
    if not os.path.exists(os.path.join(save_path,time)):
        os.mkdir(os.path.join(save_path,time))
    if dataset == 'mnist':
        mpp = np.abs(torch.sum(torch.tensor(conv1mpp),dim=1).reshape(28,28).detach().numpy())
        mnn = np.abs(torch.sum(torch.tensor(conv1mnn),dim=1).reshape(28,28).detach().numpy())
        pi_s,pj_s = np.where(mpp>=np.sort(mpp.reshape(-1))[int(np.round(mpp.size*(1-pertube)))])
        ni_s,nj_s = np.where(mnn>=np.sort(mnn.reshape(-1))[int(np.round(mnn.size*(1-pertube)))])
        pidxs = np.concatenate((pi_s.reshape(-1,1),pj_s.reshape(-1,1)),axis=1)
        nidxs = np.concatenate((ni_s.reshape(-1,1),nj_s.reshape(-1,1)),axis=1)
        for k in range(50000):
            (x,y)=iter(trainloader).next()
            x=x.to(device)
            noisesp = torch.randn(int(np.round(mpp.size*pertube)))*0.3081+0.1307
            noisesn = torch.randn(int(np.round(mpp.size*pertube)))*0.3081+0.1307
            x_p = x.clone()
            x_n = x.clone()
            for i in range(pidxs.shape[0]):    
                x_p[0,0,pidxs[i,0],pidxs[i,1]] = noisesp[i]
                x_n[0,0,nidxs[i,0],nidxs[i,1]] = noisesn[i]
            y_p = net_unhook(x_p)
            y_n = net_unhook(x_n)
            yori = net_unhook(x)
            labelp = getl(y_p)
            labeln = getl(y_n)
            labelori = getl(yori)
            
            if labelp != labelori:
                rand_is_adv += 1
                x_p = x_p.cpu().numpy()[0][0]
                x_p -= x_p.min()
                x_p *= (255/x_p.max())
                Image.fromarray(np.uint8(x_p)).save(os.path.join(save_path,time,str(rand_is_adv)+'.png'))
            if labeln != labelori:
                rand_is_adv += 1
                x_n = x_n.cpu().numpy()[0][0]
                x_n -= x_n.min()
                x_n *= (255/x_n.max())
                Image.fromarray(np.uint8(x_n)).save(os.path.join(save_path,time,str(rand_is_adv)+'.png'))
    else:
        raise Exception('当前只支持MNIST')
                          
    return rand_is_adv


def DeepSst(dataset, pertube, gpu, save_path=None, model=None, modelname=None, path=None, m_dir=None):
    #DeepSst支持所有顺序结构的CNN/全连接DNN
    
    #输入
    #dataset的加载由load_data函数控制，可按需调整
    #pertube控制敏感神经元的选取数量占输入层的比例，取值范围0-1，推荐0.05
    #save_path为输出文件夹
    #model的加载有两种方式，第一种直接传入model，第二种通过path加载，modelname只支持LeNet、VGG11、VGG13、VGG16、VGG19
    #m_dir为敏感神经元列表的加载目录，如果已经有输出的mpp与mnn，可以直接加载免去计算步骤
    
    
    #输出
    #rand_is_adv表示找到了多少个测试样本
    #测试样本保存在save_path中，其中mpp.npy mnn.npy记录了输入层神经元的敏感度，内部子文件夹中保存了所有导致错误的测试样本
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if save_path is None:
        save_path = '.'
    if model is None:
        if (modelname is not None) and (path is not None):
            model = load_model_from_path(path, modelname)
        else:
            raise Exception('模型加载失败')
    print('模型加载成功')
    
    model_cp = copy.deepcopy(model)
    trainloader, ref_input = load_data(dataset)
    
    if m_dir is None:
        activation, model_cp, mpp, mnn, first_layer = initialization(model_cp, ref_input)
        mpp, mnn = get_sensitive_neurons(model_cp, activation, mpp, mnn, first_layer, trainloader, save_path)
    else:
        mpp = np.load(os.path.join(m_dir,'mpp.npy'))
        mnn = np.load(os.path.join(m_dir,'mnn.npy'))
        print('已读取敏感度数据')
        
    rand_is_adv = fuzz(model, dataset, trainloader, mpp, mnn, pertube, save_path)
    print('共找到引发错误的测试样本 '+ str(rand_is_adv) + '个')    
        
        
        
    
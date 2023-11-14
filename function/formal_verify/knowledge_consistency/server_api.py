import torch.cuda as cuda
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as T
from torch.utils.data import DataLoader
import numpy as np
from dc_util import *
from Datasets import Generate_Dataloader
import os
from torch.nn.modules.loss import _Loss
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import PIL.Image as Image
import Model_zoo as models
import datetime
from PIL import Image
import matplotlib.colors as colors
import sqlite3
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor()
device_ids=[0,1]

def load_checkpoint(resume, model):
    if os.path.isfile(resume):
        # print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume, map_location=torch.device("cuda:{}".format(device_ids[0])))
        state_dict = checkpoint['state_dict']
        keys = list(state_dict.keys())
        for key in keys:
            if key.find('module') >= 0:
                state_dict[key.replace('module.','')] = state_dict.pop(key)

        model.load_state_dict(state_dict)
        # print("=> loaded checkpoint '{}' (epoch {} acc1 {})"
        #       .format(resume, checkpoint['epoch'], checkpoint['best_acc1']))
    else:
        print("=> no checkpoint found at '{}'".format(resume))

def ResBlock_beforeReLU(block, x): # Only for ResNet 50 101 152!!
    identity = x

    out = block.conv1(x)
    out = block.bn1(out)
    out = block.relu(out)

    out = block.conv2(out)
    out = block.bn2(out)
    if block.__class__.__name__ != 'BasicBlock':
        out = block.relu(out)

        out = block.conv3(out)
        out = block.bn3(out)

    if block.downsample is not None:
        identity = block.downsample(x)

    out += identity
    return out

def get_feature(img, net, arch, conv_layer):
    input = img.unsqueeze(0)
    net.eval()
    with torch.no_grad():
        if arch.startswith("alexnet"):
            x = net.features[:conv_layer + 1](input)

        elif arch.startswith("vgg"):
            x = net.features[:conv_layer + 1](input)


        elif arch.startswith("resnet"):
            x = input
            x = net.conv1(x)
            x = net.bn1(x)
            x = net.relu(x)
            x = net.maxpool(x)

            x = net.layer1(x)
            x = net.layer2(x)
            x = net.layer3[:-1](x)
            x = ResBlock_beforeReLU(net.layer3[-1], x)

        return x.squeeze()
def knowledge_consistency(arch,dataset,img_path,base):
    base_path=os.path.join(base,'model_checkpoints')
    if arch=='vgg16_bn' and dataset=='mnist':
        conv_layer=30
        model_path1=os.path.join(base_path,'checkpoint_mnist_vgg16_bn_lr-2_sd0.pth.tar')
        model_path2=os.path.join(base_path,'checkpoint_mnist_vgg16_bn_lr-2_sd5.pth.tar')
        resume=os.path.join(base_path,f'{dataset}_{arch}','checkpoint_L30__a0.1_lr-4.pth.tar')
    elif arch=='vgg16_bn' and dataset=='cifar10':
        conv_layer=30
        model_path1=os.path.join(base_path,'checkpoint_cifar10_vgg16_bn_lr-2_sd0.pth.tar')
        model_path2=os.path.join(base_path,'checkpoint_cifar10_vgg16_bn_lr-2_sd5.pth.tar')
        resume=os.path.join(base_path,f'{dataset}_{arch}','checkpoint_L30__a0.1_lr-4.pth.tar')
    elif arch=='resnet18' and dataset=='mnist':
        conv_layer=3
        model_path1=os.path.join(base_path,'checkpoint_mnist_resnet18_lr-2_sd0.pth.tar')
        model_path2=os.path.join(base_path,'checkpoint_mnist_resnet18_lr-2_sd5.pth.tar')
        resume=os.path.join(base_path,f'{dataset}_{arch}','checkpoint_L3__a0.1_lr-4.pth.tar')
    elif arch=='resnet18' and dataset=='cifar10':
        conv_layer=3
        model_path1=os.path.join(base_path,'checkpoint_cifar10_resnet18_lr-2_sd0.pth.tar')
        model_path2=os.path.join(base_path,'checkpoint_cifar10_resnet18_lr-2_sd5.pth.tar')
        resume=os.path.join(base_path,f'{dataset}_{arch}','checkpoint_L3__a0.1_lr-3.pth.tar')
    else:
        print(arch,dataset)
        return None
    if dataset=='mnist':
        transform=T.Compose([T.Resize((224,224)),T.Grayscale(num_output_channels=3),
            T.ToTensor(), 
            T.Normalize((0.1307,), (0.3081,))])
    elif dataset=='cifar10':
        transform=T.Compose([T.Resize((224,224)),T.ToTensor(), 
                                  T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])
    
    
    net1 = models.__dict__[arch](num_classes=10)
    load_checkpoint(model_path1, net1)
    net2 = models.__dict__[arch](num_classes=10)
    # load_checkpoint(model_path2, net2)
    img = Image.open(img_path)
    img = img.convert('RGB')

    x_ori = transform(img)
    y = net1(x_ori.unsqueeze(0))
    y = np.argmax(y.cpu().detach().numpy())
    print(y)
    x = get_feature(x_ori, net1,arch,conv_layer)
    input_size = x.shape
    output_size = x.shape
    
    gpu=1
    model = models.LinearTester(input_size,output_size, gpu_id= gpu, affine=False, bn = False, instance_bn=True).cuda(gpu)
    checkpoint = torch.load(resume, map_location=torch.device("cuda:{}".format(gpu)))
    model.load_state_dict(checkpoint['state_dict'])
    del checkpoint
    input = get_feature(x_ori,net1,arch,conv_layer)
    target = get_feature(x_ori,net2,arch,conv_layer)

    model.eval()
    output, output_n, output_contrib, res = model.val_linearity(input.unsqueeze(0).cuda(gpu))
    t=0
    img_name=os.path.basename(img_path).split('.')[0]
    executor.submit(compute_val,arch,arch,model_path1,model_path2,resume,dataset,conv_layer,img_name,y)
    conn  = sqlite3.connect("./history.db")
    c = conn.cursor()
    sql="""INSERT INTO task (id, img_name , arch_in , arch_tar , dataset , status ,l2, var , sub_time, end_time)
    values(null,'{img_name}','{arch_in}','{arch_tar}','{dataset}',{status},{l2},{var},'{sub_time}','{sub_time}')
    """
    sub_time=datetime.datetime.now()
    print(sub_time)
    sql_query = sql.format(img_name=img_name, 
            arch_in=arch,
            arch_tar=arch,
            dataset=dataset,
            status=0,
            l2=0,
            var=0,
            sub_time=sub_time

            )
    print(sql_query)
    c.execute(sql_query)
    print(c.fetchall())
    conn.commit()
    conn.close()
    

    
    plt.figure(frameon=False)
    plt.imshow(output[t], cmap='jet', norm=None, vmin=output.min(), vmax=output.max())
    plt.axis('off')
    plt.savefig(f'static/gen_imgs/{os.path.basename(img_path).split(".")[0]}_{arch}_{dataset}_output.png',bbox_inches='tight')
    plt.imshow(input[t], cmap='jet', norm=None, vmin=input.min(), vmax=input.max())
    plt.savefig(f'static/gen_imgs/{os.path.basename(img_path).split(".")[0]}_{arch}_{dataset}_input.png',bbox_inches='tight')
    plt.imshow(target[t], cmap='jet', norm=None, vmin=target.min(), vmax=target.max())
    plt.savefig(f'static/gen_imgs/{os.path.basename(img_path).split(".")[0]}_{arch}_{dataset}_target.png',bbox_inches='tight')
    delta = target - output
    plt.imshow(delta[t], cmap='jet', norm=None, vmin=delta.min(), vmax=delta.max())
    plt.savefig(f'static/gen_imgs/{os.path.basename(img_path).split(".")[0]}_{arch}_{dataset}_delta.png',bbox_inches='tight')
    return torch.norm(delta, p=2).numpy().tolist()
def get_variance(net, transNet, net2, dataloader,conv_layer,arch_in,arch_tar,y=0,gpu=3):
    meter = AverageMeter()
    net.eval()
    transNet.eval()
    num_batches =  len(dataloader)

    with torch.no_grad():
        for i, (datas, target) in tqdm(enumerate(dataloader)):
            batch_size = target.size(0)
            x = conv_feature(datas,net, conv_layer,arch_in, gpu)
            out, out_n = transNet.val_batch(x)
            mean_batch = out_n.mean([1,3,4])
            meter.update(mean_batch, batch_size)
        
        Mean_all = meter.avg.reshape(3,1,-1,1,1)
        
        meter.reset()
        for i, (datas, target) in tqdm(enumerate(dataloader)):
            batch_size = target.size(0)
            x = conv_feature(datas,net, conv_layer,arch_in, gpu)
            out, out_n = transNet.val_batch(x)
            var_batch = ((out_n-Mean_all)**2).mean([1,2,3,4])
            meter.update(var_batch, batch_size)  
        
        Mean_all = Mean_all.reshape(3,-1)
        Var_all = meter.avg
        
        print(Var_all)
        meter.reset()
        for i, (datas, target) in tqdm(enumerate(dataloader)):
            batch_size = target.size(0)
            t = conv_feature(datas, net2, conv_layer,arch_tar, gpu)
            x = conv_feature(datas, net, conv_layer,arch_in, gpu)
            out, out_n = transNet.val_batch(x)
            diff = t - out
            d_mean = diff.mean([0,2,3])
            meter.update(d_mean, batch_size)
        diff_mean = meter.avg.reshape(1, -1, 1, 1)
        
        meter.reset()
        for i, (datas, target) in tqdm(enumerate(dataloader)):
            t = conv_feature(datas, net2, conv_layer,arch_tar, gpu)
            x = conv_feature(datas, net, conv_layer,arch_in, gpu)
            out, out_n = transNet.val_batch(x)
            diff = t - out
            diff_var = ((diff-diff_mean)**2).mean([0,1,2,3])
            meter.update(diff_var, batch_size) 
        diff_var = meter.avg
        
        return Mean_all, Var_all, diff_var
def compute_val(arch_in,arch_tar,net_in_path,net_tar_path,transnet,dataset,conv_layer,img_name='',y=0):

    
    if arch_in.startswith("vgg16_bn"):
        channels = 512
        kernel_size = 28 if conv_layer<=30 else 14
    elif arch_in.startswith("alexnet"):
        channels = 256
        kernel_size = 13
    elif arch_in.startswith('resnet'):
        if arch_in.startswith('resnet18') or arch_in.startswith('resnet34'):
            if conv_layer == 3:
                channels = 256
                kernel_size = 14
            elif conv_layer == 4:
                channels = 512
                kernel_size = 7
        else:
            if conv_layer == 3:
                channels = 1024
                kernel_size = 14
            elif conv_layer == 4:
                channels = 2048
                kernel_size = 7
    if dataset.startswith("VOC"):
        net_in = models.__dict__[arch_in](num_classes=20)
        net_tar = models.__dict__[arch_tar](num_classes=20)
    elif dataset.startswith("CUB"):
        net_in = models.__dict__[arch_in](num_classes=200)
        net_tar = models.__dict__[arch_tar](num_classes=200)
    elif dataset.startswith("DOG"):
        net_in = models.__dict__[arch_in](num_classes=200)
        net_tar = models.__dict__[arch_tar](num_classes=200)
    elif dataset.startswith("mnist") or dataset.startswith("cifar10"):
        net_in = models.__dict__[arch_in](num_classes=10)
        net_tar = models.__dict__[arch_tar](num_classes=10)
    if arch_in.endswith('DC'):
        net_in = load_masked_model_as_original_model(net_in_path)
    else:
        load_checkpoint(net_in_path, net_in)
        pass

    # if arch_tar.endswith('DC'):
    #     net_tar = load_masked_model_as_original_model(net_tar_path)
    # else:
    #     load_checkpoint(net_tar_path, net_tar)

    if arch_in.startswith("vgg"):
        if conv_layer <= 30:
            input_size = output_size = torch.zeros((512, 28, 28)).shape
        else:
            input_size = output_size = torch.zeros((512, 14, 14)).shape
    elif arch_in.startswith("alexnet"):
        input_size = output_size = torch.zeros((256, 13, 13)).shape
    elif arch_in.startswith("resnet"):
        if arch_in.startswith('resnet18') or arch_in.startswith('resnet34'):
            input_size = output_size = torch.zeros((256, 14, 14)).shape   
        else:
            input_size = output_size = torch.zeros((1024, 14, 14)).shape
    gpu=3
    model_Ys = models.LinearTester(input_size, output_size, gpu_id=gpu, fix_p=True, bn=False, instance_bn=True)
    load_checkpoint(transnet, model_Ys)
    model_Ys = model_Ys.cpu()
    net_in.cuda(gpu)
    net_tar.cuda(gpu)
    model_Ys.cuda(gpu)
    train_loader, val_loader = \
        Generate_Dataloader(dataset, 128, 2,
                        '', '',onlyone=y)

    Mean_all, Var_all, diff_var=(get_variance(net_in, model_Ys, net_tar, val_loader,conv_layer,arch_in,arch_tar,y,gpu))
    print(Mean_all)
    var=diff_var/Var_all.sum()
    var=var.cpu().numpy().tolist()
    print(type(var))
    end_time=datetime.datetime.now()
    sql="""
    update task set var={var}, status=1,end_time='{end_time}'
    where img_name='{img_name}' and arch_in='{arch_in}' and arch_tar='{arch_tar}' and dataset='{dataset}'
    """
    conn  = sqlite3.connect("./history.db")
    c = conn.cursor()

    sub_time=datetime.datetime.now()
    sql_query = sql.format(var=var,img_name=img_name, 
            arch_in=arch_in,
            arch_tar=arch_tar,
            dataset=dataset,
            end_time=end_time
            )
    print(sql_query)
    c.execute(sql_query)

    conn.commit()
    conn.close()

if __name__ == '__main__':
    conv_layer=30
    arch_in='vgg16_bn'
    arch_tar='vgg16_bn'
    net_in_path='model_checkpoints/checkpoint_mnist_vgg16_bn_lr-2_sd0.pth.tar'
    net_tar_path='model_checkpoints/checkpoint_mnist_vgg16_bn_lr-2_sd5.pth.tar'
    transnet='model_checkpoints/mnist_vgg16_bn/checkpoint_L30__a0.1_lr-4.pth.tar'
    dataset='mnist'
    compute_val(arch_in,arch_tar,net_in_path,net_tar_path,transnet,dataset,conv_layer)
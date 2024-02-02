from .model_VGG import VGG
from .model_Resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .model_Densenet import densenet121, densenet161, densenet169, densenet201
from .model_Lenet import lenet
from .train_model import Train

import torch
import torchvision.models as models
import os.path as osp

import textattack

def load_model(model_name, dataset, device, root, reference_model, logging, pretrained=True, attribution=True):
    if dataset == "imagenet":
        output_size = 7
        input_channel = 3
        num_classes = 1000
    elif dataset == "cifar10":
        output_size = 1
        input_channel = 3
        num_classes = 10
    elif dataset == "mnist":
        output_size = 1
        input_channel = 1
        num_classes = 10
    else:
        raise ValueError("暂不支持{:s}数据集".format(dataset))
    
    model = None
    model_name = model_name.lower()
    
    # mnist 和 cifar10需要重新训练
    if reference_model == None and (dataset == "mnist" or dataset == "cifar10"):
        pretrained = False 
    
    if model_name == 'vgg11':
        model = VGG().forward("A", pretrained=pretrained, reference_model=reference_model, num_classes=num_classes,
                  output_size=output_size, input_channel=input_channel)
    elif model_name == 'vgg13':
        model = VGG().forward("B", pretrained=pretrained, reference_model=reference_model, num_classes=num_classes,
                  output_size=output_size, input_channel=input_channel)
    elif model_name == 'vgg16':
        model = VGG().forward("D", pretrained=pretrained, reference_model=reference_model, num_classes=num_classes,
                  output_size=output_size, input_channel=input_channel)
    elif model_name == 'vgg19':
        model = VGG().forward("E", pretrained=pretrained, reference_model=reference_model, num_classes=num_classes,
                  output_size=output_size, input_channel=input_channel)
    elif model_name == "resnet18":
        model = resnet18(pretrained=pretrained,
                       reference_model=reference_model, num_classes=num_classes, input_channel=input_channel)
    elif model_name == 'resnet34':
        model = resnet34(pretrained=pretrained,
                       reference_model=reference_model, num_classes=num_classes, input_channel=input_channel)
    elif model_name == 'resnet50':
        model = resnet50(pretrained=pretrained,
                       reference_model=reference_model, num_classes=num_classes, input_channel=input_channel)
    elif model_name == 'resnet101':
        model = resnet101(pretrained=pretrained,
                        reference_model=reference_model, num_classes=num_classes, input_channel=input_channel)
    elif model_name == 'resnet152':
        model = resnet152(pretrained=pretrained,
                        reference_model=reference_model, num_classes=num_classes, input_channel=input_channel)
    elif model_name == 'densenet121':
        model = densenet121(
            pretrained=pretrained, reference_model=reference_model, num_classes=num_classes, input_channel=input_channel)
    elif model_name == 'densenet169':
        model = densenet169(
            pretrained=pretrained, reference_model=reference_model, num_classes=num_classes,input_channel=input_channel)
    elif model_name == 'densenet201':
        model = densenet201(
            pretrained=pretrained, reference_model=reference_model, num_classes=num_classes,input_channel=input_channel)
    elif model_name == 'densenet161':
        model = densenet161(
            pretrained=pretrained, reference_model=reference_model, num_classes=num_classes,input_channel=input_channel)
    elif model_name == 'lenet':
        model = lenet(reference_model=reference_model, input_channel=input_channel)
    else:
        raise NotImplementedError("暂不支持{:s}模型".format(model_name))
    
    if pretrained == False:
        save_root = osp.join(root,f"model/ckpt")
        path = osp.join(save_root, "{:s}_{:s}.pt".format(dataset, model_name))
        if not osp.exists(path):
            # logging.info("找不到缓存区中的{:s}数据集上训练的{:s}模型，重新进行训练".format(dataset, model_name))
            print("si")
            acc, model = Train(model_name, model, dataset, device)
            # logging.info("{:s}模型训练完成，在{:s}数据集上的准确率为{:.2f}".format(model_name, dataset, acc*100))
            model.to(device="cpu")
            torch.save(model.state_dict(), path)
        else:
            logging.info("加载到缓存区中的{:s}数据集上训练的{:s}模型".format(dataset, model_name))
            model.load_state_dict(torch.load(path))
    return model

def load_torch_model(model_name):
    model = None
    if model_name == "vgg16":
        model = models.vgg16(pretrained=True)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=True)
    elif model_name == "densenet121":
        model = models.densenet121(pretrained=True)
    elif model_name == "mobilenet":
        model = models.mobilenet_v2(pretrained=True)
    elif model_name == "alexnet":
        model = models.alexnet(pretrained=True)
    elif model_name == "inception":
        model = models.inception_v3(pretrained=True)
    elif model_name == "efficientnet":
        model = models.efficientnet_b0(pretrained=True)
    else:
        raise NotImplementedError("暂不支持{:s}模型".format(model_name))
    return model

def load_text_model(model_name):
    model = None
    if model_name == "lstm":
        model = textattack.models.helpers.LSTMForClassification.from_pretrained("lstm-sst2")
    elif model_name == "wordcnn":
        model = textattack.models.helpers.WordCNNForClassification.from_pretrained("cnn-sst2")
    else:
        raise NotImplementedError("暂不支持{:s}模型".format(model_name))
    return model
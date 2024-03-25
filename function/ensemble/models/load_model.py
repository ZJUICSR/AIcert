# from model_VGG import VGG
from function.ensemble.models.resnet_attack import ResNet18
import torch
import os.path as osp
# from torchvision.models.resnet import resnet18
import os

def load_model():
    model = ResNet18(1)
    # model_info = torch.load(os.path.join(os.path.dirname(__file__), 'mnist_resnet18.pt'))
    model_info = torch.load('/root/fairness/AI-Platform/model/ckpt/MNIST_resnet18.pth')
    # train_epoch = model_info['epoch']
    # model_param = model_info['model']
    # model.load_state_dict(model_param)
    model.load_state_dict(model_info)
    return model


if __name__ == '__main__':
    model = load_model()
    print(model)
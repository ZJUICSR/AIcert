# -*- coding: utf-8 -*-
from torchvision.models import resnet50
import torch
import os


def get_resnet50_model(pretrain=False, device='cpu'):
    net = resnet50(pretrained=pretrain)
    net.fc = torch.nn.Linear(in_features=2048, out_features=2, bias=True)
    model_dir = r'resnet.pkl'

    if os.path.exists(model_dir):
        check_point = torch.load(model_dir)
        net.load_state_dict(check_point['model'])
        print(f'load over')
    net.to(device)
    return net


if __name__ == '__main__':
    model = get_resnet50_model()


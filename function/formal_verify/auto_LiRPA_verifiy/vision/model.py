from function.formal_verify.auto_LiRPA_verifiy.vision.cnn import mnist_model, mnist_model_with_different_activate_function
from function.formal_verify.auto_LiRPA_verifiy.vision.resnet import resnet18
from function.formal_verify.auto_LiRPA_verifiy.vision.densenet import Densenet_cifar_32
import torch
import os


def get_mnist_cnn_model(load=True, device='cpu', activate=True):
    model = mnist_model() if not activate else mnist_model_with_different_activate_function()
    if load:
        name = 'kw_mnist.pth' if not activate else 'cnn_act_func.pkl'
        name = os.path.join(os.path.dirname(__file__), name)
        checkpoint = torch.load(name, map_location=device)
        state = checkpoint if not activate else checkpoint['model']
        model.load_state_dict(state)
        model.to(device)

    return model


def get_cifar_resnet18(in_planes=2, load=True, device='cpu'):
    model = resnet18(in_planes=in_planes)
    if load and in_planes == 2:
        name = os.path.join(os.path.dirname(__file__), "resnet18_natural_cifar.pth")
        checkpoint = torch.load(name, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
    model.to(device)

    return model


def get_gtsrb_resnet18(in_planes=25, load=False, device='cpu'):
    model = resnet18(in_planes=in_planes)
    model.linear = torch.nn.Linear(in_features=in_planes * 8, out_features=43, bias=True)

    if load and in_planes == 25:
        name = os.path.join(os.path.dirname(__file__), "model_sss_in25.pth")
        checkpoint = torch.load(name, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
    model.to(device)

    return model


def get_MTFL_resnet18(in_planes=3, load=True, device='cpu'):
    model = resnet18(in_planes=in_planes)
    model.linear = torch.nn.Linear(in_features=in_planes * 8, out_features=2, bias=True)
    if load and in_planes == 3:
        name = os.path.join(os.path.dirname(__file__), "resnet18_mtfl_in3.pth")
        checkpoint = torch.load(name, map_location=device)
        model.load_state_dict(checkpoint['model'])
    model.to(device)

    return model


def get_cifar_densenet_model(load=True, device='cpu'):
    model = Densenet_cifar_32()
    if load:
        name = os.path.join(os.path.dirname(__file__), "model_densenet_cifar.pth")
        checkpoint = torch.load(name, map_location=device)
        model.load_state_dict(checkpoint['model'])
    model.to(device)
    return model


if __name__ == '__main__':
    model = get_gtsrb_resnet18()
    print(model)


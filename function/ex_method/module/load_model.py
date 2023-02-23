from .model_VGG import VGG
from .model_Resnet import resnet18, resnet34, resnet50, resnet101
from .model_Densenet import densenet121
from .model_mnist import mnist_self
from .model_cifar10 import cifar_self


def load_model(model, pretrained=True, reference_model=None, num_classes=1000):
    net = None
    output_size = 7 if num_classes == 1000 else 1
    model = model.lower()
    if model == 'vgg11':
        net = VGG().forward("A", pretrained=pretrained, reference_model=reference_model, num_classes=num_classes,
                            output_size=output_size)
    elif model == 'vgg13':
        net = VGG().forward("B", pretrained=pretrained, reference_model=reference_model, num_classes=num_classes,
                            output_size=output_size)
    elif model == 'vgg16':
        net = VGG().forward("D", pretrained=pretrained, reference_model=reference_model, num_classes=num_classes,
                            output_size=output_size)
    elif model == 'vgg19':
        net = VGG().forward("E", pretrained=pretrained, reference_model=reference_model, num_classes=num_classes,
                            output_size=output_size)
    elif model == 'resnet18':
        net = resnet18(pretrained=pretrained, reference_model=reference_model, num_classes=num_classes)
    elif model == 'resnet34':
        net = resnet34(pretrained=pretrained, reference_model=reference_model, num_classes=num_classes)
    elif model == 'resnet50':
        net = resnet50(pretrained=pretrained, reference_model=reference_model, num_classes=num_classes)
    elif model == 'resnet101':
        net = resnet101(pretrained=pretrained, reference_model=reference_model, num_classes=num_classes)
    elif model == 'densenet121':
        net = densenet121(pretrained=pretrained, reference_model=reference_model, num_classes=num_classes)
    elif model == 'cifar10':
        net = cifar_self(pretrained=pretrained, reference_model=reference_model)
    elif model == 'mnist':
        net = mnist_self(pretrained=pretrained, reference_model=reference_model)
    else:
        print("model not find!")
    return net

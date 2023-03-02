from .model_VGG import VGG
from .model_Resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .model_Densenet import densenet121, densenet161, densenet169, densenet201
from .model_CNN import cnn


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
    elif model == "resnet18":
        net = resnet18(pretrained=pretrained,
                       reference_model=reference_model, num_classes=num_classes)
    elif model == 'resnet34':
        net = resnet34(pretrained=pretrained,
                       reference_model=reference_model, num_classes=num_classes)
    elif model == 'resnet50':
        net = resnet50(pretrained=pretrained,
                       reference_model=reference_model, num_classes=num_classes)
    elif model == 'resnet101':
        net = resnet101(pretrained=pretrained,
                        reference_model=reference_model, num_classes=num_classes)
    elif model == 'resnet152':
        net = resnet152(pretrained=pretrained,
                        reference_model=reference_model, num_classes=num_classes)
    elif model == 'densenet121':
        net = densenet121(
            pretrained=pretrained, reference_model=reference_model, num_classes=num_classes)
    elif model == 'densenet169':
        net = densenet169(
            pretrained=pretrained, reference_model=reference_model, num_classes=num_classes)
    elif model == 'densenet201':
        net = densenet201(
            pretrained=pretrained, reference_model=reference_model, num_classes=num_classes)
    elif model == 'densenet161':
        net = densenet161(
            pretrained=pretrained, reference_model=reference_model, num_classes=num_classes)
    elif model == 'cnn':
        net = cnn(pretrained=pretrained, reference_model=reference_model)
    else:
        raise NotImplementedError("selected model is not supported!")
    return net

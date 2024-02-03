import Model_zoo as models
import torch

# create modelquick a qudef Generate_Model(dataset, arch, device_ids, train_layer,seed=None, pretrained = False):
def Generate_Model(dataset, arch, device_ids, train_layer,seed=None, pretrained = False):
    if seed is not None:
        torch.manual_seed(seed)

    if pretrained and dataset == 'ilsvrc':
        print("=> using pre-trained model '{}'".format(arch))
        model = models.__dict__[arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(arch))
        if dataset == 'cifar10':
            
            if arch.endswith('ft'):
                model = models.__dict__[arch](train_layer=train_layer, num_classes=10)
            else:
                model = models.__dict__[arch](num_classes=10)
        elif dataset == 'mnist':
            if arch.endswith('ft'):
                model = models.__dict__[arch](train_layer=train_layer, num_classes=10)
            else:
                model = models.__dict__[arch](num_classes=10)
        elif dataset == 'CUB':
            model = models.alexnet(num_classes = 2)
        elif dataset == 'CUB200':
            if arch.endswith('ft'):
                model = models.__dict__[arch](train_layer=train_layer, num_classes=200)
            else:
                model = models.__dict__[arch](num_classes=200)
        elif dataset == 'DOG120':
            if arch.endswith('ft'):
                model = models.__dict__[arch](train_layer=train_layer, num_classes=120)
            else:
                model = models.__dict__[arch](num_classes=120)
        elif dataset.startswith('VOC2012'):
            if arch.endswith('ft'):
                model = models.__dict__[arch](train_layer=train_layer, num_classes=20)
            else:
                model = models.__dict__[arch](num_classes=20)
        elif dataset.startswith('mix320'):
            if arch.endswith('ft'):
                model = models.__dict__[arch](train_layer=train_layer, num_classes=320)
            else:
                model = models.__dict__[arch](num_classes=320)
        else:
            model = models.__dict__[arch]()

        print('GPU used: ', device_ids)
        if arch.startswith('alexnet') or arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features, device_ids=device_ids)
            model.cuda(device_ids[0])
        else:
            model = torch.nn.DataParallel(model).cuda(device_ids[0])

    return model

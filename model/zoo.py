'''
About model zoo:
    We only provide models for imagenet/cifar/gtsrb/mnist
    You may load your private model using api in: models.loader.load_model_from_path
'''

import argp
import torch
import os.path as osp
from argp.models import cifar


def get_model(arch, task, pretrained=None, **kwargs):
    '''
    Get model from model zoo
    Args:
        arch: str, name of model file
        task: str, name of task
        pretrained: bool
        **kwargs:

    Returns:
        model: nn.Module
    '''
    task = "CIFAR" if ("CIFAR" in task.upper()) else task
    assert task in ('MNIST', 'CIFAR', 'Imagenet1k', 'GTSRB')

    if "num_classes" not in kwargs.keys():
        kwargs["num_classes"] = 10

    if pretrained and pretrained is not None:
        return get_pretrainednet(arch, task, pretrained, **kwargs)
    else:
        try:
            # This should have ideally worked
            model = eval("argp.models.{}.{}".format(task.lower(), arch.lower()))(**kwargs)
        except AssertionError:
            # But, there's a bug in pretrained models which ignores the num_classes attribute.
            # So, temporarily load the model and replace the last linear layer
            model = eval('argp.models.{}.{}'.format(task.lower(), arch.lower()))()
            if "num_classes" in kwargs:
                num_classes = kwargs["num_classes"]
                in_feat = model.last_linear.in_features
                model.last_linear = torch.nn.Linear(in_feat, num_classes)
        return model


def get_pretrainednet(arch, task, pretrained="imagenet", num_classes=10, **kwargs):
    if pretrained == "imagenet":
        return get_imagenet_pretrainednet(arch, num_classes, **kwargs)
    elif osp.exists(pretrained):
        try:
            # This should have ideally worked:
            model = eval("argp.models.{}.{}".format(task, arch))(num_classes=num_classes, **kwargs)
        except AssertionError:
            # But, there's a bug in pretrained models which ignores the num_classes attribute.
            # So, temporarily load the model and replace the last linear layer
            model = eval('argp.models.{}.{}'.format(task, arch))()
            in_feat = model.last_linear.in_features
            model.last_linear = torch.nn.Linear(in_feat, num_classes)
        checkpoint = torch.load(pretrained)
        pretrained_state_dict = checkpoint.get("state_dict", checkpoint)
        copy_weights_(pretrained_state_dict, model.state_dict())
        return model
    else:
        raise ValueError('Currently only supported for imagenet or existing pretrained models')


def get_imagenet_pretrainednet(arch, num_classes=1000, **kwargs):
    valid_models = argp.models.imagenet.__dict__.keys()
    assert arch in valid_models, 'Model not recognized, Supported models = {}'.format(valid_models)
    model = argp.models.imagenet.__dict__[arch](pretrained="imagenet")
    if num_classes != 1000:
        # Replace last linear layer
        in_features = model.last_linear.in_features
        out_features = num_classes
        model.last_linear = torch.nn.Linear(in_features, out_features, bias=True)
    return model


def copy_weights_(src_state_dict, dst_state_dict):
    n_params = len(src_state_dict)
    n_success, n_skipped, n_shape_mismatch = 0, 0, 0
    for i, (src_param_name, src_param) in enumerate(src_state_dict.items()):
        if src_param_name in dst_state_dict:
            dst_param = dst_state_dict[src_param_name]
            if dst_param.data.shape == src_param.data.shape:
                dst_param.data.copy_(src_param.data)
                n_success += 1
            else:
                print('Mismatch: {} ({} != {})'.format(src_param_name, dst_param.data.shape, src_param.data.shape))
                n_shape_mismatch += 1
        else:
            n_skipped += 1
    print('=> # Success param blocks loaded = {}/{}, '
          '# Skipped = {}, # Shape-mismatch = {}'.format(n_success, n_params, n_skipped, n_shape_mismatch))
from train import main
from certify import certify
from visualize import visualize
from predict import predict
args_dict = {
    'dataset': 'cifar10',
    'arch': 'cifar_resnet110',
    'outdir': '/data/user/WZT/models/smoothing/cifar10/resnet110/noise_0.00',
    'workers': 4,
    'epochs': 90,
    'batch': 256,
    'lr': 0.1,
    'lr_step_size': 30,
    'gamma': 0.1,
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'noise_sd': 0.0,
    'gpu': 0,
    'print-freq': 10
}

args_dict2 = {
    'dataset': 'cifar10',
    'base_classifier': "/data/user/WZT/models/smoothing/cifar10/resnet110/noise_0.00/checkpoint.pth.tar",
    'sigma': 0.25,
    'outfile': "/data/user/WZT/Datasets /smoothing/data/predict/cifar10/resnet110/noise_0.12/test/sigma_0.12",
    'batch': 1000,
    'skip': 1,
    'max': -1,
    'split': 'test',
    'N0': 100,
    'N': 100000,
    'alpha': 0.001
}
args_dict3 = {
    'dataset': 'cifar10',
    'outdir': '/data/user/WZT/Datasets/smoothing/figures/example_images/cifar10',
    'idx': 10,
    'noise_sds': 10,
    'split': 'test'
}
args_dict4 = {
    'dataset': 'imagenet',
    'base_classifier': "/data/user/WZT/models/smoothing//imagenet/resnet50/noise_0.25/checkpoint.pth.tar",
    'sigma': 0.25,
    'outfile': "/data/user/WZT/Datasets/smoothing/data/predict/imagenet/resnet50/noise_0.25/test/N_100",
    'batch': 1000,
    'skip': 1,
    'max': -1,
    'split': 'test',
    'N': 100000,
    'alpha': 0.001
}


main(args_dict)
# certify(args_dict2)
# predict(args_dict4)
# visualize(args_dict3)

# evaluate a smoothed classifier on a dataset
import argparse
import os
import setGPU
from datasets import get_dataset, DATASETS, get_num_classes
from core import Smooth
from time import time
import torch
import datetime
from architectures import get_architecture

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
args = parser.parse_args()


def certify(args_dict):
    # load the base classifier
    global prediction, correct
    checkpoint = torch.load(args_dict['base_classifier'])
    base_classifier = get_architecture(checkpoint["arch"], args_dict['dataset'])
    base_classifier.load_state_dict(checkpoint['state_dict'])

    # create the smooothed classifier g
    smoothed_classifier = Smooth(base_classifier, get_num_classes(args_dict['dataset']), args_dict['sigma'])

    # prepare output file
    f = open(args_dict['outfile'], 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    # iterate through the dataset
    dataset = get_dataset(args_dict['dataset'], args.split)
    for i in range(len(dataset)):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args_dict['skip'] != 0:
            continue
        if i == args.max:
            break

        (x, label) = dataset[i]

        before_time = time()
        # certify the prediction of g around x
        x = x.cuda()
        prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args_dict['batch'])
        after_time = time()
        correct = int(prediction == label)

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
            i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)

    f.close()
    return prediction, correct


if __name__ == "__main__":
    args_dict = {
        'dataset': 'cifar10',  # 数据集
        'base_classifier': '/data/user/WZT/models/smoothing/cifar10/resnet110/noise_0.00/checkpoint.pth.tar',  # 模型文件
        'sigma': 0.50,  # 噪声水平
        'outfile': '/data/user/WZT/models/smoothing/certification_output/result',  # 输出g对一堆输入进行预测的结果文件
        'batch': 400,  # 训练批次大小
        'skip': 100,  # 每隔一百个图像
    }

    certify(args_dict)

""" This script loads a base classifier and then runs PREDICT on many examples from a dataset.
"""
import argparse
import setGPU
from datasets import get_dataset, DATASETS, get_num_classes
from core import Smooth
from time import time
import torch
from architectures import get_architecture
import datetime

parser = argparse.ArgumentParser(description='Predict on many examples')
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
args = parser.parse_args()


# if __name__ == "__main__":
def predict(args_dict):
    # load the base classifier
    checkpoint = torch.load(args_dict['base_classifier'])
    base_classifier = get_architecture(checkpoint["arch"], args_dict['dataset'])
    base_classifier.load_state_dict(checkpoint['state_dict'])

    # create the smoothed classifier g
    smoothed_classifier = Smooth(base_classifier, get_num_classes(args_dict['dataset']), args_dict['sigma'])

    # prepare output file
    f = open(args_dict['outfile'], 'w')
    print("idx\tlabel\tpredict\tcorrect\ttime", file=f, flush=True)
    global correct
    global prediction
    # iterate through the dataset
    dataset = get_dataset(args_dict['dataset'], args.split)
    for i in range(len(dataset)):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args_dict['skip'] != 0:
            continue
        if i == args.max:
            break

        (x, label) = dataset[i]
        x = x.cuda()
        before_time = time()

        # make the prediction
        prediction = smoothed_classifier.predict(x, args_dict['N'], args_dict['alpha'], args_dict['batch'])

        after_time = time()
        correct = int(prediction == label)

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))

        # log the prediction and whether it was correct
        print("{}\t{}\t{}\t{}\t{}".format(i, label, prediction, correct, time_elapsed), file=f, flush=True)

    f.close()
    return prediction, correct


if __name__ == "__main__":
    args_dict = {
        'dataset': 'cifar10',  # 数据集
        'base_classifier': '/data/user/WZT/models/smoothing/cifar10/resnet110/noise_0.00/checkpoint.pth.tar',  # 模型文件
        'sigma': 0.50,  # 噪声水平
        'outfile': '/data/user/WZT/models/smoothing/prediction_outupt/result',  # 输出g对一堆输入进行预测结果
        'batch': 400,  # 训练批次大小
        'skip': 100,  # 每隔一百个图像
        'alpha': 0.001,  #
        'N': 100000
    }

    predict(args_dict)

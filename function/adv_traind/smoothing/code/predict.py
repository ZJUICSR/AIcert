import setGPU
from datasets import get_dataset, DATASETS, get_num_classes
from core import Smooth
from time import time
import torch
from architectures import get_architecture
import datetime

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

    # iterate through the dataset
    dataset = get_dataset(args_dict['dataset'], args_dict['split'])
    for i in range(len(dataset)):

        # only certify every args_dict['skip'] examples, and stop after args_dict['max'] examples
        if i % args_dict['skip'] != 0:
            continue
        if i == args_dict['max']:
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



if __name__ == "__main__":
    args_dict = {
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
    predict(args_dict)

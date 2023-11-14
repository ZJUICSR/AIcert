import parse_args
from image import utils
from metrics.multilabel_metrics import *

def main(model, opt):
    utils.set_random_seed(opt['random_seed'])
    
    if not opt['test_mode']:
        for epoch in range(opt['total_epochs']):
            model.train()
    
    result = model.test()
    
    value = calculate_demographic_parity(metric_func=mean_f1_score, **result)
    print("f1 score value: ", value)


if __name__ == '__main__':
    model, opt = parse_args.collect_args()
    main(model, opt)
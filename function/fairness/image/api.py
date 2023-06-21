from parse_args import create_exerpiment_setting
import utils
import torch
from metrics import METRICS_FULL_NAME

def collect_args(dataset_name='celeba', algorithm_name='baseline', experiment_name='test'):
    assert dataset_name in ['cifar_color', 'cifar_gray', 'cifar-s', 'cifar-i', 'cifar-c_28', 'cifar-d_16', 'cifar-d_8', 'celeba'], "Invalid dataset_name"
    assert algorithm_name in ['baseline', 'sampling', 'domain_discriminative', 'domain_independent', 'uniconf_adv', 'gradproj_adv'], "Invalid algorithm_name"

    experiment = dataset_name + '_' + algorithm_name
    cuda = torch.cuda.is_available()

    opt = {
        'experiment': experiment,
        'experiment_name': experiment_name,
        'cuda': cuda,
        'random_seed': 0
    }
    
    model, opt = create_exerpiment_setting(opt)
    return model, opt

def model_evaluate(dataset_name, model_name="Resnet50", metrics=['mPre', 'mFPR', 'mFNR', 'mTNR', 'mTPR', 'mAcc', 'mRec', 'mSpec', 'mF1', 'mBA'], test_mode=True):
    
    model, opt = collect_args(dataset_name)
    opt['test_mode'] = test_mode
    utils.set_random_seed(opt["random_seed"])
    if not opt['test_mode']:
        for epoch in range(opt['total_epochs']):
            model.train()
    
    td = model.test()
    
    result = {}
    cal_func = opt['evaluate_func']
    metrics_ls = opt['metrics_list']
    for m in metrics:
        value = cal_func(metric_func=metrics_ls[m], **td)
        result[METRICS_FULL_NAME[m]] = value

    return result

def model_debias(dataset_name,  model_name="Resnet50", algorithm_name="sampling", metrics=['DI', 'DP', 'PE', 'EOD', 'PP', 'OMd', 'FOd', 'FNd'], test_mode=True):
    
    model, opt = collect_args(dataset_name, algorithm_name=algorithm_name)
    opt['test_mode'] = test_mode
    utils.set_random_seed(opt["random_seed"])
    if not opt['test_mode']:
        for epoch in range(opt['total_epochs']):
            model.train()
    
    td = model.test()
    
    result = {}
    cal_func = opt['evaluate_func']
    metrics_ls = opt['metrics_list']
    for m in metrics:
        value = cal_func(metric_func=metrics_ls[m], **td)
        result[METRICS_FULL_NAME[m]] = value

    return result

if __name__ == "__main__":
    result = model_evaluate("cifar-s")
    print(result)
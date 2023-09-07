from image.parse_args import create_exerpiment_setting
from image import utils
import torch
import numpy as np
from image.metrics import METRICS_FULL_NAME

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

def model_overall_fairness(result, metrics=['mPre', 'mFPR', 'mFNR', 'mTNR', 'mTPR', 'mAcc', 'mF1', 'mBA']):
    # print(result)
    valid_metrics = ['mPre', 'mFPR', 'mFNR', 'mTNR', 'mTPR', 'mAcc', 'mF1', 'mBA']
    # valid_metrics_name = [METRICS_FULL_NAME[key] for key in valid_metrics]
    eval_metrics = [key for key in metrics if ((METRICS_FULL_NAME[key] in result) and (key in valid_metrics))]
    values = [result[METRICS_FULL_NAME[key]] for key in eval_metrics]
    overall_group_fairness = 1 - np.mean(values)
    return overall_group_fairness

def model_evaluate(dataset_name, model_name="Resnet50", metrics=['mPre', 'mFPR', 'mFNR', 'mTNR', 'mTPR', 'mAcc', 'mF1', 'mBA'], test_mode=True, logger=None):
    logger.info(f"start evaluating fairness of model: \'{model_name}\' on dataset: \'{dataset_name}\'.")
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
    logger.info(f'calculating fairness metrics for model: \'{model_name}\' on dataset: \'{dataset_name}\'.')
    for m in metrics:
        logger.info(f'start with algoritm \'{m}\'.')
        value = cal_func(metric_func=metrics_ls[m], **td)
        result[METRICS_FULL_NAME[m]] = value
        
    # calculate overall fairness
    logger.info(f'calculating overall fairness for model: \'{model_name}\' on dataset: \'{dataset_name}\'.')
    result['Overall fairness'] = model_overall_fairness(result=result)
    logger.info(f'model evaluation done.')
    return result

def model_debias(dataset_name,  model_name="Resnet50", algorithm_name="sampling", metrics=['mPre', 'mFPR', 'mFNR', 'mTNR', 'mTPR', 'mAcc', 'mF1', 'mBA'], test_mode=True, logger=None):
    logger.info(f"start improving fairness of model: \'{model_name}\' on dataset: \'{dataset_name}\'.")
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
        logger.info(f'start in-process debiasing with algoritm \'{m}\'.')
        value = cal_func(metric_func=metrics_ls[m], **td)
        result[METRICS_FULL_NAME[m]] = value
        
    # calculate overall fairness
    logger.info(f'calculating overall fairness for model: \'{model_name}\' on dataset: \'{dataset_name}\'.')
    result['Overall fairness'] = model_overall_fairness(result=result)
    logger.info(f'model debiasing done.')
    return result

if __name__ == "__main__":
    result = model_evaluate("cifar-s", test_mode=True)
    print(result)
    result = model_debias("cifar-s", test_mode=True)
    print(result)
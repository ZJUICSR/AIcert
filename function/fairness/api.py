import torch
from .fairness_datasets import CompasDataset, AdultDataset, GermanDataset
from .metrics.dataset_metric import DatasetMetrics
from .debias.preprocess import *
# from models.models import LR, Net
from torch import nn, threshold
import random
import numpy as np
from .metrics.model_metrics import *
import copy
from .debias.inprocess import *
from .debias.postprocess import *
from .dataset_analysis import *
import os
import logging

LOGGING_FILE = './log.txt'

# Configure logging
logging.basicConfig(filename=LOGGING_FILE, level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create a logger object
logger = logging.getLogger(__name__)

seed = 11
random.seed(seed)
# os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True


def get_all_values(d):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from get_all_values(v)
        else:
            yield v

def split_result(result):
    if isinstance(result, list): # if is list, reform each item and return list
        for item in result:
            item = split_result(item)
        return result
    elif isinstance(result, dict): # if is dict, and without list, return input directly, else split into two dicts
        result_ls = [{}, {}]
        flag = False
        for key in result:
            item = split_result(result[key])
            if isinstance(item, list):
                flag = True
                result_ls[0][key] = item[0]
                result_ls[1][key] = item[1]
            else:
                result_ls[0][key] = item
                result_ls[1][key] = item
        return result_ls if flag else result
    else: # if is value, return directly
        return result

def dataset_overall_fairness(result):
    values = list(get_all_values(result['Favorable Rate Difference']))
    overall_group_fairness = 1 - np.mean(values)
    overall_individual_fairness = result['Consistency']
    overall_fairness = np.mean([overall_group_fairness, overall_individual_fairness])
    return [overall_group_fairness, overall_individual_fairness, overall_fairness]

def model_overall_fairness(result, metrics=['DP', 'PE', 'EOP','EOD', 'PP', 'OMd', 'FDd', 'FOd']):
    
    values = {key: result[METRICS_FULL_NAME[key]] for key in metrics if METRICS_FULL_NAME[key] in result}
    values = list(get_all_values(values))
    overall_group_fairness = 1 - np.mean(values)
    overall_individual_fairness = result['Consistency']
    overall_fairness = np.mean([overall_group_fairness, overall_individual_fairness])
    return [overall_group_fairness, overall_individual_fairness, overall_fairness]

# dataset evaluation
def dataset_evaluate(dataset_name, sensattrs=[], targetattrs=[]):
    """Evaluates fairness of built-in dataset 

    Args:
        dataset_name (str): name of built in dataset, in ['Compas', 'Adult', 'German']
        sensattrs (list, optional): names of sensitive attributes. Defaults to [], which include all sensitive attributes available.
        targetattrs (list, optional): names of target attributes. Defaults to [], which include all target attributes available.

    Returns:
        Dict: result of the evaluation
    """
    logger.info(f"start evaluating fairness of dataset: \'{dataset_name}\'.")
    dataset_cls = None
    result = {
        'Favorable Rate Difference': {},
        'Favorable Rate Ratio': {},
        'Consistency': None,
        'Proportion':{}
    }
    if dataset_name == 'Compas':
        dataset_cls = CompasDataset
    elif dataset_name == 'Adult':
        dataset_cls = AdultDataset
    elif dataset_name == 'German':
        dataset_cls = GermanDataset
    else:
        logger.exception('invalid data set: \'{}\''.format(dataset_name))
        raise ValueError('invalid data set: \'{}\''.format(dataset_name))
    
    # check attrs validity
    if not all(attr in dataset_cls.favorable.keys() for attr in targetattrs):
        logger.exception('invalid target attribute: \'{}\''.format(targetattrs))
        raise ValueError('invalid target attribute: \'{}\''.format(targetattrs))
    if not all(attr in dataset_cls.privileged for attr in sensattrs):
        logger.exception('invalid sensitive attribute: \'{}\''.format(sensattrs))
        raise ValueError('invalid sensitive attribute: \'{}\''.format(sensattrs))
    
    # take all sensitive/target attributes as default
    if len(sensattrs) == 0:
        logger.info('take all available sensitive attribute by default')
        sensattrs = list(dataset_cls.privileged.keys())
    if len(targetattrs) == 0:
        logger.info('take all available target attribute by default')
        targetattrs = list(dataset_cls.favorable.keys())
    
    # group and individual fairness metrics
    for target in targetattrs:
        logger.info(f'calculating fairness metrics for target attribute \'{target}\'.')
        dataset = dataset_cls(favorable=target)
        metrics = DatasetMetrics(dataset=dataset)
        fd = metrics.favorable_diff()
        fr = metrics.favorable_ratio()
        result['Favorable Rate Difference'][target] = {k:fd[k] for k in sensattrs}
        result['Favorable Rate Ratio'][target] = {k:fr[k] for k in sensattrs}
        if result['Consistency'] is None:
            result['Consistency'] = metrics.consistency()

    # proportion of goups
    logger.info(f'calculating proportion of groups in dataset: \'{dataset_name}\'.')
    dataset = dataset_cls()
    prop = dataset.proportion()
    for attr in sensattrs:
        result['Proportion'][attr] = {
            SENSITIVE_GROUPS[dataset_name][attr][0]: prop[attr],
            SENSITIVE_GROUPS[dataset_name][attr][1]: 1 - prop[attr],
        }
        
    # overall fairness
    logger.info(f'calculating overall fairness for dataset: \'{dataset_name}\'.')
    result1 = dataset_overall_fairness(result=result)
    result['Overall group fairness'] = result1[0]
    result['Overall individual fairness'] = result1[1]
    result['Overall fairness'] = result1[2]

    logger.info(f'data set evalution done.')
    return result


# dataset debiasing e.g. the first element in the list is original, the second is improved

def dataset_debias(dataset_name, algorithm_name, sensattrs=[], targetattrs=[]):
    """Debias built-in datasets

    Args:
        dataset_name (str): name of built in dataset, in ['Compas', 'Adult', 'German']
        algorithm_name (str): name of built-in dataset debias algorithms
        sensattrs (list, optional): names of sensitive attributes. Defaults to [], which include all sensitive attributes available.
        targetattrs (list, optional): names of target attributes. Defaults to [], which include all target attributes available.

    Returns:
        _type_: _description_
    """
    logger.info(f"start debiasing dataset: \'{dataset_name}\' using algorithm: \'{algorithm_name}\'.")
    dataset_cls = None
    algorithm_cls = None
    result = {
        'Favorable Rate Difference': {},
        'Favorable Rate Ratio': {},
        'Consistency': None,
        'Proportion':{}
    }
    if dataset_name == 'Compas':
        dataset_cls = CompasDataset
    elif dataset_name == 'Adult':
        dataset_cls = AdultDataset
    elif dataset_name == 'German':
        dataset_cls = GermanDataset
    else:
        logger.exception('invalid data set: \'{}\''.format(dataset_name))
        raise ValueError('invalid data set: \'{}\''.format(dataset_name))

    if algorithm_name == 'LFR':
        algorithm_cls = LFR
    elif algorithm_name == 'Reweighing':
        algorithm_cls = Reweighing
    else:
        logger.exception('invalid algoritm: \'{}\''.format(algorithm_name))
        raise ValueError('invalid algoritm: \'{}\''.format(algorithm_name))
    
    # check attrs validity
    if not all(attr in dataset_cls.favorable.keys() for attr in targetattrs):
        logger.exception('invalid target attribute: \'{}\''.format(targetattrs))
        raise ValueError('invalid target attribute: \'{}\''.format(targetattrs))
    if not all(attr in dataset_cls.privileged for attr in sensattrs):
        logger.exception('invalid sensitive attribute: \'{}\''.format(sensattrs))
        raise ValueError('invalid sensitive attribute: \'{}\''.format(sensattrs))

    # take all sensitive/target attributes as default
    if len(sensattrs) == 0:
        logger.info('take all available sensitive attribute by default')
        sensattrs = list(dataset_cls.privileged.keys())
    if len(targetattrs) == 0:
        logger.info('take all available target attribute by default')
        targetattrs = [list(dataset_cls.favorable.keys())[0]]
        
    dataset = dataset_cls()
    metrics = DatasetMetrics(dataset=dataset)

    # group and individual fairness metrics
    for target in targetattrs:
        logger.info(f'debiasing target attribute \'{target}\'.')
        dataset = dataset_cls(favorable=target)
        metrics = DatasetMetrics(dataset=dataset)
        # debiasing
        new_dataset = algorithm_cls(dataset).fit().transform()
        logger.info(f'calcuating fairness metrics for original and debiased dataset: \'{dataset_name}\'.')
        new_metrics = DatasetMetrics(new_dataset)
        fd, nfd = metrics.favorable_diff(), new_metrics.favorable_diff()
        fr, nfr = metrics.favorable_ratio(), new_metrics.favorable_ratio()
        result['Favorable Rate Difference'][target] = [{k:d[k] for k in sensattrs} for d in [fd, nfd]]
        result['Favorable Rate Ratio'][target] = [{k:d[k] for k in sensattrs} for d in [fr, nfr]]
        if result['Consistency'] is None:
            result['Consistency'] = [metrics.consistency(), new_metrics.consistency()]
        
        # proportion of goups
        logger.info(f'calculating proportion of groups in dataset: \'{dataset_name}\'.')
        dataset = dataset_cls()
        prop = dataset.proportion()
        for attr in sensattrs:
            result['Proportion'][attr] = {
                SENSITIVE_GROUPS[dataset_name][attr][0]: prop[attr],
                SENSITIVE_GROUPS[dataset_name][attr][1]: 1 - prop[attr],
            }
            
        # overall fairness
        logger.info(f'calculating overall fairness for dataset: \'{dataset_name}\'.')
        result1, result2 = split_result(result)
        result1 = dataset_overall_fairness(result=result1)
        result2 = dataset_overall_fairness(result=result2)
        result['Overall group fairness'] = [result1[0], result2[0]]
        result['Overall individual fairness'] = [result1[1], result2[1]]
        result['Overall fairness'] = [result1[2], result2[2]]

        logger.info(f'data debiasing done.')
        return result

# evaluate model fairness
def model_evaluate(dataset_name, model_name, metrics=['DI', 'DP', 'PE', 'EOD', 'PP', 'OMd', 'FOd', 'FNd'], sensattrs=[], targetattr=None, generalized=False):
    """Evaluate fairness of model trained on built in dataset

    Args:
        dataset_name (str): name of built in dataset, in ['Compas', 'Adult', 'German']
        model_name (str): name of model structure
        metrics (list, optional): list of metric names to be evaluated upon. Defaults to ['DI', 'DP', 'PE', 'EOD', 'PP', 'OMd', 'FOd', 'FNd'].
        sensattrs (list, optional): names of sensitive attributes. Defaults to [], which include all sensitive attributes available.
        targetattr (str, optional): name of target attribute. Defaults to None, which use the default target attribute of the dataset.
        generalized (bool, optional): if, the metrics are evaluated in a generalized way(according to soft prediction rather than hard). Defaults to False.

    Returns:
        _type_: _description_
    """
    logger.info(f"start evaluating fairness of model: \'{model_name}\' on dataset: \'{dataset_name}\'.")
    
    dataset_cls = None
    model_cls = None
    result = {
        'Consistency': None,
        'Proportion': {},
    }
    if dataset_name == 'Compas':
        dataset_cls = CompasDataset
    elif dataset_name == 'Adult':
        dataset_cls = AdultDataset
    elif dataset_name == 'German':
        dataset_cls = GermanDataset
    else:
        logger.exception('invalid data set: \'{}\''.format(dataset_name))
        raise ValueError('invalid data set: \'{}\''.format(dataset_name))
    
    # check attrs validity
    if all((not targetattr in dataset_cls.favorable.keys(), targetattr is not None)):
        logger.exception('invalid target attribute: \'{}\''.format(targetattr))
        raise ValueError('invalid target attribute: \'{}\''.format(targetattr))
    if not all(attr in dataset_cls.privileged for attr in sensattrs):
        logger.exception('invalid sensitive attribute: \'{}\''.format(sensattrs))
        raise ValueError('invalid sensitive attribute: \'{}\''.format(sensattrs))

    # take all sensitive attributes as default
    if len(sensattrs) == 0:
        logger.info('take all available sensitive attribute by default')
        sensattrs = list(dataset_cls.privileged.keys())

    dataset = dataset_cls(favorable=targetattr)

    if model_name == '3 Hidden-layer FCN':
        model_cls = Classifier
    else:
        logger.exception('invalid model: \'{}\''.format(model_name))
        raise ValueError('invalid model: \'{}\''.format(model_name))

    # training model
    logger.info(f'training model: \'{model_name}\' on dataset: \'{dataset_name}\'.')
    model = model_cls(input_shape=dataset.num_features, device='cuda')
    model.train(dataset=dataset, epochs=2000)

    # evaluating model
    logger.info(f'calculating fairness metrics for model: \'{model_name}\' on dataset: \'{dataset_name}\'.')
    pred_dataset = model.predicted_dataset(dataset=dataset)
    model_metrics = ModelMetrics(dataset=dataset, classified_dataset=pred_dataset)
    for metric in metrics:
        m = None
        if generalized:
            m = model_metrics.general_group_fairness_metrics(metrics=metric)
        else:
            m = model_metrics.group_fairness_metrics(metrics=metric)
        result[METRICS_FULL_NAME[metric]] = {k: m[k] for k in sensattrs}
    result['Consistency'] = model_metrics.consistency()

    # proportion of groups given favorable prediction
    logger.info(f'calculating proportion of groups given favorable prediction for model: \'{model_name}\' on dataset: \'{dataset_name}\'.')
    dataset = dataset_cls(favorable=targetattr)
    prop = dataset.proportion(favorable=True)
    for attr in sensattrs:
        result['Proportion'][attr] = {
            SENSITIVE_GROUPS[dataset_name][attr][0]: prop[attr],
            SENSITIVE_GROUPS[dataset_name][attr][1]: 1 - prop[attr],
        }

    # overall fairness
    logger.info(f'calculating overall fairness for model: \'{model_name}\' on dataset: \'{dataset_name}\'.')
    result1 = model_overall_fairness(result=result)
    result['Overall group fairness'] = result1[0]
    result['Overall individual fairness'] = result1[1]
    result['Overall fairness'] = result1[2]

    logger.info(f'model evaluation done.')
    return result


# model debiasing
def model_debias(dataset_name, model_name, algorithm_name, metrics=['DI', 'DP', 'PE', 'EOD', 'PP', 'OMd', 'FOd', 'FNd'], sensattrs=[], targetattr=None, generalized=False):
    """Debias model trained on built-in dataset

    Args:
        dataset_name (str): name of built in dataset, in ['Compas', 'Adult', 'German']
        model_name (str): name of model structure
        algorithm_name (str): the name of built in model debias algorithms
        metrics (list, optional): list of metric names to be evaluated upon. Defaults to ['DI', 'DP', 'PE', 'EOD', 'PP', 'OMd', 'FOd', 'FNd'].
        sensattrs (list, optional): names of sensitive attributes. Defaults to [], which include all sensitive attributes available.
        targetattr (str, optional): name of target attribute. Defaults to None, which use the default target attribute of the dataset.
        generalized (bool, optional): if, the metrics are evaluated in a generalized way(according to soft prediction rather than hard). Defaults to False.

    Returns:
        _type_: _description_
    """
    logger.info(f"start improving fairness of model: \'{model_name}\' on dataset: \'{dataset_name}\'.")
    
    dataset_cls = None
    model_cls = None
    algorithm_cls = None
    result = {
        'Consistency': None,
        'Proportion': {},
    }
    if dataset_name == 'Compas':
        dataset_cls = CompasDataset
    elif dataset_name == 'Adult':
        dataset_cls = AdultDataset
    elif dataset_name == 'German':
        dataset_cls = GermanDataset
    else:
        logger.exception('invalid data set: \'{}\''.format(dataset_name))
        raise ValueError('invalid data set: \'{}\''.format(dataset_name))
    
    # check attrs validity
    if all((not targetattr in dataset_cls.favorable.keys(), targetattr is not None)):
        logger.exception('invalid target attribute: \'{}\''.format(targetattr))
        raise ValueError('invalid target attribute: \'{}\''.format(targetattr))
    if not all(attr in dataset_cls.privileged for attr in sensattrs):
        logger.exception('invalid sensitive attribute: \'{}\''.format(sensattrs))
        raise ValueError('invalid sensitive attribute: \'{}\''.format(sensattrs))

    # take all sensitive attributes as default
    if len(sensattrs) == 0:
        logger.info('take all available sensitive attribute by default')
        sensattrs = list(dataset_cls.privileged.keys())

    dataset = dataset_cls(favorable=targetattr)

    if model_name == '3 Hidden-layer FCN':
        model_cls = Classifier
    else:
        logger.exception('invalid model: \'{}\''.format(model_name))
        raise ValueError('invalid model: \'{}\''.format(model_name))

    if algorithm_name == 'Domain Independent':
        algorithm_cls = DomainIndependentClassifier
    elif algorithm_name == 'Adersarial Debiasing':
        algorithm_cls = FADClassifier
    elif algorithm_name in ['Calibrated EOD-'+c for c in ['fnr', 'fpr', 'weighted']]:
        algorithm_cls = CalibratedEqOdds
    elif algorithm_name in ['Reject Option-'+c for c in ['SPd', 'AOd', 'EOd']]:
        algorithm_cls = RejectOptionClassification
    else:
        logger.exception('invalid algorithm: \'{}\''.format(algorithm_name))
        raise ValueError('invalid algorithm: \'{}\''.format(algorithm_name))

    # training model
    logger.info(f'training model: \'{model_name}\' on dataset: \'{dataset_name}\'.')
    model = model_cls(input_shape=dataset.num_features, device='cuda')
    model.train(dataset=dataset, epochs=2000)

    # evaluating model
    logger.info(f'calcuating fairness metrics for original model: \'{model_name}\' on dataset: \'{dataset_name}\'.')
    pred_dataset = model.predicted_dataset(dataset=dataset)
    model_metrics = ModelMetrics(dataset=dataset, classified_dataset=pred_dataset)

    #debiasing
    new_pred_dataset = None
    if issubclass(algorithm_cls, Classifier): # in process debiasing
        logger.info(f'start in-process debiasing with algoritm \'{algorithm_name}\'.')
        new_model = algorithm_cls(input_shape=dataset.num_features, device='cuda')
        new_model.train(dataset=dataset, epochs=2000)
        new_pred_dataset = new_model.predicted_dataset(dataset=dataset)
    else: # post process debaising
        logger.info(f'start post-process debiasing with algoritm \'{algorithm_name}\'.')
        algorithm = algorithm_cls()
        if issubclass(algorithm_cls, CalibratedEqOdds):
            constraint = algorithm_name.split('-')[1]
            algorithm = algorithm_cls(cost_constraint=constraint)
        if issubclass(algorithm_cls, RejectOptionClassification):
            metric_name = algorithm_name.split('-')[1]
            algorithm = algorithm_cls(metric_name=metric_name)
        algorithm.fit(dataset=dataset, dataset_pred=pred_dataset)
        new_pred_dataset = algorithm.transform(pred_dataset)
    new_metrics = ModelMetrics(dataset=dataset, classified_dataset=new_pred_dataset)

    logger.info(f'calcuating fairness metrics for debiased model: \'{model_name}\' on dataset: \'{dataset_name}\'.')
    for metric in metrics:
        m = None
        new_m = None
        if generalized:
            m = model_metrics.general_group_fairness_metrics(metrics=metric)
            new_m = new_metrics.general_group_fairness_metrics(metrics=metric)
        else:
            m = model_metrics.group_fairness_metrics(metrics=metric)
            new_m = new_metrics.group_fairness_metrics(metrics=metric)
        result[METRICS_FULL_NAME[metric]] = [{k:d[k] for k in sensattrs} for d in[m, new_m]]
    result['Consistency'] = [model_metrics.consistency(), new_metrics.consistency()]
    # if result['Consistency'] is None:
    #     result['Consistency'] = [metrics.consistency(), new_metrics.consistency()]

    # proportion of goups given favorable prediction
    logger.info(f'calculating proportion of groups given favorable prediction for model: \'{model_name}\' on dataset: \'{dataset_name}\'.')
    dataset = dataset_cls(favorable=targetattr)
    prop = dataset.proportion(favorable=True)
    for attr in sensattrs:
        result['Proportion'][attr] = {
            SENSITIVE_GROUPS[dataset_name][attr][0]: prop[attr],
            SENSITIVE_GROUPS[dataset_name][attr][1]: 1 - prop[attr],
        }

    # overall fairness
    logger.info(f'calculating overall fairness for model: \'{model_name}\' on dataset: \'{dataset_name}\'.')
    result1, result2 = split_result(result)
    result1 = model_overall_fairness(result=result1)
    result2 = model_overall_fairness(result=result2)
    result['Overall group fairness'] = [result1[0], result2[0]]
    result['Overall individual fairness'] = [result1[1], result2[1]]
    result['Overall fairness'] = [result1[2], result2[2]]

    logger.info(f'model debiasing done.')
    return result

def dataset_analysis(dataset_name, attrs=[], targetattrs=[], prptn_thd=0.1):
    """Analyize statistical characteristics (correlation coefficient and proportion) of built-in datasets. For m attribute and n target attributes there will be m * n pairs of correlation coefficient sets (4 correlation coefficients supported). Categorical attributes among 'attrs' will be given proportion analysis.

    Args:
        dataset_name (str): name of built in dataset, in ['Compas', 'Adult', 'German']
        attrs (list, optional): names of attributes to analysis. Defaults to [], which include all attributes available.
        targetattrs (list, optional): names of target attributes (could be any attribute, not necessarily target attribute). Defaults to [], which include all target attributes available.
        prptn_thd (float, optional): threshold of proportion. Categories with proportion lower then the threshold will be merged into one category named 'others'. 

    Returns:
        Dict: result of the analysis
    """
    logger.info(f"start analysis on dataset: \'{dataset_name}\'.")
    dataset_cls = None
    result = {
        'Correlation coefficients': [],
        'Proportion':{},
        'Overall Correlation': None,
        'Overall uniformity': None,
    }
    if dataset_name == 'Compas':
        dataset_cls = CompasDataset
    elif dataset_name == 'Adult':
        dataset_cls = AdultDataset
    elif dataset_name == 'German':
        dataset_cls = GermanDataset
    else:
        logger.exception('invalid data set: \'{}\''.format(dataset_name))
        raise ValueError('invalid data set: \'{}\''.format(dataset_name))
    
    # take all sensitive/target attributes as default
    if len(attrs) == 0:
        logger.info('take all available attribute by default')
        attrs = dataset_cls.features_to_keep
    if len(targetattrs) == 0:
        logger.info('take all available target attribute by default')
        targetattrs = list(dataset_cls.favorable.keys())
    
    privileged = dataset_cls.privileged
    favorable = dataset_cls.favorable
    
    # prepare dataset
    logger.info('preprocessing dataset')
    dataset = dataset_cls()
    dataset = preprocess(dataset, factorize=True)
    
    # categorical features
    logger.info('analysing all categorical features in dataset')
    categotical_features = dataset.__class__.categotical_features + [key for key in favorable.keys() if isinstance(favorable[key], list)] + [key for key in privileged.keys() if isinstance(privileged[key], list)]
    categotical_features = list(set(categotical_features))
    # categotical_features = dataset_cls.categotical_features + list(favorable.keys()) + list(privileged.keys())

    # calculate correlation
    logger.info('calculating correlations of attributes in dataset')
    for attr in attrs:
        attr2_type = 'continuous' if attr not in categotical_features else 'discrete'
        attr2 = attr # if attr2_type == 'discrete' else attr + '_cat'
        for target in targetattrs:
            attr1_type = 'continuous' if target not in categotical_features else 'discrete'
            attr1 = target # if attr1_type == 'discrete' else target + '_cat'
            result['Correlation coefficients'].append({"attr": attr, "target": target, "values": correlation_analysis(dataset.df, attr1, attr2, attr1_type, attr2_type)})

    # calculate proportion
    logger.info('calculating proportion of categorical attributes in dataset')
    dataset = preprocess(dataset, factorize=False) # disable factorize so sens and target attr remains string format
    cat_attrs = [attr for attr in attrs if attr in categotical_features]
    result['Proportion'] = calculate_category_proportions(dataset.df, cat_attrs, prptn_thd)
    
    # overall analysis
    logger.info('calculating overall analysis of dataset')
    result['Overall Correlation'] = calculate_overall_correlation(result['Correlation coefficients'])
    result['Overall uniformity'] = calculate_distribution_uniformity(result['Proportion'])

    return result

def test1():
    # testing
    # for dataset in ['Compas', 'Adult', 'German']: # go through all data sets
    # np.random.seed(seed)

    # print(model_evaluate("Compas", '3 Hidden-layer FCN'))

    # exit()
    for dataset in ['Compas']: # go through all data sets
        print('======dataset evaluation: {}======'.format(dataset))
        print(dataset_evaluate(dataset_name=dataset))
        for dataset_algorithm in ['LFR', 'Reweighing']: # go through all data set debias algorithms
        # for dataset_algorithm in ['LFR']: # go through all data set debias algorithms
            print('====dataset debiasing: {}, {}===='.format(dataset, dataset_algorithm))
            print(dataset_debias(dataset_name=dataset, algorithm_name=dataset_algorithm))
        for model in ['3 Hidden-layer FCN']: # go through all models
            print('====model evaluation: {}, {}===='.format(dataset, model))
            print(model_evaluate(dataset_name=dataset, model_name=model))
            # for model_algorithm in ['Reject Option-SPd', 'Reject Option-AOd', 'Reject Option-EOd']: # go through all model debias algorithms
            for model_algorithm in ['Domain Independent', 'Adersarial Debiasing', 'Calibrated EOD-fnr', 'Calibrated EOD-fpr', 'Calibrated EOD-weighted', 'Reject Option-SPd', 'Reject Option-AOd', 'Reject Option-EOd']: # go through all model debias algorithms
                print('====model debiasing: {}, {}, {}===='.format(dataset, model, model_algorithm))
                print(model_debias(dataset_name=dataset, model_name=model, algorithm_name=model_algorithm,generalized=True))  

TARGET_ATTR = {
    'Compas': list(CompasDataset.favorable.keys()),
    'Adult': list(AdultDataset.favorable.keys()),
    'German': list(GermanDataset.favorable.keys()),
}

def test2():
    # testing
    # for dataset in ['Compas', 'Adult', 'German']: # go through all data sets
    # np.random.seed(seed)

    # print(model_evaluate("Compas", '3 Hidden-layer FCN'))

    # exit()
    # for dataset in ['Compas', 'Adult', 'German']:
    for dataset in ['Adult']: # go through all data sets
        # print('======dataset evaluation: {}======'.format(dataset))
        # print(dataset_evaluate(dataset_name=dataset))
        # print(dataset_analysis(dataset_name=dataset))
        # continue
        # for dataset_algorithm in ['LFR', 'Reweighing']: # go through all data set debias algorithms
        # for dataset_algorithm in ['LFR']: # go through all data set debias algorithms
            # print('====dataset debiasing: {}, {}===='.format(dataset, dataset_algorithm))
            # result = dataset_debias(dataset_name=dataset, algorithm_name=dataset_algorithm)
            # print(result)
            # print("transformed debias result: ", split_result(result=result))
            # print("original result: ", result)
        for model in ['3 Hidden-layer FCN']: # go through all models
            for target_attr in TARGET_ATTR[dataset]:
                print('====model evaluation: {}-{}, {}===='.format(dataset, target_attr, model))
                result = model_evaluate(dataset_name=dataset, model_name=model, targetattr=target_attr)
                print(result)
                # print("transformed result: ", split_result(result=result))
                # for model_algorithm in ['Reject Option-SPd', 'Reject Option-AOd', 'Reject Option-EOd']: # go through all model debias algorithms
                for model_algorithm in ['Domain Independent', 'Adersarial Debiasing', 'Calibrated EOD-fnr', 'Calibrated EOD-fpr', 'Calibrated EOD-weighted', 'Reject Option-SPd', 'Reject Option-AOd', 'Reject Option-EOd']: # go through all model debias algorithms
                    print('====model debiasing: {}-{}, {}, {}===='.format(dataset,target_attr, model, model_algorithm))
                    result = model_debias(dataset_name=dataset, model_name=model, algorithm_name=model_algorithm,generalized=True, targetattr=target_attr)
                    print(result)  
                    print("transformed result: ", split_result(result=result))
        
    # print(result) 

if __name__ == '__main__':
    test2()
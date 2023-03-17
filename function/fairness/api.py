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
import os

seed = 11
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

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
        raise ValueError('invalid data set: \'{}\''.format(dataset_name))
    
    # check attrs validity
    if not all(attr in dataset_cls.favorable.keys() for attr in targetattrs):
        raise ValueError('invalid target attribute: \'{}\''.format(targetattrs))
    if not all(attr in dataset_cls.privileged for attr in sensattrs):
        raise ValueError('invalid sensitive attribute: \'{}\''.format(sensattrs))
    
    # take all sensitive/target attributes as default
    if len(sensattrs) == 0:
        sensattrs = list(dataset_cls.privileged.keys())
    if len(targetattrs) == 0:
        targetattrs = list(dataset_cls.favorable.keys())
    
    # group and individual fairness metrics
    for target in targetattrs:
        dataset = dataset_cls(favorable=target)
        metrics = DatasetMetrics(dataset=dataset)
        fd = metrics.favorable_diff()
        fr = metrics.favorable_ratio()
        result['Favorable Rate Difference'][target] = {k:fd[k] for k in sensattrs}
        result['Favorable Rate Ratio'][target] = {k:fr[k] for k in sensattrs}
        if result['Consistency'] is None:
            result['Consistency'] = metrics.consistency()

    # proportion of goups
    dataset = dataset_cls()
    prop = dataset.proportion()
    for attr in sensattrs:
        result['Proportion'][attr] = {
            SENSITIVE_GROUPS[dataset_name][attr][0]: prop[attr],
            SENSITIVE_GROUPS[dataset_name][attr][1]: 1 - prop[attr],
        }

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
        raise ValueError('invalid data set: \'{}\''.format(dataset_name))

    if algorithm_name == 'LFR':
        algorithm_cls = LFR
    elif algorithm_name == 'Reweighing':
        algorithm_cls = Reweighing
    else:
        raise ValueError('invalid algoritm: \'{}\''.format(algorithm_name))
    
    # check attrs validity
    if not all(attr in dataset_cls.favorable.keys() for attr in targetattrs):
        raise ValueError('invalid target attribute: \'{}\''.format(targetattrs))
    if not all(attr in dataset_cls.privileged for attr in sensattrs):
        raise ValueError('invalid sensitive attribute: \'{}\''.format(sensattrs))

    # take all sensitive/target attributes as default
    if len(sensattrs) == 0:
        sensattrs = list(dataset_cls.privileged.keys())
    if len(targetattrs) == 0:
        targetattrs = list(dataset_cls.favorable.keys())
        
    dataset = dataset_cls()
    # algorithm = algorithm_cls(dataset=dataset)
    metrics = DatasetMetrics(dataset=dataset)
    # algorithm.fit()
    # new_dataset = algorithm.transform()
    # new_metrics = DatasetMetrics(new_dataset)

    # group and individual fairness metrics
    for target in targetattrs:
        print('==debias target: {}=='.format(target))
        dataset = dataset_cls(favorable=target)
        metrics = DatasetMetrics(dataset=dataset)
        # debiasing
        new_dataset = algorithm_cls(dataset).fit().transform()
        new_metrics = DatasetMetrics(new_dataset)
        fd, nfd = metrics.favorable_diff(), new_metrics.favorable_diff()
        fr, nfr = metrics.favorable_ratio(), new_metrics.favorable_ratio()
        result['Favorable Rate Difference'][target] = [{k:d[k] for k in sensattrs} for d in [fd, nfd]]
        result['Favorable Rate Ratio'][target] = [{k:d[k] for k in sensattrs} for d in [fr, nfr]]
        if result['Consistency'] is None:
            result['Consistency'] = [metrics.consistency(), new_metrics.consistency()]

    # proportion of goups
    dataset = dataset_cls()
    prop = dataset.proportion()
    for attr in sensattrs:
        result['Proportion'][attr] = {
            SENSITIVE_GROUPS[dataset_name][attr][0]: prop[attr],
            SENSITIVE_GROUPS[dataset_name][attr][1]: 1 - prop[attr],
        }

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
        raise ValueError('invalid data set: \'{}\''.format(dataset_name))
    
    # check attrs validity
    if all((not targetattr in dataset_cls.favorable.keys(), targetattr is not None)):
        raise ValueError('invalid target attribute: \'{}\''.format(targetattr))
    if not all(attr in dataset_cls.privileged for attr in sensattrs):
        raise ValueError('invalid sensitive attribute: \'{}\''.format(sensattrs))

    # take all sensitive attributes as default
    if len(sensattrs) == 0:
        sensattrs = list(dataset_cls.privileged.keys())

    dataset = dataset_cls(favorable=targetattr)

    if model_name == '3 Hidden-layer FCN':
        model_cls = Classifier
    else:
        raise ValueError('invalid model: \'{}\''.format(model_name))

    # training model
    model = model_cls(input_shape=dataset.num_features, device='cuda')
    model.train(dataset=dataset, epochs=2000)

    # evaluating model
    pred_dataset = model.predicted_dataset(dataset=dataset)
    model_metrics = ModelMetrics(dataset=dataset, classified_dataset=pred_dataset)
    for metric in metrics:
        m = None
        if generalized:
            m = model_metrics.general_group_fairness_metrics(metrics=metric)
        else:
            m = model_metrics.group_fairness_metrics(metrics=metric)
        result[METRICS_FULL_NAME[metric]] = {k:m[k] for k in sensattrs}
    result['Consistency'] = model_metrics.consistency()
    # if result['Consistency'] is None:
    #     result['Consistency'] = [metrics.consistency(), new_metrics.consistency()]

    # proportion of goups given favorable prediction
    dataset = dataset_cls(favorable=targetattr)
    prop = dataset.proportion(favorable=True)
    for attr in sensattrs:
        result['Proportion'][attr] = {
            SENSITIVE_GROUPS[dataset_name][attr][0]: prop[attr],
            SENSITIVE_GROUPS[dataset_name][attr][1]: 1 - prop[attr],
        }

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
        raise ValueError('invalid data set: \'{}\''.format(dataset_name))
    
    # check attrs validity
    if all((not targetattr in dataset_cls.favorable.keys(), targetattr is not None)):
        raise ValueError('invalid target attribute: \'{}\''.format(targetattr))
    if not all(attr in dataset_cls.privileged for attr in sensattrs):
        raise ValueError('invalid sensitive attribute: \'{}\''.format(sensattrs))

    # take all sensitive attributes as default
    if len(sensattrs) == 0:
        sensattrs = list(dataset_cls.privileged.keys())

    dataset = dataset_cls(favorable=targetattr)

    if model_name == '3 Hidden-layer FCN':
        model_cls = Classifier
    else:
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
        raise ValueError('invalid algorithm: \'{}\''.format(algorithm_name))

    # training model
    model = model_cls(input_shape=dataset.num_features, device='cuda')
    model.train(dataset=dataset, epochs=2000)

    # evaluating model
    pred_dataset = model.predicted_dataset(dataset=dataset)
    model_metrics = ModelMetrics(dataset=dataset, classified_dataset=pred_dataset)

    #debiasing
    new_pred_dataset = None
    if issubclass(algorithm_cls, Classifier): # in process debiasing
        new_model = algorithm_cls(input_shape=dataset.num_features, device='cuda')
        new_model.train(dataset=dataset, epochs=2000)
        new_pred_dataset = new_model.predicted_dataset(dataset=dataset)
    else: # post process debaising
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
    dataset = dataset_cls(favorable=targetattr)
    prop = dataset.proportion(favorable=True)
    for attr in sensattrs:
        result['Proportion'][attr] = {
            SENSITIVE_GROUPS[dataset_name][attr][0]: prop[attr],
            SENSITIVE_GROUPS[dataset_name][attr][1]: 1 - prop[attr],
        }

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
    for dataset in ['Adult', 'German']:
    # for dataset in ['Compas']: # go through all data sets
        print('======dataset evaluation: {}======'.format(dataset))
        print(dataset_evaluate(dataset_name=dataset))
        for dataset_algorithm in ['LFR', 'Reweighing']: # go through all data set debias algorithms
        # for dataset_algorithm in ['LFR']: # go through all data set debias algorithms
            print('====dataset debiasing: {}, {}===='.format(dataset, dataset_algorithm))
            print(dataset_debias(dataset_name=dataset, algorithm_name=dataset_algorithm))
        for model in ['3 Hidden-layer FCN']: # go through all models
            for target_attr in TARGET_ATTR[dataset]:
                print('====model evaluation: {}-{}, {}===='.format(dataset, target_attr, model))
                print(model_evaluate(dataset_name=dataset, model_name=model, targetattr=target_attr))
                # for model_algorithm in ['Reject Option-SPd', 'Reject Option-AOd', 'Reject Option-EOd']: # go through all model debias algorithms
                for model_algorithm in ['Domain Independent', 'Adersarial Debiasing', 'Calibrated EOD-fnr', 'Calibrated EOD-fpr', 'Calibrated EOD-weighted', 'Reject Option-SPd', 'Reject Option-AOd', 'Reject Option-EOd']: # go through all model debias algorithms
                    print('====model debiasing: {}-{}, {}, {}===='.format(dataset,target_attr, model, model_algorithm))
                    print(model_debias(dataset_name=dataset, model_name=model, algorithm_name=model_algorithm,generalized=True, targetattr=target_attr))  
        
    # print(result) 

if __name__ == '__main__':
    test2()
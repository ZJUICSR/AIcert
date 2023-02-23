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

# dataset evaluation e.g.
def dataset_evaluate(dataset_name):
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
    
    # group and individual fairness metrics
    for target in dataset_cls.favorable.keys():
        dataset = dataset_cls(favorable=target)
        metrics = DatasetMetrics(dataset=dataset)
        result['Favorable Rate Difference'][target] = metrics.favorable_diff()
        result['Favorable Rate Ratio'][target] = metrics.favorable_ratio()
        if result['Consistency'] is None:
            result['Consistency'] = metrics.consistency()

    # proportion of goups
    dataset = dataset_cls()
    prop = dataset.proportion()
    for attr in dataset.privileged.keys():
        result['Proportion'][attr] = {
            SENSITIVE_GROUPS[dataset_name][attr][0]: prop[attr],
            SENSITIVE_GROUPS[dataset_name][attr][1]: 1 - prop[attr],
        }

    return result

# dataset debiasing e.g. the first element in the list is original, the second is improved
def dataset_debias(dataset_name, algorithm_name):
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

    dataset = dataset_cls()
    # algorithm = algorithm_cls(dataset=dataset)
    metrics = DatasetMetrics(dataset=dataset)
    # algorithm.fit()
    # new_dataset = algorithm.transform()
    # new_metrics = DatasetMetrics(new_dataset)

    # group and individual fairness metrics
    for target in dataset_cls.favorable.keys():
        dataset = dataset_cls(favorable=target)
        metrics = DatasetMetrics(dataset=dataset)
        # debiasing
        new_dataset = algorithm_cls(dataset).fit().transform()
        new_metrics = DatasetMetrics(new_dataset)

        result['Favorable Rate Difference'][target] = [metrics.favorable_diff(), new_metrics.favorable_diff()]
        result['Favorable Rate Ratio'][target] = [metrics.favorable_ratio(), new_metrics.favorable_ratio()]
        if result['Consistency'] is None:
            result['Consistency'] = [metrics.consistency(), new_metrics.consistency()]

    # proportion of goups
    dataset = dataset_cls()
    prop = dataset.proportion()
    for attr in dataset.privileged.keys():
        result['Proportion'][attr] = {
            SENSITIVE_GROUPS[dataset_name][attr][0]: prop[attr],
            SENSITIVE_GROUPS[dataset_name][attr][1]: 1 - prop[attr],
        }

    return result

# evaluate model fairness
def model_evaluate(dataset_name, model_name, metrics=['DI', 'DP', 'PE', 'EOD', 'PP', 'OMd', 'FOd', 'FNd']):
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

    dataset = dataset_cls()

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
        m = model_metrics.group_fairness_metrics(metrics=metric)
        result[METRICS_FULL_NAME[metric]] = m
    result['Consistency'] = model_metrics.consistency()
    # if result['Consistency'] is None:
    #     result['Consistency'] = [metrics.consistency(), new_metrics.consistency()]

    # proportion of goups given favorable prediction
    dataset = dataset_cls()
    prop = dataset.proportion(favorable=True)
    for attr in dataset.privileged.keys():
        result['Proportion'][attr] = {
            SENSITIVE_GROUPS[dataset_name][attr][0]: prop[attr],
            SENSITIVE_GROUPS[dataset_name][attr][1]: 1 - prop[attr],
        }

    return result

# model debiasing
def model_debias(dataset_name, model_name, algorithm_name, metrics=['DI', 'DP', 'PE', 'EOD', 'PP', 'OMd', 'FOd', 'FNd']):
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

    dataset = dataset_cls()

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
        algorithm.fit(dataset=dataset, dataset_pred=pred_dataset)
        new_pred_dataset = algorithm.transform(pred_dataset)
    new_metrics = ModelMetrics(dataset=dataset, classified_dataset=new_pred_dataset)

    for metric in metrics:
        m = model_metrics.group_fairness_metrics(metrics=metric)
        new_m = new_metrics.group_fairness_metrics(metrics=metric)
        result[METRICS_FULL_NAME[metric]] = [m, new_m]
    result['Consistency'] = [model_metrics.consistency(), new_metrics.consistency()]
    # if result['Consistency'] is None:
    #     result['Consistency'] = [metrics.consistency(), new_metrics.consistency()]

    # proportion of goups given favorable prediction
    dataset = dataset_cls()
    prop = dataset.proportion(favorable=True)
    for attr in dataset.privileged.keys():
        result['Proportion'][attr] = {
            SENSITIVE_GROUPS[dataset_name][attr][0]: prop[attr],
            SENSITIVE_GROUPS[dataset_name][attr][1]: 1 - prop[attr],
        }

    return result




    

if __name__ == '__main__':

    # result = dataset_evaluate('Compas')
    # {
    #     "Favorable Rate Difference":{
    #         "two_year_recid":{
    #             "sex":0.1279983309134417,
    #             "race":0.09745618466614947
    #         },
    #         "decile_score":{
    #             "sex":0.06719725452292646,
    #             "race":0.17530399000503327
    #         },
    #         "score_text":{
    #             "sex":0.05016678091961568,
    #             "race":0.17408231543674751
    #         }
    #     },
    #     "Favorable Rate Ratio":{
    #         "two_year_recid":{
    #             "sex":0.8026272456387218,
    #             "race":0.8400075282178671
    #         },
    #         "decile_score":{
    #             "sex":0.9044106851520114,
    #             "race":0.7705884934781674
    #         },
    #         "score_text":{
    #             "sex":0.9156710048919193,
    #             "race":0.739804470957015
    #         }
    #     },
    #     "Consistency":0.6579714841218336,
    #     "Proportion":{
    #         "sex":{
    #             "Female":0.1903758911211925,
    #             "Male":0.8096241088788075
    #         },
    #         "race":{
    #             "Caucasian":0.34073233959818533,
    #             "non-Caucasian":0.6592676604018146
    #         }
    #     }
    # }

    # result = dataset_debias('Compas', 'LFR')
    # {
    #     "Favorable Rate Difference":{
    #         "two_year_recid":[
    #             {
    #                 "sex":0.1279983309134417,
    #                 "race":0.09745618466614947
    #             },
    #             {
    #                 "sex":0.014617281006902072,
    #                 "race":0.03182395639086899
    #             }
    #         ],
    #         "decile_score":[
    #             {
    #                 "sex":0.06719725452292646,
    #                 "race":0.17530399000503327
    #             },
    #             {
    #                 "sex":0.002401440864518656,
    #                 "race":0.002949127549766506
    #             }
    #         ],
    #         "score_text":[
    #             {
    #                 "sex":0.05016678091961568,
    #                 "race":0.17408231543674751
    #             },
    #             {
    #                 "sex":-0.013771326625762637,
    #                 "race":0.03432398356126665
    #             }
    #         ]
    #     },
    #     "Favorable Rate Ratio":{
    #         "two_year_recid":[
    #             {
    #                 "sex":0.8026272456387218,
    #                 "race":0.8400075282178671
    #             },
    #             {
    #                 "sex":0.9843860861971728,
    #                 "race":0.966335120578472
    #             }
    #         ],
    #         "decile_score":[
    #             {
    #                 "sex":0.9044106851520114,
    #                 "race":0.7705884934781674
    #             },
    #             {
    #                 "sex":0.9975985591354813,
    #                 "race":0.9970508724502335
    #             }
    #         ],
    #         "score_text":[
    #             {
    #                 "sex":0.9156710048919193,
    #                 "race":0.739804470957015
    #             },
    #             {
    #                 "sex":1.0152509979126023,
    #                 "race":0.9633587119648002
    #             }
    #         ]
    #     },
    #     "Consistency":[
    #         0.6586519766688193,
    #         0.9999675955930006
    #     ],
    #     "Proportion":{
    #         "sex":{
    #             "Female":0.1903758911211925,
    #             "Male":0.8096241088788075
    #         },
    #         "race":{
    #             "Caucasian":0.34073233959818533,
    #             "non-Caucasian":0.6592676604018146
    #         }
    #     }
    # }

    # result = model_evaluate('Compas', '3 Hidden-layer FCN')
    # {
    #     "Consistency":0.9690134,
    #     "Proportion":{
    #         "sex":{
    #             "Female":0.9390425215581326,
    #             "Male":0.060957478441867385
    #         },
    #         "race":{
    #             "Caucasian":0.9717514124293786,
    #             "non-Caucasian":0.02824858757062143
    #         }
    #     },
    #     "Dsiaprate Impact":{
    #         "sex":0.7365824116239865,
    #         "race":0.7516651530804042
    #     },
    #     "Demographic Parity":{
    #         "sex":0.20131755649134164,
    #         "race":0.17854527236833662
    #     },
    #     "Predictive Equality":{
    #         "sex":0.05917126675917461,
    #         "race":0.011487003319321287
    #     },
    #     "Equal Odds":{
    #         "sex":0.31581726722288417,
    #         "race":0.2978373870770743
    #     },
    #     "Overall Misclassification Difference":{
    #         "sex":-0.03995248212757441,
    #         "race":0.008757165242879394
    #     },
    #     "False Omission Difference":{
    #         "sex":0.015086747393038169,
    #         "race":0.05409303503612756
    #     },
    #     "False Negative Difference":{
    #         "sex":-0.1368840573129051,
    #         "race":-0.10071607421255457
    #     }
    # }

    # result = model_debias('Compas', '3 Hidden-layer FCN', 'Adersarial Debiasing')
    # {
    #     "Consistency":[
    #         0.96968377,
    #         0.9789661
    #     ],
    #     "Proportion":{
    #         "sex":{
    #             "Female":0.9390425215581326,
    #             "Male":0.060957478441867385
    #         },
    #         "race":{
    #             "Caucasian":0.9717514124293786,
    #             "non-Caucasian":0.02824858757062143
    #         }
    #     },
    #     "Dsiaprate Impact":[
    #         {
    #             "sex":0.8174554678767078,
    #             "race":0.7641244120047502
    #         },
    #         {
    #             "sex":0.7239927198784254,
    #             "race":0.7139070490458581
    #         }
    #     ],
    #     "Demographic Parity":[
    #         {
    #             "sex":0.14184016793054555,
    #             "race":0.18495351291037965
    #         },
    #         {
    #             "sex":0.2201004006659315,
    #             "race":0.2183441202733588
    #         }
    #     ],
    #     "Predictive Equality":[
    #         {
    #             "sex":0.07810102697602617,
    #             "race":0.026048302175713967
    #         },
    #         {
    #             "sex":0.05851263888488267,
    #             "race":-0.004159869267948313
    #         }
    #     ],
    #     "Equal Odds":[
    #         {
    #             "sex":0.19747520219644127,
    #             "race":0.31098108252676304
    #         },
    #         {
    #             "sex":0.35978561223270633,
    #             "race":0.3867523748823671
    #         }
    #     ],
    #     "Overall Misclassification Difference":[
    #         {
    #             "sex":-0.04190531340080644,
    #             "race":-0.009242609681052283
    #         },
    #         {
    #             "sex":-0.04550526060317045,
    #             "race":0.017236315965197146
    #         }
    #     ],
    #     "False Omission Difference":[
    #         {
    #             "sex":0.03934459079900704,
    #             "race":0.0012883635189267495
    #         },
    #         {
    #             "sex":-0.01400162443519265,
    #             "race":0.047004431898804266
    #         }
    #     ],
    #     "False Negative Difference":[
    #         {
    #             "sex":-0.07872199366082701,
    #             "race":-0.11237505821055686
    #         },
    #         {
    #             "sex":-0.15412909026510094,
    #             "race":-0.12611724899720364
    #         }
    #     ]
    # }

    # testing
    for dataset in ['Compas', 'Adult', 'German']: # go through all data sets
        print('======dataset evaluation: {}======'.format(dataset))
        print(dataset_evaluate(dataset_name=dataset))
        for dataset_algorithm in ['LFR', 'Reweighing']: # go through all data set debias algorithms
            print('====dataset debiasing: {}, {}===='.format(dataset, dataset_algorithm))
            print(dataset_debias(dataset_name=dataset, algorithm_name=dataset_algorithm))
        for model in ['3 Hidden-layer FCN']: # go through all models
            print('====model evaluation: {}, {}===='.format(dataset, model))
            print(model_evaluate(dataset_name=dataset, model_name=model))
            for model_algorithm in ['Domain Independent', 'Adersarial Debiasing', 'Calibrated EOD-fnr', 'Calibrated EOD-fpr', 'Calibrated EOD-weighted']: # go through all model debias algorithms
                print('====model debiasing: {}, {}, {}===='.format(dataset, model, model_algorithm))
                print(model_debias(dataset_name=dataset, model_name=model, algorithm_name=model_algorithm))  
        
    # print(result)
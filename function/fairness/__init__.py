
from .tabular.dataset_analysis import calculate_distribution_uniformity
from function.fairness.tabular.api import dataset_evaluate, dataset_debias, model_evaluate, model_debias, dataset_analysis
from function.fairness.image.api import model_evaluate as image_model_evaluate
from function.fairness.image.api import model_debias as image_model_debias

__version__="0.2"

def run_dataset_evaluate(dataset_name, sensattrs=[], targetattrs=[], staAttrList=[], logging=None):
    res = dataset_evaluate(dataset_name, sensattrs=sensattrs, targetattrs=targetattrs, logger=logging)
    res1 = dataset_analysis(dataset_name, attrs=staAttrList, targetattrs=staAttrList, logger=logging)
    res['Corelation coefficients'] = res1['Correlation coefficients']
    res['Overall Correlation'] = res1['Overall Correlation']
    for key in res1['Proportion'].keys():
        if key not in res['Proportion'].keys():
            res['Proportion'][key] = res1['Proportion'][key]
    res['Overall uniformity'] = calculate_distribution_uniformity(res['Proportion'])
    return res

def run_dataset_debias(dataset_name, algorithm_name, sensattrs=[], targetattrs=[], staAttrList=[], logging=None):
    res = dataset_debias(dataset_name, algorithm_name, sensattrs, targetattrs, logger=logging)
    res1 = dataset_analysis(dataset_name, attrs=staAttrList, targetattrs=staAttrList, logger=logging)
    res['Corelation coefficients'] = res1['Correlation coefficients']
    res['Overall Correlation'] = res1['Overall Correlation']
    for key in res1['Proportion'].keys():
        if key not in res['Proportion'].keys():
            res['Proportion'][key] = res1['Proportion'][key]
    res['Overall uniformity'] = calculate_distribution_uniformity(res['Proportion'])
    return res

def run_model_evaluate(dataset_name, model_path='', model_name='', metrics=['DI', 'DP', 'PE', 'EOD', 'PP', 'OMd', 'FOd', 'FNd'], sensattrs=[], targetattr=None, staAttrList=[], generalized=False, logging=None):
    res = model_evaluate(dataset_name, model_name, metrics, sensattrs, targetattr, generalized, logger=logging, model_path=model_path)
    res1 = dataset_analysis(dataset_name, attrs=staAttrList, targetattrs=staAttrList, logger=logging)
    res['Corelation coefficients'] = res1['Correlation coefficients']
    res['Overall Correlation'] = res1['Overall Correlation']
    for key in res1['Proportion'].keys():
        if key not in res['Proportion'].keys():
            res['Proportion'][key] = res1['Proportion'][key]
    res['Overall uniformity'] = calculate_distribution_uniformity(res['Proportion'])
    return res

def run_model_debias(dataset_name, model_name, algorithm_name, metrics=['DI', 'DP', 'PE', 'EOD', 'PP', 'OMd', 'FOd', 'FNd'], 
                     sensattrs=[], targetattr=None, staAttrList=[], generalized=False, logging=None, model_path='', save_folder=''):
    res = model_debias(dataset_name, model_name, algorithm_name, metrics, sensattrs, targetattr, generalized, logger=logging, model_path=model_path, save_folder=save_folder)
    res1 = dataset_analysis(dataset_name, attrs=staAttrList, targetattrs=staAttrList, logger=logging)
    res['Corelation coefficients'] = res1['Correlation coefficients']
    res['Overall Correlation'] = res1['Overall Correlation']
    for key in res1['Proportion'].keys():
        if key not in res['Proportion'].keys():
            res['Proportion'][key] = res1['Proportion'][key]
    res['Overall uniformity'] = calculate_distribution_uniformity(res['Proportion'])
    return res


def run_image_model_evaluate(dataset_name, model_path='', model_name="Resnet50", metrics=['mPre', 'mFPR', 'mFNR', 'mTNR', 'mTPR', 'mAcc', 'mF1', 'mBA'], test_mode=True, logging=None):
    res = image_model_evaluate(dataset_name, model_path, model_name, metrics, test_mode, logger=logging)
    # res1 = dataset_analysis(dataset_name, attrs=staAttrList, targetattrs=staAttrList, logger=logging)
    # res['Corelation coefficients'] = res1['Correlation coefficients']
    # res['Overall Correlation'] = res1['Overall Correlation']
    # for key in res1['Proportion'].keys():
    #     if key not in res['Proportion'].keys():
    #         res['Proportion'][key] = res1['Proportion'][key]
    # res['Overall uniformity'] = calculate_distribution_uniformity(res['Proportion'])
    return res

def run_image_model_debias(dataset_name, model_name="Resnet50", algorithm_name='', metrics=['mPre', 'mFPR', 'mFNR', 'mTNR', 'mTPR', 'mAcc', 'mF1', 'mBA'], 
                           test_mode=True, logging=None, save_folder=''):
    res = image_model_debias(dataset_name, model_name, algorithm_name, metrics, test_mode, logger=logging, save_folder=save_folder)
    # res1 = dataset_analysis(dataset_name, attrs=staAttrList, targetattrs=staAttrList, logger=logging)
    # res['Corelation coefficients'] = res1['Correlation coefficients']
    # res['Overall Correlation'] = res1['Overall Correlation']
    # for key in res1['Proportion'].keys():
    #     if key not in res['Proportion'].keys():
    #         res['Proportion'][key] = res1['Proportion'][key]
    # res['Overall uniformity'] = calculate_distribution_uniformity(res['Proportion'])
    return res

if __name__=='__main__':
    dataset_name = "German"
    model_name = "3 Hidden-layer FCN"
    # ['Domain Independent', 'Adersarial Debiasing', 'Calibrated EOD-fnr', 'Calibrated EOD-fpr', 'Calibrated EOD-weighted', 'Reject Option-SPd', 'Reject Option-AOd', 'Reject Option-EOd']
    algorithm_name = 'Domain Independent'
    metrics=['DI', 'DP']
    sensattrs = ["age"]
    targetattr = ["credit_amount"]
    generalized = False
    result = run_image_model_debias(dataset_name, model_name, algorithm_name, metrics, sensattrs, targetattr, generalized)
    print(result)
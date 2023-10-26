import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import copy
import os.path as osp

ROOT = osp.dirname(osp.abspath(__file__))
# names of groups according to sensitive attribute, [privileged, unprivileged]
seed = 11
SENSITIVE_GROUPS = {
    'Compas': {
        "sex": ['Female', 'Male'],
        "race": ['Caucasian', 'non-Caucasian'],
    },
    'Adult': {
        "sex": ['Male', 'Female'],
        "race": ['White', 'non-White'],
    },
    'German': {
        "sex": ['male', 'female'],
        "age": ['>25', '<=25'],
    }
}

class FairnessDataset():
    def __init__(self, path, favorable=None, privileged=None, features_to_keep=None, features_to_drop=None):

        self.privileged = privileged
        self.features_to_drop = features_to_drop
        self.features_to_keep = features_to_keep # labels with be automatically removed from training features
        self.favorable = favorable
        self.path = path

        self.load_data()
        self.custom_preprocess()
        self.preprocess()

    def load_data(self):
        self.df = pd.read_csv(self.path)

    def custom_preprocess(self):
        # dataset-specific preprocess
        pass

    def preprocess(self):
        # pre-process sensitive attributes and label
        for key in self.privileged:
            if callable(self.privileged[key]): # if priviledged is given by a function
                self.df[key] = self.df[key].map(self.privileged[key])
            else:
                self.df[key] = np.where(self.df[key].isin(self.privileged[key]), 1, 0) # 1 for privileged, 0 for unprivileged
        for key in self.favorable:
            if callable(self.favorable[key]):
                self.df[key] = self.df[key].map(self.favorable[key])
            else:
                self.df[key] = np.where(self.df[key].isin(self.favorable[key]), 1, 0)
        # print(self.df)

        # select sensitive attributes
        self.Z = self.df[self.privileged.keys()]

        # select features
        target = list(self.favorable.keys())[0]
        categotical_features = copy.deepcopy(self.categotical_features)
        if target in categotical_features:
            categotical_features.remove(target)
        categotical_features = intersection(categotical_features, self.features_to_keep)
        for i in self.features_to_drop:
            if i in categotical_features:
                categotical_features.remove(i)
        selected = (set(self.features_to_keep) | set(categotical_features)) - set(self.features_to_drop) - set(self.favorable.keys())
        self.df = self.df[list(selected | set(self.favorable.keys()))]

        # process categorial features
        self.df = pd.get_dummies(self.df, columns=categotical_features) 

        # # drop na
        # self.df = self.df.dropna()
        # sort data frame
        self.df = self.df.sort_index(axis=1)

        # split feature and label
        self.X = self.df.drop(self.favorable, axis=1)# if self.favorable in self.df else self.df # (5278, 7)
        self.Y = self.df[list(self.favorable.keys())] # (5278, 1)
        self.num_features = self.X.shape[1]

        self.weights = np.ones((self.X.shape[0], 1))

    def proportion(self, favorable=None, thresh=0.5):
        result = {}
        mask = None
        if favorable is None:
            mask = np.ones_like(self.Y) == 1
        elif favorable:
            mask = np.array(self.Y) > thresh
        else:
            mask = np.array(self.Y) < thresh
        # go through sensitive attributes
        for (i, attr) in enumerate(self.privileged.keys()):
            w_z = self.weights[(np.array(self.Z)[:, i] == 1) & np.squeeze(mask)]
            total = np.sum(self.weights[mask])
            result[attr] = np.sum(w_z) / total
        return result

    def split(self, train_ratio):
        return map(np.array, train_test_split(self.X, self.Y, self.Z, train_size=train_ratio, random_state=seed))

    def __len__(self):
        return self.X.shape[0]

    def copy(self, deepcopy=True):
        return copy.deepcopy(self) if deepcopy else self


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

class CompasDataset(FairnessDataset):

    # dataset-specific information
    sensitive_attr = ['sex', 'race']
    favorable = {
        'two_year_recid': [0],
        'decile_score': lambda x: 1 if x <= 5 else 0,
        'score_text': ['Low']
        }
    privileged = {
        "sex": ['Female'],
        "race": ['Caucasian'],
    }
    features_to_keep=['sex', 'age', 'age_cat', 'race',
                     'juv_fel_count', 'juv_misd_count', 'juv_other_count',
                     'priors_count', 'c_charge_degree', 'c_charge_desc', 'two_year_recid'
                     ]
    features_to_drop=['c_charge_desc']
    categotical_features = ['age_cat', 'c_charge_degree', 'c_charge_desc']
    # categotical_features = ['age_cat', 'c_charge_degree', 'c_charge_desc']

    def __init__(self, path=osp.join(ROOT,r'datasets/Compas/compas-scores-two-years.csv'), favorable=None, privileged=None, features_to_keep=None, features_to_drop=None): # TODO: use key word argues

        privileged = privileged if privileged else CompasDataset.privileged
        features_to_drop = features_to_drop if features_to_drop else CompasDataset.features_to_drop
        features_to_keep = features_to_keep if features_to_keep else CompasDataset.features_to_keep
        favorable = favorable if favorable else 'two_year_recid'
        if isinstance(favorable, str):
            if favorable in CompasDataset.favorable.keys():
                favorable = {favorable: CompasDataset.favorable[favorable]}
            else:
                raise ValueError('target label \'{}\' ')
    
        super().__init__(path, favorable, privileged, features_to_keep, features_to_drop) #num of feature 401
        # self.feature_select()

    def custom_preprocess(self):
        # custom preprocessing
        # Perform the same preprocessing as the original analysis: https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb
        self.df = self.df[(self.df['days_b_screening_arrest'] <= 30) 
            & (self.df['days_b_screening_arrest'] >= -30)
            & (self.df.is_recid != -1)
            & (self.df.c_charge_degree != 'O')
            & (self.df.score_text != 'N/A')]

class AdultDataset(FairnessDataset):

    # dataset-specific information
    favorable = {
        'income-per-year': ['>50K', '>50K.'],
        'capital-gain': lambda x: 1 if x > 0 else 0,
        'capital-loss': lambda x: 1 if x == 0 else 0,
        }
    privileged = {
        "sex": ['Male'],
        "race": ['White'],
    }
    features_to_keep=['age', 'workclass', 'fnlwgt', 'education',
            'education-num', 'marital-status', 'occupation', 'relationship',
            'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
            'native-country', 'income-per-year'] # include all featrue
    features_to_drop=['fnlwgt']
    categotical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'native-country', 'income-per-year']
    # categotical_features = ['age_cat', 'c_charge_degree', 'c_charge_desc']

    def __init__(self, train_path=r'datasets/Adult/adult.data', test_path=r'datasets/Adult/adult.test', favorable=None, privileged=None, features_to_keep=None, features_to_drop=None): # TODO: use key word argues

        privileged = privileged if privileged else AdultDataset.privileged
        features_to_drop = features_to_drop if features_to_drop else AdultDataset.features_to_drop
        features_to_keep = features_to_keep if features_to_keep else AdultDataset.features_to_keep
        favorable = favorable if favorable else 'income-per-year'
        if isinstance(favorable, str) and favorable in AdultDataset.favorable.keys():
            favorable = {favorable: AdultDataset.favorable[favorable]}

        self.train_path = osp.join(ROOT,train_path)
        self.test_path = osp.join(ROOT,test_path)
        super().__init__('', favorable, privileged, features_to_keep, features_to_drop) #num of feature 401
        # self.feature_select()
    
    def load_data(self):
        column_names = ['age', 'workclass', 'fnlwgt', 'education',
            'education-num', 'marital-status', 'occupation', 'relationship',
            'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
            'native-country', 'income-per-year']
        # # read train and test file
        # train = pd.read_csv(self.train_path, header=None, names=column_names,
        #         skipinitialspace=True)
        # test = pd.read_csv(self.test_path, header=None, names=column_names,
        #         skipinitialspace=True)
        # # merge train and test file
        # self.df = pd.concat([test, train], ignore_index=True)
        # too many samples, take first 2000 samples from test set
        self.df = pd.read_csv(self.test_path, header=None, names=column_names, skipinitialspace=True)
        self.df = self.df.iloc[:2000]
        return

class GermanDataset(FairnessDataset):

    # dataset-specific information
    favorable = {
        'credit': [1],
        # 'number_of_credits': lambda x: 1 if x >= 2 else 0,
        'skill_level': ['A173', 'A174'],
        'credit_amount': lambda x: 1 if x > 3000 else 0
        }
    privileged = {
        "sex": ['male'],
        "age": lambda x: 1 if x > 25 else 0,
    }
    features_to_keep=['status', 'month', 'credit_history',
            'purpose', 'credit_amount', 'savings', 'employment',
            'investment_as_income_percentage', 'personal_status',
            'other_debtors', 'residence_since', 'property', 'age',
            'installment_plans', 'housing', 'number_of_credits',
            'skill_level', 'people_liable_for', 'telephone',
            'foreign_worker', 'credit', 'sex'] # include all features
    features_to_drop=['personal_status']
    categotical_features = ['status', 'credit_history', 'purpose',
                     'savings', 'employment', 'other_debtors', 'property',
                     'installment_plans', 'housing', 'skill_level', 'telephone',
                     'foreign_worker', 'personal_status']
    # categotical_features = ['age_cat', 'c_charge_degree', 'c_charge_desc']

    def __init__(self, path=osp.join(ROOT,r'datasets/German/german.data'), favorable=None, privileged=None, features_to_keep=None, features_to_drop=None): # TODO: use key word argues
        privileged = privileged if privileged else GermanDataset.privileged
        features_to_drop = features_to_drop if features_to_drop else GermanDataset.features_to_drop
        features_to_keep = features_to_keep if features_to_keep else GermanDataset.features_to_keep
        favorable = favorable if favorable else 'credit'
        if isinstance(favorable, str) and favorable in GermanDataset.favorable.keys():
            favorable = {favorable: GermanDataset.favorable[favorable]}

        super().__init__(path, favorable, privileged, features_to_keep, features_to_drop) #num of feature 401
        # self.feature_select()
    
    def load_data(self):
        # read train and test file
        self.df = pd.read_csv(self.path, sep=' ', names=GermanDataset.features_to_keep)

    def custom_preprocess(self):
        status_map = {'A91': 'male', 'A93': 'male', 'A94': 'male',
                  'A92': 'female', 'A95': 'female'}
        self.df['sex'] = self.df['personal_status'].replace(status_map)
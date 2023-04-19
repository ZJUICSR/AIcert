import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import normalized_mutual_info_score
import math
from fairness_datasets import FairnessDataset, intersection

def correlation_analysis(dataframe, attr1, attr2, attr1_type, attr2_type):
    """
    计算两个变量之间的各种相关性分析结果，包括Pearson相关系数、Spearman秩相关系数、
    Kendall Tau相关系数和互信息。对于不适用于输入数据类型的相关性分析，该相关性
    的分析结果为None。
    
    Args:
    dataframe: pd.DataFrame, 数据集的dataframe
    attr1: str, 数据集中的第一个属性名称
    attr2: str, 数据集中的第二个属性名称
    attr1_type: str, 第一个属性的数据类型，取值为'continuous'或'discrete'或'ordinal'
    attr2_type: str, 第二个属性的数据类型，取值为'continuous'或'discrete'或'ordinal'
    
    Returns:
    results: dict, 包含各种相关性分析结果的字典
    """

    # 判断变量类型，如果变量是离散变量或离散型连续变量，则将其转换为数值型
    # if attr1_type in ['discrete', 'ordinal']:
    #     var1_values = pd.Categorical(df[attr1]).codes
    # else:
    #     var1_values = df[attr1].values

    # if attr2_type in ['discrete', 'ordinal']:
    #     var2_values = pd.Categorical(df[attr1]).codes
    # else:
    #     var2_values = df[attr1].values

    results = {}

    # Pearson相关系数
    if attr1_type == 'continuous' and attr2_type == 'continuous':
        results['pearson'] = pearsonr(dataframe[attr1], dataframe[attr2])[0]
    else:
        results['pearson'] = None

    # Spearman秩相关系数和Kendall Tau相关系数
    if attr1_type in ['continuous', 'ordinal'] and attr2_type in ['continuous', 'ordinal']:
        results['spearman'] = spearmanr(dataframe[attr1], dataframe[attr2])[0]
        results['kendalltau'] = kendalltau(dataframe[attr1], dataframe[attr2])[0]
    else:
        results['spearman'] = None
        results['kendalltau'] = None

    # 互信息
    results['mutual_info'] = normalized_mutual_info_score(dataframe[attr1], dataframe[attr2])
    # if attr1_type == 'discrete' and attr2_type == 'discrete':
    #     results['mutual_info'] = mutual_info_score(dataframe[attr1], dataframe[attr2])
    # else:
    #     results['mutual_info'] = None

    return results


def calculate_category_proportions(df, attribute_names, threshold=0.05):
    """
    Function to calculate the proportions of each category for given attributes in a DataFrame.
    When an attribute has too many categories, the categories with proportions less than the threshold will be combined into an "others" category.

    Args:
    df (pandas.DataFrame): The dataset in the form of a DataFrame.
    attribute_names (list): A list of strings, each representing the name of an attribute in the dataset for which proportions need to be calculated.
    threshold (float): The threshold value to decide if categories with small proportions should be combined into an "others" category. Defaults to 0.05.

    Returns:
    dict: A dictionary where the keys are attribute names and the values are dictionaries representing the proportions of each category for that attribute.
    """
    # Initialize an empty dictionary to store the results
    proportions = {}

    # Iterate through the given attribute names
    for attribute in attribute_names:
        # Check if the attribute exists in the DataFrame
        if attribute in df.columns:
            # Calculate the proportion of each category for the current attribute
            category_proportions = df[attribute].value_counts(normalize=True)

            # Combine categories with proportions less than the threshold into an "others" category
            others_proportion = category_proportions[category_proportions < threshold].sum()
            filtered_proportions = category_proportions[category_proportions >= threshold].to_dict()

            # Add the "others" category and its proportion to the results, if applicable
            if others_proportion > 0:
                filtered_proportions['others'] = others_proportion

            # Add the proportions to the results dictionary
            proportions[attribute] = filtered_proportions
        else:
            print(f"Attribute '{attribute}' not found in the dataset.")

    return proportions

def calculate_overall_correlation(correlation_data):
    total_corr_values = 0
    count = 0

    for item in correlation_data:
        for key, value in item["values"].items():
            if value is not None:
                total_corr_values += abs(value)
                count += 1

    if count > 0:
        overall_correlation = total_corr_values / count
    else:
        overall_correlation = None

    return overall_correlation

def calculate_distribution_uniformity(distribution_data):
    normalized_entropy_sum = 0
    total_attributes = len(distribution_data)

    for _, category_probs in distribution_data.items():
        entropy = 0
        num_categories = len(category_probs)
        max_entropy = math.log2(num_categories)

        for category, prob in category_probs.items():
            if prob > 0:
                entropy -= prob * math.log(prob, 2)

        normalized_entropy = entropy / max_entropy
        normalized_entropy_sum += normalized_entropy

    average_normalized_entropy = normalized_entropy_sum / total_attributes

    return average_normalized_entropy

def preprocess(self: FairnessDataset, factorize=False):
    # reload data
    self.load_data()
    self.custom_preprocess()
    
    # pre-process all sensitive attributes and target label, add redundant collumn for binarized variable
    if factorize:
        for key in self.__class__.privileged:
            if callable(self.__class__.privileged[key]): # if priviledged is given by a function
                self.df[key] = self.df[key].map(self.__class__.privileged[key])
            else:
                self.df[key] = np.where(self.df[key].isin(self.__class__.privileged[key]), 1, 0) # 1 for privileged, 0 for unprivileged
        for key in self.__class__.favorable:
            if callable(self.__class__.favorable[key]):
                self.df[key] = self.df[key].map(self.__class__.favorable[key])
            else:
                self.df[key] = np.where(self.df[key].isin(self.__class__.favorable[key]), 1, 0)
    # print(self.df)

    # process categorial features
    # # binarizied 
    # categotical_features = self.__class__.categotical_features + [key + '_' for key in self.__class__.privileged]
        for cat in self.__class__.categotical_features:
            self.df[cat], _ = pd.factorize(self.df[cat])
    # self.df = pd.get_dummies(self.df, columns=categotical_features) 
    
    # # drop na
    # self.df = self.df.dropna()
    # sort data frame
    self.df = self.df.sort_index(axis=1)
    return self

if __name__ == "__main__":
    
    # 示例
    data = {'attr1': [1, 2, 3, 4, 5],
            'attr2': [2, 3, 4, 5, 6],
            'attr3': [0, 1, 0, 1, 0],
            'attr4': [1, 0, 1, 0, 1]}
    df = pd.DataFrame(data)

    results = correlation_analysis(df, 'attr1', 'attr2', 'continuous', 'continuous')
    print(results)
    
    # Create a sample dataset
    data = {
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male'],
        'Color': ['Red', 'Blue', 'Green', 'Blue', 'Red', 'Green', 'Blue']
    }
    df = pd.DataFrame(data)

    # Define the attributes for which proportions need to be calculated
    attributes_to_analyze = ['Gender', 'Color']

    # Call the function and print the results
    result = calculate_category_proportions(df, attributes_to_analyze)
    print(result)
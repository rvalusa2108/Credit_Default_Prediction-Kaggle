# -*- coding: utf-8 -*-
"""
Created on Fri May  1 22:55:15 2020

@author: Raghu
"""

import logging
import os
import yaml
from pprint import pformat
from pathlib import Path
import arrow
from joblib import Parallel, delayed
from collections import Counter
import random
import itertools
from itertools import permutations
import logging
from io import StringIO
import pandas as pd
pd.options.display.max_rows=999
pd.set_option('display.max_columns', 500)
import numpy as np
import missingno as msno
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import make_scorer, f1_score, accuracy_score, roc_auc_score, f1_score, mean_squared_error, mean_squared_log_error
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import StackingClassifier, StackingRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.feature_selection import SelectFromModel, VarianceThreshold

from kaggle.api.kaggle_api_extended import KaggleApi

import pickle
import pickle as cPickle
from collections import OrderedDict

import matplotlib.pyplot as plt

from numpy.random.mtrand import _rand as global_randstate
np.random.seed(42)

def set_pandas_display_options() -> None:
    """Set pandas display options."""
    # Ref: https://stackoverflow.com/a/52432757/
    display = pd.options.display

    display.max_columns = 1000
    display.max_rows = 1000
    display.max_colwidth = 199
    display.width = None
    display.precision = 8  # set as needed

set_pandas_display_options()

logger = logging.getLogger(__name__)
logger.handlers.clear()
if not logger.hasHandlers():
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
    file_handler = logging.FileHandler('credit_default_prediction_classification.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
#=============================================================================

def dataset_datatypes_counts(data, ):
    #Caterogies mentioned as numerical values need to be considered as categories
    #Be cautious about this = pClass = 1,2,3(it is a category) in Titanic dataset
    buf = StringIO()
    data.info(buf=buf)
    logger.info(f"""\n:: dataset(DataFrame) Info ::\n\n{buf.getvalue()}""")

    string_vars_list = []
    categorical_vars_list = []
    numerical_vars_list = []
    # numerical_vars_list = list(data.select_dtypes(include=[np.number]).columns)

    for i in list(data.select_dtypes(include=[np.number]).columns):
        if len(data[i].unique()) <= 5:
            categorical_vars_list.append(i)
        else:
            numerical_vars_list.append(i)

    for cat_name in list(data.select_dtypes(include=['category', 'object']).columns):
        if len(data[cat_name].unique()) > 30:
            string_vars_list.append(cat_name)
        else:
            categorical_vars_list.append(cat_name)

    temporal_vars_list = list(data.select_dtypes(include=['datetime', 'datetime64', 'timedelta', 'timedelta64', 'datetimetz']).columns)

    # print("-"*20)
    # print(f'::numerical_vars_list::\n{numerical_vars_list}\n{len(numerical_vars_list)}')
    # print("-"*20)
    # print(f'::categorical_vars_list::\n{categorical_vars_list}\n{len(categorical_vars_list)}')
    # print("-"*20)
    # print(f'::string_vars_list::\n{string_vars_list}\n{len(string_vars_list)}')
    # print("-"*20)
    # print(f'::temporal_vars_list::\n{temporal_vars_list}\n{len(temporal_vars_list)}')
    # print("-"*20)

    logger.info(f"""\n:: numerical_features :: {len(numerical_vars_list)}\n{chr(10).join(sorted(numerical_vars_list))}\n""")
    logger.info(f"\n:: categorical_features :: {len(categorical_vars_list)}\n{chr(10).join(sorted(categorical_vars_list))}\n")
    logger.info(f"\n:: string/text_features :: {len(string_vars_list)}\n{string_vars_list}\n")
    logger.info(f"\n:: temporal/datetime_features :: {len(temporal_vars_list)}\n{chr(10).join(sorted(temporal_vars_list))}\n")

    return numerical_vars_list, categorical_vars_list, string_vars_list, temporal_vars_list

#=============================================================================

def train_val_test_split(X, y):
    #splitting the data into train and test
    # X_train, X_test, y_train, y_test = train_test_split(X,
    #                                                     y,
    #                                                     test_size=0.2,
    #                                                     random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X,
                                                        y,
                                                        test_size=0.1,
                                                        random_state=42)


    # X_train, X_val, y_train, y_val = train_test_split(X_train,
    #                                                   y_train,
    #                                                   test_size=0.2,
    #                                                   random_state=42,
    #                                                   stratify=y_train)

    # print(f'X_train shape : {X_train.shape} :: X_val shape : {X_val.shape} :: X_test shape : {X_test.shape}')
    # print(f'y_train shape : {y_train.shape} :: y_val shape : {y_val.shape} :: y_test shape : {y_test.shape}')

    print(f'X_train shape : {X_train.shape} :: X_val shape : {X_val.shape}')
    print(f'y_train shape : {y_train.shape} :: y_val shape : {y_val.shape}')


    X_train = X_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    # X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    # y_test = y_test.reset_index(drop=True)

    # return X_train, X_val, X_test, y_train, y_val, y_test
    return X_train, X_val, y_train, y_val



#=============================================================================

class PassthroughTransformer(BaseEstimator, TransformerMixin):

    # I corrected the `fit()` method here, it should take X, y as input
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self.X = X
        return X

    # I have corrected the output here, See point 2
    def get_feature_names(self):
        return self.X.columns.tolist()
#=============================================================================

def Outlier_Detection(X, y, outlier_detection_transform_dict):
    X_copy = X.copy()
    y_copy = y.copy()
    outlier_indices = []
    # iterate over features(columns)
    for col in outlier_detection_transform_dict['features_list']:
        # 1st quartile (25%)
        Q1 = np.percentile(X_copy[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(X_copy[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1

        # outlier step
        outlier_step = 1.5 * IQR

        # Determine a list of indices of outliers for feature col
        outlier_list_col = X_copy[(X_copy[col] < Q1 - outlier_step) | (X_copy[col] > Q3 + outlier_step )].index

        # append the found outlier indices for col to the list of outlier indices
        outlier_indices.extend(outlier_list_col)

    # select observations containing more than 'min_outliers' outliers
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > outlier_detection_transform_dict['min_outliers'])

    logger.info(f"\nOutlier rows in features - {outlier_detection_transform_dict['features_list']} :\n{X_copy.loc[multiple_outliers].to_string()}")
    logger.info(f"Shape of input X dataframe: {X_copy.shape}")
    logger.info(f"Shape of label y: {y_copy.shape}")
    logger.info(f"No. of outliers detected: {len(multiple_outliers)}")

    if outlier_detection_transform_dict['drop_outliers']:
        X_copy = X_copy.drop(multiple_outliers, axis = 0).reset_index(drop=True)
        y_copy = y_copy.drop(index=multiple_outliers, axis = 0).reset_index(drop=True)
        logger.info(f"\nShape of input dataframe X post outliers rows dropped because of outliers in features - {outlier_detection_transform_dict['features_list']}: {X_copy.shape}")
        logger.info(f"\nShape of label y post outliers rows dropped: {y_copy.shape}")

    return X_copy, y_copy


'''
class Custom_Outlier_Detection_Transformer(BaseEstimator, TransformerMixin):
    def __init__(self, outlier_detection_transform_dict, loginfo=False):
        self.outlier_detection_transform_dict = outlier_detection_transform_dict
        self.loginfo = loginfo

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X):
        return self


    def fit_transform(self, X, y):
        X_copy = X.copy()
        y_copy = y.copy()
        outlier_indices = []
        # iterate over features(columns)
        for col in self.outlier_detection_transform_dict['features_list']:
            # 1st quartile (25%)
            Q1 = np.percentile(X_copy[col], 25)
            # 3rd quartile (75%)
            Q3 = np.percentile(X_copy[col],75)
            # Interquartile range (IQR)
            IQR = Q3 - Q1

            # outlier step
            outlier_step = 1.5 * IQR

            # Determine a list of indices of outliers for feature col
            outlier_list_col = X_copy[(X_copy[col] < Q1 - outlier_step) | (X_copy[col] > Q3 + outlier_step )].index

            # append the found outlier indices for col to the list of outlier indices
            outlier_indices.extend(outlier_list_col)

        # select observations containing more than 'min_outliers' outliers
        outlier_indices = Counter(outlier_indices)
        multiple_outliers = list( k for k, v in outlier_indices.items() if v > self.outlier_detection_transform_dict['min_outliers'])

        if self.loginfo:
            logger.info(f"\nOutlier rows in features - {self.outlier_detection_transform_dict['features_list']} :\n{X_copy.loc[multiple_outliers].to_string()}")
            logger.info(f"Shape of input dataframe: {X_copy.shape}")
            logger.info(f"No. of outliers detected: {len(multiple_outliers)}")

        if self.outlier_detection_transform_dict['drop_outliers']:
            X_copy = X_copy.drop(multiple_outliers, axis = 0).reset_index(drop=True)
            if self.loginfo:
                logger.info(f"\nShape of input dataframe post outliers rows dropped because of outliers in features - {self.outlier_detection_transform_dict['features_list']}: {X_copy.shape}")

        # return multiple_outliers
        return (X_copy, y_copy)
'''

#=============================================================================

class Custom_Missing_Values_Check_Column_Drop(BaseEstimator, TransformerMixin):
    def __init__(self, missing_val_percentage=0.7, loginfo=False):
        self.missing_val_percentage = missing_val_percentage
        self.loginfo = loginfo

    def fit(self, X, y=None, **kwargs):
        X_copy = X.copy()
        self.filtered_data = msno.nullity_filter(X_copy,
                                            filter='bottom',
                                            n=X_copy.shape[1],
                                            p=1-self.missing_val_percentage)
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy.drop(columns=self.filtered_data.columns.tolist(), axis=1, inplace=True)
        if self.loginfo:
            logger.info(f"""
                        Columns dropped because of large no. of missing values ({self.missing_val_percentage*100}%): {self.filtered_data.columns.tolist()}""")
            logger.info(f"""
                        Shape of dataset post above columns dropped : {X_copy.shape}""")
        return X_copy

#=============================================================================

class Custom_Features_Transformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_transform_dict, loginfo=False):
        self.feature_transform_dict = feature_transform_dict
        self.loginfo = loginfo

    def fit(self, X, y=None, **kwargs):
        # X_copy = X.copy()
        self.feat_trans_dict = {}
        for key, value in self.feature_transform_dict.items():
# even though for loop is defined above, use this transformation separately for each feature in pipeline
# example dict for "value_mapping" transformation::
# feature_transform_dict = {'Years in current job': {
#                                                 'strategy': 'value_mapping',
#                                                 'map_values': {'10+ years': 10, 
#                                                                '9 years': 9, 
#                                                                '8 years': 8, 
#                                                                '7 years': 7, 
#                                                                '6 years': 6, 
#                                                                '5 years': 5, 
#                                                                '4 years': 4,
#                                                                '3 years': 3,
#                                                                '2 years': 2,
#                                                                '1 year': 1,
#                                                                '< 1 year': 0.5}}}          
            if value['strategy'] == 'value_mapping':
                self.feat_trans_dict[key] = value['map_values']
                
        return self

    def transform(self, X):
        X_copy = X.copy()
        for key, value in self.feature_transform_dict.items():
# example dict for "regular_expression" transformation::
# feature_transform_dict = OrderedDict({'Name': {'strategy': 'regular_expression',
#                                                     'reg_exp': ' ([A-Za-z]+)\.',
#                                                     'column_name': 'Title',
#                                                     'drop_orig_column': True,
#                                                     'top_categories_count': 4},
#                                           'Cabin': {'strategy': 'regular_expression',
#                                                     'reg_exp': '(^[a-zA-Z])',
#                                                     'column_name': 'CabImputed',
#                                                     'drop_orig_column': True,
#                                                     'top_categories_count': 100}})            
            
            if value['strategy'] == 'regular_expression':
                X_copy[value['column_name']] = X_copy[key].str.extract(value['reg_exp'], expand=False)

                all_categories_list = X_copy[value['column_name']].value_counts().to_frame().index.tolist()
                top_categories_list = X_copy[value['column_name']].value_counts().to_frame().head(value['top_categories_count']).index.tolist()
                mapping_dict = {k: (v if k in top_categories_list else len(top_categories_list)) for v, k in enumerate(all_categories_list)}

                X_copy[value['column_name']] = X_copy[value['column_name']].map(mapping_dict)

                if self.loginfo:
                    logger.info(f"""Applied Regular Expression Text Extract on feature -
                                '{key}' with regular_expression - {value['reg_exp']}""")

                    logger.info(f"""Categories in feature - '{key}'
                                post regular_expression applied - {top_categories_list}""")

                    logger.info(f"""Categories mapping_dict for feature - '{key}' in new column -
                                '{value['column_name']}' post regular_expression
                                applied - {mapping_dict}""")

                if value['drop_orig_column'] == True:
                    X_copy.drop(key, axis=1, inplace=True)
                    if self.loginfo:
                        logger.info(f"""Original feature - '{key}' dropped post
                                    regular expression text extraction""")
                else:
                    if self.loginfo:
                        logger.info(f"""Original feature - '{key}' is NOT dropped
                                    post regular expression text extraction""")

            if value['strategy'] == 'value_mapping':
                # X_copy[key] = X_copy[key].map(self.feat_trans_dict[key])
                X_copy[key] = X_copy[key].replace(self.feat_trans_dict[key])

                if self.loginfo:
                    logger.info(f"""Custom value mapping transformation applied to column - '{key}'
                                mappings values are - {self.feat_trans_dict[key]}""")

        return X_copy

#=============================================================================

class Custom_New_Features_Creation(BaseEstimator, TransformerMixin):
    def __init__(self, new_features_creation_dict):
        self.new_features_creation_dict = new_features_creation_dict

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X):
        X_copy = X.copy()
        if self.new_features_creation_dict['strategy'] == 'summation':
            new_col = self.new_features_creation_dict['column_name']
            X_copy[new_col] = 0
            for feat in self.new_features_creation_dict['feature_list']:
                X_copy[new_col] = X_copy[new_col] + X_copy[feat]

        return X_copy

#=============================================================================

class Custom_Features_Binning_Transformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_binning_dict, loginfo=False):
        self.feature_binning_dict = feature_binning_dict
        self.loginfo = loginfo

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X):
# example input for this class
# feature_binning_dict = OrderedDict({'Monthly Debt': {'column_name': 'Monthly_Debt_Binned',
#                                                      'bins': 10,
#                                                      'labels': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#                                                      'drop_orig_column': False},
#                                     'Current Loan Amount': {'column_name': 'Current_Loan_Amount_Binned',
#                                                      'bins': 10,
#                                                      'labels': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#                                                      'drop_orig_column': False},
#                                     })        
        X_copy = X.copy()
        for col, col_dict in self.feature_binning_dict.items():
            X_copy[col_dict['column_name']] = pd.cut(X_copy[col],bins=col_dict['bins'],
                                                     labels=col_dict['labels'])
            if self.loginfo:
                logger.info(f"""Applied binning on feature -
                            '{col}' with bins - {col_dict['bins']}, label - {col_dict['labels']}""")

            if col_dict['drop_orig_column'] == True:
                X_copy.drop(col, axis=1, inplace=True)
                if self.loginfo:
                    logger.debug(f"""Original feature - '{col}' dropped post binning""")
            else:
                if self.loginfo:
                    logger.debug(f"""Original feature - '{col}' is NOT dropped post binning""")

        return X_copy


#=============================================================================

class Custom_SimpleImputer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_imputer_dict, loginfo=False):
        self.feature_imputer_dict = feature_imputer_dict
        self.loginfo = loginfo

    def fit(self, X, y=None, **kwargs):
        self.fit_obj_dict = {}
        X_copy = X.copy()
        for key, value in self.feature_imputer_dict.items():
            if value['strategy'] == 'mean':
                self.imputer_mean_obj = SimpleImputer(missing_values=value['missing_values'],
                                            strategy=value['strategy'])
                self.imputer_mean_obj.fit(X_copy[key].values.reshape(-1, 1))

                self.fit_obj_dict[key] = self.imputer_mean_obj

            elif value['strategy'] == 'most_frequent':
                self.imputer_most_frequent_obj = SimpleImputer(missing_values=value['missing_values'],
                                            strategy=value['strategy'])
                self.imputer_most_frequent_obj.fit(X_copy[key].values.reshape(-1, 1))

                self.fit_obj_dict[key] = self.imputer_most_frequent_obj

            elif value['strategy'] == 'median':
                self.imputer_median_obj = SimpleImputer(missing_values=value['missing_values'],
                                            strategy=value['strategy'])
                self.imputer_median_obj.fit(X_copy[key].values.reshape(-1, 1))

                self.fit_obj_dict[key] = self.imputer_median_obj

            elif value['strategy'] == 'constant':
                self.imputer_constant_obj = SimpleImputer(missing_values=value['missing_values'],
                                            strategy=value['strategy'],
                                            fill_value=value['constant_value'])
                self.imputer_constant_obj.fit(X_copy[key].values.reshape(-1, 1))

                self.fit_obj_dict[key] = self.imputer_constant_obj

        return self

    def transform(self, X):
        X_copy = X.copy()
        impute_feat_list = []
        for key, value in self.feature_imputer_dict.items():
            impute_feat_list.append(key)
            if value['strategy'] == 'mean':
                if self.loginfo:
                    logger.info(f"""
                                :: Mean Imputer in feature - {key} ::
                                {key} - {value}
                                Shape of X in Custom_SimpleImputer in Tranform Method - {X_copy.shape}
                                self.fit_obj_dict[key].statistics_ - {self.fit_obj_dict[key].statistics_}""")
                X_copy[key] = self.fit_obj_dict[key].transform(X_copy[key].values.reshape(-1,1))

            elif value['strategy'] == 'most_frequent':
                if self.loginfo:
                    logger.info(f"""
                                :: Most_Frequent(Mode) Imputer in feature - {key} ::
                                {key} - {value}
                                Shape of X in Custom_SimpleImputer in Tranform Method - {X_copy.shape}
                                self.fit_obj_dict[key].statistics_ - {self.fit_obj_dict[key].statistics_}""")
                X_copy[key] = self.fit_obj_dict[key].transform(X_copy[key].values.reshape(-1,1))

            elif value['strategy'] == 'median':
                if self.loginfo:
                    logger.info(f"""
                                :: Median Imputer in feature - {key} ::
                                {key} - {value}
                                Shape of X in Custom_SimpleImputer in Tranform Method - {X_copy.shape}
                                self.fit_obj_dict[key].statistics_ - {self.fit_obj_dict[key].statistics_}""")
                X_copy[key] = self.fit_obj_dict[key].transform(X_copy[key].values.reshape(-1,1))

            elif value['strategy'] == 'constant':
                if self.loginfo:
                    logger.info(f"""
                                :: Constant Value Imputer in feature - {key} ::
                                {key} - {value}
                                Shape of X in Custom_SimpleImputer in Tranform Method - {X_copy.shape}
                                self.fit_obj_dict[key].statistics_ - {self.fit_obj_dict[key].statistics_}""")
                X_copy[key] = self.fit_obj_dict[key].transform(X_copy[key].values.reshape(-1,1))

            elif value['strategy'] == 'based_on_other_column':
                # input example for this condition
                # OrderedDict({'Bankruptcies': {'strategy': 'based_on_other_column',
                #                               'other_column': 'Number of Credit Problems'},
                #                               'group_by_transformation': False,})                
                # get the null value indexes as list before imputing
                if not value['group_by_transformation']:
                    null_value_index_list = X_copy[X_copy[key].isnull()].index.tolist()
                   
                    #fillna based on other column values
                    X_copy[key] = X_copy[key].fillna(X_copy[value['other_column']])
                    
                    if self.loginfo:
                        logger.info(f"""column - {key} imputed the null values based on other column({value['other_column']}) values
                                    Below are the missing value indexes and values filled 
                                    {X_copy[key].iloc[null_value_index_list]}""") 

                else:
                    null_value_index_list = X_copy[X_copy[key].isnull()].index.tolist()
                    
                    # fillna based on other column values
                    X_copy[key] = X_copy.groupby(value['other_column']).transform(lambda x: x.fillna(x.median()))[key]
                    
                    if self.loginfo:
                        logger.info(f"""column - {key} imputed the null values based on other column({value['other_column']}) values
                                    Below are the missing value indexes and values filled 
                                    {X_copy[key].iloc[null_value_index_list]}""")                     

        return X_copy


#=============================================================================

class Custom_LabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, feature_lbl_encode_list, loginfo=False):
        self.feature_lbl_encode_list = feature_lbl_encode_list
        self.loginfo = loginfo

    def fit(self, X, y=None, **kwargs):
        X_copy = X.copy()
        self.fit_obj_dict = {}
        for feat in self.feature_lbl_encode_list:
            self.lbl_enc = LabelEncoder()
            # self.lbl_enc.fit(X_copy[feat].values)
            if X_copy[feat].dtype in [np.int16, np.int32, np.int64]:
                self.lbl_enc.fit(X_copy[feat].values.tolist()+[-999])
            else:
                self.lbl_enc.fit(X_copy[feat].values.tolist()+['Unknown'])
                # pass
            # self.lbl_enc.fit(X_copy[feat].values.tolist()+[-999])
            # self.lbl_enc.fit(X_copy[feat].values)
            self.fit_obj_dict[feat] = self.lbl_enc


            # self.lbl_enc_w_unknown = LabelEncoder()
            # self.lbl_enc_w_unknown.fit(X_copy[feat].tolist()+['Unknown'])
            # self.fit_obj_dict[feat+'_w_unknown'] = self.lbl_enc_w_unknown

        return self

    def transform(self, X):
        X_copy = X.copy()
        # lbl_enc_feat_list = []
        for feat in self.feature_lbl_encode_list:
            # print(f'\nLabel Encode - {feat}\nShape of X in Custom_LabelEncoder in Tranform Method - {X_copy.shape}\nself.fit_obj_dict[feat].statistics_ - {self.fit_obj_dict[feat].statistics_}\n')
            # lbl_enc_feat_list.append(feat)
            # X_copy[feat] = self.lbl_enc.transform(X_copy[feat].values)

            # for unique_item in np.unique(X_copy[feat].values.tolist()):
            for unique_item in list(set(X_copy[feat].values.tolist())):
                if self.fit_obj_dict[feat].classes_.dtype.__str__() in ['int16', 'int32', 'int64']:
                    if unique_item not in self.fit_obj_dict[feat].classes_.tolist():
                        # X_copy[feat] = ['Unknown' if x == unique_item else x for x in X_copy[feat].values.tolist()]
                        X_copy[feat] = [-999 if x == unique_item else x for x in X_copy[feat].values.tolist()]
                else:
                    if unique_item not in self.fit_obj_dict[feat].classes_.tolist():
                        X_copy[feat] = ['Unknown' if x == unique_item else x for x in X_copy[feat].values.tolist()]


                #     X_copy[feat] = self.fit_obj_dict[feat+'_w_unknown'].transform(X_copy[feat].values)
                # else:
                #     X_copy[feat] = self.fit_obj_dict[feat].transform(X_copy[feat].values)

            X_copy[feat] = self.fit_obj_dict[feat].transform(X_copy[feat].values)


            #X_copy[feat] = self.fit_obj_dict[feat].transform(new_data_list)

            if self.loginfo:
                #logging the label and its corresponing mapped value as part of label encoding
                values = self.fit_obj_dict[feat].inverse_transform(sorted(X_copy[feat].unique()))
                keys = sorted(X_copy[feat].unique())
                dictionary = dict(zip(keys, values))
                logger.info(f"""
                            Label Encoded Feature: {feat}
                            Label Encoded to categories in feature - '{feat}':
                            {dictionary}""")
        return X_copy
#=============================================================================

class Custom_OneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, feature_1hot_encode_list, drop_first=True, handle_unknown='error', loginfo=False):
        self.feature_1hot_encode_list = feature_1hot_encode_list
        self.drop_first = drop_first
        self.handle_unknown = handle_unknown
        self.loginfo = loginfo

    def fit(self, X, y=None, **kwargs):
        self.fit_obj_dict = {}
        X_copy = X.copy()

        if self.drop_first:
            self.drop_first = 'first'
        else:
            self.drop_first = None

        for feat in self.feature_1hot_encode_list:
            if feat in X_copy.columns.values.tolist():
                # self.onehot_enc = OneHotEncoder(drop=self.drop_first,
                #                                 sparse=False,
                #                                 handle_unknown=self.handle_unknown)
                # self.onehot_enc.fit(X_copy[feat].values.reshape(-1,1))
                # self.fit_obj_dict[feat] = self.onehot_enc

                self.onehot_enc = OneHotEncoder(drop='first',
                                                sparse=False,
                                                handle_unknown='error')
                # self.onehot_enc.fit(X_copy[feat].values.reshape(-1,1))
                # self.fit_obj_dict[feat] = self.onehot_enc


                # self.onehot_enc_unknown_ignore = OneHotEncoder(drop=None,
                #                                 sparse=False,
                #                                 handle_unknown='ignore')
                # self.onehot_enc_unknown_ignore.fit(X_copy[feat].values.reshape(-1,1))
                # self.fit_obj_dict[feat+'_unknown_ignore'] = self.onehot_enc_unknown_ignore

                if X_copy[feat].dtype in [np.int16, np.int32, np.int64]:
                    # self.onehot_enc.fit(X_copy[feat].values.tolist()+[-999])
                    self.onehot_enc.fit(np.append(X_copy[feat].values.tolist(), -999).reshape(-1,1))
                else:
                    # self.onehot_enc.fit(X_copy[feat].values.tolist()+['Unknown'])
                    self.onehot_enc.fit(np.append(X_copy[feat].values.tolist(), 'Unknown').reshape(-1,1))

                self.fit_obj_dict[feat] = self.onehot_enc



        return self

    def transform(self, X):
        X_copy = X.copy()
        df = pd.DataFrame()
        df = df.append(X)
        for feat in self.feature_1hot_encode_list:
            if feat in X_copy.columns.values.tolist():
                # if not all([True if x in self.fit_obj_dict[feat].categories_[0].tolist() else False for x in np.unique(X_copy[feat].values.tolist())]):
                #     for unique_item in np.unique(X_copy[feat].values.tolist()):
                #         if unique_item not in self.fit_obj_dict[feat].categories_[0].tolist():
                #             X_copy[feat] = [-999 if x == unique_item else x for x in X_copy[feat].values.tolist()]
                #     transformed  = self.fit_obj_dict[feat+'_unknown_ignore'].transform(X_copy[feat].values.reshape(-1,1))
                # else:
                #     transformed  = self.fit_obj_dict[feat].transform(X_copy[feat].values.reshape(-1,1))

                '''
                if all([True if x in self.fit_obj_dict[feat].categories_[0].tolist() else False for x in np.unique(X_copy[feat].values.tolist())]):
                    transformed  = self.fit_obj_dict[feat].transform(X_copy[feat].values.reshape(-1,1))
                    ohe_df = pd.DataFrame(data=transformed, columns=self.fit_obj_dict[feat].get_feature_names([feat]))
                else:
                    # for unique_item in np.unique(X_copy[feat].values.tolist()):
                    for unique_item in list(set(X_copy[feat].values.tolist())):
                        if unique_item not in self.fit_obj_dict[feat].categories_[0].tolist():
                            X_copy[feat] = ['Unknown' if x == unique_item else x for x in X_copy[feat].values.tolist()]
                    transformed  = self.fit_obj_dict[feat+'_unknown_ignore'].transform(X_copy[feat].values.reshape(-1,1))
                    ohe_df = pd.DataFrame(data=transformed, columns=self.fit_obj_dict[feat+'_unknown_ignore'].get_feature_names([feat]))
                '''

                # transformed  = self.fit_obj_dict[feat].transform(X_copy[feat].values.reshape(-1,1))
                # ohe_df = pd.DataFrame(data=transformed, columns=self.fit_obj_dict[feat].get_feature_names([feat]))
                # df.reset_index(drop=True, inplace=True)
                # ohe_df.reset_index(drop=True, inplace=True)
                # df = pd.concat([df, ohe_df], axis=1).drop([feat], axis=1)

                for unique_item in list(set(X_copy[feat].values.tolist())):
                    if self.fit_obj_dict[feat].categories_[0].dtype.__str__() in ['int16', 'int32', 'int64']:
                        if unique_item not in self.fit_obj_dict[feat].categories_[0].tolist():
                            # X_copy[feat] = ['Unknown' if x == unique_item else x for x in X_copy[feat].values.tolist()]
                            X_copy[feat] = [-999 if x == unique_item else x for x in X_copy[feat].values.tolist()]
                    else:
                        if unique_item not in self.fit_obj_dict[feat].categories_[0].tolist():
                            X_copy[feat] = ['Unknown' if x == unique_item else x for x in X_copy[feat].values.tolist()]

                transformed  = self.fit_obj_dict[feat].transform(X_copy[feat].values.reshape(-1,1))
                ohe_df = pd.DataFrame(data=transformed, columns=self.fit_obj_dict[feat].get_feature_names([feat]))
                df.reset_index(drop=True, inplace=True)
                ohe_df.reset_index(drop=True, inplace=True)
                df = pd.concat([df, ohe_df], axis=1).drop([feat], axis=1)

                if self.loginfo:
                    logger.info(f"""OneHot Encoded column names for the feature - '{feat}': {self.fit_obj_dict[feat].get_feature_names([feat])}""")

        return df
        # return ohe_df
#=============================================================================

class Custom_TfIdfVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_tfidf_vectorizer_list):
        self.feature_tfidf_vectorizer_list = feature_tfidf_vectorizer_list

    def fit(self, X, y=None, **kwargs):
        X_copy = X.copy()
        X_copy['Text'] = X_copy[self.feature_tfidf_vectorizer_list].apply(lambda x: ' '.join(x.map(str)), axis=1)

        self.tfv = TfidfVectorizer(input='content'
                                ,encoding='utf-8'
                                ,strip_accents='unicode'
                                ,analyzer='word'
                                ,stop_words='english'
                                #,token_pattern='r\w{1,}'
                                ,ngram_range=(1, 2)
                                ,min_df=3
                                ,max_features=None
                                ,vocabulary=None
                                ,binary=False
                                ,norm='l2'
                                ,use_idf=True
                                ,smooth_idf=True
                                ,sublinear_tf=True)

        self.tfv.fit(X_copy['Text'].tolist())

        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy['Text'] = X_copy[self.feature_tfidf_vectorizer_list].apply(lambda x: ' '.join(x.map(str)), axis=1)
        df = pd.DataFrame()
        df = df.append(X)

        tfidf_transformed  = self.tfv.transform(X_copy['Text'].tolist())
        self.tfidf_df = pd.DataFrame(data=list(tfidf_transformed.toarray()), columns=self.tfv.get_feature_names())
        df = pd.concat([df, self.tfidf_df], axis=1).drop(self.feature_tfidf_vectorizer_list, axis=1)

        return df
        # return self.tfidf_df

    def get_feature_names(self):
        return self.tfidf_df.columns.tolist()

#=============================================================================

class custom_tfidf(BaseEstimator,TransformerMixin):
    def __init__(self, tfidf):
        self.tfidf = tfidf

    def fit(self, X, y=None):
        joined_X = X.apply(lambda x: ' '.join(x), axis=1)
        self.tfidf.fit(joined_X)
        return self

    def transform(self, X):
        joined_X = X.apply(lambda x: ' '.join(x), axis=1)

        return self.tfidf.transform(joined_X)

#=============================================================================
class Custom_Feature_Selection(BaseEstimator, TransformerMixin):
    def __init__(self, feature_selection_dict, loginfo=False):
        self.feature_selection_dict = feature_selection_dict
        self.loginfo = loginfo

    def fit(self, X, y=None, **kwargs):
        X_copy = X.copy()
        feat_remove_list = []

        # =============================================================
        # Constant Features
        # -----------------
        # Removing constant features using VarianceThreshold from Scikit-learn
        self.vt_cf = VarianceThreshold(threshold=0)
        self.vt_cf.fit(X_copy)
        self.vt_cf_feat_names = X_copy.columns[self.vt_cf.get_support()] #This gives which features are retained
        constant_feat = X_copy.columns[~self.vt_cf.get_support()].to_list()

        feat_remove_list += constant_feat

        if self.loginfo:
            if len(constant_feat) > 0:
                logger.info(f"""No. of constant features dropped : {len(constant_feat)}\n{constant_feat}""")
            else:
                logger.info("""No Constant Features are found""")

        # =============================================================
        # Quasi-Constant Features
        # -----------------------
        # Remove quasi-constant features using VarianceThreshold from Scikit-learn
        self.vt_qcf = VarianceThreshold(threshold=0.01)
        X_copy = X_copy[X_copy.columns.difference(feat_remove_list)]
        self.vt_qcf.fit(X_copy)
        self.vt_qcf_feat_names = X_copy.columns[self.vt_qcf.get_support()]
        quasi_constant_feat = X_copy.columns[~self.vt_qcf.get_support()].to_list()

        feat_remove_list += quasi_constant_feat

        if self.loginfo:
            if len(quasi_constant_feat) > 0:
                logger.info(f"""No. of quasi-constant features dropped : {len(quasi_constant_feat)}\n{quasi_constant_feat}""")
            else:
                logger.info("""No Quasi-Constant Features are found""")

        # =============================================================
        # Duplicate Features
        # ------------------
        # Remove duplicate features using iteration
        self.duplicated_feat = []
        # iterate over every feature in our dataset:
        X_copy = X_copy[X_copy.columns.difference(feat_remove_list)]
        for i in range(0, len(X_copy.columns)):
            feat_1 = X_copy.columns[i]
            if feat_1 not in self.duplicated_feat:
                # now, iterate over the remaining features of the dataset:
                for feat_2 in X_copy.columns[i + 1:]:
                    # check if this second feature is identical to the first one
                    if X_copy[feat_1].equals(X_copy[feat_2]):
                        self.duplicated_feat.append(feat_2)

        feat_remove_list += self.duplicated_feat

        if self.loginfo:
            if len(self.duplicated_feat) > 0:
                logger.info(f"""No. of duplicate features dropped : {len(self.duplicated_feat)}\n{self.duplicated_feat}""")
            else:
                logger.info("""No Duplicate Features are found""")

        # =============================================================
        # Correlated Features
        # -------------------
        # Identify the correlated feature in the training data set
        X_copy = X_copy[X_copy.columns.difference(feat_remove_list)]

        corrmat_threshold = self.feature_selection_dict['featCorrelationThreshold']
        corrmat = X_copy.corr()
        corrmat = corrmat.abs().unstack() # absolute value of corr coef
        corrmat = corrmat.sort_values(ascending=False)
        corrmat = corrmat[corrmat >= corrmat_threshold]
        corrmat = corrmat[corrmat < 1]
        corrmat = pd.DataFrame(corrmat).reset_index()
        corrmat.columns = ['feature1', 'feature2', 'corr']
        corrmat.head()

        grouped_feature_ls = []
        correlated_groups = []
        for feature in corrmat.feature1.unique():
            if feature not in grouped_feature_ls:
                # find all features correlated to a single feature
                correlated_block = corrmat[corrmat.feature1 == feature]
                grouped_feature_ls = grouped_feature_ls + list(correlated_block.feature2.unique()) + [feature]
                # append the block of features to the list
                correlated_groups.append(correlated_block)

        if self.loginfo:
            logger.info(f'Found {len(correlated_groups)} correlated groups out of {X_copy.shape[1]} total features')
            for group in correlated_groups:
                logger.info(f"\n{group}")
            # print()

        self.corr_features_drop = []
        for group in correlated_groups:
            # add all features of the group to a list
            features = group['feature2'].unique().tolist()+group['feature1'].unique().tolist()

            if self.feature_selection_dict['model_type'] == 'Regression':
                # train a random forest
                rf = RandomForestRegressor(n_estimators=400, random_state=42, max_depth=5)
                rf.fit(X_copy[features], y)

                importance = pd.concat([pd.Series(features), pd.Series(rf.feature_importances_)], axis=1)
                importance.columns = ['feature', 'importance']
                importance = importance.sort_values(by='importance', ascending=False)
                importance.reset_index(drop=True, inplace=True)
                self.corr_features_drop += importance['feature'].values.tolist()[1:]

                if self.loginfo:
                    logger.info(f"""Correlated Feature group Importance\n{importance}""")

            if self.feature_selection_dict['model_type'] == 'Classification':
                # print("Write the code for classification in here")
                rf = RandomForestClassifier(n_estimators=400, random_state=42, max_depth=5)
                rf.fit(X_copy[features], y)

                importance = pd.concat([pd.Series(features), pd.Series(rf.feature_importances_)], axis=1)
                importance.columns = ['feature', 'importance']
                importance = importance.sort_values(by='importance', ascending=False)
                importance.reset_index(drop=True, inplace=True)
                self.corr_features_drop += importance['feature'].values.tolist()[1:]

                if self.loginfo:
                    logger.info(f"""Correlated Feature group Importance\n{importance}""")                

        feat_remove_list += self.corr_features_drop

        if self.loginfo:
            logger.info(f"""Correlated Features to be dropped : {len(self.corr_features_drop)}\n{self.corr_features_drop} """)

        # =============================================================
        if self.feature_selection_dict['model_type'] == 'Regression':
            X_copy = X_copy[X_copy.columns.difference(feat_remove_list)]
            self.sfm = SelectFromModel(estimator=RandomForestRegressor(n_estimators=500,
                                                                       random_state=42,
                                                                       max_depth=5),
                                                                       threshold=self.feature_selection_dict['threshold'])
            self.sfm.fit(X_copy, y)
            feat_importances = pd.Series(self.sfm.estimator_.feature_importances_,
                                          index=X_copy.columns).sort_values(ascending=False)

            plt.figure(figsize=(6,7))
            feat_importances.nlargest(self.feature_selection_dict['n_largest_features']).plot(kind='barh')
            plt.show()

            self.selectfrommodel_important_feats = X_copy.columns.values[self.sfm.get_support()].tolist()

            if self.loginfo:
                logger.info(f"""Feature Importance Threshold:\n {self.feature_selection_dict['threshold']}""")
                logger.info(f"""Feature Importance:\n{feat_importances}""")
                logger.info(f"""Features selected based on importance: {len(self.selectfrommodel_important_feats)}\n{self.selectfrommodel_important_feats}""")

        if self.feature_selection_dict['model_type'] == 'Classification':
            self.sfm = SelectFromModel(estimator=RandomForestClassifier(n_estimators=500,
                                                                       random_state=42,
                                                                       max_depth=5),
                                                                       threshold=self.feature_selection_dict['threshold'])
            self.sfm.fit(X_copy, y)
            feat_importances = pd.Series(self.sfm.estimator_.feature_importances_,
                                          index=X_copy.columns).sort_values(ascending=False)

            plt.figure(figsize=(6,7))
            feat_importances.nlargest(self.feature_selection_dict['n_largest_features']).plot(kind='barh')
            plt.show()

            self.selectfrommodel_important_feats = X_copy.columns.values[self.sfm.get_support()].tolist()

            if self.loginfo:
                logger.info(f"""Feature Importance Threshold:\n {self.feature_selection_dict['threshold']}""")
                logger.info(f"""Feature Importance:\n{feat_importances}""")
                logger.info(f"""Features selected based on importance: {len(self.selectfrommodel_important_feats)}\n{self.selectfrommodel_important_feats}""")

        return self

    def transform(self, X):
        X_copy = X.copy()

        # Removing constant features using VarianceThreshold from Scikit-learn
        X_copy = X_copy[self.vt_cf_feat_names]

        # Removing quasi-constant features using VarianceThreshold from Scikit-learn
        X_copy = X_copy[self.vt_qcf_feat_names]

        # Drop duplicate features
        X_copy.drop(columns=self.duplicated_feat, axis=1, inplace=True)

        if self.feature_selection_dict['model_type'] == 'Regression':
            # Dropping correlated features
            X_copy.drop(columns=self.corr_features_drop, axis=1, inplace=True)
            X_copy = X_copy[self.selectfrommodel_important_feats]

        return X_copy
#=============================================================================

def get_stacking_ensemble_classifiers():
    # define the base models
    level0 = list()
    level0.append(('gbc', GradientBoostingClassifier()))
    level0.append(('dtc', DecisionTreeClassifier()))
    level0.append(('svc', SVC()))
    level0.append(('abc', AdaBoostClassifier()))
    level0.append(('etc', ExtraTreesClassifier()))
    level0.append(('bc', BaggingClassifier()))
    level0.append(('lgbmc', LGBMClassifier()))
    level0.append(('xgbc', XGBClassifier()))
    level0.append(('cbc', CatBoostClassifier()))

    # define meta learner model
    level1 = RandomForestClassifier()
    # define the stacking ensemble
    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
    return model

#=============================================================================

def get_transformer_feature_names(columnTransformer):
    output_features = []

    for name, pipe, features in columnTransformer.transformers_:
        if name!='remainder' and name!='passthrough':
            for i in pipe:
                trans_features = []
                if hasattr(i,'categories_'):
                    trans_features.extend(i.get_feature_names(features))
                else:
                    trans_features = features
            output_features.extend(trans_features)
        # else:
        #     trans_features = features
        #     output_features.extend(trans_features)

    return output_features

#=============================================================================

def evaluate_model(model,
               X,
               y,
               scoring,
               n_splits=10,
               n_repeats=3,
               random_state=42):
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    scores = cross_val_score(model, X, y, scoring=scoring, cv=cv, n_jobs=-1, error_score='raise')
    return scores
#=============================================================================

def binary_classification_tree_models_perf(feat_trans_pipeline,
                                           X,
                                           y, ):
    def model_fit_pred_eval(X,
                            y,
                            feat_trans_pipeline,
                            estimator,
                            scoring,
                            repeatedkf_n_splits=5,
                            repeatedkf_n_repeats=2,
                            repeatedkf_random_state=42):

        clf_pipeline = Pipeline([('feat_trans', feat_trans_pipeline),
                         (estimator)])

        cv = RepeatedStratifiedKFold(n_splits=repeatedkf_n_splits,
                                     n_repeats=repeatedkf_n_repeats,
                                     random_state=repeatedkf_random_state)

        scores = cross_validate(clf_pipeline,
                                 X,
                                 y,
                                 scoring=scoring,
                                 cv=cv,
                                 n_jobs=-1,
                                 error_score='raise',
                                 return_train_score=False,
                                 return_estimator=True,
                                 verbose=1)

        return scores

    scoring = {'Accuracy': make_scorer(accuracy_score),
               'ROC_AUC_Score': make_scorer(roc_auc_score),
               'F1_Score': make_scorer(f1_score, average='weighted')}

    models_list = [('RandomForestClassifier', RandomForestClassifier()),
                  ('LGBMClassifier', LGBMClassifier()),
                  ('XGBClassifier', XGBClassifier()),
                  ('CatBoostClassifier', CatBoostClassifier(verbose=False)),
                  ('BaggingClassifier', BaggingClassifier()),
                  ('ExtraTreesClassifier', ExtraTreesClassifier()),
                  ('AdaBoostClassifier', AdaBoostClassifier()),
                  ('SVC', SVC()),
                  ('DecisionTreeClassifier', DecisionTreeClassifier()),
                  ('GradientBoostingClassifier', GradientBoostingClassifier())]

    model_scores = Parallel(n_jobs=-1, verbose=5)(delayed(model_fit_pred_eval)\
                                       (X=X,
                                        y=y,
                                        feat_trans_pipeline=feat_trans_pipeline,
                                        estimator=model_item,
                                        scoring=scoring) \
                            for model_item in models_list)

    model_perf_dict = {}
    for i in range(len(model_scores)):
        estimator_name = model_scores[i]['estimator'][0].steps[-1][0]
        for key in scoring:
            if key == 'Accuracy':
                Accuracy = round(model_scores[i]['test_'+key].mean(), 5)
            if key == 'F1_Score':
                F1_Score = round(model_scores[i]['test_'+key].mean(), 5)
            if key == 'ROC_AUC_Score':
                ROC_AUC_Score = round(model_scores[i]['test_'+key].mean(), 5)


        model_perf_dict[estimator_name] = dict(Accuracy=Accuracy,
                                             ROC_AUC_Score=ROC_AUC_Score,
                                              F1_Score=F1_Score)

    model_perf_df = pd.DataFrame.from_dict(model_perf_dict, orient='index').reset_index().rename(columns={'index': 'Model'})


    return model_perf_df

#=============================================================================
# Gridsearch model performance tuning

def model_perf_tuning(X, y,
                      feature_trans,
                      estimator_list,
                      model_type,
                      score_eval,
                      greater_the_better=True,
                      cv_n_splits=2,
                      randomsearchcv_n_iter=25,
                      n_jobs=-1):

    model_perf_tuning_dict = {}

    with open('config.yaml') as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)

    for estimator in estimator_list:
        print(f"""\n##################\nModel Performance Tuning : {estimator}\n##################""")
        model_pipeline = Pipeline([('feat_trans', feature_trans),
                                   ('model_estimator', globals()[estimator]())])

        if model_type == 'Classification':
            model_params = config_data['classification_models_parameters'][estimator]
            cv = StratifiedKFold(n_splits=cv_n_splits, shuffle=True, random_state=42)
        if model_type == 'Regression':
            model_params = config_data['regression_models_parameters'][estimator]
            cv = KFold(n_splits=cv_n_splits, shuffle=True, random_state=42)

        grid_params = {f'model_estimator__{k}': v for k, v in model_params.items()}

        # scoring = {'Accuracy': make_scorer(accuracy_score),
        #            'ROC_AUC_Score': make_scorer(roc_auc_score),
        #            'F1_Score': make_scorer(f1_score, average='weighted')}

        # scoring = {'ROC_AUC_Score': make_scorer(score_eval)}
        # scoring = config_data['classification_eval_metrics']

        if score_eval == 'rmse':
            scorer = make_scorer(mean_squared_error, squared=False, greater_is_better=False)
        if score_eval == 'rmsle':
            def rmsle(real, predicted):
                real = real.to_numpy()
                sum=0.0
                for x in range(len(predicted)):
                    if predicted[x]<0 or real[x]<0: #check for negative values
                        continue
                    p = np.log(predicted[x]+1)
                    r = np.log(real[x]+1)

                    # p = np.log(abs(predicted[x])+1)
                    # r = np.log(abs(real[x])+1)
                    sum = sum + (p - r)**2
                return (sum/len(predicted))**0.5
            scorer = make_scorer(rmsle, greater_is_better=False)
        if score_eval == 'mse':
            scorer = make_scorer(mean_squared_error, squared=True, greater_is_better=False)
        if score_eval == 'roc_auc_score':
            scorer = make_scorer(roc_auc_score,
                                 average='macro',
                                 sample_weight=None,
                                 max_fpr=None,
                                 multi_class='raise',
                                 labels=None,
                                 greater_is_better=True)
        if score_eval == 'f1_score':
            scorer = make_scorer(f1_score,
                                 labels=None, 
                                 pos_label=1, # The class to report if average='binary' and the data is binary.
                                 average='binary', 
                                 sample_weight=None, 
                                 zero_division='warn',
                                 greater_is_better=True)
            
            
            
        clf = RandomizedSearchCV(estimator=model_pipeline,
                                 param_distributions=grid_params,
                                 n_iter=randomsearchcv_n_iter,
                                 scoring=scorer,
                                 n_jobs=n_jobs,
                                 #n_jobs=-1,
                                 #n_jobs=1,
                                 refit=True,
                                 cv=cv,
                                 verbose=1,
                                 random_state=42,
                                 error_score='raise',
                                 return_train_score=True)
        clf.fit(X, y)

        model_pipeline.set_params(**clf.best_params_)

        oof_preds = cross_val_predict(estimator=model_pipeline,
                                      X=X,
                                      y=y,
                                      cv=cv,
                                      n_jobs=n_jobs,
                                      #n_jobs=-1,
                                      method='predict',
                                      # fit_params=clf.best_params_,
                                      verbose=1)

        if model_type =='Regression':
            if score_eval == 'rmse':
                oof_eval_score = mean_squared_error(y_true=y.to_numpy(), y_pred=oof_preds, squared=False)
            if score_eval == 'rmsle':
                oof_eval_score = np.sqrt(mean_squared_log_error(y_true=y.to_numpy(), y_pred=oof_preds))
            if score_eval == 'mse':
                oof_eval_score = mean_squared_error(y_true=y.to_numpy(), y_pred=oof_preds, squared=True)

        if model_type == 'Classification':
            if score_eval == 'roc_auc_score':
                oof_eval_score = roc_auc_score(y_true=y.to_numpy(), y_score=oof_preds)
            if score_eval == 'f1_score':
                    oof_eval_score = f1_score(y_true=y.to_numpy(), y_pred=oof_preds)                

        print(f"""\n{score_eval} OOF Model Evaluation Score : {estimator} - {oof_eval_score}""")
        #oof_eval_score = score_eval(y, oof_preds)

        esti_eval_dict = {}
        esti_eval_dict['OOF_'+score_eval] = round(oof_eval_score, 5)
        esti_eval_dict['Total_Fits'] = clf.n_splits_*len(clf.cv_results_['mean_train_score'])
        esti_eval_dict['Best_Score'] = round(clf.best_score_,5)
        esti_eval_dict['Best_Params'] = str(clf.best_params_)
        esti_eval_dict['Mean_Train_Score'] = round(np.mean(clf.cv_results_['mean_train_score']),5)
        esti_eval_dict['Std_Train_Score'] = round(np.mean(clf.cv_results_['std_train_score']),5)
        esti_eval_dict['Mean_Test_Score'] = round(np.mean(clf.cv_results_['mean_test_score']),5)
        esti_eval_dict['Std_Test_Score'] = round(np.mean(clf.cv_results_['std_test_score']),5)

        model_perf_tuning_dict[estimator] = esti_eval_dict

    model_perf_tuning_df = pd.DataFrame.from_dict(model_perf_tuning_dict,
                                                  orient='index').reset_index().rename(columns={'index': 'Model'})

    if greater_the_better:
        # model_perf_tuning_df = model_perf_tuning_df.sort_values(by='OOF_'+score_eval.__name__, ascending=False)
        model_perf_tuning_df = model_perf_tuning_df.sort_values(by='OOF_'+score_eval, ascending=False)
    else:
        # model_perf_tuning_df = model_perf_tuning_df.sort_values(by='OOF_'+score_eval.__name__, ascending=True)
        model_perf_tuning_df = model_perf_tuning_df.sort_values(by='OOF_'+score_eval, ascending=True)

    # logger.info(f"\n:: Single Model Evaluation Metric ::\n{model_perf_tuning_df[['Model', 'OOF_'+score_eval.__name__, 'Total_Fits', 'Best_Score', 'Mean_Train_Score', 'Std_Train_Score', 'Mean_Test_Score', 'Std_Test_Score']].to_string()}")
    logger.info(f"\n:: Single Model Evaluation Metric ::\n{model_perf_tuning_df[['Model', 'OOF_'+score_eval, 'Total_Fits', 'Best_Score', 'Mean_Train_Score', 'Std_Train_Score', 'Mean_Test_Score', 'Std_Test_Score']].to_string()}")

    best_model_best_params = eval(model_perf_tuning_df.iloc[0].to_dict()['Best_Params'])
    # model_pipeline = Pipeline([('feat_trans', feature_trans),
    #                             ('estimator', globals()[model_perf_tuning_df.iloc[0].to_dict()['Model']]())])
    model_pipeline = Pipeline([('feat_trans', feature_trans),
                            ('model_estimator', globals()[model_perf_tuning_df.iloc[0].to_dict()['Model']]())])
    model_pipeline.set_params(**best_model_best_params)

    logger.info(f"\nBest Single Model : {model_perf_tuning_df.iloc[0].to_dict()['Model']}")
    logger.info(f"\nBest Single Model - {model_perf_tuning_df.iloc[0].to_dict()['Model']} Params ::\n {pformat(eval(model_perf_tuning_df.iloc[0].to_dict()['Best_Params']))}")
    logger.info(f"\nBest Single Model - - {model_perf_tuning_df.iloc[0].to_dict()['Model']} Pipeline :\n {pformat(model_pipeline.__dict__)}")

    with open('best_single_model.plk', 'wb') as f:
        pickle.dump(model_pipeline, f)

    return model_perf_tuning_df
#=============================================================================
'''
def model_ensemble_classification1(X,
                                   y,
                                   feature_trans,
                                   estimator_list,
                                   score_eval,
                                   model_perf_tuning_df):

    stacked_model_eval_dict = {}

    with open('config.yaml') as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)

    tree_models_short_name_dict = config_data['tree_models_short_name']
    df = model_perf_tuning_df.copy()
    df.set_index('Model', inplace=True)

    # Below is the code for model stacking with different combinations
    model_combinations_tuple = permutations(estimator_list)
    for models_tuple in model_combinations_tuple:
        level0 = list()
        model_params_dict = {}
        for i in models_tuple[:1]:
            level0.append((tree_models_short_name_dict[i][0],  globals()[i]()))

            for key, val in eval(df.loc[i, 'Best_Params']).items():
                key = key.replace('__', '__'+tree_models_short_name_dict[i][0]+'__')
                model_params_dict[key] = val

        # define meta learner model
        level1 = globals()[models_tuple[-1]]()
        final_model_short_name = tree_models_short_name_dict[models_tuple[-1]][0]

        for key, val in eval(df.loc[models_tuple[-1], 'Best_Params']).items():
            # key = key.replace('__', '__'+tree_models_short_name_dict[models_tuple[-1]][0]+'__')
            key = key.replace('__', '__final_estimator__')
            model_params_dict[key] = val

        # define the stacking ensemble
        model_stack = StackingClassifier(estimators=level0, final_estimator=level1)

        model_pipeline = Pipeline([('feat_trans', feature_trans),
                                   ('estimator', model_stack)])

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        model_pipeline.set_params(**model_params_dict)
        oof_preds = cross_val_predict(estimator=model_pipeline,
                                      X=X,
                                      y=y,
                                      cv=cv,
                                      n_jobs=-1,
                                      method='predict',
                                      # fit_params=clf.best_params_,
                                      verbose=1)

        oof_roc_auc_score = roc_auc_score(y, oof_preds)

        stacking_str = ''
        for j in level0:
            stacking_str = stacking_str+j[0]+'+'
        stacking_str = stacking_str[:-1]
        stacking_str = stacking_str+'-->'+final_model_short_name

        stacked_model_eval_dict[stacking_str] = dict(OOF_ROC_AUC_Score=round(oof_roc_auc_score, 5),
                                          Best_Params=str(model_params_dict))

    stacked_model_eval_df = pd.DataFrame.from_dict(stacked_model_eval_dict,
                                                  orient='index').reset_index().rename(columns={'index': 'Model'})

    return stacked_model_eval_df
'''
#=============================================================================

def model_ensemble_classification(X,
                                  y,
                                  feature_trans,
                                  estimator_list,
                                  score_eval,
                                  model_perf_tuning_df):

    stacked_model_eval_dict = {}

    with open('config.yaml') as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)

    tree_models_short_name_dict = config_data['tree_models_short_name']
    df = model_perf_tuning_df.copy()
    df.set_index('Model', inplace=True)

    ll = estimator_list
    model_combinations_list = []
    for ii in range(2, len(ll)+1):
        if ii == 2:
            res = [list(ele) for ele in list(itertools.permutations(ll, ii))]
            model_combinations_list.extend(res)
        else:
            res = [list(ele) for ele in list(itertools.permutations(ll, ii))]
            final_list = []
            for i in res:
                pop_item = i.pop()
                sorted_list = sorted(i)
                final_list.append(sorted_list+[pop_item])

            final_list.sort()
            complete_list = list(final_list for final_list,_ in itertools.groupby(final_list))
            model_combinations_list.extend(complete_list)

    # model_combinations_tuple = permutations(estimator_list)
    for models_list in model_combinations_list:
        level0 = list()
        model_params_dict = {}
        for i in models_list[:-1]:
            level0.append((tree_models_short_name_dict[i][0],  globals()[i]()))

            for key, val in eval(df.loc[i, 'Best_Params']).items():
                key = key.replace('__', '__'+tree_models_short_name_dict[i][0]+'__')
                model_params_dict[key] = val

        # define meta learner model
        level1 = globals()[models_list[-1]]()
        final_model_short_name = tree_models_short_name_dict[models_list[-1]][0]

        for key, val in eval(df.loc[models_list[-1], 'Best_Params']).items():
            # key = key.replace('__', '__'+tree_models_short_name_dict[models_list[-1]][0]+'__')
            key = key.replace('__', '__final_estimator__')
            model_params_dict[key] = val

        # define the stacking ensemble
        model_stack = StackingClassifier(estimators=level0, final_estimator=level1)

        model_pipeline = Pipeline([('feat_trans', feature_trans),
                                   ('estimator', model_stack)])

        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        model_pipeline.set_params(**model_params_dict)
        oof_preds = cross_val_predict(estimator=model_pipeline,
                                      X=X,
                                      y=y,
                                      cv=cv,
                                      n_jobs=-1,
                                      method='predict',
                                      # fit_params=clf.best_params_,
                                      verbose=1)

        # oof_roc_auc_score = roc_auc_score(y, oof_preds)
        oof_eval_score = score_eval(y, oof_preds)

        stacking_str = ''
        for j in level0:
            stacking_str = stacking_str+j[0]+'+'
        stacking_str = stacking_str[:-1]
        stacking_str = stacking_str+'-->'+final_model_short_name

        stacked_model_eval_dict[stacking_str] = {'OOF_'+score_eval.__name__: round(oof_eval_score, 5),
                                          'Best_Params': str(model_params_dict)}

    stacked_model_eval_df = pd.DataFrame.from_dict(stacked_model_eval_dict,
                                                  orient='index').reset_index().rename(columns={'index': 'Model'})


    # stacked_model_eval_df = stacked_model_eval_df.sort_values(by='OOF_ROC_AUC_Score', ascending=False)
    #=========================================================================
    # Below is the code for pickle the best stacked model
    stacked_model_eval_df = stacked_model_eval_df.sort_values(by='OOF_'+score_eval.__name__, ascending=False)
    best_stacked_model_best_params = eval(stacked_model_eval_df.iloc[0].to_dict()['Best_Params'])
    # logger.info(f'Best Stacked')
    models_split = stacked_model_eval_df.iloc[0].to_dict()['Model'].split('-->')
    logger.info(f"Best Stacked Tree Model - {stacked_model_eval_df.iloc[0].to_dict()['Model']}")
    logger.info(f"\nBest Stacked Tree Model - {stacked_model_eval_df.iloc[0].to_dict()['Model']} Params:\n{pformat(best_stacked_model_best_params)}")
    final_esti = [key for key, value in tree_models_short_name_dict.items() if value[0] == models_split[-1]]
    level1 = globals()[final_esti[0]]()

    level0 = []
    for i in models_split[0].split('+'):
        level0_esti = [key for key, value in tree_models_short_name_dict.items() if value[0] == i]
        level0.append((i,  globals()[level0_esti[0]]()))

    model_stack = StackingClassifier(estimators=level0, final_estimator=level1)
    model_pipeline = Pipeline([('feat_trans', feature_trans),
                               ('estimator', model_stack)])
    model_pipeline.set_params(**best_stacked_model_best_params)
    with open('best_stacked_tree_model.plk', 'wb') as f:
        pickle.dump(model_pipeline, f)
    #=========================================================================

    # stacked_model_eval_df = stacked_model_eval_df.append(model_perf_tuning_df[['Model', 'OOF_ROC_AUC_Score', 'Best_Params']])
    all_tree_model_eval_df = stacked_model_eval_df.append(model_perf_tuning_df[['Model', 'OOF_'+score_eval.__name__, 'Best_Params']])


    all_tree_model_eval_df['Model'] = all_tree_model_eval_df['Model'].map(lambda x: tree_models_short_name_dict[x][0] if x in tree_models_short_name_dict else x)
    all_tree_model_eval_df = all_tree_model_eval_df.sort_values(by='OOF_'+score_eval.__name__, ascending=False)
    all_tree_model_eval_df.reset_index(drop=True, inplace=True)
    logger.info(f"\nAll Tree Models Evaluation Dataframe\n{all_tree_model_eval_df[['Model', 'OOF_'+score_eval.__name__]].to_string()}")
    best_tree_model = all_tree_model_eval_df.iloc[0].to_dict()['Model']
    best_tree_model_params = eval(all_tree_model_eval_df.iloc[0].to_dict()['Best_Params'])
    logger.info(f"Best Tree Model (Single+Stacked) : {best_tree_model}")
    logger.info(f"Best Tree Model (Single+Stacked) - {best_tree_model} Params\n{pformat(best_tree_model_params)}")
    # breakpoint()
    return all_tree_model_eval_df, best_tree_model


#=============================================================================

def model_ensemble(X,
                    y,
                    feature_trans,
                    estimator_list,
                    score_eval,
                    greater_the_better,
                    model_type,
                    model_perf_tuning_df,
                    n_splits,
                    n_jobs=-1):

    stacked_model_eval_dict = {}

    with open('config.yaml') as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)

    if model_type == 'Regression':
        models_short_name_dict = config_data['regression_models_short_name']
    if model_type == 'Classification':
        models_short_name_dict = config_data['classification_models_short_name']

    df = model_perf_tuning_df.copy()
    df.set_index('Model', inplace=True)

    ll = estimator_list
    model_combinations_list = []
    for ii in range(2, len(ll)+1):
        if ii == 2:
            res = [list(ele) for ele in list(itertools.permutations(ll, ii))]
            model_combinations_list.extend(res)
        else:
            res = [list(ele) for ele in list(itertools.permutations(ll, ii))]
            final_list = []
            for i in res:
                pop_item = i.pop()
                sorted_list = sorted(i)
                final_list.append(sorted_list+[pop_item])

            final_list.sort()
            complete_list = list(final_list for final_list,_ in itertools.groupby(final_list))
            model_combinations_list.extend(complete_list)

    # model_combinations_tuple = permutations(estimator_list)
    for models_list in model_combinations_list:
        print(f"""\n##################\n Ensemble Model : {models_list}\n##################""")
        level0 = list()
        model_params_dict = {}
        for i in models_list[:-1]:
            level0.append((models_short_name_dict[i][0],  globals()[i]()))

            for key, val in eval(df.loc[i, 'Best_Params']).items():
                key = key.replace('__', '__'+models_short_name_dict[i][0]+'__')
                model_params_dict[key] = val

        # define meta learner model
        level1 = globals()[models_list[-1]]()
        final_model_short_name = models_short_name_dict[models_list[-1]][0]

        for key, val in eval(df.loc[models_list[-1], 'Best_Params']).items():
            # key = key.replace('__', '__'+models_short_name_dict[models_list[-1]][0]+'__')
            key = key.replace('__', '__final_estimator__')
            model_params_dict[key] = val

        # define the stacking ensemble
        if model_type == 'Regression':
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            model_stack = StackingRegressor(estimators=level0, final_estimator=level1)

        if model_type == 'Classification':
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            model_stack = StackingClassifier(estimators=level0, final_estimator=level1)

        # model_pipeline = Pipeline([('feat_trans', feature_trans),
        #                            ('estimator', model_stack)])
        model_pipeline = Pipeline([('feat_trans', feature_trans),
                                   ('model_estimator', model_stack)])

        #cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        model_pipeline.set_params(**model_params_dict)
        oof_preds = cross_val_predict(estimator=model_pipeline,
                                      X=X,
                                      y=y,
                                      cv=cv,
                                      n_jobs=n_jobs,
                                      method='predict',
                                      # fit_params=clf.best_params_,
                                      verbose=1)

        if model_type == 'Regression':
            if score_eval == 'rmse':
                oof_eval_score = mean_squared_error(y_true=y.to_numpy(), y_pred=oof_preds, squared=False)
            if score_eval == 'mse':
                oof_eval_score = mean_squared_error(y_true=y.to_numpy(), y_pred=oof_preds, squared=True)
            if score_eval == 'rmsle':
                oof_eval_score = np.sqrt(mean_squared_log_error(y_true=y.to_numpy(), y_pred=oof_preds))

        if model_type == 'Classification':
            if score_eval == 'roc_auc_score':
                oof_eval_score = roc_auc_score(y_true=y.to_numpy(), y_pred=oof_preds)
            if score_eval == 'f1_score':
                oof_eval_score = f1_score(y_true=y.to_numpy(), y_pred=oof_preds)
                
        print(f"""\n{score_eval} OOF Model Evaluation Score : {models_list} - {oof_eval_score}""")
        # oof_roc_auc_score = roc_auc_score(y, oof_preds)
        # oof_eval_score = score_eval(y, oof_preds)

        stacking_str = ''
        for j in level0:
            stacking_str = stacking_str+j[0]+'+'
        stacking_str = stacking_str[:-1]
        stacking_str = stacking_str+'-->'+final_model_short_name

        # stacked_model_eval_dict[stacking_str] = {'OOF_'+score_eval.__name__: round(oof_eval_score, 5),
        #                                   'Best_Params': str(model_params_dict)}
        stacked_model_eval_dict[stacking_str] = {'OOF_'+score_eval : round(oof_eval_score, 5),
                                          'Best_Params': str(model_params_dict)}

    stacked_model_eval_df = pd.DataFrame.from_dict(stacked_model_eval_dict,
                                                  orient='index').reset_index().rename(columns={'index': 'Model'})


    # stacked_model_eval_df = stacked_model_eval_df.sort_values(by='OOF_ROC_AUC_Score', ascending=False)
    #=========================================================================
    # Below is the code for pickle the best stacked model
    # stacked_model_eval_df = stacked_model_eval_df.sort_values(by='OOF_'+score_eval.__name__, ascending=False)
    if greater_the_better:
        stacked_model_eval_df = stacked_model_eval_df.sort_values(by='OOF_'+score_eval, ascending=False)
    else:
        stacked_model_eval_df = stacked_model_eval_df.sort_values(by='OOF_'+score_eval, ascending=True)

    best_stacked_model_best_params = eval(stacked_model_eval_df.iloc[0].to_dict()['Best_Params'])
    # logger.info(f'Best Stacked')
    models_split = stacked_model_eval_df.iloc[0].to_dict()['Model'].split('-->')
    logger.info(f"Best Stacked Model - {stacked_model_eval_df.iloc[0].to_dict()['Model']}")
    logger.info(f"\nBest Stacked Model - {stacked_model_eval_df.iloc[0].to_dict()['Model']} Params:\n{pformat(best_stacked_model_best_params)}")
    final_esti = [key for key, value in models_short_name_dict.items() if value[0] == models_split[-1]]
    level1 = globals()[final_esti[0]]()

    level0 = []
    for i in models_split[0].split('+'):
        level0_esti = [key for key, value in models_short_name_dict.items() if value[0] == i]
        level0.append((i,  globals()[level0_esti[0]]()))

    if model_type == 'Regression':
        model_stack = StackingRegressor(estimators=level0, final_estimator=level1)
    if model_type == 'Classification':
        model_stack = StackingClassifier(estimators=level0, final_estimator=level1)

    model_pipeline = Pipeline([('feat_trans', feature_trans),
                               ('model_estimator', model_stack)])

    model_pipeline.set_params(**best_stacked_model_best_params)

    with open('best_stacked_model.plk', 'wb') as f:
        pickle.dump(model_pipeline, f)
    #=========================================================================

    # stacked_model_eval_df = stacked_model_eval_df.append(model_perf_tuning_df[['Model', 'OOF_ROC_AUC_Score', 'Best_Params']])
    # all_tree_model_eval_df = stacked_model_eval_df.append(model_perf_tuning_df[['Model', 'OOF_'+score_eval.__name__, 'Best_Params']])
    # all_tree_model_eval_df = stacked_model_eval_df.append(model_perf_tuning_df[['Model', 'OOF_'+score_eval, 'Best_Params']])


    # all_tree_model_eval_df['Model'] = all_tree_model_eval_df['Model'].map(lambda x: tree_models_short_name_dict[x][0] if x in tree_models_short_name_dict else x)
    # # all_tree_model_eval_df = all_tree_model_eval_df.sort_values(by='OOF_'+score_eval.__name__, ascending=False)
    # all_tree_model_eval_df = all_tree_model_eval_df.sort_values(by='OOF_'+score_eval, ascending=False)
    # all_tree_model_eval_df.reset_index(drop=True, inplace=True)
    # # logger.info(f"\nAll Tree Models Evaluation Dataframe\n{all_tree_model_eval_df[['Model', 'OOF_'+score_eval.__name__]].to_string()}")
    # logger.info(f"\nAll Tree Models Evaluation Dataframe\n{all_tree_model_eval_df[['Model', 'OOF_'+score_eval]].to_string()}")
    # best_tree_model = all_tree_model_eval_df.iloc[0].to_dict()['Model']
    # best_tree_model_params = eval(all_tree_model_eval_df.iloc[0].to_dict()['Best_Params'])
    # logger.info(f"Best Tree Model (Single+Stacked) : {best_tree_model}")
    # logger.info(f"Best Tree Model (Single+Stacked) - {best_tree_model} Params\n{pformat(best_tree_model_params)}")
    # # breakpoint()
    # return all_tree_model_eval_df, best_tree_model

    all_model_eval_df = stacked_model_eval_df.append(model_perf_tuning_df[['Model', 'OOF_'+score_eval, 'Best_Params']])
    all_model_eval_df['Model'] = all_model_eval_df['Model'].map(lambda x: models_short_name_dict[x][0] if x in models_short_name_dict else x)

    if greater_the_better:
        all_model_eval_df = all_model_eval_df.sort_values(by='OOF_'+score_eval, ascending=False)
    else:
        all_model_eval_df = all_model_eval_df.sort_values(by='OOF_'+score_eval, ascending=True)

    all_model_eval_df.reset_index(drop=True, inplace=True)
    logger.info(f"\nAll Models Evaluation Dataframe\n{all_model_eval_df[['Model', 'OOF_'+score_eval]].to_string()}")
    best_model = all_model_eval_df.iloc[0].to_dict()['Model']
    best_model_params = eval(all_model_eval_df.iloc[0].to_dict()['Best_Params'])
    logger.info(f"Best Model (Single+Stacked) : {best_model}")
    logger.info(f"Best Model (Single+Stacked) - {best_model} Params\n{pformat(best_model_params)}")

    return all_model_eval_df, best_model

#=============================================================================

def final_tree_model_training_pred(complete_X_train,
                                   complete_y_train,
                                   test_data,
                                   best_tree_model):
    #Method for final training with complete dataset and test data prediction using
    #best tree based model
    # best_score_tree_model = all_tree_model_eval_df.sort_values(by='OOF_'+score_eval.__name__, ascending=False)
    # breakpoint()
    if '-->' in best_tree_model:
        with open('best_stacked_tree_model.plk', 'rb') as f:
            best_tree_model_pipeline = pickle.load(f)
        # best_tree_model_pipeline.fit(complete_X_train, complete_y_train)
        # test_pred = best_tree_model_pipeline.predict(test_data)
    else:
        with open('best_single_tree_model.plk', 'rb') as f:
            best_tree_model_pipeline = pickle.load(f)

    # breakpoint()
    best_tree_model_pipeline.fit(complete_X_train, complete_y_train)
    test_pred = best_tree_model_pipeline.predict(test_data)

    return test_pred

#=============================================================================

def final_model_training(complete_X_train,
                            complete_y_train,
                            best_model):
    # Method for final training with complete dataset and test data prediction using best model
    if '-->' in best_model:
        with open('best_stacked_model.plk', 'rb') as f:
            best_model_pipeline = pickle.load(f)
    else:
        with open('best_single_model.plk', 'rb') as f:
            best_model_pipeline = pickle.load(f)

    best_model_pipeline.fit(complete_X_train, complete_y_train)

    with open('final_best_model.plk', 'wb') as f:
        pickle.dump(best_model_pipeline, f)

    return None

#=============================================================================

def kaggle_submission(test_df,
                      test_pred,
                      id_feature,
                      target_feature,
                      sub_file_name,
                      submission_msg,
                      competition_name,):

    my_submission = pd.DataFrame({id_feature: test_df[id_feature],
                                  target_feature: test_pred})

    sub_file_name = sub_file_name.split('.')[0]+str(arrow.now().format('YYYY-MM-DD_HH-mm-ss'))+'.csv'
    my_submission.to_csv(sub_file_name, index=False)


    #============================================================================
    #Kaggle Submission:
    #Check this for more info-https://technowhisp.com/kaggle-api-python-documentation/
    os.environ['KAGGLE_USERNAME'] = 'raghuvalusa'
    os.environ['KAGGLE_KEY'] = '10728ec404bfc529c649a7af141bd857'

    api = KaggleApi()
    api.authenticate()

    # competitions = api.competitions_list(search='cat',category="playground")
    # api.competitions_list_cli(search='titanic')

    # competition_submit(file_name, message, competition, quiet=False)
    api.competition_submit(sub_file_name, submission_msg, competition_name)

    # competition_view_leaderboard(id, **kwargs)
    leaderboard = api.competition_view_leaderboard(competition_name)

#=============================================================================

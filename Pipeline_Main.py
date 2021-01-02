
import yaml
import os
import pprint
import pickle
from pprint import pformat
import pandas as pd
pd.options.display.max_rows=999
pd.set_option('display.max_columns', 500)
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))
import numpy as np
import arrow
from collections import OrderedDict
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, make_scorer
from kaggle.api.kaggle_api_extended import KaggleApi
import logging
np.random.seed(42)
import Custom_Transformers_Methods as ctm


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
if not logger.hasHandlers():
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
    file_handler = logging.FileHandler('credit_default_prediction_classification.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

logger.info(f"####################### {arrow.now().format('MM/DD/YYYY HH:mm:ss - dddd, MMMM,YYYY')} ################################")

train_data = pd.read_csv('./train.csv')
test_data  = pd.read_csv('./test.csv')

train_data.columns = ['_'.join(col.split(' ')) for col in train_data.columns]
test_data.columns = ['_'.join(col.split(' ')) for col in test_data.columns]

train_data['Credit_Score'] = [score/10 if score > 800 else score for score in train_data['Credit_Score']]
test_data['Credit_Score'] = [score/10 if score > 800 else score for score in test_data['Credit_Score']]



id_feat = ['Id']
target_feat = ['Credit_Default']


X = train_data.drop(labels=target_feat+id_feat, axis=1).reset_index(drop=True)
# X = train_data.drop(labels=target_feat, axis=1).reset_index(drop=True)
y = train_data[target_feat[0]].reset_index(drop=True)

numerical_vars_list, categorical_vars_list, string_vars_list, temporal_vars_list = ctm.dataset_datatypes_counts(X)

logger.info(f"""\nTrain dataset columns with null values: {np.count_nonzero(X.isnull().sum().sort_values(ascending=False).values)}
{X.isnull().sum().sort_values(ascending=False)}""")

s = X[numerical_vars_list].isnull().sum()
logger.info(f"""
:: Numerical Features with null values ::
{s[s>0].sort_values(ascending=False)}""")

s = X[categorical_vars_list].isnull().sum()
logger.info(f"""
:: Categorical Features with null values ::
{s[s>0].sort_values(ascending=False)}""")

feature_transform_dict = OrderedDict({'Years_in_current_job': {
                                                'strategy': 'value_mapping',
                                                'map_values': {'10+ years': 10, 
                                                               '9 years': 9, 
                                                               '8 years': 8, 
                                                               '7 years': 7, 
                                                               '6 years': 6, 
                                                               '5 years': 5, 
                                                               '4 years': 4,
                                                               '3 years': 3,
                                                               '2 years': 2,
                                                               '1 year': 1,
                                                               '< 1 year': 0.5}},
                                    'Term': {'strategy': 'value_mapping',
                                             'map_values': {'Short Term': 0,
                                             'Long Term': 1}},
                                    'Purpose': {'strategy': 'value_mapping',
                                                'map_values': {'other': 'Other', 'medical bills': 'Other', 'major purchase': 'Other', 'take a trip': 'Other', 
                                                'buy house': 'Other', 'small business': 'Other', 'wedding': 'Other', 'moving': 'Other', 
                                                'educational expenses': 'Other', 'vacation': 'Other', 'renewable energy': 'Other', 'business loan': 'Other',
                                                'buy a car': 'Other'}},
                                    })

feature_binning_dict = OrderedDict({'Monthly_Debt': {'column_name': 'Monthly_Debt_Binned',
                                                     'bins': 10,
                                                     'labels': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                                     'drop_orig_column': False},
                                    'Current_Loan_Amount': {'column_name': 'Current_Loan_Amount_Binned',
                                                     'bins': 10,
                                                     'labels': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                                     'drop_orig_column': False},
                                    })


num_feature_impute_dict = OrderedDict({'Bankruptcies': {'strategy': 'based_on_other_column',
                                            'other_column': 'Number_of_Credit_Problems',
                                            'group_by_transformation': False},
                                       'Years_in_current_job': {'missing_values': np.nan,
                                                                'strategy': 'constant',
                                                                'constant_value': 0},
                                       'Annual_Income': {'strategy': 'based_on_other_column',
                                            'other_column': ['Monthly_Debt_Binned'],
                                            'group_by_transformation': True},                                       
                                       'Credit_Score': {'strategy': 'based_on_other_column',
                                            'other_column': ['Current_Loan_Amount_Binned'],
                                            'group_by_transformation': True},
                                       })


cat_lbl_encode_list = ['Home_Ownership', 'Purpose',]

cat_1hot_encode_list = ['Home_Ownership', 'Purpose',]

feature_selection_dict = {'featCorrelationThreshold': 0.8,
                          'model_type': 'Classification',
                          'threshold': 0.001,
                          'n_largest_features': 20}


analyze_feature_transform_pipeline = Pipeline([
('Feature_Drop_W_High_Missing_Values', ctm.Custom_Missing_Values_Check_Column_Drop(missing_val_percentage=0.5, loginfo=True)),
('Custom_Feat_Transformation', ctm.Custom_Features_Transformer(feature_transform_dict=feature_transform_dict, loginfo=True)),
('Feature_Binning', ctm.Custom_Features_Binning_Transformer(feature_binning_dict=feature_binning_dict, loginfo=True)),
('Num_Feat_Imputation', ctm.Custom_SimpleImputer(feature_imputer_dict=num_feature_impute_dict, loginfo=True)),
('Cat_LabelEncoder', ctm.Custom_LabelEncoder(feature_lbl_encode_list=cat_lbl_encode_list, loginfo=True)),
('Feat_Selection', ctm.Custom_Feature_Selection(feature_selection_dict=feature_selection_dict, loginfo=True)),
('Cat_OneHotEncoder', ctm.Custom_OneHotEncoder(feature_1hot_encode_list=cat_1hot_encode_list, loginfo=True)),
])


# analyze_feature_transform_pipeline1 = Pipeline([
# ('Feature_Drop_W_High_Missing_Values', ctm.Custom_Missing_Values_Check_Column_Drop(missing_val_percentage=0.5, loginfo=True)),
# ('Custom_Feat_Transformation', ctm.Custom_Features_Transformer(feature_transform_dict=feature_transform_dict, loginfo=True)),
# ('Feature_Binning', ctm.Custom_Features_Binning_Transformer(feature_binning_dict=feature_binning_dict, loginfo=True)),
# ('Num_Feat_Imputation', ctm.Custom_SimpleImputer(feature_imputer_dict=num_feature_impute_dict, loginfo=True)),
# ('Cat_LabelEncoder', ctm.Custom_LabelEncoder(feature_lbl_encode_list=cat_lbl_encode_list, loginfo=True)),
# #('Feat_Selection', ctm.Custom_Feature_Selection(feature_selection_dict=feature_selection_dict, loginfo=True)),
# ('Cat_OneHotEncoder', ctm.Custom_OneHotEncoder(feature_1hot_encode_list=cat_1hot_encode_list, loginfo=True)),
# ])

breakpoint()
# train_transformed_df = analyze_feature_transform_pipeline1.fit_transform(X)
# cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
# test_transformed_df = analyze_feature_transform_pipeline1.transform(test_data)



transformed_df = analyze_feature_transform_pipeline.fit_transform(X, y)
logger.info(f"\nFinal Tranformed Dataframe:\n{transformed_df.sample(20).to_string()}")
logger.info(f"\nFinal Tranformed Dataframe shape:\n{transformed_df.shape}")


feature_transform_pipeline = Pipeline([
('Feature_Drop_W_High_Missing_Values', ctm.Custom_Missing_Values_Check_Column_Drop(missing_val_percentage=0.5, loginfo=False)),
('Custom_Feat_Transformation', ctm.Custom_Features_Transformer(feature_transform_dict=feature_transform_dict, loginfo=False)),
('Feature_Binning', ctm.Custom_Features_Binning_Transformer(feature_binning_dict=feature_binning_dict, loginfo=False)),
('Num_Feat_Imputation', ctm.Custom_SimpleImputer(feature_imputer_dict=num_feature_impute_dict, loginfo=False)),
('Cat_LabelEncoder', ctm.Custom_LabelEncoder(feature_lbl_encode_list=cat_lbl_encode_list, loginfo=False)),
('Feat_Selection', ctm.Custom_Feature_Selection(feature_selection_dict=feature_selection_dict, loginfo=False)),
('Cat_OneHotEncoder', ctm.Custom_OneHotEncoder(feature_1hot_encode_list=cat_1hot_encode_list, loginfo=False)),
])



cv_n_splits = 2
randomsearchcv_n_iter = 100
model_perf_tuning_df = ctm.model_perf_tuning(X=X,  #X=X_train,
                                y=y, #y=y_train,
                                feature_trans=feature_transform_pipeline,
                                estimator_list=[#'LGBMRegressor',
                                                #'RandomForestRegressor',
                                                #'GradientBoostingRegressor',
                                                #'RandomForestClassifier',                                                
                                                'LGBMClassifier',
                                                #'GradientBoostingClassifier',
                                                #'CatBoostClassifier',
                                                ],
                                model_type='Classification',
                                score_eval='f1_score',
                                greater_the_better=True,
                                cv_n_splits=cv_n_splits,
                                randomsearchcv_n_iter=randomsearchcv_n_iter,
                                n_jobs=6)

'''
all_model_eval_df, best_model = ctm.model_ensemble(X=X, #X=X_train,
                                y=y, #y=y_train,
                                feature_trans=feature_transform_pipeline,
                                estimator_list=[#'LGBMRegressor',
                                                #'RandomForestRegressor',
                                                #'GradientBoostingRegressor',
                                                'RandomForestClassifier',
                                                'LGBMClassifier',
                                                #'GradientBoostingClassifier',
                                                #'CatBoostClassifier'
                                                 ],
                                model_type='Classification',
                                score_eval='f1_score',
                                greater_the_better=True,
                                model_perf_tuning_df=model_perf_tuning_df,
                                n_splits=cv_n_splits,
                                n_jobs=6)
'''

test_dataset = test_data.drop(labels=id_feat, axis=1).reset_index(drop=True)
# transformed_test_dataset = analyze_feature_transform_pipeline.transform(test_dataset)
# logger.info(f"""\nTransformed Test Dataset:\n{transformed_test_dataset.to_string()} """)
# logger.info(f"""\nTransformed Test Dataset shape:\n{transformed_test_dataset.shape} """)

best_model = 'lgbmc'
# Training the best model with the complete dataset to pickle the final best model for test data prediction
ctm.final_model_training(complete_X_train=X,
                        complete_y_train=y,
                        best_model=best_model)

# Final Test Data Predictions
with open('final_best_model.plk', 'rb') as f:
    best_model_pipeline = pickle.load(f)

test_pred = best_model_pipeline.predict(test_dataset)


#============================================================================
#Kaggle Submission:
#Check this for more info-https://technowhisp.com/kaggle-api-python-documentation/
if True:
#if False:
    test_df = test_data
    test_pred = test_pred
    id_feature = id_feat[0]
    target_feature = target_feat[0]
    sub_file_name = 'credit_default_prediction'
    submission_msg = 'Final_Test_Data_Submission'
    competition_name = 'credit-default-prediction-ai-big-data'
    ctm.kaggle_submission(test_df=test_df,
                      test_pred=test_pred,
                      id_feature=id_feature,
                      target_feature=target_feature,
                      sub_file_name=sub_file_name,
                      submission_msg=submission_msg,
                      competition_name=competition_name)

#========================================================



















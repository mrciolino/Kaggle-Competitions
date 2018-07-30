# Tutorial found at kaggle kernel Will K. - Intro to Auto Feature Engr
# https://www.kaggle.com/willkoehrsen/automated-feature-engineering-basics

import pickle
import featuretools as ft
from progress.bar import FillingSquaresBar
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 22
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# Read in the datasets and limit to the first 1000 rows (sorted by SK_ID_CURR)
# This allows us to actually see the results in a reasonable amount of time!
bar = FillingSquaresBar('Loading Files', max=8)
app_train = pd.read_csv('/Users/mrciolino/Documents/Documents/PythonLearning/Kaggle Competetions/Home Credit/Home_Credit_Data/application_train.csv').sort_values(
    'SK_ID_CURR').reset_index(drop=True).loc[:1000, :]
bar.next()
app_test = pd.read_csv('/Users/mrciolino/Documents/Documents/PythonLearning/Kaggle Competetions/Home Credit/Home_Credit_Data/application_test.csv').sort_values(
    'SK_ID_CURR').reset_index(drop=True).loc[:1000, :]
bar.next()
bureau = pd.read_csv('/Users/mrciolino/Documents/Documents/PythonLearning/Kaggle Competetions/Home Credit/Home_Credit_Data/bureau.csv').sort_values(
    ['SK_ID_CURR', 'SK_ID_BUREAU']).reset_index(drop=True).loc[:1000, :]
bar.next()
bureau_balance = pd.read_csv('/Users/mrciolino/Documents/Documents/PythonLearning/Kaggle Competetions/Home Credit/Home_Credit_Data/bureau_balance.csv').sort_values(
    'SK_ID_BUREAU').reset_index(drop=True).loc[:1000, :]
bar.next()
cash = pd.read_csv('/Users/mrciolino/Documents/Documents/PythonLearning/Kaggle Competetions/Home Credit/Home_Credit_Data/POS_CASH_balance.csv').sort_values(
    ['SK_ID_CURR', 'SK_ID_PREV']).reset_index(drop=True).loc[:1000, :]
bar.next()
credit = pd.read_csv('/Users/mrciolino/Documents/Documents/PythonLearning/Kaggle Competetions/Home Credit/Home_Credit_Data/credit_card_balance.csv').sort_values(
    ['SK_ID_CURR', 'SK_ID_PREV']).reset_index(drop=True).loc[:1000, :]
bar.next()
previous = pd.read_csv('/Users/mrciolino/Documents/Documents/PythonLearning/Kaggle Competetions/Home Credit/Home_Credit_Data/previous_application.csv').sort_values(
    ['SK_ID_CURR', 'SK_ID_PREV']).reset_index(drop=True).loc[:1000, :]
bar.next()
installments = pd.read_csv('/Users/mrciolino/Documents/Documents/PythonLearning/Kaggle Competetions/Home Credit/Home_Credit_Data/installments_payments.csv').sort_values([
    'SK_ID_CURR', 'SK_ID_PREV']).reset_index(drop=True).loc[:1000, :]
bar.next()
bar.finish()


# Add identifying column
app_train['set'] = 'train'
app_test['set'] = 'test'
app_test["TARGET"] = np.nan

# Append the dataframes
app = app_train.append(app_test, ignore_index=True)

# Entity set with id applications
es = ft.EntitySet(id='clients')

# Entities with a unique index
es = es.entity_from_dataframe(entity_id='app', dataframe=app, index='SK_ID_CURR')
es = es.entity_from_dataframe(entity_id='bureau', dataframe=bureau, index='SK_ID_BUREAU')
es = es.entity_from_dataframe(entity_id='previous', dataframe=previous, index='SK_ID_PREV')

# Entities that do not have a unique index
es = es.entity_from_dataframe(entity_id='bureau_balance', dataframe=bureau_balance, make_index=True, index='bureaubalance_index')
es = es.entity_from_dataframe(entity_id='cash', dataframe=cash, make_index=True, index='cash_index')
es = es.entity_from_dataframe(entity_id='installments', dataframe=installments, make_index=True, index='installments_index')
es = es.entity_from_dataframe(entity_id='credit', dataframe=credit, make_index=True, index='credit_index')


print('Parent: app, Parent Variable: SK_ID_CURR \n \n', app.iloc[:, 111:115].head())
print('\nChild: bureau, Child Variable: SK_ID_CURR \n \n', bureau.iloc[10:30, :4].head())


# Relationship between app and bureau
r_app_bureau = ft.Relationship(es['app']['SK_ID_CURR'], es['bureau']['SK_ID_CURR'])

# Relationship between bureau and bureau balance
r_bureau_balance = ft.Relationship(es['bureau']['SK_ID_BUREAU'], es['bureau_balance']['SK_ID_BUREAU'])

# Relationship between current app and previous apps
r_app_previous = ft.Relationship(es['app']['SK_ID_CURR'], es['previous']['SK_ID_CURR'])

# Relationships between previous apps and cash, installments, and credit
r_previous_cash = ft.Relationship(es['previous']['SK_ID_PREV'], es['cash']['SK_ID_PREV'])
r_previous_installments = ft.Relationship(es['previous']['SK_ID_PREV'], es['installments']['SK_ID_PREV'])
r_previous_credit = ft.Relationship(es['previous']['SK_ID_PREV'], es['credit']['SK_ID_PREV'])

# Add in the defined relationships
es = es.add_relationships([r_app_bureau, r_bureau_balance, r_app_previous,
                           r_previous_cash, r_previous_installments, r_previous_credit])

# Print out the EntitySet
print es

# Default primitives from featuretools
default_agg_primitives =  ["sum", "std", "max", "skew", "min", "mean", "count", "percent_true", "num_unique", "mode"]
default_trans_primitives =  ["day", "year", "month", "weekday", "haversine", "numwords", "characters"]

# Specify the aggregation primitives
feature_matrix_spec, feature_names_spec = ft.dfs(entityset = es, target_entity = 'app',
                                                 agg_primitives = ['sum', 'count', 'min', 'max', 'mean', 'mode'],
                                                 max_depth = 2, features_only = False, verbose = False)


features_train, features_test, labels_train, labels_test = train_test_split(
    feature_matrix_spec, feature_names_spec, test_size=.2, random_state=42)

print len(features_train)
print len(labels_train)

lgb_train = lgb.Dataset(features_train, labels_train)
lgb_eval = lgb.Dataset(features_test, labels_test, reference=lgb_train)

# specify your configurations as a dict
params = {'task': 'train', 'boosting_type': 'gbdt', 'objective': 'regression', 'metric': 'auc', 'num_leaves': 31,
          'learning_rate': 0.05, 'feature_fraction': 0.9, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'verbose': 0}

print('Start training...')
gbm = lgb.train(params, lgb_train, num_boost_round=1000, valid_sets=lgb_eval, early_stopping_rounds=50)

print('Start predicting...')
y_pred = gbm.predict(features_test, num_iteration=gbm.best_iteration)
print('The Mean-Squared-Error of prediction is:', mean_squared_error(labels_test, y_pred) ** 0.5)
print('Feature importances:', list(gbm.feature_importance())[0:25])

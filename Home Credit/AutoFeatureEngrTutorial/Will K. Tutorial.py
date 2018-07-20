# Tutorial found at kaggle kernel Will K. - Intro to Auto Feature Engr
# https://www.kaggle.com/willkoehrsen/automated-feature-engineering-basics

import pickle
import featuretools as ft
from progress.bar import FillingSquaresBar
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

# List the primitives in a dataframe
primitives = ft.list_primitives()
pd.options.display.max_colwidth = 100
primitives[primitives['type'] == 'aggregation'].head(10)
primitives[primitives['type'] == 'transform'].head(10)

default_agg_primitives = ["sum", "std", "max", "skew", "min", "mean", "count", "percent_true", "num_unique", "mode"]
default_trans_primitives = ["day", "year", "month", "weekday", "haversine", "numwords", "characters"]


# DFS with default primitives
feature_matrix, feature_names = ft.dfs(entityset=es, target_entity='app',
                                       trans_primitives=default_trans_primitives,
                                       agg_primitives=default_agg_primitives,
                                       max_depth=2, features_only=False, verbose=True)

pd.options.display.max_columns = 1700
feature_matrix.head(10)
print "Last 20 Feature Names \n", feature_names[-20:]


# Specify the aggregation primitives
feature_matrix_spec, feature_names_spec = ft.dfs(entityset=es, target_entity='app',
                                                 agg_primitives=['sum', 'count', 'min', 'max', 'mean', 'mode'],
                                                 max_depth=2, features_only=False, verbose=True)

pd.options.display.max_columns = 1000
feature_matrix_spec.head(10)
print "Last 20 Feature Names \n", feature_names[-20:]

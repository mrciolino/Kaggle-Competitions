from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold, cross_validate

from home_credit_data_tools import data_creation_method, create_submission, save_classifier
import lightgbm as lgb
import numpy as np
import warnings
import pickle
import time

# Data Creation Options - if you dont have the dataset you must
# set both options to false to use the included pickle dataset
# dataset can be found on Kaggle within the Home_Credit competetion
create_new_data = True
pickle_new_data = False
save_clf = True

# creating/feteching the data
features_train, features_test, labels_train, labels_test, features_test_final =\
    data_creation_method(create_new_data, pickle_new_data, train_data_size=307511)

# Simple LightGBM with preset settings
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

# save classifier to file
if save_clf == True:
    try:
        save_classifier(clf)
    except:
        try:
            save_classifier(gbm)
        except:
            gbm.save_model('/Users/mrciolino/Documents/Documents/PythonLearning/Kaggle Competetions/Home Credit/LightGBM.txt')

create_submission(y_pred)

import pickle
import warnings
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold, cross_validate
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import classification_report
from home_credit_data_tools import data_creation_method, create_submission, save_clf_data_to_pickle


# Data Creation Options - if you dont have the dataset you must
# set both options to false to use the included pickle dataset
# dataset can be found on Kaggle within the Home_Credit competetion
create_new_data = True
pickle_new_data = False
save_classifier = True

# creating/feteching the data
features_train, features_test, labels_train, labels_test, features_test_final =\
    data_creation_method(create_new_data, pickle_new_data, train_data_size = 10000)

# Tuned Support Vector Regression Classifier using GridSearchCV
grid_values = {'tol': np.linspace(.0003, .001, 5), 'C': np.linspace(1, 3, 5), }
clf = GridSearchCV(LinearSVC(), grid_values)

print "********** Classifier Creation **********"

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    # select k-best feature selection
    print "Selecting the 25-best Features"
    selection = SelectKBest(k=10)
    selection.fit(features_train, labels_train)
    features_train_Kbest = selection.transform(features_train)
    features_test_Kbest = selection.transform(features_test)
    features_test_final_Kbest = selection.transform(features_test_final)
    print " "

    # Kfold Corss Validation
    print "==Cross Validation on the training data=="
    X = np.array(features_train_Kbest)
    y = np.array(labels_train)
    k_fold = StratifiedKFold(n_splits=10, shuffle=True)
    cv = cross_validate(clf, X, y, cv=k_fold, n_jobs=1, scoring=['accuracy'])
    print "Cross Validation Accuracy test:", "%.2f" % (100 * np.mean(cv['test_accuracy']))
    print " "

    # Predicting the Testing Data
    clf.fit(features_train_Kbest, labels_train)
    pred = clf.predict(features_test_Kbest)
    print "Parameters for the Classifier"
    print clf.best_estimator_
    print " "
    print "Classification Report"
    print classification_report(pred, labels_test)


    if save_classifier == True:
        clf_pickle_file = '/Users/mrciolino/Documents/Documents/PythonLearning/Kaggle Competetions/Home Credit/pickled_clf.pickle'
        save_clf_data_to_pickle(clf_pickle_file, clf)

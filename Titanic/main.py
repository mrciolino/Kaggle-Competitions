"""
Competition Description

The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during
her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and
crew. This sensational tragedy shocked the international community and led to better safety regulations
for ships.

One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the
passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of
people were more likely to survive than others, such as women, children, and the upper - class.

In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In
particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.
"""

# print __doc__
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append("Kaggle Competetions/Titanic")
from tools import data_format, data_split, bar_graph


# Import CSV file into pandas and fill in empty data with numerical values and convert to numpy array
train_data = data_format("Kaggle Competetions/Titanic/titanic - data/train.csv")
test_data = data_format("Kaggle Competetions/Titanic/titanic - data/test.csv")


# Format/split data into usable training and testing data
from sklearn.model_selection import train_test_split
targets, features, features_test_final = data_split(train_data, test_data)
features_train, features_test, labels_train, labels_test = train_test_split(
    features, targets, test_size=0.5, random_state=42)


########################################  Tuned Support Vector Machine  ############################################
"""
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

grid_values = {'C': [2], }
clf = GridSearchCV(SVC(kernel='linear'), grid_values)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print("Tuned SVM Accuracy Score: %.2f Percent" % (100 * clf.best_score_))
print clf.best_estimator_
"""

##################################### Tuned Logistic Regression ##############################################
"""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

grid_values = {'penalty': ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
clf = GridSearchCV(LogisticRegression(), grid_values)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print("Tuned Logistic Regression Accuracy: %.2f Percent" % (100 * clf.best_score_))
print clf.best_estimator_
"""
##################################### Tuned AdaBoostClassifier  #############################################
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

grid_values = {'n_estimators': [80, 90],
               'learning_rate': np.linspace(1, 1.5, 4),
               'algorithm': ['SAMME', 'SAMME.R'], }
clf_final = GridSearchCV(AdaBoostClassifier(), grid_values)
clf_final.fit(features_train, labels_train)
pred = clf_final.predict(features_test)
print("Tuned AdaBoostClassifier Accuracy: %.2f Percent" % (100 * clf_final.best_score_))
print clf_final.best_estimator_
bar_graph("Kaggle Competetions/Titanic/titanic - data/train.csv", clf_final.best_estimator_)

# Validation
# confusion_matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test, pred)
print(cm)

####################### Tuned AdaBoostClassifier With Priniciple Component Analysis #########################
"""
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

parameters = {'clf__learning_rate': np.linspace(.75, 1.25, 3),
              'pca__n_components': [2, 5, 7], }

pipeline = Pipeline([('pca', PCA()),
                     ('clf', AdaBoostClassifier()), ])

gs = GridSearchCV(estimator=pipeline, param_grid=parameters, scoring='accuracy', cv=5)
gs.fit(features_train, labels_train)
print('Tuned AdaBoostClassifier With PCA Accuracy: %.3f Percent' % (100 * gs.best_score_))
print gs.best_estimator_
"""

# predicting the test data and formatting for submission
prediction = clf_final.predict(features_test_final)
ids = range(892, 1310)
output = pd.DataFrame({'PassengerId': ids, 'Survived': prediction})
output.to_csv('Kaggle Competetions/Titanic/submission_data.csv', index=False)

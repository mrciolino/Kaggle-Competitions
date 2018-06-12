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

# visualize strength of each feature
bar_graph("Kaggle Competetions/Titanic/titanic - data/train.csv", clf_final.best_estimator_)

# predicting the test data and formatting for submission
prediction = clf_final.predict(features_test_final)
ids = range(892, 1310)
output = pd.DataFrame({'PassengerId': ids, 'Survived': prediction})
output.to_csv('Kaggle Competetions/Titanic/submission_data.csv', index=False)

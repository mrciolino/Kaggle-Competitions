"""
Competition Description

    Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement
ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much
more influences price negotiations than the number of bedrooms or a white-picket fence.

    With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this
competition challenges you to predict the final price of each home.
"""

print __doc__

import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("Kaggle Competetions/House Prices")

from tools import data_format
train_data = data_format("Kaggle Competetions/House Prices/House Prices - Data/train.csv")
test_data = data_format("Kaggle Competetions/House Prices/House Prices - Data/test.csv")

from tools import data_split
from sklearn.model_selection import train_test_split
targets, features, features_test_final = data_split(train_data, test_data)
features_train, features_test, labels_train, labels_test = train_test_split(
    features, targets, test_size=0.9, random_state=42)


# Remove Outliers
import matplotlib.pyplot as plt
plt.subplot(1, 2, 1)
boxplot = plt.boxplot(labels_train, notch=True)
outliers = boxplot["fliers"][0].get_data()[1]
indices = [i for i, x in enumerate(labels_train) if x in outliers]
i = 0
for num in indices:
    del labels_train[num - i]
    del features_train[num - i]
    i += 1

# select k-best feature selection
from sklearn.feature_selection import SelectKBest
selection = SelectKBest(k=10)
selection.fit(features_train, labels_train)
features_train_Kbest = selection.transform(features_train)
features_test_Kbest = selection.transform(features_test)
features_test_final_Kbest = selection.transform(features_test_final)

# Tuned Support Vector Regression Classifier using GridSearchCV
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
grid_values = {'C': [550, 600, 650], 'epsilon': np.linspace(15, 25, 3), }
clf = GridSearchCV(SVR(kernel='linear'), grid_values)

# Kfold Corss Validation
from sklearn.model_selection import KFold, cross_validate
X = np.array(features_train_Kbest)
y = np.array(labels_train)
metrics = ('explained_variance', 'r2')
k_fold = KFold(n_splits=5, shuffle=True)
cv = cross_validate(clf, X, y, cv=k_fold, n_jobs=1, scoring=metrics)
print "Cross Validation r2 test:", "%.3f" % np.mean(cv['test_r2'])
print "Cross Validation explained_variance test:", "%.3f" % np.mean(cv['test_explained_variance'])

# Predicting the Testing Data
clf.fit(features_train_Kbest, labels_train)
pred = clf.predict(features_test_Kbest)
print("Tuned SVR Accuracy Score: %.2f Percent" % (100 * clf.best_score_))
print clf.best_estimator_

# Plot the Predictions vs. True Value
import matplotlib.pyplot as plt
plt.subplot(1, 2, 2)
scatter = plt.scatter(labels_test, pred)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()

# create submission data
from tools import create_submission
create_submission(clf, features_test_final_Kbest)

"""
Competition Description

    Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement
ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much
more influences price negotiations than the number of bedrooms or a white-picket fence.

    With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this
competition challenges you to predict the final price of each home.
"""

# print __doc__

import sys
sys.path.append("Kaggle Competetions/House Prices")
from tools import data_format, data_split, bar_graph

train_data = data_format("Kaggle Competetions/House Prices/House Prices - Data/train.csv")
test_data = data_format("Kaggle Competetions/House Prices/House Prices - Data/test.csv")


from sklearn.model_selection import train_test_split
targets, features, features_test_final = data_split(train_data, test_data)
features_train, features_test, labels_train, labels_test = train_test_split(
    features, targets, test_size=0.875, random_state=42)


from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
grid_values = {'C': [150, 200, 250], }
clf = GridSearchCV(SVR(kernel='linear'), grid_values)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print("Tuned SVR Accuracy Score: %.2f Percent" % (100 * clf.best_score_))
print clf.best_estimator_

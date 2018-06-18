# Tools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def data_format(file):
    # Read the data from a csv file
    data = pd.read_csv(file)
    # Select features you want
    try:
        my_data = data[["MSSubClass", "LotArea", "LotFrontage", "OverallQual", "OverallCond", "YearBuilt",
                        "ExterQual", "ExterCond", "1stFlrSF", "2ndFlrSF", "FullBath", "HalfBath",
                        "TotRmsAbvGrd", "GarageArea", "Fireplaces", "PoolArea", "SalePrice"]]
    except:
        my_data = data[["MSSubClass", "LotArea", "LotFrontage", "OverallQual", "OverallCond", "YearBuilt",
                        "ExterQual", "ExterCond", "1stFlrSF", "2ndFlrSF", "FullBath", "HalfBath",
                        "TotRmsAbvGrd", "GarageArea", "Fireplaces", "PoolArea"]]

    # Convert text-classification to numerical range (1-200)
    pd.options.mode.chained_assignment = None  # default='warn'  # handle replacement warning
    my_data.replace({'Ex': 200, 'Gd': 150, 'TA': 100, 'Fa': 50, 'Po': 0}, inplace=True)
    # Convert to list of arrays
    features_array = my_data.values
    # handle NANs
    features_array[np.isnan(features_array)] = 0
    # return dataset
    return features_array


def data_split(train_data, test_data):
    target_train = []
    features_train = []
    features_test = []

    for item in train_data:
        target_train.append(item[-1])
        features_train.append(item[:-1])

    for item in test_data:
        features_test.append(item[0:])

    return target_train, features_train, features_test


def create_submission(clf, features_test_final):
    prediction = clf.predict(features_test_final)
    ids = range(1461, 2920)
    output = pd.DataFrame({'Id': ids, 'SalePrice': prediction})
    output.to_csv('Kaggle Competetions/House Prices/submission.csv', index=False)

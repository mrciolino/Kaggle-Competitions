# Data Formatting
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import confusion_matrix


def data_format(file):
    # Read the data from a csv file
    data = pd.read_csv(file, index_col=0)

    # Select what data you want to take out of the pandas dataframe
    #   disregard Cabin since only 22.89% of people have data on cabin
    #   disregard ticket # since no observable pattern of number distrubution
    #   disregard names since it does not affect the data
    try:
        raw_data = data[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    except:
        raw_data = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

    # Fill in blank gaps in numberical data points with the median values of that data point
    median_data = raw_data.dropna().median()
    filled_data = raw_data.fillna(median_data)

    # Convert to homogeneous numpy array of floating point values and create a copy of the dataset
    features_array = filled_data.values
    my_dataset = features_array

    # convert male/female to 0/1
    length = len(my_dataset)
    for name in range(0, length):
        value = (len(my_dataset[1]) - 6)
        gender = my_dataset[name][value]
        if gender == 'female':
            my_dataset[name][value] = 1
        else:
            my_dataset[name][value] = 0

    # convert embarked (C = Cherbourg, Q = Queenstown, S = Southampton) to C=1,Q=2,S=3
    for name in range(0, length):
        value = (len(my_dataset[1]) - 1)
        location = my_dataset[name][value]
        if location == 'C':
            my_dataset[name][value] = 1
        elif location == 'Q':
            my_dataset[name][value] = 2
        else:
            my_dataset[name][value] = 3

    return my_dataset


def data_split(train_data, test_data):
    target_train = []
    features_train = []
    features_test = []

    for item in train_data:
        print item
        target_train.append(item[0])
        features_train.append(item[1:])

    for item in test_data:
        features_test.append(item[0:])

    return target_train, features_train, features_test


def bar_graph(file, logreg):
    data = pd.read_csv(file, index_col=0)
    numerical_features = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    feature_names = numerical_features.columns
    x = np.arange(len(feature_names))
    plt.bar(x, logreg.feature_importances_.ravel())
    plt.xticks(x + 0.5, feature_names, rotation=30)
    plt.show()

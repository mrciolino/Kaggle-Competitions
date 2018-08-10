# Tools for data handling for the home credit kaggle competetions

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def data_import(file):

    print "Importing Data from", file[-50:]
    df = pd.read_csv(file)
    print df.index
    return df


def data_format(data, size):

    # Handle error
    pd.options.mode.chained_assignment = None
    # Reshape data to given size
    data = data[:size]
    # Fill in NaNs with zeroes
    data.fillna(0, inplace=True)
    print "Resized Data to size", size

    return data


def data_split(df):

    try:
        if any(df['TARGET']):
            labels = df['TARGET']
            features = df.drop(['TARGET', 'SK_ID_CURR'], 1)
            print "Train Data Split into Target and", len(list(features)), "Features"
            return features, labels
    except:
        features = df.drop(['SK_ID_CURR'], 1)
        return features, 0


def categorical_data_to_numerical(features):

    features = pd.get_dummies(features)
    print "Encoding Complete"
    return features


def add_missing_colums(features_train_imported, features_test_final):

    print "Train", len(list(features_train_imported)), "Test", len(list(features_test_final))

    if len(list(features_train_imported)) >= len(list(features_test_final)):
        # Adding missing columns back to the test/train dataset
        missing_cols = set(features_train_imported.columns) - set(features_test_final.columns)
        # Add a missing column in test set with default value equal to 0
        for number in missing_cols:
            features_test_final[number] = 0
        # Ensure the order of column in the test set is in the same order than in train set
        features_test_final = features_test_final[features_train_imported.columns]
    else:
        missing_cols = set(features_test_final.columns) - set(features_train_imported.columns)
        for number in missing_cols:
            features_train_imported[number] = 0
        features_test_final = features_test_final[features_train_imported.columns]

    print "Train", len(list(features_train_imported)), "Test", len(list(features_test_final))

    return features_train_imported, features_test_final


def convert_dataframe_to_list(features, labels):

    try:
        features, labels = features.values.tolist(), labels.values.tolist()
    except:
        features, labels = features.values.tolist(), 0

    print "Data converted to list of lists"
    return features, labels


def handle_outliers(features_train, labels_train):

    print "Classifcation data contains no outliers"

    return features_train, labels_train


def normalize_data():
    pass


def pickle_data(features_train, features_test, labels_train, labels_test, features_test_final, pickle_file):
    with open(pickle_file, 'wb') as f:
        pickle.dump([features_train, features_test, labels_train, labels_test, features_test_final], f)


def data_creation(train_file, train_size, test_file, pickle, data_pickle_file):

    print "********** Data Creation **********"

    print "==Importing Data=="
    data_train = data_import(train_file)
    data_test = data_import(test_file)
    print " "

    print "==Formatting Data=="
    data_train = data_format(data_train, train_size)
    data_test = data_format(data_test, len(data_test))
    print " "

    print "==Splitting Data=="
    features_train, labels_train = data_split(data_train)
    features_test, labels_test = data_split(data_test)
    print " "

    print "==One Hot Encoding Caterogircal Data=="
    features_train = categorical_data_to_numerical(features_train)
    features_test = categorical_data_to_numerical(features_test)
    print " "

    print "==Adding Missing Columns to Train/Test Data=="
    features_train, features_test = add_missing_colums(features_train, features_test)
    print " "

    print "==Converting Data to List of Lists=="
    features_train, labels_train = convert_dataframe_to_list(features_train, labels_train)
    features_test_final, _ = convert_dataframe_to_list(features_test, labels_test)
    print " "

    print "==Handling Outliers=="
    features_train, labels_train = handle_outliers(features_train, labels_train)
    print " "

    print "==Train Data Splitting=="
    features_train, features_test, labels_train, labels_test = train_test_split(
        features_train, labels_train, test_size=.2, random_state=42)
    print "Split train data with test size of", .2
    print " "

    print "==Normalizing Data=="
    normalize_data()
    print " "

    if pickle == True:
        print "==Pickling Data=="
        print " "
        pickle_data(features_train, labels_train, features_test,
                    labels_test, features_test_final, data_pickle_file)
    else:
        print "==Returning Data=="
        print " "
        return features_train, labels_train, features_test, labels_test, features_test_final


def data_creation_method(create_new_data, pickle_new_data, train_data_size):

    train_data_file = '/Users/mrciolino/Documents/Documents/PythonLearning/Kaggle Competetions/Home Credit/Home_Credit_Data/application_train.csv'
    test_data_file = '/Users/mrciolino/Documents/Documents/PythonLearning/Kaggle Competetions/Home Credit/Home_Credit_Data/application_test.csv'
    data_pickle_file = '/Users/mrciolino/Documents/Documents/PythonLearning/Kaggle Competetions/Home Credit/pickled_data.pickle'
                    
    # Create Pickled data or load pickled data
    if os.path.isfile(data_pickle_file) == False and create_new_data == False:
        print "You do not have a pickle_data file, set create_new_data = True to created dataset"
        sys.exit()

    if pickle_new_data == False and create_new_data == True:
        features_train, labels_train, features_test, labels_test, features_test_final = data_creation(
            train_data_file, train_data_size, test_data_file, pickle_new_data, data_pickle_file)

    if pickle_new_data == True and create_new_data == False:
        print "You cannot pickle new data if you dont create new data. Set pickle_new_data to false to load previous dataset"
        sys.exit()

    if pickle_new_data == False and create_new_data == False:
        data = pickle.load(open(data_pickle_file, 'rb'))

    if pickle_new_data == True and create_new_data == True:
        data_creation(train_data_file, train_data_size, test_data_file, pickle_new_data, data_pickle_file)
        data = pickle.load(open(data_pickle_file, 'rb'))

    # Convert pickle data back into usable lists
    try:
        features_train, features_test, labels_train, labels_test, features_test_final =\
            data[0], data[1], data[2], data[3], data[4]
        print "Converting pickled file into our dataset"
    except:
        pass

    return features_train, features_test, labels_train, labels_test, features_test_final


def save_classifier(clf):
    clf_pickle_file = '/Users/mrciolino/Documents/Documents/PythonLearning/Kaggle Competetions/Home Credit/pickled_clf.pickle'
    with open(clf_pickle_file, 'wb') as f:
        pickle.dump(clf, f)


def create_submission(pred):
    file = '/Users/mrciolino/Documents/Documents/PythonLearning/Kaggle Competetions/Home Credit/Home_Credit_Data/application_test.csv'
    df = pd.read_csv(file)
    pred = abs(pred)

    i = 0
    id = []
    target = []

    for label in df['SK_ID_CURR']:
        # line = ("%s,%.2f" % (label, pred[i]))
        id.append(label)
        target.append(pred[i])
        i += 1

    submission = pd.DataFrame({'SK_ID_CURR': id, 'TARGET': target})
    submission.to_csv('/Users/mrciolino/Documents/Documents/PythonLearning/Kaggle Competetions/Home Credit/submission.csv',
                      index=False)

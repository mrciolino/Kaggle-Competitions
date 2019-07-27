"""
Matthew Ciolino - Time Series Analysis
Wikipedia Web Traffic
"""
from os.path import isfile
import pandas as pd
import sys
import os


def data_collection(load_file):

    print("Collecting Data")
    if not os.path.isfile(load_file):
        print("No unzipped training data found \n")
        print("use 'src/data/download_data.py' to download the kaggle dataset")
        sys.exit(0)
    else:
        df = pd.read_csv(load_file)
        print("Dataset 1 Shape:", df.shape)

    return df


def dataframe_transform(df, load_file):

    print("Transforming Data")
    if 'train_1' in load_file:
        save_file = "data/page_data/page_logs_1.csv"
    else:
        save_file = "data/page_data/page_logs_2.csv"

    # save page column to new file to use later if needed
    if not os.path.isfile(save_file):
        data = df.Page
        data.to_csv(save_file, header=True)

    # convert to datetime object
    df = df.drop(columns='Page')
    df = df.transpose()

    # fill in missing values with a rolling mean and if not then fill with 0
    df = df.fillna(df.rolling(7, min_periods=1).mean())
    df = df.fillna(0)

    return df


def save_processed_data(df, save_file):

    print("Saving Processed Data")
    df.to_csv(save_file, index=True)


def check_and_process(save_file, load_file, override):
    if not os.path.isfile(save_file) or override:
        print("Building the features for %s" % load_file)
        # get the data
        df = data_collection(load_file)
        # transpose data
        df = dataframe_transform(df, load_file)
        # save the processed data
        save_processed_data(df, save_file)
    else:
        print("You already have built the features for %s" % load_file)


if __name__ == "__main__":

    override = False
    files = {"data/train_data/train_1.csv": "data/processed_data/train_1_processed.csv",
             "data/train_data/train_2.csv": "data/processed_data/train_2_processed.csv"}

    for load_file, save_file in files.items():
        check_and_process(save_file, load_file, override)

    if [isfile(x) for x in files.values()]:
        print("\n ---------Your all set to train the forcaster---------")

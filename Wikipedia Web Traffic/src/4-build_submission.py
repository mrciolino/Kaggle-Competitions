"""
Matthew Ciolino - Time Series Analysis
Wikipedia Web Traffic
"""
import pandas as pd
import numpy as np
import datetime
import os


def build_submission(files):

    page_file = files[0]
    prediction_file = files[1]
    key_file = files[2]
    submission_file = files[3]
    dates_start = files[4]
    dates_end = files[5]

    # load the page log
    page_df = pd.read_csv(page_file, index_col=0)

    # grab datetime
    start_date = datetime.date(*dates_start)
    end_date = datetime.date(*dates_end)
    dates = [start_date + datetime.timedelta(n) for n in range(int((end_date - start_date).days))]

    # load the predictions
    df = pd.read_csv(prediction_file, header=None)
    df.drop(df.columns[len(df.columns) - 1], axis=1, inplace=True)

    # fill any nans
    df = df.fillna(df.rolling(7, min_periods=1).mean())
    df = df.fillna(0)

    # round to int and replace negative numbers
    df = df.round(0).astype(int)
    df[df < 0] = 0

    # make the dates the column names and the pages the index names
    df.columns = dates
    df.index = page_df.Page

    # convert the dataframe into one column submission
    df = df.assign(counter=df.index).melt('counter')
    df['name'] = df['counter'] + "_" + df['variable'].astype(str)
    df = df[['name', 'value']]

    # grab the keys
    keys_df = pd.read_csv(key_file)
    page_to_id = dict(zip(keys_df.Page, keys_df.Id))

    # convert name column to hash with dict
    df.name = df.name.map(page_to_id)
    df.columns = ['Id', 'Visits']

    # save the submission file
    df.to_csv(submission_file, index=False)


if __name__ == "__main__":

    files_1 = ["data/page_data/page_logs_1.csv",
               "data/prediction_data/train_1_predictions.txt",
               "data/keys_data/key_1.csv",
               "data/submission_data/submission_1.csv",
               (2017, 1, 1),
               (2017, 3, 2)]

    files_2 = ["data/page_data/page_logs_2.csv",
               "data/prediction_data/train_2_predictions.txt",
               "data/keys_data/key_2.csv",
               "data/submission_data/submission_2.csv",
               (2017, 9, 13),
               (2017, 11, 14)]

    if not os.path.isfile("data/submission_data/submission_1.csv"):
        print("Building Submission 1")
        build_submission(files_1)
    else:
        print("Submission 1 already built")

    if not os.path.isfile("data/submission_data/submission_2.csv"):
        print("Building Submission 2")
        build_submission(files_2)
    else:
        print("Submission 2 already built")

    print("\n ---------Your all set to upload to kaggle---------")

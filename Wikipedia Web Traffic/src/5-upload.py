"""
Matthew Ciolino - Time Series Analysis
Wikipedia Web Traffic
"""
import os
import sys
import pandas as pd


def upload_data(submission_data_file):
    # download the zipped dataset
    if os.path.isfile(submission_data_file):
        print("Uploading the submission files")
        cmd = "kaggle competitions submit -c web-traffic-time-series-forecasting -f " + submission_data_file + " -m 'optimized rnn'"
        os.system(cmd)
        print("File submitted")
    else:
        print("Could not upload the files")


if __name__ == "__main__":

    files = {"file_1": "data/submission_data/submission_2.csv"}

    for submission_data in files.values():
        upload_data(submission_data)

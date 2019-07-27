"""
Matthew Ciolino - Time Series Analysis
Wikipedia Web Traffic
"""

"""
The dataset for the wikipeida time series analysis is in total 2.47 GB unzipped
The following script will download the data to Project Path->data/ via Kaggle's API

There are some prerequiists however:
1) pip install kaggle
	- run just "kaggle" in the command line to get the file location you should put kaggle.json in
2) go to the "account" section of kaggle and press "Create New API Token" to get your kaggle.json file
	- place that file into the location suggested in step 1
"""
import os
import sys
import subprocess
from os.path import isfile

def create_file_structure():

    def make_dir(dirName):
        try:
            os.mkdir(dirName)
            print("Directory %s Created" % dirName)
        except:
            print("Directory %s Already Exists" % dirName)

    folders = {'data': 'data',
               'keys': 'data/keys_data',
               'page': 'data/page_data',
               'prediction': 'data/prediction_data',
               'processed': 'data/processed_data',
               'submission': 'data/submission_data',
               'train': 'data/train_data'}

    for folder in folders.values():
        make_dir(folder)


def download_data():
    # download the zipped dataset
    if not os.path.isdir("data/zipped"):
        print("Downloading the dataset")
        cmd = "kaggle competitions download -c web-traffic-time-series-forecasting --p data/zipped"
        os.system(cmd)
    else:
        print("You already have the zipped dataset")


def unzip_data(zip_file, save_file):

	# find where to put the files
	if 'key' in save_file:
		store_dir = "data/keys_data"
	else:
		store_dir = "data/train_data"

	# unzip the train data
	if not os.path.isfile(save_file) and os.path.isdir("data/zipped"):
		print("Unzipping data")
		if sys.platform == "darwin":
			cmd = "unzip " + zip_file + " -d " + store_dir  # macOS
			os.system(cmd)
		elif sys.platform == "win32":
			p = subprocess.Popen(["powershell.exe",
			                      "Expand-Archive " + zip_file + ' ' + store_dir],
			                     stdout=sys.stdout)
			p.communicate()
		else:
			print("Could find the operating system. Linux not supported")
			sys.exit(1)
	else:
	    print("You already have the unzipped the training data for %s" % save_file)


if __name__ == "__main__":

    create_file_structure()
    download_data()

    files = {"data/zipped/train_1.csv.zip": "data/train_data/train_1.csv",
             "data/zipped/train_2.csv.zip": "data/train_data/train_2.csv",
             "data/zipped/key_1.csv.zip": "data/keys_data/key_1.csv",
             "data/zipped/key_2.csv.zip": "data/keys_data/key_2.csv"}

    for zip_file, save_file in files.items():
        unzip_data(zip_file, save_file)

    if [isfile(x) for x in files.values()]:
        print("\n ---------Your all set to build features---------")

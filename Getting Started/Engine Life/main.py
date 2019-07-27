import numpy as np
import pandas as pd
import featuretools as ft
import utils

utils.download_data()

data_path = 'data/train_FD004.txt'
data = utils.load_data(data_path)
data.head()

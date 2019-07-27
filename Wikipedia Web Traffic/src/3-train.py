"""
Matthew Ciolino - Time Series Analysis
Wikipedia Web Traffic
"""
from keras import backend, optimizers
from keras.models import Sequential, Model
from keras.layers import Input, Dense, LSTM, Bidirectional
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# make sure keras sees our gpu
backend.tensorflow_backend._get_available_gpus()


def rnn_model(shape):
    model = Sequential()
    model.add(Bidirectional(LSTM(units=35, activation='relu', input_shape=shape, return_sequences=True)))
    model.add(Bidirectional(LSTM(units=28, activation='relu', return_sequences=True)))
    model.add(Bidirectional(LSTM(units=21, activation='relu', return_sequences=False)))

    model.add(Dense(units=60, activation='relu'))
    model.add(Dense(units=30, activation='relu'))
    model.add(Dense(units=1))

    model.compile(loss='mean_absolute_error', optimizer=optimizers.Adam(lr=.001))

    return model


def fit_predict_model(data, model):
    x = data[0]
    y = data[1]
    x_val = data[2]
    model.fit(x, y, batch_size=20, epochs=5, verbose=0)
    preds = model.predict(x_val)
    return preds


def train_predict_1(df):

    def single_row_data(row):
        data = df.iloc[:, row].values
        x, y, x_test = data[0:-120], data[60:-60], data[-60:]
        x = x.reshape((x.shape[0], 1, 1))
        x_test = x_test.reshape((x_test.shape[0], 1, 1))
        return x, y, x_test

    # compile a model
    model = rnn_model((460, 1, 1))

    # open txt file for writing
    f = open("data/prediction_data/train_1_predictions.txt", 'w+')

    # write to file with predictions
    for row in range(len(df.columns)):
        preds = fit_predict_model(single_row_data(row), model)
        np.savetxt(f, preds, newline=',')
        f.write("\n")
        if row % 100 == 0:
            print("We are on row %d" % row)

    f.close()


def train_predict_2(df):

    def single_row_data(row):
        data = df.iloc[:, row].values
        x, y, x_test = data[0:-124], data[62:-62], data[-62:]
        x = x.reshape((x.shape[0], 1, 1))
        x_test = x_test.reshape((x_test.shape[0], 1, 1))
        return x, y, x_test

    # compile a model
    model = rnn_model((742, 1, 1))

    # open txt file for writing
    f = open("data/prediction_data/train_2_predictions.txt", 'w+')

    # write to file with predictions
    for row in range(len(df.columns)):
        preds = fit_predict_model(single_row_data(row), model)
        np.savetxt(f, preds, newline=',')
        f.write("\n")
        if row % 100 == 0:
            print("We are on row %d" % row)

    f.close()


if __name__ == "__main__":

    if not os.path.isfile("data/prediction_data/train_1_predictions.txt"):
        # load in processed data
        load_file = "data/processed_data/train_1_processed.csv"
        df = pd.read_csv(load_file, index_col=0)
        # predict
        train_predict_1(df)
    else:
        print("You have already made predictions on train_1 data")

    if not os.path.isfile("data/prediction_data/train_2_predictions.txt"):
        # load in processed data
        load_file = "data/processed_data/train_2_processed.csv"
        df = pd.read_csv(load_file, index_col=0)
        # predict
        train_predict_2(df)
    else:
        print("You have already made predictions on train_2 data")

    print("\n ---------Your all set to build the submit files---------")

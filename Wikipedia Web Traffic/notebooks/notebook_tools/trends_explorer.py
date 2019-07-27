"""
Matthew Ciolino - Time Series Analysis
Wikipedia Web Traffic
"""

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.tsa.seasonal import seasonal_decompose


class visaulize_trends():

    def __init__(self, df, row=False):

        data = self.data_get(df, row)
        self.data = self.feature_create(data)
        self.data_length = len(self.data)

    def data_get(self, df, row):

        if row == 'Total':
            pd.options.mode.chained_assignment = None  # default='warn'
            df.loc['Total'] = df.sum()

        return df

    def feature_create(self, df):

        # convert to datetime object
        df = df.drop(columns='Page')
        df = df.transpose()
        df.index = pd.to_datetime(df.index)

        # make weekly and monthly column so we can groupby and plot
        df['month'] = df.index.month
        df['day of week'] = df.index.dayofweek
        df['week'] = df.index.week

        return df

    def trend_visual(self, row, plot_group):

        # grab values
        y = self.data[row]
        self.show_patterns(y, plot=True)

        if plot_group:
            # grab dataframe
            df = self.data

            # plot month, day, and time of week
            fig = plt.figure(figsize=(14, 16))
            fig.set_facecolor('white')
            plt.style.use('seaborn-dark')

            # plot the unit time counts
            day_of_week = df.groupby('day of week')[row].mean()
            week = df.groupby('week')[row].mean()
            monthly = df.groupby('month')[row].mean()

            # plot
            plt.subplot(611)
            plt.plot(day_of_week, label="Day of Week Group")
            plt.legend(loc='upper right')
            plt.subplot(612)
            plt.plot(week, label="Week Group")
            plt.legend(loc='upper right')
            plt.subplot(613)
            plt.plot(monthly, label="Month Group")
            plt.legend(loc='upper right')

            # Converting to daily, weekly, monthly mean
            daily = df.resample('D')[row].mean()
            weekly = df.resample('W')[row].mean()
            monthly = df.resample('M')[row].mean()

            # plot
            plt.subplot(614)
            plt.plot(daily, label='Daily Means')
            plt.legend(loc='upper left')
            plt.subplot(615)
            plt.plot(weekly, label='Weekly Means')
            plt.legend(loc='upper left')
            plt.subplot(616)
            plt.plot(monthly, label='Monthly Means')
            plt.legend(loc='upper left')

    def show_patterns(self, values, plot, freq=30):

        decomposition = seasonal_decompose(list(values), freq=freq)

        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid

        if plot:
            fig = plt.figure(figsize=(12, 8))
            plt.subplot(411)
            plt.plot(values, label='Original')
            plt.legend(loc='best')
            plt.subplot(412)
            plt.plot(trend, label='Trend')
            plt.legend(loc='best')
            plt.subplot(413)
            plt.plot(seasonal, label='Seasonality')
            plt.legend(loc='best')
            plt.subplot(414)
            plt.plot(residual, label='Residuals')
            plt.legend(loc='best')
            plt.tight_layout()
            plt.show()

        return residual

    def stationality_check(self, x):
        # removing nan's
        x = x[~np.isnan(x)]
        # adfuller test
        result = adfuller(x)
        print('ADF Statistic: %f' % result[0])
        print('p-value: %E' % result[1])
        print('Critical Values:')
        for key, value in result[4].items():
            print('\t%s: %.3f' % (key, value))

    def correlation_plots(self, row, show_num, decompose, station_check):

        # grab values
        y = self.data[row]

        if decompose:
            y = self.show_patterns(y, plot=False)

        if station_check:
            self.stationality_check(y)

        # make sure no nans
        y = y[~np.isnan(y)]

        # MA
        # calling auto correlation function
        lag_acf = acf(y, nlags=75)
        cutoff = 1.96 / np.sqrt(len(y))
        acf_element = [i for i, val in enumerate(lag_acf) if val < cutoff][0]
        # Plot PACF:
        plt.figure(figsize=(12, 4))
        plt.plot(lag_acf, marker='+')
        plt.axhline(y=0, linestyle='--', color='gray')
        plt.axhline(y=-cutoff, linestyle='--', color='gray')
        plt.axhline(y=cutoff, linestyle='--', color='gray')
        plt.title('Autocorrelation Function')
        plt.xlabel('number of lags')
        plt.ylabel('correlation')
        plt.tight_layout()

        # AR
        # calling partial correlation function
        lag_pacf = pacf(y, nlags=30, method='ols')
        cutoff = 1.96 / np.sqrt(len(y))
        pacf_element = [i for i, val in enumerate(lag_pacf) if val < cutoff][0]
        # Plot PACF:
        plt.figure(figsize=(12, 4))
        plt.plot(lag_pacf, marker='+')
        plt.axhline(y=0, linestyle='--', color='gray')
        plt.axhline(y=-cutoff, linestyle='--', color='gray')
        plt.axhline(y=cutoff, linestyle='--', color='gray')
        plt.title('Partial Autocorrelation Function')
        plt.xlabel('Number of lags')
        plt.ylabel('correlation')
        plt.tight_layout()

        if show_num:
            print("ACF (MA) Element: %d, PACF (AR) Element: %d" % (acf_element, pacf_element))

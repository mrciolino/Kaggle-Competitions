"""
Matthew Ciolino - Time Series Analysis
Wikipedia Web Traffic
"""

import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class advanced_models():

    def __init__(self, df, row, test_size):

        self.dataframe = df
        self.test_size = test_size
        self.data = self.data_get(df, row)
        self.train, self.test = self.data[0:-(test_size)], self.data[-(test_size):]

    def data_get(self, df, row):
        if row == 'Total':
            pd.options.mode.chained_assignment = None  # default='warn'
            df.loc['Total'] = df.sum()

        data = df.loc[[row]].values[0][1:]

        return list(data)

    def root_mean_squared_error(self, predicted):
        sum_error = 0.0
        for i in range(len(self.test)):
            prediction_error = predicted[i] - self.test[i]
            sum_error += (prediction_error ** 2)
        mean_error = sum_error / float(len(self.test))
        return mean_error**.5

    def forcast_visualize(self, model, zoom, residuals):
        # visual for statsmodel's Autoregression methods
        pred = model.predict(len(self.train), len(self.train) + len(self.test) - 1)
        fig = plt.figure(figsize=(50, 9))

        if not zoom:
            plt.plot(range(-len(self.train), 0), self.train, label='Train')
            plt.plot(range(-len(self.train), 0), model.fittedvalues, color='green', label='Fitted: %f' % (self.root_mean_squared_error(model.fittedvalues)))
            plt.plot(self.test, label='True Test')
            plt.plot(pred, color='red', label='Preds: %f' % (self.root_mean_squared_error(pred)))
        else:
            plt.plot(self.test, label='True Test')
            plt.plot(pred, color='red', label='Preds: %f' % (self.root_mean_squared_error(pred)))

        plt.legend(loc='best')
        plt.show()

        if residuals:
            # residual error / autocorrelation
            residuals = pd.DataFrame(model.resid)
            pd.plotting.autocorrelation_plot(residuals)
            residuals.plot(kind='kde')

    def holts_method(self, seasonal_periods, trend, seasonal, zoom, residuals):
        """
        Holts-Winters method using Exponential Smoothing
        """
        model = ExponentialSmoothing(self.train, seasonal_periods=seasonal_periods, trend=trend, seasonal=seasonal).fit()
        self.forcast_visualize(model, zoom=zoom, residuals=residuals)

    def ARMA(self, order, zoom, residuals):
        """
        Autoregression Moving Average
        """
        model = ARMA(self.train, order).fit(disp=-1)
        self.forcast_visualize(model, zoom=zoom, residuals=residuals)

    def SARIMAX(self, order, seasonal_order, zoom, residuals):
        """
        Seasonal Autoregression Integrated Moving Average (eXogenous)
        """
        model = SARIMAX(self.train, order=order, seasonal_order=seasonal_order).fit(disp=-1)
        self.forcast_visualize(model, zoom=zoom, residuals=residuals)

    def prohpet(self, row, component, zoom):
        """
        Facebook's Prophet
        """
        # prohpet
        dfp = self.dataframe.drop(columns='Page')
        dfp = dfp.transpose()
        dfp.index = pd.to_datetime(dfp.index)
        dfp['ds'] = dfp.index
        dfp['y'] = dfp[row]
        prophet_df = dfp[['ds', 'y']].iloc[:-(self.test_size)]

        m = Prophet()
        m.fit(prophet_df)
        future = m.make_future_dataframe(periods=self.test_size, freq='D')
        forecast = m.predict(future)
        fig = m.plot(forecast)

        if component:
            fig = m.plot_components(forecast)

        if zoom:
            fig = plt.figure(figsize=(12, 8))
            preds = forecast['yhat'].values[-(self.test_size):]
            plt.plot(forecast['yhat'].values[-(self.test_size):], label='Preds: %f' % (self.root_mean_squared_error(preds)))
            plt.plot(dfp['y'].iloc[-(self.test_size):].values, label='True Test')
            plt.legend(loc='best')

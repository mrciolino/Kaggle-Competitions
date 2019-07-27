"""
Matthew Ciolino - Time Series Analysis
Wikipedia Web Traffic
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd


class baseline_time_series_forcast_visual():

    def __init__(self, df, row):

        self.data = self.data_get(df, row)
        self.data_length = len(self.data)
        self.data_range = range(self.data_length)
        self.std_hypers = {'persistance': [1, False],
                           'rolling': [5, False],
                           'simple_exp': [.2, False],
                           'holt_exp': [.2, .1, False],
                           'holt_winter_exp': [.2, .2, .2, 4, False]}

    def data_get(self, df, row):

        if row == 'Total':
            pd.options.mode.chained_assignment = None  # default='warn'
            df.loc['Total'] = df.sum()

        data = df.loc[[row]].values[0][1:]  # used b/c first value in my df is a string

        return data

    def run_show(self, search_best, plot=True):

        if search_best:
            hypers, _ = self.search_best()
            for hyper in hypers:
                hypers[hyper][-1:] = False
            preds, errors = self.run_models(hypers)
        else:
            preds, errors = self.run_models(self.std_hypers)

        if plot:
            self.plot(preds, errors)

    def run_models(self, hypers):

        persistance_pred, persistance_error = self.offset_persistance(hypers['persistance'])  # [offset, search]
        rolling_avg_pred, rolling_avg_error = self.rolling_average(hypers['rolling'])  # [window, search]
        simple_exp_pred, simple_exp_error = self.simple_exp_smoothing(hypers['simple_exp'])  # [alpha, search]
        holt_exp_pred, holt_exp_error = self.holts_smoothing(hypers['holt_exp'])  # [alpha,beta, search]
        holt_winter_exp_pred, holt_winter_exp_error = self.holt_winter_additive(hypers['holt_winter_exp'])  # [alpha,beta,gamma,m, search]

        print("Using Hypers: ", [{line: hypers[line]} for line in hypers])

        preds = {'persistance': persistance_pred,
                 'rolling': rolling_avg_pred,
                 'simple_exp': simple_exp_pred,
                 'holt_exp': holt_exp_pred,
                 'holt_winter_exp': holt_winter_exp_pred}

        errors = {'persistance': persistance_error,
                  'rolling': rolling_avg_error,
                  'simple_exp': simple_exp_error,
                  'holt_exp': holt_exp_error,
                  'holt_winter_exp': holt_winter_exp_error}

        return preds, errors

    def plot(self, preds, errors):

        i = 10  # only start plotting after i values for forecasting
        fig = plt.figure(figsize=(100, 9))
        fig.set_facecolor('white')
        plt.style.use('seaborn-dark')
        plt.plot(self.data_range, self.data, 'b', linewidth=5, label='True Data')
        plt.plot(self.data_range[i:], preds['persistance'][i:],     'r--', label='Persistance offset, RMSE: %.2f' % errors['persistance'])
        plt.plot(self.data_range[i:], preds['rolling'][i:],         'g--', label='Rolling Average, RMSE: %.2f' % errors['rolling'])
        plt.plot(self.data_range[i:], preds['simple_exp'][i:],      'y--', label='Simple Smoothing, RMSE: %.2f' % errors['simple_exp'])
        plt.plot(self.data_range[i:], preds['holt_exp'][i:],        'm--', label='Holt Smoothing, RMSE: %.2f' % errors['holt_exp'])
        plt.plot(self.data_range[i:], preds['holt_winter_exp'][i:], 'k--', label='Holt Winter, RMSE: %.2f' % errors['holt_winter_exp'])
        plt.legend(loc=2)
        plt.show()

    def search_best(self):

        x0 = {'alpha': .8,
              'beta': .5,
              'gamma': .05,
              'm': 3,
              'offset': 1,
              'window': 5,
              'search':  True}

        bounds = {'alpha': (0, 1),
                  'beta': (0, 1),
                  'gamma': (0, 1),
                  'm': (1, 20),
                  'offset': (1, 10),
                  'window': (1, 10),
                  'search':  (True, True)}

        offset_x0 = [x0[x] for x in ['offset', 'search']]
        rolling_x0 = [x0[x] for x in ['window', 'search']]
        simple_x0 = [x0[x] for x in ['alpha', 'search']]
        holt_x0 = [x0[x] for x in ['alpha', 'beta', 'search']]
        holts_winter_x0 = [x0[x] for x in ['alpha', 'beta', 'gamma', 'm', 'search']]

        offset_bounds = [bounds[x] for x in ['offset', 'search']]
        rolling_bounds = [bounds[x] for x in ['window', 'search']]
        simple_bounds = [bounds[x] for x in ['alpha', 'search']]
        holt_bounds = [bounds[x] for x in ['alpha', 'beta', 'search']]
        holts_winter_bounds = [bounds[x] for x in ['alpha', 'beta', 'gamma', 'm', 'search']]

        method = "TNC"
        offset_best = minimize(self.offset_persistance,        offset_x0,       method=method, bounds=offset_bounds)
        rolling_best = minimize(self.rolling_average,          rolling_x0,      method=method, bounds=rolling_bounds)
        simple_best = minimize(self.simple_exp_smoothing,      simple_x0,       method=method, bounds=simple_bounds)
        holt_best = minimize(self.holts_smoothing,             holt_x0,         method=method, bounds=holt_bounds)
        holt_winter_best = minimize(self.holt_winter_additive, holts_winter_x0, method=method, bounds=holts_winter_bounds)

        optimized_hypers = {'persistance': offset_best.x,
                            'rolling': rolling_best.x,
                            'simple_exp': simple_best.x,
                            'holt_exp': holt_best.x,
                            'holt_winter_exp': holt_winter_best.x}

        models_rmse = {'persistance': offset_best.fun,
                       'rolling': rolling_best.fun,
                       'simple_exp': simple_best.fun,
                       'holt_exp': holt_best.fun,
                       'holt_winter_exp': holt_winter_best.fun}

        return optimized_hypers, models_rmse

    def root_mean_squared_error(self, predicted):
        sum_error = 0.0
        for i in self.data_range:
            prediction_error = predicted[i] - self.data[i]
            sum_error += (prediction_error ** 2)
        mean_error = sum_error / float(self.data_length)
        return mean_error**.5

    def offset_persistance(self, args):
        """
        y(t+1) = y(t-offset)
        offset
        """
        offset = args[0]
        serach = args[1]

        predictions = []
        offset = int(offset)

        for i in self.data_range:
            try:
                prediction = self.data[i - offset]
            except:
                prediction = self.data[i]
            predictions.append(prediction)
        error = self.root_mean_squared_error(predictions)

        if serach:
            return error
        else:
            return predictions, error

    def rolling_average(self, args):
        """
        y(t+1) = sum(y(t-window):y(t))/len(window)
        window
        """
        window = args[0]
        serach = args[1]

        predictions = []
        window = int(window)

        for i in self.data_range:
            if i > window:
                prediction_set = self.data[i - window:i]
            else:
                prediction_set = self.data[i:i + 1]
            prediction = sum(prediction_set) / len(prediction_set)
            predictions.append(prediction)
        error = self.root_mean_squared_error(predictions)

        if serach:
            return error
        else:
            return predictions, error

    def simple_exp_smoothing(self, args):
        """
        y(t+1)= alpha *y(t) + alpha(1-alpha)*y(t-1) + alpha(1-alpha^2)*y(t-2) ...
        alpha
        """
        alpha = args[0]
        serach = args[1]

        predictions = []

        for i in self.data_range:
            smoothing_set = self.data[0:i]
            prediction = sum([alpha * (1 - alpha) ** i * x for i, x in enumerate(reversed(smoothing_set))])
            predictions.append(prediction)
        error = self.root_mean_squared_error(predictions)

        if serach:
            return error
        else:
            return predictions, error

    def holts_smoothing(self, args):
        """
        y(t+h) = l(t) + h*b(t)
        l(t) = alpha*y(t) + (1-alpha)(l(t-1) + b(t-1))
        b(t) = beta * (l(t) - l(t-1)) + (1-beta)b(t-1)
        alpha, beta, h
        """
        alpha = args[0]
        beta = args[1]
        serach = args[2]

        level = [1]
        trend = [1]
        predictions = [self.data[0]]

        for i in self.data_range[1:]:
            level.append((alpha * self.data[i - 1]) + (1 - alpha) * (level[i - 1] + trend[i - 1]))
            trend.append(beta * (level[i] - level[i - 1]) + (1 - beta) * trend[i - 1])
            predictions.append(level[i] + trend[i])
        error = self.root_mean_squared_error(predictions)

        if serach:
            return error
        else:
            return predictions, error

    def holt_winter_additive(self, args):
        """
        y(t+h) = l(t) + h*b(t) + s(t+h-m(k+1))
        l(t) = alpha*(y(t) - s(t-m)) + (1-alpha)*(l(t-1)+b(t-1))
        b(t) = beta*(l(t)-l(t-1)) + (1-beta)*b(t-1)
        s(t) = gamma*(y(t)-l(t-1)-b(t-1))+(1-gamma)*s(t-m)
        alpha, beta, gamma, h, m
        """
        alpha = args[0]
        beta = args[1]
        gamma = args[2]
        m = args[3]
        serach = args[4]

        m = int(m)
        level = [1] * m
        trend = [1] * m
        season = [1] * m
        predictions = []
        for element in range((m)):
            predictions.append(self.data[element])

        for i in self.data_range[m:]:
            level.append(alpha * (self.data[i - 1] - season[i - m]) + (1 - alpha) * (level[i - 1] + trend[i - 1]))
            trend.append(beta * (level[i] - level[i - 1]) + (1 - beta) * trend[i - 1])
            season.append(gamma * (self.data[i - 1] - level[i - 1] - trend[i - 1] + (1 - gamma) * (season[i - m])))
            predictions.append(level[i] + trend[i] + season[i - m])
        error = self.root_mean_squared_error(predictions)

        if serach:
            return error
        else:
            return predictions, error

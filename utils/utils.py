import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import os
import errno
import sys
import joblib
from keras.models import load_model

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

sns.set_style("whitegrid")


class Utils:

    @staticmethod
    def save_pipeline(pipe, config):
        ticker_name = config['data']['ticker']
        dir_path = config['model']['dir_path']
         
        Utils.mkdir_ifnot_exist(dir_path)

        # LSTM Model
        fitted_model = pipe.regressor_['regressor'].model
        fitted_model.save(dir_path+ticker_name+'_keras_model.h5')
        # Scikit-learn pipeline
        pipe.regressor_['regressor'].model = None
        joblib.dump(pipe, dir_path+ticker_name+'_sklearn_pipeline.pkl')
        pipe.regressor_['regressor'].model = fitted_model


    @staticmethod
    def load_pipeline(config):
        ticker_name = config['data']['ticker']
        dir_path = config['model']['dir_path']

        if not os.path.isdir(dir_path):
            sys.exit("pipeline and model do not exists....")

        # Scikit-learn pipeline
        pipe = joblib.load(dir_path+ticker_name+'_sklearn_pipeline.pkl')
        # LSTM Model
        pipe.regressor_['regressor'].model = load_model(dir_path+ticker_name+'_keras_model.h5')

        return pipe

    
    @staticmethod
    def train_test_split(X, y, test_size):
        """ """
        train_size = 1.0 - test_size
        X_train = X.iloc[0:int(np.ceil(len(X)*train_size)), :]
        y_train = y.iloc[0:int(np.ceil(len(X)*train_size))]
        
        X_test = X.iloc[int(np.ceil(len(X)*train_size)):, :]
        y_test = y.iloc[int(np.ceil(len(X)*train_size)):]

        return X_train, X_test, y_train, y_test


    @staticmethod
    def evaluation_metrics(y_true, y_pred):
        metrics = []
        metrics.append(('mae',mean_absolute_error(y_true, y_pred)))
        metrics.append(('mse', mean_squared_error(y_true, y_pred)))
        metrics.append(('rmse', mean_squared_error(y_true, y_pred, squared=False)))
        metrics.append(('r2', r2_score(y_true, y_pred)))
        
        print('\n','*'*15,'EVALUATION METRICES','*'*15)
        for metric, val in metrics:
            print("{}: {}".format(metric.upper(), round(val,5)))


    @staticmethod
    def mkdir_ifnot_exist(dir_path):
        try:
            os.makedirs(dir_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


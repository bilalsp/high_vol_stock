import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

import tensorflow as tf

from data import Data
from pipeline.lstm_autoencoder import LSTMAutoEncoderPipeline
from pipeline.lstm import LSTMPipeline

from utils import *


def run_lstm(pipeline, data, config, **kwargs):
    """Run Regular LSTM pipeline"""
    y = data['Adj Close']
    X = y.to_frame()

    # train and test set split
    test_size = config['test_size']
    X_train, X_test, y_train, y_test = Utils.train_test_split(X, y, test_size)

    # Model Training
    if config['model']['train']:
        # build and train pipeline
        pipe = pipeline.build()
        pipe.fit(X_train,y_train)
        # plot model training loss
        # Plot.model_loss(pipe.regressor_['regressor'].model.history.history, config)
        
        Plot.model_history(pipe.regressor_['regressor'].model, config)
        
        # save trained pipeline
        Utils.save_pipeline(pipe, config)

    # Model Testing
    pipe = Utils.load_pipeline(config)
    y_pred = pipe.predict(X_test)

    print(y_pred[:10], y_test[:10])

    # plot histogram of prediction error
    Plot.model_error(y_test, y_pred, config)

    # plot true and predicted value
    Plot.model_forecast(y_train, y_test, y_pred, config)

    # Trade profit 
    (
        max_profit,
        success_rate,
        profitable_trade_count,
        unprofitable_trade_count
    ) = Trade.cal_max_profit(y_test, y_pred)
    
    print("Max profit: {}, Success rate: {}, Profitable Count: {}, Unprofitable Count: {}"\
    .format(max_profit, success_rate, profitable_trade_count, unprofitable_trade_count))
    


def run_lstm_autoencoder(pipeline, data, config, **kwargs):
    """Run autoencoder pipeline"""
    y = data['Adj Close']
    X = y.to_frame()

    # plot ticker data
    Plot.ticker_data(y, config)

    # train and test set split
    test_size = config['test_size']
    X_train, X_test, y_train, y_test = Utils.train_test_split(X, y, test_size)

    # Model Training
    if config['model']['train']:
        # build and train pipeline
        pipe = pipeline.build()
        pipe.fit(X_train, y_train)
        # plot model training loss
        Plot.model_loss(pipe.regressor_['regressor'].model.history.history, config)
        # save trained pipeline
        Utils.save_pipeline(pipe, config)  

    pipe = Utils.load_pipeline(config)

    # instance is considered anomaly if reconstruction error is large
    # MAE: Train Set
    Xt_train = pipe.regressor_[:-1].transform(X_train)
    X_pred = pipe.regressor_['regressor'].model.predict(Xt_train)
    mae_train = np.mean(np.abs(X_pred - Xt_train), axis=1)
    Plot.model_mae(mae_train, "mae_train", config)

    threshold = 1.0
    
    # MAE: Test Set
    Xt_test = pipe.regressor_[:-1].transform(X_test)
    X_pred = pipe.regressor_['regressor'].model.predict(Xt_test)
    mae_test = np.mean(np.abs(X_pred - Xt_test), axis=1)
    Plot.model_mae(mae_test, "mae_test", config)

    # Anomalies
    anomaly_df = y_test.to_frame().reset_index()
    anomaly_df['Date'] = pd.to_datetime(anomaly_df["Date"])
    anomaly_df['mae'] = mae_test
    anomaly_df['threshold'] = threshold
    anomaly_df['anomaly'] = anomaly_df['mae'] > anomaly_df['threshold']
    
    #
    Plot.mae_vs_threshold(anomaly_df, config)
    
    #
    Plot.anomaly(anomaly_df, config)

    # Trade profit 
    (
        max_profit,
        success_rate,
        profitable_trade_count,
        unprofitable_trade_count
    ) = Trade.cal_max_profit_based_on_breakouts(anomaly_df)
   
    print("Max profit: {}, Success rate: {}, Profitable Count: {}, Unprofitable Count: {}"\
    .format(max_profit, success_rate, profitable_trade_count, unprofitable_trade_count))
    


def run_buy_hold(data, **kwargs):
    """Run autoencoder pipeline"""

    y = data['Adj Close']
    X = y.to_frame()

    # train and test set split
    test_size = config['test_size']
    X_train, X_test, y_train, y_test = Utils.train_test_split(X, y, test_size)

    max_profits = X_test['Adj Close'].iloc[-1] - X_test['Adj Close'].iloc[0]

    print("Buy and Hold profit for period {} to {} is ${}" \
    .format(X_test.index.to_list()[0], X_test.index.to_list()[-1], max_profits))




if __name__ == "__main__":

    ########################################################################
    # Strategy-1: Stock price prediction using Regular LSTM Network
    ########################################################################

    config = json.load(open('config/lstm.json'))
    n_features = 1
    look_back = config['look_back']
    model_params = config['model']
    pipeline = LSTMPipeline(look_back, n_features, params=model_params)
    
    # RUN pipeline for SPY Ticker data
    stock_data = Data(config['data']).get()
    run_lstm(pipeline, stock_data, config)
    
    # RUN pipeline for GME Ticker data
    config['data']['ticker'] = 'GME'
    stock_data = Data(config['data']).get()
    run_lstm(pipeline, stock_data, config)

    ########################################################################
    # Strategy-2: LSTM autoencoder network for consolidation breakouts
    ########################################################################

    config = json.load(open('config/lstm_autoencoder.json'))
    n_features = 1
    look_back = config['look_back']
    model_params = config['model']
    pipeline = LSTMAutoEncoderPipeline(look_back, n_features, params=model_params)
    
    # RUN pipeline for SPY Ticker data
    stock_data = Data(config['data']).get()
    run_lstm_autoencoder(pipeline, stock_data, config)
    
    # RUN pipeline for GME Ticker data
    config['data']['ticker'] = 'GME'
    stock_data = Data(config['data']).get()
    run_lstm_autoencoder(pipeline, stock_data, config)

    ########################################################################
    # Strategy-3: Simply Buy and Hold
    ########################################################################

    # RUN pipeline for SPY Ticker data
    stock_data = Data(config['data']).get()
    run_buy_hold(stock_data)

    # RUN for GME Ticker data
    config['data']['ticker'] = 'GME'
    stock_data = Data(config['data']).get()
    run_buy_hold(stock_data)


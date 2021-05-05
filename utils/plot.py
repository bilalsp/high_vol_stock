import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils.utils import Utils

sns.set_style("whitegrid")

class Plot:

    """Plot Utility class for ploting graphs """

    @staticmethod
    def model_history(model, config):
        """ Plot model training loss per epochs during the training """
        
        ticker_name = config['data']['ticker']
        dir_path = config['image']['dir_path']
        
        Utils.mkdir_ifnot_exist(dir_path)
      
        history = pd.DataFrame(model.history.history)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax = sns.lineplot(data=history['loss'], label='Training Loss', ax=ax)
        ax.set_xticks(range(0,len(history)+1,2), minor=False)
        ax.set_title('Model History - '+ticker_name, fontsize=15)
        ax.set_xlabel('Epochs')
        ax.figure.savefig(dir_path+ticker_name+'_model_history.jpg')


    @staticmethod
    def model_forecast(y_train, y_test, y_pred, config):
        """Plot observed and predicted timeseries"""
        ticker_name = config['data']['ticker']
        dir_path = config['image']['dir_path']
        
        Utils.mkdir_ifnot_exist(dir_path)

        y = pd.concat([y_train, y_test], axis=0).to_frame().reset_index()
        y['Date'] = pd.to_datetime(y["Date"])
        y['Adj Close Predicted'] = np.concatenate([[None]*len(y_train),y_pred]).astype('float64')

        fig, ax = plt.subplots(figsize=(10, 5))
        ax = sns.lineplot(x='Date', y='Adj Close', data=y, label='Observed', ax=ax)
        ax = sns.lineplot(x='Date', y='Adj Close Predicted', data=y, label='One-step ahead Forecast', ax=ax)
        ax.set_title('Model Forecast - '+ticker_name, fontsize=15)
        ax.set_ylabel('Adj Close Stock Price USD ($)')
        ax.figure.savefig(dir_path+ticker_name+'_model_forecast.jpg')
        

    @staticmethod
    def model_error(y_test, y_pred, config, bins=50, kde=True):
        """Plot histogram of predicted error"""
        ticker_name = config['data']['ticker']
        dir_path = config['image']['dir_path']
        
        Utils.mkdir_ifnot_exist(dir_path)

        error = (y_pred - y_test).abs().to_frame().rename({'Adj Close':'Error'}, axis=1)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax = sns.histplot(data=error, x='Error', bins=bins, kde=kde, label='Error', ax=ax)
        ax.set_title('Model Prediction MAE - '+ticker_name, fontsize=15)
        ax.set_xlabel('Error')
        ax.figure.savefig(dir_path+ticker_name+'_model_error.jpg')

    
    @staticmethod
    def ticker_data(y, config):
        """Plot ticker data"""
        ticker_name = config['data']['ticker']
        dir_path = config['image']['dir_path']

        Utils.mkdir_ifnot_exist(dir_path)

        y = y.to_frame().reset_index()
        y['Date'] = pd.to_datetime(y['Date'])
        fig, ax = plt.subplots(figsize=(10, 5))
        ax = sns.lineplot(x='Date', y='Adj Close', data=y, label='Observed', ax=ax)
        ax.set_title('Ticker Data - '+ticker_name, fontsize=15)
        ax.set_xlabel('Date')
        ax.figure.savefig(dir_path+ticker_name+'_ticker_data.jpg')


    @staticmethod
    def model_loss(history, config):
        """Plot training & validation loss values"""
        ticker_name = config['data']['ticker']
        dir_path = config['image']['dir_path']

        Utils.mkdir_ifnot_exist(dir_path)
      
        history = pd.DataFrame(history)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax = sns.lineplot(data=history['loss'], label='Training Loss', ax=ax)
        ax = sns.lineplot(data=history['val_loss'], label='Validation Loss', ax=ax)
        ax.set_title('Model Loss - '+ticker_name, fontsize=15)
        ax.set_xlabel('Epochs')
        ax.figure.savefig(dir_path+ticker_name+'_model_loss.jpg')


    @staticmethod
    def model_mae(mae, title, config, bins=30, kde=True):
        """Plot histogram for autoencoder MAE """
        ticker_name = config['data']['ticker']
        dir_path = config['image']['dir_path']

        Utils.mkdir_ifnot_exist(dir_path)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax = sns.histplot(data=mae, bins=bins, kde=kde, label='Error', ax=ax)
        ax.set_title('Model Prediction MAE - '+ticker_name, fontsize=15)
        ax.set_xlabel('MAE')
        ax.figure.savefig(dir_path+ticker_name+'_'+title+'.jpg')


    @staticmethod
    def mae_vs_threshold(df, config):
        """Plot mae vs maximum threshold"""
        ticker_name = config['data']['ticker']
        dir_path = config['image']['dir_path']

        Utils.mkdir_ifnot_exist(dir_path)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax = sns.lineplot(x='Date', y='mae', data=df,  label='MAE', ax=ax)
        ax = sns.lineplot(x='Date', y='threshold', data=df,  label='threshold', ax=ax)
        ax.set_title('MAE vs Threshold - '+ticker_name, fontsize=15)
        ax.figure.savefig(dir_path+ticker_name+'_mae_vs_threshold.jpg')


    @staticmethod
    def anomaly(df, config):
        """ """
        ticker_name = config['data']['ticker']
        dir_path = config['image']['dir_path']

        Utils.mkdir_ifnot_exist(dir_path)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax = sns.lineplot(x='Date', y='Adj Close', data=df,  label='Adj Close', ax=ax)
        ax = sns.scatterplot(x='Date', y='Adj Close', data=df[df['anomaly']==True],  label='anomaly', color='red', ax=ax)
        ax.set_title('Consolidation Breakout Detections - '+ticker_name, fontsize=15)
        ax.figure.savefig(dir_path+ticker_name+'_breakout_detection.jpg')


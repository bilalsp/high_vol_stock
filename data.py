import pandas as pd
import yfinance as yf
from datetime import date, datetime

from utils import *


class Data:
    """Collect historical market data from Yahoo! finance """
    
    def __init__(self, config):
        self.config = config

    def get(self, **kwargs):
        """ """
        config = self.config

        ticker = config['ticker']
        dir_path = config['dir_path']
        start_date, end_date = config['start_date'], config['end_date']

        file_path = dir_path+ticker+'.csv'
        Utils.mkdir_ifnot_exist(dir_path)

        if config.get('download','True'):
            yf.download(
                ticker,
                start=start_date, 
                end= end_date
            ).to_csv(file_path)

        return pd.read_csv(file_path, index_col='Date')

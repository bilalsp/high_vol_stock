import numpy as np
import pandas as pd


class Trade:
    """Trade Utility class for profit calculation"""

    @staticmethod
    def cal_max_profit(y_test, y_pred):
        """Calculate max profit based on LSTM predictions."""
        y = y_test.to_frame().rename({'Adj Close':'true'}, axis=1)
        y['predicted'] = y_pred
        
        max_profit = 0
        profitable_trade_count = unprofitable_trade_count = 0
        for i in range(len(y)-1):
            if y['true'].iloc[i+1] > y['true'].iloc[i] and y['predicted'].iloc[i+1] > y['predicted'].iloc[i] :
                # Correct Prediction. Execute trade and collect profits.
                max_profit += y['true'].iloc[i+1] - y['true'].iloc[i]
                profitable_trade_count +=1
            elif y['true'].iloc[i+1] < y['true'].iloc[i] and y['predicted'].iloc[i+1] > y['predicted'].iloc[i]:
                # Incorrect Prediction.  Execute the trade to stop loss.
                max_profit += y['true'].iloc[i+1] - y['true'].iloc[i]  # Loss
                unprofitable_trade_count +=1
            elif y['predicted'].iloc[i+1] < y['predicted'].iloc[i]:
                # LSTM predicts that stock price will drop. We don't enter a trade.
                pass
        
        success_rate = profitable_trade_count / (profitable_trade_count + unprofitable_trade_count) * 100
        
        return max_profit, success_rate, profitable_trade_count, unprofitable_trade_count

    
    @staticmethod
    def cal_max_profit_based_on_breakouts(breakout, **kwargs):
        """Calculate max profit based on LSTM breakouts"""
        max_profit = 0
        profitable_trade_count = unprofitable_trade_count = 0
        for i in range(len(breakout)-1):
            if breakout['Adj Close'].iloc[i+1] > breakout['Adj Close'].iloc[i] \
            and breakout['anomaly'].iloc[i] == True:
                # Correct Prediction. Execute trade and collect profits.
                max_profit += breakout['Adj Close'].iloc[i+1] - breakout['Adj Close'].iloc[i]
                profitable_trade_count +=1
            elif breakout['Adj Close'].iloc[i+1] < breakout['Adj Close'].iloc[i] \
            and breakout['anomaly'].iloc[i] == True:
                # Incorrect Prediction.  Execute the trade to stop loss.
                max_profit += breakout['Adj Close'].iloc[i+1] - breakout['Adj Close'].iloc[i]
                unprofitable_trade_count +=1
            elif breakout['anomaly'].iloc[i] == False:
                # LSTM predicts no breakout. We don't enter a trade.
                pass
        
        success_rate = profitable_trade_count / (profitable_trade_count + unprofitable_trade_count) * 100
        
        return max_profit, success_rate, profitable_trade_count, unprofitable_trade_count

                
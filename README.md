<!-- Add banner here -->
<img src="img/banner.jpg" width="55%" height="40%">

# Stock Market: High Volatility Prediction

<!-- Add buttons here -->
![GitHub last commit](https://img.shields.io/github/last-commit/bilalsp/high_vol_stock)
![GitHub pull requests](https://img.shields.io/github/issues-pr/bilalsp/high_vol_stock)
![GitHub issues](https://img.shields.io/github/issues-raw/bilalsp/high_vol_stock)
![GitHub stars](https://img.shields.io/github/stars/bilalsp/high_vol_stock)


<!-- Describe your project in brief -->
In this project, we implemented different strategies for stock trading using deep learning in the presence of extreme volatility.

# Table of contents
- [Description](#Description)
- [Methodologies](#Methodologies)
- [Data Collection](#Data-Collection)
- [Experimental Results](#Experimental-Result)
- [Conclusion](#Conclusion)

## Description
We analyze the high volatility stock and compare it to less volatility stock. Two deep learning-based approaches have been implemented to make a prediction for stock trading. For strategy-1, we implemented LSTM (Long Short-term Memory) Network with a short sequence in case of high volatility to predict stock price. For strategy-2, we implemented an LSTM autoencoder network to trade only on consolidation breakouts after detecting anomalies in stock price. In 2021, a surge in volatility has been observed for certain stocks such as GameStop Corporation stock (GME ticker). We consider GME high volatility stock data for our case study and SPY ticker as a benchmark for comparison since it is a less volatile stock.

## Methodologies
Three different methods have been implemented for high volatility stock trading.

**Strategy-1: Stock price prediction using Regular LSTM Network**

We design a pipeline of transformers and a regular LSTM network to make stock price predictions. We use 80% of data to train our regular LSTM neural network of two LSTM and two dense layers and tested on 20% of future data. We consider 60 as sequence size or look back value to make a prediction. Considering very small look back result in failing to learn from data and very large look back result in accumulating error.

Trading based on prediction: We bought stock based on the stock price prediction. If the actual stock price on test data is higher than the price we bought stock then we close a trade with profit. On the other hand, if the actual stock price on test data is lower than the price we bought stock then we close a trade with a loss. We report the total number of profitable and unprofitable trades as well as total profit and success rate.

**Strategy-2: LSTM autoencoder network for consolidation breakouts**

We design a pipeline of transformers and an LSTM autoencoder network to detect anomalies in stock price. We take data point from the test set and try to reconstruct using trained autoencoder. If reconstruction error is above a certain threshold, we label that data point as a breakout.

Trading based on consolidation breakouts: We bought stock based on the consolidation breakouts prediction. If the actual stock price on test data is higher than the price we bought stock then we close a trade with profit. On the other hand, if the actual stock price on test data is lower than the price we bought stock then we close a trade with a loss. We report the total number of profitable and unprofitable trades as well as total profit and success rate.

**Strategy-3: Simply Buy and Hold**

Buy and Hold strategy means to buy the stock at the start date of the test set and sell the stock at the market close on the last date of the test set. We report a total profit for that duration.

## Data Collection

Around two decades of historical market data for SPY and GME ticker have been collected from Yahoo! finance using [yfinance](https://pypi.org/project/yfinance/) python library.  

## Experimental Result

#### Model Forcasting and Consolidation breakout detection on test set

| ![Model Forcasting](/img/lstm/SPY_model_forecast.jpg) | ![Model Forcasting](/img/lstm/GME_model_forecast.jpg)|
| :------- | :---------- |
| ![Model Breakout](/img/lstm_autoencoder/SPY_breakout_detection.jpg) | ![Model Breakout](/img/lstm_autoencoder/GME_breakout_detection.jpg) |

#### **SPY - Stock Trading on test set for period 2017-04-28 to 2021-02-12**

| Strategy | Profits ($) | Success Rate | Profitable | Unprofitable |
| :------- | :---------: | :---------:  | :--------: | :----------: |
| # 1      | 90.55       | 55.98%       | 304        | 239          |
| # 2      | 72.80       | 56.37%       | 358        | 277          |
| # 3      | **170.31**  | N/A          | N/A        |  N/A         |    

#### **GME - Stock Trading on test set for period 2017-04-28 to 2021-02-12**

| Strategy | Profits ($) | Success Rate | Profitable | Unprofitable |
| :------- | :---------: | :----------: | :--------: | :----------: |
| # 1      | **65.35**   | 49.67%       | 230        | 233          |
| # 2      | 20.11       | 52.87%       | 92         | 82           |
| # 3      | 33.66       | N/A          | N/A        |  N/A         |

## Conclusion
In this case study of high volatility stock, we explored GME high volatility stock and compared it with SPY less volatile stock. We implemented three different approaches for stock trading. In the presence of high volatility, strategy-1 "Regular LSTM Network" outperforms but in the case of less volatility strategy-3 "Simple Buy and Hold" outperforms. 

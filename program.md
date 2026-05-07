# Program: Financial Autoresearch (SPY Directional Forecasting)

## Objective
Develop an end-to-end automated machine learning pipeline to predict the daily directional movement (Up/Down) of the S&P 500 (SPY). Evaluate the model not just on statistical accuracy, but on simulated financial performance (backtesting) to determine if the strategy generates alpha over a buy-and-hold baseline.

## 1. Data & Hypothesis
* **Asset:** S&P 500 ETF (SPY)
* **Timeframe:** 2015-01-01 to Present.
* **Hypothesis:** Short-term volatility and momentum indicators can provide a slight edge in predicting the next day's directional close.
* **Target:** `Target = 1` if Close_{t+1} > Close_t, else `0`.

## 2. Feature Engineering Log
*Current Features:*
* `Returns`: Daily percentage change.
* `Vol_20`: 20-day rolling standard deviation of returns.
* `SMA_10`: 10-day simple moving average.
* `SMA_50`: 50-day simple moving average.

*Ideas for Next Iteration:*
* [ ] Add MACD (Moving Average Convergence Divergence).
* [ ] Add RSI (Relative Strength Index) to capture overbought/oversold conditions.
* [ ] Fetch VIX (Volatility Index) data as an external macro feature.

## 3. Model Architecture
*Current:* * **Type:** Multi-Layer Perceptron (PyTorch)
* **Structure:** `Linear(Input, 64) -> ReLU -> Linear(64, 32) -> ReLU -> Linear(32, 1) -> Sigmoid`
* **Loss:** Binary Cross Entropy (BCE).
* **Optimizer:** Adam (lr=0.001).

*Ideas for Next Iteration:*
* [ ] Switch to an LSTM or GRU to better capture the sequential, time-series nature of the data.
* [ ] Try an XGBoost baseline; tree-based models often handle tabular financial data better than simple MLPs.

## 4. Experiment & Backtest Log

| Date | Model | Features | Acc (Test) | Strat Return | Market Return | Sharpe | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| YYYY-MM-DD | Base MLP | Ret, Vol20, SMA10/50 | 51.2% | 12.4% | 15.1% | 0.85 | *Underperformed market. Needs better features.* |
| *Pending* | LSTM | + RSI, MACD | - | - | - | - | *To be run in next batch.* |

## 5. Known Issues & Real-World Constraints
* **Trading Costs:** The current backtest assumes 0 transaction fees and 0 slippage. This is unrealistic. 
    * *Fix:* Add a 0.1% friction penalty per trade in `backtest.py`.
* **Lookahead Bias:** Must ensure that calculating rolling windows does not accidentally leak `t+1` data into row `t`.
* **Class Imbalance:** The stock market naturally drifts upward; there are slightly more "Up" days than "Down" days. Might need to adjust class weights in the loss function.
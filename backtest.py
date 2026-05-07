import torch
import pandas as pd
import numpy as np
from train import FinanceModel

def run_backtest():
    test_df = pd.read_csv('data/test.csv', index_col=0)
    features = ['Returns', 'Vol_20', 'SMA_10', 'SMA_50']
    
    # Load Model
    model = FinanceModel(input_dim=len(features))
    model.load_state_dict(torch.load('model_weights.pt'))
    model.eval()
    
    # Generate Predictions
    X_test = torch.tensor(test_df[features].values, dtype=torch.float32)
    with torch.no_grad():
        preds = model(X_test).numpy().flatten()
    
    # Trading Strategy: Buy if model predicts UP (>0.5), stay in cash if DOWN (<0.5)
    test_df['Signal'] = (preds > 0.5).astype(int)
    
    # Calculate Strategy Returns (Signal applied to next day's return)
    test_df['Strategy_Return'] = test_df['Signal'] * test_df['Returns']
    
    # Calculate Cumulative Returns
    test_df['Cum_Market'] = (1 + test_df['Returns']).cumprod()
    test_df['Cum_Strategy'] = (1 + test_df['Strategy_Return']).cumprod()
    
    # Calculate Metrics
    strategy_total_return = test_df['Cum_Strategy'].iloc[-1] - 1
    market_total_return = test_df['Cum_Market'].iloc[-1] - 1
    
    # Annualized Volatility
    strat_vol = test_df['Strategy_Return'].std() * np.sqrt(252) 
    sharpe_ratio = (test_df['Strategy_Return'].mean() * 252) / strat_vol if strat_vol > 0 else 0
    
    print("--- Backtest Results ---")
    print(f"Market Return:   {market_total_return * 100:.2f}%")
    print(f"Strategy Return: {strategy_total_return * 100:.2f}%")
    print(f"Sharpe Ratio:    {sharpe_ratio:.2f}")
    
    test_df.to_csv('data/backtest_results.csv')
    print(f"BACKTEST_METRIC: sharpe={sharpe_ratio:.6f}")

if __name__ == '__main__':
    run_backtest()

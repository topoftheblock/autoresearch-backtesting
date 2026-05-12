import torch
import pandas as pd
import numpy as np
import sys
from train import FinanceModel

def run_backtest(mode="val"):
    # Select the dataset based on the mode to prevent data leakage
    file_path = 'data/val.csv' if mode == 'val' else 'data/test.csv'
    print(f"Running backtest on {file_path}...")
    
    df = pd.read_csv(file_path, index_col=0)
    features = ['Returns', 'Vol_20', 'SMA_10', 'SMA_50']
    
    # Load Model
    model = FinanceModel(input_dim=len(features))
    model.load_state_dict(torch.load('model_weights.pt'))
    model.eval()
    
    # Generate Predictions
    X = torch.tensor(df[features].values, dtype=torch.float32)
    with torch.no_grad():
        raw_outputs = model(X).numpy().flatten()
        
        # SAFETY CHECK: Convert logits to probabilities if the model lacks a Sigmoid layer
        # (Logits can be negative or > 1. Probabilities are strictly between 0 and 1)
        if (raw_outputs < 0).any() or (raw_outputs > 1).any():
            preds = 1 / (1 + np.exp(-raw_outputs))  # Sigmoid formula
        else:
            preds = raw_outputs
    
    # --- DYNAMIC POSITION SIZING ---
    # Strategy: Buy proportionally to our confidence above the 50% threshold.
    # We map the probability range [0.5, 1.0] to a position size [0.0, 1.0]
    # Example: 
    #   Pred = 0.50 -> (0.50 - 0.5) * 2 = 0.0 (0% invested, stay in cash)
    #   Pred = 0.65 -> (0.65 - 0.5) * 2 = 0.3 (30% invested)
    #   Pred = 0.90 -> (0.90 - 0.5) * 2 = 0.8 (80% invested)
    
    df['Signal'] = np.where(preds > 0.5, (preds - 0.5) * 2, 0.0)
    
    # OPTIONAL: If you want to enable Long/Short trading, comment out the line above 
    # and uncomment the line below. This scales from -100% (Short) to +100% (Long).
    # df['Signal'] = (preds - 0.5) * 2 
    
    # Calculate Strategy Returns (Signal applied to next day's return)
    df['Strategy_Return'] = df['Signal'] * df['Returns']
    
    # Calculate Cumulative Returns
    df['Cum_Market'] = (1 + df['Returns']).cumprod()
    df['Cum_Strategy'] = (1 + df['Strategy_Return']).cumprod()
    
    # Calculate Metrics
    strategy_total_return = df['Cum_Strategy'].iloc[-1] - 1
    market_total_return = df['Cum_Market'].iloc[-1] - 1
    
    # Annualized Volatility
    strat_vol = df['Strategy_Return'].std() * np.sqrt(252) 
    sharpe_ratio = (df['Strategy_Return'].mean() * 252) / strat_vol if strat_vol > 0 else 0
    
    print(f"--- {mode.upper()} Backtest Results ---")
    print(f"Market Return:   {market_total_return * 100:.2f}%")
    print(f"Strategy Return: {strategy_total_return * 100:.2f}%")
    print(f"BACKTEST_METRIC: sharpe={sharpe_ratio:.4f}")

if __name__ == '__main__':
    # Default to validation mode unless '--test' is explicitly passed
    mode = "test" if len(sys.argv) > 1 and sys.argv[1] == "--test" else "val"
    run_backtest(mode)
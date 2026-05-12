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
        if (raw_outputs < 0).any() or (raw_outputs > 1).any():
            preds = 1 / (1 + np.exp(-raw_outputs))  # Sigmoid formula
        else:
            preds = raw_outputs
    
    # --- DYNAMIC LONG/SHORT POSITION SIZING ---
    # Strategy:
    # 1. Prediction > 0.6 (High confidence UP): Go Long. 
    #    Formula: (preds - 0.6) * 2.5 maps [0.6 to 1.0] -> [0.0 to 1.0] (0% to 100% Long)
    # 2. Prediction < 0.4 (High confidence DOWN): Go Short. 
    #    Formula: (preds - 0.4) * 2.5 maps [0.4 to 0.0] -> [0.0 to -1.0] (0% to -100% Short)
    # 3. Prediction between 0.4 and 0.6: Neutral zone. Stay in cash (0.0).
    
    conditions = [
        preds > 0.6,
        preds < 0.4
    ]
    choices = [
        (preds - 0.6) * 2.5,  # Scale into Long
        (preds - 0.4) * 2.5   # Scale into Short (results in a negative multiplier)
    ]
    
    df['Signal'] = np.select(conditions, choices, default=0.0)
    
    # Calculate Strategy Returns 
    # (A negative signal multiplied by a negative market return equals a positive strategy return!)
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
import torch
import pandas as pd
import numpy as np
import sys
from train import FinanceModel

def run_backtest(split="test"):
    # Load the requested dataset split (val or test)
    file_path = f'data/{split}.csv'
    try:
        df = pd.read_csv(file_path, index_col=0)
    except FileNotFoundError:
        print(f"Error: Could not find {file_path}")
        sys.exit(1)
        
    features = ['Returns', 'Vol_20', 'SMA_10', 'SMA_50']
    
    # Load Model
    model = FinanceModel(input_dim=len(features))
    model.load_state_dict(torch.load('model_weights.pt'))
    model.eval()
    
    # Generate Predictions
    X = torch.tensor(df[features].values, dtype=torch.float32)
    with torch.no_grad():
        preds = model(X).numpy().flatten()
    
    # Trading Strategy: Buy if model predicts UP (>0.5), stay in cash if DOWN (<0.5)
    df['Signal'] = (preds > 0.5).astype(int)
    
    # Calculate Strategy Returns
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
    
    print(f"--- {split.upper()} Backtest Results ---")
    print(f"Market Return:   {market_total_return * 100:.2f}%")
    print(f"Strategy Return: {strategy_total_return * 100:.2f}%")
    print(f"BACKTEST_METRIC: sharpe={sharpe_ratio:.4f}")

if __name__ == "__main__":
    # Default to test, but allow command-line override
    dataset_split = "test"
    if len(sys.argv) > 1:
        dataset_split = sys.argv[1].lower()
        
    if dataset_split not in ["val", "test", "train"]:
        print("Invalid split provided. Use 'val' or 'test'.")
        sys.exit(1)
        
    run_backtest(dataset_split)
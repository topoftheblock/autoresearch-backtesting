import yfinance as yf
import pandas as pd
import numpy as np
import os

def prepare_data(ticker="SPY", start="2015-01-01", end="2025-01-01"):
    print(f"Downloading data for {ticker}...")
    df = yf.download(ticker, start=start, end=end)
    
    # Flatten MultiIndex columns if yfinance returns them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    # --- Feature Engineering ---
    df['Returns'] = df['Close'].pct_change()
    df['Vol_20'] = df['Returns'].rolling(20).std()
    
    # Ensure Stationarity: Calculate percentage difference from the SMA rather than raw price
    raw_sma_10 = df['Close'].rolling(10).mean()
    raw_sma_50 = df['Close'].rolling(50).mean()
    
    df['SMA_10'] = (df['Close'] - raw_sma_10) / raw_sma_10
    df['SMA_50'] = (df['Close'] - raw_sma_50) / raw_sma_50
    
    # Target: 1 if next day's return is positive, 0 if negative
    df['Target'] = (df['Returns'].shift(-1) > 0).astype(int)
    
    # Drop NaNs created by rolling windows and shifting
    df = df.dropna()
    
    # --- Train/Val/Test Split (Time-series split, NO random shuffling) ---
    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.15)
    
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size+val_size]
    test_df = df.iloc[train_size+val_size:]
    
    # Save to disk
    os.makedirs('data', exist_ok=True)
    train_df.to_csv('data/train.csv')
    val_df.to_csv('data/val.csv')
    test_df.to_csv('data/test.csv')
    
    print("Data preparation complete. Stationary splits saved to /data.")

if __name__ == '__main__':
    prepare_data()
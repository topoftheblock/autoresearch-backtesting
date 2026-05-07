# AutoResearch Backtesting: AI-Driven Quantitative Trading Pipeline

An automated machine learning pipeline that predicts the daily directional movement (Up/Down) of the S&P 500 ETF (SPY). The system uses an autonomous Large Language Model (LLM) agent to iteratively rewrite its own PyTorch neural network architecture, optimizing for improved Sharpe Ratio and Strategy Return in a simulated trading environment.

## Overview

Unlike standard model training, this project closes the loop: train → backtest → analyse → redesign.  
An LLM (GPT-4o) receives backtest performance metrics, proposes a new architecture, and the pipeline automatically integrates it for the next iteration. All data handling respects chronological order to avoid lookahead bias.

## Project Structure

| File | Purpose |
|------|---------|
| `llm_agent.py` | Orchestrator: runs training, backtesting, feeds metrics to the LLM, extracts generated PyTorch code, and overwrites `train.py` for the next research cycle. |
| `prepare.py` | Data ingestion and preprocessing. Downloads SPY data (2015–2025) via `yfinance`, engineers momentum/volatility features, and creates chronological train/validation/test splits. |
| `train.py` | PyTorch model definition and training loop. **This file is dynamically overwritten by the LLM agent during the research loop.** Default architecture: a Multi-Layer Perceptron (MLP) with Binary Cross Entropy loss. |
| `backtest.py` | Evaluates the trained model on unseen test data. Simulates a simple trading strategy (long if predicted probability > 50%, otherwise cash) and computes Strategy Return, cumulative returns, and Sharpe Ratio. |
| `program.md` | Living research log: feature ideas, experiment tracking, known limitations (transaction costs, class imbalance). |

## Data & Features

The model attempts to find an edge using short-term volatility and momentum indicators derived from daily SPY prices.

**Current Engineered Features:**
- `Returns`: Daily percentage change.
- `Vol_20`: 20-day rolling standard deviation of returns.
- `SMA_10`: 10-day simple moving average.
- `SMA_50`: 50-day simple moving average.

**Target:**
- Binary variable: `1` if next day’s close > current close, else `0`.

Data is split chronologically: **train** (earliest), **validation**, and **test** (most recent) to prevent future information leakage.

## How It Works

1. **Data preparation** (`prepare.py`): Downloads SPY data and creates features.
2. **Initial training** (`train.py`): A baseline PyTorch model is trained on the training set, validated on the validation set.
3. **Backtesting** (`backtest.py`): The trained model is tested on the hold‑out test set. The strategy buys SPY when the model predicts an up‑day with >50% confidence; otherwise holds cash. Performance metrics (Strategy Return, Sharpe Ratio, etc.) are recorded.
4. **LLM hypothesis** (`llm_agent.py`): The agent sends the current model code and performance results to OpenAI’s API, asking the LLM to propose a better architecture (e.g., adding LSTM, attention, regularisation).
5. **Code extraction & rewrite**: The agent extracts the Python code from the LLM’s response and overwrites `train.py` with the new `FinanceModel` definition and training loop.
6. **Loop**: Steps 2–5 repeat, with the agent logging each iteration’s performance and the generated architecture.

## Getting Started

### Prerequisites
- Python 3.8+
- Libraries: `torch`, `pandas`, `numpy`, `yfinance`, `openai`

Install them with:
```bash
pip install torch pandas numpy yfinance openai
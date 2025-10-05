# Bidirectional 3-Layer LSTM — 15m Next-10-Candle Forecaster

This project trains **once per day** on the last 60 days of 15-minute candles (from `yfinance`) and then produces rolling predictions every 15 minutes for the **next 10 intervals** (i.e., next 150 minutes).

**Inputs (features):**
- OHLC (Open, High, Low, Close)
- RSI (14)
- EMA20, EMA50
- MACD (12, 26, 9) — line, signal, histogram
- Return (pct change of Close) and **Change in return** (difference of successive returns)

**Target (labels):**
- Vector of the next 10 Close prices (t+1 ... t+10) — multi-step direct regression.

**Lookback:** 200 candles (last 200×15m ≈ 2.08 days of market time)

> Note on "accuracy > 80%": price prediction is a regression task. We report **MAPE**, **MAE**, **RMSE**, and **Directional Accuracy** (percentage of correctly predicted up/down moves). "80% accuracy" is interpreted as directional accuracy; hitting this consistently is market-dependent and not guaranteed.

## Quickstart

```bash
# 1) Install dependencies
pip install -r requirements.txt

# 2) Train once (e.g., daily)
python run_train.py --ticker TCS.NS   # or "AAPL" for US

# 3) Predict the next 10 intervals from the latest data
python run_predict.py --ticker TCS.NS
```

Outputs:
- `artifacts/` — saved Keras model, feature scaler, target scaler, metadata
- `outputs/` — JSON and CSV prediction files with next 10 timestamps and prices

## Project Structure
```
bidir_lstm_predictor/
├─ config.py
├─ data.py
├─ features.py
├─ model.py
├─ trainer.py
├─ predictor.py
├─ pipeline.py
├─ utils.py
├─ run_train.py
├─ run_predict.py
├─ requirements.txt
└─ README.md
```

## Design Highlights
- Uses **Bidirectional LSTM (3 stacked layers)** with residual skip and dropout for regularization.
- Multi-output head (10-dim) predicts all 10 steps at once.
- Robust scaling; targets are also scaled and later inverse-transformed.
- Evaluation reports: MAPE/MAE/RMSE + Directional Accuracy for step+1 and averaged across horizons.
- Safe slicing to avoid look-ahead bias.

## Notes
- `yfinance` 15-minute interval supports last ~60 days of data.
- Market calendars can cause gaps; we forward-fill only technical features (not prices) and drop NA sequences safely.
- For production, consider:
  - Model/feature drift monitoring
  - Walk-forward or expanding-window retraining
  - Better hyperparameter search (Optuna/KerasTuner)
  - Ensembling and confidence intervals
```
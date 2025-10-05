import os
import numpy as np
import pandas as pd
from config import cfg
from utils import ensure_dirs, save_json
from data import download_ohlc, add_time_features, split_train_test_by_ratio
from features import build_feature_frame, make_supervised
from trainer import train_model, evaluate_model
from predictor import load_artifacts, predict_next_10

def prepare_data(ticker: str, interval: str, lookback: int, history_days: int, horizon: int):
    raw = download_ohlc(ticker, interval, history_days)
    feats = build_feature_frame(raw)
    X, y, idx = make_supervised(feats, lookback, horizon)
    return raw, feats, X, y, idx

def train_pipeline(ticker: str, cfg, artifacts_dir: str):
    raw, feats, X, y, idx = prepare_data(ticker, cfg.data.interval, cfg.data.lookback, cfg.data.history_days, cfg.train.horizon)
    split = int(len(X) * (1 - cfg.train.test_ratio))
    X_train, y_train = X[:split], y[:split]
    X_val, y_val     = X[split:], y[split:]

    info = train_model(X_train, y_train, X_val, y_val, cfg.train.horizon, cfg, artifacts_dir)
    # After save, load to evaluate on validation set
    from predictor import load_artifacts
    import tensorflow as tf
    model, feat_scaler, targ_scaler = load_artifacts(artifacts_dir)
    from trainer import evaluate_model
    metrics = evaluate_model(model, X_val, y_val, feat_scaler, targ_scaler)
    return {"train_info": info, "val_metrics": metrics}

def predict_pipeline(ticker: str, cfg, artifacts_dir: str, outputs_dir: str):
    raw, feats, X, y, idx = prepare_data(ticker, cfg.data.interval, cfg.data.lookback, cfg.data.history_days, cfg.train.horizon)
    # Use the **latest** window
    X_last = X[-1]
    last_index = feats.index[-1]
    model, feat_scaler, targ_scaler = load_artifacts(artifacts_dir)
    df_pred = predict_next_10(model, feat_scaler, targ_scaler, X_last, last_index, interval_minutes=15)
    os.makedirs(outputs_dir, exist_ok=True)
    csv_path = os.path.join(outputs_dir, f"{ticker.replace('.','_')}_next10.csv")
    json_path = os.path.join(outputs_dir, f"{ticker.replace('.','_')}_next10.json")
    df_pred.to_csv(csv_path, index=False)
    df_pred.to_json(json_path, orient='records', date_format='iso')
    return {"csv": csv_path, "json": json_path, "preview": df_pred.head().to_dict(orient='records')}
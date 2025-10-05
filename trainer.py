import os
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from typing import Dict
from utils import ensure_dirs, set_seed, fit_feature_scaler, apply_feature_scaler, fit_target_scaler, apply_target_scaler, save_json
from model import build_bidir_lstm, default_callbacks
import tensorflow as tf

def train_model(X_train, y_train, X_val, y_val, horizon, cfg, artifacts_dir) -> Dict:
    set_seed(cfg.train.seed)
    ensure_dirs(artifacts_dir)

    # Fit scalers on train
    feat_scaler = fit_feature_scaler(X_train)
    targ_scaler = fit_target_scaler(y_train)

    X_train_s = apply_feature_scaler(X_train, feat_scaler)
    X_val_s   = apply_feature_scaler(X_val, feat_scaler)
    y_train_s = apply_target_scaler(y_train, targ_scaler)
    y_val_s   = apply_target_scaler(y_val, targ_scaler)

    model = build_bidir_lstm((X_train.shape[1], X_train.shape[2]), horizon=horizon, lr=cfg.train.lr)

    hist = model.fit(
        X_train_s, y_train_s,
        validation_data=(X_val_s, y_val_s),
        epochs=cfg.train.epochs,
        batch_size=cfg.train.batch_size,
        callbacks=default_callbacks(cfg.train.patience),
        verbose=2
    )

    # Save artifacts
    model_path = os.path.join(artifacts_dir, "model.keras")
    model.save(model_path)
    np.save(os.path.join(artifacts_dir, "feat_scaler_mean.npy"), feat_scaler.mean_)
    np.save(os.path.join(artifacts_dir, "feat_scaler_scale.npy"), feat_scaler.scale_)
    np.save(os.path.join(artifacts_dir, "targ_scaler_mean.npy"), targ_scaler.mean_)
    np.save(os.path.join(artifacts_dir, "targ_scaler_scale.npy"), targ_scaler.scale_)

    return {
        "model_path": model_path,
        "history": {k: [float(v) for v in hist.history[k]] for k in hist.history}
    }

def evaluate_model(model, X, y, feat_scaler, targ_scaler) -> Dict:
    Xs = apply_feature_scaler(X, feat_scaler)
    preds_s = model.predict(Xs, verbose=0)

    # Inverse transform back to the original price scale while preserving any
    # extra singleton dimension the network may emit (e.g. ``(batch, horizon,
    # 1)``).
    preds = targ_scaler.inverse_transform(
        preds_s.reshape(-1, preds_s.shape[-1])
    ).reshape(preds_s.shape)

    # Both y and preds may include a trailing singleton dimension depending on
    # how the model was defined (some configurations use ``TimeDistributed``
    # heads).  Collapse it so that scikit-learn receives <=2-D inputs.
    if preds.ndim > 2 and preds.shape[-1] == 1:
        preds = np.squeeze(preds, axis=-1)
    y_eval = y
    if y_eval.ndim > 2 and y_eval.shape[-1] == 1:
        y_eval = np.squeeze(y_eval, axis=-1)

    # sklearn==1.7.x does not support the ``squared`` keyword argument, so compute
    # the RMSE explicitly to stay compatible across versions.
    mae = mean_absolute_error(y_eval, preds)
    rmse = np.sqrt(mean_squared_error(y_eval, preds))
    mape = mean_absolute_percentage_error(y_eval, preds)

    # Directional accuracy on t+1 step
    true_dir = (y_eval[:,1-1] - y_eval[:,0-1]) if y_eval.shape[1] > 1 else (y_eval[:,0] - y_eval[:,0])
    # safer: use previous close vs next close - but in y we only have t+1..t+10; direction for t+1 vs last input close is external.
    # We'll compute direction within y itself: t+2 vs t+1 as proxy; and average over horizons.
    dir_acc_h = []
    for h in range(y_eval.shape[1]-1):
        td = np.sign(y_eval[:,h+1] - y_eval[:,h])
        pd = np.sign(preds[:,h+1] - preds[:,h])
        dir_acc_h.append((td == pd).mean())
    dir_acc = float(np.mean(dir_acc_h)) if dir_acc_h else 0.0

    return {"mae": float(mae), "rmse": float(rmse), "mape": float(mape), "directional_accuracy": float(dir_acc)}
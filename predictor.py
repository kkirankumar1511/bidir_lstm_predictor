import os
import numpy as np
import pandas as pd
import tensorflow as tf

class ScalerWrapper:
    def __init__(self, mean, scale):
        self.mean_ = mean
        self.scale_ = scale
    def transform(self, X):
        return (X - self.mean_) / self.scale_
    def inverse_transform(self, Y):
        return (Y * self.scale_) + self.mean_

def load_artifacts(artifacts_dir: str):
    model = tf.keras.models.load_model(os.path.join(artifacts_dir, "model.keras"))
    f_mean = np.load(os.path.join(artifacts_dir, "feat_scaler_mean.npy"))
    f_scale = np.load(os.path.join(artifacts_dir, "feat_scaler_scale.npy"))
    t_mean = np.load(os.path.join(artifacts_dir, "targ_scaler_mean.npy"))
    t_scale = np.load(os.path.join(artifacts_dir, "targ_scaler_scale.npy"))
    feat_scaler = ScalerWrapper(f_mean, f_scale)
    targ_scaler = ScalerWrapper(t_mean, t_scale)
    return model, feat_scaler, targ_scaler

def predict_next_10(model, feat_scaler, targ_scaler, X_last_window: np.ndarray, last_index: pd.Timestamp, interval_minutes: int = 15):
    Xs = feat_scaler.transform(X_last_window.reshape(-1, X_last_window.shape[-1])).reshape(1, X_last_window.shape[0], X_last_window.shape[1])
    pred_s = model.predict(Xs, verbose=0)[0]  # (10,)
    preds = targ_scaler.inverse_transform(pred_s)
    # Build timestamps for next 10 intervals
    times = [last_index + pd.Timedelta(minutes=interval_minutes * (i+1)) for i in range(len(preds))]
    return pd.DataFrame({"timestamp": times, "pred_close": preds})
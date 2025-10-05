import os, json, random, numpy as np
from typing import Tuple
from sklearn.preprocessing import StandardScaler

def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def set_seed(seed: int = 42):
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def fit_feature_scaler(X: np.ndarray) -> StandardScaler:
    s = StandardScaler()
    s.fit(X.reshape(-1, X.shape[-1]))
    return s

def apply_feature_scaler(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    X2 = scaler.transform(X.reshape(-1, X.shape[-1]))
    return X2.reshape(X.shape)

def fit_target_scaler(y: np.ndarray) -> StandardScaler:
    s = StandardScaler()
    s.fit(y.reshape(-1, y.shape[-1]))
    return s

def apply_target_scaler(y: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    y2 = scaler.transform(y.reshape(-1, y.shape[-1]))
    return y2.reshape(y.shape)
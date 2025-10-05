import numpy as np
import pandas as pd

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    close = df['Close']
    open_ = df['Open']
    high = df['High']
    low = df['Low']
    vol = df.get('Volume', pd.Series(index=df.index, dtype=float)).fillna(0)

    # Technicals
    df['ema20'] = ema(close, 20)
    df['ema50'] = ema(close, 50)
    df['rsi14'] = rsi(close, 14)
    macd_line, macd_signal, macd_hist = macd(close, 12, 26, 9)

    # Returns & change in return
    df['ret'] = close.pct_change()
    df['DRet'] = df['ret'].diff()

    # Assemble features
    #feats = pd.DataFrame({
        #'Open': open_,
        #'High': high,
        #'Low': low,
        #'Close': close,
        #'RSI14': rsi14,
        #'EMA20': ema20,
        #'EMA50': ema50,
        #'MACD': macd_line,
       # 'MACD_signal': macd_signal,
      #  'MACD_hist': macd_hist,
     #   'Ret': ret,
    #    'DRet': dret
   # }, index=df.index)

    feats = df.copy()
    feats.dropna()
    return feats

def make_supervised(feats: pd.DataFrame, lookback: int, horizon: int):
    X_list, y_list, idx_list = [], [], []
    vals = feats.values
    close_idx = feats.columns.get_loc('Close')
    for i in range(lookback, len(feats) - horizon):
        X = vals[i - lookback:i, :]  # (lookback, n_feats)
        y = vals[i:i + horizon, close_idx]  # next horizon closes
        X_list.append(X)
        y_list.append(y)
        idx_list.append(feats.index[i-1])  # last input timestamp
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y, np.array(idx_list)
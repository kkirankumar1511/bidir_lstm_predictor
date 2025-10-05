import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, timezone

def download_ohlc(ticker: str, interval: str = "15m", history_days: int = 60) -> pd.DataFrame:
    # yfinance supports period keyword for intraday
    period = f"{history_days}d"
    df = yf.download(tickers=ticker, interval=interval, period=period, auto_adjust=True, prepost=False)
    #df = df.rename(columns=str.title)  # Ensure 'Open','High','Low','Close','Volume'
    #df = df.dropna(inplace=True)
    #df.index = pd.to_datetime(df.index, utc=True)
    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    # Optional: cyclical time encoding could be added here if desired.
    return df

def split_train_test_by_ratio(df: pd.DataFrame, test_ratio: float = 0.2):
    n = len(df)
    split = int(n * (1 - test_ratio))
    return df.iloc[:split], df.iloc[split:]
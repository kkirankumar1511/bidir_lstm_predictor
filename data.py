import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, timezone

def download_ohlc(ticker: str, interval: str = "15m", history_days: int = 60) -> pd.DataFrame:
    """Download OHLC data for *ticker*.

    The public CI environment used for kata style exercises does not have
    unrestricted internet access.  When the download fails yfinance returns an
    empty frame, which later causes shape errors once we try to scale features.
    It is much more helpful to surface that situation explicitly so callers can
    decide whether to supply cached data or simply skip the run.
    """

    # yfinance supports the period keyword for intraday data.
    period = f"{history_days}d"
    df = yf.download(
        tickers=ticker,
        interval=interval,
        period=period,
        auto_adjust=True,
        prepost=False,
    )

    if df.empty:
        raise RuntimeError(
            f"No OHLC data was retrieved for '{ticker}' with interval '{interval}'. "
            "This is usually caused by the upstream data service being "
            "unreachable in the current environment."
        )

    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    # Optional: cyclical time encoding could be added here if desired.
    return df

def split_train_test_by_ratio(df: pd.DataFrame, test_ratio: float = 0.2):
    n = len(df)
    split = int(n * (1 - test_ratio))
    return df.iloc[:split], df.iloc[split:]
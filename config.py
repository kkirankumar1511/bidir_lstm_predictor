from dataclasses import dataclass


class DataConfig:
    ticker: str = "AAPL"
    interval: str = "15m"
    lookback: int = 200
    history_days: int = 60


class TrainConfig:
    test_ratio: float = 0.2
    batch_size: int = 64
    epochs: int = 20
    lr: float = 1e-3
    patience: int = 5
    seed: int = 42
    horizon: int = 10  # predict next 10 intervals


class Paths:
    artifacts_dir: str = "artifacts"
    outputs_dir: str = "outputs"

class Config:
    data: DataConfig = DataConfig()
    train: TrainConfig = TrainConfig()
    paths: Paths = Paths()

cfg = Config()
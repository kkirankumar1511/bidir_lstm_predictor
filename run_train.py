import argparse, os, json
from config import cfg, Config, DataConfig, TrainConfig, Paths
from utils import ensure_dirs, save_json
from pipeline import train_pipeline

def main():
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--ticker', type=str, required=True, help='Ticker symbol (e.g., AAPL, TCS.NS)')
    #args = parser.parse_args()

    cfg.data.ticker = "BEL.NS"
    ensure_dirs(cfg.paths.artifacts_dir, cfg.paths.outputs_dir)
    res = train_pipeline( cfg.data.ticker, cfg, cfg.paths.artifacts_dir)

    print("Validation metrics:\n", json.dumps(res['val_metrics'], indent=2))
    save_json(os.path.join(cfg.paths.artifacts_dir, 'val_metrics.json'), res['val_metrics'])
    save_json(os.path.join(cfg.paths.artifacts_dir, 'train_history.json'), res['train_info']['history'])

if __name__ == '__main__':
    main()
import argparse, os, json
from config import cfg
from utils import ensure_dirs, save_json
from pipeline import predict_pipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str, required=True, help='Ticker symbol (e.g., AAPL, TCS.NS)')
    args = parser.parse_args()

    cfg.data.ticker = args.ticker
    ensure_dirs(cfg.paths.artifacts_dir, cfg.paths.outputs_dir)
    res = predict_pipeline(args.ticker, cfg, cfg.paths.artifacts_dir, cfg.paths.outputs_dir)

    print("Saved:", res['csv'], res['json'])
    print("Preview:", res['preview'])

if __name__ == '__main__':
    main()
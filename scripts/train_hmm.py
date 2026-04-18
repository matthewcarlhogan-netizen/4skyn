"""
OpenClaw V7 — HMM training

Changes vs V6:
  - Drops 3 noise features (funding constant, oichg=0, btcdomz=0)
  - Fetches REAL funding rate + OI history from mainnet
  - Preserves EM-learned transition matrix — no hand-stamp override
  - Consistent feature pipeline with live inference (shared via RegimeDetector)
"""
from __future__ import annotations
import os
import pickle
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from hmmlearn import hmm
from pybit.unified_trading import HTTP
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# Add parent dir to path so `from core...` works when running this script directly
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.regime_hmm import RegimeDetector, FEATURES, N_FEATURES


def _fetch_klines(client, symbol: str, days: int, interval: str = "60") -> pd.DataFrame:
    bars_per_day = {"15": 96, "60": 24, "240": 6, "D": 1}.get(interval, 24)
    all_rows = []
    end = int(time.time() * 1000)
    target = days * bars_per_day
    retries = 0
    while len(all_rows) < target and retries < 5:
        try:
            r = client.get_kline(
                category="linear", symbol=symbol,
                interval=interval, limit=1000, end=end,
            )
            if r["retCode"] != 0:
                break
            rows = r["result"]["list"]
            if not rows:
                break
            all_rows.extend(rows)
            end = int(rows[-1][0]) - 1
            time.sleep(0.2)
        except Exception as e:
            retries += 1
            print(f"fetch retry {retries}: {e}")
            time.sleep(1)
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(
        all_rows,
        columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    return df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def train_and_save(cfg: dict):
    symbol = cfg["execution"]["symbol"]
    days = cfg["regime"]["training_days"]
    model_path = Path(cfg["infra"]["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)

    interval = cfg["execution"]["timeframe"]
    print(f"Fetching {days} days of {interval}m klines for {symbol} from MAINNET...")
    client = HTTP(testnet=False)
    df = _fetch_klines(client, symbol, days, interval=interval)
    if df.empty or len(df) < 1000:
        print(f"Failed — got {len(df)} rows")
        return

    print(f"Engineering features (real funding + OI, no noise)...")
    # Use the shared feature-engineering path — but we need a bootstrap detector
    # to call engineer_features(). Since we don't have a model yet, use a shim:
    class _Shim(RegimeDetector):
        def __init__(self):
            self.http = HTTP(testnet=False)
    shim = _Shim()
    feat_df = shim.engineer_features(df, symbol).dropna(subset=FEATURES)
    print(f"  {len(feat_df)} clean rows after feature engineering")

    X = feat_df[FEATURES].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    print(f"  feature shape = {Xs.shape}")

    print(f"Training 3-state GaussianHMM (cov={cfg['regime']['cov_type']})...")
    model = hmm.GaussianHMM(
        n_components=cfg["regime"]["n_states"],
        covariance_type=cfg["regime"]["cov_type"],
        n_iter=200, verbose=False, random_state=42,
    )
    model.fit(Xs)

    # Sort states by mean log_return (feature 0): 0=CRISIS, 1=RANGE, 2=BULL
    order = np.argsort(model.means_[:, 0])
    model.means_ = model.means_[order]
    model.transmat_ = model.transmat_[order][:, order]     # remap rows AND cols
    model.startprob_ = model.startprob_[order]
    model._covars_ = model.covars_[order].copy()
    if cfg["regime"]["cov_type"] == "diag":
        model._covars_ = np.clip(model._covars_, 1e-8, None)

    # ✅ IMPORTANT: we do NOT overwrite transmat with a hand-picked matrix.
    # The learned dynamics are kept intact.
    print("Learned transition matrix (rows=from, cols=to):")
    print(np.round(model.transmat_, 3))

    bundle = {
        "model": model,
        "scaler": scaler,
        "state_mapping": {0: "CRISIS", 1: "RANGE", 2: "BULL"},
        "features": FEATURES,
        "training_days": days,
        "trained_at": pd.Timestamp.utcnow().isoformat(),
    }
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"✅ Saved to {model_path}")


if __name__ == "__main__":
    cfg_path = Path(__file__).parent.parent / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    train_and_save(cfg)

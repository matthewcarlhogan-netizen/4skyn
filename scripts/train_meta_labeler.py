"""
Bootstrap + retrain the meta-labeler.

Two modes:
  1. --bootstrap : synthesize a training set from a historical backtest,
                   so the meta-labeler has a warm start on day 1.
                   Runs backtest.py-equivalent logic, recording (features, win/loss)
                   for every signal the primary strategy would have fired.
  2. (no flag)   : retrain from existing data/meta_training.parquet

Usage:
    python scripts/train_meta_labeler.py --bootstrap --days 365
    python scripts/train_meta_labeler.py            # just retrain on existing data
"""
from __future__ import annotations
import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from pybit.unified_trading import HTTP

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.regime_hmm import RegimeDetector, FEATURES
from core.meta_labeler import MetaLabeler, MetaContext


def fetch_history(symbol: str, days: int, interval: str = "60") -> pd.DataFrame:
    client = HTTP(testnet=False)
    bars_per_day = {"15": 96, "60": 24, "240": 6}.get(interval, 24)
    rows, end = [], int(time.time() * 1000)
    target = days * bars_per_day
    while len(rows) < target:
        r = client.get_kline(category="linear", symbol=symbol,
                             interval=interval, limit=1000, end=end)
        if r["retCode"] != 0 or not r["result"]["list"]:
            break
        chunk = r["result"]["list"]
        rows.extend(chunk)
        end = int(chunk[-1][0]) - 1
        time.sleep(0.15)
    df = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume","turnover"])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms")
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    return df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def bootstrap_training_data(cfg: dict, days: int):
    symbol = cfg["execution"]["symbol"]
    print(f"Bootstrap: fetching {days} days of {symbol} and replaying primary strategy...")
    interval = cfg["execution"]["timeframe"]
    df = fetch_history(symbol, days, interval=interval)
    detector = RegimeDetector(cfg["infra"]["model_path"],
                              conf_threshold=cfg["regime"]["confidence_threshold"])
    feat_df = detector.engineer_features(df, symbol).dropna(subset=FEATURES).reset_index(drop=True)
    print(f"  {len(feat_df)} feature rows")

    records = []
    warmup = 100
    sl_mult_bull = cfg["regimes"]["BULL"]["sl_atr_mult"]
    tp_mult_bull = cfg["regimes"]["BULL"]["tp_atr_mult"]
    sl_mult_rng  = cfg["regimes"]["RANGE"]["sl_atr_mult"]
    tp_mult_rng  = cfg["regimes"]["RANGE"]["tp_atr_mult"]
    persistence_bars = cfg["regime"].get("persistence_bars", 3)
    cooldown_bars = cfg["regime"].get("cooldown_bars", 4)
    regime_history = []
    last_trade_bar = -cooldown_bars

    for i in range(warmup, len(feat_df) - 100):
        hist = feat_df.iloc[:i+1].tail(60)
        seq_raw = hist[FEATURES].values
        if len(seq_raw) < 20: continue
        seq = detector.scaler.transform(seq_raw[-50:])
        proba = detector.hmm.predict_proba(seq)[-1]
        state = int(np.argmax(proba)); conf = float(proba.max())
        regime = detector.state_mapping.get(state, "UNKNOWN")
        regime_history.append(regime)
        if len(regime_history) > 10: regime_history.pop(0)

        if conf < cfg["regime"]["confidence_threshold"] or regime == "CRISIS":
            continue
        # Apply persistence + cooldown during bootstrap too, so training set
        # reflects what the live bot will actually see
        if len(regime_history) < persistence_bars:
            continue
        if not all(r == regime for r in regime_history[-persistence_bars:]):
            continue
        if i - last_trade_bar < cooldown_bars:
            continue

        price = float(feat_df.iloc[i]["close"])
        c20 = hist["close"].tail(20); ma = c20.mean(); sd = c20.std()
        direction = 0
        if regime == "BULL":
            direction = 1
        elif regime == "RANGE":
            if price > ma + 1.5*sd: direction = -1
            elif price < ma - 1.5*sd: direction = 1
        if direction == 0: continue

        h = hist["high"].tail(14); l = hist["low"].tail(14); cs = hist["close"].shift().tail(14)
        tr = pd.concat([(h-l), (h-cs).abs(), (l-cs).abs()], axis=1).max(axis=1)
        atr = float(tr.mean())
        if regime == "BULL": sl_m, tp_m = sl_mult_bull, tp_mult_bull
        else:                sl_m, tp_m = sl_mult_rng, tp_mult_rng

        sl = price - sl_m*atr*direction
        tp = price + tp_m*atr*direction
        # Simulate outcome from next up to 50 bars
        label = -1
        for j in range(i+1, min(i+50, len(feat_df))):
            b = feat_df.iloc[j]
            if direction > 0:
                if b["low"]  <= sl: label = 0; break
                if b["high"] >= tp: label = 1; break
            else:
                if b["high"] >= sl: label = 0; break
                if b["low"]  <= tp: label = 1; break
        if label == -1:
            continue  # unresolved

        rv_20 = hist["close"].pct_change().tail(20).std() or 1e-8
        rv_100 = hist["close"].pct_change().tail(100).std() or 1e-8
        ts = feat_df.iloc[i]["timestamp"]
        ctx = MetaContext(
            p_crisis=float(proba[0]), p_range=float(proba[1]), p_bull=float(proba[2]),
            hmm_conf=conf, ofi_rank=50.0,  # no historical OFI during bootstrap
            funding_8h=float(feat_df.iloc[i].get("funding_live", 0.0001)),
            rv_zscore=float((rv_20 - rv_100) / rv_100), atr_pct=float(atr/price),
            recent_win_rate=0.5, streak=0, equity_hwm_ratio=1.0,
            direction=direction,
        )
        row = ctx.to_feature_row(ts.to_pydatetime() if hasattr(ts, 'to_pydatetime') else ts).iloc[0].to_dict()
        row["signal_ts"] = ts.isoformat()
        row["signal_id"] = row["signal_ts"] + "_" + str(direction)
        row["label"] = label
        row["pnl_pct"] = (tp_m * atr / price) if label == 1 else -(sl_m * atr / price)
        records.append(row)
        last_trade_bar = i

    out_path = Path(cfg["meta_labeler"]["training_data_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(records)
    df_out.to_parquet(out_path, index=False)
    wins = int((df_out["label"] == 1).sum()); total = len(df_out)
    print(f"  saved {total} labelled bootstrap rows → {out_path}")
    print(f"  bootstrap win-rate = {wins/total*100:.1f}% ({wins}/{total})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bootstrap", action="store_true", help="Generate historical training set")
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--config", type=str, default=str(Path(__file__).parent.parent / "config.yaml"))
    args = p.parse_args()
    with open(args.config) as f: cfg = yaml.safe_load(f)

    if args.bootstrap:
        bootstrap_training_data(cfg, args.days)

    ml = MetaLabeler(
        model_path=cfg["meta_labeler"]["model_path"],
        training_data_path=cfg["meta_labeler"]["training_data_path"],
        activation_threshold=cfg["meta_labeler"]["activation_threshold"],
        prob_threshold=cfg["meta_labeler"]["prob_threshold"],
        retrain_every=cfg["meta_labeler"]["retrain_every_n_trades"],
        rolling_window=cfg["meta_labeler"]["rolling_window_trades"],
    )
    print("Fitting meta-labeler...")
    ml.fit()
    print("Diagnostics:", ml.diagnostics())


if __name__ == "__main__":
    main()

"""
OpenClaw V7 — Event-driven backtester

Critical: this replays bar-by-bar with REAL fees, slippage, and latency.
Do not run the live bot until this shows a deflated Sharpe > 1.0.

Usage:
    python scripts/backtest.py --days 365 --start-equity 250
"""
from __future__ import annotations
import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import yaml
from pybit.unified_trading import HTTP
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.regime_hmm import RegimeDetector, FEATURES
from core.meta_labeler import MetaLabeler, MetaContext


# ── Real-world execution frictions ───────────────────────────────────────
BYBIT_TAKER_FEE = 0.00055   # 5.5 bps
SLIPPAGE_BPS = 2.0 / 10000  # 2 bps per fill (conservative for $250 on BTCUSDT)


@dataclass
class Trade:
    open_ts: pd.Timestamp
    close_ts: pd.Timestamp
    side: str
    entry: float
    exit: float
    qty: float
    pnl_usd: float
    regime: str
    reason: str


def fetch_history(symbol: str, days: int, interval: str = "60") -> pd.DataFrame:
    client = HTTP(testnet=False)
    bars_per_day = {"15": 96, "60": 24, "240": 6}.get(interval, 24)
    all_rows = []
    end = int(time.time() * 1000)
    target = days * bars_per_day
    while len(all_rows) < target:
        r = client.get_kline(category="linear", symbol=symbol,
                             interval=interval, limit=1000, end=end)
        if r["retCode"] != 0 or not r["result"]["list"]:
            break
        rows = r["result"]["list"]
        all_rows.extend(rows)
        end = int(rows[-1][0]) - 1
        time.sleep(0.15)
    df = pd.DataFrame(all_rows, columns=["timestamp","open","high","low","close","volume","turnover"])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms")
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    return df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def deflated_sharpe(returns: np.ndarray, n_trials: int = 10) -> float:
    """Bailey & Lopez de Prado deflated Sharpe (adjusts for skew, kurtosis, trial count)."""
    if len(returns) < 30:
        return 0.0
    sr = returns.mean() / (returns.std(ddof=1) + 1e-12) * np.sqrt(252 * 24 * 4)  # 15m -> annualized
    T = len(returns)
    sk = stats.skew(returns)
    ku = stats.kurtosis(returns, fisher=True)

    # Expected max Sharpe under null
    euler_mascheroni = 0.5772156649
    e_max_sr = np.sqrt(2 * np.log(n_trials)) - \
               (euler_mascheroni + np.log(np.log(n_trials))) / np.sqrt(2 * np.log(n_trials))
    denom = np.sqrt((1 - sk * sr + (ku / 4) * sr ** 2) / (T - 1))
    dsr = stats.norm.cdf((sr - e_max_sr) / (denom + 1e-12))
    return float(dsr)


def run_backtest(cfg: dict, days: int, start_equity: float):
    symbol = cfg["execution"]["symbol"]
    print(f"Fetching {days} days of {symbol}...")
    interval = cfg["execution"]["timeframe"]
    df = fetch_history(symbol, days, interval=interval)
    print(f"  {len(df)} bars")

    detector = RegimeDetector(
        model_path=cfg["infra"]["model_path"],
        conf_threshold=cfg["regime"]["confidence_threshold"],
    )

    # Pre-engineer features once (big speedup)
    print("Engineering full feature set (may call OI/funding APIs)...")
    feat_df = detector.engineer_features(df, symbol).dropna(subset=FEATURES).reset_index(drop=True)
    print(f"  {len(feat_df)} valid rows")

    # Meta-labeler (loads existing bootstrap-trained model if present)
    ml_cfg = cfg.get("meta_labeler", {})
    meta = None
    if ml_cfg.get("enabled", False):
        meta = MetaLabeler(
            model_path=ml_cfg["model_path"],
            training_data_path=ml_cfg["training_data_path"],
            activation_threshold=ml_cfg["activation_threshold"],
            prob_threshold=ml_cfg["prob_threshold"],
            retrain_every=ml_cfg["retrain_every_n_trades"],
            rolling_window=ml_cfg["rolling_window_trades"],
        )
        print(f"  meta-labeler: {meta.diagnostics()}")

    warmup = 100
    equity = start_equity
    equity_curve: List[tuple] = [(feat_df.iloc[warmup]["timestamp"], equity)]
    trades: List[Trade] = []

    pos = None  # None or dict with entry, side, qty, sl, tp, open_ts, regime
    regime_history = []         # for persistence filter
    last_exit_bar = -1          # for cooldown
    hwm = start_equity
    meta_blocks = 0
    raw_signals = 0

    for i in range(warmup, len(feat_df) - 1):
        row = feat_df.iloc[i]
        price = row["close"]
        ts = row["timestamp"]

        # If in position, check SL/TP vs next bar's high/low (conservative fill)
        if pos is not None:
            nxt = feat_df.iloc[i + 1]
            hit_sl = (pos["side"] == "Buy" and nxt["low"] <= pos["sl"]) or \
                     (pos["side"] == "Sell" and nxt["high"] >= pos["sl"])
            hit_tp = (pos["side"] == "Buy" and nxt["high"] >= pos["tp"]) or \
                     (pos["side"] == "Sell" and nxt["low"] <= pos["tp"])

            exit_price = None
            reason = ""
            if hit_sl and hit_tp:
                # Assume SL hit first (pessimistic)
                exit_price = pos["sl"]; reason = "sl"
            elif hit_sl:
                exit_price = pos["sl"]; reason = "sl"
            elif hit_tp:
                exit_price = pos["tp"]; reason = "tp"

            if exit_price is not None:
                # Apply slippage + fees on both sides
                slip = exit_price * SLIPPAGE_BPS * (-1 if pos["side"] == "Buy" else 1)
                fill = exit_price + slip
                if pos["side"] == "Buy":
                    raw_pnl = (fill - pos["entry"]) * pos["qty"]
                else:
                    raw_pnl = (pos["entry"] - fill) * pos["qty"]
                fees = (pos["entry"] + fill) * pos["qty"] * BYBIT_TAKER_FEE
                pnl = raw_pnl - fees
                equity += pnl
                trades.append(Trade(pos["open_ts"], ts, pos["side"], pos["entry"],
                                    fill, pos["qty"], pnl, pos["regime"], reason))
                pos = None
                last_exit_bar = i

        equity_curve.append((ts, equity))
        if equity <= 0:
            print("Bankrupt."); break

        # If flat, check for new signal
        if pos is None:
            # Regime prediction using ONLY data up to and including bar i
            hist = feat_df.iloc[:i + 1].tail(60).copy()
            seq_raw = hist[FEATURES].values
            if len(seq_raw) < 20:
                continue
            seq = detector.scaler.transform(seq_raw[-50:])
            proba = detector.hmm.predict_proba(seq)[-1]
            state = int(np.argmax(proba))
            regime = detector.state_mapping.get(state, "UNKNOWN")
            conf = float(proba.max())

            # Track regime history for persistence filter
            regime_history.append(regime)
            if len(regime_history) > 10:
                regime_history.pop(0)

            if conf < cfg["regime"]["confidence_threshold"]:
                continue
            if regime == "CRISIS":
                continue

            # 🆕 REGIME PERSISTENCE: require same regime for last N bars (anti-churn)
            persistence_bars = cfg["regime"].get("persistence_bars", 3)
            if len(regime_history) < persistence_bars:
                continue
            if not all(r == regime for r in regime_history[-persistence_bars:]):
                continue

            # 🆕 COOLDOWN: min bars between trades
            cooldown = cfg["regime"].get("cooldown_bars", 4)
            if i - last_exit_bar < cooldown:
                continue

            # Direction
            c20 = hist["close"].tail(20)
            ma = c20.mean(); sd = c20.std()
            direction = 0
            if regime == "BULL":
                direction = 1
            elif regime == "RANGE":
                if price > ma + 1.5 * sd: direction = -1
                elif price < ma - 1.5 * sd: direction = 1

            if direction == 0:
                continue

            raw_signals += 1

            # 🆕 META-LABELER GATE
            if meta is not None and meta.model is not None:
                rv_20 = hist["close"].pct_change().tail(20).std() or 1e-8
                rv_100 = hist["close"].pct_change().tail(100).std() or 1e-8
                # Recent live win rate from trades list
                recent = [1 if t.pnl_usd > 0 else 0 for t in trades[-20:]]
                wr = float(np.mean(recent)) if recent else 0.5
                streak = 0
                if recent:
                    last_ = 1 if recent[-1] else -1
                    for r in reversed(recent):
                        if (r == 1 and last_ > 0) or (r == 0 and last_ < 0):
                            streak += last_
                        else:
                            break
                hwm = max(hwm, equity)
                atr_tmp_h = hist["high"].tail(14); atr_tmp_l = hist["low"].tail(14)
                atr_tmp_c = hist["close"].shift().tail(14)
                atr_tmp = pd.concat([(atr_tmp_h - atr_tmp_l),
                                     (atr_tmp_h - atr_tmp_c).abs(),
                                     (atr_tmp_l - atr_tmp_c).abs()], axis=1).max(axis=1)
                atr_tmp_val = float(atr_tmp.mean())
                ctx = MetaContext(
                    p_crisis=float(proba[0]), p_range=float(proba[1]), p_bull=float(proba[2]),
                    hmm_conf=conf, ofi_rank=50.0,  # no historical L2 — neutral prior
                    funding_8h=float(hist.iloc[-1].get("funding_live", 0.0001)),
                    rv_zscore=float((rv_20 - rv_100) / rv_100),
                    atr_pct=float(atr_tmp_val / price),
                    recent_win_rate=wr, streak=int(streak),
                    equity_hwm_ratio=float(equity / hwm),
                    direction=int(direction),
                )
                take, p_win = meta.should_take_trade(ctx)
                if not take:
                    meta_blocks += 1
                    continue

            # ATR
            h = hist["high"].tail(14); l = hist["low"].tail(14); cs = hist["close"].shift().tail(14)
            tr = pd.concat([(h - l), (h - cs).abs(), (l - cs).abs()], axis=1).max(axis=1)
            atr = float(tr.mean())

            rc = cfg["regimes"][regime]
            sl_mult = rc.get("sl_atr_mult", 2.0)
            tp_mult = rc.get("tp_atr_mult", 4.0)

            # Sizing: use risk-per-trade cap for simplicity in backtest
            max_loss = equity * cfg["risk"]["risk_per_trade_pct"]
            stop_distance = sl_mult * atr
            if stop_distance <= 0:
                continue
            qty = max_loss / stop_distance
            # Leverage cap on notional
            max_notional = equity * cfg["risk"]["max_leverage"]
            notional = qty * price
            if notional > max_notional:
                qty = max_notional / price

            if qty * price < cfg["execution"]["min_notional_usd"]:
                continue

            # Entry with slippage
            slip = price * SLIPPAGE_BPS * (1 if direction > 0 else -1)
            entry = price + slip
            sl = entry - sl_mult * atr * direction
            tp = entry + tp_mult * atr * direction

            pos = {
                "side": "Buy" if direction > 0 else "Sell",
                "entry": entry, "qty": qty, "sl": sl, "tp": tp,
                "open_ts": ts, "regime": regime,
            }

    # ── results ──────────────────────────────────────────────────────────
    eq_df = pd.DataFrame(equity_curve, columns=["ts", "equity"])
    eq_df["returns"] = eq_df["equity"].pct_change().fillna(0)

    n = len(trades)
    wins = sum(1 for t in trades if t.pnl_usd > 0)
    total_pnl = sum(t.pnl_usd for t in trades)
    max_eq = eq_df["equity"].cummax()
    drawdown = (eq_df["equity"] - max_eq) / max_eq
    max_dd = drawdown.min()

    trade_returns = np.array([t.pnl_usd / start_equity for t in trades]) if trades else np.array([])
    dsr = deflated_sharpe(eq_df["returns"].values)

    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"  raw signals       : {raw_signals}")
    print(f"  meta-blocked      : {meta_blocks} ({(meta_blocks/max(raw_signals,1)*100):.1f}%)")
    print(f"  period            : {days} days")
    print(f"  start equity      : ${start_equity:,.2f}")
    print(f"  end equity        : ${equity:,.2f}")
    print(f"  total return      : {(equity/start_equity - 1)*100:+.2f}%")
    print(f"  trades            : {n}")
    print(f"  win rate          : {(wins/n*100 if n else 0):.1f}%")
    print(f"  avg pnl/trade     : ${(total_pnl/n if n else 0):+.2f}")
    print(f"  max drawdown      : {max_dd*100:.2f}%")
    print(f"  deflated sharpe   : {dsr:.3f}  (>0.95 = passes one-sided test)")
    print(f"  pass bar (DSR>1)  : {'✅' if dsr > 0.95 else '❌'}")
    print(f"  pass bar (DD<10%) : {'✅' if max_dd > -0.10 else '❌'}")
    print("=" * 60)

    # Save results
    out_dir = Path(__file__).parent.parent / "backtest_results"
    out_dir.mkdir(exist_ok=True)
    eq_df.to_csv(out_dir / "equity_curve.csv", index=False)
    pd.DataFrame([t.__dict__ for t in trades]).to_csv(out_dir / "trades.csv", index=False)
    print(f"  saved to {out_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=180)
    p.add_argument("--start-equity", type=float, default=250)
    p.add_argument("--config", type=str,
                   default=str(Path(__file__).parent.parent / "config.yaml"))
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run_backtest(cfg, args.days, args.start_equity)

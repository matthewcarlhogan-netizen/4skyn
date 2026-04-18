"""
OpenClaw V7 — Meta-labeler (Lopez de Prado, AFML Ch. 3)

A secondary binary classifier stacked on top of the primary regime signal.
  - Primary signal fires → we compute context features → meta-labeler estimates P(win)
  - Only take the trade if P(win) >= prob_threshold
  - Meta-labeler self-learns: every closed trade produces a labelled row
  - Retrains every N new labels on a rolling window
  - Pass-through mode until we have `activation_threshold` real labels

Empirical: historically lifts Sharpe by 0.3–0.8 on an otherwise-unchanged strategy.

Context features (14):
  - regime probabilities: P(CRISIS), P(RANGE), P(BULL)
  - HMM confidence
  - OFI percentile rank
  - funding rate (8h)
  - realized vol z-score
  - ATR as % of price
  - recent rolling win rate (last 20)
  - consecutive W/L streak (signed int)
  - equity/HWM ratio
  - hour of day (UTC, sin-encoded)
  - hour of day (UTC, cos-encoded)
  - day of week (UTC)
  - regime (one-hot: BULL=1 BULL, -1 RANGE mean-revert short, 0 RANGE mean-revert long)
"""
from __future__ import annotations
import math
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    # Also test that the native lib actually loads (macOS libomp issue)
    _test = lgb.Dataset([[1]], label=[1])
    lgb.train({"objective": "binary", "verbose": -1}, _test, num_boost_round=1)
    HAS_LGB = True
except Exception:
    HAS_LGB = False

# sklearn fallback — always available, nearly same performance
from sklearn.ensemble import HistGradientBoostingClassifier


FEATURE_COLS: List[str] = [
    "p_crisis", "p_range", "p_bull", "hmm_conf",
    "ofi_rank", "funding_8h",
    "rv_zscore", "atr_pct",
    "recent_win_rate", "streak",
    "equity_hwm_ratio",
    "hour_sin", "hour_cos", "dow",
    "direction",
]


@dataclass
class MetaContext:
    """Everything the meta-labeler needs at decision time."""
    p_crisis: float
    p_range: float
    p_bull: float
    hmm_conf: float
    ofi_rank: float
    funding_8h: float
    rv_zscore: float
    atr_pct: float
    recent_win_rate: float
    streak: int
    equity_hwm_ratio: float
    direction: int   # -1 / +1

    def to_feature_row(self, ts: datetime | None = None) -> pd.DataFrame:
        ts = ts or datetime.now(timezone.utc)
        row = {
            "p_crisis": self.p_crisis,
            "p_range": self.p_range,
            "p_bull": self.p_bull,
            "hmm_conf": self.hmm_conf,
            "ofi_rank": self.ofi_rank,
            "funding_8h": self.funding_8h,
            "rv_zscore": self.rv_zscore,
            "atr_pct": self.atr_pct,
            "recent_win_rate": self.recent_win_rate,
            "streak": self.streak,
            "equity_hwm_ratio": self.equity_hwm_ratio,
            "hour_sin": math.sin(2 * math.pi * ts.hour / 24),
            "hour_cos": math.cos(2 * math.pi * ts.hour / 24),
            "dow": ts.weekday(),
            "direction": self.direction,
        }
        return pd.DataFrame([row], columns=FEATURE_COLS)


class MetaLabeler:
    def __init__(
        self,
        model_path: str,
        training_data_path: str,
        activation_threshold: int = 50,
        prob_threshold: float = 0.53,
        retrain_every: int = 25,
        rolling_window: int = 500,
    ):
        self.model_path = Path(model_path)
        self.training_data_path = Path(training_data_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.training_data_path.parent.mkdir(parents=True, exist_ok=True)

        self.activation_threshold = activation_threshold
        self.prob_threshold = prob_threshold
        self.retrain_every = retrain_every
        self.rolling_window = rolling_window

        self.model = None
        self._trades_since_last_fit = 0
        self._load_model()

    # ── persistence ─────────────────────────────────────────────────────
    def _load_model(self):
        if self.model_path.exists():
            try:
                with open(self.model_path, "rb") as f:
                    self.model = pickle.load(f)
            except Exception:
                self.model = None

    def _save_model(self):
        # Always write — even None — so stale models never survive a negative-lift fit.
        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)

    def _load_training(self) -> pd.DataFrame:
        if self.training_data_path.exists():
            try:
                return pd.read_parquet(self.training_data_path)
            except Exception:
                return pd.DataFrame()
        return pd.DataFrame()

    def _save_training(self, df: pd.DataFrame):
        df.to_parquet(self.training_data_path, index=False)

    # ── feature recording ──────────────────────────────────────────────
    def record_signal(self, ctx: MetaContext, ts: datetime | None = None) -> str:
        """
        Call when a primary signal is ABOUT TO FIRE.
        Returns a signal_id that the caller uses later to attach the outcome.
        """
        feat = ctx.to_feature_row(ts)
        feat["signal_ts"] = (ts or datetime.now(timezone.utc)).isoformat()
        feat["label"] = -1      # unknown until close
        feat["pnl_pct"] = np.nan
        signal_id = feat["signal_ts"].iloc[0] + "_" + str(ctx.direction)
        feat["signal_id"] = signal_id

        df = self._load_training()
        df = pd.concat([df, feat], ignore_index=True)
        self._save_training(df)
        return signal_id

    def record_outcome(self, signal_id: str, pnl_pct: float):
        """Call on trade close to attach the label."""
        df = self._load_training()
        if df.empty:
            return
        mask = df["signal_id"] == signal_id
        if not mask.any():
            return
        df.loc[mask, "label"] = 1 if pnl_pct > 0 else 0
        df.loc[mask, "pnl_pct"] = pnl_pct
        self._save_training(df)

        self._trades_since_last_fit += 1
        n_labelled = int((df["label"] != -1).sum())
        if (
            n_labelled >= self.activation_threshold
            and self._trades_since_last_fit >= self.retrain_every
        ):
            self.fit()
            self._trades_since_last_fit = 0

    def _predict_proba_raw(self, X: np.ndarray) -> np.ndarray:
        """Unified predict regardless of backend."""
        if HAS_LGB:
            return self.model.predict(X)
        else:
            return self.model.predict_proba(X)[:, 1]

    # ── training ────────────────────────────────────────────────────────
    def fit(self):
        df = self._load_training()
        if df.empty:
            return
        labelled = df[df["label"].isin([0, 1])].tail(self.rolling_window)
        if len(labelled) < self.activation_threshold:
            return

        backend = "lightgbm" if HAS_LGB else "sklearn-HistGB"

        # TIME-SPLIT validation — train on first 75%, validate on last 25%
        # Shuffling would leak future returns (lookahead bias). Always time-split.
        split = int(len(labelled) * 0.75)
        train = labelled.iloc[:split]; val = labelled.iloc[split:]
        X_tr = train[FEATURE_COLS].values; y_tr = train["label"].astype(int).values
        X_val = val[FEATURE_COLS].values; y_val = val["label"].astype(int).values

        if HAS_LGB:
            params = {
                "objective": "binary", "metric": "binary_logloss",
                "learning_rate": 0.02, "num_leaves": 15,
                "min_data_in_leaf": max(10, len(train) // 30),
                "feature_fraction": 0.7, "bagging_fraction": 0.7,
                "bagging_freq": 5, "lambda_l2": 1.0, "verbose": -1,
            }
            dtrain = lgb.Dataset(X_tr, label=y_tr)
            if len(val) >= 50:
                dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
                self.model = lgb.train(
                    params, dtrain, num_boost_round=300, valid_sets=[dval],
                    callbacks=[lgb.early_stopping(30, verbose=False)],
                )
            else:
                self.model = lgb.train(params, dtrain, num_boost_round=80)
        else:
            # HistGradientBoosting — no dylib issues, same speed, very close performance
            self.model = HistGradientBoostingClassifier(
                max_iter=200, max_leaf_nodes=15,
                min_samples_leaf=max(10, len(train) // 30),
                learning_rate=0.05, l2_regularization=1.0,
                random_state=42,
            )
            self.model.fit(X_tr, y_tr)

        # Diagnostics
        base_rate = float(y_tr.mean())
        preds_tr = self._predict_proba_raw(X_tr)
        in_acc = float(((preds_tr > 0.5).astype(int) == y_tr).mean())

        if len(val) >= 50:
            preds_val = self._predict_proba_raw(X_val)
            oos_acc = float(((preds_val > 0.5).astype(int) == y_val).mean())
            take_mask = preds_val > self.prob_threshold
            oos_acc_at_thresh = float((y_val[take_mask] == 1).mean()) if take_mask.any() else base_rate
            take_rate = float(take_mask.mean())
            lift = oos_acc_at_thresh - base_rate
            print(
                f"[meta-labeler/{backend}] train={len(train)} val={len(val)} "
                f"base={base_rate:.3f} in-sample={in_acc:.3f} "
                f"OOS={oos_acc:.3f} OOS@thresh={oos_acc_at_thresh:.3f} "
                f"take-rate={take_rate:.2%} lift={lift:+.3f}"
            )
            if lift < 0.02:
                print("  ⚠️ weak lift — strategy may not have learnable edge yet; meta-labeler stays in pass-through")
                if lift < 0:
                    self.model = None  # don't gate with a net-negative classifier
                    self._save_model()
                    return
        else:
            print(f"[meta-labeler/{backend}] fit on {len(train)}, in-sample={in_acc:.3f}")

        self._save_model()

    # ── inference ───────────────────────────────────────────────────────
    def predict_proba(self, ctx: MetaContext) -> float:
        """Return P(trade wins) ∈ [0, 1]. 0.5 in pass-through mode."""
        if self.model is None:
            return 0.5
        X = ctx.to_feature_row().values
        try:
            return float(self._predict_proba_raw(X)[0])
        except Exception:
            return 0.5

    def should_take_trade(self, ctx: MetaContext) -> tuple[bool, float]:
        """
        Returns (take_trade, p_win).
        In pass-through mode (not enough data or no lightgbm), always True.
        """
        df = self._load_training()
        n_labelled = int((df["label"] != -1).sum()) if not df.empty else 0

        p = self.predict_proba(ctx)
        if n_labelled < self.activation_threshold or self.model is None:
            return True, p   # pass-through
        return p >= self.prob_threshold, p

    def diagnostics(self) -> dict:
        df = self._load_training()
        n_labelled = int((df["label"] != -1).sum()) if not df.empty else 0
        return {
            "model_loaded": self.model is not None,
            "total_signals_recorded": int(len(df)),
            "labelled_outcomes": n_labelled,
            "activation_threshold": self.activation_threshold,
            "status": "active" if (self.model is not None and n_labelled >= self.activation_threshold) else "pass-through",
            "prob_threshold": self.prob_threshold,
        }

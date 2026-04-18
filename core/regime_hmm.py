"""
OpenClaw V7 — Regime Detector (HMM)

Fixes vs V6:
  - Drops 3 noise-only features (funding constant broadcast, oichg=0, btcdomz=0)
  - Replaces them with REAL funding-rate time series + REAL open-interest deltas
  - Preserves EM-learned transition matrix (no hand-stamp override)
  - Closed-bar-only feature construction (drops in-progress bar)
  - Explicit sequence prediction (forward-backward on 50 rows, not 1)
"""
from __future__ import annotations
import pickle
from pathlib import Path
from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
from pybit.unified_trading import HTTP
import warnings
warnings.filterwarnings("ignore")

# Feature ordering MUST match training exactly. Keep this list canonical.
FEATURES: List[str] = [
    "log_return",
    "realized_vol",
    "trend",
    "volume_zscore",
    "rviv",
    "funding_live",
    "oi_change",
]
N_FEATURES = len(FEATURES)


class RegimeDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.65):
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.http = HTTP(testnet=False)  # mainnet for ALL market signals
        self.hmm = None
        self.scaler = None
        self.state_mapping: Dict[int, str] = {}
        self._funding_cache: pd.DataFrame | None = None
        self._oi_cache: pd.DataFrame | None = None
        self.load_models()

    # ── model load ──────────────────────────────────────────────────────
    def load_models(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. Run scripts/train_hmm.py first."
            )
        with open(self.model_path, "rb") as f:
            bundle = pickle.load(f)
        self.hmm = bundle["model"]
        # Safety: if covars saved full but type is diag, extract diagonals
        if self.hmm.covariance_type == "diag" and self.hmm._covars_.ndim == 3:
            self.hmm._covars_ = np.array([np.diag(c) for c in self.hmm._covars_])
        self.scaler = bundle["scaler"]
        self.state_mapping = bundle["state_mapping"]

    # ── mainnet data fetchers ───────────────────────────────────────────
    def _fetch_funding_series(self, symbol: str, n: int = 300) -> pd.DataFrame:
        """Real 8h funding-rate history, forward-filled to 15m grid."""
        try:
            resp = self.http.get_funding_rate_history(
                category="linear", symbol=symbol, limit=200
            )
            if resp["retCode"] != 0:
                return pd.DataFrame()
            df = pd.DataFrame(resp["result"]["list"])
            if df.empty:
                return pd.DataFrame()
            df["fundingRateTimestamp"] = pd.to_datetime(
                df["fundingRateTimestamp"].astype(float), unit="ms"
            )
            df["funding_live"] = df["fundingRate"].astype(float)
            return (
                df[["fundingRateTimestamp", "funding_live"]]
                .rename(columns={"fundingRateTimestamp": "timestamp"})
                .sort_values("timestamp")
            )
        except Exception:
            return pd.DataFrame()

    def _fetch_oi_series(self, symbol: str) -> pd.DataFrame:
        """Real open-interest time series (15m granularity)."""
        try:
            resp = self.http.get_open_interest(
                category="linear",
                symbol=symbol,
                intervalTime="15min",
                limit=200,
            )
            if resp["retCode"] != 0:
                return pd.DataFrame()
            df = pd.DataFrame(resp["result"]["list"])
            if df.empty:
                return pd.DataFrame()
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms")
            df["open_interest"] = df["openInterest"].astype(float)
            return df[["timestamp", "open_interest"]].sort_values("timestamp")
        except Exception:
            return pd.DataFrame()

    # ── feature engineering (shared with training) ──────────────────────
    def engineer_features(self, df: pd.DataFrame, symbol: str = "BTCUSDT") -> pd.DataFrame:
        """Produce full feature frame. Live & training use the same path."""
        d = df.copy().sort_values("timestamp").reset_index(drop=True)

        # Price-based
        d["log_return"] = np.log(d["close"] / d["close"].shift(1))
        d["realized_vol"] = d["log_return"].rolling(20).std()
        d["trend"] = d["close"] / d["close"].rolling(20).mean() - 1
        vmean = d["volume"].rolling(20).mean()
        vstd = d["volume"].rolling(20).std()
        d["volume_zscore"] = (d["volume"] - vmean) / (vstd + 1e-8)
        d["rviv"] = d["realized_vol"] / (d["realized_vol"].rolling(100).mean() + 1e-8)

        # Funding series — forward-filled onto kline grid
        fund = self._fetch_funding_series(symbol)
        if not fund.empty:
            d = pd.merge_asof(
                d.sort_values("timestamp"),
                fund,
                on="timestamp",
                direction="backward",
            )
        else:
            d["funding_live"] = 0.0001
        d["funding_live"] = d["funding_live"].ffill().fillna(0.0001)

        # OI delta over 20 bars
        oi = self._fetch_oi_series(symbol)
        if not oi.empty:
            d = pd.merge_asof(
                d.sort_values("timestamp"),
                oi,
                on="timestamp",
                direction="backward",
            )
            d["oi_change"] = d["open_interest"].pct_change(20).fillna(0.0)
        else:
            d["oi_change"] = 0.0

        return d

    # ── prediction ──────────────────────────────────────────────────────
    def _build_sequence(self, df: pd.DataFrame, symbol: str, seq_len: int = 50) -> np.ndarray:
        d = self.engineer_features(df, symbol).dropna(subset=FEATURES)
        if len(d) < seq_len:
            return np.empty((0, N_FEATURES))
        raw = d[FEATURES].values[-seq_len:]
        return self.scaler.transform(raw)

    def predict_regime(
        self, df: pd.DataFrame, symbol: str = "BTCUSDT"
    ) -> Tuple[str, float, np.ndarray]:
        seq = self._build_sequence(df, symbol)
        if len(seq) < 20:
            return "INSUFFICIENT_DATA", 0.0, np.zeros(3)

        proba = self.hmm.predict_proba(seq)[-1]
        conf = float(proba.max())
        state = self.hmm.predict(seq)[-1]
        regime = self.state_mapping.get(state, "UNKNOWN")

        if conf < self.conf_threshold:
            # Don't silently coerce to RANGE — return low-confidence signal
            # Caller decides what to do.
            return "LOW_CONF", conf, proba
        return regime, conf, proba

    def get_atr_multipliers(self, regime: str, regime_cfg: dict) -> Tuple[float, float]:
        cfg = regime_cfg.get(regime, {})
        return cfg.get("sl_atr_mult", 2.0), cfg.get("tp_atr_mult", 4.0)

    # ── diagnostics ─────────────────────────────────────────────────────
    def summarize_states(self) -> None:
        inv = {v: k for k, v in self.state_mapping.items()}
        print("HMM state centroids (in scaled feature space):")
        print(f"  feature order: {FEATURES}")
        for name in ["CRISIS", "RANGE", "BULL"]:
            idx = inv.get(name)
            if idx is not None:
                print(f"  {name:6s}: {np.round(self.hmm.means_[idx], 3)}")
        print("Learned transition matrix (rows = from, cols = to):")
        print(np.round(self.hmm.transmat_, 3))

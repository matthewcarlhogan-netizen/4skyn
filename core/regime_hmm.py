"""
OpenClaw V7 — Regime Detector (HMM)
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
        self.http = HTTP(testnet=False)
        self.hmm = None
        self.scaler = None
        self.state_mapping: Dict[int, str] = {}
        self._funding_cache: pd.DataFrame | None = None
        self._oi_cache: pd.DataFrame | None = None
        self.load_models()

    def load_models(self):
        if not self.model_path.exists():
            print(f"[WARNING] Model not found at {self.model_path}. Using fallback regime logic.")
            self.hmm = None
            self.scaler = None
            self.state_mapping = {0: "RANGE", 1: "BULL", 2: "CRISIS"}
            return
        with open(self.model_path, "rb") as f:
            bundle = pickle.load(f)
        self.hmm = bundle["model"]
        if self.hmm.covariance_type == "diag" and self.hmm._covars_.ndim == 3:
            self.hmm._covars_ = np.array([np.diag(c) for c in self.hmm._covars_])
        self.scaler = bundle["scaler"]
        self.state_mapping = bundle["state_mapping"]

    def predict_regime(self, df: pd.DataFrame, symbol: str = "BTCUSDT") -> Tuple[str, float, np.ndarray]:
        seq = self._build_sequence(df, symbol)
        if len(seq) < 20:
            return "INSUFFICIENT_DATA", 0.0, np.zeros(3)

        if self.hmm is None:
            trend = seq[-1, 2] if seq.shape[0] > 0 else 0.0
            if trend > 0.5:
                return "BULL", 0.6, np.array([0.2, 0.6, 0.2])
            elif trend < -0.5:
                return "CRISIS", 0.6, np.array([0.6, 0.2, 0.2])
            else:
                return "RANGE", 0.6, np.array([0.2, 0.6, 0.2])

        proba = self.hmm.predict_proba(seq)[-1]
        conf = float(proba.max())
        state = self.hmm.predict(seq)[-1]
        regime = self.state_mapping.get(state, "UNKNOWN")

        if conf < self.conf_threshold:
            return "LOW_CONF", conf, proba
        return regime, conf, proba

    def get_atr_multipliers(self, regime: str, regime_cfg: dict) -> Tuple[float, float]:
        cfg = regime_cfg.get(regime, {})
        return cfg.get("sl_atr_mult", 2.0), cfg.get("tp_atr_mult", 4.0)

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

    def _build_sequence(self, df: pd.DataFrame, symbol: str) -> np.ndarray:
        df = df.copy()
        if "close" not in df.columns:
            df["close"] = df.iloc[:, 3]
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["realized_vol"] = df["log_return"].rolling(20).std() * np.sqrt(252)
        df["trend"] = (df["close"] - df["close"].rolling(50).mean()) / df["close"].rolling(50).std()
        df["volume_zscore"] = (df["volume"] - df["volume"].rolling(20).mean()) / df["volume"].rolling(20).std()
        df["rviv"] = df["realized_vol"] / (df["volume"].rolling(20).std() + 1e-8)
        funding = self._fetch_funding_series(symbol)
        oi = self._fetch_oi_series(symbol)
        df = df.merge(funding, on="timestamp", how="left")
        df = df.merge(oi, on="timestamp", how="left")
        df["funding_live"] = df["funding_live"].ffill().fillna(0.0)
        df["oi_change"] = df["open_interest"].diff().fillna(0.0)
        df = df.dropna()
        return self.scaler.transform(df[FEATURES].values) if self.scaler is not None else df[FEATURES].values

    def _fetch_funding_series(self, symbol: str, n: int = 300) -> pd.DataFrame:
        try:
            resp = self.http.get_funding_rate_history(category="linear", symbol=symbol, limit=200)
            if resp["retCode"] != 0:
                return pd.DataFrame()
            df = pd.DataFrame(resp["result"]["list"])
            if df.empty:
                return pd.DataFrame()
            df["fundingRateTimestamp"] = pd.to_datetime(df["fundingRateTimestamp"].astype(float), unit="ms")
            df["funding_live"] = df["fundingRate"].astype(float)
            return df[["fundingRateTimestamp", "funding_live"]].rename(columns={"fundingRateTimestamp": "timestamp"}).sort_values("timestamp")
        except Exception:
            return pd.DataFrame()

    def _fetch_oi_series(self, symbol: str) -> pd.DataFrame:
        try:
            resp = self.http.get_open_interest(category="linear", symbol=symbol, intervalTime="15min", limit=200)
            if resp["retCode"] != 0:
                return pd.DataFrame()
            df = pd.DataFrame(resp["result"]["list"])
            if df.empty:
                return pd.DataFrame()
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms")
            df["open_interest"] = df["openInterest"].astype(float)
            return df[["timestamp", "open_interest"]]
        except Exception:
            return pd.DataFrame()

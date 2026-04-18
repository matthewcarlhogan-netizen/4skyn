"""
OpenClaw V7.5 — Position Engine

V7.5 changes over V7:
  - CycleDetector integrated: evaluate() + get_macro_composite() called on every decide_trade()
  - Hard MACRO_VETO gate: macro_composite < cfg["cycle"]["macro_veto_threshold"] → HOLD
  - Kelly notional multiplied by cycle_comp.kelly_multiplier (0.3× – 1.5×)
  - TradeSignal.reason now includes cycle=, macro=, kelly_mult= for all signals
  - Cycle config block in config.yaml controls thresholds

All original V7 logic preserved and unchanged:
  - RANGE maps to mean-revert vs midline (not blanket SELL)
  - Consecutive-loss circuit breaker
  - Daily loss kill switch
  - Closed-bar-only klines
  - OFI mainnet confirmation
  - Meta-labeler gate (Lopez de Prado AFML Ch. 3)
"""
from __future__ import annotations
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
import numpy as np
import pandas as pd
from pybit.unified_trading import HTTP

from core.regime_hmm import RegimeDetector
from core.kelly_sizer import KellySizer
from core.ofi_calculator import OFICalculator
from core.meta_labeler import MetaLabeler, MetaContext
from core.cycle_detector import CycleDetector


@dataclass
class TradeSignal:
    side: str                         # "Buy" | "Sell" | "HOLD" | "PAUSE"
    notional_usd: float
    sl_price: Optional[float]
    tp_price: Optional[float]
    regime: str
    confidence: float
    reason: str
    price: float
    meta_signal_id: Optional[str] = None
    meta_p_win: Optional[float] = None


class PositionEngine:

    def __init__(self, cfg: dict, testnet_orders: bool = True):
        self.cfg = cfg
        exec_cfg = cfg.get("execution", {})
        infra_cfg = cfg.get("infra", {})
        regime_cfg = cfg.get("regime", {})
        kelly_cfg = cfg.get("kelly", {})
        ofi_cfg = cfg.get("ofi", {})

        self.symbol = exec_cfg.get("symbol", "ETHUSDT")
        self.timeframe = exec_cfg.get("timeframe", "60")

        # Split clients: orders to testnet (during validation), data to mainnet
        self.http_orders = HTTP(testnet=testnet_orders)
        self.http_data   = HTTP(testnet=False)

        self.detector = RegimeDetector(
            model_path=infra_cfg.get("model_path", "models/hmm_btc_v7.pkl"),
            conf_threshold=regime_cfg.get("confidence_threshold", 0.75),
        )
        self.kelly = KellySizer(
            tradelog_path=infra_cfg.get("trade_log", "data/tradelog.json"),
            hwm_path=infra_cfg.get("hwm_path", "data/highwatermark.json"),
            c=kelly_cfg.get("fraction_c", 0.50),
            fmax=kelly_cfg.get("fmax", 0.08),
            prior_mu=kelly_cfg.get("prior_mu", 0.002),
            prior_sigma2=kelly_cfg.get("prior_sigma2", 0.0004),
            prior_weight=kelly_cfg.get("prior_weight", 30),
            drawdown_clamp_threshold=kelly_cfg.get("drawdown_clamp_threshold", 0.80),
        )
        self.ofi = OFICalculator(
            testnet=(not ofi_cfg.get("use_mainnet_orderbook", True)),
            window_size=ofi_cfg.get("window", 20),
        )

        ml_cfg = cfg.get("meta_labeler", {})
        self.meta_enabled = ml_cfg.get("enabled", False)
        self.meta = None
        if self.meta_enabled:
            self.meta = MetaLabeler(
                model_path=ml_cfg["model_path"],
                training_data_path=ml_cfg["training_data_path"],
                activation_threshold=ml_cfg["activation_threshold"],
                prob_threshold=ml_cfg["prob_threshold"],
                retrain_every=ml_cfg["retrain_every_n_trades"],
                rolling_window=ml_cfg["rolling_window_trades"],
            )

        # V7.5: CycleDetector — symbol follows execution symbol
        cycle_symbol = cfg.get("cycle", {}).get("bybit_symbol", self.symbol)
        self.cycle_detector = CycleDetector(bybit_symbol=cycle_symbol)
        self._macro_veto_threshold: float = cfg.get("cycle", {}).get(
            "macro_veto_threshold", -0.40
        )

        # Circuit breaker state
        self._consecutive_losses = 0
        self._circuit_open_until = 0.0
        self._daily_start_equity = None
        self._daily_start_day = None

    # ── market data ──────────────────────────────────────────────────────────

    def _get_klines(self, limit: int = 300) -> pd.DataFrame:
        resp = self.http_data.get_kline(
            category="linear", symbol=self.symbol,
            interval=self.timeframe, limit=limit,
        )
        df = pd.DataFrame(
            resp["result"]["list"],
            columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"],
        )
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = df[c].astype(float)
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms")
        df = df.sort_values("timestamp").reset_index(drop=True)
        # CLOSED-BAR ONLY — drop in-progress bar
        if self.cfg["execution"]["closed_bar_only"] and len(df) > 1:
            df = df.iloc[:-1].reset_index(drop=True)
        return df

    def _atr(self, df: pd.DataFrame, period: int = 14) -> float:
        h = df["high"]; l = df["low"]; c = df["close"].shift()
        tr = pd.concat([(h - l), (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
        return float(tr.rolling(period).mean().iloc[-1])

    def _extreme_vol(self, df: pd.DataFrame) -> bool:
        r20 = (df["high"].rolling(20).mean() - df["low"].rolling(20).mean()).iloc[-1]
        return self._atr(df) > 3.0 * float(r20)

    def _top_depth_usd(self) -> float:
        ob = self.http_data.get_orderbook(
            category="linear", symbol=self.symbol, limit=5,
        )
        bids = sum(float(x[1]) for x in ob["result"]["b"][:5])
        asks = sum(float(x[1]) for x in ob["result"]["a"][:5])
        px = float(ob["result"]["b"][0][0])
        return (bids + asks) * px * 0.5

    def get_current_position(self) -> dict:
        try:
            resp = self.http_orders.get_positions(
                category="linear", symbol=self.symbol,
            )
            if resp["retCode"] == 0 and resp["result"]["list"]:
                p = resp["result"]["list"][0]
                return {
                    "size":         float(p.get("size", 0) or 0),
                    "entry_price":  float(p.get("avgPrice", 0) or 0),
                    "realized_pnl": float(p.get("realisedPnl", 0) or 0),
                    "side":         p.get("side"),
                }
        except Exception:
            pass
        return {"size": 0.0, "entry_price": 0.0, "realized_pnl": 0.0, "side": None}

    # ── circuit breaker ──────────────────────────────────────────────────────

    def on_trade_close(self, pnl_pct: float):
        if pnl_pct < 0:
            self._consecutive_losses += 1
            risk_cfg = self.cfg.get("risk", {})
            limit = risk_cfg.get("consecutive_loss_circuit_breaker", 4)
            if self._consecutive_losses >= limit:
                self._circuit_open_until = time.time() + 24 * 3600
        else:
            self._consecutive_losses = 0

    def _circuit_tripped(self) -> bool:
        return time.time() < self._circuit_open_until

    def _daily_loss_tripped(self, equity: float) -> bool:
        today = datetime.now(timezone.utc).date()
        if self._daily_start_day != today:
            self._daily_start_day = today
            self._daily_start_equity = equity
            return False
        if self._daily_start_equity and self._daily_start_equity > 0:
            drop = (self._daily_start_equity - equity) / self._daily_start_equity
            risk_cfg = self.cfg.get("risk", {})
            return drop >= risk_cfg.get("daily_loss_stop_pct", 0.03)
        return False

    # ── main decision ────────────────────────────────────────────────────────

    def decide_trade(self, current_equity: float) -> TradeSignal:
        # 0. Hard guards
        if self._circuit_tripped():
            return TradeSignal("HOLD", 0, None, None, "CIRCUIT", 0,
                               "consecutive-loss circuit breaker", 0)
        if self._daily_loss_tripped(current_equity):
            return TradeSignal("HOLD", 0, None, None, "DAILY_STOP", 0,
                               "daily loss cap hit", 0)

        df = self._get_klines()
        if len(df) < 150:
            return TradeSignal("HOLD", 0, None, None, "NO_DATA", 0,
                               "insufficient klines", 0)

        price = float(df["close"].iloc[-1])

        # 1. Extreme vol pause
        if self._extreme_vol(df):
            return TradeSignal("PAUSE", 0, None, None, "EXTREME_VOL", 0,
                               "ATR14 > 3x ATR20", price)

        # 2. Regime gate
        regime, conf, proba = self.detector.predict_regime(df, self.symbol)
        if regime in ("CRISIS", "LOW_CONF", "INSUFFICIENT_DATA", "UNKNOWN"):
            return TradeSignal("HOLD", 0, None, None, regime, conf,
                               f"regime gate = {regime}", price)

        # 3. V7.5: Cycle composite + macro composite
        now = datetime.now(timezone.utc)
        cycle_comp = self.cycle_detector.evaluate(now=now)
        macro_score = cycle_comp.macro_composite  # already computed in evaluate()

        # V7.5: Hard macro veto — structural headwind from K-Wave+Juglar+Kitchin
        if macro_score < self._macro_veto_threshold:
            return TradeSignal(
                "HOLD", 0, None, None, "MACRO_VETO", 0,
                f"MACRO_VETO macro={macro_score:.3f} < {self._macro_veto_threshold} "
                f"(K-Wave+Juglar+Kitchin structural headwind)",
                price,
            )

        # 4. Direction
        regime_cfg = self.cfg["regimes"]
        action = regime_cfg[regime]["action"]
        direction = self._direction_for(regime, action, df)
        if direction == 0:
            return TradeSignal("HOLD", 0, None, None, regime, conf,
                               f"{action}: no directional edge", price)

        # 5. OFI confirmation
        ofi_rank = 50.0
        ofi_cfg = self.cfg.get("ofi", {})
        if ofi_cfg.get("enabled", True):
            ofi_val, ofi_rank = self.ofi.compute(self.symbol)
            threshold = ofi_cfg.get("percentile_threshold", 75)
            aligned = (
                (direction > 0 and ofi_rank >= threshold and ofi_val > 0)
                or (direction < 0 and ofi_rank <= (100 - threshold) and ofi_val < 0)
            )
            if not aligned:
                return TradeSignal("HOLD", 0, None, None, regime, conf,
                                   "OFI not aligned", price)

        # 6. Meta-labeler gate
        meta_signal_id = None
        meta_p_win = None
        if self.meta is not None:
            ctx = self._build_meta_context(proba, conf, ofi_rank, df, direction, current_equity)
            take, p_win = self.meta.should_take_trade(ctx)
            meta_p_win = p_win
            if not take:
                return TradeSignal(
                    "HOLD", 0, None, None, regime, conf,
                    f"meta-labeler blocked: P(win)={p_win:.2f} < {self.meta.prob_threshold}",
                    price, None, p_win,
                )
            meta_signal_id = self.meta.record_signal(ctx)

        # 7. Size — V7.5: multiply Kelly notional by cycle kelly_multiplier
        risk_cfg = self.cfg.get("risk", {})
        exec_cfg = self.cfg.get("execution", {})
        leverage = risk_cfg.get("max_leverage", 3)
        notional = self.kelly.get_notional(current_equity, regime, leverage)
        notional *= cycle_comp.kelly_multiplier   # 0.3× – 1.5× depending on cycle alignment
        depth_cap = exec_cfg.get("depth_haircut", 0.03) * self._top_depth_usd()
        notional = min(notional, depth_cap)

        min_notional = exec_cfg.get("min_notional_usd", 20)
        if notional < min_notional:
            return TradeSignal(
                "HOLD", 0, None, None, regime, conf,
                f"notional ${notional:.1f} below min "
                f"(after cycle mult {cycle_comp.kelly_multiplier:.2f})",
                price,
            )

        # 8. ATR stops
        atr = self._atr(df)
        sl_mult, tp_mult = self.detector.get_atr_multipliers(regime, regime_cfg)

        risk_cfg = self.cfg.get("risk", {})
        exec_cfg = self.cfg.get("execution", {})
        max_loss_usd = current_equity * risk_cfg.get("risk_per_trade_pct", 0.005)
        sl_distance_usd = notional * (sl_mult * atr / price)
        if sl_distance_usd > max_loss_usd:
            notional = max_loss_usd / (sl_mult * atr / price)
            if notional < exec_cfg.get("min_notional_usd", 20):
                return TradeSignal(
                    "HOLD", 0, None, None, regime, conf,
                    "shrunk notional below min to respect risk cap", price,
                )

        if direction > 0:
            side = "Buy";  sl = price - sl_mult * atr;  tp = price + tp_mult * atr
        else:
            side = "Sell"; sl = price + sl_mult * atr;  tp = price - tp_mult * atr

        reason = (
            f"all gates passed | cycle={cycle_comp.composite:+.3f} [{cycle_comp.sentiment}] "
            f"macro={macro_score:+.3f} kelly_mult={cycle_comp.kelly_multiplier:.2f}"
        )

        return TradeSignal(
            side, notional, sl, tp, regime, conf, reason, price,
            meta_signal_id=meta_signal_id, meta_p_win=meta_p_win,
        )

    # ── meta-labeler context builder ─────────────────────────────────────────

    def _build_meta_context(
        self, proba, conf, ofi_rank, df, direction, current_equity,
    ) -> "MetaContext":
        rv = df["close"].pct_change().tail(20).std()
        rv_100 = df["close"].pct_change().tail(100).std() or 1e-8
        rv_z = (rv - rv_100) / (rv_100 + 1e-8)

        atr = self._atr(df)
        atr_pct = atr / float(df["close"].iloc[-1])

        pnls = self.kelly.trade_pnls[-20:]
        wr = float(np.mean([1 if p > 0 else 0 for p in pnls])) if pnls else 0.5
        streak = 0
        if pnls:
            last = 1 if pnls[-1] > 0 else -1
            for p in reversed(pnls):
                if (p > 0 and last > 0) or (p <= 0 and last < 0):
                    streak += last
                else:
                    break

        hwm = max(self.kelly.high_water_mark, current_equity, 1.0)
        eq_ratio = current_equity / hwm

        try:
            f_df = self.detector._fetch_funding_series(self.symbol)
            funding = float(f_df["funding_live"].iloc[-1]) if not f_df.empty else 0.0001
        except Exception:
            funding = 0.0001

        return MetaContext(
            p_crisis=float(proba[0]),
            p_range=float(proba[1]),
            p_bull=float(proba[2]),
            hmm_conf=float(conf),
            ofi_rank=float(ofi_rank),
            funding_8h=funding,
            rv_zscore=float(rv_z),
            atr_pct=float(atr_pct),
            recent_win_rate=wr,
            streak=int(streak),
            equity_hwm_ratio=float(eq_ratio),
            direction=int(direction),
        )

    # ── direction resolver ───────────────────────────────────────────────────

    def _direction_for(self, regime: str, action: str, df: pd.DataFrame) -> int:
        """
        Map regime action to directional intent.
        All signals filtered through 200-period EMA trend gate.
        BULL: only long when price > EMA200.
        RANGE: mean-revert against band extreme, filtered by EMA200 direction.
        """
        c = df["close"]
        price = float(c.iloc[-1])
        ema200 = float(c.ewm(span=200, adjust=False).mean().iloc[-1])
        above_ema = price > ema200

        if action == "TREND_LONG":
            return 1 if above_ema else 0

        if action == "MEAN_REVERT":
            ma20 = c.rolling(20).mean().iloc[-1]
            sd20 = c.rolling(20).std().iloc[-1]
            upper = ma20 + 1.5 * sd20
            lower = ma20 - 1.5 * sd20
            if price > upper and above_ema:
                return -1
            if price < lower and not above_ema:
                return 1
            return 0
        return 0

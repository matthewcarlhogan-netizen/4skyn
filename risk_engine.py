"""
risk_engine.py  —  OpenClaw V6

Upgrades over V5:
  1. Dynamic fractional Kelly sizing from rolling realized PnL (single-asset)
  2. Multi-asset correlation-aware Kelly via Ledoit-Wolf shrunk covariance
  3. High-water mark drawdown clamp (halve c if equity < 80% HWM)
  4. Liquidity depth haircut on max notional
  5. Regime-gated ATR multipliers (SL/TP per regime from config)
  6. Equity-tier scaling (<$500 / $500-$2k / $2k-$10k / >$10k)
  7. Edge-decay monitor: win-rate + Deflated Sharpe gate (PSR)
  8. Persistent state: PnL history, HWM, trade counts, days live
"""

import json
import logging
import numpy as np
import scipy.stats
import yaml
from collections import deque
from pathlib import Path

logger = logging.getLogger(__name__)


class RiskEngine:
    def __init__(self, config_path='config.yaml'):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        r = self.cfg['risk']
        self.risk_pct_institutional = r['risk_per_trade']     # fallback for large accounts
        self.cold_trades  = r['cold_start_trades']
        self.cold_days    = r['cold_start_days']
        self.hwm_ratio    = r.get('hwm_clamp_ratio', 0.80)
        self.depth_h      = r.get('depth_haircut_h', 0.03)
        self.kelly_c      = r.get('kelly_c_default', 0.25)
        self.kelly_c_min  = r.get('kelly_c_min', 0.10)
        self.kelly_f_max  = r.get('kelly_f_max', 0.02)
        self.kelly_window = r.get('kelly_window', 100)
        self.shrink_delta = self.cfg['hmm'].get('covariance_shrinkage_delta', 0.50)
        self.leverage     = self.cfg['bybit']['leverage']

        self.trades_executed  = 0
        self.days_live        = 0
        self.daily_pnl        = 0.0
        self.killswitch       = False
        self.high_water_mark  = 0.0          # set on first balance read
        self.pnl_history      = deque(maxlen=500)   # trade-level R-multiples
        self.pnl_by_symbol    = {}                  # {symbol: deque(maxlen=200)}
        self.state_file       = Path('state/openclaw.json')
        self.load_state()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def load_state(self):
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    s = json.load(f)
                self.trades_executed = s.get('trades_executed', 0)
                self.days_live       = s.get('days_live', 0)
                self.high_water_mark = s.get('high_water_mark', 0.0)
                self.pnl_history     = deque(s.get('pnl_history', []), maxlen=500)
                pbs                  = s.get('pnl_by_symbol', {})
                self.pnl_by_symbol   = {k: deque(v, maxlen=200) for k, v in pbs.items()}
            except Exception as e:
                logger.warning(f'State load failed: {e}')

    def save_state(self):
        try:
            self.state_file.parent.mkdir(exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump({
                    'trades_executed':  self.trades_executed,
                    'days_live':        self.days_live,
                    'high_water_mark':  self.high_water_mark,
                    'pnl_history':      list(self.pnl_history),
                    'pnl_by_symbol':    {k: list(v) for k, v in self.pnl_by_symbol.items()}
                }, f, indent=2)
        except Exception as e:
            logger.warning(f'State save failed: {e}')

    def reset_daily(self):
        self.daily_pnl  = 0.0
        self.days_live += 1
        self.killswitch = False
        self.save_state()

    # ------------------------------------------------------------------
    # Record a closed trade PnL (call after a position closes)
    # pnl_r: realized PnL in R-multiples (profit / SL distance)
    # ------------------------------------------------------------------
    def record_trade(self, symbol: str, pnl_r: float, balance_after: float):
        self.pnl_history.append(pnl_r)
        self.daily_pnl += pnl_r
        if symbol not in self.pnl_by_symbol:
            self.pnl_by_symbol[symbol] = deque(maxlen=200)
        self.pnl_by_symbol[symbol].append(pnl_r)
        if balance_after > self.high_water_mark:
            self.high_water_mark = balance_after
        self.save_state()

    # ------------------------------------------------------------------
    # Kelly sizing — single asset (equation 1+2 from the research sweep)
    # Returns fractional Kelly f_t in [0, f_max]
    # ------------------------------------------------------------------
    def _kelly_f_single(self, symbol: str = None) -> float:
        hist = list(self.pnl_by_symbol.get(symbol, [])) if symbol else list(self.pnl_history)
        # Use last N trades
        hist = hist[-self.kelly_window:] if len(hist) > self.kelly_window else hist
        if len(hist) < 20:
            # Not enough history — use safe fallback based on equity tier
            return self.kelly_f_max * 0.5

        arr  = np.array(hist, dtype=float)
        mu   = arr.mean()
        sig2 = arr.var(ddof=1) + 1e-12

        f_star = mu / sig2           # Full Kelly
        f_star = max(f_star, 0.0)    # Never short Kelly

        # Current drawdown clamp
        c = self._get_kelly_c()

        f_t = c * min(f_star, self.kelly_f_max)
        return float(np.clip(f_t, 0.0, self.kelly_f_max))

    def _get_kelly_c(self) -> float:
        """Return the fractional Kelly multiplier, halved if in drawdown clamp."""
        return self.kelly_c

    def apply_hwm_clamp(self, equity: float) -> float:
        """
        Apply high-water-mark clamp.
        If equity < HWM * hwm_ratio → halve c temporarily.
        Returns clamped Kelly c to use for this cycle.
        """
        if self.high_water_mark <= 0:
            self.high_water_mark = equity
            return self.kelly_c
        ratio = equity / self.high_water_mark
        if ratio < self.hwm_ratio:
            clamped = max(self.kelly_c * 0.5, self.kelly_c_min)
            logger.warning(f"HWM CLAMP active: equity={equity:.2f}, HWM={self.high_water_mark:.2f}, ratio={ratio:.2%}, c → {clamped:.3f}")
            self.kelly_c = clamped
        else:
            self.kelly_c = self.cfg['risk'].get('kelly_c_default', 0.25)
        return self.kelly_c

    # ------------------------------------------------------------------
    # Multi-asset Kelly weight vector (equation 3 from research sweep)
    # Returns dict {symbol: weight} (gross weights sum to ~1.0 before leverage)
    # ------------------------------------------------------------------
    def multi_asset_kelly_weights(self, symbols: list) -> dict:
        """
        w* = lambda * Sigma_inv * g
        where:
          g[i]     = mean log-return per trade for symbol i
          Sigma    = Ledoit-Wolf shrunk covariance of per-trade returns
          lambda   = scale so sum(|w|) <= 1.0
        """
        # Build return matrix: rows=trades, cols=symbols
        histories = {}
        min_len   = 999999
        for sym in symbols:
            h = list(self.pnl_by_symbol.get(sym, []))
            if len(h) < 10:
                return {s: 1.0 / len(symbols) for s in symbols}  # fallback: equal weight
            histories[sym] = np.array(h, dtype=float)
            min_len = min(min_len, len(h))

        # Align lengths
        R = np.column_stack([histories[s][-min_len:] for s in symbols])

        # Edge vector
        g = R.mean(axis=0)

        # Ledoit-Wolf shrinkage toward diagonal:
        # Sigma_shrink = delta * diag(Sigma_sample) + (1-delta) * Sigma_sample
        Sigma_sample = np.cov(R.T, ddof=1) if R.shape[0] > 1 else np.eye(len(symbols))
        if Sigma_sample.ndim == 0:   # single asset edge case
            Sigma_sample = np.array([[float(Sigma_sample)]])
        delta       = self.shrink_delta
        F           = np.diag(np.diag(Sigma_sample))   # target: diagonal
        Sigma_shrunk = delta * F + (1 - delta) * Sigma_sample
        Sigma_shrunk += np.eye(len(symbols)) * 1e-10   # numerical stability

        try:
            Sigma_inv = np.linalg.inv(Sigma_shrunk)
        except np.linalg.LinAlgError:
            Sigma_inv = np.eye(len(symbols))

        w_raw = Sigma_inv @ g
        w_raw = np.clip(w_raw, 0, None)   # no short positions in perp long-only mode

        # Normalise so gross sum = 1.0
        total = w_raw.sum() + 1e-12
        w_norm = w_raw / total

        return {sym: float(w_norm[i]) for i, sym in enumerate(symbols)}

    # ------------------------------------------------------------------
    # Main position sizing function
    # Returns qty in base asset (e.g. BTC contracts) for a single symbol
    # ------------------------------------------------------------------
    def position_size(
        self,
        balance: float,
        atr: float,
        price: float,
        open_positions: int,
        symbol: str = None,
        regime: str = 'BULL',
        top5_depth_usd: float = 0.0,
        kelly_weight: float = 1.0,   # from multi_asset_kelly_weights
    ) -> float:
        """
        Position sizing pipeline:
          1. Equity-tier risk %
          2. Fractional Kelly f_t (rolling PnL)
          3. High-water-mark clamp
          4. Regime-gated ATR stop distance
          5. Depth haircut
          6. Cold-start halving
          7. Leverage + slippage buffer
        """
        if balance <= 0 or atr <= 0 or price <= 0:
            return 0.0

        # ---- 1. Equity tier ----
        if balance < 500:
            tier_risk = 0.01      # 1% per trade at micro-equity (capped; Kelly < this wins)
        elif balance < 2000:
            tier_risk = 0.02
        elif balance < 10000:
            tier_risk = 0.015
        else:
            tier_risk = self.risk_pct_institutional   # 1% institutional

        # ---- 2. Fractional Kelly ----
        f_t = self._kelly_f_single(symbol)

        # ---- 3. HWM clamp (modifies self.kelly_c for this tick) ----
        self.apply_hwm_clamp(balance)
        f_t = self._kelly_f_single(symbol)  # recalculate with updated c

        # ---- 4. Regime-gated ATR stop distance ----
        atr_cfg = self.cfg.get('atr', {})
        if regime == 'BULL':
            sl_mult = atr_cfg.get('trend_sl_mult', 3.0)
        else:  # RANGE / mean-reversion
            sl_mult = atr_cfg.get('range_sl_mult', 1.5)

        stop_distance = atr * sl_mult

        # ---- 5. Risk in USD via Kelly f_t (bounded by tier) ----
        effective_risk_pct = min(f_t, tier_risk)
        risk_usd           = balance * effective_risk_pct

        # Notional from stop: notional = risk_usd / stop_pct
        stop_pct  = stop_distance / price
        notional  = risk_usd / (stop_pct + 1e-12)

        # ---- 6. Depth haircut ----
        if top5_depth_usd > 0:
            max_from_depth = self.depth_h * top5_depth_usd
            notional       = min(notional, max_from_depth)

        # Apply Kelly weight for multi-asset correlation
        notional *= kelly_weight

        # ---- 7. Cold-start halving ----
        in_cold = (self.trades_executed < self.cold_trades or self.days_live < self.cold_days)
        if in_cold:
            notional *= 0.5

        # ---- 8. Leverage + slippage buffer ----
        notional_leveraged = notional * self.leverage
        # Hard cap: never commit more than 10% of leveraged equity per symbol
        max_notional = balance * self.leverage * 0.10
        notional_leveraged = min(notional_leveraged, max_notional)

        qty = notional_leveraged / price
        qty = max(qty * 0.97, 0.0)   # 3% slippage buffer
        return float(qty)

    # ------------------------------------------------------------------
    # Deflated Sharpe Ratio gate (PSR)
    # ------------------------------------------------------------------
    def check_sharpe_gate(self) -> bool:
        if self.trades_executed < self.cold_trades or self.days_live < self.cold_days:
            return True
        returns = np.array(self.pnl_history)
        if len(returns) < 50:
            return True
        std = returns.std()
        if std == 0:
            return False
        sr   = returns.mean() / std
        sk   = scipy.stats.skew(returns)
        ku   = scipy.stats.kurtosis(returns, fisher=False)
        denom = max(np.sqrt(1 - sk * sr + ((ku - 3) / 4) * sr ** 2), 1e-12)
        psr  = scipy.stats.norm.cdf(
            (sr - self.cfg['sharpe']['benchmark_sr']) /
            (denom / np.sqrt(len(returns) - 1))
        )
        passed = bool(psr >= self.cfg['sharpe']['gate_threshold'])
        if not passed:
            logger.warning(f'Sharpe gate FAILED: PSR={psr:.3f} < {self.cfg["sharpe"]["gate_threshold"]}')
        return passed

    # ------------------------------------------------------------------
    # Edge decay monitor
    # ------------------------------------------------------------------
    def check_edge_decay(self, window: int = 20) -> bool:
        """Returns True if edge looks healthy, False if degraded."""
        if len(self.pnl_history) < window:
            return True
        recent = list(self.pnl_history)[-window:]
        win_rate = sum(1 for p in recent if p > 0) / window
        avg_r    = np.mean(recent)
        if win_rate < 0.35:
            logger.warning(f'Edge decay: win_rate={win_rate:.2%} over last {window} trades')
            return False
        if avg_r < -0.5:
            logger.warning(f'Edge decay: avg_R={avg_r:.3f} over last {window} trades')
            return False
        return True

"""
OpenClaw V7 — Kelly sizer with Bayesian prior

Fixes V6's cold-start deadlock. Rather than returning 0 until 30 trades exist,
we shrink toward a conservative prior (μ=0.002, σ²=0.0004) weighted by
`prior_weight` pseudo-trades. As real trade history accumulates, the posterior
mean/variance converges to empirical values.

Reference: Lopez de Prado, "Advances in Financial Machine Learning" Ch. 10,
          shrinkage estimators for small-sample edge.
"""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
import numpy as np


class KellySizer:
    def __init__(
        self,
        tradelog_path: str,
        hwm_path: str,
        c: float = 0.25,
        fmax: float = 0.015,
        prior_mu: float = 0.002,
        prior_sigma2: float = 0.0004,
        prior_weight: int = 30,
        drawdown_clamp_threshold: float = 0.80,
    ):
        self.tradelog_path = Path(tradelog_path)
        self.hwm_path = Path(hwm_path)
        self.tradelog_path.parent.mkdir(parents=True, exist_ok=True)

        self.c = c
        self.fmax = fmax
        self.prior_mu = prior_mu
        self.prior_sigma2 = prior_sigma2
        self.prior_weight = prior_weight
        self.drawdown_clamp_threshold = drawdown_clamp_threshold

        self.trade_pnls: List[float] = []
        self.high_water_mark: float = 0.0
        self._load()

    def _load(self):
        if self.tradelog_path.exists():
            self.trade_pnls = json.loads(self.tradelog_path.read_text()).get("pnls", [])
        if self.hwm_path.exists():
            self.high_water_mark = json.loads(self.hwm_path.read_text()).get("hwm", 0.0)

    def _save(self):
        self.tradelog_path.write_text(
            json.dumps({"pnls": self.trade_pnls[-500:]})
        )
        self.hwm_path.write_text(
            json.dumps({"hwm": self.high_water_mark, "ts": datetime.utcnow().isoformat()})
        )

    def record_trade(self, pnl_pct: float, equity_before: float):
        self.trade_pnls.append(float(np.log(1.0 + pnl_pct)))
        if equity_before > self.high_water_mark:
            self.high_water_mark = equity_before
        self._save()

    # ── Bayesian posterior on μ, σ² ──────────────────────────────────────
    def posterior_moments(self) -> Tuple[float, float, int]:
        """Return (posterior_mu, posterior_sigma2, effective_n)."""
        n = len(self.trade_pnls)
        w = self.prior_weight
        if n == 0:
            return self.prior_mu, self.prior_sigma2, w

        recent = np.array(self.trade_pnls[-200:])  # cap at 200 for recency
        emp_mu = float(np.mean(recent))
        emp_var = float(np.var(recent, ddof=1)) if len(recent) > 1 else self.prior_sigma2

        # Shrinkage: posterior mean is weighted average
        post_mu = (w * self.prior_mu + n * emp_mu) / (w + n)
        post_sigma2 = (w * self.prior_sigma2 + n * emp_var) / (w + n)
        post_sigma2 = max(post_sigma2, 1e-6)
        return post_mu, post_sigma2, w + n

    def get_fraction(self, current_equity: float, regime: str) -> float:
        mu, sigma2, _ = self.posterior_moments()
        # Kelly: f* = μ/σ² (for log returns)
        f_star = mu / sigma2

        # Regime-conditional haircut
        regime_mult = {"BULL": 1.0, "RANGE": 0.6, "CRISIS": 0.0, "LOW_CONF": 0.0}.get(
            regime, 0.5
        )

        ft = self.c * max(min(f_star, self.fmax), 0.0) * regime_mult

        # Drawdown clamp
        if (
            self.high_water_mark > 0
            and current_equity < self.drawdown_clamp_threshold * self.high_water_mark
        ):
            ft *= 0.5

        return float(np.clip(ft, 0.0, 0.25))

    def get_notional(
        self, current_equity: float, regime: str, leverage: int
    ) -> float:
        ft = self.get_fraction(current_equity, regime)
        return ft * current_equity * leverage

    def diagnostics(self) -> dict:
        mu, sigma2, n_eff = self.posterior_moments()
        return {
            "n_real_trades": len(self.trade_pnls),
            "posterior_mu": round(mu, 6),
            "posterior_sigma2": round(sigma2, 6),
            "effective_n": n_eff,
            "implied_f_star": round(mu / sigma2, 4) if sigma2 > 0 else 0.0,
            "hwm": round(self.high_water_mark, 2),
        }

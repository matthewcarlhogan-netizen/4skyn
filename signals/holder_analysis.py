"""
Holder wallet age distribution and concentration analysis.
Detects bot wallet clusters, real user adoption, and wealth concentration.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HolderSnapshot:
    """Snapshot of a holder's position."""
    wallet_address: str
    balance: float  # tokens held
    balance_usd: float
    wallet_age_days: int
    first_transaction_time: datetime


@dataclass
class HolderScore:
    """Composite holder health score."""

    # Raw metrics
    gini_coefficient: float  # 0-1: wealth concentration (0=equal, 1=one holder)
    max_wallet_pct: float  # Largest holder % of circulating
    pct_wallets_new_7days: float  # % created <7 days ago (bot proxy)
    pct_wallets_old_90days: float  # % created >90 days ago (real users)
    holder_count_growth_rate: float  # New holders per hour
    buy_sell_ratio: float  # Buy txs / sell txs across all holders

    # Derived
    composite_score: float = field(init=False)  # 0-100, higher is better
    bot_probability: float = field(init=False)  # 0-1

    def __post_init__(self):
        """Compute composite score and bot probability."""
        # Gini score: lower (more distributed) is better
        gini_score = max(0, 50 - self.gini_coefficient * 50)

        # Max wallet: <5% ideal, >15% is red flag
        if self.max_wallet_pct < 5:
            max_wallet_score = 25
        elif self.max_wallet_pct < 10:
            max_wallet_score = 15
        elif self.max_wallet_pct < 15:
            max_wallet_score = 5
        else:
            max_wallet_score = -20

        # New wallets: many new wallets (especially <7d) = high bot probability
        new_wallet_penalty = -self.pct_wallets_new_7days * 50

        # Old wallets: >50% established users = +25 pts
        old_wallet_score = min(25, self.pct_wallets_old_90days * 25)

        # Holder growth: moderate growth = +10, explosive = -10
        if 0.5 < self.holder_count_growth_rate < 5:
            growth_score = 10
        elif self.holder_count_growth_rate > 10:
            growth_score = -10
        else:
            growth_score = 0

        # Buy/sell ratio: >1 (more buys than sells) = +10
        if self.buy_sell_ratio > 1.0:
            bs_score = min(10, (self.buy_sell_ratio - 1) * 10)
        else:
            bs_score = -5

        raw_score = gini_score + max_wallet_score + new_wallet_penalty + old_wallet_score + growth_score + bs_score
        self.composite_score = max(0, min(100, raw_score))

        # Bot probability: high % of new wallets, high concentration, poor B/S ratio
        bot_prob = 0.0
        bot_prob += self.pct_wallets_new_7days * 0.4
        bot_prob += max(0, (self.gini_coefficient - 0.5) * 0.3)  # High concentration
        bot_prob += max(0, (self.max_wallet_pct - 0.15) * 0.3)  # Large whale

        if self.buy_sell_ratio < 0.8:
            bot_prob += 0.2  # More sells than buys = suspicious

        self.bot_probability = min(0.95, max(0.0, bot_prob))


class HolderAnalyzer:
    """Analyzes holder distribution and wallet patterns."""

    def __init__(self):
        pass

    def analyze(self, token_address: str, holder_list: List[HolderSnapshot]) -> HolderScore:
        """
        Analyze holder composition and distribution health.

        Args:
            token_address: Token address (for logging)
            holder_list: List of holders and their wallet details

        Returns:
            HolderScore with concentration metrics and bot probability
        """
        if not holder_list:
            logger.warning(f"No holders found for {token_address}")
            return HolderScore(
                gini_coefficient=1.0,
                max_wallet_pct=0.0,
                pct_wallets_new_7days=0.0,
                pct_wallets_old_90days=0.0,
                holder_count_growth_rate=0.0,
                buy_sell_ratio=0.0
            )

        logger.info(f"Analyzing {len(holder_list)} holders for {token_address}")

        # Compute Gini coefficient (wealth concentration)
        gini = self._compute_gini(holder_list)

        # Max wallet percentage
        max_balance = max(h.balance for h in holder_list) if holder_list else 0
        total_balance = sum(h.balance for h in holder_list) if holder_list else 1
        max_pct = (max_balance / total_balance) if total_balance > 0 else 0

        # Wallet age distribution
        now = datetime.utcnow()
        new_wallets = sum(1 for h in holder_list if h.wallet_age_days < 7)
        old_wallets = sum(1 for h in holder_list if h.wallet_age_days > 90)

        pct_new_7d = new_wallets / len(holder_list) if holder_list else 0
        pct_old_90d = old_wallets / len(holder_list) if holder_list else 0

        # Holder growth rate (approximated from snapshot count observation)
        # In production: track historical snapshots
        growth_rate = len(holder_list) / 24.0  # Approximate holders per hour

        # Buy/sell ratio: inferred from holder ages and balance changes
        # In production: query transaction history
        bs_ratio = self._estimate_buy_sell_ratio(holder_list)

        logger.info(
            f"Token {token_address}: gini={gini:.3f}, max_pct={max_pct:.2%}, "
            f"new_7d={pct_new_7d:.2%}, old_90d={pct_old_90d:.2%}, "
            f"growth_rate={growth_rate:.1f}/h, bs_ratio={bs_ratio:.2f}"
        )

        return HolderScore(
            gini_coefficient=gini,
            max_wallet_pct=max_pct,
            pct_wallets_new_7days=pct_new_7d,
            pct_wallets_old_90days=pct_old_90d,
            holder_count_growth_rate=growth_rate,
            buy_sell_ratio=bs_ratio
        )

    def _compute_gini(self, holders: List[HolderSnapshot]) -> float:
        """
        Compute Gini coefficient for wealth concentration.
        0 = perfect equality, 1 = perfect inequality (one holder has all)
        """
        if not holders or len(holders) < 2:
            return 1.0

        balances = np.array([h.balance for h in holders])
        balances = balances[balances > 0]  # Remove zero balances

        if len(balances) == 0:
            return 1.0

        # Sort ascending
        balances_sorted = np.sort(balances)
        n = len(balances_sorted)

        # Gini formula: sum of (2 * rank) / n * (n+1) - 1
        cumsum = np.cumsum(balances_sorted)
        gini = (2 * np.sum(np.arange(1, n + 1) * balances_sorted)) / (n * np.sum(balances_sorted)) - (n + 1) / n

        return max(0.0, min(1.0, gini))

    def _estimate_buy_sell_ratio(self, holders: List[HolderSnapshot]) -> float:
        """
        Estimate buy/sell ratio from holder wallet ages.
        Newer wallet ages suggest recent buys. Old wallet ages with small balances suggest sells.
        This is a heuristic; real implementation queries tx history.
        """
        if not holders:
            return 0.0

        # Count "buy-like" holders (young wallets, low age variance in recent holders)
        recent_holders = [h for h in holders if h.wallet_age_days < 1]
        established_holders = [h for h in holders if h.wallet_age_days >= 30]

        if len(recent_holders) == 0 and len(established_holders) == 0:
            return 1.0

        # Heuristic: more recent = more buys happening
        buy_score = len(recent_holders)
        sell_score = max(1, len(established_holders) / 3)  # Established holders less likely to sell aggressively

        return buy_score / max(1, sell_score)

    def get_concentration_risk_level(self, score: HolderScore) -> str:
        """
        Classify concentration risk based on holder score.

        Returns:
            "low", "medium", "high", or "critical"
        """
        if score.max_wallet_pct > 0.30:
            return "critical"
        elif score.max_wallet_pct > 0.15 or score.gini_coefficient > 0.7:
            return "high"
        elif score.max_wallet_pct > 0.10 or score.gini_coefficient > 0.5:
            return "medium"
        else:
            return "low"

    def get_bot_risk_level(self, score: HolderScore) -> str:
        """
        Classify bot risk based on holder score.

        Returns:
            "low", "medium", "high", or "critical"
        """
        if score.bot_probability > 0.7:
            return "critical"
        elif score.bot_probability > 0.5:
            return "high"
        elif score.bot_probability > 0.3:
            return "medium"
        else:
            return "low"

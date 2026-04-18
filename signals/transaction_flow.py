"""
First-N transaction buy/sell pattern analyzer.
Detects organic growth vs bot-driven pump patterns.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class TransactionType(Enum):
    """Transaction type classification."""
    BUY = "buy"
    SELL = "sell"
    UNKNOWN = "unknown"


@dataclass
class Transaction:
    """On-chain transaction record."""
    tx_hash: str
    timestamp: datetime
    tx_type: TransactionType
    from_wallet: str
    to_wallet: str
    amount: float  # tokens
    price: float  # SOL per token
    liquidity_added: float  # SOL (for swaps via bonding curve)


@dataclass
class FlowScore:
    """Transaction flow pattern health score."""

    # Raw metrics
    buy_sell_ratio_first_100: float  # Buy count / sell count in first 100 txs
    buy_sell_ratio_first_500: float  # Buy count / sell count in first 500 txs
    avg_buy_size: float  # Average token amount per buy
    avg_sell_size: float  # Average token amount per sell
    buy_interval_variance: float  # Variance in time between buys (high = organic)
    unique_buyer_count: int
    whale_entry_detected: bool  # Single tx >5% of liquidity
    whale_entry_size_pct: float  # Size of largest single transaction as % of liquidity

    # Derived
    organic_score: float = field(init=False)  # 0-100

    def __post_init__(self):
        """Compute organic growth score."""
        # B/S ratio: >1 is healthy (more buys)
        ratio_100 = min(self.buy_sell_ratio_first_100, 3.0)  # Cap at 3x
        ratio_score_100 = max(-20, (ratio_100 - 1) * 20)

        # Large buy/sell size disparity: organic = similar sizes
        size_ratio = self.avg_buy_size / max(0.01, self.avg_sell_size)
        if 0.8 < size_ratio < 1.3:
            size_score = 15  # Balanced
        elif 0.5 < size_ratio < 2.0:
            size_score = 5   # Somewhat balanced
        else:
            size_score = -15  # Imbalanced (pump pattern)

        # Buy interval variance: high variance = organic, low variance = bot
        if self.buy_interval_variance > 30:
            interval_score = 20
        elif self.buy_interval_variance > 10:
            interval_score = 10
        elif self.buy_interval_variance > 1:
            interval_score = 0
        else:
            interval_score = -25  # Very regular = bot

        # Whale detection: large single transaction = warning
        whale_penalty = 0
        if self.whale_entry_detected:
            whale_penalty = -max(10, self.whale_entry_size_pct * 30)

        # Unique buyers: more unique = healthier
        unique_buyer_score = min(15, (self.unique_buyer_count / 50) * 15)

        raw_score = ratio_score_100 + size_score + interval_score + whale_penalty + unique_buyer_score
        self.organic_score = max(0, min(100, raw_score))

        logger.debug(
            f"FlowScore computed: bs_ratio={self.buy_sell_ratio_first_100:.2f}, "
            f"interval_var={self.buy_interval_variance:.1f}, organic={self.organic_score:.1f}"
        )


class TransactionFlowAnalyzer:
    """Analyzes early transaction patterns for organic vs bot behavior."""

    def __init__(self):
        pass

    def analyze(self, token_address: str, first_txs: List[Transaction]) -> FlowScore:
        """
        Analyze first-N transactions for organic growth signals.

        Args:
            token_address: Token address (for logging)
            first_txs: Chronologically ordered list of transactions

        Returns:
            FlowScore with organic assessment
        """
        if not first_txs:
            logger.warning(f"No transactions for {token_address}")
            return FlowScore(
                buy_sell_ratio_first_100=0.0,
                buy_sell_ratio_first_500=0.0,
                avg_buy_size=0.0,
                avg_sell_size=0.0,
                buy_interval_variance=0.0,
                unique_buyer_count=0,
                whale_entry_detected=False,
                whale_entry_size_pct=0.0
            )

        logger.info(f"Analyzing {len(first_txs)} transactions for {token_address}")

        # Filter to first 500 max for analysis
        txs_500 = first_txs[:500]
        txs_100 = first_txs[:100]

        # Compute B/S ratios
        bs_100 = self._compute_buy_sell_ratio(txs_100)
        bs_500 = self._compute_buy_sell_ratio(txs_500)

        # Compute average sizes
        avg_buy, avg_sell = self._compute_avg_sizes(txs_500)

        # Compute buy interval variance
        interval_var = self._compute_buy_interval_variance(txs_500)

        # Count unique buyers
        unique_buyers = self._count_unique_buyers(txs_500)

        # Detect whale entry
        whale_detected, whale_size_pct = self._detect_whale_entry(first_txs)

        logger.info(
            f"Token {token_address}: bs_100={bs_100:.2f}, bs_500={bs_500:.2f}, "
            f"avg_buy={avg_buy:.1f}, avg_sell={avg_sell:.1f}, "
            f"interval_var={interval_var:.1f}, unique_buyers={unique_buyers}, "
            f"whale_detected={whale_detected}"
        )

        return FlowScore(
            buy_sell_ratio_first_100=bs_100,
            buy_sell_ratio_first_500=bs_500,
            avg_buy_size=avg_buy,
            avg_sell_size=avg_sell,
            buy_interval_variance=interval_var,
            unique_buyer_count=unique_buyers,
            whale_entry_detected=whale_detected,
            whale_entry_size_pct=whale_size_pct
        )

    def _compute_buy_sell_ratio(self, txs: List[Transaction]) -> float:
        """Compute buy/sell transaction count ratio."""
        if not txs:
            return 0.0

        buy_count = sum(1 for tx in txs if tx.tx_type == TransactionType.BUY)
        sell_count = sum(1 for tx in txs if tx.tx_type == TransactionType.SELL)

        if sell_count == 0:
            return float('inf') if buy_count > 0 else 0.0

        return buy_count / sell_count

    def _compute_avg_sizes(self, txs: List[Transaction]) -> tuple:
        """Compute average buy and sell transaction sizes."""
        buy_amounts = [tx.amount for tx in txs if tx.tx_type == TransactionType.BUY]
        sell_amounts = [tx.amount for tx in txs if tx.tx_type == TransactionType.SELL]

        avg_buy = np.mean(buy_amounts) if buy_amounts else 0.0
        avg_sell = np.mean(sell_amounts) if sell_amounts else 0.0

        return avg_buy, avg_sell

    def _compute_buy_interval_variance(self, txs: List[Transaction]) -> float:
        """
        Compute variance in time between buy transactions.
        High variance = organic (variable buy timing)
        Low variance = bot (regular intervals)
        """
        buy_txs = [tx for tx in txs if tx.tx_type == TransactionType.BUY]

        if len(buy_txs) < 3:
            return 0.0

        # Sort by timestamp
        buy_txs = sorted(buy_txs, key=lambda t: t.timestamp)

        # Compute time deltas in seconds
        intervals = []
        for i in range(1, len(buy_txs)):
            delta = (buy_txs[i].timestamp - buy_txs[i-1].timestamp).total_seconds()
            if delta >= 0:
                intervals.append(delta)

        if not intervals:
            return 0.0

        # Return coefficient of variation (std / mean)
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)

        if mean_interval == 0:
            return 0.0

        # Coefficient of variation in percentage
        return (std_interval / mean_interval) * 100

    def _count_unique_buyers(self, txs: List[Transaction]) -> int:
        """Count unique wallet addresses making buys."""
        buyers = set()
        for tx in txs:
            if tx.tx_type == TransactionType.BUY:
                buyers.add(tx.from_wallet)

        return len(buyers)

    def _detect_whale_entry(self, txs: List[Transaction]) -> tuple:
        """
        Detect single large transaction >5% of total liquidity.

        Returns:
            Tuple of (whale_detected, whale_size_pct)
        """
        if not txs:
            return False, 0.0

        total_liquidity = sum(tx.liquidity_added for tx in txs)

        if total_liquidity <= 0:
            return False, 0.0

        # Find largest single transaction
        max_tx = max(txs, key=lambda t: t.liquidity_added)
        max_size_pct = max_tx.liquidity_added / total_liquidity

        whale_detected = max_size_pct > 0.05  # >5% threshold

        return whale_detected, max_size_pct

    def get_pump_risk_level(self, score: FlowScore) -> str:
        """
        Classify pump/dump risk based on transaction patterns.

        Returns:
            "low", "medium", "high", or "critical"
        """
        if score.organic_score < 20:
            return "critical"
        elif score.organic_score < 40:
            return "high"
        elif score.organic_score < 60:
            return "medium"
        else:
            return "low"

    def detect_bot_pump_pattern(self, score: FlowScore) -> bool:
        """
        Detect classic bot pump pattern:
        - Very high B/S ratio (artificial buying)
        - Low interval variance (regular buy timing)
        - Large buy/sell size mismatch
        """
        return (
            score.buy_sell_ratio_first_100 > 2.5
            and score.buy_interval_variance < 5
            and (score.avg_buy_size / max(0.01, score.avg_sell_size)) > 2.0
        )

    def detect_rug_prep_pattern(self, score: FlowScore) -> bool:
        """
        Detect rug pull preparation:
        - High sells relative to buys in later window
        - Whales accumulating position
        - Declining unique buyers
        """
        return (
            score.whale_entry_detected
            and score.whale_entry_size_pct > 0.10
            and score.buy_sell_ratio_first_500 < 1.5
        )

"""
Bonding curve velocity profile analyzer.
Detects curve shape patterns and computes velocity metrics for graduation prediction.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List
import numpy as np

logger = logging.getLogger(__name__)


class CurveShape(Enum):
    """Classification of bonding curve shape."""
    ORGANIC = "organic"  # Exponential early, deceleration later
    BOT_PUMPED = "bot_pumped"  # Linear ramp then cliff
    SLOW_BLEED = "slow_bleed"  # Dying curve
    WHALE_SPIKE = "whale_spike"  # Single large buy
    UNKNOWN = "unknown"


@dataclass
class CurveSnapshot:
    """Snapshot of bonding curve state at a point in time."""
    timestamp: datetime
    market_cap: float  # SOL or USD
    liquidity_pool_size: float
    total_supply: float
    holder_count: int
    transaction_count: int


@dataclass
class CurveScore:
    """Composite bonding curve risk/opportunity score."""

    # Raw metrics
    shape_type: CurveShape
    velocity_first_10min: float  # Market cap increase per minute
    velocity_10_30min: float  # Market cap increase per minute
    velocity_ratio: float  # early velocity / late velocity (organic > 1)
    percent_completion_per_hour: float  # % of curve completed (0-100)
    estimated_graduation_hours: float  # ETA to likely graduation threshold

    # Derived
    velocity_score: float = field(init=False)  # 0-100
    graduation_probability: float = field(init=False)  # 0-1

    def __post_init__(self):
        """Compute composite metrics."""
        # Velocity score: higher ratio = more organic (better)
        # penalize extremely high absolute velocities (bot pump indicator)
        ratio_score = min(40, (self.velocity_ratio - 1) * 40)  # ratio > 1 is good
        velocity_magnitude = min(10, self.velocity_first_10min / 1000)  # cap at 1000 SOL/min
        magnitude_penalty = max(-30, -velocity_magnitude * 3) if self.velocity_first_10min > 500 else 0

        shape_bonuses = {
            CurveShape.ORGANIC: 25,
            CurveShape.BOT_PUMPED: -30,
            CurveShape.WHALE_SPIKE: -20,
            CurveShape.SLOW_BLEED: -15,
            CurveShape.UNKNOWN: -10
        }
        shape_score = shape_bonuses.get(self.shape_type, 0)

        raw_velocity = max(0, ratio_score + magnitude_penalty + shape_score)
        self.velocity_score = min(100, raw_velocity)

        # Graduation probability: based on velocity, completion %, and shape
        # Organic curve with good velocity = high graduation probability
        base_prob = 0.2  # baseline

        if self.shape_type == CurveShape.ORGANIC:
            base_prob += 0.4

        if self.velocity_ratio > 1.5:  # Strong deceleration = healthy
            base_prob += 0.15

        if 20 < self.percent_completion_per_hour < 80:  # Not too fast, not too slow
            base_prob += 0.15

        if self.estimated_graduation_hours > 6:  # Still time to grow
            base_prob += 0.1

        self.graduation_probability = min(0.95, max(0.0, base_prob))

        logger.debug(
            f"CurveScore computed: shape={self.shape_type.value}, "
            f"velocity={self.velocity_score:.1f}, grad_prob={self.graduation_probability:.2%}"
        )


class BondingCurveAnalyzer:
    """Analyzes bonding curve behavior and shape."""

    # Graduation thresholds
    MIN_MARKET_CAP_GRADUATION = 50_000  # SOL
    MIN_HOLDER_GRADUATION = 1000
    MIN_TRANSACTION_GRADUATION = 5000

    def __init__(self):
        pass

    def analyze(self, token_address: str, curve_snapshots: List[CurveSnapshot]) -> CurveScore:
        """
        Analyze bonding curve shape and velocity.

        Args:
            token_address: Token address (for logging)
            curve_snapshots: Ordered list of curve snapshots (chronological)

        Returns:
            CurveScore with shape classification and metrics
        """
        if not curve_snapshots or len(curve_snapshots) < 2:
            logger.warning(f"Insufficient snapshots for {token_address}")
            return CurveScore(
                shape_type=CurveShape.UNKNOWN,
                velocity_first_10min=0.0,
                velocity_10_30min=0.0,
                velocity_ratio=0.0,
                percent_completion_per_hour=0.0,
                estimated_graduation_hours=float('inf')
            )

        snapshots = sorted(curve_snapshots, key=lambda s: s.timestamp)
        time_range_min = (snapshots[-1].timestamp - snapshots[0].timestamp).total_seconds() / 60

        # Compute velocity windows
        first_10min_snapshots = [s for s in snapshots
                               if (s.timestamp - snapshots[0].timestamp).total_seconds() <= 600]
        mid_window_snapshots = [s for s in snapshots
                              if 600 < (s.timestamp - snapshots[0].timestamp).total_seconds() <= 1800]

        velocity_0_10 = self._compute_velocity(first_10min_snapshots) if len(first_10min_snapshots) >= 2 else 0.0
        velocity_10_30 = self._compute_velocity(mid_window_snapshots) if len(mid_window_snapshots) >= 2 else velocity_0_10 * 0.5

        velocity_ratio = velocity_0_10 / max(0.01, velocity_10_30)

        # Detect shape
        shape = self._classify_shape(snapshots)

        # Compute % completion per hour
        percent_per_hour = self._compute_completion_rate(snapshots, time_range_min)

        # Estimate graduation time
        graduation_eta = self._estimate_graduation_time(snapshots, shape, velocity_0_10)

        logger.info(
            f"Token {token_address}: shape={shape.value}, "
            f"v0-10={velocity_0_10:.1f} SOL/min, v10-30={velocity_10_30:.1f} SOL/min, "
            f"ratio={velocity_ratio:.2f}, grad_eta={graduation_eta:.1f}h"
        )

        return CurveScore(
            shape_type=shape,
            velocity_first_10min=velocity_0_10,
            velocity_10_30min=velocity_10_30,
            velocity_ratio=velocity_ratio,
            percent_completion_per_hour=percent_per_hour,
            estimated_graduation_hours=graduation_eta
        )

    def _compute_velocity(self, snapshots: List[CurveSnapshot]) -> float:
        """Compute market cap increase per minute."""
        if len(snapshots) < 2:
            return 0.0

        sorted_snaps = sorted(snapshots, key=lambda s: s.timestamp)
        first, last = sorted_snaps[0], sorted_snaps[-1]
        time_delta_min = (last.timestamp - first.timestamp).total_seconds() / 60

        if time_delta_min <= 0:
            return 0.0

        cap_delta = last.market_cap - first.market_cap
        return cap_delta / time_delta_min

    def _classify_shape(self, snapshots: List[CurveSnapshot]) -> CurveShape:
        """Classify curve shape as organic, bot_pumped, slow_bleed, or whale_spike."""
        if len(snapshots) < 3:
            return CurveShape.UNKNOWN

        # Extract market cap and compute differences
        caps = np.array([s.market_cap for s in sorted(snapshots, key=lambda s: s.timestamp)])
        diffs = np.diff(caps)

        if len(diffs) == 0:
            return CurveShape.UNKNOWN

        # Compute velocity changes (second derivative)
        if len(diffs) > 1:
            accel = np.diff(diffs)
            avg_accel = np.mean(accel) if len(accel) > 0 else 0
            accel_std = np.std(accel) if len(accel) > 0 else 0

            # Organic: consistently decelerating (negative acceleration)
            if avg_accel < -100 and accel_std < 500:
                return CurveShape.ORGANIC

            # Whale spike: single large jump followed by no growth
            if caps[-1] - caps[0] > 0 and diffs[0] > diffs[-1] * 3:
                return CurveShape.WHALE_SPIKE

        # Bot pumped: linear then cliff (velocity constant then drops)
        early_vel = np.mean(diffs[:len(diffs)//2]) if len(diffs) > 2 else diffs[0]
        late_vel = np.mean(diffs[len(diffs)//2:]) if len(diffs) > 2 else diffs[-1]

        if early_vel > 100 and late_vel / max(1, early_vel) < 0.2:
            return CurveShape.BOT_PUMPED

        # Slow bleed: velocity declining throughout
        if diffs[-1] < diffs[0] * 0.3 and late_vel < 50:
            return CurveShape.SLOW_BLEED

        return CurveShape.UNKNOWN

    def _compute_completion_rate(self, snapshots: List[CurveSnapshot], time_range_min: float) -> float:
        """Compute % of curve completed per hour."""
        if time_range_min <= 0 or len(snapshots) < 2:
            return 0.0

        sorted_snaps = sorted(snapshots, key=lambda s: s.timestamp)
        first, last = sorted_snaps[0], sorted_snaps[-1]

        # Simple heuristic: holder growth as % of token supply
        holder_growth = last.holder_count - first.holder_count
        holder_growth_rate = (holder_growth / max(1, first.holder_count)) * 100 if first.holder_count > 0 else 0

        # Normalize to hourly rate
        hours = time_range_min / 60
        if hours <= 0:
            return 0.0

        return min(100, holder_growth_rate / hours)

    def _estimate_graduation_time(self, snapshots: List[CurveSnapshot], shape: CurveShape,
                                 velocity: float) -> float:
        """Estimate hours until graduation threshold."""
        if len(snapshots) < 1:
            return float('inf')

        sorted_snaps = sorted(snapshots, key=lambda s: s.timestamp)
        current_state = sorted_snaps[-1]

        # Graduation thresholds
        needs_mcap = max(0, self.MIN_MARKET_CAP_GRADUATION - current_state.market_cap)
        needs_holders = max(0, self.MIN_HOLDER_GRADUATION - current_state.holder_count)

        # Estimate time to graduation
        if velocity > 0:
            hours_to_mcap = needs_mcap / (velocity * 60) if needs_mcap > 0 else 0
        else:
            hours_to_mcap = float('inf')

        # Holder growth rate (per snapshot interval)
        if len(sorted_snaps) >= 2:
            time_between = (sorted_snaps[-1].timestamp - sorted_snaps[-2].timestamp).total_seconds() / 3600
            if time_between > 0:
                holders_per_hour = (current_state.holder_count - sorted_snaps[-2].holder_count) / time_between
                hours_to_holders = needs_holders / max(1, holders_per_hour) if needs_holders > 0 else 0
            else:
                hours_to_holders = float('inf')
        else:
            hours_to_holders = float('inf')

        # Return maximum (both thresholds must be met)
        eta = max(hours_to_mcap, hours_to_holders, 0.5)

        # Penalize slow_bleed and whale_spike shapes
        if shape == CurveShape.SLOW_BLEED:
            eta *= 2.0
        elif shape == CurveShape.WHALE_SPIKE:
            eta *= 1.5

        return min(float('inf'), eta)

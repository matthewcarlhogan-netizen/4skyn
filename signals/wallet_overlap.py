"""
Cross-token wallet overlap detector.
Identifies orchestrated pump schemes via wallet clustering across tokens.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Set, List, Dict
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class OverlapScore:
    """Orchestrated pump detection score."""

    # Raw metrics
    overlap_pct: float  # % of top holders that appear in recent failed tokens
    linked_rugs_count: int  # Number of rugged tokens with overlap
    known_bot_cluster_overlap: float  # % of wallets in known bot clusters

    # Derived
    orchestration_probability: float = field(init=False)  # 0-1

    def __post_init__(self):
        """Compute orchestration probability."""
        overlap_score = self.overlap_pct * 0.5
        rug_score = min(0.3, self.linked_rugs_count * 0.05)  # Each rug adds 5%
        bot_score = self.known_bot_cluster_overlap * 0.2

        raw_prob = overlap_score + rug_score + bot_score
        self.orchestration_probability = min(0.95, max(0.0, raw_prob))


class WalletOverlapDetector:
    """Detects coordinated trading via wallet overlap across tokens."""

    def __init__(self, lookback_hours: int = 72):
        """
        Initialize detector with historical token tracking.

        Args:
            lookback_hours: How far back to track failed/rugged tokens
        """
        self.lookback_hours = lookback_hours

        # Cache of recent token holder sets
        self._token_holder_cache: Dict[str, Set[str]] = {}

        # Cache of recent failed tokens with timestamps
        self._failed_tokens: Dict[str, datetime] = {}

        # Known bot wallet clusters (hard-coded patterns)
        self._bot_clusters = self._load_bot_clusters()

    async def check_overlap(self, token_address: str, top_holders: List[str]) -> OverlapScore:
        """
        Check for wallet overlap with failed tokens and bot clusters.

        Args:
            token_address: Current token address
            top_holders: Top holder wallet addresses (typically top 20)

        Returns:
            OverlapScore with orchestration probability
        """
        logger.info(f"Checking overlap for {token_address} with {len(top_holders)} top holders")

        top_holders_set = set(top_holders)

        # Get overlap with recent failed tokens
        failed_overlap_pct, linked_rugs = await self._compute_failed_token_overlap(top_holders_set)

        # Check against known bot clusters
        bot_overlap_pct = self._check_bot_cluster_overlap(top_holders_set)

        logger.info(
            f"Token {token_address}: failed_overlap={failed_overlap_pct:.2%}, "
            f"linked_rugs={linked_rugs}, bot_overlap={bot_overlap_pct:.2%}"
        )

        return OverlapScore(
            overlap_pct=failed_overlap_pct,
            linked_rugs_count=linked_rugs,
            known_bot_cluster_overlap=bot_overlap_pct
        )

    async def register_failed_token(self, token_address: str):
        """
        Register a token as failed/rugged for future overlap detection.

        Args:
            token_address: Address of the failed token
        """
        self._failed_tokens[token_address] = datetime.utcnow()
        logger.info(f"Registered failed token: {token_address}")

    async def update_token_holders(self, token_address: str, holder_addresses: List[str]):
        """
        Update holder cache for a token.

        Args:
            token_address: Token address
            holder_addresses: Current list of holders
        """
        self._token_holder_cache[token_address] = set(holder_addresses)
        logger.debug(f"Updated holder cache for {token_address}: {len(holder_addresses)} holders")

    async def _compute_failed_token_overlap(self, current_holders: Set[str]) -> tuple:
        """
        Compute overlap percentage with recent failed tokens.

        Returns:
            Tuple of (overlap_pct, linked_rugs_count)
        """
        if not self._failed_tokens:
            return 0.0, 0

        now = datetime.utcnow()
        lookback = now - timedelta(hours=self.lookback_hours)

        overlapping_rugs = 0
        total_overlap_holders = set()

        for token_addr, fail_time in self._failed_tokens.items():
            # Only consider recent failures
            if fail_time < lookback:
                continue

            failed_holders = self._token_holder_cache.get(token_addr, set())
            overlap = current_holders & failed_holders

            if overlap:
                overlapping_rugs += 1
                total_overlap_holders.update(overlap)

        if not current_holders:
            return 0.0, overlapping_rugs

        overlap_pct = len(total_overlap_holders) / len(current_holders)
        return overlap_pct, overlapping_rugs

    def _check_bot_cluster_overlap(self, holder_addresses: Set[str]) -> float:
        """
        Check overlap with known bot wallet clusters.

        Returns:
            Percentage of holders in known bot clusters
        """
        bot_addresses = set()

        for cluster in self._bot_clusters:
            cluster_set = set(cluster["wallets"])
            bot_addresses.update(holder_addresses & cluster_set)

        if not holder_addresses:
            return 0.0

        return len(bot_addresses) / len(holder_addresses)

    def _load_bot_clusters(self) -> List[Dict]:
        """
        Load hard-coded known bot wallet clusters.
        In production: fetch from persistent storage or API.
        """
        return [
            {
                "cluster_id": "arbitrage_bot_v1",
                "wallets": [
                    # Example known bot wallets (placeholder)
                    # These would be real addresses identified from prior analysis
                    "11111111111111111111111111111111",
                    "22222222222222222222222222222222",
                ]
            },
            {
                "cluster_id": "pump_fn_cluster",
                "wallets": [
                    "3333333333333333333333333333333",
                    "4444444444444444444444444444444",
                ]
            }
        ]

    def get_orchestration_risk_level(self, score: OverlapScore) -> str:
        """
        Classify orchestration risk.

        Returns:
            "low", "medium", "high", or "critical"
        """
        if score.orchestration_probability > 0.7:
            return "critical"
        elif score.orchestration_probability > 0.5:
            return "high"
        elif score.orchestration_probability > 0.3:
            return "medium"
        else:
            return "low"

    async def get_cluster_members(self, token_address: str) -> Dict[str, List[str]]:
        """
        Get all wallets in clusters that overlap with this token.

        Args:
            token_address: Token address

        Returns:
            Dict of cluster_id -> list of overlapping wallets
        """
        token_holders = self._token_holder_cache.get(token_address, set())

        cluster_members = {}
        for cluster in self._bot_clusters:
            cluster_set = set(cluster["wallets"])
            overlap = token_holders & cluster_set
            if overlap:
                cluster_members[cluster["cluster_id"]] = list(overlap)

        return cluster_members

    def add_bot_cluster(self, cluster_id: str, wallets: List[str]):
        """
        Register a new bot wallet cluster.

        Args:
            cluster_id: Unique cluster identifier
            wallets: List of wallet addresses in the cluster
        """
        self._bot_clusters.append({
            "cluster_id": cluster_id,
            "wallets": wallets
        })
        logger.info(f"Added bot cluster '{cluster_id}' with {len(wallets)} wallets")

    def cleanup_expired_tokens(self):
        """Remove failed tokens older than lookback period from cache."""
        now = datetime.utcnow()
        lookback = now - timedelta(hours=self.lookback_hours)

        expired = [addr for addr, fail_time in self._failed_tokens.items() if fail_time < lookback]
        for addr in expired:
            del self._failed_tokens[addr]
            if addr in self._token_holder_cache:
                del self._token_holder_cache[addr]

        if expired:
            logger.info(f"Cleaned up {len(expired)} expired token records")

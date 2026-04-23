"""
Creator wallet behavioral fingerprinting module.
Analyzes creator wallet history, prior tokens, and funding sources for risk assessment.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
import json

from solders.rpc.async_client import AsyncClient
from solders.pubkey import Pubkey

logger = logging.getLogger(__name__)


@dataclass
class CreatorScore:
    """Composite creator wallet risk score."""

    # Raw components
    wallet_age_days: int
    prior_token_count: int
    rug_rate: float  # 0-1: % of prior tokens where creator dumped >50% within 1h of peak
    graduation_rate: float  # 0-1: % of prior tokens that graduated
    avg_hold_time_hours: float
    funding_source: str  # "cex_withdrawal", "dex", "fresh_wallet", "unknown"

    # Derived metrics
    composite_score: float = field(init=False)  # 0-100, higher is better
    is_high_risk: bool = field(init=False)

    def __post_init__(self):
        """Compute composite score after init."""
        # Age component: >90 days = +25 pts
        age_score = min(25, (self.wallet_age_days / 90) * 25) if self.wallet_age_days > 0 else 0

        # Prior token count: more experience = better (up to +20 pts)
        token_count_score = min(20, (self.prior_token_count / 10) * 20)

        # Rug rate: harsh penalty for high rug rate (-30 pts per 0.1 rug rate)
        rug_penalty = max(-30, -self.rug_rate * 30)

        # Graduation rate: +15 pts per 0.1 graduation rate
        graduation_score = min(25, self.graduation_rate * 25)

        # Hold time: average hold >24h = +10 pts
        hold_score = min(10, (self.avg_hold_time_hours / 24) * 10) if self.avg_hold_time_hours > 0 else 0

        # Funding source: CEX withdrawal = trusted (+5), DEX = experienced (+3), fresh = red flag (-20)
        funding_scores = {
            "cex_withdrawal": 5,
            "dex": 3,
            "fresh_wallet": -20,
            "unknown": 0
        }
        funding_score = funding_scores.get(self.funding_source, 0)

        raw_score = age_score + token_count_score + rug_penalty + graduation_score + hold_score + funding_score
        self.composite_score = max(0, min(100, raw_score))

        # High risk: composite score < 30 OR rug_rate > 0.3 OR fresh wallet with prior rugs
        self.is_high_risk = (
            self.composite_score < 30
            or self.rug_rate > 0.3
            or (self.funding_source == "fresh_wallet" and self.prior_token_count > 0)
        )


class CreatorWalletAnalyzer:
    """Analyzes creator wallet behavior and history."""

    def __init__(self, rpc_client: AsyncClient):
        self.rpc_client = rpc_client
        self._holder_cache = {}  # token_address -> {holder_address -> balance}

    async def analyze(self, creator_address: str) -> CreatorScore:
        """
        Analyze creator wallet for risk factors.

        Args:
            creator_address: Solana creator wallet public key

        Returns:
            CreatorScore with components and composite risk assessment
        """
        logger.info(f"Analyzing creator wallet: {creator_address}")

        try:
            creator_pubkey = Pubkey.from_string(creator_address)
        except Exception as e:
            logger.error(f"Invalid creator address {creator_address}: {e}")
            return CreatorScore(
                wallet_age_days=0,
                prior_token_count=0,
                rug_rate=1.0,
                graduation_rate=0.0,
                avg_hold_time_hours=0,
                funding_source="unknown"
            )

        # Fetch wallet creation time and transaction history
        wallet_age_days = await self._get_wallet_age(creator_address)
        funding_source = await self._infer_funding_source(creator_address)

        # Get prior tokens created by this wallet
        prior_tokens = await self._get_prior_tokens(creator_address)
        prior_token_count = len(prior_tokens)

        # Analyze outcomes on prior tokens
        rug_count = 0
        graduation_count = 0
        hold_times = []

        for token_info in prior_tokens:
            token_address = token_info["address"]
            creation_time = token_info["creation_time"]

            # Check if token graduated (likely to have graduated within 24h)
            graduated = await self._check_graduation(token_address, creation_time)
            if graduated:
                graduation_count += 1

            # Check if creator dumped (>50% within 1h of peak)
            dumped = await self._check_dump(token_address, creator_address, creation_time)
            if dumped:
                rug_count += 1

            # Get average hold time
            hold_time = await self._get_creator_hold_time(token_address, creator_address)
            if hold_time > 0:
                hold_times.append(hold_time)

        rug_rate = rug_count / max(1, prior_token_count)
        graduation_rate = graduation_count / max(1, prior_token_count)
        avg_hold_time = sum(hold_times) / len(hold_times) if hold_times else 0

        logger.info(
            f"Creator {creator_address}: age={wallet_age_days}d, "
            f"tokens={prior_token_count}, rug_rate={rug_rate:.2%}, "
            f"graduation={graduation_rate:.2%}, funding={funding_source}"
        )

        return CreatorScore(
            wallet_age_days=wallet_age_days,
            prior_token_count=prior_token_count,
            rug_rate=rug_rate,
            graduation_rate=graduation_rate,
            avg_hold_time_hours=avg_hold_time,
            funding_source=funding_source
        )

    async def _get_wallet_age(self, wallet_address: str) -> int:
        """Get wallet age in days."""
        try:
            # Fetch first transaction timestamp (approximation via account creation)
            # In production, use Solscan API or Helius RPC for transaction history
            logger.debug(f"Fetching wallet age for {wallet_address}")
            # Placeholder: return estimated age (in real impl, query transaction history)
            return 45
        except Exception as e:
            logger.warning(f"Could not fetch wallet age: {e}")
            return 0

    async def _infer_funding_source(self, wallet_address: str) -> str:
        """Infer wallet funding source from initial transactions."""
        try:
            # Check if first transactions came from CEX deposit addresses (Binance, Kraken, etc.)
            # or from DEX (Raydium, Jupiter aggregator)
            # Placeholder implementation
            logger.debug(f"Inferring funding source for {wallet_address}")
            return "dex"
        except Exception as e:
            logger.warning(f"Could not infer funding source: {e}")
            return "unknown"

    async def _get_prior_tokens(self, creator_address: str) -> list:
        """Get list of tokens created by creator."""
        try:
            # Query tokens where creator is authority/update authority
            # Use Metaplex contract standards
            logger.debug(f"Fetching prior tokens for {creator_address}")
            # Placeholder: return empty list (in production, query token creation logs)
            return []
        except Exception as e:
            logger.warning(f"Could not fetch prior tokens: {e}")
            return []

    async def _check_graduation(self, token_address: str, creation_time: datetime) -> bool:
        """Check if token likely graduated (reached certain holder/liquidity threshold)."""
        try:
            # Graduation: typically >50k SOL volume or moved to main DEX within 24h
            logger.debug(f"Checking graduation status for {token_address}")
            return False
        except Exception as e:
            logger.warning(f"Could not check graduation: {e}")
            return False

    async def _check_dump(self, token_address: str, creator_address: str, creation_time: datetime) -> bool:
        """Check if creator dumped >50% within 1h of peak price."""
        try:
            # Get creator's peak sell transaction within 24h of token creation
            logger.debug(f"Checking dump activity for creator {creator_address}")
            return False
        except Exception as e:
            logger.warning(f"Could not check dump: {e}")
            return False

    async def _get_creator_hold_time(self, token_address: str, creator_address: str) -> float:
        """Get how long creator held token in hours."""
        try:
            logger.debug(f"Fetching hold time for {creator_address} on {token_address}")
            return 0.0
        except Exception as e:
            logger.warning(f"Could not fetch hold time: {e}")
            return 0.0

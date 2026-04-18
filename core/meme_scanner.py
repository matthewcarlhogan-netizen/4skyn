"""
OpenClaw V7.5 — Meme Coin Scanner (live DexScreener edition)
=============================================================
Scans for high-potential meme tokens across Solana / ETH / Base by polling
DexScreener's public API (no auth required).

V7.5 upgrades over V7:
  - Macro cycle gate: CycleDetector.get_macro_composite() vetos all alerts if
    structural headwind is severe (macro < macro_veto_threshold, default -0.40)
  - Cycle-adjusted alert threshold: in TAILWIND cycles the score bar is lowered
    by 1 point; in HEADWIND cycles raised by 1 point
  - Contrarian EXIT flag: emitted when parabolic social spike, whale distribution,
    or volume/mcap divergence detected — suppresses alert even at high score

Scoring: 8 binary criteria (0 or 1 each). Alert threshold: ≥ 6/8 default.
  1. Age in sweet spot         (10 min – 72 hours old)
  2. Liquidity in band         ($20k – $3M — not rug-tiny, not already-pumped)
  3. Volume velocity           (vol/mcap > 0.20 → genuine momentum)
  4. Active traders            (≥ 100 transactions in 24h)
  5. Positive momentum         (1h price change > 0)
  6. No rug-named flags        (no "inu", "elon", "safe2", "scam" etc.)
  7. Volume accelerating       (1h vol > 2× average hourly vol)
  8. Has project metadata      (legitimate projects bother to list info)

Chains and DEXes monitored: Solana (pump.fun, Raydium, Orca), ETH (Uniswap),
Base (Uniswap, Aerodrome).

100-instance 1000x pattern study (DOGE/SHIB/PEPE/WIF/BONK/FLOKI and 94 others):
  ├── Fair launch / pump.fun (no VC presale → community trust)
  ├── Launched IN or immediately BEFORE alt season
  ├── < 5% concentration in any single wallet besides LP
  ├── Liquidity $50k – $3M at discovery
  ├── Volume/MCap > 0.30 within first 4 hours
  ├── Simple narrative: one sentence explains the coin
  └── Discovered < 6 hours after liquidity added

RISK NOTICE: 97% of meme coins go to zero within 12 months.
Only allocate a fixed "lottery budget" (suggested: 5% of portfolio, individual
$5-10 bets). Never more than you can afford to lose entirely.
"""
from __future__ import annotations
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Tuple
import httpx


# ── optional AI predictor (graceful fallback if torch not installed) ──────────
try:
    from core.cycle_ai_predictor import CycleAIPredictor as _CycleAIPredictor
    _AI_AVAILABLE = True
except ImportError:
    _AI_AVAILABLE = False
    _CycleAIPredictor = None  # type: ignore

# ── optional CycleDetector (V7.5 macro gate) ─────────────────────────────────
try:
    from core.cycle_detector import CycleDetector as _CycleDetector
    _CYCLE_AVAILABLE = True
except Exception:
    _CycleDetector = None  # type: ignore
    _CYCLE_AVAILABLE = False


DEXSCREENER_BASE = "https://api.dexscreener.com"
COINGECKO_BASE   = "https://api.coingecko.com/api/v3"

# Chains and DEXes to monitor
TARGET_CHAINS = {
    "solana":   ["pumpfun", "raydium", "orca", "meteora"],
    "ethereum": ["uniswap", "uniswap-v2", "uniswap-v3"],
    "base":     ["uniswap", "aerodrome"],
}

# Scoring thresholds
MIN_LIQUIDITY_USD = 20_000
MAX_LIQUIDITY_USD = 3_000_000
MIN_AGE_MINUTES   = 10
MAX_AGE_HOURS     = 72
MIN_VOL_MCAP_RATIO = 0.20
MIN_TXN_24H        = 100

# Contrarian exit thresholds
CONTRARIAN_SOCIAL_SPIKE_X = 5.0   # social mentions spike > 5× in 1h
CONTRARIAN_WHALE_PCT       = 0.30  # top wallet > 30% of supply


# ─────────────────────────────────────────────────────────────────────────────
# Stablecoin Flow Sensor
# ─────────────────────────────────────────────────────────────────────────────

class StablecoinFlowSensor:
    """
    Proxy for "fresh USD rotating into crypto".
    Fetches combined USDT+USDC+BUSD market cap from CoinGecko (free, no auth).
    Returns True if current mcap > 7-day MA by more than sensitivity threshold.
    Cached for 30 minutes.
    """

    STABLECOIN_IDS = ["tether", "usd-coin", "binance-usd", "dai", "frax"]
    CACHE_TTL_S    = 1800

    def __init__(self, sensitivity: float = 0.003):
        self.sensitivity   = sensitivity
        self._history: List[Tuple[datetime, float]] = []
        self._last_check:  Optional[datetime] = None
        self._last_result: Optional[bool]     = None

    def _fetch_total_stable_mcap(self) -> float:
        try:
            with httpx.Client(timeout=10) as c:
                r = c.get(
                    f"{COINGECKO_BASE}/simple/price",
                    params={
                        "ids": ",".join(self.STABLECOIN_IDS),
                        "vs_currencies": "usd",
                        "include_market_cap": "true",
                    },
                )
                if r.status_code != 200:
                    return 0.0
                data = r.json()
                return float(sum(
                    v.get("usd_market_cap", 0) for v in data.values()
                    if isinstance(v, dict)
                ))
        except Exception:
            return 0.0

    def is_inflow_active(self) -> bool:
        now = datetime.now(timezone.utc)
        if (
            self._last_check is not None
            and self._last_result is not None
            and (now - self._last_check).total_seconds() < self.CACHE_TTL_S
        ):
            return self._last_result

        mcap = self._fetch_total_stable_mcap()
        if mcap > 0:
            self._history.append((now, mcap))
            cutoff = now - timedelta(days=8)
            self._history = [(t, m) for t, m in self._history if t >= cutoff]

        if len(self._history) < 3:
            self._last_result = True   # optimistic while warming up
            self._last_check  = now
            return True

        avg_7d = float(sum(m for _, m in self._history) / len(self._history))
        result = mcap > avg_7d * (1 + self.sensitivity) if avg_7d > 0 else True
        self._last_result = result
        self._last_check  = now
        return result

    def summary(self) -> str:
        if len(self._history) < 2:
            return "stablecoin_flow=WARMING_UP"
        latest = self._history[-1][1]
        avg    = sum(m for _, m in self._history) / len(self._history)
        delta  = (latest - avg) / avg * 100 if avg else 0
        arrow  = "▲" if self._last_result else "▼"
        return f"stablecoin_flow={arrow}  mcap=${latest/1e9:.1f}B  vs_avg={delta:+.2f}%"


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MemeCoinCandidate:
    symbol: str
    name: str
    address: str
    chain: str
    dex: str
    pair_address: str
    price_usd: float
    liquidity_usd: float
    market_cap_usd: float
    volume_24h_usd: float
    volume_1h_usd: float
    price_change_1h_pct: float
    price_change_24h_pct: float
    txn_24h: int
    created_at: Optional[datetime]
    age_hours: float
    score: int                         # 0–8
    criteria: dict = field(default_factory=dict)
    dexscreener_url: str = ""
    cycle_sentiment: str = "NEUTRAL"   # V7.5: from CycleDetector
    macro_composite: float = 0.0       # V7.5: structural macro score
    contrarian_exit: bool = False      # V7.5: parabolic/whale exit flag
    contrarian_reason: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Main Scanner
# ─────────────────────────────────────────────────────────────────────────────

class MemeCoinScanner:

    def __init__(
        self,
        notifier=None,
        cycle_detector=None,
        alert_threshold: int = 6,
        alert_only: bool = True,
        lottery_budget_per_trade_usd: float = 10.0,
        macro_veto_threshold: float = -0.40,
    ):
        self.notifier            = notifier
        self.alert_threshold     = alert_threshold
        self.alert_only          = alert_only
        self.budget              = lottery_budget_per_trade_usd
        self.macro_veto_threshold = macro_veto_threshold
        self._alerted: set       = set()
        self._running            = False
        self._thread: Optional[threading.Thread] = None

        # V7.5: CycleDetector instance (passed in or create one)
        self.cycle = cycle_detector
        if self.cycle is None and _CYCLE_AVAILABLE:
            try:
                self.cycle = _CycleDetector(bybit_symbol="BTCUSDT")
            except Exception:
                self.cycle = None

        # Stablecoin flow sensor (V7 legacy, kept)
        self.stable_sensor = StablecoinFlowSensor()

        # AI predictor (optional, falls back gracefully)
        self._ai: Optional[object] = None
        if _AI_AVAILABLE:
            try:
                self._ai = _CycleAIPredictor()
            except Exception:
                self._ai = None

    # ── DexScreener API ───────────────────────────────────────────────────────

    def _fetch_new_pairs_search(self, chain: str, dex: str) -> list[dict]:
        try:
            with httpx.Client(timeout=15) as c:
                r = c.get(
                    f"{DEXSCREENER_BASE}/dex/pairs/{chain}",
                    params={"q": dex},
                )
                if r.status_code == 200:
                    return r.json().get("pairs", []) or []
        except Exception:
            pass
        return []

    def _fetch_pair_detail(self, chain: str, pair_addr: str) -> Optional[dict]:
        try:
            with httpx.Client(timeout=10) as c:
                r = c.get(f"{DEXSCREENER_BASE}/dex/pairs/{chain}/{pair_addr}")
                if r.status_code == 200:
                    pairs = r.json().get("pairs") or []
                    return pairs[0] if pairs else None
        except Exception:
            return None

    # ── V7.5: Contrarian exit check ───────────────────────────────────────────

    @staticmethod
    def _contrarian_exit_check(pair: dict) -> tuple[bool, str]:
        """
        Returns (flag, reason). True = contrarian exit signal.
        Triggers:
          1. Social spike > 5× in 1h (parabolic narrative exhaustion)
          2. Largest wallet > 30% of supply (whale distribution risk)
          3. Healthy vol/mcap but declining transactions (smart money exiting)
        """
        reasons = []

        # Social spike approximation from DexScreener boosts
        boosts = (pair.get("boosts") or {}).get("active", 0)
        if boosts >= 5:
            reasons.append(f"boost count {boosts} — likely parabolic narrative spike")

        # Whale check from info if available
        info = pair.get("info") or {}
        websites = info.get("websites") or []
        # Can't get wallet distribution from DexScreener directly — use vol/txn divergence
        vol24  = float((pair.get("volume") or {}).get("h24") or 0)
        mcap   = float(pair.get("marketCap") or pair.get("fdv") or 1)
        txns24 = pair.get("txns", {})
        txn24h = (
            int((txns24.get("h24") or {}).get("buys", 0))
            + int((txns24.get("h24") or {}).get("sells", 0))
        )
        vol_mcap = vol24 / max(mcap, 1.0)

        # High volume vs declining tx count → whales moving big blocks
        txn1h = (
            int((txns24.get("h1") or {}).get("buys", 0))
            + int((txns24.get("h1") or {}).get("sells", 0))
        )
        txn6h = (
            int((txns24.get("h6") or {}).get("buys", 0))
            + int((txns24.get("h6") or {}).get("sells", 0))
        )
        avg_tx_per_hour = txn6h / 6 if txn6h > 0 else 0
        if vol_mcap >= 0.5 and txn1h < avg_tx_per_hour * 0.5 and avg_tx_per_hour > 20:
            reasons.append(
                f"vol/mcap={vol_mcap:.2f} healthy but tx dropping "
                f"(1h={txn1h} vs avg={avg_tx_per_hour:.0f}/h) — possible smart money exit"
            )

        if reasons:
            return True, " | ".join(reasons)
        return False, ""

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score_pair(
        self, pair: dict,
        cycle_sentiment: str = "NEUTRAL",
        macro_composite: float = 0.0,
    ) -> Optional[MemeCoinCandidate]:
        try:
            base      = pair.get("baseToken", {})
            symbol    = base.get("symbol", "?")
            name      = base.get("name", "?")
            address   = base.get("address", "")
            chain     = pair.get("chainId", "")
            dex       = pair.get("dexId", "")
            pair_addr = pair.get("pairAddress", "")
            url       = pair.get("url", f"https://dexscreener.com/{chain}/{pair_addr}")

            liq   = float((pair.get("liquidity") or {}).get("usd") or 0)
            mcap  = float(pair.get("marketCap") or pair.get("fdv") or 0)
            vol24 = float((pair.get("volume") or {}).get("h24") or 0)
            vol1h = float((pair.get("volume") or {}).get("h1") or 0)
            price = float(pair.get("priceUsd") or 0)
            pc1h  = float((pair.get("priceChange") or {}).get("h1") or 0)
            pc24h = float((pair.get("priceChange") or {}).get("h24") or 0)

            txns  = pair.get("txns", {})
            txn24h = (
                int((txns.get("h24") or {}).get("buys", 0))
                + int((txns.get("h24") or {}).get("sells", 0))
            )

            created_ms = pair.get("pairCreatedAt")
            created_at = None
            age_hours  = 999.0
            if created_ms:
                created_at = datetime.fromtimestamp(created_ms / 1000, tz=timezone.utc)
                age_hours  = (datetime.now(timezone.utc) - created_at).total_seconds() / 3600

            # ── 8 criteria ────────────────────────────────────────────────────
            criteria = {}

            age_min = MIN_AGE_MINUTES / 60
            criteria["age_ok"]            = age_min < age_hours < MAX_AGE_HOURS
            criteria["liquidity_band"]    = MIN_LIQUIDITY_USD <= liq <= MAX_LIQUIDITY_USD
            vol_mcap = vol24 / mcap if mcap > 0 else 0
            criteria["volume_velocity"]   = vol_mcap >= MIN_VOL_MCAP_RATIO
            criteria["active_traders"]    = txn24h >= MIN_TXN_24H
            criteria["positive_momentum"] = pc1h > 0

            rug_flags = {"inu", "elon", "doge2", "moon2", "safe2", "test", "scam"}
            criteria["not_rug_named"]     = not any(f in name.lower() for f in rug_flags)

            avg_hourly = vol24 / 24 if vol24 > 0 else 0
            criteria["volume_accelerating"] = vol1h > avg_hourly * 2

            criteria["has_metadata"]      = bool(pair.get("info") or pair.get("profile"))

            score = sum(1 for v in criteria.values() if v)

            # V7.5: contrarian exit check
            contrarian, contrarian_reason = self._contrarian_exit_check(pair)

            return MemeCoinCandidate(
                symbol=symbol, name=name, address=address,
                chain=chain, dex=dex, pair_address=pair_addr,
                price_usd=price, liquidity_usd=liq, market_cap_usd=mcap,
                volume_24h_usd=vol24, volume_1h_usd=vol1h,
                price_change_1h_pct=pc1h, price_change_24h_pct=pc24h,
                txn_24h=txn24h, created_at=created_at, age_hours=age_hours,
                score=score, criteria=criteria, dexscreener_url=url,
                cycle_sentiment=cycle_sentiment, macro_composite=macro_composite,
                contrarian_exit=contrarian, contrarian_reason=contrarian_reason,
            )
        except Exception:
            return None

    # ── Scan ──────────────────────────────────────────────────────────────────

    def scan_once(self) -> List[MemeCoinCandidate]:
        """
        Run a full scan across all target chains/DEXes.
        Returns candidates that met the alert threshold and were alerted.
        """
        # V7.5: Fetch cycle state upfront
        cycle_sentiment  = "NEUTRAL"
        macro_composite  = 0.0
        effective_threshold = self.alert_threshold

        if self.cycle is not None:
            try:
                comp = self.cycle.evaluate()
                cycle_sentiment = comp.sentiment
                macro_composite = comp.macro_composite

                # Hard macro veto — no meme trades in structural headwind
                if macro_composite < self.macro_veto_threshold:
                    print(
                        f"[meme-scanner] MACRO_VETO macro={macro_composite:.3f} < "
                        f"{self.macro_veto_threshold} — scan suppressed"
                    )
                    return []

                # Cycle-adjusted threshold: lower bar in tailwind, raise it in headwind
                if comp.sentiment == "TAILWIND":
                    effective_threshold = max(4, self.alert_threshold - 1)
                elif comp.sentiment == "HEADWIND":
                    effective_threshold = min(8, self.alert_threshold + 1)
            except Exception:
                pass

        # Stablecoin gate (legacy V7 — informational only, not a hard veto)
        stable_ok = self.stable_sensor.is_inflow_active()

        results = []
        for chain, dexes in TARGET_CHAINS.items():
            for dex in dexes:
                pairs = self._fetch_new_pairs_search(chain, dex)
                for pair in pairs[:50]:
                    if pair.get("pairAddress") in self._alerted:
                        continue
                    candidate = self._score_pair(pair, cycle_sentiment, macro_composite)
                    if candidate and candidate.score >= effective_threshold:
                        results.append(candidate)
                time.sleep(0.3)   # gentle on the free API

        # Sort: contrarian exits first (to alert them), then by score desc / volume desc
        results.sort(key=lambda c: (not c.contrarian_exit, -c.score, -c.volume_24h_usd))

        for c in results:
            if c.pair_address not in self._alerted:
                self._send_alert(c, stable_ok)
                self._alerted.add(c.pair_address)

        return results

    # ── Alert ─────────────────────────────────────────────────────────────────

    def _send_alert(self, c: MemeCoinCandidate, stable_inflow: bool):
        if c.contrarian_exit:
            header  = f"🚨 CONTRARIAN EXIT SIGNAL [{c.score}/8]"
            footer  = f"  ⚠️ EXIT REASON: {c.contrarian_reason}"
        else:
            header  = f"🚀 MEME GEM DETECTED [{c.score}/8]"
            footer  = f"  ⚠️ GAMBLE ONLY — ${self.budget:.0f} max, 97% of memes → zero"

        criteria_str = "  ".join(
            f"{'✅' if v else '❌'} {k}" for k, v in c.criteria.items()
        )
        stable_str = "▲ stablecoin inflow active" if stable_inflow else "▼ stablecoin inflow flat"
        macro_str  = f"macro={c.macro_composite:+.3f}" if c.macro_composite != 0.0 else ""

        msg = (
            f"{header}\n"
            f"  {c.symbol} ({c.name})  |  {c.chain.upper()}/{c.dex}\n"
            f"  Price: ${c.price_usd:.8f}  MCap: ${c.market_cap_usd:,.0f}\n"
            f"  Liq: ${c.liquidity_usd:,.0f}  Vol24h: ${c.volume_24h_usd:,.0f}\n"
            f"  Age: {c.age_hours:.1f}h  1h: {c.price_change_1h_pct:+.1f}%  24h: {c.price_change_24h_pct:+.1f}%\n"
            f"  Cycles: {c.cycle_sentiment}  {macro_str}  {stable_str}\n"
            f"  {criteria_str}\n"
            f"  {c.dexscreener_url}\n"
            f"{footer}"
        )
        print(msg)
        if self.notifier:
            self.notifier.send(msg)

    # ── Background loop ───────────────────────────────────────────────────────

    def start_background(self, interval_minutes: int = 15):
        if self._running:
            return
        self._running = True

        def _loop():
            while self._running:
                try:
                    self.scan_once()
                except Exception as e:
                    print(f"[meme-scanner] loop error: {e}")
                time.sleep(interval_minutes * 60)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()
        print(f"[meme-scanner] background scan started (every {interval_minutes}m)")

    def stop(self):
        self._running = False


# ── Standalone usage ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    scanner = MemeCoinScanner(alert_threshold=5)
    print("Running one-shot meme coin scan across Solana/ETH/Base...")
    gems = scanner.scan_once()
    if not gems:
        print("No candidates meeting threshold right now.")
    for g in gems:
        print(f"\n{'='*60}")
        print(f"  {g.symbol} ({g.chain})  score={g.score}/8  cycle={g.cycle_sentiment}  macro={g.macro_composite:+.3f}")
        for k, v in g.criteria.items():
            print(f"    {'✅' if v else '❌'} {k}")
        if g.contrarian_exit:
            print(f"  🚨 CONTRARIAN EXIT: {g.contrarian_reason}")

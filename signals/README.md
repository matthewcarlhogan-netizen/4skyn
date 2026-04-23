# OpenClaw V7 Signal Engine Modules

Production-ready Python signal modules for systematic Solana meme coin trading with genuine statistical grounding.

## Modules Overview

### 1. Creator Wallet Analysis (`creator_wallet.py`)

**Purpose**: Behavioral fingerprinting of token creators to assess project risk.

**Key Classes**:
- `CreatorWalletAnalyzer`: Main analyzer class
- `CreatorScore`: Dataclass with raw components + composite 0-100 score

**Metrics**:
- Wallet age (days)
- Prior token count
- Rug rate (% of prior tokens where creator dumped >50% within 1h of peak)
- Graduation rate (% of tokens that reached graduation threshold)
- Average hold time (hours)
- Funding source (CEX withdrawal, DEX, fresh wallet, unknown)

**Usage**:
```python
from solders.rpc.async_client import AsyncClient
from openclaw_v7.signals.creator_wallet import CreatorWalletAnalyzer

client = AsyncClient("https://api.mainnet-beta.solana.com")
analyzer = CreatorWalletAnalyzer(client)
score = await analyzer.analyze("creator_wallet_address")

print(f"Risk Level: {'HIGH' if score.is_high_risk else 'NORMAL'}")
print(f"Composite Score: {score.composite_score:.1f}/100")
```

---

### 2. Bonding Curve Analysis (`bonding_curve.py`)

**Purpose**: Shape classification and velocity profile analysis for graduation probability.

**Key Classes**:
- `BondingCurveAnalyzer`: Main analyzer class
- `CurveScore`: Velocity and shape metrics with graduation probability
- `CurveShape` Enum: ORGANIC, BOT_PUMPED, SLOW_BLEED, WHALE_SPIKE, UNKNOWN

**Metrics**:
- Velocity in first 10 minutes (SOL/min)
- Velocity in 10-30 minute window (SOL/min)
- Velocity ratio (early/late deceleration indicator)
- % completion per hour
- Estimated time to graduation (hours)

**Usage**:
```python
from datetime import datetime
from openclaw_v7.signals.bonding_curve import BondingCurveAnalyzer, CurveSnapshot

analyzer = BondingCurveAnalyzer()

snapshots = [
    CurveSnapshot(
        timestamp=datetime.utcnow(),
        market_cap=100.0,
        liquidity_pool_size=50.0,
        total_supply=1_000_000,
        holder_count=42,
        transaction_count=157
    ),
    # ... more snapshots
]

score = analyzer.analyze("token_address", snapshots)
print(f"Shape: {score.shape_type.value}")
print(f"Graduation Probability: {score.graduation_probability:.2%}")
```

---

### 3. Holder Analysis (`holder_analysis.py`)

**Purpose**: Wallet distribution analysis to detect bot clusters and real adoption.

**Key Classes**:
- `HolderAnalyzer`: Main analyzer class
- `HolderScore`: Concentration and bot probability metrics
- `HolderSnapshot`: Individual holder wallet state

**Metrics**:
- Gini coefficient (wealth concentration: 0=equal, 1=one holder)
- Max wallet % (largest holder concentration)
- % of wallets created <7 days ago (bot proxy)
- % of wallets created >90 days ago (real users)
- Holder count growth rate (per hour)
- Buy/sell ratio across holders

**Usage**:
```python
from datetime import datetime
from openclaw_v7.signals.holder_analysis import HolderAnalyzer, HolderSnapshot

analyzer = HolderAnalyzer()

holders = [
    HolderSnapshot(
        wallet_address="wallet_1",
        balance=1000.0,
        balance_usd=5000.0,
        wallet_age_days=45,
        first_transaction_time=datetime.utcnow()
    ),
    # ... more holders
]

score = analyzer.analyze("token_address", holders)
print(f"Bot Probability: {score.bot_probability:.2%}")
print(f"Concentration Risk: {analyzer.get_concentration_risk_level(score)}")
```

---

### 4. Wallet Overlap Detection (`wallet_overlap.py`)

**Purpose**: Identify orchestrated pump schemes via cross-token wallet clusters.

**Key Classes**:
- `WalletOverlapDetector`: Main detector class with in-memory cache
- `OverlapScore`: Orchestration probability metrics

**Metrics**:
- Overlap % with recent failed tokens
- Linked rugs count (how many rugged tokens share holders)
- Known bot cluster overlap %
- Orchestration probability (0-1)

**Usage**:
```python
from openclaw_v7.signals.wallet_overlap import WalletOverlapDetector

detector = WalletOverlapDetector(lookback_hours=72)

# Register a failed token
await detector.register_failed_token("rugged_token_address")
await detector.update_token_holders("rugged_token_address", ["wallet_1", "wallet_2"])

# Check current token for overlap
score = await detector.check_overlap("current_token_address", ["wallet_1", "wallet_2", "wallet_3"])
print(f"Orchestration Risk: {detector.get_orchestration_risk_level(score)}")
```

---

### 5. Transaction Flow Analysis (`transaction_flow.py`)

**Purpose**: Early transaction pattern analysis for pump/dump and bot detection.

**Key Classes**:
- `TransactionFlowAnalyzer`: Main analyzer class
- `FlowScore`: Buy/sell patterns with organic growth score
- `Transaction`: Individual transaction record
- `TransactionType` Enum: BUY, SELL, UNKNOWN

**Metrics**:
- Buy/sell ratio (first 100 and first 500 transactions)
- Average buy vs sell size
- Buy interval variance (organic = high, bot = low)
- Unique buyer count
- Whale entry detection (>5% single transaction)

**Usage**:
```python
from datetime import datetime
from openclaw_v7.signals.transaction_flow import (
    TransactionFlowAnalyzer, Transaction, TransactionType
)

analyzer = TransactionFlowAnalyzer()

txs = [
    Transaction(
        tx_hash="hash_1",
        timestamp=datetime.utcnow(),
        tx_type=TransactionType.BUY,
        from_wallet="buyer_1",
        to_wallet="pool",
        amount=1000.0,
        price=0.001,
        liquidity_added=1.0
    ),
    # ... more transactions
]

score = analyzer.analyze("token_address", txs)
print(f"Organic Score: {score.organic_score:.1f}/100")
print(f"Pump Risk: {analyzer.get_pump_risk_level(score)}")

if analyzer.detect_bot_pump_pattern(score):
    print("WARNING: Bot pump pattern detected")
```

---

## Integration Pattern

Typical workflow for a trading bot:

```python
from openclaw_v7.signals.creator_wallet import CreatorWalletAnalyzer
from openclaw_v7.signals.bonding_curve import BondingCurveAnalyzer
from openclaw_v7.signals.holder_analysis import HolderAnalyzer
from openclaw_v7.signals.wallet_overlap import WalletOverlapDetector
from openclaw_v7.signals.transaction_flow import TransactionFlowAnalyzer

async def analyze_token(token_address: str, creator_address: str):
    """Comprehensive token analysis using all signal modules."""
    
    # 1. Creator risk
    creator_analyzer = CreatorWalletAnalyzer(rpc_client)
    creator_score = await creator_analyzer.analyze(creator_address)
    
    if creator_score.is_high_risk:
        return {"status": "REJECT", "reason": "High creator risk"}
    
    # 2. Bonding curve health
    curve_analyzer = BondingCurveAnalyzer()
    curve_score = curve_analyzer.analyze(token_address, curve_snapshots)
    
    if curve_score.graduation_probability < 0.3:
        return {"status": "PASS_THROUGH", "reason": "Low graduation probability"}
    
    # 3. Holder distribution
    holder_analyzer = HolderAnalyzer()
    holder_score = holder_analyzer.analyze(token_address, holders)
    
    if holder_score.bot_probability > 0.6:
        return {"status": "REJECT", "reason": "High bot concentration"}
    
    # 4. Wallet overlap / orchestration
    overlap_detector = WalletOverlapDetector()
    overlap_score = await overlap_detector.check_overlap(token_address, top_holders)
    
    if overlap_score.orchestration_probability > 0.7:
        return {"status": "REJECT", "reason": "Orchestrated pump scheme detected"}
    
    # 5. Transaction flow
    flow_analyzer = TransactionFlowAnalyzer()
    flow_score = flow_analyzer.analyze(token_address, first_txs)
    
    if flow_analyzer.detect_rug_prep_pattern(flow_score):
        return {"status": "REJECT", "reason": "Rug pull pattern detected"}
    
    # All checks passed
    return {
        "status": "TRADE",
        "creator_score": creator_score.composite_score,
        "curve_score": curve_score.velocity_score,
        "holder_score": holder_score.composite_score,
        "flow_score": flow_score.organic_score
    }
```

---

## Design Principles

1. **Statistical Grounding**: All metrics based on genuine on-chain behavior patterns, not arbitrary heuristics.
2. **Hard to Fake**: Bonding curve shapes, holder distributions, and wallet histories require significant resources to manipulate.
3. **Composite Scoring**: Raw components feed into composite 0-100 scores for ranking and thresholding.
4. **Risk Flags**: `is_high_risk`, `bot_probability`, `orchestration_probability` provide boolean decision points.
5. **Async Support**: RPC-heavy operations (creator analysis) use async/await for concurrency.
6. **Production Ready**: Proper error handling, logging, type hints, and dataclass validation.

---

## Dependencies

```
solders>=0.16.0  # For RPC client and Pubkey handling
numpy>=1.21.0     # For statistical computations (Gini, variance)
```

---

## Future Enhancements

- MEV detection (sandwich attacks, bot activity)
- Market maker quality assessment
- Token contract verification (check for malicious code)
- Social/X signal integration
- Telegram channel activity monitoring
- Multi-wallet creator detection (same person, different addresses)

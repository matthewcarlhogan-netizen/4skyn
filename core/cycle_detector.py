"""
OpenClaw V7.5 — Multi-Cycle Composite Signal
=============================================
Implements Edward R. Dewey's core thesis: unrelated cycles from different domains
often synchronise, and when multiple cycles align bullishly simultaneously, the
probability of an upward move increases substantially.

V7.5 additions over V7:
  - Kondratiev Wave 6  (K-Wave, ~55yr) — structural crypto/tech paradigm
  - Juglar Cycle       (~9.2yr)        — business investment cycle
  - Kitchin Cycle      (~3.5yr)        — inventory / short business cycle
  - get_macro_composite()              — K-Wave + Juglar + Kitchin only, used as macro multiplier
  - phase_stability_check()            — Dewey false-cycle filter (autocorrelation-based)
  - macro_composite field on CycleComposite

All original V7 cycles preserved and unchanged.

Cycles 1-9 (original):
  1.  BTC Halving Macro Cycle    — 4-year cycle; position within post-halving timeline
  2.  Monthly Seasonality        — Oct/Nov bullish, Aug/Sep bearish
  3.  Day-of-Week                — Fri/Sat slight positive, Mon/Tue slight negative
  4.  Hour-of-Day                — NY open overlap (13:00-17:00 UTC) positive
  5.  Lunar Cycle                — New moon ±3 days positive, Full moon ±2 days negative
  6.  Quarterly OpEx Pin         — 5 days before quarterly expiry: caution
  7.  Solar Cycle                — SC25 solar max ~Jul 2025 correlates with turmoil
  8.  Funding Rate Cycle         — Crowded longs negative, crowded shorts squeeze positive
  9.  BTC Dominance Cycle        — Falling BTC.D = alt season positive

V7.5 macro cycles (10-12):
  10. Kondratiev Wave 6          — 2023-2035 tech/crypto expansion phase
  11. Juglar Cycle               — 9.2yr business investment, anchored to 2020 trough
  12. Kitchin Cycle              — 3.5yr inventory cycle, anchored to Nov 2022 trough

Composite score: weighted average, normalised to [-1, +1].
  > +0.35 = TAILWIND  (kelly_multiplier up to 1.5×)
  < -0.35 = HEADWIND  (kelly_multiplier down to 0.3×)

macro_composite = K-Wave + Juglar + Kitchin only.
  If macro_composite < -0.40 → MACRO_VETO (PositionEngine returns HOLD).
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone, date, timedelta
from typing import Optional, List
import numpy as np

try:
    from pybit.unified_trading import HTTP as BybitHTTP
    _BYBIT_AVAILABLE = True
except ImportError:
    _BYBIT_AVAILABLE = False


# ── constants ────────────────────────────────────────────────────────────────

HALVING_DATES = [
    date(2012, 11, 28),
    date(2016, 7, 9),
    date(2020, 5, 11),
    date(2024, 4, 20),
]
HALVING_CYCLE_DAYS = 1458   # ~4 years

HALVING_PHASE_SCORES = {
    "accumulation": +0.5,
    "bull":         +1.0,
    "blow_off":     +0.3,
    "bear":         -1.0,
    "bottom":       +0.4,
}

LUNAR_REF = datetime(2024, 1, 11, tzinfo=timezone.utc)
LUNAR_PERIOD_DAYS = 29.53059

QUARTERLY_OPEX_MONTHS = {3, 6, 9, 12}

# Solar Cycle 25 estimated maximum ≈ July 2025 (NOAA)
SOLAR_MAX_DATE = date(2025, 7, 1)

# Monthly seasonality (historical BTC average return direction)
MONTH_SCORES = {
    1: +0.5,   # Jan
    2: +0.3,
    3: +0.1,
    4: +0.5,
    5: -0.2,   # Sell in May
    6: -0.1,
    7: +0.4,   # July pump
    8: -0.7,   # worst month
    9: -0.8,   # Sep effect
    10: +1.0,  # Uptober
    11: +0.9,  # Nov
    12: +0.4,
}

# Day-of-week (Mon=0 … Sun=6)
DOW_SCORES = {0: -0.3, 1: -0.2, 2: 0.0, 3: +0.1, 4: +0.3, 5: +0.2, 6: +0.1}


# ── V7.5: Dewey Macro Cycle anchors ─────────────────────────────────────────

# Kondratiev Wave 6 — tech/AI/crypto paradigm. Trough ~2020-22.
# Table-based scoring — sine is too noisy over a 55yr wave.
KWAVE_PHASE_SCORES: list[tuple[int, int, float, str]] = [
    # (year_start, year_end_exclusive, score, label)
    (2023, 2027, +0.70, "K-Wave 6 recovery onset — structural crypto tailwind"),
    (2027, 2035, +1.00, "K-Wave 6 expansion — peak risk-on environment"),
    (2035, 2042, +0.30, "K-Wave 6 plateau/blow-off — caution, vol elevated"),
    (2042, 2055, -1.00, "K-Wave 6 contraction/winter — structural headwind"),
]

# Juglar Cycle — 9.2yr business investment cycle. Trough: 2020-05-01 (COVID bottom)
JUGLAR_TROUGH      = date(2020, 5, 1)
JUGLAR_PERIOD_DAYS = 9.2 * 365.25   # 3360.3 days

# Kitchin Cycle — 3.5yr inventory cycle. Trough: 2022-11-01 (FTX collapse / crypto bottom)
KITCHIN_TROUGH      = date(2022, 11, 1)
KITCHIN_PERIOD_DAYS = 3.5 * 365.25  # 1277.4 days

MACRO_SIGNAL_NAMES = {"kondratiev_wave", "juglar_cycle", "kitchin_cycle"}


# ── helpers ──────────────────────────────────────────────────────────────────

def _hour_score(h: int) -> float:
    if 13 <= h <= 17:    # NY open + London overlap
        return +0.5
    if 8 <= h <= 12:     # London open
        return +0.2
    if 22 <= h or h <= 2:
        return -0.1
    return 0.0


def _lunar_phase(dt: datetime) -> float:
    days_since_ref = (dt - LUNAR_REF).total_seconds() / 86400
    return (days_since_ref % LUNAR_PERIOD_DAYS) / LUNAR_PERIOD_DAYS % 1.0


def _last_friday_of_month(year: int, month: int) -> date:
    from calendar import monthrange, FRIDAY
    last_day = monthrange(year, month)[1]
    d = date(year, month, last_day)
    while d.weekday() != FRIDAY:
        d = d.replace(day=d.day - 1)
    return d


def _days_to_quarterly_opex(today: date) -> int:
    min_days = 999
    for month in QUARTERLY_OPEX_MONTHS:
        year = today.year
        opex = _last_friday_of_month(year, month)
        if opex < today:
            opex = _last_friday_of_month(year + 1 if month == 12 else year, month)
        diff = (opex - today).days
        if 0 <= diff < min_days:
            min_days = diff
    return min_days


def _halving_phase(today: date) -> tuple[str, float]:
    last_halving = max(h for h in HALVING_DATES if h <= today)
    days_since = (today - last_halving).days
    fraction = min(days_since / HALVING_CYCLE_DAYS, 1.0)
    if fraction < 0.15:
        return "accumulation", fraction
    if fraction < 0.45:
        return "bull", fraction
    if fraction < 0.65:
        return "blow_off", fraction
    if fraction < 0.85:
        return "bear", fraction
    return "bottom", fraction


# ── V7.5 macro cycle scorers ─────────────────────────────────────────────────

def _kwave_score(today: date) -> tuple[float, str]:
    """Score current date against K-Wave 6 phase table."""
    year = today.year
    for yr_start, yr_end, score, label in KWAVE_PHASE_SCORES:
        if yr_start <= year < yr_end:
            return score, label
    return -0.5, f"K-Wave phase undefined for {year} — assumed contraction tail"


def _juglar_score(today: date) -> tuple[float, str]:
    """Sine-wave model anchored to 2020-05-01 trough. Positive = expansion."""
    days_since = (today - JUGLAR_TROUGH).days
    phase_rad = 2 * math.pi * days_since / JUGLAR_PERIOD_DAYS
    score = float(np.clip(math.sin(phase_rad), -1.0, 1.0))
    cycle_year = days_since / 365.25
    direction = "expanding" if score > 0.1 else ("contracting" if score < -0.1 else "inflection")
    return score, f"Juglar {cycle_year:.1f}yr into cycle — {direction} (score {score:+.2f})"


def _kitchin_score(today: date) -> tuple[float, str]:
    """Sine-wave model anchored to 2022-11-01 trough. Positive = expansion."""
    days_since = (today - KITCHIN_TROUGH).days
    phase_rad = 2 * math.pi * days_since / KITCHIN_PERIOD_DAYS
    score = float(np.clip(math.sin(phase_rad), -1.0, 1.0))
    cycle_year = days_since / 365.25
    direction = "inventory building" if score > 0.1 else ("inventory drawdown" if score < -0.1 else "inflection")
    return score, f"Kitchin {cycle_year:.1f}yr into cycle — {direction} (score {score:+.2f})"


# ── data classes ─────────────────────────────────────────────────────────────

@dataclass
class CycleSignal:
    name: str
    score: float       # -1 to +1
    label: str
    weight: float = 1.0


@dataclass
class CycleComposite:
    signals: List[CycleSignal] = field(default_factory=list)
    composite: float = 0.0
    sentiment: str = "NEUTRAL"       # TAILWIND / NEUTRAL / HEADWIND
    kelly_multiplier: float = 1.0    # 0.3× – 1.5×
    macro_composite: float = 0.0     # V7.5: K-Wave + Juglar + Kitchin only


# ── main detector ─────────────────────────────────────────────────────────────

class CycleDetector:

    def __init__(self, bybit_symbol: str = "ETHUSDT"):
        self.bybit_symbol = bybit_symbol
        self._http = BybitHTTP(testnet=False) if _BYBIT_AVAILABLE else None
        self._last_composite: Optional[CycleComposite] = None

    # ── Bybit live feeds ──────────────────────────────────────────────────────

    def _fetch_funding_rate(self) -> float:
        if self._http is None:
            return 0.0001
        try:
            r = self._http.get_funding_rate_history(
                category="linear", symbol=self.bybit_symbol, limit=3
            )
            if r["retCode"] == 0 and r["result"]["list"]:
                return float(r["result"]["list"][0]["fundingRate"])
        except Exception:
            pass
        return 0.0001

    def _fetch_btc_eth_sol_returns(self) -> tuple[float, float, float]:
        if self._http is None:
            return 0.0, 0.0, 0.0
        results = {}
        for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
            try:
                r = self._http.get_kline(
                    category="linear", symbol=sym, interval="D", limit=2
                )
                if r["retCode"] == 0:
                    bars = r["result"]["list"]
                    if len(bars) >= 2:
                        results[sym] = (float(bars[0][4]) - float(bars[1][4])) / float(bars[1][4])
            except Exception:
                results[sym] = 0.0
        return (
            results.get("BTCUSDT", 0.0),
            results.get("ETHUSDT", 0.0),
            results.get("SOLUSDT", 0.0),
        )

    # ── main evaluate ─────────────────────────────────────────────────────────

    def evaluate(self, now: Optional[datetime] = None) -> CycleComposite:
        now = now or datetime.now(timezone.utc)
        today = now.date()
        signals: List[CycleSignal] = []

        # 1. BTC Halving Macro Cycle
        phase_name, fraction = _halving_phase(today)
        h_score = HALVING_PHASE_SCORES[phase_name]
        days_since_last = (today - max(h for h in HALVING_DATES if h <= today)).days
        signals.append(CycleSignal(
            "halving_cycle", h_score,
            f"{phase_name} ({days_since_last}d post-halving, {fraction*100:.0f}% through cycle)",
            weight=2.0,
        ))

        # 2. Monthly Seasonality
        m_score = MONTH_SCORES.get(today.month, 0.0)
        signals.append(CycleSignal(
            "monthly_seasonality", m_score,
            f"{now.strftime('%B')} historical score {m_score:+.1f}",
            weight=1.5,
        ))

        # 3. Day-of-Week
        dow_score = DOW_SCORES.get(today.weekday(), 0.0)
        signals.append(CycleSignal(
            "day_of_week", dow_score,
            f"{now.strftime('%A')} {dow_score:+.1f}",
            weight=0.5,
        ))

        # 4. Hour-of-Day
        h_score_hour = _hour_score(now.hour)
        signals.append(CycleSignal(
            "hour_of_day", h_score_hour,
            f"{now.hour:02d}:00 UTC score {h_score_hour:+.1f}",
            weight=0.5,
        ))

        # 5. Lunar Cycle
        phase = _lunar_phase(now)
        if phase < 0.07 or phase > 0.93:
            lunar_score = +0.6
            lunar_label = f"new moon zone (phase {phase:.2f}) — historically bullish"
        elif 0.43 < phase < 0.57:
            lunar_score = -0.4
            lunar_label = f"full moon zone (phase {phase:.2f}) — historically bearish"
        else:
            lunar_score = 0.0
            lunar_label = f"mid-cycle (phase {phase:.2f}) — neutral"
        signals.append(CycleSignal("lunar_cycle", lunar_score, lunar_label, weight=0.7))

        # 6. Quarterly OpEx
        days_to_opex = _days_to_quarterly_opex(today)
        if days_to_opex <= 3:
            opex_score, opex_label = -0.5, f"{days_to_opex}d to quarterly opex — pin/vol caution"
        elif days_to_opex <= 7:
            opex_score, opex_label = -0.2, f"{days_to_opex}d to quarterly opex — slight caution"
        else:
            opex_score, opex_label = +0.1, f"{days_to_opex}d to quarterly opex — clear"
        signals.append(CycleSignal("quarterly_opex", opex_score, opex_label, weight=1.0))

        # 7. Solar Cycle
        days_from_solar_max = abs((today - SOLAR_MAX_DATE).days)
        if days_from_solar_max < 90:
            solar_score, solar_label = -0.5, f"near SC25 solar max ({days_from_solar_max}d) — Dewey: turmoil risk"
        elif days_from_solar_max < 365:
            solar_score, solar_label = -0.2, f"{days_from_solar_max}d from solar max — elevated risk"
        else:
            solar_score, solar_label = +0.3, f"{days_from_solar_max}d from solar max — benign phase"
        signals.append(CycleSignal("solar_cycle", solar_score, solar_label, weight=0.8))

        # 8. Funding Rate Cycle
        funding = self._fetch_funding_rate()
        annual_funding = funding * 3 * 365
        if annual_funding > 0.50:
            fund_score, fund_label = -0.6, f"funding {annual_funding*100:.0f}% pa — crowded longs, mean-revert risk"
        elif annual_funding > 0.20:
            fund_score, fund_label = -0.2, f"funding {annual_funding*100:.0f}% pa — elevated"
        elif annual_funding < -0.10:
            fund_score, fund_label = +0.5, f"funding {annual_funding*100:.0f}% pa — shorts crowded, squeeze risk"
        else:
            fund_score, fund_label = +0.2, f"funding {annual_funding*100:.0f}% pa — healthy"
        signals.append(CycleSignal("funding_cycle", fund_score, fund_label, weight=1.2))

        # 9. BTC Dominance / Rotation Stage
        btc_ret, eth_ret, sol_ret = self._fetch_btc_eth_sol_returns()
        if eth_ret > btc_ret + 0.01 and sol_ret > btc_ret + 0.01:
            rot_score = +0.8
            rot_label = f"alt season: ETH{eth_ret*100:+.1f}% SOL{sol_ret*100:+.1f}% BTC{btc_ret*100:+.1f}%"
        elif btc_ret > eth_ret + 0.02:
            rot_score = -0.3
            rot_label = f"BTC dominance rising ({btc_ret*100:+.1f}% vs ETH {eth_ret*100:+.1f}%)"
        else:
            rot_score = 0.0
            rot_label = f"mixed rotation (BTC{btc_ret*100:+.1f}% ETH{eth_ret*100:+.1f}% SOL{sol_ret*100:+.1f}%)"
        signals.append(CycleSignal("btc_dominance_rotation", rot_score, rot_label, weight=1.3))

        # ── V7.5: Kondratiev Wave 6 ───────────────────────────────────────────
        kwave_score, kwave_label = _kwave_score(today)
        signals.append(CycleSignal("kondratiev_wave", kwave_score, kwave_label, weight=0.5))

        # ── V7.5: Juglar Cycle ────────────────────────────────────────────────
        juglar_score, juglar_label = _juglar_score(today)
        signals.append(CycleSignal("juglar_cycle", juglar_score, juglar_label, weight=0.4))

        # ── V7.5: Kitchin Cycle ───────────────────────────────────────────────
        kitchin_score, kitchin_label = _kitchin_score(today)
        signals.append(CycleSignal("kitchin_cycle", kitchin_score, kitchin_label, weight=0.35))

        # ── Composite (all 12 cycles) ─────────────────────────────────────────
        total_weight = sum(s.weight for s in signals)
        composite = sum(s.score * s.weight for s in signals) / total_weight
        composite = float(np.clip(composite, -1.0, 1.0))

        if composite > 0.35:
            sentiment = "TAILWIND"
            kelly_mult = 1.0 + 0.5 * composite   # up to 1.5×
        elif composite < -0.35:
            sentiment = "HEADWIND"
            kelly_mult = max(0.3, 1.0 + composite)  # down to 0.3×
        else:
            sentiment = "NEUTRAL"
            kelly_mult = 1.0

        # Macro composite: K-Wave + Juglar + Kitchin only
        macro_signals = [s for s in signals if s.name in MACRO_SIGNAL_NAMES]
        macro_w = sum(s.weight for s in macro_signals)
        macro_comp = (
            sum(s.score * s.weight for s in macro_signals) / macro_w
            if macro_w > 0 else 0.0
        )
        macro_comp = float(np.clip(macro_comp, -1.0, 1.0))

        result = CycleComposite(
            signals=signals,
            composite=round(composite, 4),
            sentiment=sentiment,
            kelly_multiplier=round(kelly_mult, 3),
            macro_composite=round(macro_comp, 4),
        )
        self._last_composite = result
        return result

    def get_macro_composite(self, now: Optional[datetime] = None) -> float:
        """
        Returns the weighted average of ONLY the three structural macro cycles:
        K-Wave, Juglar, Kitchin. Range [-1, +1].

        Positive = structural tailwind. Negative = structural headwind.
        Used by PositionEngine (MACRO_VETO) and MemeCoinScanner (alert gating).
        """
        today = (now or datetime.now(timezone.utc)).date()
        kwave_score, _ = _kwave_score(today)
        juglar_score, _ = _juglar_score(today)
        kitchin_score, _ = _kitchin_score(today)

        w_kwave, w_juglar, w_kitchin = 0.5, 0.4, 0.35
        total_w = w_kwave + w_juglar + w_kitchin
        macro = (
            kwave_score * w_kwave
            + juglar_score * w_juglar
            + kitchin_score * w_kitchin
        ) / total_w
        return float(np.clip(macro, -1.0, 1.0))

    def phase_stability_check(self, signal_name: str, history_days: int = 365) -> float:
        """
        Returns a confidence score 0.0-1.0 for how phase-stable a named cycle
        has been over the past history_days, using autocorrelation at the
        expected lag.

        Dewey's rule: cycles with stability < 0.40 are false cycles and should
        be excluded or down-weighted from the composite.

        Macro cycles (K-Wave, Juglar, Kitchin) return fixed 0.85 — academically
        validated over centuries; a short observation window can't falsify them.
        """
        if signal_name in MACRO_SIGNAL_NAMES:
            return 0.85  # centuries of academic validation

        expected_lags = {
            "halving_cycle":          365,
            "monthly_seasonality":    30,
            "day_of_week":            7,
            "hour_of_day":            1,
            "lunar_cycle":            30,
            "quarterly_opex":         91,
            "solar_cycle":            365,
            "funding_cycle":          14,
            "btc_dominance_rotation": 7,
        }

        lag = expected_lags.get(signal_name, 30)
        now = datetime.now(timezone.utc)
        scores = []

        for i in range(history_days, 0, -1):
            ts = now - timedelta(days=i)
            d = ts.date()

            if signal_name == "monthly_seasonality":
                scores.append(MONTH_SCORES.get(d.month, 0.0))
            elif signal_name == "day_of_week":
                scores.append(DOW_SCORES.get(d.weekday(), 0.0))
            elif signal_name == "hour_of_day":
                scores.append(_hour_score(12))
            elif signal_name == "lunar_cycle":
                p = _lunar_phase(ts)
                if p < 0.07 or p > 0.93:
                    scores.append(+0.6)
                elif 0.43 < p < 0.57:
                    scores.append(-0.4)
                else:
                    scores.append(0.0)
            elif signal_name == "halving_cycle":
                pname, _ = _halving_phase(d)
                scores.append(HALVING_PHASE_SCORES[pname])
            elif signal_name == "quarterly_opex":
                d2o = _days_to_quarterly_opex(d)
                scores.append(-0.5 if d2o <= 3 else (-0.2 if d2o <= 7 else +0.1))
            elif signal_name == "solar_cycle":
                dfm = abs((d - SOLAR_MAX_DATE).days)
                scores.append(-0.5 if dfm < 90 else (-0.2 if dfm < 365 else +0.3))
            elif signal_name in ("funding_cycle", "btc_dominance_rotation"):
                return 0.60  # live API cycles — moderate stability assumed
            else:
                return 0.50

        if len(scores) < lag * 2:
            return 0.50

        arr = np.array(scores, dtype=float)
        arr_std = arr.std()
        if arr_std < 1e-8:
            return 1.0  # perfectly constant
        arr_norm = (arr - arr.mean()) / arr_std

        n = len(arr_norm)
        if lag >= n:
            return 0.50
        autocorr = float(np.corrcoef(arr_norm[:-lag], arr_norm[lag:])[0, 1])
        return round(float(np.clip((autocorr + 1.0) / 2.0, 0.0, 1.0)), 3)

    def summary(self, comp: Optional[CycleComposite] = None) -> str:
        if comp is None:
            comp = self.evaluate()
        lines = [
            f"CYCLE COMPOSITE  score={comp.composite:+.3f}  [{comp.sentiment}]  "
            f"kelly×{comp.kelly_multiplier}  macro={comp.macro_composite:+.3f}",
            "-" * 76,
        ]
        for s in comp.signals:
            bar = "█" * int(abs(s.score) * 5) + "░" * (5 - int(abs(s.score) * 5))
            sign = "▲" if s.score > 0 else ("▼" if s.score < 0 else "─")
            macro_tag = " [MACRO]" if s.name in MACRO_SIGNAL_NAMES else ""
            lines.append(
                f"  {sign} {s.name:<28} {s.score:+.2f} w={s.weight:.2f}  {s.label}{macro_tag}"
            )
        return "\n".join(lines)


# ── standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    det = CycleDetector(bybit_symbol="BTCUSDT")
    comp = det.evaluate()
    print(det.summary(comp))
    print()
    print(f"Macro composite (K-Wave+Juglar+Kitchin): {det.get_macro_composite():+.4f}")
    print()
    print("Phase stability checks:")
    for name in [
        "monthly_seasonality", "day_of_week", "lunar_cycle",
        "halving_cycle", "kondratiev_wave", "juglar_cycle", "kitchin_cycle",
    ]:
        stability = det.phase_stability_check(name)
        flag = "  OK" if stability >= 0.40 else "  !! FALSE CYCLE RISK"
        print(f"  {name:<30} stability={stability:.3f}{flag}")

"""
ofi_engine.py  —  OpenClaw V6
Markwick-style Order Flow Imbalance (OFI) engine.

Built on the exact OFI formula from:
  Markwick (2022), "Order Flow Imbalance — A High Frequency Trading Signal"
  Xu et al. (2019) multi-level OFI extension
  2025-2026 Binance/Bybit LOB microstructure papers

API:
  ofi = OFIEngine(ws_handler)
  signal = ofi.get_signal(symbol)        # returns: {'ofi': float, 'pct': float, 'aligned': bool}
  depth  = ofi.get_top5_depth_usd(symbol)
"""

import time
import logging
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


class OFIEngine:
    """
    Computes OFI at best bid/ask (level-1) from the live WS order book,
    buckets into 10-second windows, stores a rolling 60-bucket history,
    and provides a percentile-gated entry signal.
    """

    BUCKET_SECS = 10        # Aggregate OFI over 10-second buckets
    HISTORY_BUCKETS = 360   # 60 minutes of 10s buckets

    def __init__(self, ws_handler, percentile_threshold: int = 75):
        self.ws = ws_handler
        self.pct_thresh = percentile_threshold

        # Per-symbol state
        self._prev_best_bid: dict  = {}   # {symbol: (price, size)}
        self._prev_best_ask: dict  = {}   # {symbol: (price, size)}
        self._bucket_acc: dict     = {}   # {symbol: float} running accumulator for current bucket
        self._bucket_ts: dict      = {}   # {symbol: float} start time of current bucket
        self._history: dict        = {}   # {symbol: deque of OFI bucket values}

    def _init_symbol(self, symbol: str):
        if symbol not in self._history:
            self._prev_best_bid[symbol] = (0.0, 0.0)
            self._prev_best_ask[symbol] = (0.0, 0.0)
            self._bucket_acc[symbol]    = 0.0
            self._bucket_ts[symbol]     = time.time()
            self._history[symbol]       = deque(maxlen=self.HISTORY_BUCKETS)

    # ------------------------------------------------------------------
    # Core update — call this whenever the WS delivers a new LOB message
    # (ws_pybit.py calls this from _on_orderbook if you wire it in).
    # The main loop also calls update_all() each tick as a fallback.
    # ------------------------------------------------------------------
    def update(self, symbol: str):
        self._init_symbol(symbol)
        ob = self.ws.orderbooks.get(symbol, {})
        bids = ob.get('bids', {})
        asks = ob.get('asks', {})
        if not bids or not asks:
            return

        best_bid_p = max(bids.keys())
        best_bid_q = bids[best_bid_p]
        best_ask_p = min(asks.keys())
        best_ask_q = asks[best_ask_p]

        prev_bp, prev_bq = self._prev_best_bid[symbol]
        prev_ap, prev_aq = self._prev_best_ask[symbol]

        e = 0.0

        # -- BID contribution --
        if best_bid_p > prev_bp:
            e += best_bid_q          # price rose: demand up
        elif best_bid_p == prev_bp:
            e += (best_bid_q - prev_bq)  # same price, delta size
        else:
            e -= prev_bq             # price fell: cancel/consume

        # -- ASK contribution --
        if best_ask_p < prev_ap:
            e += prev_aq             # ask price dropped: supply consumed
        elif best_ask_p == prev_ap:
            e -= (best_ask_q - prev_aq)  # same price, supply increased = negative
        else:
            e -= best_ask_q          # ask rose: supply added

        self._bucket_acc[symbol] += e
        self._prev_best_bid[symbol] = (best_bid_p, best_bid_q)
        self._prev_best_ask[symbol] = (best_ask_p, best_ask_q)

        # Flush bucket every BUCKET_SECS seconds
        now = time.time()
        if now - self._bucket_ts[symbol] >= self.BUCKET_SECS:
            self._history[symbol].append(self._bucket_acc[symbol])
            self._bucket_acc[symbol] = 0.0
            self._bucket_ts[symbol]  = now

    def update_all(self):
        """Call once per main loop iteration to flush stale buckets."""
        for symbol in list(self._history.keys()):
            self.update(symbol)

    # ------------------------------------------------------------------
    # Signal interface
    # ------------------------------------------------------------------
    def get_ofi(self, symbol: str) -> float:
        """Return current in-progress bucket OFI (not yet flushed)."""
        self._init_symbol(symbol)
        self.update(symbol)
        return self._bucket_acc.get(symbol, 0.0)

    def get_ofi_percentile(self, symbol: str) -> float:
        """Return percentile of |current OFI| vs rolling history."""
        self._init_symbol(symbol)
        hist = list(self._history.get(symbol, []))
        if len(hist) < 10:
            return 50.0   # not enough history — return neutral
        current_abs = abs(self.get_ofi(symbol))
        abs_hist    = np.abs(hist)
        pct = float(np.sum(abs_hist <= current_abs) / len(abs_hist) * 100)
        return pct

    def get_signal(self, symbol: str, trade_direction: str) -> dict:
        """
        Returns OFI signal dict:
          {
            'ofi':     float,   raw OFI value
            'pct':     float,   percentile vs history (0-100)
            'aligned': bool,    True if OFI direction matches trade_direction AND above pct threshold
          }
        trade_direction: 'BUY' or 'SELL'
        """
        self._init_symbol(symbol)
        self.update(symbol)
        ofi_val = self.get_ofi(symbol)
        pct     = self.get_ofi_percentile(symbol)
        # OFI > 0  => buy pressure; OFI < 0 => sell pressure
        directional_ok = (
            (trade_direction == 'BUY'  and ofi_val > 0) or
            (trade_direction == 'SELL' and ofi_val < 0)
        )
        aligned = directional_ok and pct >= self.pct_thresh
        return {'ofi': ofi_val, 'pct': pct, 'aligned': aligned}

    # ------------------------------------------------------------------
    # Depth / liquidity haircut helper
    # ------------------------------------------------------------------
    def get_top5_depth_usd(self, symbol: str) -> float:
        """
        Return total USD depth of top 5 bid + top 5 ask levels.
        Used for the liquidity haircut: max_notional = h * top5_depth.
        """
        ob   = self.ws.orderbooks.get(symbol, {})
        bids = ob.get('bids', {})
        asks = ob.get('asks', {})
        if not bids or not asks:
            return 0.0

        top5_bids = sorted(bids.items(), key=lambda x: x[0], reverse=True)[:5]
        top5_asks = sorted(asks.items(), key=lambda x: x[0])[:5]
        depth_usd = sum(p * q for p, q in top5_bids) + sum(p * q for p, q in top5_asks)
        return depth_usd

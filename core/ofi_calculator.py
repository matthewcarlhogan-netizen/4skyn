"""
OpenClaw V7 — OFI Calculator

Fixes V6's testnet-orderbook problem (synthetic/illiquid data).
Uses mainnet L1 by default. Computes a proper percentile rank
(not the binary 0/100 V6 produced).
"""
from __future__ import annotations
from collections import deque
from typing import Tuple
import numpy as np
from pybit.unified_trading import HTTP


class OFICalculator:
    def __init__(self, testnet: bool = False, window_size: int = 20):
        self.http = HTTP(testnet=testnet)
        self.window_size = window_size
        self.lob_buf: deque = deque(maxlen=window_size)
        self.history: deque = deque(maxlen=200)  # rolling cumulative-OFI history for rank

    def _snapshot(self, symbol: str):
        try:
            r = self.http.get_orderbook(category="linear", symbol=symbol, limit=10)
            if r["retCode"] != 0:
                return None
            b = r["result"]["b"]; a = r["result"]["a"]
            if not b or not a:
                return None
            return (float(b[0][0]), float(b[0][1]), float(a[0][0]), float(a[0][1]))
        except Exception:
            return None

    def compute(self, symbol: str) -> Tuple[float, float]:
        snap = self._snapshot(symbol)
        if snap is None:
            return 0.0, 50.0
        self.lob_buf.append(snap)
        if len(self.lob_buf) < 2:
            return 0.0, 50.0

        deltas = []
        for i in range(1, len(self.lob_buf)):
            pb, qb, pa, qa = self.lob_buf[i - 1]
            nb, nqb, na, nqa = self.lob_buf[i]
            d = 0.0
            if nb > pb or (nb == pb and nqb > qb):
                d += (nqb - qb) if nb == pb else nqb
            elif nb < pb or (nb == pb and nqb < qb):
                d -= (qb - nqb) if nb == pb else qb
            if na < pa or (na == pa and nqa > qa):
                d -= (nqa - qa) if na == pa else nqa
            elif na > pa or (na == pa and nqa < qa):
                d += (qa - nqa) if na == pa else qa
            deltas.append(d)

        cum_ofi = float(np.sum(deltas))
        self.history.append(cum_ofi)

        # Real percentile rank (0-100), not a binary indicator
        arr = np.array(self.history)
        rank = float((arr <= cum_ofi).mean() * 100.0)
        return cum_ofi, rank

    def is_aligned(self, direction: int, symbol: str, threshold: float = 75.0) -> bool:
        ofi, rank = self.compute(symbol)
        if direction > 0:
            return rank >= threshold and ofi > 0
        return rank <= (100 - threshold) and ofi < 0

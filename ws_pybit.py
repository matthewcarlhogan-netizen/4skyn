import asyncio
import json
import logging
import time
from pybit.unified_trading import WebSocket

logger = logging.getLogger(__name__)

class BybitWSHandler:
    def __init__(self, symbols=None, testnet=True):
        self.symbols     = symbols or ['BTCUSDT', 'ETHUSDT']
        self.testnet     = testnet
        self.orderbooks  = {s: {'bids': {}, 'asks': {}} for s in self.symbols}
        self.tickers     = {s: {} for s in self.symbols}
        self.last_fills  = []
        self.connected   = False
        self.ws          = None
        self._retries    = 0
        self._max_retry  = 5

    def _on_orderbook(self, msg):
        try:
            topic  = msg.get('topic', '')
            symbol = topic.split('.')[-1]
            data   = msg.get('data', {})
            mtype  = msg.get('type', '')
            if symbol not in self.orderbooks:
                return
            ob = self.orderbooks[symbol]
            if mtype == 'snapshot':
                ob['bids'] = {float(p): float(q) for p, q in data.get('b', [])}
                ob['asks'] = {float(p): float(q) for p, q in data.get('a', [])}
            elif mtype == 'delta':
                for p, q in data.get('b', []):
                    fp, fq = float(p), float(q)
                    if fq == 0:
                        ob['bids'].pop(fp, None)
                    else:
                        ob['bids'][fp] = fq
                for p, q in data.get('a', []):
                    fp, fq = float(p), float(q)
                    if fq == 0:
                        ob['asks'].pop(fp, None)
                    else:
                        ob['asks'][fp] = fq
            ob['ts'] = msg.get('ts', time.time() * 1000)
        except Exception as e:
            logger.error(f"Orderbook parse error: {e}")

    def _on_ticker(self, msg):
        try:
            data   = msg.get('data', {})
            symbol = data.get('symbol', '')
            if symbol in self.tickers:
                self.tickers[symbol] = {
                    'mark_price':  float(data.get('markPrice', 0)),
                    'last_price':  float(data.get('lastPrice', 0)),
                    'bid1':        float(data.get('bid1Price', 0)),
                    'ask1':        float(data.get('ask1Price', 0)),
                    'ts':          time.time()
                }
        except Exception as e:
            logger.error(f"Ticker parse error: {e}")

    def _on_execution(self, msg):
        try:
            for fill in msg.get('data', []):
                record = {
                    'symbol':     fill.get('symbol'),
                    'side':       fill.get('side'),
                    'price':      float(fill.get('execPrice', 0)),
                    'qty':        float(fill.get('execQty', 0)),
                    'fee':        float(fill.get('execFee', 0)),
                    'order_id':   fill.get('orderId'),
                    'ts':         fill.get('execTime')
                }
                self.last_fills.append(record)
                if len(self.last_fills) > 100:
                    self.last_fills.pop(0)
                logger.info(f"FILL: {record['symbol']} {record['side']} "
                            f"{record['qty']} @ {record['price']}")
        except Exception as e:
            logger.error(f"Execution parse error: {e}")

    def connect(self, api_key=None, api_secret=None):
        try:
            self.ws = WebSocket(
                testnet=self.testnet,
                channel_type='linear',
                api_key=api_key,
                api_secret=api_secret
            )
            for sym in self.symbols:
                self.ws.orderbook_stream(50, sym, self._on_orderbook)
                self.ws.ticker_stream(sym, self._on_ticker)
            if api_key:
                self.ws.execution_stream(self._on_execution)
            self.connected  = True
            self._retries   = 0
            logger.info(f"WS connected | testnet={self.testnet} | symbols={self.symbols}")
        except Exception as e:
            logger.error(f"WS connect failed: {e}")
            self.connected = False

    def get_mid_price(self, symbol):
        ob  = self.orderbooks.get(symbol, {})
        bids = ob.get('bids', {})
        asks = ob.get('asks', {})
        if not bids or not asks:
            t = self.tickers.get(symbol, {})
            return t.get('mark_price', 0)
        best_bid = max(bids.keys())
        best_ask = min(asks.keys())
        return (best_bid + best_ask) / 2

    def get_spread_bps(self, symbol):
        ob   = self.orderbooks.get(symbol, {})
        bids = ob.get('bids', {})
        asks = ob.get('asks', {})
        if not bids or not asks:
            return 999
        best_bid = max(bids.keys())
        best_ask = min(asks.keys())
        mid      = (best_bid + best_ask) / 2
        return ((best_ask - best_bid) / mid) * 10000

    def get_slippage_estimate(self, symbol, side, qty_usd):
        ob     = self.orderbooks.get(symbol, {})
        levels = ob.get('asks', {}) if side == 'BUY' else ob.get('bids', {})
        if not levels:
            return 0.001
        mid    = self.get_mid_price(symbol)
        if mid <= 0:
            return 0.001
        qty_base  = qty_usd / mid
        sorted_lv = sorted(levels.items(), key=lambda x: x[0],
                           reverse=(side == 'SELL'))
        filled    = 0.0
        cost      = 0.0
        for price, size in sorted_lv:
            take    = min(size, qty_base - filled)
            cost   += take * price
            filled += take
            if filled >= qty_base:
                break
        if filled <= 0:
            return 0.001
        avg_fill = cost / filled
        return abs(avg_fill - mid) / mid

    def is_stale(self, symbol, max_age_s=10):
        ob = self.orderbooks.get(symbol, {})
        ts = ob.get('ts', 0)
        return (time.time() * 1000 - ts) > (max_age_s * 1000)

    def disconnect(self):
        if self.ws:
            try:
                self.ws.exit()
            except Exception:
                pass
        self.connected = False
        logger.info("WS disconnected")

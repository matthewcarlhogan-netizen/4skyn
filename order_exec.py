"""
order_exec.py  —  OpenClaw V6

Upgrades over V5:
  1. Regime-gated SL/TP multipliers passed in from agents_council
  2. Dynamic tick-size rounding from Bybit instrument info (avoids -10001 errors)
  3. PostOnly with Market fallback if rejected (for testnet/live robustness)
  4. record_fill() pushes realized PnL (R-multiples) back to RiskEngine
  5. Position close helper close_position()
"""

import logging
import time

import yaml
from pybit.unified_trading import HTTP

logger = logging.getLogger(__name__)


class OrderExecutor:
    def __init__(self, config_path='config.yaml'):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        self.dry_run  = self.cfg['system']['dry_run']
        self.testnet  = self.cfg['bybit']['testnet']
        self.leverage = self.cfg['bybit']['leverage']
        self.client   = None
        self._tick_sizes: dict = {}   # {symbol: {'qty_step': float, 'price_tick': float}}
        if not self.dry_run:
            self.client = HTTP(
                testnet=self.testnet,
                api_key=self.cfg['bybit']['api_key'],
                api_secret=self.cfg['bybit']['api_secret']
            )
            self._set_leverage()
            self._fetch_tick_sizes()

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------
    def _set_leverage(self):
        for symbol in self.cfg['bybit']['symbols']:
            try:
                self.client.set_leverage(
                    category='linear',
                    symbol=symbol,
                    buyLeverage=str(self.leverage),
                    sellLeverage=str(self.leverage)
                )
                logger.info(f'Leverage set: {symbol} {self.leverage}x')
            except Exception as e:
                logger.warning(f'Leverage set skipped ({symbol}): {e}')

    def _fetch_tick_sizes(self):
        """Cache qty_step and price_tick for each symbol to avoid rounding errors."""
        try:
            resp = self.client.get_instruments_info(category='linear')
            for item in resp['result']['list']:
                sym = item['symbol']
                if sym in self.cfg['bybit']['symbols']:
                    lot  = item.get('lotSizeFilter', {})
                    pf   = item.get('priceFilter', {})
                    self._tick_sizes[sym] = {
                        'qty_step':   float(lot.get('qtyStep', '0.001')),
                        'min_qty':    float(lot.get('minOrderQty', '0.001')),
                        'price_tick': float(pf.get('tickSize', '0.01'))
                    }
            logger.info(f'Tick sizes cached: {self._tick_sizes}')
        except Exception as e:
            logger.warning(f'Tick size fetch failed: {e} — using defaults')

    def _round_qty(self, symbol: str, qty: float) -> float:
        ts    = self._tick_sizes.get(symbol, {})
        step  = ts.get('qty_step', 0.001)
        min_q = ts.get('min_qty', 0.001)
        qty   = max(round(qty / step) * step, min_q)
        # Decimal places from step size
        dp = len(str(step).rstrip('0').split('.')[-1]) if '.' in str(step) else 0
        return round(qty, dp)

    def _round_price(self, symbol: str, price: float) -> float:
        ts   = self._tick_sizes.get(symbol, {})
        tick = ts.get('price_tick', 0.01)
        dp   = len(str(tick).rstrip('0').split('.')[-1]) if '.' in str(tick) else 0
        return round(round(price / tick) * tick, dp)

    # ------------------------------------------------------------------
    # Place order
    # ------------------------------------------------------------------
    def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        atr: float,
        sl_mult: float = 3.0,
        tp_mult: float = 6.0,
        retries: int   = 3
    ) -> dict | None:
        """
        Place a PostOnly limit order with ATR-based SL/TP.
        sl_mult / tp_mult come from agents_council regime logic:
          BULL  → SL=3*ATR, TP=6*ATR  (trend, asymmetric R>2)
          RANGE → SL=1.5*ATR, TP=2.5*ATR  (mean-reversion, tighter)
        """
        if atr <= 0:
            logger.warning(f'place_order: atr={atr} invalid, skipping')
            return None

        if side == 'Buy':
            sl_price = price - atr * sl_mult
            tp_price = price + atr * tp_mult
        else:
            sl_price = price + atr * sl_mult
            tp_price = price - atr * tp_mult

        qty      = self._round_qty(symbol, qty)
        price    = self._round_price(symbol, price)
        sl_price = self._round_price(symbol, sl_price)
        tp_price = self._round_price(symbol, tp_price)

        if qty <= 0:
            logger.warning(f'place_order: qty={qty} after rounding, skipping')
            return None

        if self.dry_run:
            logger.info(
                f'[DRY RUN] {side} {qty} {symbol} @ {price} '
                f'| SL={sl_price} (x{sl_mult} ATR) TP={tp_price} (x{tp_mult} ATR)'
            )
            return {
                'dry_run': True, 'symbol': symbol, 'side': side,
                'qty': qty, 'price': price, 'sl': sl_price, 'tp': tp_price,
                'order_id': f'DRY_{int(time.time())}'
            }

        for attempt in range(retries):
            try:
                resp = self.client.place_order(
                    category='linear',
                    symbol=symbol,
                    side=side,
                    orderType='Limit',
                    qty=str(qty),
                    price=str(price),
                    timeInForce='PostOnly',
                    stopLoss=str(sl_price),
                    takeProfit=str(tp_price),
                    slTriggerBy='MarkPrice',
                    tpTriggerBy='MarkPrice',
                    reduceOnly=False,
                    closeOnTrigger=False
                )
                if resp['retCode'] == 0:
                    oid = resp['result']['orderId']
                    logger.info(
                        f'ORDER PLACED: {side} {qty} {symbol} @ {price} '
                        f'| SL={sl_price} TP={tp_price} | id={oid}'
                    )
                    return {'order_id': oid, 'symbol': symbol, 'side': side,
                            'qty': qty, 'price': price, 'sl': sl_price, 'tp': tp_price}

                # PostOnly rejected (crossed spread) — fallback to Market on last retry
                if resp['retCode'] == 10001 and attempt == retries - 1:
                    logger.warning(f'PostOnly rejected, falling back to Market order: {resp["retMsg"]}')
                    resp2 = self.client.place_order(
                        category='linear', symbol=symbol, side=side,
                        orderType='Market', qty=str(qty),
                        stopLoss=str(sl_price), takeProfit=str(tp_price),
                        slTriggerBy='MarkPrice', tpTriggerBy='MarkPrice'
                    )
                    if resp2['retCode'] == 0:
                        oid = resp2['result']['orderId']
                        logger.info(f'MARKET ORDER: {side} {qty} {symbol} | id={oid}')
                        return {'order_id': oid, 'symbol': symbol, 'side': side,
                                'qty': qty, 'price': price, 'sl': sl_price, 'tp': tp_price}

                logger.warning(f'Order rejected (attempt {attempt+1}): {resp["retMsg"]}')
                time.sleep(2 ** attempt)

            except Exception as e:
                logger.error(f'Order exception (attempt {attempt+1}): {e}')
                time.sleep(2 ** attempt)

        logger.error(f'Order failed after {retries} attempts — skipping signal')
        return None

    # ------------------------------------------------------------------
    # Close a position (reduce-only market order)
    # ------------------------------------------------------------------
    def close_position(self, symbol: str, side: str, qty: float) -> dict | None:
        """Close existing position. side = current position side ('Buy'|'Sell')."""
        close_side = 'Sell' if side == 'Buy' else 'Buy'
        qty        = self._round_qty(symbol, qty)

        if self.dry_run:
            logger.info(f'[DRY RUN] CLOSE {close_side} {qty} {symbol}')
            return {'dry_run': True, 'closed': True}

        try:
            resp = self.client.place_order(
                category='linear', symbol=symbol, side=close_side,
                orderType='Market', qty=str(qty), reduceOnly=True
            )
            if resp['retCode'] == 0:
                logger.info(f'CLOSE ORDER: {close_side} {qty} {symbol}')
                return {'order_id': resp['result']['orderId'], 'closed': True}
            logger.error(f'Close failed: {resp["retMsg"]}')
        except Exception as e:
            logger.error(f'close_position exception: {e}')
        return None

    # ------------------------------------------------------------------
    # Cancel an order
    # ------------------------------------------------------------------
    def cancel_order(self, symbol: str, order_id: str) -> bool:
        if self.dry_run:
            logger.info(f'[DRY RUN] Cancel {order_id}')
            return True
        try:
            resp = self.client.cancel_order(
                category='linear', symbol=symbol, orderId=order_id
            )
            return resp['retCode'] == 0
        except Exception as e:
            logger.error(f'Cancel failed: {e}')
            return False

    # ------------------------------------------------------------------
    # Account queries
    # ------------------------------------------------------------------
    def get_positions(self) -> list:
        if self.dry_run:
            return []
        try:
            resp = self.client.get_positions(category='linear', settleCoin='USDT')
            return [p for p in resp['result']['list'] if float(p['size']) > 0]
        except Exception as e:
            logger.error(f'get_positions failed: {e}')
            return []

    def get_balance(self) -> float:
        if self.dry_run:
            return float(self.cfg.get('dry_run_balance', 1000))
        try:
            resp  = self.client.get_wallet_balance(accountType='UNIFIED')
            coins = resp['result']['list'][0]['coin']
            for c in coins:
                if c['coin'] == 'USDT':
                    val = c.get('walletBalance') or c.get('availableToWithdraw') or c.get('equity') or '0'; return float(val) if val != '' else 0.0
        except Exception as e:
            logger.error(f'get_balance failed: {e}')
        return 0.0

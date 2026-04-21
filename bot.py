"""
OpenClaw V7 — main loop.
"""
from __future__ import annotations
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import yaml
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass

from pybit.unified_trading import HTTP
sys.path.insert(0, str(Path(__file__).parent))
from core.position_engine import PositionEngine
from core.telegram_notifier import TelegramNotifier

CFG_PATH = Path(__file__).parent / "config" / "settings.yaml"
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

MODE = os.getenv("TRADING_MODE", "testnet").lower()
if MODE == "testnet":
    API_KEY = os.getenv("BYBIT_TESTNET_API_KEY")
    API_SECRET = os.getenv("BYBIT_TESTNET_API_SECRET")
else:
    API_KEY = os.getenv("BYBIT_LIVE_API_KEY")
    API_SECRET = os.getenv("BYBIT_LIVE_API_SECRET")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT = os.getenv("TELEGRAM_CHAT_ID", "")

if not API_KEY or not API_SECRET:
    print("⚠️ WARNING: Missing Bybit API credentials. Running in PAPER mode.")
    API_KEY = "DUMMY"
    API_SECRET = "DUMMY"

_log_dir = CFG.get("infra", {}).get("log_dir", "logs")
Path(_log_dir).mkdir(exist_ok=True)
_logger = logging.getLogger("openclaw_v7")
_logger.setLevel(logging.DEBUG)
_last_date = None

def get_logger():
    global _last_date
    today = datetime.utcnow().strftime("%Y-%m-%d")
    if _last_date != today:
        for h in _logger.handlers[:]:
            _logger.removeHandler(h)
        fh = logging.FileHandler(f"{_log_dir}/openclaw_{today}.log")
        fh.setFormatter(logging.Formatter("%(asctime)s UTC | %(levelname)s | %(message)s"))
        _logger.addHandler(fh)
        _last_date = today
    return _logger

def get_equity(client) -> float:
    if getattr(client, '_api_key', None) == "DUMMY":
        return 1000.0
    try:
        r = client.get_wallet_balance(accountType="UNIFIED")
        if r.get("retCode") != 0:
            return 0.0
        lst = r.get("result", {}).get("list", [])
        if not lst:
            return 0.0
        eq = lst[0].get("totalEquity")
        return float(eq) if eq not in (None, "") else 0.0
    except Exception as e:
        print(f"wallet fetch error: {e}")
        return 0.0

def main():
    testnet_orders = (MODE == "testnet")
    client = HTTP(testnet=testnet_orders, api_key=API_KEY, api_secret=API_SECRET)
    engine = PositionEngine(CFG, testnet_orders=testnet_orders)
    engine.http_orders = client
    notifier = TelegramNotifier(TELEGRAM_TOKEN, TELEGRAM_CHAT)
    log = get_logger()

    exec_cfg = CFG.get("execution", {})
    risk_cfg = CFG.get("risk", {})
    startup = (
        f"BOOT mode={MODE} symbol={exec_cfg.get('symbol', 'BTCUSDT')} "
        f"lev={risk_cfg.get('max_leverage', 5)}x "
        f"risk/trade={risk_cfg.get('risk_per_trade_pct', 0.005)*100:.2f}% "
        f"maxDD={risk_cfg.get('max_drawdown_pct', 0.10)*100:.0f}%"
    )
    log.info(startup)
    notifier.send(startup)
    print("🦾 " + startup)

    last_pos_size = 0.0
    last_equity = 0.0
    last_heartbeat = 0.0
    pending_meta_signal_id = None

    while True:
        try:
            log = get_logger()
            equity = get_equity(client)
            if equity == 0.0:
                log.warning("WALLET_EMPTY")
                print("Wallet empty. Deposit at your Bybit account.")
                time.sleep(60)
                continue

            risk_cfg = CFG.get("risk", {})
            max_dd = risk_cfg.get("max_drawdown_pct", 0.10)
            hwm = max(engine.kelly.high_water_mark, equity)
            if equity < hwm * (1 - max_dd):
                msg = f"🛑 MAX DRAWDOWN HIT — equity ${equity:.2f} < {(1-max_dd)*100:.0f}% of HWM ${hwm:.2f}. Halting."
                log.error(msg)
                notifier.send(msg)
                print(msg)
                break

            signal = engine.decide_trade(equity)
            k_diag = engine.kelly.diagnostics()
            log.info(f"LOOP eq={equity:.2f} regime={signal.regime} conf={signal.confidence:.2f} side={signal.side} notional={signal.notional_usd:.2f} reason={signal.reason} kelly={k_diag['implied_f_star']:.3f} n_eff={k_diag['effective_n']}")
            print(f"${equity:,.2f} | {signal.regime} ({signal.confidence:.2f}) | {signal.side} ${signal.notional_usd:.0f} | {signal.reason}")

            infra_cfg = CFG.get("infra", {})
            if time.time() - last_heartbeat >= infra_cfg.get("telegram_heartbeat_s", 3600):
                notifier.send(f"💓 eq=${equity:.2f} | {signal.regime} {signal.confidence:.2f} | {signal.side} | kelly={k_diag['implied_f_star']:.3f}")
                last_heartbeat = time.time()

            cur = engine.get_current_position()
            exec_cfg = CFG.get("execution", {})
            risk_cfg = CFG.get("risk", {})
            if signal.side in ("Buy", "Sell") and cur["size"] == 0.0:
                qty = round(signal.notional_usd / signal.price, 3)
                if qty > 0:
                    try:
                        client.set_leverage(category="linear", symbol=exec_cfg.get("symbol", "BTCUSDT"), buyLeverage=str(risk_cfg.get("max_leverage", 5)), sellLeverage=str(risk_cfg.get("max_leverage", 5)))
                    except Exception:
                        pass
                    # In testnet mode, prices from mainnet klines don't match testnet mark price.
                # Recalculate SL/TP from testnet mark price using the same ATR-based %.
                order_kwargs = dict(
                    category="linear",
                    symbol=exec_cfg.get("symbol", "BTCUSDT"),
                    side=signal.side,
                    orderType="Market",
                    qty=qty,
                    positionIdx=0,
                )
                if testnet_orders:
                    # Testnet uses fake prices; derive SL/TP as % offsets from signal price
                    mainnet_px = signal.price
                    if mainnet_px and mainnet_px > 0:
                        sl_pct = abs(signal.sl_price - mainnet_px) / mainnet_px
                        tp_pct = abs(signal.tp_price - mainnet_px) / mainnet_px
                        # Get current testnet mark price
                        try:
                            tk = client.get_tickers(category="linear", symbol=exec_cfg.get("symbol", "BTCUSDT"))
                            tn_px = float(tk["result"]["list"][0]["lastPrice"])
                            if signal.side == "Buy":
                                tn_sl = round(tn_px * (1 - sl_pct), 2)
                                tn_tp = round(tn_px * (1 + tp_pct), 2)
                            else:
                                tn_sl = round(tn_px * (1 + sl_pct), 2)
                                tn_tp = round(tn_px * (1 - tp_pct), 2)
                            order_kwargs["stopLoss"] = str(tn_sl)
                            order_kwargs["takeProfit"] = str(tn_tp)
                        except Exception as sl_err:
                                log.warning(f"[WARN] Testnet SL/TP price fetch failed, order placed without SL/TP: {sl_err}")
                else:
                    order_kwargs["stopLoss"] = str(round(signal.sl_price, 2))
                    order_kwargs["takeProfit"] = str(round(signal.tp_price, 2))
                # Hard p_win floor: skip trade if meta-labeler is active and p_win < 0.35
                if signal.meta_p_win is not None and signal.meta_p_win < 0.35:
                    skip_msg = f"[SKIP] p_win={signal.meta_p_win:.2f} below floor 0.35 — trade blocked"
                    log.warning(skip_msg)
                    notifier.send(skip_msg)
                    continue
                order = client.place_order(**order_kwargs)
                pending_meta_signal_id = signal.meta_signal_id
                pwin_str = f" p_win={signal.meta_p_win:.2f}" if signal.meta_p_win is not None else ""
                msg = (f"OPEN {signal.side} {qty} @ ${signal.price:.2f} regime={signal.regime} notional=${signal.notional_usd:.0f}{pwin_str} SL={signal.sl_price:.2f} TP={signal.tp_price:.2f}")
                log.info("TRADE_OPEN " + msg)
                notifier.send(msg)
                print(msg)

            elif last_pos_size > 0 and cur["size"] == 0.0:
                pnl_pct = (equity - last_equity) / last_equity if last_equity else 0.0
                engine.kelly.record_trade(pnl_pct, last_equity)
                engine.on_trade_close(pnl_pct)
                if engine.meta is not None and pending_meta_signal_id:
                    engine.meta.record_outcome(pending_meta_signal_id, pnl_pct)
                    pending_meta_signal_id = None
                msg = f"CLOSE pnl={pnl_pct*100:+.2f}% equity=${equity:.2f}"
                log.info("TRADE_CLOSE " + msg)
                notifier.send(msg)
                print(msg)

            if signal.regime == "EXTREME_VOL":
                notifier.send("🚨 Extreme vol — pausing 5m")
                time.sleep(300)
                continue

            last_pos_size = cur["size"]
            last_equity = equity
            infra_cfg = CFG.get("infra", {})
            time.sleep(infra_cfg.get("loop_interval_s", 60))

        except KeyboardInterrupt:
            print("\n👋 stopped by user")
            break
        except Exception as e:
            log = get_logger()
            log.error(f"LOOP_ERROR {type(e).__name__}: {e}")
        err_str = str(e)
        print(f"loop error: {err_str}")
        # Send critical errors to Telegram (throttle ErrCode 10024 - known testnet restriction)
        if "10024" not in err_str:
            notifier.send(f"⚠️ LOOP ERROR: {err_str[:200]}")
            time.sleep(30)

if __name__ == "__main__":
    main()

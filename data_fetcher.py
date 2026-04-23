import time
import logging
import pandas as pd
from pybit.unified_trading import HTTP

logger = logging.getLogger(__name__)

def fetch_ohlcv(symbol='BTCUSDT', interval='60', limit=100, testnet=False):
    """
    Fetch recent OHLCV candles via pybit REST.
    Returns DataFrame with columns: timestamp, open, high, low, close, volume
    interval: '60' = 1h, '240' = 4h, 'D' = daily
    """
    try:
        client = HTTP(testnet=testnet)
        resp   = client.get_kline(
            category='linear',
            symbol=symbol,
            interval=interval,
            limit=limit
        )
        if resp['retCode'] != 0:
            logger.error(f"fetch_ohlcv error: {resp['retMsg']}")
            return None

        rows = resp['result']['list']
        if not rows:
            return None

        df = pd.DataFrame(rows, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        df = df.sort_values('timestamp').reset_index(drop=True)
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        return df

    except Exception as e:
        logger.error(f"fetch_ohlcv exception: {e}")
        return None


def fetch_ohlcv_history(symbol='BTCUSDT', interval='60', days=730, testnet=False):
    """
    Fetch full historical OHLCV by paginating backwards.
    Returns DataFrame sorted oldest → newest.
    """
    client     = HTTP(testnet=testnet)
    all_rows   = []
    end_time   = int(time.time() * 1000)
    target     = days * 24 * (60 // int(interval)) if interval.isdigit() else days

    print(f"Fetching {days} days of {interval}min candles for {symbol}...")

    while len(all_rows) < target:
        try:
            resp = client.get_kline(
                category='linear',
                symbol=symbol,
                interval=interval,
                limit=1000,
                end=end_time
            )
            if resp['retCode'] != 0:
                logger.error(f"History fetch error: {resp['retMsg']}")
                break
            rows = resp['result']['list']
            if not rows:
                break
            all_rows.extend(rows)
            earliest = int(rows[-1][0])
            end_time = earliest - 1
            print(f"  Fetched {len(rows)} candles, total: {len(all_rows)}")
            time.sleep(0.1)
        except Exception as e:
            logger.error(f"History fetch exception: {e}")
            break

    if not all_rows:
        return None

    df = pd.DataFrame(all_rows, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    df = df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    print(f"Total candles: {len(df)}")
    return df


if __name__ == '__main__':
    df = fetch_ohlcv_history(symbol='BTCUSDT', interval='60', days=730, testnet=False)
    if df is not None:
        df.to_csv('logs/BTCUSDT_1h_historical.csv', index=False)
        print(f"Saved to logs/BTCUSDT_1h_historical.csv")

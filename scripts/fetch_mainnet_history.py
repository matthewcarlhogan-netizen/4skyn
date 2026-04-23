#!/usr/bin/env python3
# Fetch REAL mainnet BTCUSDT 1h candles for HMM training.
# Public klines — NO API keys required. testnet=False.
# Overwrites logs/BTCUSDT_1h_historical.csv
# Run ONCE before training. Never run while bot is live.

from pybit.unified_trading import HTTP
import pandas as pd
from datetime import datetime, timedelta
import time
import os

def fetch_mainnet_klines(symbol='BTCUSDT', interval='60', days=730):
    # Public endpoint — no keys needed, testnet=False for real price data
    session = HTTP(testnet=False)

    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    all_data = []
    current_end = end_time

    print(f"Fetching {days} days of {interval}min MAINNET candles for {symbol}...")
    print("(No API keys needed — public endpoint)")

    while current_end > start_time:
        try:
            response = session.get_kline(
                category='linear',
                symbol=symbol,
                interval=interval,
                start=start_time,
                end=current_end,
                limit=1000
            )
            if response['retCode'] != 0:
                print(f"Error: {response['retMsg']}")
                break
            klines = response['result']['list']
            if not klines:
                break
            all_data.extend(klines)
            print(f"  Fetched {len(klines)} candles, total: {len(all_data)}")
            current_end = int(klines[-1][0]) - 1
            time.sleep(0.1)
        except Exception as e:
            print(f"Error: {e}")
            break

    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
        df[col] = df[col].astype(float)
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = df.drop_duplicates(subset='timestamp')

    # Sanity check — reject if price looks like testnet garbage
    max_price = df['close'].max()
    min_price = df['close'].min()
    print(f"\nPrice sanity check:")
    print(f"  Min close: ${min_price:,.0f}")
    print(f"  Max close: ${max_price:,.0f}")
    if max_price > 500_000:
        print("WARNING: Max price > $500k — possible testnet data contamination!")
        print("Check your network/DNS is not routing to testnet.")
    else:
        print("  Prices look like real mainnet data. Good.")

    print(f"\nTotal candles: {len(df)}")
    return df

if __name__ == "__main__":
    os.makedirs('logs', exist_ok=True)
    df = fetch_mainnet_klines('BTCUSDT', interval='60', days=730)
    out = 'logs/BTCUSDT_1h_historical.csv'
    df.to_csv(out, index=False)
    print(f"Saved to {out}")
    print("\nNow retrain the model:")
    print("  caffeinate -i python scripts/train_hmm.py")

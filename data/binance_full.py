import requests
import pandas as pd
import time
from datetime import datetime
import os
from multiprocessing import Pool, cpu_count

BASE_URL = "https://data-api.binance.vision/api/v3/klines"

def fetch_klines(symbol, interval, start_str, end_str, limit=1000):
    start_ts = int(pd.to_datetime(start_str).timestamp() * 1000)
    end_ts = int(pd.to_datetime(end_str).timestamp() * 1000)
    all_klines = []
    batch_count = 0

    while start_ts < end_ts:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ts,
            "endTime": end_ts,
            "limit": limit
        }

        response = requests.get(BASE_URL, params=params)
        batch_count += 1
        print(f"📦 {symbol} [{interval}] Batch {batch_count}: {datetime.utcfromtimestamp(start_ts/1000).strftime('%Y-%m-%d %H:%M')} — fetched {len(all_klines)} total rows")

        try:
            data = response.json()
        except Exception as e:
            print(f"❌ Failed to parse response: {e}")
            break

        if not isinstance(data, list) or len(data) == 0:
            print(f"⚠️ No more data returned at {datetime.utcfromtimestamp(start_ts / 1000)}")
            break

        all_klines += data
        start_ts = data[-1][0] + 1
        time.sleep(0.25)  # Avoid rate limit

    if not all_klines:
        raise Exception(f"No data fetched for {symbol} from {start_str} to {end_str}.")

    df = pd.DataFrame(all_klines, columns=[
        "Open time", "Open", "High", "Low", "Close", "Volume",
        "Close time", "Quote asset volume", "Number of trades",
        "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"
    ])
    df["Open time"] = pd.to_datetime(df["Open time"], unit='ms')
    df.set_index("Open time", inplace=True)
    return df[["Open", "High", "Low", "Close", "Volume"]].astype(float)

def download_yearly_data(args):
    symbol, interval, year, output_dir = args
    start_date = f"{year}-01-01"
    end_date = f"{year + 1}-01-01"
    filename = f"{symbol}_{interval}_{year}.csv"
    output_path = os.path.join(output_dir, filename)

    if os.path.exists(output_path):
        print(f"✅ Skipped (exists): {filename}")
        return

    try:
        print(f"\n📥 Starting download: {symbol} {interval} for {year}")
        df = fetch_klines(symbol, interval, start_date, end_date)
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(output_path)
        print(f"✅ Finished and saved: {filename} ({len(df)} rows)")
    except Exception as e:
        print(f"❌ Failed {symbol} {year}: {e}")


def download_all(symbols, intervals, start_year=2020, end_year=None, output_dir="1m data", parallel=True):
    if end_year is None:
        end_year = datetime.utcnow().year

    tasks = []
    for symbol in symbols:
        for interval in intervals:
            for year in range(start_year, end_year + 1):
                tasks.append((symbol, interval, year, output_dir))

    if parallel:
        try:
            with Pool(cpu_count()) as pool:
                for _ in pool.imap_unordered(download_yearly_data, tasks):
                    pass  # All logging is handled inside download_yearly_data
        except KeyboardInterrupt:
            print("\n🛑 Keyboard interrupt detected. Terminating downloads...")
            pool.terminate()
            pool.join()
    else:
        for task in tasks:
            try:
                download_yearly_data(task)
            except KeyboardInterrupt:
                print("\n🛑 Stopped by user.")
                break


if __name__ == "__main__":
    symbols = ["XLMUSDT", "XRPUSDT", "LINKUSDT", "HBARUSDT"]
    intervals = ["1m"]
    start_year = 2020
    end_year =  2025 # or None for current year

    download_all(symbols, intervals, start_year=start_year, end_year=end_year, output_dir="1m data", parallel=True)

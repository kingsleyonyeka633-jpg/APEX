"""Binance public bulk archive (data.binance.vision): monthly zipped klines back to each coin's listing date.
This is the best FREE source for deep history. Only coins that existed back then have it, so 10 years
exists for just a handful of coins (BTC, ETH, BNB, LTC, XRP, ADA, ...). Most mid-caps start 2019-2021.
"""
import io
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone

import pandas as pd

from . import exchange, store

ARCHIVE = "https://data.binance.vision/data/spot/monthly/klines"


def parse_bulk_csv(raw: bytes) -> pd.DataFrame:
    """Archive CSVs have no header; 12 columns. Since 2025 spot timestamps are in MICROseconds."""
    df = pd.read_csv(io.BytesIO(raw), header=None)
    ts = df[0].astype("int64")
    ts = ts.where(ts < 10**14, ts // 1000)
    out = pd.DataFrame({"ts": ts, "open": df[1], "high": df[2], "low": df[3], "close": df[4],
                        "volume": df[5], "quote_volume": df[7]})
    return out.astype({c: float for c in out.columns if c != "ts"})


def month_zip(symbol, interval, year, month, opener=urllib.request.urlopen):
    url = f"{ARCHIVE}/{symbol}/{interval}/{symbol}-{interval}-{year}-{month:02d}.zip"
    try:
        with opener(urllib.request.Request(url, headers={"User-Agent": "apex/2.0"}), timeout=60) as r:
            data = r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None                       # coin not listed yet that month (or delisted)
        raise
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        return parse_bulk_csv(z.read(z.namelist()[0]))


def months_between(start: datetime, end: datetime):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1


def download_symbol(symbol, interval, years, data_dir="data", opener=urllib.request.urlopen, fetch=exchange._get, log=print):
    """Bulk months up to last month, then REST for the current month. Resumable."""
    path = store.path_for(data_dir, symbol, interval)
    now = datetime.now(timezone.utc)
    first = datetime(now.year - years, now.month, 1, tzinfo=timezone.utc)
    done = store.last_ts(path)
    if done:
        first = max(first, datetime.fromtimestamp(done / 1000, timezone.utc).replace(day=1))
    last_full = datetime(now.year, now.month, 1, tzinfo=timezone.utc) - pd.Timedelta(days=1)
    got = 0
    for y, m in months_between(first, last_full):
        df = month_zip(symbol, interval, y, m, opener)
        if df is not None:
            store.save_merge(df, path)
            got += len(df)
        time.sleep(0.05)
    start_ms = (store.last_ts(path) or int(datetime(now.year, now.month, 1, tzinfo=timezone.utc).timestamp() * 1000)) + exchange.TF_MS[interval]
    tail = exchange.klines_range(symbol, interval, start_ms, int(time.time() * 1000), fetch=fetch)
    store.save_merge(tail, path)
    log(f"{symbol} {interval}: +{got + len(tail)} candles -> {path}")
    return got + len(tail)

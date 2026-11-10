"""Public market data from Binance (no API key). Default host is Binance's market-data mirror,
which works in more countries than api.binance.com. Override with APEX_BASE_URL.

NOTE: written against Binance's public docs. I could not reach the live API from my build sandbox,
so the paging/parsing logic is unit-tested against fake responses only. Run one small download first.
"""
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

BASE = os.getenv("APEX_BASE_URL", "https://data-api.binance.vision")
TF_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}


def _get(path, params=None, retries=6):
    url = f"{BASE}{path}" + (f"?{urllib.parse.urlencode(params)}" if params else "")
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "apex/2.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (418, 429):
                time.sleep(5 * (k + 1))          # rate limited: back off hard
            elif k == retries - 1:
                raise
            else:
                time.sleep(1.5 * (k + 1))
        except Exception:
            if k == retries - 1:
                raise
            time.sleep(1.5 * (k + 1))


def rows_to_frame(rows) -> pd.DataFrame:
    """Binance kline row: [openTime, o, h, l, c, vol, closeTime, quoteVol, trades, ...]"""
    if not rows:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume", "quote_volume"])
    df = pd.DataFrame([[r[0], r[1], r[2], r[3], r[4], r[5], r[7]] for r in rows],
                      columns=["ts", "open", "high", "low", "close", "volume", "quote_volume"])
    df["ts"] = df["ts"].astype("int64")
    return df.astype({c: float for c in df.columns if c != "ts"})


def klines(symbol, interval, start_ms=None, end_ms=None, limit=1000, fetch=_get) -> pd.DataFrame:
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_ms is not None:
        params["startTime"] = int(start_ms)
    if end_ms is not None:
        params["endTime"] = int(end_ms)
    return rows_to_frame(fetch("/api/v3/klines", params))


def klines_range(symbol, interval, start_ms, end_ms, fetch=_get, sleep=0.1) -> pd.DataFrame:
    """Forward pager, 1000 candles per request."""
    frames, cur = [], int(start_ms)
    while cur < end_ms:
        df = klines(symbol, interval, cur, end_ms, fetch=fetch)
        if df.empty:
            break
        frames.append(df)
        nxt = int(df.ts.iloc[-1]) + TF_MS[interval]
        if nxt <= cur:
            break
        cur = nxt
        time.sleep(sleep)
    return pd.concat(frames).drop_duplicates("ts").sort_values("ts") if frames else rows_to_frame([])


def recent_closed(symbol, interval, limit=500, fetch=_get) -> pd.DataFrame:
    """Latest candles with the still-forming one removed."""
    df = klines(symbol, interval, limit=limit, fetch=fetch)
    if df.empty:
        return df
    now_ms = int(time.time() * 1000)
    if int(df.ts.iloc[-1]) + TF_MS[interval] > now_ms:
        df = df.iloc[:-1]
    return df


def tickers_24h(fetch=_get):
    return fetch("/api/v3/ticker/24hr")


def last_price(symbol, fetch=_get) -> float:
    return float(fetch("/api/v3/ticker/price", {"symbol": symbol})["price"])

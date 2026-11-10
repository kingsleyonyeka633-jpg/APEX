"""CSV.gz candle store: data/<SYMBOL>_<tf>.csv.gz with columns ts(ms), open, high, low, close, volume, quote_volume."""
import os

import pandas as pd

COLS = ["ts", "open", "high", "low", "close", "volume", "quote_volume"]


def path_for(data_dir, symbol, tf):
    return os.path.join(data_dir, f"{symbol}_{tf}.csv.gz")


def save_merge(df: pd.DataFrame, path: str):
    """Append + dedupe + sort, so downloads are safe to re-run (resumable)."""
    if df is None or df.empty:
        return
    if os.path.exists(path):
        df = pd.concat([pd.read_csv(path), df[COLS]])
    df = df[COLS].drop_duplicates("ts").sort_values("ts")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False, compression="gzip")


def last_ts(path):
    return int(pd.read_csv(path, usecols=["ts"])["ts"].max()) if os.path.exists(path) else None


def load(data_dir, symbol, tf, start=None, end=None) -> pd.DataFrame:
    df = pd.read_csv(path_for(data_dir, symbol, tf))
    df.index = pd.to_datetime(df.pop("ts"), unit="ms", utc=True)
    if "quote_volume" not in df or df["quote_volume"].isna().all():
        df["quote_volume"] = df["volume"] * df["close"]
    df = df[~df.index.duplicated()].sort_index().astype(float)
    if start:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df.index < pd.Timestamp(end, tz="UTC")]
    return df


def quality_report(df: pd.DataFrame, tf_minutes: int = 15) -> dict:
    """Catch gaps/bad candles BEFORE they poison a backtest."""
    step = pd.Timedelta(minutes=tf_minutes)
    gaps = df.index.to_series().diff().dropna()
    bad = ((df.high < df.low) | (df.close > df.high) | (df.close < df.low) | (df.open > df.high) | (df.open < df.low)).sum()
    return {"rows": len(df), "start": str(df.index[0]), "end": str(df.index[-1]),
            "missing_bars": int(((gaps / step) - 1).clip(lower=0).sum()),
            "largest_gap_hours": float(gaps.max() / pd.Timedelta(hours=1)) if len(gaps) else 0.0,
            "bad_candles": int(bad), "zero_volume_bars": int((df.volume <= 0).sum())}

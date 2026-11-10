"""Pick a mid-cap trading universe from the exchange's 24h tickers (ranked by USDT volume)."""
import json
import os
import re
import time

from ..config import Config

STABLES = {"USDC", "FDUSD", "TUSD", "USDP", "DAI", "BUSD", "EUR", "GBP", "TRY", "BRL", "AEUR", "USD1", "USDE", "XUSD"}
LEVERAGED = re.compile(r"(UP|DOWN|BULL|BEAR)$")


def select_universe(tickers, cfg: Config):
    rows = []
    for t in tickers:
        sym = t["symbol"]
        if not sym.endswith(cfg.quote):
            continue
        base = sym[: -len(cfg.quote)]
        if base in STABLES or base in cfg.mega_caps or LEVERAGED.search(base):
            continue
        qv = float(t.get("quoteVolume", 0) or 0)
        if qv < cfg.min_daily_quote_volume or float(t.get("lastPrice", 0) or 0) <= 0:
            continue
        rows.append((sym, qv))
    rows.sort(key=lambda r: -r[1])
    # ranks are counted among non-mega coins, so rank_from skips the biggest of the remainder
    picked = rows[cfg.universe_rank_from - 1: cfg.universe_rank_to]
    return [{"symbol": s, "quote_volume_24h": v} for s, v in picked]


def save(universe, path="data/universe.json"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"updated": time.time(), "symbols": universe}, fh, indent=1)


def load(path="data/universe.json"):
    with open(path) as fh:
        return [x["symbol"] for x in json.load(fh)["symbols"]]


def ensure(cfg: Config, fetch_tickers, max_age_days: float = 7.0, log=print):
    """Create or refresh data/universe.json. Keeps the old list if the refresh fails."""
    path = os.path.join(cfg.data_dir, "universe.json")
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < max_age_days * 86400:
        return load(path)
    try:
        u = select_universe(fetch_tickers(), cfg)
        if not u:
            raise RuntimeError("no coins passed the filters")
        save(u, path)
        log(f"universe refreshed: {len(u)} coins")
        return [x["symbol"] for x in u]
    except Exception as e:
        if os.path.exists(path):
            log(f"universe refresh failed ({e}); keeping the old list")
            return load(path)
        raise

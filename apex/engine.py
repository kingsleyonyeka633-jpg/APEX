"""Portfolio backtester: replays history through the SAME Portfolio class the live bot uses."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import metrics
from .config import Config
from .portfolio import Portfolio
from .strategies import P, scan, to_1h


@dataclass
class Result:
    equity: pd.Series
    trades: pd.DataFrame
    metrics: dict
    by_layer: pd.DataFrame
    by_symbol: pd.DataFrame
    signals_seen: int
    signals_taken: int
    portfolio: Portfolio


def _ts(x):
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def compute_signals(data: dict, cfg: Config) -> dict:
    btc = data.get(cfg.btc_symbol)
    btc1h = to_1h(btc) if btc is not None else None
    params = {**P, "rr": cfg.reward_risk}
    return {s: scan(df, to_1h(df), btc1h if s != cfg.btc_symbol else None, params) for s, df in data.items()}


def run_backtest(data: dict, cfg: Config, start=None, end=None, ml=None, tradable=None, sigs=None) -> Result:
    sigs = sigs or compute_signals(data, cfg)
    syms = list(data)
    tradable = set(tradable or [s for s in syms if s != cfg.btc_symbol])
    timeline = sorted(set().union(*[df.index for df in data.values()]))
    timeline = pd.DatetimeIndex(timeline)
    if start:
        timeline = timeline[timeline >= _ts(start)]
    if end:
        timeline = timeline[timeline < _ts(end)]
    arr = {}
    for s in syms:
        df, sg = data[s], sigs[s]
        arr[s] = dict(o=df.open.to_numpy(), h=df.high.to_numpy(), l=df.low.to_numpy(), c=df.close.to_numpy(),
                      qv=df.quote_volume.to_numpy(), atr=sg.atr_pct.to_numpy(), sig=sg.signal.to_numpy(),
                      pos=data[s].index.get_indexer(timeline))
    pf = Portfolio(cfg, ml=ml)
    for k, ts in enumerate(timeline):
        bars = {}
        for s in syms:
            i = arr[s]["pos"][k]
            if i < 0:
                continue
            a = arr[s]
            atr = a["atr"][i] if np.isfinite(a["atr"][i]) else 0.01
            row = sigs[s].iloc[i].to_dict() if (a["sig"][i] == 1 and s in tradable) else None
            bars[s] = dict(o=a["o"][i], h=a["h"][i], l=a["l"][i], c=a["c"][i], qv=a["qv"][i], atr_pct=atr, sig=row)
        pf.step(ts, bars, tradable)
    eq = pd.Series([e for _, e in pf.eq_log], index=[t for t, _ in pf.eq_log])
    tr = pd.DataFrame(pf.trades, columns=["sym", "layer", "entry_ts", "exit_ts", "entry_px", "exit_px", "qty",
                                         "pnl", "r", "fees", "slippage", "reason", "bars"])
    m = metrics.compute(eq, tr, pf.cost, metrics.bars_per_year(timeline))
    return Result(eq, tr, m, metrics.breakdown(tr, "layer"), metrics.breakdown(tr, "sym"),
                  pf.signals_seen, pf.signals_taken, pf)

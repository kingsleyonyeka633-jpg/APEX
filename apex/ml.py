"""Optional ML filter: a plain logistic regression (numpy only) that scores each strategy signal by the
probability that the trade is profitable AFTER costs. It filters signals; it never creates them.

Honesty rules baked in: costs inside the labels, chronological split with a purge gap (labels look ahead
up to max_hold_bars), threshold picked on validation, verdict judged on an untouched test slice, and the
model only activates if it beats "take every signal" on that test slice.
"""
import json
import math
import os

import numpy as np
import pandas as pd

from .broker import FillModel
from .config import Config
from .engine import compute_signals
from .strategies import FEATURES, LAYERS

NAMES = FEATURES + [f"layer_{x}" for x in LAYERS]


def vectorize(row: dict) -> np.ndarray:
    v = [float(row.get(f, 0.0) or 0.0) for f in FEATURES]
    v += [1.0 if row.get("layer") == x else 0.0 for x in LAYERS]
    return np.nan_to_num(np.array(v, dtype=float))


def auc(y, p) -> float:
    y, p = np.asarray(y), np.asarray(p)
    pos, neg = (y == 1).sum(), (y == 0).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = pd.Series(p).rank().to_numpy()
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


class LogisticFilter:
    def __init__(self, w=None, b=0.0, mean=None, std=None, threshold=None, active=False, report=None):
        self.w, self.b, self.mean, self.std = w, b, mean, std
        self.threshold, self.active, self.report = threshold, active, report or {}

    def fit(self, X, y, l2=1.0, lr=0.1, iters=800):
        self.mean, self.std = X.mean(0), X.std(0) + 1e-9
        Z = (X - self.mean) / self.std
        self.w, self.b = np.zeros(Z.shape[1]), 0.0
        for _ in range(iters):
            p = 1 / (1 + np.exp(-(Z @ self.w + self.b)))
            self.w -= lr * (Z.T @ (p - y) / len(y) + l2 * self.w / len(y))
            self.b -= lr * (p - y).mean()
        return self

    def proba(self, X):
        return 1 / (1 + np.exp(-(((X - self.mean) / self.std) @ self.w + self.b)))

    def predict(self, row: dict) -> float:
        return float(self.proba(vectorize(row)[None, :])[0])

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as fh:
            json.dump({"w": self.w.tolist(), "b": self.b, "mean": self.mean.tolist(), "std": self.std.tolist(),
                       "threshold": self.threshold, "active": self.active, "report": self.report, "names": NAMES}, fh, indent=1)

    @classmethod
    def load(cls, path):
        d = json.load(open(path))
        return cls(np.array(d["w"]), d["b"], np.array(d["mean"]), np.array(d["std"]), d["threshold"], d["active"], d["report"])


def label_signal(o, h, l, c, i, stop, tp, atr_pct, cfg: Config, fm: FillModel):
    """Net R of one signal: enter next open, SL-first, costs included. Returns (r_net, exit_bar) or None."""
    if i + 1 >= len(o):
        return None
    entry = o[i + 1] * (1 + fm.half_spread(atr_pct))
    dist = entry - stop
    if dist <= 0 or entry >= tp:
        return None
    end = min(i + 1 + cfg.max_hold_bars, len(o) - 1)
    exit_px, maker, stopped, exit_bar = c[end], False, False, end
    for j in range(i + 1, end + 1):
        if l[j] <= stop:
            exit_px, stopped, exit_bar = min(o[j], stop), True, j
            break
        if h[j] >= tp * (1 + cfg.tp_trade_through_bps / 1e4):
            exit_px, maker, exit_bar = tp, True, j
            break
    exit_slip = 0.0 if maker else fm.half_spread(atr_pct) * (cfg.stop_slip_mult if stopped else 1.0)
    exit_px *= (1 - exit_slip)
    fees = cfg.taker_fee * entry + (cfg.maker_fee if maker else cfg.taker_fee) * exit_px
    return (exit_px - entry - fees) / dist, exit_bar


def build_dataset(data: dict, cfg: Config, sigs=None) -> pd.DataFrame:
    sigs = sigs or compute_signals(data, cfg)
    fm = FillModel(cfg)
    rows = []
    for sym, df in data.items():
        if sym == cfg.btc_symbol:
            continue
        sg = sigs[sym]
        o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
        for i in np.flatnonzero(sg.signal.to_numpy() == 1):
            row = sg.iloc[i]
            lab = label_signal(o, h, l, c, i, row.stop_loss, row.take_profit, row.atr_pct, cfg, fm)
            if lab is None or not np.isfinite(row.atr_pct):
                continue
            d = row.to_dict()
            d.update(ts=sg.index[i], sym=sym, r_net=lab[0])
            rows.append(d)
    ds = pd.DataFrame(rows)
    if not ds.empty:
        ds["y"] = (ds.r_net > 0).astype(int)
        ds = ds.sort_values("ts").reset_index(drop=True)
    return ds


def train(data: dict, cfg: Config, min_test_trades: int = 100):
    ds = build_dataset(data, cfg)
    if len(ds) < 300:
        return LogisticFilter(report={"error": f"only {len(ds)} labelled signals; need >= 300"}), ds
    gap = pd.Timedelta(minutes=15 * cfg.max_hold_bars)           # purge: labels look this far ahead
    t0, t1 = ds.ts.quantile(0.6), ds.ts.quantile(0.8)
    tr = ds[ds.ts < t0 - gap]
    va = ds[(ds.ts >= t0) & (ds.ts < t1 - gap)]
    te = ds[ds.ts >= t1]
    X = lambda d: np.vstack([vectorize(r) for r in d.to_dict("records")])
    m = LogisticFilter().fit(X(tr), tr.y.to_numpy())
    pv, pt = m.proba(X(va)), m.proba(X(te))
    best_thr, best = None, -9
    for thr in np.arange(0.40, 0.76, 0.02):                      # threshold chosen on VALIDATION only
        sel = va[pv >= thr]
        if len(sel) >= 30 and sel.r_net.mean() > best:
            best, best_thr = sel.r_net.mean(), float(thr)
    te_sel = te[pt >= best_thr] if best_thr is not None else te.iloc[:0]
    base, filt = float(te.r_net.mean()), float(te_sel.r_net.mean()) if len(te_sel) else float("nan")
    m.threshold = best_thr
    m.active = bool(best_thr is not None and len(te_sel) >= min_test_trades and filt > max(0.0, base))
    m.report = {"rows": {"train": len(tr), "val": len(va), "test": len(te)}, "test_auc": auc(te.y, pt),
                "test_base_expectancy_r": base, "test_filtered_expectancy_r": filt,
                "test_filtered_trades": int(len(te_sel)), "threshold": best_thr, "active": m.active,
                "test_start": str(t1)}
    return m, ds

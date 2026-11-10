"""APEX strategies -- SINGLE SOURCE OF TRUTH for signal generation (backtest, paper and live all call this).

Input: candle DataFrames indexed by candle OPEN time (UTC) with columns
open, high, low, close, volume. Pass only CLOSED candles.

Everything is causal: the signal on bar i uses data up to and including bar i
(higher-timeframe state comes only from fully completed 1h bars).

Gates (applied to every strategy)
  G1 higher-timeframe (1h) trend must not oppose the trade
  G2 volatility sanity: skip dead or chaotic markets
  G3 cost gate: reward must be >= `min_reward_to_cost` x round-trip fees+slippage
  G4 optional BTC 1h trend gate for altcoins

Strategies (priority order)
  SWEEP      liquidity-sweep reversal (stop-hunt wick that closes back inside the range)
  SQUEEZE    volatility-squeeze breakout with volume
  PULLBACK   buy/sell the dip to EMA20 in a confirmed higher-timeframe trend
  MEANREV    Bollinger + RSI fade, only when the market is NOT trending
"""
import numpy as np
import pandas as pd

STRATEGY_VERSION = "v2"

P = dict(
    ema_fast=20, ema_slow=50, rsi_n=14, atr_n=14, bb_n=20, bb_k=2.0, don_n=20, vol_n=20,
    sl_atr=1.5, rr=2.0,
    min_atr_pct=0.0015, max_atr_pct=0.03,
    fee_roundtrip=0.0011, slip_roundtrip=0.0010, min_reward_to_cost=4.0,
    squeeze_q=0.20, squeeze_lookback=200, squeeze_recent=6,
    sweep_n=48, sweep_wick=0.6, sweep_max_sl_atr=2.5,
    er_n=30, er_range=0.25, cooldown_bars=3,
)


# ---------- indicators (all causal) ----------
def ema(s, n): return s.ewm(span=n, adjust=False).mean()


def atr(df, n):
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rsi(c, n):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def efficiency_ratio(c, n):
    return (c - c.shift(n)).abs() / c.diff().abs().rolling(n).sum().replace(0, np.nan)


def htf_state(df1h: pd.DataFrame, p=P) -> pd.Series:
    """+1 up, -1 down, 0 flat. Shifted one bar so only COMPLETED 1h bars are visible."""
    c = df1h["close"]
    f, s = ema(c, p["ema_fast"]), ema(c, p["ema_slow"])
    up = (f > s) & (c > s) & (f.diff(3) > 0)
    dn = (f < s) & (c < s) & (f.diff(3) < 0)
    return pd.Series(np.select([up, dn], [1, -1], default=0), index=df1h.index).shift(1)


def _align(state: pd.Series, idx) -> np.ndarray:
    return state.reindex(idx, method="ffill").fillna(0).to_numpy()


# ---------- the scanner ----------
def scan(df15: pd.DataFrame, df1h: pd.DataFrame, btc1h: pd.DataFrame | None = None, p=P) -> pd.DataFrame:
    """Signal for EVERY bar (use this to build the training dataset). Last row = live signal."""
    o, h, l, c, v = (df15[k] for k in ("open", "high", "low", "close", "volume"))
    a, r = atr(df15, p["atr_n"]), rsi(c, p["rsi_n"])
    e20, e50 = ema(c, p["ema_fast"]), ema(c, p["ema_slow"])
    vr = v / v.rolling(p["vol_n"]).mean()
    mid, sd = c.rolling(p["bb_n"]).mean(), c.rolling(p["bb_n"]).std()
    up_b, lo_b = mid + p["bb_k"] * sd, mid - p["bb_k"] * sd
    bw = (up_b - lo_b) / mid
    squeezed = (bw <= bw.rolling(p["squeeze_lookback"]).quantile(p["squeeze_q"])).astype(float)
    sq_recent = squeezed.shift(1).rolling(p["squeeze_recent"]).max().fillna(0).astype(bool)
    don_hi = h.rolling(p["don_n"]).max().shift(1)
    don_lo = l.rolling(p["don_n"]).min().shift(1)
    sw_hi = h.rolling(p["sweep_n"]).max().shift(1)
    sw_lo = l.rolling(p["sweep_n"]).min().shift(1)
    rng = (h - l).replace(0, np.nan)
    lower_wick, upper_wick = (np.minimum(o, c) - l) / rng, (h - np.maximum(o, c)) / rng
    er = efficiency_ratio(c, p["er_n"])

    htf = _align(htf_state(df1h, p), df15.index)
    btc = _align(htf_state(btc1h, p), df15.index) if btc1h is not None else np.zeros(len(df15))
    htf_s = pd.Series(htf, index=df15.index)

    # entry conditions: (long, short) per strategy
    sweep_l = (l < sw_lo) & (c > sw_lo) & (lower_wick >= p["sweep_wick"]) & (vr >= 1.2) & (htf_s >= 0)
    sweep_s = (h > sw_hi) & (c < sw_hi) & (upper_wick >= p["sweep_wick"]) & (vr >= 1.2) & (htf_s <= 0)
    sq_l = sq_recent & (c > don_hi) & (vr >= 1.3) & (htf_s >= 0)
    sq_s = sq_recent & (c < don_lo) & (vr >= 1.3) & (htf_s <= 0)
    pb_l = (htf_s == 1) & (e20 > e50) & (l <= e20) & (c > e20) & (c > o) & r.between(40, 58)
    pb_s = (htf_s == -1) & (e20 < e50) & (h >= e20) & (c < e20) & (c < o) & r.between(42, 60)
    rng_mkt = er < p["er_range"]
    mr_l = rng_mkt & (c < lo_b) & (r < 30) & (htf_s >= 0)
    mr_s = rng_mkt & (c > up_b) & (r > 70) & (htf_s <= 0)

    n = len(df15)
    sig, layer = np.zeros(n, int), np.array([""] * n, dtype=object)
    sl, tp = np.full(n, np.nan), np.full(n, np.nan)
    ca, ch, cl, cmid = a.to_numpy(), h.to_numpy(), l.to_numpy(), mid.to_numpy()
    cc = c.to_numpy()
    book = [("SWEEP", sweep_l, sweep_s), ("SQUEEZE", sq_l, sq_s), ("PULLBACK", pb_l, pb_s), ("MEANREV", mr_l, mr_s)]
    conds = {name: (L.to_numpy(), S.to_numpy()) for name, L, S in book}
    cost = p["fee_roundtrip"] + p["slip_roundtrip"]
    warm = max(p["squeeze_lookback"], p["sweep_n"], p["ema_slow"], p["er_n"]) + 5
    last_sig = -10**9

    for i in range(warm, n):
        if i - last_sig <= p["cooldown_bars"] or not np.isfinite(ca[i]) or ca[i] <= 0:
            continue
        atr_pct = ca[i] / cc[i]
        if not (p["min_atr_pct"] <= atr_pct <= p["max_atr_pct"]):          # G2
            continue
        for name in ("SWEEP", "SQUEEZE", "PULLBACK", "MEANREV"):
            Lc, Sc = conds[name]
            for side, ok in ((1, Lc[i]), (-1, Sc[i])):
                if not ok:
                    continue
                if (side == 1 and btc[i] == -1) or (side == -1 and btc[i] == 1):   # G4
                    continue
                entry = cc[i]
                if name == "SWEEP":
                    sl_d = (entry - cl[i] + 0.25 * ca[i]) if side == 1 else (ch[i] - entry + 0.25 * ca[i])
                    if sl_d > p["sweep_max_sl_atr"] * ca[i]:
                        continue
                else:
                    sl_d = p["sl_atr"] * ca[i]
                tp_d = p["rr"] * sl_d
                if name == "MEANREV":
                    tp_d = (cmid[i] - entry) if side == 1 else (entry - cmid[i])
                    if tp_d < sl_d:                                           # need >= 1R to the mean
                        continue
                if tp_d / entry < p["min_reward_to_cost"] * cost:               # G3
                    continue
                sig[i], layer[i] = side, name
                sl[i], tp[i] = entry - side * sl_d, entry + side * tp_d
                last_sig = i
                break
            if sig[i]:
                break

    hours = df15.index.hour + df15.index.minute / 60
    out = pd.DataFrame({"signal": sig, "layer": layer, "stop_loss": sl, "take_profit": tp,
                        "htf": htf, "atr_pct": (a / c).to_numpy(), "strategy_version": STRATEGY_VERSION},
                       index=df15.index)
    # features for the optional ML filter (all causal)
    out["f_rsi"] = (r / 100).to_numpy()
    out["f_vol_ratio"] = np.clip(vr, 0, 10).to_numpy()
    out["f_dist_e20"] = ((c - e20) / a).clip(-10, 10).to_numpy()
    out["f_dist_e50"] = ((c - e50) / a).clip(-10, 10).to_numpy()
    out["f_er"] = er.to_numpy()
    out["f_bb_pos"] = ((c - lo_b) / (up_b - lo_b).replace(0, np.nan)).clip(-1, 2).to_numpy()
    out["f_ret_1h"] = (c.pct_change(4) / (a / c)).clip(-10, 10).to_numpy()
    out["f_ret_4h"] = (c.pct_change(16) / (a / c)).clip(-10, 10).to_numpy()
    out["f_btc"] = btc
    out["f_hsin"] = np.sin(2 * np.pi * hours / 24)
    out["f_hcos"] = np.cos(2 * np.pi * hours / 24)
    return out


def to_1h(df15: pd.DataFrame) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return df15.resample("1h").agg(agg).dropna()


FEATURES = ["f_rsi", "f_vol_ratio", "f_dist_e20", "f_dist_e50", "f_er", "f_bb_pos", "f_ret_1h", "f_ret_4h",
            "f_btc", "f_hsin", "f_hcos", "htf", "atr_pct"]
LAYERS = ["SWEEP", "SQUEEZE", "PULLBACK", "MEANREV"]


def generate_signal(df15, df1h, btc1h=None, p=P) -> dict:
    """Live: pass closed candles only. Returns the signal for the latest closed bar."""
    row = scan(df15, df1h, btc1h, p).iloc[-1]
    side = {1: "BUY", -1: "SELL", 0: "HOLD"}[int(row["signal"])]
    return {"signal": side, "layer": row["layer"] or None, "stop_loss": None if side == "HOLD" else float(row["stop_loss"]),
            "take_profit": None if side == "HOLD" else float(row["take_profit"]), "htf_trend": int(row["htf"]),
            "strategy_version": STRATEGY_VERSION}

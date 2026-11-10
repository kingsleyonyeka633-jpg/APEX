"""Fake multi-coin market for offline testing. Proves the machinery, says NOTHING about profitability."""
import numpy as np
import pandas as pd


def make_market(n_symbols=6, n=20_000, seed=5, freq="15min"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")

    def regime_returns(scale):
        rets, i, prev = np.zeros(n), 0, 0.0
        while i < n:
            length = int(rng.integers(300, 2000))
            mu, sig, ac = [(0.0002, 0.0025, 0.0), (-0.0002, 0.003, 0.0), (0.0, 0.002, -0.3)][rng.integers(0, 3)]
            for j in range(i, min(n, i + length)):
                prev = mu * scale + ac * prev + sig * scale * rng.standard_normal()
                rets[j] = prev
            i += length
        return rets

    btc_r = regime_returns(0.8)
    out = {}
    names = ["BTCUSDT"] + [f"ALT{k}USDT" for k in range(1, n_symbols)]
    for k, name in enumerate(names):
        r = btc_r if k == 0 else 0.5 * btc_r + regime_returns(1.0 + 0.15 * k)
        close = 100 * np.exp(np.cumsum(r))
        open_ = np.r_[100.0, close[:-1]]
        wick = np.abs(rng.standard_normal(n)) * 0.0025
        liquidity = 5e6 if k == 0 else 4e4 * (1.6 ** (n_symbols - k))      # later alts are thinner
        qv = liquidity * rng.lognormal(0, 0.6, n)
        out[name] = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * (1 + wick),
                                  "low": np.minimum(open_, close) * (1 - wick), "close": close,
                                  "volume": qv / close, "quote_volume": qv}, index=idx)
    return out

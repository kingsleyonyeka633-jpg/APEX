import os, sys
import numpy as np
import pandas as pd
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from apex.broker import FillModel, round_qty
from apex.config import DEFAULT
from apex.data.synthetic import make_market
from apex.engine import run_backtest, compute_signals
from apex.portfolio import Portfolio
from apex.strategies import scan, to_1h, FEATURES

DATA = make_market(n_symbols=6, n=12_000)


def test_fill_model_costs_rise_with_size_and_thin_volume():
    fm = FillModel(DEFAULT)
    small = fm.buy_price(100, 100, 1e6, 0.005)
    big = fm.buy_price(100, 50_000, 1e6, 0.005)
    thin = fm.buy_price(100, 100, 1e3, 0.005)
    assert 100 < small < big and small < thin
    assert fm.sell_price(100, 100, 1e6, 0.005) < 100
    assert fm.max_notional(1e6) == DEFAULT.participation_cap * 1e6
    assert abs(round_qty(123.456789) - 123.4) < 1e-9 and abs(round_qty(0.00123456) - 0.001234) < 1e-12


def test_features_are_causal():
    df = DATA["ALT2USDT"]
    k = 9000
    a = scan(df.iloc[:k], to_1h(df.iloc[:k]))
    b = scan(df, to_1h(df)).iloc[:k]
    pd.testing.assert_frame_equal(a, b)
    assert all(f in a.columns for f in FEATURES)


def test_backtest_invariants():
    r = run_backtest(DATA, DEFAULT)
    pf = r.portfolio
    assert pf.cash > -1e-6, "spot: cash must never go negative"
    assert (r.equity > 0).all()
    assert r.signals_taken <= r.signals_seen
    if len(r.trades):
        assert (r.trades.fees >= 0).all() and (r.trades.slippage >= -1e-9).all()
        assert r.trades.r.min() > -3.0, "a single trade should not lose several times its risk budget"
    assert abs(r.metrics["net_pnl"] - (pf.equity() - DEFAULT.initial_capital)) < 1e-6
    print("   trades:", r.metrics["trades"], "net:", round(r.metrics["net_pnl"], 2), "costs:", round(r.metrics["fees_paid"] + r.metrics["slippage_paid"], 2))


def test_costs_matter():
    free = replace(DEFAULT, taker_fee=0, maker_fee=0, min_half_spread_bps=0, spread_vol_k=0, impact_coef=0)
    a = run_backtest(DATA, free).metrics["net_pnl"]
    b = run_backtest(DATA, DEFAULT).metrics["net_pnl"]
    assert b < a


def test_risk_caps_respected():
    cfg = replace(DEFAULT, max_open_positions=2)
    pf = run_backtest(DATA, cfg).portfolio
    assert pf.max_open_seen <= 2
    assert pf.max_exposure_seen <= cfg.max_total_exposure + 0.02   # small drift from price moves between marks


def test_drawdown_halt_flattens():
    cfg = replace(DEFAULT, max_drawdown=0.01, daily_loss_limit=0.5)
    r = run_backtest(DATA, cfg)
    assert r.portfolio.guard.halted_dd or r.metrics["max_drawdown"] > -0.01
    assert len(r.portfolio.pos) == 0 or not r.portfolio.guard.halted_dd


def test_state_roundtrip():
    r = run_backtest(DATA, DEFAULT, end="2024-03-01")
    pf = r.portfolio
    ts = pd.Timestamp("2024-03-01", tz="UTC")
    pf2 = Portfolio.from_dict(DEFAULT, pf.to_dict(), ts)
    assert abs(pf2.equity() - pf.equity()) < 1e-9 and pf2.pos.keys() == pf.pos.keys()


def test_reward_risk_setting_changes_target_distance():
    for rr in (1.0, 3.0):
        cfg = replace(DEFAULT, reward_risk=rr)
        sg = compute_signals({k: DATA[k] for k in ("BTCUSDT", "ALT2USDT")}, cfg)["ALT2USDT"]
        x = sg[(sg.signal == 1) & (sg.layer != "MEANREV") & (sg.layer != "SWEEP")]
        if len(x):
            c = DATA["ALT2USDT"].close.loc[x.index]
            ratio = (x.take_profit - c) / (c - x.stop_loss)
            assert (abs(ratio - rr) < 1e-6).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)

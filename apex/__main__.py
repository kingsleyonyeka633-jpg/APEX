import argparse
import json
import os
import sys
from dataclasses import replace

import pandas as pd

from .config import DEFAULT
from .data import bulk, exchange, store, universe
from .data.synthetic import make_market


def _fmt(m):
    pct = {"total_return", "cagr", "max_drawdown", "win_rate", "costs_pct_of_gross"}
    out = []
    for k, v in m.items():
        if v is None:
            out.append(f"  {k:24s} {'n/a':>12}")
        elif k in pct:
            out.append(f"  {k:24s} {v:>12.1%}")
        else:
            out.append(f"  {k:24s} {v:>12.2f}")
    return "\n".join(out)


def _syms(a, cfg):
    if a.symbols:
        syms = a.symbols
    else:
        syms = universe.load(os.path.join(cfg.data_dir, "universe.json"))
    return sorted(set(syms) | {cfg.btc_symbol})


def _data(a, cfg):
    if a.synthetic:
        return make_market(n_symbols=a.n_symbols, n=a.bars)
    data = {}
    for s in _syms(a, cfg):
        try:
            data[s] = store.load(cfg.data_dir, s, cfg.timeframe)
        except FileNotFoundError:
            print(f"  (no data for {s}, skipping; run `python -m apex download` first)")
    if cfg.btc_symbol not in data:
        sys.exit("BTCUSDT data is required for the market-trend gate. Download it first.")
    return data


def _cfg(a):
    cfg = DEFAULT
    if getattr(a, "ml", False) and os.path.exists(cfg.model_path):
        from .ml import LogisticFilter
        m = LogisticFilter.load(cfg.model_path)
        if m.active:
            cfg = replace(cfg, ml_threshold=m.threshold)
            return cfg, m
        print("ML filter exists but did not beat 'take every signal' on its test slice, so it is OFF.")
    return cfg, None


def main():
    p = argparse.ArgumentParser(prog="apex", description="Mid-cap crypto quant system (paper trading first).")
    p.add_argument("cmd", choices=["universe", "download", "quality", "backtest", "sweep", "train", "auto", "paper", "run", "dashboard", "status", "resume"])
    p.add_argument("--symbols", nargs="*")
    p.add_argument("--years", type=int, default=3)
    p.add_argument("--synthetic", action="store_true", help="fake market: tests the machinery only")
    p.add_argument("--n-symbols", type=int, default=6)
    p.add_argument("--bars", type=int, default=20000)
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--ml", action="store_true", help="use the trained ML filter if it passed its test")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    cfg = DEFAULT
    pd.set_option("display.width", 200)

    if a.cmd == "universe":
        u = universe.select_universe(exchange.tickers_24h(), cfg)
        universe.save(u, os.path.join(cfg.data_dir, "universe.json"))
        for i, x in enumerate(u, 1):
            print(f"{i:3d} {x['symbol']:14s} ${x['quote_volume_24h'] / 1e6:8.1f}M / day")
    elif a.cmd == "download":
        for s in _syms(a, cfg):
            bulk.download_symbol(s, cfg.timeframe, a.years, cfg.data_dir)
    elif a.cmd == "quality":
        for s in _syms(a, cfg):
            print(s, json.dumps(store.quality_report(store.load(cfg.data_dir, s, cfg.timeframe))))
    elif a.cmd == "backtest":
        from .engine import run_backtest
        cfg, ml = _cfg(a)
        r = run_backtest(_data(a, cfg), cfg, start=a.start, end=a.end, ml=ml)
        print(_fmt(r.metrics))
        print(f"\n  signals seen {r.signals_seen}, taken {r.signals_taken}")
        print("\nBy strategy:\n", r.by_layer.round(3).to_string() if len(r.by_layer) else "  no trades")
        print("\nBy coin:\n", r.by_symbol.round(3).head(10).to_string() if len(r.by_symbol) else "  no trades")
    elif a.cmd == "sweep":
        from .engine import compute_signals, run_backtest
        data = _data(a, cfg)
        print("Win rate vs the win rate you NEED just to break even (after fees and slippage), per take-profit size.")
        print("Margin = actual win rate minus break-even. Positive = real edge at that target.\n")
        print(f"{'target':>7} {'signals':>8} {'trades':>7} {'win%':>6} {'need%':>6} {'margin':>7} {'avgR':>6} {'net$':>9} {'costs/gross':>12}")
        for rr in (0.5, 1.0, 1.5, 2.0, 3.0):
            c = replace(cfg, reward_risk=rr)
            r = run_backtest(data, c, start=a.start, end=a.end, sigs=compute_signals(data, c))
            t = r.trades
            if len(t) < 10:
                print(f"{'1:' + str(rr) if rr >= 1 else str(rr) + ':1':>7} {r.signals_seen:>8} {len(t):>7}   too few trades to judge")
                continue
            w, l = t[t.r > 0].r.mean(), -t[t.r <= 0].r.mean()
            need = l / (w + l)
            win = (t.r > 0).mean()
            cg = r.metrics["costs_pct_of_gross"]
            print(f"{rr:>6}R {r.signals_seen:>8} {len(t):>7} {win:>6.1%} {need:>6.1%} {win - need:>+7.1%} {t.r.mean():>6.2f} {r.metrics['net_pnl']:>9.2f} {('n/a' if cg is None else format(cg, '.0%')):>12}")
        print("\nDo NOT just pick the best row. Choose using an early period, then confirm on a later one:")
        print("  python -m apex sweep --end 2025-06-01      (choose)")
        print("  python -m apex sweep --start 2025-06-01    (confirm)")
    elif a.cmd == "train":
        from .ml import train
        m, ds = train(_data(a, cfg), cfg)
        print(json.dumps(m.report, indent=1, default=str))
        if m.w is not None:
            m.save(cfg.model_path)
            print("saved ->", cfg.model_path)
    elif a.cmd == "paper":
        from .live import PaperRunner, setup_logging
        cfg, ml = _cfg(a)
        setup_logging()
        PaperRunner(cfg, ml=ml).run()
    elif a.cmd == "auto":
        from .auto import AutoRunner
        from .live import setup_logging
        setup_logging()
        AutoRunner(cfg, years=a.years).run(open_browser=True)
    elif a.cmd == "run":
        # One process for hosts like Railway: dashboard in a thread, paper bot in the main loop.
        from .dashboard.server import serve_background
        from .live import PaperRunner, setup_logging
        cfg, ml = _cfg(a)
        setup_logging()
        universe.ensure(cfg, exchange.tickers_24h)
        port = int(os.getenv("PORT", a.port))
        serve_background(cfg, "0.0.0.0", port)
        print(f"Dashboard listening on port {port}")
        PaperRunner(cfg, ml=ml).run()
    elif a.cmd == "dashboard":
        from .dashboard.server import serve
        serve(cfg, a.host, a.port)
    elif a.cmd == "status":
        from .dashboard.server import summary
        s = summary(cfg)
        print(json.dumps({k: s[k] for k in ("status", "equity", "risk", "costs")}, indent=1, default=str))
    elif a.cmd == "resume":
        from .db import DB
        db = DB(cfg.db_path)
        st = db.kv_get("portfolio")
        if not st:
            sys.exit("Nothing to resume.")
        st["manual_halt"] = None
        if st.get("guard"):
            st["guard"].update(halted_dd=False, halted_day=False, peak=st["cash"] + sum(
                p["qty"] * st["last"].get(p["sym"], p["entry_px"]) for p in st["pos"].values()))
        db.kv_set("portfolio", st)
        if os.path.exists(cfg.kill_file):
            os.remove(cfg.kill_file)
        print("Halts cleared. Drawdown peak reset to current equity. Restart the paper bot.")


if __name__ == "__main__":
    main()

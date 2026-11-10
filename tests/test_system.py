import io, json, os, sys, tempfile, threading, urllib.request, urllib.error, zipfile
import numpy as np
import pandas as pd
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from apex.config import DEFAULT
from apex.data import bulk, exchange, store, universe
from apex.data.synthetic import make_market
from apex.db import DB
from apex.engine import run_backtest
from apex.ml import train, LogisticFilter, auc, vectorize
from apex.live import PaperRunner

DATA = make_market(n_symbols=6, n=14_000)


def ts_ms(idx):
    return ((idx - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)).astype("int64")


def candles(df):
    out = df.copy()
    out.insert(0, "ts", ts_ms(df.index))
    return out[store.COLS].reset_index(drop=True)


def write_store(d, data):
    for sym, df in data.items():
        store.save_merge(candles(df), store.path_for(d, sym, "15m"))


def test_bulk_parser_handles_microseconds_and_zip():
    ms = "1700000000000,10,11,9,10.5,100,1700000899999,1050,10,50,500,0\n"
    us = "1735689600000000,10,11,9,10.5,100,1735690499999999,1050,10,50,500,0\n"
    df = bulk.parse_bulk_csv((ms + us).encode())
    assert list(df.ts) == [1700000000000, 1735689600000] and df.quote_volume.iloc[0] == 1050
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("x.csv", ms)
    class R:
        def __init__(s, d): s.d = d
        def read(s): return s.d
        def __enter__(s): return s
        def __exit__(s, *a): pass
    out = bulk.month_zip("TESTUSDT", "15m", 2024, 1, opener=lambda req, timeout: R(buf.getvalue()))
    assert len(out) == 1
    def missing(req, timeout): raise urllib.error.HTTPError("u", 404, "nf", {}, None)
    assert bulk.month_zip("TESTUSDT", "15m", 2015, 1, opener=missing) is None


def test_klines_paging_with_fake_exchange():
    step, t0 = 900_000, 1_700_000_000_000
    all_ts = [t0 + i * step for i in range(3500)]
    def fetch(path, params):
        lo, hi, lim = params["startTime"], params["endTime"], params["limit"]
        sel = [t for t in all_ts if lo <= t <= hi][:lim]
        return [[t, "1", "2", "0.5", "1.5", "10", t + step - 1, "15", 1, 0, 0, 0] for t in sel]
    exchange.time.sleep = lambda *_: None
    df = exchange.klines_range("X", "15m", t0, t0 + 3500 * step, fetch=fetch)
    assert len(df) == 3500 and df.ts.is_unique and df.ts.is_monotonic_increasing


def test_universe_selection_skips_megas_stables_leveraged():
    t = lambda s, v: {"symbol": s, "quoteVolume": str(v), "lastPrice": "1.0"}
    tickers = [t("BTCUSDT", 9e9), t("ETHUSDT", 5e9), t("USDCUSDT", 4e9), t("BNBUPUSDT", 9e8), t("ADAUSDT", 3e8),
               t("LINKUSDT", 2e8), t("DOGEUSDT", 1e8), t("TINYUSDT", 1e5), t("ADABTC", 9e9)]
    cfg = replace(DEFAULT, universe_rank_from=1, universe_rank_to=10)
    syms = [u["symbol"] for u in universe.select_universe(tickers, cfg)]
    assert syms == ["ADAUSDT", "LINKUSDT", "DOGEUSDT"]


def test_store_roundtrip_and_quality():
    with tempfile.TemporaryDirectory() as d:
        df = DATA["ALT1USDT"].copy()
        path = store.path_for(d, "ALT1USDT", "15m")
        store.save_merge(candles(df), path)
        store.save_merge(candles(df).iloc[-100:], path)      # re-run must not duplicate
        back = store.load(d, "ALT1USDT", "15m")
        assert len(back) == len(df)
        q = store.quality_report(back)
        assert q["missing_bars"] == 0 and q["bad_candles"] == 0


def test_ml_trains_with_purge_and_honest_verdict():
    m, ds = train(DATA, DEFAULT)
    assert m.w is not None and "test_auc" in m.report
    assert 0 <= m.predict(ds.iloc[0].to_dict()) <= 1
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.json")
        m.save(path)
        m2 = LogisticFilter.load(path)
        assert abs(m2.predict(ds.iloc[0].to_dict()) - m.predict(ds.iloc[0].to_dict())) < 1e-9
    print("   ML report:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.report.items() if k != "rows"})
    assert abs(auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) - 1.0) < 1e-9


def test_paper_runner_resumes_and_logs():
    """Feed the runner fake 'live' candles in two steps; it must resume from SQLite without re-trading."""
    with tempfile.TemporaryDirectory() as d:
        cfg = replace(DEFAULT, db_path=os.path.join(d, "a.db"), kill_file=os.path.join(d, "KILL"), data_dir=d)
        full = DATA
        cut = {"n": 11_000}
        def fetch(sym, tf, limit=500):
            return candles(full[sym].iloc[: cut["n"]].iloc[-limit:])
        syms = [s for s in full if s != cfg.btc_symbol]
        import apex.live as live
        live.time.time = lambda: 4_000_000_000        # pretend it is always far past the last bar
        r1 = PaperRunner(cfg, symbols=syms + [cfg.btc_symbol], fetch=fetch, notifier=lambda *_: None)
        r1.tick()                               # first run: only records the starting point
        for n in range(11_200, 14_000, 200):    # then 'time passes'
            cut["n"] = n
            r1.tick()
        eq1 = r1.pf.equity()
        assert len(r1.db.equity_rows()) > 100
        r2 = PaperRunner(cfg, symbols=syms + [cfg.btc_symbol], fetch=fetch, notifier=lambda *_: None)   # restart
        assert abs(r2.pf.equity() - eq1) < 1e-6 and r2.pf.pos.keys() == r1.pf.pos.keys()
        n_trades = len(r1.db.trades())
        r2.tick()                               # nothing new: must not duplicate anything
        assert len(r2.db.trades()) == n_trades
        print("   paper bars logged:", len(r1.db.equity_rows()), "trades:", n_trades, "equity:", round(eq1, 2))


def test_kill_file_flattens_and_halts():
    with tempfile.TemporaryDirectory() as d:
        cfg = replace(DEFAULT, db_path=os.path.join(d, "b.db"), kill_file=os.path.join(d, "KILL"), data_dir=d)
        cut = {"n": 6000}
        def fetch(sym, tf, limit=500):
            return candles(DATA[sym].iloc[: cut["n"]].iloc[-limit:])
        syms = list(DATA)
        import apex.live as live
        live.time.time = lambda: 4_000_000_000
        r = PaperRunner(cfg, symbols=syms, fetch=fetch, notifier=lambda *_: None)
        r.tick()
        for n in range(6100, 9000, 100):
            cut["n"] = n; r.tick()
        open(cfg.kill_file, "w").close()
        cut["n"] = 9100; r.tick()
        assert r.pf.manual_halt and len(r.pf.pos) == 0 and not r.pf.pending


def test_dashboard_api_and_kill_auth():
    from apex.dashboard.server import make_handler, summary
    from http.server import ThreadingHTTPServer
    with tempfile.TemporaryDirectory() as d:
        cfg = replace(DEFAULT, db_path=os.path.join(d, "c.db"), kill_file=os.path.join(d, "KILL"))
        db = DB(cfg.db_path)
        db.add_equity("2024-01-01T00:00:00+00:00", 1000.0, 1000.0)
        db.add_equity("2024-01-01T00:15:00+00:00", 1010.0, 1010.0)
        db.add_trade({"sym": "ADAUSDT", "layer": "PULLBACK", "entry_ts": "a", "exit_ts": "b", "entry_px": 1, "exit_px": 1.1,
                      "qty": 10, "pnl": 10.0, "r": 1.0, "fees": 0.5, "slippage": 0.3, "reason": "tp", "bars": 5})
        s = summary(cfg)
        assert abs(s["costs"]["gross"] - 10.8) < 1e-9 and s["equity"]["now"] == 1010.0 and s["layers"][0]["trades"] == 1
        srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(cfg))
        port = srv.server_address[1]
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{port}"
        assert b"Apex paper desk" in urllib.request.urlopen(base + "/").read()
        assert json.loads(urllib.request.urlopen(base + "/api/summary").read())["mode"] == "paper"
        def post(token):
            req = urllib.request.Request(base + "/api/kill", method="POST", headers={"X-Token": token} if token else {})
            try:
                return urllib.request.urlopen(req).status
            except urllib.error.HTTPError as e:
                return e.code
        os.environ.pop("DASHBOARD_TOKEN", None)
        assert post("x") == 403 and not os.path.exists(cfg.kill_file)      # no token configured: refuse
        os.environ["DASHBOARD_TOKEN"] = "secret"
        assert post("wrong") == 403 and not os.path.exists(cfg.kill_file)
        assert post("secret") == 200 and os.path.exists(cfg.kill_file)
        srv.shutdown()




def test_dashboard_password_and_health_and_public_guard():
    import base64
    from apex.dashboard.server import make_handler, _check_exposure
    from http.server import ThreadingHTTPServer
    with tempfile.TemporaryDirectory() as d:
        cfg = replace(DEFAULT, db_path=os.path.join(d, "p.db"), kill_file=os.path.join(d, "KILL"))
        DB(cfg.db_path)
        os.environ["DASHBOARD_PASSWORD"] = "pw123"
        srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(cfg))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        def get(path, pw=None):
            req = urllib.request.Request(base + path)
            if pw is not None:
                req.add_header("Authorization", "Basic " + base64.b64encode(f"me:{pw}".encode()).decode())
            try:
                return urllib.request.urlopen(req).status
            except urllib.error.HTTPError as e:
                return e.code
        assert get("/health") == 200                      # open, carries no data
        assert get("/api/summary") == 401 and get("/") == 401
        assert get("/api/summary", "wrong") == 401
        assert get("/api/summary", "pw123") == 200 and get("/", "pw123") == 200
        srv.shutdown()
        os.environ.pop("DASHBOARD_PASSWORD")
        try:
            _check_exposure("0.0.0.0")
            raise AssertionError("must refuse public bind without password")
        except SystemExit:
            pass
        _check_exposure("127.0.0.1")


def test_universe_ensure_creates_refreshes_and_survives_failure():
    from apex.data import universe as U
    t = lambda s, v: {"symbol": s, "quoteVolume": str(v), "lastPrice": "1.0"}
    good = lambda: [t("ADAUSDT", 3e8), t("LINKUSDT", 2e8)]
    def bad(): raise RuntimeError("blocked")
    with tempfile.TemporaryDirectory() as d:
        cfg = replace(DEFAULT, data_dir=d, universe_rank_from=1, universe_rank_to=10)
        assert U.ensure(cfg, good, log=lambda *_: None) == ["ADAUSDT", "LINKUSDT"]
        old = os.path.join(d, "universe.json")
        os.utime(old, (0, 0))                              # make it stale
        assert U.ensure(cfg, bad, log=lambda *_: None) == ["ADAUSDT", "LINKUSDT"]   # keeps old list
    with tempfile.TemporaryDirectory() as d:
        try:
            U.ensure(replace(DEFAULT, data_dir=d), bad, log=lambda *_: None)
            raise AssertionError("no list and no network must fail loudly")
        except RuntimeError:
            pass


def test_healthy_symbols_drops_short_and_broken_coins():
    from apex.auto import healthy_symbols
    with tempfile.TemporaryDirectory() as d:
        cfg = replace(DEFAULT, data_dir=d, min_rows_per_coin=3000)
        good, short, broken = DATA["ALT1USDT"], DATA["ALT2USDT"].iloc[:1000], DATA["ALT3USDT"].copy()
        broken.iloc[500, broken.columns.get_loc("high")] = broken.iloc[500]["low"] * 0.5      # high below low
        for name, df in (("GOODUSDT", good), ("SHORTUSDT", short), ("BROKENUSDT", broken)):
            store.save_merge(candles(df), store.path_for(d, name, "15m"))
        assert healthy_symbols(cfg, ["GOODUSDT", "SHORTUSDT", "BROKENUSDT", "NOFILEUSDT"], log=lambda *_: None) == ["GOODUSDT"]


def test_set_model_swaps_filter_on_and_off():
    from apex.ml import NAMES
    with tempfile.TemporaryDirectory() as d:
        cfg = replace(DEFAULT, db_path=os.path.join(d, "x.db"), kill_file=os.path.join(d, "KILL"), data_dir=d)
        r = PaperRunner(cfg, symbols=list(DATA), fetch=lambda *a, **k: None, notifier=lambda *_: None)
        rng = np.random.default_rng(1)
        m = LogisticFilter().fit(rng.standard_normal((80, len(NAMES))), (rng.random(80) > 0.5).astype(float))
        m.active, m.threshold = True, 0.99
        r.set_model(m)
        assert r.pf.ml is m and r.pf.cfg.ml_threshold == 0.99
        m.active = False
        r.set_model(m)
        assert r.pf.ml is None and r.pf.cfg.ml_threshold is None


def test_auto_pipeline_end_to_end_and_ab_comparison():
    from apex.auto import AutoRunner
    from apex.config import arm_db_path
    from apex.dashboard.server import summary
    with tempfile.TemporaryDirectory() as d:
        cfg = replace(DEFAULT, data_dir=d, db_path=os.path.join(d, "s", "apex.db"), kill_file=os.path.join(d, "KILL"),
                      model_path=os.path.join(d, "m.json"), universe_rank_from=1, universe_rank_to=10)
        write_store(d, DATA)
        cut = {"n": 12_000}
        fetch = lambda sym, tf, limit=500: candles(DATA[sym].iloc[: cut["n"]].iloc[-limit:])
        tickers = lambda: [{"symbol": s, "quoteVolume": "9e8", "lastPrice": "1"} for s in DATA if s != "BTCUSDT"]
        msgs = []
        ar = AutoRunner(cfg, years=1, notifier=msgs.append, fetch=fetch, download=lambda *a, **k: 0,
                        tickers=tickers, log=lambda *_: None, fetch_ttl=0)
        rep = ar.prepare()
        assert "verdict" in rep and rep["rules"]["trades"] >= 0 and ar.db.kv_get("progress") == ""
        assert os.path.exists(cfg.model_path)
        ar.build_arms()
        import apex.live as live
        live.time.time = lambda: 4_000_000_000
        ar.tick_all()                                  # first tick: records the starting point only
        for n in range(12_200, 14_000, 200):
            cut["n"] = n
            ar.tick_all()
        assert len(ar.rules.db.equity_rows()) > 50 and len(ar.mlarm.db.equity_rows()) > 50
        assert ar.rules.db.path != ar.mlarm.db.path and ar.mlarm.db.path == arm_db_path(cfg, "ml")
        lf, lt = ar.maintenance(0, 0, now=10**10)       # daily refresh + weekly retrain are due
        assert lt == 10**10 and ar.mlarm.cfg.ml_threshold == (ar.model.threshold if ar.model.active else None)
        s_rules, s_ml = summary(cfg, "rules"), summary(cfg, "ml")
        assert s_rules["arm"] == "rules" and s_ml["arm"] == "ml"
        assert s_rules["compare"]["rules"]["equity"] is not None and s_rules["compare"]["ml"]["equity"] is not None
        assert s_rules["holdout"]["verdict"] and s_rules["ml"]["has_model"]
        assert any("ML retrained" in m or "ML not trained" in m for m in msgs)
        print("   verdict:", rep["verdict"][:70], "| ML filter on:", ar.model.active)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)

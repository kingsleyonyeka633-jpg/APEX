"""One-command operation: pick coins -> download -> check -> train -> test on unseen history -> paper trade.
Keeps refreshing data and retraining by itself. Two paper bots run side by side on the same market:
  rules  = strategy only
  ml     = strategy + ML filter (filter is ON only while the latest model passes its unseen-data test)
so the dashboard can show, with real paper results, whether the ML is actually helping."""
import time
import webbrowser
import os
from dataclasses import replace

from . import ml as mlmod
from .alerts import notify
from .config import Config, arm_db_path
from .data import bulk, exchange, store, universe
from .db import DB
from .engine import compute_signals, run_backtest
from .live import PaperRunner
from .ml import LogisticFilter


class CachedFetch:
    """Both paper bots need the same candles: fetch once per short window."""
    def __init__(self, fetch, ttl=45):
        self.f, self.ttl, self.c = fetch, ttl, {}

    def __call__(self, sym, tf, limit=500):
        k, now = (sym, tf, limit), time.time()
        hit = self.c.get(k)
        if hit and now - hit[0] < self.ttl:
            return hit[1].copy()
        df = self.f(sym, tf, limit=limit)
        self.c[k] = (now, df)
        return df.copy()


def healthy_symbols(cfg: Config, symbols, log=print):
    ok = []
    for s in symbols:
        try:
            q = store.quality_report(store.load(cfg.data_dir, s, cfg.timeframe))
        except FileNotFoundError:
            log(f"  skipped {s}: no data")
            continue
        if q["rows"] < cfg.min_rows_per_coin:
            log(f"  skipped {s}: only {q['rows']} candles (too new or too thin)")
        elif q["bad_candles"] > 0 or q["missing_bars"] / q["rows"] > 0.02:
            log(f"  skipped {s}: broken data ({q['bad_candles']} bad candles, {q['missing_bars']} missing)")
        else:
            ok.append(s)
    return ok


def download_all(cfg: Config, symbols, years, download=bulk.download_symbol, progress=lambda *_: None, log=print):
    for i, s in enumerate(symbols, 1):
        progress(f"Downloading price history {i}/{len(symbols)} ({s})")
        try:
            download(s, cfg.timeframe, years, cfg.data_dir, log=lambda *_: None)
        except Exception as e:
            log(f"  {s}: download problem ({e})")


def load_data(cfg, symbols, years=None):
    """Keep only the newest `years` of candles per coin (one coin read at a time): keeps memory small on free hosts."""
    keep = int(years * 365 * 86_400_000 / exchange.TF_MS[cfg.timeframe]) if years else None
    out = {}
    for s in symbols:
        df = store.load(cfg.data_dir, s, cfg.timeframe)
        out[s] = df.tail(keep).copy() if keep else df
    return out


def train_and_save(cfg, data, db, notifier=print):
    m, _ = mlmod.train(data, cfg)
    status = {"trained_at": time.time(), "active": bool(m.active), "threshold": m.threshold, "report": m.report}
    if m.w is not None:
        m.save(cfg.model_path)
    db.kv_set("ml_status", status)
    hist = (db.kv_get("ml_history", []) or []) + [status]
    db.kv_set("ml_history", hist[-20:])
    r = m.report
    if "error" in r:
        notifier(f"ML not trained: {r['error']}")
    else:
        notifier(f"ML retrained. Filter is {'ON' if m.active else 'OFF (found no edge on unseen data)'}. "
                 f"Unseen-data expectancy: {r['test_base_expectancy_r']:+.2f}R without filter, "
                 f"{r['test_filtered_expectancy_r']:+.2f}R with it.")
    return m


def verdict(rules: dict) -> str:
    if rules["trades"] < 100:
        return "Too few trades on unseen history to judge. Collect more data before believing any result."
    if rules["net_pnl"] > 0 and rules["avg_r"] > 0.05:
        return "Positive after fees and slippage on history it never saw. Promising, but not proven: paper trade 4+ weeks."
    return "NOT profitable after fees and slippage on unseen history. It will still paper trade, but expect losses. Do not use real money."


def holdout_report(cfg, data, model=None):
    if model is not None and model.report.get("test_start"):
        start = model.report["test_start"]
    else:
        tl = sorted(set().union(*[d.index for d in data.values()]))
        start = str(tl[int(len(tl) * 0.8)])
    sigs = compute_signals(data, cfg)
    out = {"start": start, "rules": run_backtest(data, cfg, start=start, sigs=sigs).metrics}
    if model is not None and model.active:
        c = replace(cfg, ml_threshold=model.threshold)
        out["ml"] = run_backtest(data, c, start=start, ml=model, sigs=sigs).metrics
    out["verdict"] = verdict(out["rules"])
    return out


class AutoRunner:
    def __init__(self, cfg: Config, years=3, notifier=notify, fetch=exchange.recent_closed,
                 download=bulk.download_symbol, tickers=exchange.tickers_24h, log=print, fetch_ttl=45):
        self.cfg, self.years, self.notifier, self.fetch, self.fetch_ttl = cfg, years, notifier, fetch, fetch_ttl
        self.download, self.tickers, self.log = download, tickers, log
        self.db = DB(cfg.db_path)
        self.symbols, self.model = [], None
        self.rules = self.mlarm = None

    def progress(self, text):
        self.db.kv_set("progress", text)
        self.log(text)

    def prepare(self):
        cfg = self.cfg
        self.progress("Step 1/5: choosing mid-cap coins")
        picked = universe.ensure(cfg, self.tickers, log=self.log)[:cfg.max_coins]   # list is ordered by volume
        syms = sorted(set(picked) | {cfg.btc_symbol})
        self.progress("Step 2/5: downloading price history (first time is slow; safe to stop and restart)")
        download_all(cfg, syms, self.years, self.download, self.progress, self.log)
        self.progress("Step 3/5: checking the data for gaps and broken candles")
        self.symbols = healthy_symbols(cfg, syms, self.log)
        if cfg.btc_symbol not in self.symbols:
            raise SystemExit("BTC data is missing or broken, and the market-trend gate needs it. Check your internet and run again.")
        data = load_data(cfg, self.symbols, self.years)
        self.progress("Step 4/5: training the ML filter")
        self.model = train_and_save(cfg, data, self.db, self.notifier)
        self.progress("Step 5/5: testing on history the model never saw")
        rep = holdout_report(cfg, data, self.model)
        self.db.kv_set("holdout_report", rep)
        self.log("\n" + rep["verdict"] + "\n")
        self.db.kv_set("progress", "")
        return rep

    def build_arms(self):
        shared = CachedFetch(self.fetch, self.fetch_ttl)
        on = self.model is not None and self.model.active
        cfg_ml = replace(self.cfg, db_path=arm_db_path(self.cfg, "ml"), ml_threshold=self.model.threshold if on else None)
        self.rules = PaperRunner(self.cfg, self.symbols, fetch=shared, ml=None, notifier=self.notifier, tag="[rules]")
        self.mlarm = PaperRunner(cfg_ml, self.symbols, fetch=shared, ml=self.model if on else None,
                                 notifier=self.notifier, tag="[rules+ML]")

    def tick_all(self):
        self.rules.tick()
        self.mlarm.tick()

    def maintenance(self, last_refresh, last_train, now=None):
        """Refresh data daily, retrain weekly. Returns the new (last_refresh, last_train)."""
        now = now or time.time()
        cfg = self.cfg
        if now - last_refresh >= cfg.refresh_hours * 3600:
            download_all(cfg, self.symbols, self.years, self.download, lambda *_: None, self.log)
            last_refresh = now
        if now - last_train >= cfg.retrain_days * 86400:
            model = train_and_save(cfg, load_data(cfg, self.symbols, self.years), self.db, self.notifier)
            self.model = model
            self.mlarm.set_model(model)
            last_train = now
        return last_refresh, last_train

    def run(self, host=None, port=None, open_browser=True):
        from .dashboard.server import serve_background
        port = port or int(os.getenv("PORT", 8000))
        host = host or os.getenv("APEX_HOST") or ("0.0.0.0" if os.getenv("PORT") else "127.0.0.1")
        serve_background(self.cfg, host, port)
        url = f"http://localhost:{port}"
        self.log(f"\nDashboard: {url}   (keep this window open; closing it stops the bot)\n")
        if open_browser and host == "127.0.0.1":
            webbrowser.open(url)
        self.prepare()
        self.build_arms()
        self.notifier(f"Apex started on paper money: {len(self.symbols) - 1} coins. "
                      f"ML filter {'ON' if self.model and self.model.active else 'OFF'}.")
        last_refresh = last_train = time.time()
        errors = 0
        while True:
            try:
                self.tick_all()
                last_refresh, last_train = self.maintenance(last_refresh, last_train)
                errors = 0
            except Exception as e:
                errors += 1
                self.log(f"problem (will retry): {e}")
                if errors in (3, 10, 30):
                    self.notifier(f"{errors} problems in a row: {e}")
            time.sleep(self.cfg.poll_seconds)

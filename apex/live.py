"""Paper trading on LIVE prices with the simulated broker (fees, spread, impact, volume caps).
Same Portfolio class as the backtest. Resumes after restarts from SQLite. No API keys needed."""
import logging
import os
import time
from logging.handlers import RotatingFileHandler

import pandas as pd

from .alerts import notify
from .config import Config
from .data import exchange, universe
from .db import DB
from .engine import compute_signals
from .portfolio import Portfolio

log = logging.getLogger("apex")
TF_MS = exchange.TF_MS


def setup_logging():
    os.makedirs("logs", exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (RotatingFileHandler("logs/apex.log", maxBytes=2_000_000, backupCount=5), logging.StreamHandler()):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.setLevel(logging.INFO)


def indexed(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.pop("ts"), unit="ms", utc=True)
    return out


class PaperRunner:
    def __init__(self, cfg: Config, symbols=None, fetch=exchange.recent_closed, ml=None, notifier=notify, tag=""):
        self.cfg, self.fetch, self.ml, self.tag = cfg, fetch, ml, tag
        self.notify = (lambda m: notifier(f"{tag} {m}".strip())) if tag else notifier
        self.db = DB(cfg.db_path)
        self.symbols = symbols or universe.load(os.path.join(cfg.data_dir, "universe.json"))
        self.tradable = set(self.symbols) - {cfg.btc_symbol}
        saved = self.db.kv_get("portfolio")
        now = pd.Timestamp.now(tz="UTC")
        self.pf = Portfolio.from_dict(cfg, saved, now, ml=ml) if saved else Portfolio(cfg, ml=ml)
        self.last_ts = self.db.kv_get("last_ts", {})
        self.saved_trades = 0

    def set_model(self, model):
        """Hot-swap the ML filter (used after a retrain). An inactive model means 'no filter'."""
        from dataclasses import replace
        on = model is not None and bool(model.active)
        self.ml = model if on else None
        self.cfg = replace(self.cfg, ml_threshold=model.threshold if on else None)
        self.pf.ml, self.pf.cfg = self.ml, self.cfg

    def _fetch_all(self):
        data = {}
        for sym in sorted(set(self.symbols) | {self.cfg.btc_symbol}):
            df = self.fetch(sym, self.cfg.timeframe, limit=500)
            if len(df) >= 300:
                data[sym] = indexed(df)
        return data

    def tick(self):
        cfg = self.cfg
        tf = TF_MS[cfg.timeframe]
        latest_closed = (int(time.time() * 1000) // tf) * tf - tf
        if self.last_ts and min(self.last_ts.values()) >= latest_closed and not os.path.exists(cfg.kill_file):
            return "idle"
        if os.path.exists(cfg.kill_file) and self.pf.manual_halt is None:
            self.pf.force_halt(pd.Timestamp.now(tz="UTC"), "KILL file")
            self.notify("KILL file found: sold everything and halted.")
        data = self._fetch_all()
        sigs = compute_signals(data, cfg)
        # group genuinely new closed bars by timestamp, oldest first
        by_ts = {}
        for sym, df in data.items():
            seen = self.last_ts.get(sym)
            if seen is None:                     # first run: start from NOW, never trade on history
                self.last_ts[sym] = int(df.index[-1].timestamp() * 1000)
                row = df.iloc[-1]
                self.pf.last[sym] = float(row.close)
                self.pf.lastinfo[sym] = (float(row.quote_volume), float(sigs[sym].atr_pct.iloc[-1]))
                continue
            for ts in df.index[df.index.map(lambda t: int(t.timestamp() * 1000)) > seen]:
                by_ts.setdefault(ts, []).append(sym)
        n_before = len(self.pf.trades)
        for ts in sorted(by_ts):
            bars = {}
            for sym in by_ts[ts]:
                r, sg = data[sym].loc[ts], sigs[sym].loc[ts]
                atr = float(sg.atr_pct) if pd.notna(sg.atr_pct) else 0.01
                row = sg.to_dict() if (sg.signal == 1 and sym in self.tradable) else None
                bars[sym] = dict(o=float(r.open), h=float(r.high), l=float(r.low), c=float(r.close),
                                 qv=float(r.quote_volume), atr_pct=atr, sig=row)
                self.last_ts[sym] = int(ts.timestamp() * 1000)
            was_ok = self.pf.can_trade
            eq = self.pf.step(ts, bars, self.tradable)
            self.db.add_equity(ts, eq, self.pf.cash)
            if was_ok and not self.pf.can_trade:
                self.notify(f"HALTED by risk limits. Equity {eq:.2f}. Everything was sold.")
        for t in self.pf.trades[n_before:]:
            self.db.add_trade(t)
            self.notify(f"CLOSED {t['sym']} {t['layer']} {t['reason']} pnl {t['pnl']:+.2f} ({t['r']:+.2f}R)")
        self.db.kv_set("portfolio", self.pf.to_dict())
        self.db.kv_set("last_ts", self.last_ts)
        return f"processed {len(by_ts)} bar(s)"

    def run(self):
        self.notify(f"Apex paper bot started: {len(self.tradable)} symbols, equity {self.pf.equity():.2f}")
        errors = 0
        while True:
            try:
                log.info(self.tick())
                errors = 0
            except Exception as e:
                errors += 1
                log.exception("tick failed")
                if errors in (3, 10, 30):
                    self.notify(f"{errors} consecutive errors: {e}")
            time.sleep(self.cfg.poll_seconds)

"""SQLite storage (stdlib, zero setup). Replaces Supabase: no cloud keys to leak."""
import contextlib
import json
import os
import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS equity(ts TEXT PRIMARY KEY, equity REAL, cash REAL);
CREATE TABLE IF NOT EXISTS trades(id INTEGER PRIMARY KEY AUTOINCREMENT, sym TEXT, layer TEXT, entry_ts TEXT,
  exit_ts TEXT, entry_px REAL, exit_px REAL, qty REAL, pnl REAL, r REAL, fees REAL, slippage REAL,
  reason TEXT, bars INTEGER, UNIQUE(sym, entry_ts));
CREATE TABLE IF NOT EXISTS kv(key TEXT PRIMARY KEY, value TEXT);
"""


class DB:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with self.conn() as c:
            c.executescript(SCHEMA)

    @contextlib.contextmanager
    def conn(self):
        c = sqlite3.connect(self.path, timeout=30)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        try:
            yield c
            c.commit()
        finally:
            c.close()

    def kv_set(self, key, value):
        with self.conn() as c:
            c.execute("INSERT OR REPLACE INTO kv VALUES (?,?)", (key, json.dumps(value, default=str)))

    def kv_get(self, key, default=None):
        with self.conn() as c:
            r = c.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return json.loads(r["value"]) if r else default

    def add_equity(self, ts, equity, cash):
        with self.conn() as c:
            c.execute("INSERT OR REPLACE INTO equity VALUES (?,?,?)", (str(ts), equity, cash))

    def add_trade(self, t: dict):
        cols = ["sym", "layer", "entry_ts", "exit_ts", "entry_px", "exit_px", "qty", "pnl", "r", "fees", "slippage", "reason", "bars"]
        with self.conn() as c:
            c.execute(f"INSERT OR IGNORE INTO trades({','.join(cols)}) VALUES ({','.join('?' * len(cols))})", [t[k] for k in cols])

    def equity_rows(self):
        with self.conn() as c:
            return [dict(r) for r in c.execute("SELECT ts, equity, cash FROM equity ORDER BY ts")]

    def trades(self, limit=None):
        with self.conn() as c:
            q = "SELECT * FROM trades ORDER BY exit_ts DESC" + (f" LIMIT {int(limit)}" if limit else "")
            return [dict(r) for r in c.execute(q)]

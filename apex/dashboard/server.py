"""Dependency-free dashboard server (Python stdlib only).

Read-only by default and bound to 127.0.0.1. For a cloud server, keep it on localhost and reach it with an SSH
tunnel (ssh -L 8000:localhost:8000 you@server), or put a reverse proxy with login in front of it.
The only write action (/api/kill) needs the DASHBOARD_TOKEN env var and a matching X-Token header.
"""
import base64
import hmac
import json
import threading
import os
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pandas as pd

from ..config import Config, arm_db_path
from ..db import DB

HERE = os.path.dirname(__file__)


def _quick(path):
    rows = DB(path).equity_rows()
    n = len(DB(path).trades())
    if not rows:
        return {"equity": None, "ret": 0.0, "trades": n}
    return {"equity": rows[-1]["equity"], "ret": rows[-1]["equity"] / rows[0]["equity"] - 1, "trades": n}


def summary(cfg: Config, arm: str = "rules") -> dict:
    main_db = DB(cfg.db_path)
    compare = {"rules": _quick(arm_db_path(cfg, "rules")), "ml": _quick(arm_db_path(cfg, "ml"))}
    ml_status, holdout = main_db.kv_get("ml_status"), main_db.kv_get("holdout_report")
    progress = main_db.kv_get("progress", "")
    cfg = replace(cfg, db_path=arm_db_path(cfg, "ml" if arm == "ml" else "rules"))
    db = DB(cfg.db_path)
    eq_rows = db.equity_rows()
    trades = db.trades()
    pf = db.kv_get("portfolio") or {}
    start_eq = eq_rows[0]["equity"] if eq_rows else cfg.initial_capital
    now_eq = eq_rows[-1]["equity"] if eq_rows else cfg.initial_capital
    peak = max([r["equity"] for r in eq_rows] + [now_eq])
    step = max(1, len(eq_rows) // 500)
    curve = [[r["ts"], r["equity"]] for r in eq_rows[::step]]
    if eq_rows and curve[-1][0] != eq_rows[-1]["ts"]:
        curve.append([eq_rows[-1]["ts"], now_eq])
    fees = sum(t["fees"] for t in trades)
    slip = sum(t["slippage"] for t in trades)
    net = sum(t["pnl"] for t in trades)
    g = pf.get("guard") or {}
    reason = pf.get("manual_halt") or ("max drawdown" if g.get("halted_dd") else ("daily loss limit" if g.get("halted_day") else None))
    last = pf.get("last", {})
    positions = []
    for p in (pf.get("pos") or {}).values():
        px = last.get(p["sym"], p["entry_px"])
        positions.append({"sym": p["sym"], "layer": p["layer"], "entry": p["entry_px"], "stop": p["stop"], "tp": p["tp"],
                          "last": px, "pnl_pct": px / p["entry_px"] - 1, "value": p["qty"] * px})
    layers = {}
    for t in trades:
        d = layers.setdefault(t["layer"], {"layer": t["layer"], "trades": 0, "wins": 0, "r": 0.0, "pnl": 0.0})
        d["trades"] += 1
        d["wins"] += t["pnl"] > 0
        d["r"] += t["r"]
        d["pnl"] += t["pnl"]
    layer_rows = [{"layer": d["layer"], "trades": d["trades"], "win_rate": d["wins"] / d["trades"],
                   "avg_r": d["r"] / d["trades"], "pnl": d["pnl"]} for d in layers.values()]
    updated = eq_rows[-1]["ts"] if eq_rows else None
    age = None
    if updated:
        age = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(updated)).total_seconds()
    return {
        "mode": "paper", "server_time": time.time(), "arm": arm, "progress": progress,
        "compare": compare,
        "ml": {"active": bool(ml_status and ml_status.get("active")), "trained_at": (ml_status or {}).get("trained_at"),
               "report": (ml_status or {}).get("report"), "has_model": ml_status is not None},
        "holdout": {"verdict": holdout.get("verdict"), "start": holdout.get("start")} if holdout else None,
        "status": {"halted": reason is not None, "reason": reason, "updated": updated, "age_seconds": age,
                   "has_data": bool(eq_rows)},
        "equity": {"now": now_eq, "start": start_eq, "ret": now_eq / start_eq - 1 if start_eq else 0,
                   "peak": peak, "drawdown": now_eq / peak - 1 if peak else 0, "cash": pf.get("cash")},
        "risk": {"day_pnl": (now_eq / g["day_start"] - 1) if g.get("day_start") else 0.0,
                 "day_limit": cfg.daily_loss_limit, "dd_limit": cfg.max_drawdown,
                 "open": len(positions), "max_open": cfg.max_open_positions},
        "costs": {"gross": net + fees + slip, "fees": fees, "slippage": slip, "net": net, "trades": len(trades)},
        "layers": sorted(layer_rows, key=lambda r: -r["pnl"]),
        "positions": positions,
        "trades": [{k: t[k] for k in ("sym", "layer", "exit_ts", "pnl", "r", "reason")} for t in trades[:15]],
        "curve": curve,
    }


def make_handler(cfg: Config):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _authed(self) -> bool:
            pw = os.getenv("DASHBOARD_PASSWORD", "")
            if not pw:
                return True                       # only allowed on localhost (see serve())
            h = self.headers.get("Authorization", "")
            try:
                given = base64.b64decode(h.split(" ", 1)[1]).decode().split(":", 1)[1] if h.startswith("Basic ") else ""
            except Exception:
                given = ""
            if hmac.compare_digest(given, pw):
                return True
            data = b"Login required"
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Basic realm="apex"')
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return False

        def do_GET(self):
            if self.path == "/health":            # no data, no login: for the host's health check
                return self._send(200, {"ok": True})
            if not self._authed():
                return
            if self.path in ("/", "/index.html"):
                with open(os.path.join(HERE, "index.html"), "rb") as fh:
                    return self._send(200, fh.read(), "text/html; charset=utf-8")
            if self.path.startswith("/api/summary"):
                arm = parse_qs(urlparse(self.path).query).get("arm", ["rules"])[0]
                return self._send(200, summary(cfg, "ml" if arm == "ml" else "rules"))
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if not self._authed():
                return
            if self.path != "/api/kill":
                return self._send(404, {"error": "not found"})
            token = os.getenv("DASHBOARD_TOKEN", "") or os.getenv("DASHBOARD_PASSWORD", "")
            if not token:
                return self._send(403, {"error": "Set the DASHBOARD_TOKEN environment variable to enable the stop button."})
            if not hmac.compare_digest(self.headers.get("X-Token", ""), token):
                return self._send(403, {"error": "Wrong token."})
            os.makedirs(os.path.dirname(cfg.kill_file) or ".", exist_ok=True)
            open(cfg.kill_file, "w").close()
            self._send(200, {"ok": True, "message": "Stop requested. The bot sells everything on its next check."})
    return Handler


def _check_exposure(host):
    if host not in ("127.0.0.1", "localhost") and not os.getenv("DASHBOARD_PASSWORD"):
        raise SystemExit("Refusing to expose the dashboard publicly without a password. Set DASHBOARD_PASSWORD.")


def serve_background(cfg: Config, host="127.0.0.1", port=8000):
    _check_exposure(host)
    srv = ThreadingHTTPServer((host, port), make_handler(cfg))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def serve(cfg: Config, host="127.0.0.1", port=8000):
    _check_exposure(host)
    srv = ThreadingHTTPServer((host, port), make_handler(cfg))
    print(f"Dashboard on http://{host}:{port}  (Ctrl+C to stop)")
    srv.serve_forever()

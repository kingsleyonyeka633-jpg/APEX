"""Portfolio + execution state machine. Backtest and live/paper BOTH drive this exact class,
one closed bar at a time, so they cannot diverge.

Per bar (timestamp t):
  1. orders decided at the previous close fill at THIS bar's open (with spread/impact/fees)
  2. open positions check stop (gap-aware) -> take-profit -> time stop
  3. mark to market, update kill switches (flatten everything if halted)
  4. new signals at THIS bar's close create orders for the NEXT open
Spot, long-only: SELL signals are ignored (shorting needs margin/perps).
"""
from dataclasses import asdict, dataclass

import pandas as pd

from .broker import FillModel, round_qty
from .config import Config
from .risk import RiskGuard


@dataclass
class Position:
    sym: str
    layer: str
    qty: float
    qty0: float
    entry_ts: str
    entry_px: float
    entry_ref: float
    stop: float
    tp: float
    risk_amt: float
    cost_basis: float
    entry_fee: float
    fees: float
    slip_cost: float
    proceeds: float = 0.0
    bars: int = 0
    exit_pending: str | None = None


class Portfolio:
    def __init__(self, cfg: Config, cash: float | None = None, ml=None):
        self.cfg, self.fm, self.ml = cfg, FillModel(cfg), ml
        self.cash = float(cfg.initial_capital if cash is None else cash)
        self.pos: dict[str, Position] = {}
        self.pending: dict[str, dict] = {}
        self.last: dict[str, float] = {}
        self.lastinfo: dict[str, tuple] = {}
        self.guard: RiskGuard | None = None
        self.manual_halt: str | None = None
        self.trades: list[dict] = []
        self.eq_log: list[tuple] = []
        self.cost = {"fees": 0.0, "slippage": 0.0}
        self.signals_seen = self.signals_taken = 0
        self.max_open_seen, self.max_exposure_seen = 0, 0.0

    # ---------- helpers ----------
    def equity(self) -> float:
        return self.cash + sum(p.qty * self.last.get(p.sym, p.entry_px) for p in self.pos.values())

    def exposure(self) -> float:
        return sum(p.qty * self.last.get(p.sym, p.entry_px) for p in self.pos.values())

    @property
    def can_trade(self) -> bool:
        return self.manual_halt is None and (self.guard is None or self.guard.can_trade)

    # ---------- bar processing ----------
    def on_bar(self, ts, sym, o, h, l, c, qv, atr_pct):
        cfg = self.cfg
        pe = self.pending.pop(sym, None)
        if pe is not None and self.can_trade:
            self._fill_entry(ts, sym, pe, o)
        p = self.pos.get(sym)
        if p is not None:
            p.bars += 1
            if p.exit_pending:
                self._sell(ts, p, o, qv, atr_pct, p.exit_pending)
            elif l <= p.stop:
                self._sell(ts, p, min(o, p.stop), qv, atr_pct, "stop", slip_mult=cfg.stop_slip_mult)
            elif h >= p.tp * (1 + cfg.tp_trade_through_bps / 1e4):
                self._sell(ts, p, p.tp, qv, atr_pct, "tp", maker=True)
            elif p.bars >= cfg.max_hold_bars:
                p.exit_pending = "time"          # executes at next open
        self.last[sym] = c
        self.lastinfo[sym] = (qv, atr_pct)

    def mark(self, ts) -> float:
        eq = self.equity()
        if self.guard is None:
            self.guard = RiskGuard(self.cfg, eq, ts)
        self.guard.update(ts, eq)
        if not self.can_trade:
            self.pending.clear()
            for sym in list(self.pos):
                p = self.pos[sym]
                qv, atr = self.lastinfo.get(sym, (1e12, 0.01))
                self._sell(ts, p, self.last.get(sym, p.entry_px), qv, atr, "risk_halt", slip_mult=self.cfg.stop_slip_mult)
            eq = self.equity()
        self.max_open_seen = max(self.max_open_seen, len(self.pos))
        self.max_exposure_seen = max(self.max_exposure_seen, self.exposure() / eq if eq > 0 else 0.0)
        self.eq_log.append((ts, eq))
        return eq

    def force_halt(self, ts, reason: str):
        self.manual_halt = reason
        self.mark(ts)

    def on_signal(self, ts, sym, row: dict, tradable_slot_check=True) -> bool:
        self.signals_seen += 1
        cfg = self.cfg
        if not self.can_trade or sym in self.pos or sym in self.pending:
            return False
        if len(self.pos) + len(self.pending) >= cfg.max_open_positions:
            return False
        if self.ml is not None and cfg.ml_threshold is not None and self.ml.predict(row) < cfg.ml_threshold:
            return False
        qv, atr_pct = self.lastinfo[sym]
        self.pending[sym] = {"stop": float(row["stop_loss"]), "tp": float(row["take_profit"]),
                             "layer": row["layer"], "qv": qv, "atr_pct": atr_pct, "signal_ts": str(ts)}
        self.signals_taken += 1
        return True

    def step(self, ts, bars: dict, tradable) -> float:
        """bars[sym] = dict(o,h,l,c,qv,atr_pct,sig) where sig is a signal-row dict or None."""
        for sym, b in bars.items():
            self.on_bar(ts, sym, b["o"], b["h"], b["l"], b["c"], b["qv"], b["atr_pct"])
        eq = self.mark(ts)
        for sym, b in bars.items():
            if sym in tradable and b.get("sig") is not None:
                self.on_signal(ts, sym, b["sig"])
        return eq

    # ---------- fills ----------
    def _fill_entry(self, ts, sym, pe, o):
        """Fill at this bar's open. Liquidity/volatility inputs come from the DECISION bar (known at the time)."""
        cfg, fm = self.cfg, self.fm
        stop, tp, qv, atr_pct = pe["stop"], pe["tp"], pe["qv"], pe["atr_pct"]
        dist = o - stop
        if dist <= 0 or o >= tp:
            return
        eq = self.equity()
        risk_amt = cfg.risk_per_trade * eq
        room = max(cfg.max_total_exposure * eq - self.exposure(), 0.0)
        cap = min(cfg.max_position_frac * eq, room, fm.max_notional(qv), self.cash / (1 + cfg.taker_fee))
        notional = min(risk_amt / dist * o, cap)
        if notional < cfg.min_notional:
            return
        qty = notional / o
        px = fm.buy_price(o, qty * o, qv, atr_pct)
        if px - stop <= 0 or px >= tp:
            return
        qty = round_qty(min(qty, risk_amt / (px - stop)))          # re-size to the REAL fill price
        cost = qty * px
        fee = fm.fee(cost)
        if cost < cfg.min_notional or cost + fee > self.cash:
            return
        self.cash -= cost + fee
        slip = (px - o) * qty
        self.cost["fees"] += fee
        self.cost["slippage"] += slip
        self.pos[sym] = Position(sym=sym, layer=pe["layer"], qty=qty, qty0=qty, entry_ts=str(ts), entry_px=px,
                                 entry_ref=o, stop=stop, tp=tp, risk_amt=risk_amt, cost_basis=cost + fee,
                                 entry_fee=fee, fees=fee, slip_cost=slip)

    def _sell(self, ts, p, ref, qv, atr_pct, reason, maker=False, slip_mult=1.0):
        cfg, fm = self.cfg, self.fm
        if maker:
            qty, px = p.qty, ref
        else:
            cap_q = fm.max_notional(qv) / max(ref, 1e-12)
            dust = p.qty * ref < cfg.min_notional
            qty = p.qty if dust else min(p.qty, cap_q)
            if qty <= 0:
                p.exit_pending = reason
                return
            px = fm.sell_price(ref, qty * ref, qv, atr_pct, slip_mult)
        notional = qty * px
        fee = fm.fee(notional, maker)
        self.cash += notional - fee
        p.proceeds += notional - fee
        p.fees += fee
        slip = 0.0 if maker else (ref - px) * qty
        p.slip_cost += slip
        self.cost["fees"] += fee
        self.cost["slippage"] += slip
        p.qty -= qty
        if p.qty <= 1e-12:
            pnl = p.proceeds - p.cost_basis
            exit_fees = p.fees - p.entry_fee
            self.trades.append({
                "sym": p.sym, "layer": p.layer, "entry_ts": p.entry_ts, "exit_ts": str(ts),
                "entry_px": p.entry_px, "exit_px": (p.proceeds + exit_fees) / p.qty0, "qty": p.qty0,
                "pnl": pnl, "r": pnl / p.risk_amt if p.risk_amt else 0.0, "fees": p.fees,
                "slippage": p.slip_cost, "reason": reason, "bars": p.bars})
            del self.pos[p.sym]
        else:
            p.exit_pending = reason       # leftover (volume cap): keep unwinding next bar

    # ---------- persistence ----------
    def to_dict(self):
        return {"cash": self.cash, "pos": {k: asdict(v) for k, v in self.pos.items()}, "pending": self.pending,
                "last": self.last, "lastinfo": {k: list(v) for k, v in self.lastinfo.items()},
                "guard": self.guard.to_dict() if self.guard else None, "manual_halt": self.manual_halt,
                "cost": self.cost}

    @classmethod
    def from_dict(cls, cfg, d, ts, ml=None):
        pf = cls(cfg, cash=d["cash"], ml=ml)
        pf.pos = {k: Position(**v) for k, v in d["pos"].items()}
        pf.pending, pf.last = d["pending"], d["last"]
        pf.lastinfo = {k: tuple(v) for k, v in d["lastinfo"].items()}
        pf.guard = RiskGuard.from_dict(cfg, d["guard"], ts) if d.get("guard") else None
        pf.manual_halt, pf.cost = d.get("manual_halt"), d.get("cost", pf.cost)
        return pf

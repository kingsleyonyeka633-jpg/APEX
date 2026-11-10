from .config import Config


class RiskGuard:
    """Daily loss limit (pause until next UTC day) and max drawdown (halt until a human resets)."""

    def __init__(self, cfg: Config, equity: float, ts):
        self.cfg, self.peak = cfg, equity
        self.day, self.day_start = ts.date(), equity
        self.halted_day = self.halted_dd = False

    def update(self, ts, equity: float):
        if ts.date() != self.day:
            self.day, self.day_start, self.halted_day = ts.date(), equity, False
        self.peak = max(self.peak, equity)
        if equity <= self.peak * (1 - self.cfg.max_drawdown):
            self.halted_dd = True
        if equity <= self.day_start * (1 - self.cfg.daily_loss_limit):
            self.halted_day = True

    @property
    def can_trade(self) -> bool:
        return not (self.halted_day or self.halted_dd)

    def to_dict(self):
        return {"peak": self.peak, "day": str(self.day), "day_start": self.day_start,
                "halted_day": self.halted_day, "halted_dd": self.halted_dd}

    @classmethod
    def from_dict(cls, cfg, d, ts):
        g = cls(cfg, d["peak"], ts)
        g.halted_dd = d["halted_dd"]
        if str(ts.date()) == d["day"]:
            g.day_start, g.halted_day = d["day_start"], d["halted_day"]
        return g

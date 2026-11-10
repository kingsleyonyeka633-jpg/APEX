"""Simulated exchange fills. The same model drives backtests AND paper trading, so paper results are
comparable to the backtest. It is deliberately pessimistic:

  buy price  = reference * (1 + half_spread + impact)
  sell price = reference * (1 - half_spread - impact)
  half_spread = max(floor, k * ATR%)           wider when the coin is volatile
  impact      = coef * ATR% * sqrt(order $ / bar traded $)     square-root market-impact law
  fill size   <= participation_cap * bar traded $              you cannot buy more than the market trades
  stop orders slip stop_slip_mult times more; limit take-profits need price to trade THROUGH them
  fees on every fill; quantities rounded down; orders under min_notional rejected
"""
import math

from .config import Config


class FillModel:
    def __init__(self, cfg: Config):
        self.c = cfg

    def half_spread(self, atr_pct: float) -> float:
        return max(self.c.min_half_spread_bps, self.c.spread_vol_k * atr_pct * 1e4) / 1e4

    def impact(self, notional: float, bar_quote_vol: float, atr_pct: float) -> float:
        return self.c.impact_coef * atr_pct * math.sqrt(max(notional, 0.0) / max(bar_quote_vol, 1.0))

    def slip(self, notional, bar_quote_vol, atr_pct, mult=1.0) -> float:
        return (self.half_spread(atr_pct) + self.impact(notional, bar_quote_vol, atr_pct)) * mult

    def buy_price(self, ref, notional, bar_quote_vol, atr_pct, mult=1.0) -> float:
        return ref * (1 + self.slip(notional, bar_quote_vol, atr_pct, mult))

    def sell_price(self, ref, notional, bar_quote_vol, atr_pct, mult=1.0) -> float:
        return ref * (1 - self.slip(notional, bar_quote_vol, atr_pct, mult))

    def max_notional(self, bar_quote_vol: float) -> float:
        return self.c.participation_cap * max(bar_quote_vol, 0.0)

    def fee(self, notional: float, maker: bool = False) -> float:
        return notional * (self.c.maker_fee if maker else self.c.taker_fee)


def round_qty(qty: float) -> float:
    """Round DOWN to 4 significant digits (stand-in for exchange lot-size rules)."""
    if qty <= 0:
        return 0.0
    step = 10 ** (math.floor(math.log10(qty)) - 3)
    return math.floor(qty / step + 1e-9) * step

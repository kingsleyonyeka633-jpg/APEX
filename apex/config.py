import os
from dataclasses import dataclass

HOME = os.getenv("APEX_HOME", ".")     # on Railway set APEX_HOME=/data (the persistent volume)


@dataclass(frozen=True)
class Config:
    # ---- market / paths ----
    quote: str = "USDT"
    timeframe: str = "15m"
    btc_symbol: str = "BTCUSDT"            # used only for the market-trend gate
    data_dir: str = os.path.join(HOME, "data")
    db_path: str = os.path.join(HOME, "state", "apex.db")
    kill_file: str = os.path.join(HOME, "state", "KILL")
    model_path: str = os.path.join(HOME, "models", "filter.json")

    # ---- mid-cap universe (ranked by 24h USDT volume) ----
    universe_rank_from: int = 15           # skip the mega caps at the top
    universe_rank_to: int = 80
    min_daily_quote_volume: float = 5_000_000
    mega_caps: tuple = ("BTC", "ETH", "BNB", "XRP", "SOL", "USDC", "FDUSD", "TUSD")

    # ---- simulated broker: make it hurt like the real market ----
    taker_fee: float = 0.0010              # 0.10% per side (Binance/Bybit spot default tier)
    maker_fee: float = 0.0010
    min_half_spread_bps: float = 2.0       # floor on half the bid/ask spread
    spread_vol_k: float = 0.08             # spread widens with volatility (8% of ATR%, in bps)
    impact_coef: float = 0.6               # square-root market impact: coef * ATR% * sqrt(order/bar volume)
    participation_cap: float = 0.10        # never fill more than 10% of a bar's traded volume
    stop_slip_mult: float = 1.5            # stop-market orders slip 50% more than normal market orders
    tp_trade_through_bps: float = 1.0      # price must trade THROUGH the target to fill it
    min_notional: float = 10.0

    # ---- risk ----
    reward_risk: float = 2.0               # take-profit distance as a multiple of the stop distance
    initial_capital: float = 1000.0
    risk_per_trade: float = 0.005          # 0.5% of equity lost if the stop is hit
    max_position_frac: float = 0.20
    max_total_exposure: float = 0.60
    max_open_positions: int = 5
    max_hold_bars: int = 192               # time stop: 192 x 15m = 2 days
    daily_loss_limit: float = 0.03
    max_drawdown: float = 0.15

    # ---- optional ML filter ----
    ml_threshold: float | None = None      # set after `apex train`; None = no ML filter

    # ---- auto mode ----
    refresh_hours: int = 24                # download newest candles this often
    retrain_days: int = 7                  # retrain the ML filter this often
    min_rows_per_coin: int = 3000          # ~31 days of 15m bars; fewer = coin skipped
    max_coins: int = int(os.getenv("APEX_MAX_COINS", 12))   # memory guard: small hosts need few coins

    # ---- runtime ----
    poll_seconds: int = 60


DEFAULT = Config()


def arm_db_path(cfg: Config, arm: str) -> str:
    """'rules' = strategy only. 'ml' = strategy + ML filter. Two paper bots, same market, separate books."""
    return cfg.db_path if arm == "rules" else cfg.db_path.replace(".db", "_ml.db")

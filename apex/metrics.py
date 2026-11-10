import numpy as np
import pandas as pd


def bars_per_year(idx) -> float:
    step = pd.Series(idx).diff().median().total_seconds()
    return 365.25 * 86400 / step


def compute(eq: pd.Series, trades: pd.DataFrame, cost: dict, bpy: float) -> dict:
    r = eq.pct_change().dropna()
    years = max(len(eq) / bpy, 1e-9)
    ratio = eq.iloc[-1] / eq.iloc[0]
    dd = (eq / eq.cummax() - 1).min()
    sd = r.std()
    n = len(trades)
    wins = trades[trades.pnl > 0].pnl.sum() if n else 0.0
    losses = -trades[trades.pnl < 0].pnl.sum() if n else 0.0
    net = float(eq.iloc[-1] - eq.iloc[0])
    gross = net + cost["fees"] + cost["slippage"]
    cagr = ratio ** (1 / years) - 1 if ratio > 0 else -1.0
    return {
        "start_equity": float(eq.iloc[0]), "end_equity": float(eq.iloc[-1]), "net_pnl": net,
        "total_return": float(ratio - 1), "cagr": float(cagr),
        "sharpe": float(r.mean() / sd * np.sqrt(bpy)) if sd > 0 else 0.0,
        "max_drawdown": float(dd), "calmar": float(cagr / abs(dd)) if dd < 0 else 0.0,
        "trades": n, "win_rate": float((trades.pnl > 0).mean()) if n else 0.0,
        "profit_factor": float(wins / losses) if losses > 0 else (float("inf") if wins > 0 else 0.0),
        "avg_r": float(trades.r.mean()) if n else 0.0,
        "gross_pnl_before_costs": float(gross), "fees_paid": float(cost["fees"]), "slippage_paid": float(cost["slippage"]),
        "costs_pct_of_gross": float((cost["fees"] + cost["slippage"]) / gross) if gross > 0 else None,
    }


def breakdown(trades: pd.DataFrame, key: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    g = trades.groupby(key)
    return pd.DataFrame({"trades": g.size(), "win_rate": g.pnl.apply(lambda s: (s > 0).mean()),
                         "avg_r": g.r.mean(), "net_pnl": g.pnl.sum()}).sort_values("net_pnl", ascending=False)

# Apex (Python)

## Easiest way (Windows)
1. Install Python from python.org (tick "Add python.exe to PATH").
2. Double-click **START.bat**. First time it installs what it needs.
3. Your browser opens the dashboard. The bot picks coins, downloads data, checks it, trains the ML filter,
   tests it on history it never saw, then paper trades. It refreshes data daily and retrains weekly by itself.
Close the black window to stop. Start again any time; it resumes where it stopped.

Mid-cap crypto quant system. One language, one brain: data, strategies, simulated broker, risk,
paper trading, ML filter and dashboard. Spot, long-only. Paper money until YOU decide otherwise.

## Quick start
    pip install -r requirements.txt
    python tests/test_core.py && python tests/test_system.py     # 15 checks
    python -m apex backtest --synthetic                           # fake market, machinery check only

## Real data (needs internet)
    python -m apex universe                      # picks mid-caps by 24h volume (skips BTC/ETH/BNB/XRP/SOL, stables, leveraged tokens)
    python -m apex download --years 5            # Binance bulk archive + REST, resumable. BTC is always fetched (trend gate)
    python -m apex quality                       # gaps / bad candles BEFORE you trust a backtest
    python -m apex backtest --end 2025-06-01     # in-sample
    python -m apex backtest --start 2025-06-01   # held-out period: THIS is the number that matters
    python -m apex train && python -m apex backtest --ml   # optional ML filter; auto-OFF if it fails its test slice

## Paper trade on live prices
    python -m apex paper        # polls closed 15m bars, same code and fill model as the backtest
    python -m apex dashboard    # http://127.0.0.1:8000
    touch state/KILL            # sells everything and halts. `python -m apex resume` to clear halts.

## The simulated broker (apex/broker.py)
fees both sides; half-spread that widens with volatility; square-root market impact vs the bar's traded
volume; fills capped at 10% of a bar's volume; stop orders slip 1.5x; take-profits must be traded through;
orders fill at the NEXT bar open; lot rounding and minimum order size. Tune everything in config.py and make
it MORE pessimistic if live results come in worse than paper.

## Railway (easiest)
See RAILWAY.md. `python -m apex run` starts the dashboard and the paper bot in one process.

## Cloud (own server)
VPS (Europe region), install Docker, copy folder, `cp .env.example .env`, edit, `docker compose up -d --build`.
Dashboard is bound to localhost: `ssh -L 8000:localhost:8000 you@server`. Never expose it without a login proxy.

## Honest limits
- Only 10 years of history exists for coins that existed 10 years ago (a handful). Most mid-caps start 2019-2021.
- Coins that were delisted vanish from "top by volume" lists: survivorship bias flatters mid-cap backtests.
- The downloaders were written from the public docs and tested against FAKE responses; run a small one first.
- Dashboard was syntax-checked and its API tested, but not viewed in a real browser by the author.
- No live-order code on purpose. Add exchange execution only after weeks of paper results you believe.
- Not financial advice.

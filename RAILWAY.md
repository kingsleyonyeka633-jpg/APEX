# Deploy on Railway (auto mode)

1. Put this folder in a PRIVATE GitHub repo (`.env`, data and state are already git-ignored).
2. railway.com -> New Project -> Deploy from GitHub repo. It builds the Dockerfile, which runs `python -m apex auto`.
3. Service -> Variables:
   - APEX_HOME = /data
   - DASHBOARD_PASSWORD = a long random password (the dashboard refuses to start publicly without it)
   - TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID (optional, recommended: trade and halt alerts)
4. Service -> add a Volume mounted at /data. Without it everything is forgotten on each restart.
5. Settings -> Region -> Europe (Binance data can be blocked from US regions).
6. Settings -> Networking -> Generate Domain. Open it: any username, your DASHBOARD_PASSWORD.
The FIRST start downloads years of data for ~60 coins and trains the model: expect it to look 'Getting ready' for a while.

Free / small plans: memory is the limit. The Dockerfile starts with `--years 1` and the bot uses at most 12 coins (APEX_MAX_COINS, default 12).
Measured peak at startup is about 200 MB. Raise APEX_MAX_COINS only after checking the Metrics tab.

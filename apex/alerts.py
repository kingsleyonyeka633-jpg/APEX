import json
import logging
import os
import urllib.request

log = logging.getLogger("quantbot")


def notify(text: str):
    """Logs always; also sends to Telegram if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are set."""
    log.info(text)
    token, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not (token and chat):
        return
    try:
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=json.dumps({"chat_id": chat, "text": f"[quantbot] {text}"}).encode(),
            headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:  # alerts must never crash the bot
        log.warning("telegram failed: %s", e)

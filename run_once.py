#!/usr/bin/env python3
"""One-shot story delivery for GitHub Actions (no Telegram polling)."""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from dotenv import load_dotenv
from telegram import Bot
from telegram.error import BadRequest, Forbidden

load_dotenv()

import state
from bot import _bootstrap_subscribers, _token, deliver, prepare_story, validate_token

log = logging.getLogger("kitab.run_once")


async def run() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _bootstrap_subscribers()
    token = _token()
    me = validate_token(token)
    username = me.get("username") or ""
    log.info("token ok bot=@%s id=%s", username, me.get("id"))
    expected = (os.getenv("EXPECTED_BOT_USERNAME") or "Bot4ttobot").lstrip("@")
    if expected and username and username.lower() != expected.lower():
        raise SystemExit(
            f"Wrong Telegram bot token: got @{username}, expected @{expected}. "
            "Update TELEGRAM_BOT_TOKEN secret."
        )

    payload = await asyncio.to_thread(prepare_story, True)
    chats = state.subscribers()
    log.info(
        "prepared title=%s subscribers=%s",
        (payload.get("title") or "")[:80],
        len(chats),
    )
    if not chats:
        log.warning("no subscribers; nothing to send")
        return

    bot = Bot(token)
    ok = 0
    failed = 0
    async with bot:
        for chat_id in chats:
            try:
                await deliver(bot, chat_id, payload)
                log.info("delivered chat_id=%s", chat_id)
                ok += 1
            except (BadRequest, Forbidden) as exc:
                failed += 1
                log.warning("skip chat_id=%s err=%s", chat_id, exc)
                # Drop dead chats so future runs stay clean
                try:
                    state.unsubscribe(chat_id)
                except Exception:
                    pass
            except Exception:
                failed += 1
                log.exception("deliver failed chat_id=%s", chat_id)

    log.info("done ok=%s failed=%s", ok, failed)
    if ok == 0:
        raise SystemExit("No successful deliveries")


def main() -> None:
    try:
        asyncio.run(run())
    except SystemExit:
        raise
    except Exception:
        log.exception("run_once failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""One-shot story delivery for GitHub Actions (no Telegram polling)."""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from dotenv import load_dotenv
from telegram import Bot

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
    log.info("token ok bot=@%s id=%s", me.get("username") or "", me.get("id"))

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
    async with bot:
        for chat_id in chats:
            try:
                await deliver(bot, chat_id, payload)
                log.info("delivered chat_id=%s", chat_id)
            except Exception:
                log.exception("deliver failed chat_id=%s", chat_id)
                raise


def main() -> None:
    try:
        asyncio.run(run())
    except Exception:
        log.exception("run_once failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

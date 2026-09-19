"""Telegram bot: folk tales 4×/day (Asia/Riyadh) — plain Arabic text+audio."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
import json
import uuid
from pathlib import Path
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, InputFile, Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters

import books
import state
import summarize
import tts

log = logging.getLogger("kitab.bot")

TZ = ZoneInfo("Asia/Riyadh")
WELCOME_AR = (
    "مرحباً بك في بوت الحكايات الشعبية.\n\n"
    "أربع مرات يومياً بتوقيت الرياض (08:00، 12:00، 16:00، 20:00):\n"
    "حكاية كاملة في ملف نص، ثم صوت ايديج تلقائياً. وللجيميناي: /gemini\n\n"
    "المصدر يتناوب بين أرشيف الإنترنت: عربي ← إنجليزي ← روسي ← أردو، "
    "والتسليم دائماً بالعربية المشكّلة.\n\n"
    "تم اشتراك هذه المحادثة.\n"
    "الأوامر:\n"
    "/now حكاية الآن\n"
    "/status الحالة واللغة التالية\n"
    "/stop إلغاء الاشتراك"
)
ALREADY_AR = "هذه المحادثة مشتركة بالفعل. أرسل /now لطلب حكاية الآن، أو /status."
STOPPED_AR = "تم إلغاء الاشتراك. يمكنك العودة متى شئت عبر /start."
NOT_SUB_AR = "هذه المحادثة غير مشتركة. أرسل /start للاشتراك."
PREPARING_AR = "جاري تجهيز الحكاية، قد يستغرق هذا دقيقة أو اثنتين…"
BUSY_AR = "يوجد طلب قيد التجهيز لهذه المحادثة. انتظر قليلاً ثم أعد المحاولة."
NOW_KEYBOARD = InlineKeyboardMarkup(
    [[InlineKeyboardButton("حكاية الآن", callback_data="now:folk")]]
)
TTS_MENU_PREFIX = "ttsmenu:"
TTS_ENGINE_PREFIX = "ttseng:"  # ttseng:gemini:<token> | ttseng:edge:<token>
TTS_HINT_AR = "لتحويل النص الى صوت اضغط تحويل"
TTS_CHOOSE_AR = "اختر محرك التحويل إلى صوت:"
CONVERTING_AR = "جارٍ تحويل النص إلى صوت…"
TTS_DONE_AR = "تم التحويل ✓"
TTS_FAIL_AR = "تعذّر التحويل إلى صوت الآن. حاول مرة أخرى."
PENDING_TTS_DIR = Path(__file__).resolve().parent / "data" / "pending_tts"
_busy_chats: set[int] = set()
_awaiting_gemini: set[int] = set()
GEMINI_ASK_AR = "أرسل النص الذي تريد تحويله إلى صوت جيميناي (نصاً أو ملف .txt)."
GEMINI_CANCEL_AR = "تم إلغاء انتظار نص جيميناي. أرسل /gemini مرة أخرى عند الحاجة."


def _token() -> str:
    return (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()


def validate_token(token: str) -> dict:
    if not token:
        print(
            "ERROR: TELEGRAM_BOT_TOKEN is missing or empty.\n"
            "ضع التوكن في ملف .env ثم أعد التشغيل.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    url = f"https://api.telegram.org/bot{token}/getMe"
    try:
        with httpx.Client(timeout=20.0) as client:
            r = client.get(url)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: could not reach Telegram API: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    try:
        data = r.json()
    except Exception:
        data = {}
    if r.status_code == 401 or not data.get("ok"):
        desc = (data.get("description") or r.text or "Unauthorized")[:200]
        print(
            "ERROR: Telegram rejected the bot token "
            f"(HTTP {r.status_code}: {desc}).",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return data.get("result") or {}


def prepare_story(advance_cycle: bool = True) -> dict:
    """Pick next unused non-Arabic history episode; Arabic rewrite (TTS on button).

    Anti-repeat formula:
      1) skip if identifier#index already in used_stories
      2) skip if Arabic title fingerprint already in used_titles
    """
    if advance_cycle:
        source_lang = state.advance_source_language() or "eng"
    else:
        source_lang = state.peek_source_language() or "eng"
    if source_lang == "ara":
        source_lang = "eng"
    used = set(state.used_stories())
    used_titles = set(state.used_title_fps())
    last_err: Exception | None = None
    max_tries = 8
    for attempt in range(1, max_tries + 1):
        story = books.pick_next_story(used=used, language=source_lang)
        ident = story["identifier"]
        idx = int(story["story_index"])
        src_title = story.get("title") or "نادرة"
        collection_title = story.get("collection_title") or ""
        author = story.get("author") or ""
        key = f"{ident}#{idx}"
        try:
            title, source_label, parts, full = summarize.build_folk_story(
                story["text"],
                src_title,
                source_lang=story.get("language") or source_lang,
                author=author,
                collection_title=collection_title,
            )
        except summarize.GeminiProhibitedError as exc:
            last_err = exc
            log.warning(
                "story blocked by Gemini safety try=%s/%s key=%s — skipping",
                attempt, max_tries, key,
            )
            state.mark_story_used(ident, idx, title=f"[skipped-prohibited] {src_title}"[:80])
            used.add(key)
            continue
        except summarize.SummarizeError as exc:
            cause = exc.__cause__
            if isinstance(cause, summarize.GeminiProhibitedError) or (
                "PROHIBITED" in str(cause or exc).upper()
            ):
                last_err = exc
                log.warning(
                    "story summarize prohibited try=%s/%s key=%s — skipping",
                    attempt, max_tries, key,
                )
                state.mark_story_used(ident, idx, title=f"[skipped-prohibited] {src_title}"[:80])
                used.add(key)
                continue
            raise


        # Reject non-narrative / generic book-title outputs
        body0 = (full or "").strip()
        if body0.startswith("رفض") or "مادة غير سردية" in body0[:80]:
            log.warning("gemini refused non-narrative slice try=%s key=%s", attempt, key)
            state.mark_story_used(ident, idx, title=f"[skipped-non-narrative] {src_title}"[:80])
            used.add(key)
            continue
        generic = {
            "سير بلوتارخ", "قصة تاريخية", "حكاية شعبية", "نادرة",
            "مئة حكاية مرحة", "نوادر إنجليزية", (src_title or "").strip(),
            (collection_title or "").strip(),
        }
        title_s = (title or "").strip()
        bookish_bits = ("نوادر إنجليز", "طرائف إنجليز", "جمعت من مصادر", "مجموعة من مصادر", "Collected from")
        if (
            title_s in generic
            or len(title_s) < 6
            or any(b in title_s for b in bookish_bits)
        ):
            log.warning("generic/bookish title %r try=%s key=%s — skipping", title, attempt, key)
            state.mark_story_used(ident, idx, title=f"[skipped-generic-title] {title}"[:80])
            used.add(key)
            continue
        # Anti-repeat part 2: title fingerprint (pre-review)
        title_fp = state.title_fingerprint(title)
        if title_fp and title_fp in used_titles:
            log.warning(
                "duplicate Arabic title fp=%s try=%s/%s key=%s — skipping",
                title_fp[:40], attempt, max_tries, key,
            )
            state.mark_story_used(ident, idx, title=f"[skipped-dup-title] {title}"[:80])
            used.add(key)
            continue

        # Cohere final review: attractive title + professional complete output
        try:
            title, full = summarize.review_finalize_story(
                full, title=title, source_label=source_label,
            )
            parts = summarize._split_exactly_three(full) or parts
        except Exception as rev_exc:  # noqa: BLE001
            log.warning("cohere review skipped: %s", rev_exc)

        # Re-check title after review
        title_s = (title or "").strip()
        bookish_bits = ("نوادر إنجليز", "طرائف إنجليز", "جمعت من مصادر", "مجموعة من مصادر", "Collected from")
        if (
            title_s in generic
            or len(title_s) < 6
            or any(b in title_s for b in bookish_bits)
        ):
            log.warning("generic/bookish title after review %r try=%s key=%s — skipping", title, attempt, key)
            state.mark_story_used(ident, idx, title=f"[skipped-generic-title] {title}"[:80])
            used.add(key)
            continue
        title_fp = state.title_fingerprint(title)
        if title_fp and title_fp in used_titles:
            log.warning(
                "duplicate Arabic title after review fp=%s try=%s/%s key=%s — skipping",
                title_fp[:40], attempt, max_tries, key,
            )
            state.mark_story_used(ident, idx, title=f"[skipped-dup-title] {title}"[:80])
            used.add(key)
            continue

        # Vocalize (full tashkeel) + split into short tweets for .txt and TTS
        try:
            tweets = summarize.vocalize_and_split_tweets(full, title=title)
        except Exception as voc_exc:  # noqa: BLE001
            log.warning("vocalize step failed: %s", voc_exc)
            tweets = summarize.split_into_tweets(full)
        if not tweets:
            tweets = [full]
        full_tweets = summarize.format_tweets_file(
            tweets,
            title=title,
            original_title=src_title,
            collection_title=collection_title,
            author=author,
        )
        tts_parts = summarize.tweets_to_tts_parts(tweets) or parts

        state.mark_story_used(ident, idx, title=title)
        used.add(key)
        if title_fp:
            used_titles.add(title_fp)
        return {
            "title": title,
            "source_label": source_label,
            "collection_title": collection_title,
            "original_title": src_title,
            "author": author,
            "source_language": source_lang,
            "source_language_label": state.language_label(source_lang),
            "identifier": ident,
            "story_index": idx,
            "parts": tts_parts,
            "full_text": full_tweets,
            "tweets": tweets,
            "mp3_path": None,
            "tts_backend": None,
        }

    raise RuntimeError(
        f"تعذّر إيجاد نادرة مقبولة بعد {max_tries} محاولات"
        + (f": {last_err}" if last_err else "")
    )


def prepare_on_demand() -> dict:
    """Immediate /now delivery — advances the language cycle like a scheduled send."""
    return prepare_story(advance_cycle=True)


def _telegram_text_chunks(text: str, limit: int = 4090) -> list[str]:
    """Split only if Telegram forces it; one continuous story, no headers/labels."""
    body = (text or "").strip()
    if not body:
        return []
    if len(body) <= limit:
        return [body]
    chunks: list[str] = []
    rest = body
    while rest:
        if len(rest) <= limit:
            chunks.append(rest)
            break
        window = rest[:limit]
        cut = -1
        for sep in ("\n\n", "۔", ".", "؟", "!", "\n", "،", " "):
            pos = window.rfind(sep)
            if pos >= int(limit * 0.55):
                cut = pos + len(sep)
                break
        if cut < 0:
            cut = limit
        chunks.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip()
    return [c for c in chunks if c]



def _pending_tts_path(token: str) -> Path:
    PENDING_TTS_DIR.mkdir(parents=True, exist_ok=True)
    return PENDING_TTS_DIR / f"{token}.json"


def save_pending_tts(payload: dict) -> str:
    """Persist story parts for on-demand TTS; return short callback token."""
    token = uuid.uuid4().hex[:16]
    data = {
        "token": token,
        "title": payload.get("title") or "",
        "parts": payload.get("parts") or [],
        "full_text": payload.get("full_text") or "",
        "identifier": payload.get("identifier") or "story",
        "story_index": int(payload.get("story_index") or 0),
        "mp3_path": None,
        "created": datetime.now(TZ).isoformat(),
    }
    _pending_tts_path(token).write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )
    return token


def load_pending_tts(token: str) -> dict | None:
    path = _pending_tts_path(token)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def update_pending_tts(token: str, **fields) -> None:
    data = load_pending_tts(token)
    if not data:
        return
    data.update(fields)
    _pending_tts_path(token).write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )


def synthesize_pending_mp3(
    pending: dict,
    engine: str = "gemini",
    gender: str | None = None,
) -> Path:
    """Build full-story MP3 via Fish / Gemini / Edge (cached per engine+gender)."""
    eng = (engine or "gemini").strip().lower()
    if eng.startswith("edge"):
        eng = "edge"
    elif eng.startswith("fish"):
        eng = "fish"
    elif eng.startswith("gem"):
        eng = "gemini"
    else:
        eng = "gemini"
    g = (gender or pending.get("gender") or "male")
    g = "female" if str(g).lower().startswith("f") else "male"
    cache_key = f"mp3_path_{eng}_{g}" if eng == "fish" else f"mp3_path_{eng}"
    existing = pending.get(cache_key)
    if not existing and eng == "gemini":
        existing = pending.get("mp3_path")
    if existing and Path(existing).is_file() and Path(existing).stat().st_size > 1000:
        return Path(existing)
    parts = list(pending.get("parts") or [])
    if not parts:
        body = (pending.get("full_text") or "").strip()
        if not body:
            raise RuntimeError("لا يوجد نص للتحويل")
        parts = [body]
    stamp = datetime.now(TZ).strftime("%Y%m%d-%H%M%S")
    ident = pending.get("identifier") or "story"
    idx = int(pending.get("story_index") or 0)
    base = tts.story_mp3_path(str(ident), idx, stamp)
    suffix = f"_{eng}_{g}" if eng == "fish" else f"_{eng}"
    mp3 = base.with_name(base.stem + suffix + base.suffix)
    tts.parts_to_mp3(
        parts,
        mp3,
        title=pending.get("title") or "",
        author="",
        engine=eng,
        gender=g if eng == "fish" else None,
    )
    token = pending.get("token")
    if token:
        fields = {cache_key: str(mp3)}
        if eng == "gemini":
            fields["mp3_path"] = str(mp3)
        update_pending_tts(token, **fields)
    return mp3


def tts_convert_keyboard(token: str) -> InlineKeyboardMarkup:
    """First step: open engine chooser."""
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton("تحويل النص الى صوت", callback_data=f"{TTS_MENU_PREFIX}{token}")]]
    )


def tts_engine_keyboard(token: str) -> InlineKeyboardMarkup:
    """Second step: Gemini or free Microsoft Edge."""
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("جيميناي", callback_data=f"{TTS_ENGINE_PREFIX}gemini:{token}"),
                InlineKeyboardButton("ايديج", callback_data=f"{TTS_ENGINE_PREFIX}edge:{token}"),
            ]
        ]
    )





def _safe_filename(title: str, fallback: str = "nadira") -> str:
    raw = (title or fallback).strip() or fallback
    cleaned = re.sub(r"[^\w\u0600-\u06FF\s\-]+", "", raw, flags=re.UNICODE)
    cleaned = re.sub(r"\s+", "_", cleaned).strip("_")
    return (cleaned[:60] or fallback)


def write_story_txt(body: str, title: str, identifier: str = "story", story_index: int = 0) -> Path:
    """Write the full story into one UTF-8 .txt file under data/stories/."""
    root = Path(__file__).resolve().parent / "data" / "stories"
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(TZ).strftime("%Y%m%d-%H%M%S")
    ident = re.sub(r"[^\w\-]+", "_", str(identifier or "story"))[:40]
    name = f"{stamp}_{ident}_s{int(story_index):04d}_{_safe_filename(title)}.txt"
    path = root / name
    path.write_text(((body or "").strip() + "\n"), encoding="utf-8")
    return path



def _tts_clean_parts(parts: list[str], body: str) -> list[str]:
    """Drop tweet numbers/labels so TTS does not speak '1)' aloud."""
    cleaned: list[str] = []
    for p in parts or []:
        s = re.sub(r"(?m)^\s*\d+[\)\.\-\u060c]\s*", "", p or "")
        s = re.sub(r"(?m)^---+\s*", "", s)
        s = s.strip()
        if s:
            cleaned.append(s)
    if cleaned:
        return cleaned
    body = re.sub(r"(?m)^\s*\d+[\)\.\-\u060c]\s*", "", body or "")
    return [body.strip()] if body.strip() else []


async def deliver(bot, chat_id: int, payload: dict) -> None:
    """Send full story as one .txt, then automatic Edge TTS audio."""
    title = (payload.get("title") or "نادرة").strip()
    body = (payload.get("full_text") or "").strip()
    if not body:
        parts = payload.get("parts") or []
        body = "\n\n".join(p.strip() for p in parts if p and str(p).strip()).strip()
    if title and body and not body.lstrip().startswith(title):
        body = f"{title}\n\n{body}"

    txt_path = write_story_txt(
        body,
        title=title,
        identifier=str(payload.get("identifier") or "story"),
        story_index=int(payload.get("story_index") or 0),
    )
    fname = f"{_safe_filename(title)}.txt"
    book = (payload.get("collection_title") or "").strip()
    orig = (payload.get("original_title") or "").strip()
    caption_bits = [title or "نادرة"]
    if orig and orig != title:
        caption_bits.append(f"الأصل: {orig}")
    if book:
        caption_bits.append(f"الكتاب: {book}")
    caption = "\n".join(caption_bits)[:1000]
    with open(txt_path, "rb") as fh:
        await bot.send_document(
            chat_id=chat_id,
            document=InputFile(fh, filename=fname),
            caption=caption,
        )

    # Auto audio: Edge TTS only
    parts = _tts_clean_parts(list(payload.get("parts") or []), body)
    pending = {
        "token": None,
        "title": title,
        "parts": parts,
        "full_text": "\n\n".join(parts) if parts else body,
        "identifier": payload.get("identifier") or "story",
        "story_index": int(payload.get("story_index") or 0),
    }
    try:
        await bot.send_message(chat_id=chat_id, text="جارٍ تجهيز الصوت…")
    except Exception:
        pass

    mp3 = None
    backend_label = "Edge TTS"
    try:
        mp3 = await asyncio.wait_for(
            asyncio.to_thread(synthesize_pending_mp3, pending, "edge"),
            timeout=180,
        )
    except Exception:
        log.exception("Edge TTS failed chat_id=%s", chat_id)
        await bot.send_message(
            chat_id=chat_id,
            text="تعذّر تجهيز الصوت (ايديج). يمكنك استخدام /gemini لاحقاً.",
        )
        return

    audio_name = os.path.basename(str(mp3))
    if not audio_name.lower().endswith(".mp3"):
        audio_name = "story.mp3"
    size = Path(mp3).stat().st_size if mp3 else 0
    log.info(
        "sending audio chat_id=%s backend=%s bytes=%s path=%s",
        chat_id, backend_label, size, mp3,
    )
    caption = f"{title} — {backend_label}" if title else backend_label
    try:
        if size > 45 * 1024 * 1024:
            with open(mp3, "rb") as fh:
                await bot.send_document(
                    chat_id=chat_id,
                    document=InputFile(fh, filename=audio_name),
                    caption=caption[:1000],
                )
        else:
            with open(mp3, "rb") as fh:
                await bot.send_audio(
                    chat_id=chat_id,
                    audio=InputFile(fh, filename=audio_name),
                    caption=caption[:1000],
                    title=(title or "نادرة")[:64],
                    performer=(backend_label[:64] or "راوي"),
                )
    except Exception:
        log.exception("send_audio failed; retry as document chat_id=%s", chat_id)
        try:
            with open(mp3, "rb") as fh:
                await bot.send_document(
                    chat_id=chat_id,
                    document=InputFile(fh, filename=audio_name),
                    caption=caption[:1000],
                )
        except Exception:
            log.exception("send_document audio also failed chat_id=%s", chat_id)
            await bot.send_message(
                chat_id=chat_id,
                text="تم تجهيز الصوت لكن تعذّر رفعه إلى تيليجرام. حاول /now لاحقاً.",
            )


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat or not update.message:
        return
    added = state.subscribe(update.effective_chat.id)
    await update.message.reply_text(WELCOME_AR if added else ALREADY_AR)


async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat or not update.message:
        return
    removed = state.unsubscribe(update.effective_chat.id)
    await update.message.reply_text(STOPPED_AR if removed else NOT_SUB_AR)


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    ymd = state.today_ymd()
    nsub = len(state.subscribers())
    last = state.last_story()
    next_lang = state.peek_source_language()
    next_label = state.language_label(next_lang)
    cycle = " ← ".join(state.language_label(c) for c in state.LANGUAGE_CYCLE)
    if last:
        last_line = (
            f"آخر حكاية: {last.get('title') or last.get('identifier')} "
            f"(#{last.get('story_index', '?')})"
        )
    else:
        last_line = "آخر حكاية: لا شيء بعد"
    body = (
        f"تاريخ اليوم ({ymd}) بتوقيت الرياض\n"
        f"{last_line}\n"
        f"اللغة التالية للمصدر: {next_label} ({next_lang})\n"
        f"دورة اللغات: {cycle}\n"
        f"المشتركون: {nsub}\n"
        "الجدول: 08:00، 12:00، 16:00، 20:00 — حكايات من «أساطير شعبية من قلب جزيرة العرب» للجهيمان + صوت MP3.\n"
        "التسليم: ملف نص مشكّل ومقسوم تغريدات قصيرة (بدون ترقيم) + صوت Edge."
    )
    await update.message.reply_text(body)


async def cmd_now(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat or not update.message:
        return
    state.subscribe(update.effective_chat.id)
    await update.message.reply_text(
        "اضغط الزر لطلب حكاية فورية:",
        reply_markup=NOW_KEYBOARD,
    )


async def cmd_gemini(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Menu: ask user to send text for Gemini TTS."""
    if not update.effective_chat or not update.message:
        return
    chat_id = update.effective_chat.id
    state.subscribe(chat_id)
    _awaiting_gemini.add(chat_id)
    await update.message.reply_text(GEMINI_ASK_AR)


async def cmd_cancel_gemini(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat or not update.message:
        return
    chat_id = update.effective_chat.id
    _awaiting_gemini.discard(chat_id)
    await update.message.reply_text(GEMINI_CANCEL_AR)


async def on_gemini_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Receive pasted text or .txt while waiting for Gemini conversion."""
    if not update.effective_chat or not update.message:
        return
    chat_id = update.effective_chat.id
    if chat_id not in _awaiting_gemini:
        return
    if chat_id in _busy_chats:
        await update.message.reply_text(BUSY_AR)
        return

    text = ""
    title = "نص جيميناي"
    msg = update.message
    if msg.document:
        doc = msg.document
        name = (doc.file_name or "").lower()
        mime = (doc.mime_type or "").lower()
        if not (name.endswith(".txt") or mime.startswith("text/")):
            await msg.reply_text("أرسل نصاً عادياً أو ملف .txt فقط.")
            return
        tg_file = await doc.get_file()
        data = await tg_file.download_as_bytearray()
        try:
            text = bytes(data).decode("utf-8")
        except UnicodeDecodeError:
            text = bytes(data).decode("utf-8", errors="replace")
        if doc.file_name:
            title = Path(doc.file_name).stem[:60] or title
    else:
        text = (msg.text or msg.caption or "").strip()

    text = (text or "").strip()
    if len(text) < 20:
        await msg.reply_text("النص قصير جداً. أرسل مقطعاً أطول، أو /cancel لإلغاء الانتظار.")
        return

    _awaiting_gemini.discard(chat_id)
    _busy_chats.add(chat_id)
    await msg.reply_text(CONVERTING_AR + "\nالمحرك: جيميناي")
    try:
        # Split long text into ~3 speech segments for Gemini TTS limits
        from summarize import sanitize_for_speech

        clean = sanitize_for_speech(text)
        # rough thirds by words
        words = clean.split()
        if len(words) <= 400:
            parts = [clean]
        else:
            n = len(words)
            cuts = [0, n // 3, (2 * n) // 3, n]
            parts = [" ".join(words[cuts[i]:cuts[i + 1]]).strip() for i in range(3)]
            parts = [p for p in parts if p]
        pending = {
            "token": None,
            "title": title,
            "parts": parts,
            "full_text": clean,
            "identifier": f"user{chat_id}",
            "story_index": 0,
        }
        mp3 = await asyncio.to_thread(synthesize_pending_mp3, pending, "gemini")
        fname = os.path.basename(str(mp3))
        if not fname.lower().endswith(".mp3"):
            fname = "gemini.mp3"
        with open(mp3, "rb") as fh:
            await context.bot.send_audio(
                chat_id=chat_id,
                audio=InputFile(fh, filename=fname),
                caption=title,
                title=title[:64],
                performer="راوي الحكايات",
            )
        await msg.reply_text(TTS_DONE_AR + " (جيميناي)")
    except Exception as exc:  # noqa: BLE001
        log.exception("gemini on-demand TTS failed chat_id=%s", chat_id)
        await msg.reply_text(
            TTS_FAIL_AR if not str(exc) else (TTS_FAIL_AR + "\n" + str(exc))
        )
        _awaiting_gemini.add(chat_id)
    finally:
        _busy_chats.discard(chat_id)


async def on_now_folk(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query or not query.data or query.data != "now:folk":
        return
    await query.answer()
    chat = update.effective_chat
    if not chat:
        return
    chat_id = chat.id
    if chat_id in _busy_chats:
        if query.message:
            await query.message.reply_text(BUSY_AR)
        return
    try:
        await query.edit_message_text(PREPARING_AR)
    except Exception:
        await context.bot.send_message(chat_id=chat_id, text=PREPARING_AR)
    _busy_chats.add(chat_id)
    try:
        payload = await asyncio.to_thread(prepare_on_demand)
        await deliver(context.bot, chat_id, payload)
    except Exception as exc:  # noqa: BLE001
        log.exception("on-demand folk tale failed")
        await context.bot.send_message(
            chat_id=chat_id,
            text=(
                str(exc)
                if str(exc)
                else f"تعذّر جلب الحكاية: {exc.__class__.__name__}. حاول مرة أخرى عبر /now."
            ),
        )
    finally:
        _busy_chats.discard(chat_id)



async def on_tts_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show Gemini / Edge choice after user presses convert."""
    query = update.callback_query
    if not query or not query.data or not query.data.startswith(TTS_MENU_PREFIX):
        return
    token = query.data[len(TTS_MENU_PREFIX):].strip()
    await query.answer()
    pending = load_pending_tts(token)
    if not pending:
        if query.message:
            await query.message.reply_text("انتهت صلاحية زر التحويل. اطلب قصة جديدة عبر /now.")
        return
    try:
        await query.edit_message_text(
            TTS_CHOOSE_AR,
            reply_markup=tts_engine_keyboard(token),
        )
    except Exception:
        chat = update.effective_chat
        if chat:
            await context.bot.send_message(
                chat_id=chat.id,
                text=TTS_CHOOSE_AR,
                reply_markup=tts_engine_keyboard(token),
            )


async def on_tts_engine(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Run full-text TTS with the chosen engine and send one complete audio."""
    query = update.callback_query
    if not query or not query.data or not query.data.startswith(TTS_ENGINE_PREFIX):
        return
    rest = query.data[len(TTS_ENGINE_PREFIX):].strip()
    if ":" not in rest:
        await query.answer()
        return
    engine, _, token = rest.partition(":")
    engine = engine.strip().lower()
    token = token.strip()
    if engine not in {"gemini", "edge"} or not token:
        await query.answer()
        return
    await query.answer()
    chat = update.effective_chat
    if not chat:
        return
    chat_id = chat.id
    pending = load_pending_tts(token)
    if not pending:
        if query.message:
            await query.message.reply_text("انتهت صلاحية زر التحويل. اطلب قصة جديدة عبر /now.")
        return
    if chat_id in _busy_chats:
        if query.message:
            await query.message.reply_text(BUSY_AR)
        return
    label = "جيميناي" if engine == "gemini" else "ايديج (مايكروسوفت)"
    _busy_chats.add(chat_id)
    try:
        converting_msg = CONVERTING_AR + "\n" + "المحرك: " + label
        try:
            await query.edit_message_text(converting_msg)
        except Exception:
            await context.bot.send_message(chat_id=chat_id, text=converting_msg)
        mp3 = await asyncio.to_thread(synthesize_pending_mp3, pending, engine)
        title = (pending.get("title") or "قصة").strip()
        fname = os.path.basename(str(mp3))
        if not fname.lower().endswith(".mp3"):
            fname = "story.mp3"
        caption = (title + " — " + label) if title else label
        with open(mp3, "rb") as fh:
            await context.bot.send_audio(
                chat_id=chat_id,
                audio=InputFile(fh, filename=fname),
                caption=caption,
                title=title[:64] or "قصة",
                performer=("راوي الحكايات" if engine == "gemini" else "Edge TTS"),
            )
        try:
            await query.edit_message_text(TTS_DONE_AR + " (" + label + ")")
        except Exception:
            pass
    except Exception as exc:  # noqa: BLE001
        log.exception("tts convert failed engine=%s token=%s", engine, token)
        fail_text = TTS_FAIL_AR if not str(exc) else (TTS_FAIL_AR + "\n" + str(exc))
        await context.bot.send_message(chat_id=chat_id, text=fail_text)
        try:
            await query.edit_message_text(
                TTS_CHOOSE_AR,
                reply_markup=tts_engine_keyboard(token),
            )
        except Exception:
            pass
    finally:
        _busy_chats.discard(chat_id)


async def scheduled_job(application: Application) -> None:
    log.info("scheduled folk-tale fire")
    try:
        payload = await asyncio.to_thread(prepare_story, True)
    except Exception:
        log.exception("scheduled prepare_story failed")
        return
    chats = state.subscribers()
    if not chats:
        log.info(
            "no subscribers; story prepared title=%s lang=%s",
            payload.get("title"),
            payload.get("source_language"),
        )
        return
    for chat_id in chats:
        try:
            await deliver(application.bot, chat_id, payload)
        except Exception:
            log.exception("send failed chat_id=%s", chat_id)


def _bootstrap_subscribers() -> None:
    """Ensure chat IDs from BOOTSTRAP_SUBSCRIBERS are in state.json (Railway volume)."""
    raw = (os.getenv("BOOTSTRAP_SUBSCRIBERS") or "").strip()
    if not raw:
        return
    added = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            cid = int(part)
        except ValueError:
            log.warning("BOOTSTRAP_SUBSCRIBERS skip invalid id=%r", part)
            continue
        if state.subscribe(cid):
            added.append(cid)
    if added:
        log.info("bootstrap subscribed chat_ids=%s", added)
    else:
        log.info("bootstrap subscribers already present")


async def post_init(application: Application) -> None:
    _bootstrap_subscribers()
    try:
        await application.bot.set_my_commands(
            [
                BotCommand("now", "حكاية الآن"),
                BotCommand("gemini", "تحويل النص الى صوت جميناي"),
                BotCommand("status", "الحالة"),
                BotCommand("cancel", "إلغاء انتظار نص جميناي"),
                BotCommand("stop", "إلغاء الاشتراك"),
                BotCommand("start", "اشتراك"),
            ]
        )
    except Exception:
        log.exception("set_my_commands failed")
    scheduler = AsyncIOScheduler(timezone=TZ)
    for hour, job_id in (
        (8, "folk-0800"),
        (12, "folk-1200"),
        (16, "folk-1600"),
        (20, "folk-2000"),
    ):
        scheduler.add_job(
            scheduled_job,
            CronTrigger(hour=hour, minute=0, timezone=TZ),
            args=[application],
            id=job_id,
            replace_existing=True,
            misfire_grace_time=3600,
        )
    scheduler.start()
    application.bot_data["scheduler"] = scheduler
    jobs = scheduler.get_jobs()
    log.info(
        "scheduler started Asia/Riyadh folk tales 08/12/16/20 — jobs=%s",
        [j.id for j in jobs],
    )


async def post_shutdown(application: Application) -> None:
    scheduler = application.bot_data.get("scheduler")
    if scheduler:
        scheduler.shutdown(wait=False)


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    token = _token()
    me = validate_token(token)
    log.info("token ok bot=@%s id=%s", me.get("username") or "", me.get("id"))

    application = (
        Application.builder()
        .token(token)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )
    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("stop", cmd_stop))
    application.add_handler(CommandHandler("status", cmd_status))
    application.add_handler(CommandHandler("now", cmd_now))
    application.add_handler(CommandHandler("gemini", cmd_gemini))
    application.add_handler(CommandHandler("cancel", cmd_cancel_gemini))
    application.add_handler(CallbackQueryHandler(on_now_folk, pattern=r"^now:folk$"))
    application.add_handler(CallbackQueryHandler(on_tts_menu, pattern=r"^ttsmenu:[0-9a-f]{8,32}$"))
    application.add_handler(CallbackQueryHandler(on_tts_engine, pattern=r"^ttseng:(gemini|edge):[0-9a-f]{8,32}$"))
    application.add_handler(
        MessageHandler(
            (filters.TEXT | filters.Document.ALL) & ~filters.COMMAND,
            on_gemini_text,
        )
    )

    log.info("starting polling")
    application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()

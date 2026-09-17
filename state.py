"""JSON persistence for subscribers and folk-tale delivery state."""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
STATE_PATH = DATA_DIR / "state.json"
TZ = ZoneInfo("Asia/Riyadh")

LANGUAGE_CYCLE = ("eng",)  # non-Arabic history sources
LANGUAGE_LABELS = {
    "ara": "عربي",
    "eng": "إنجليزي",
    "rus": "روسي",
    "urd": "أردو",
}

_lock = threading.RLock()

_DEFAULT: dict[str, Any] = {
    "subscribers": [],
    "used_stories": [],
    "used_titles": [],  # normalized Arabic title fingerprints (anti-repeat)
    "last_story": None,
    "language_cycle_index": 0,
    "fish_voice_gender": "male",  # next Fish narrator: male|female
    # legacy keys kept for gentle migration / status
    "used_arabic": [],
    "used_english": [],
    "arabic": None,
    "english": None,
}


def _ensure_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def today_ymd() -> str:
    return datetime.now(TZ).date().isoformat()


def story_key(identifier: str, story_index: int) -> str:
    return f"{identifier}#{int(story_index)}"

def title_fingerprint(title: str) -> str:
    """Normalize Arabic/Latin title for anti-repeat matching."""
    import re as _re
    t = (title or "").strip().lower()
    # strip tashkeel and punctuation/whitespace
    t = _re.sub(r"[\u064B-\u065F\u0670\u0640]", "", t)
    t = _re.sub(r"[^\w\u0600-\u06FF]+", "", t, flags=_re.UNICODE)
    return t[:80]


def used_title_fps() -> set[str]:
    with _lock:
        state = load()
        return {str(x) for x in (state.get("used_titles") or []) if x}


def mark_title_used(title: str) -> None:
    fp = title_fingerprint(title)
    if not fp:
        return
    with _lock:
        state = load()
        used = list(state.get("used_titles") or [])
        if fp not in used:
            used.append(fp)
        state["used_titles"] = used
        save(state)



def parse_story_key(key: str) -> tuple[str, int] | None:
    s = str(key or "")
    if "#" not in s:
        return None
    ident, _, idx = s.rpartition("#")
    try:
        return ident, int(idx)
    except ValueError:
        return None


def load() -> dict[str, Any]:
    with _lock:
        _ensure_dir()
        if not STATE_PATH.exists():
            return json.loads(json.dumps(_DEFAULT))
        try:
            raw = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return json.loads(json.dumps(_DEFAULT))
        state = json.loads(json.dumps(_DEFAULT))
        if isinstance(raw, dict):
            state.update(raw)

        # migrate old used_identifiers → used_arabic (legacy)
        old_used = state.get("used_identifiers") or []
        if old_used:
            ua = list(state.get("used_arabic") or [])
            for x in old_used:
                s = str(x)
                if s and s not in ua:
                    ua.append(s)
            state["used_arabic"] = ua

        # migrate book identifiers into used_stories as identifier#0 markers
        # only if used_stories is empty (first folk-tale boot)
        used_stories = [str(x) for x in (state.get("used_stories") or []) if x]
        if not used_stories:
            seeded: list[str] = []
            for key in ("used_arabic", "used_english"):
                for ident in state.get(key) or []:
                    sk = story_key(str(ident), 0)
                    # do not auto-seed whole books as used stories — keep empty
                    # so folk tales can reuse Alf-Layla / Kalila collections.
                    _ = sk
            state["used_stories"] = seeded
        else:
            state["used_stories"] = used_stories

        subs = state.get("subscribers") or []
        cleaned: list[int] = []
        for item in subs:
            try:
                cleaned.append(int(item))
            except (TypeError, ValueError):
                continue
        state["subscribers"] = cleaned
        for key in ("used_arabic", "used_english"):
            state[key] = [str(x) for x in (state.get(key) or []) if x]
        if state.get("last_story") is not None and not isinstance(state.get("last_story"), dict):
            state["last_story"] = None
        try:
            state["language_cycle_index"] = int(state.get("language_cycle_index") or 0) % len(LANGUAGE_CYCLE)
        except (TypeError, ValueError):
            state["language_cycle_index"] = 0
        return state


def save(state: dict[str, Any]) -> None:
    with _lock:
        _ensure_dir()
        tmp = STATE_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(STATE_PATH)


def subscribe(chat_id: int) -> bool:
    with _lock:
        state = load()
        cid = int(chat_id)
        if cid in state["subscribers"]:
            return False
        state["subscribers"].append(cid)
        save(state)
        return True


def unsubscribe(chat_id: int) -> bool:
    with _lock:
        state = load()
        cid = int(chat_id)
        if cid not in state["subscribers"]:
            return False
        state["subscribers"] = [s for s in state["subscribers"] if s != cid]
        save(state)
        return True


def subscribers() -> list[int]:
    return list(load()["subscribers"])


def used_stories() -> set[str]:
    return set(load().get("used_stories") or [])


def used_for(language: str) -> set[str]:
    """Legacy helper: book identifiers used (kept for old call sites)."""
    state = load()
    key = "used_arabic" if language == "ara" else "used_english"
    return set(state.get(key) or [])


def last_story() -> dict[str, Any] | None:
    rec = load().get("last_story")
    return rec if isinstance(rec, dict) else None


def mark_story_used(
    identifier: str,
    story_index: int,
    title: str | None = None,
) -> None:
    with _lock:
        state = load()
        key = story_key(identifier, story_index)
        used = list(state.get("used_stories") or [])
        if key not in used:
            used.append(key)
        state["used_stories"] = used
        # Anti-repeat formula part 2: normalized Arabic title fingerprint
        fp = title_fingerprint(title or "")
        if fp:
            ut = list(state.get("used_titles") or [])
            if fp not in ut:
                ut.append(fp)
            state["used_titles"] = ut
        state["last_story"] = {
            "date": today_ymd(),
            "identifier": str(identifier),
            "story_index": int(story_index),
            "title": title or "",
            "key": key,
            "title_fp": fp,
        }
        # also remember collection identifier for legacy status
        ua = list(state.get("used_arabic") or [])
        if str(identifier) not in ua:
            ua.append(str(identifier))
        state["used_arabic"] = ua
        save(state)


def unmark_story_used(identifier: str, story_index: int) -> None:
    with _lock:
        state = load()
        key = story_key(identifier, story_index)
        state["used_stories"] = [x for x in (state.get("used_stories") or []) if str(x) != key]
        save(state)


# --- legacy book-slot helpers (status / soft compat) ---

def slot(language: str) -> dict[str, Any] | None:
    state = load()
    rec = state.get("arabic" if language == "ara" else "english")
    return rec if isinstance(rec, dict) else None


def todays_slot(language: str) -> dict[str, Any] | None:
    rec = slot(language)
    if rec and rec.get("date") == today_ymd():
        return rec
    return None


def set_daily_book(language: str, identifier: str, title: str) -> dict[str, Any]:
    with _lock:
        state = load()
        rec = {
            "date": today_ymd(),
            "identifier": str(identifier),
            "title": title,
        }
        slot_key = "arabic" if language == "ara" else "english"
        state[slot_key] = rec
        save(state)
        return rec


def mark_used(language: str, identifier: str, title: str | None = None) -> None:
    with _lock:
        state = load()
        used_key = "used_arabic" if language == "ara" else "used_english"
        used = list(state.get(used_key) or [])
        ident = str(identifier)
        if ident not in used:
            used.append(ident)
        state[used_key] = used
        save(state)


def unmark_used(language: str, identifier: str) -> None:
    with _lock:
        state = load()
        used_key = "used_arabic" if language == "ara" else "used_english"
        ident = str(identifier)
        state[used_key] = [x for x in (state.get(used_key) or []) if str(x) != ident]
        save(state)


def language_label(code: str) -> str:
    return LANGUAGE_LABELS.get(code, code)


def get_cycle_index() -> int:
    return int(load().get("language_cycle_index") or 0) % len(LANGUAGE_CYCLE)


def peek_source_language() -> str:
    """Language for the next delivery (does not advance)."""
    return LANGUAGE_CYCLE[get_cycle_index()]


def advance_source_language() -> str:
    """Return current language then advance index for the following send."""
    with _lock:
        state = load()
        idx = int(state.get("language_cycle_index") or 0) % len(LANGUAGE_CYCLE)
        lang = LANGUAGE_CYCLE[idx]
        state["language_cycle_index"] = (idx + 1) % len(LANGUAGE_CYCLE)
        save(state)
        return lang


def set_cycle_index(index: int) -> None:
    with _lock:
        state = load()
        state["language_cycle_index"] = int(index) % len(LANGUAGE_CYCLE)
        save(state)


def next_fish_voice_gender() -> str:
    """Return next Fish narrator gender and flip for the following story."""
    with _lock:
        state = load()
        cur = str(state.get("fish_voice_gender") or "male").lower()
        if cur not in {"male", "female"}:
            cur = "male"
        nxt = "female" if cur == "male" else "male"
        state["fish_voice_gender"] = nxt
        save(state)
        return cur


def peek_fish_voice_gender() -> str:
    g = str(load().get("fish_voice_gender") or "male").lower()
    return g if g in {"male", "female"} else "male"

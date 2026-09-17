"""Dry-run: pick folk story slice → Gemini Arabic+tashkeel → Gemini TTS MP3."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

import books
import state
import summarize
import tts

SAMPLE_MP3 = ROOT / "data" / "audio" / "folk_sample.mp3"
SAMPLE_TXT = ROOT / "data" / "audio" / "folk_sample.txt"


def main() -> int:
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    lang = state.peek_source_language()
    print(f"== dry-run folk tale (peek source lang={lang}) ==")
    used = state.used_stories()
    story = books.pick_next_story(used=used, language=lang)
    print(
        f"picked {story['identifier']}#{story['story_index']} "
        f"lang={story.get('language')} words={len(story['text'].split())}"
    )
    print(f"title hint: {str(story.get('title') or '')[:80]}")

    print("\n== Gemini folk story (Arabic + tashkeel) ==")
    title, source_label, parts, full = summarize.build_folk_story(
        story["text"],
        story.get("title") or story.get("collection_title") or "",
        source_lang=story.get("language") or lang,
        author=story.get("author") or "",
    )
    print(f"title={title}")
    print(f"source_label={source_label}")
    print(f"parts={len(parts)} full_words={len(full.split())} full_chars={len(full)}")
    SAMPLE_TXT.parent.mkdir(parents=True, exist_ok=True)
    SAMPLE_TXT.write_text(
        f"العنوان: {title}\nالمصدر: {source_label}\n\n" + "\n\n---\n\n".join(parts),
        encoding="utf-8",
    )
    print(f"wrote {SAMPLE_TXT}")

    print("\n== TTS (Gemini primary, edge fallback) ==")
    # For dry-run keep TTS shorter if story is huge — still exercise pipeline.
    speak = tts.speech_text(parts, title=title)
    # Cap dry-run audio to first ~2200 chars to save quota/time if needed,
    # but prefer full story when under 6000 chars.
    if len(speak) > 6000:
        speak = speak[:5500]
        cut = speak.rfind(" ")
        if cut > 4000:
            speak = speak[:cut]
        print(f"dry-run TTS truncated to {len(speak)} chars")
    tts.text_to_mp3(speak, SAMPLE_MP3)
    backend = getattr(tts.text_to_mp3, "_last_backend", "unknown")
    size = SAMPLE_MP3.stat().st_size if SAMPLE_MP3.exists() else 0
    print(f"wrote {SAMPLE_MP3} ({size} bytes) backend={backend}")
    if size < 500:
        print("ERROR: MP3 too small", file=sys.stderr)
        return 1

    print("\nDRY-RUN OK")
    print(f"TITLE={title}")
    print(f"IDENTIFIER={story['identifier']}")
    print(f"STORY_INDEX={story['story_index']}")
    print(f"SOURCE_LANG={lang}")
    print(f"PARTS={len(parts)}")
    print(f"MP3={SAMPLE_MP3}")
    print(f"MP3_BYTES={size}")
    print(f"TTS_BACKEND={backend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

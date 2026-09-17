"""Smoke-test Gemini summarizer. Never prints API keys or Telegram tokens."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

import summarize


FALLBACK_SAMPLE = (
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, "
    "must be in want of a wife. However little known the feelings or views of such a man may be "
    "on his first entering a neighbourhood, this truth is so well fixed in the minds of the "
    "surrounding families, that he is considered as the rightful property of some one or other "
    "of their daughters.\n\n"
    "Mr. Bennet was among the earliest of those who waited on Mr. Bingley. He had always intended "
    "to visit him, though to the last always assuring his wife that he should not go; and till "
    "the evening after the visit was paid, she had no knowledge of it. This was her favourite "
    "son-in-law elect, and she spoke of him with rapture."
)


def _arabic_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha() or "\u0600" <= c <= "\u06FF"]
    if not letters:
        return 0.0
    ar = sum(1 for c in letters if "\u0600" <= c <= "\u06FF")
    return ar / len(letters)


def _load_excerpt() -> tuple[str, str, str]:
    books_dir = ROOT / "data" / "books"
    kalila = books_dir / "kalila-wa-dimna-arabic-text.txt"
    treasure = books_dir / "treasureisland00stevrich.txt"
    if kalila.exists() and kalila.stat().st_size > 2000:
        text = kalila.read_text(encoding="utf-8", errors="replace")[:8000]
        return text, "كليلة ودمنة", "ara"
    if treasure.exists() and treasure.stat().st_size > 2000:
        text = treasure.read_text(encoding="utf-8", errors="replace")[:8000]
        return text, "Treasure Island", "eng"
    return FALLBACK_SAMPLE, "Pride and Prejudice", "eng"


def main() -> int:
    if not summarize._gemini_api_key():
        print("fail")
        print("status: 0")
        print("error: GEMINI_API_KEY missing")
        return 1

    excerpt, title, language = _load_excerpt()
    prompt = summarize._gemini_prompt(excerpt, title, language, kind="full")
    try:
        out, _reason = summarize._gemini_generate(prompt)
    except summarize.GeminiAuthError as exc:
        print("fail")
        print("model:", summarize._gemini_model or "")
        print("status:", exc.status)
        print("error:", summarize._redact_secrets(exc.snippet)[:300])
        return 1
    except summarize.GeminiError as exc:
        print("fail")
        print("model:", summarize._gemini_model or "")
        print("status:", exc.status)
        print("error:", summarize._redact_secrets(exc.snippet)[:300])
        return 1

    out = summarize._redact_secrets(out or "")
    ratio = _arabic_ratio(out)
    preview = out[:120].replace("\n", " ")
    print("ok" if out and ratio >= 0.35 else "fail")
    print("model:", summarize._gemini_model or "")
    print("output_chars:", len(out))
    print("arabic_ratio:", f"{ratio:.3f}")
    print("arabic:", "yes" if ratio >= 0.35 else "no")
    print("preview:", preview)
    return 0 if out and ratio >= 0.35 else 1


if __name__ == "__main__":
    raise SystemExit(main())

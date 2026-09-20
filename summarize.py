"""Gemini-first Arabic summaries, with extractive+translate as last-resort fallback."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import Counter

import httpx

log = logging.getLogger("kitab.summarize")

SENT_SPLIT = re.compile(r"(?<=[.!?؟۔;؛])\s+|\n{2,}")
WORD_RE = re.compile(r"[\w\u0600-\u06FF]+", re.UNICODE)
ARABIC_CHAR = re.compile(r"[\u0600-\u06FF]")

STOPWORDS = {
    # Arabic
    "في", "من", "على", "إلى", "الى", "أن", "إن", "ان", "هذا", "هذه", "ذلك", "تلك",
    "الذي", "التي", "الذين", "كان", "كانت", "يكون", "هو", "هي", "هم", "هن",
    "ما", "لا", "لم", "لن", "قد", "عن", "مع", "أو", "او", "ثم", "بعد", "قبل",
    "كل", "بعض", "أي", "اى", "ولا", "وما", "فقد", "وقد", "كما", "بين", "حتى",
    "إذا", "اذا", "لكن", "ولكن", "هناك", "هنا", "حيث", "عند", "عليه", "فيها",
    "منها", "وهو", "وهي", "ولم", "فلا", "بل", "أيضا", "ايضا", "ذلكم",
    "يا", "ألا", "الا", "أما", "اما", "ولو", "وإن", "وان", "ليس", "ليست",
    "كنت", "كنا", "قال", "قالت", "يقول", "قيل", "لقد", "وقد",
    # English
    "the", "a", "an", "and", "or", "but", "if", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "are", "was", "were", "be", "been",
    "it", "its", "this", "that", "these", "those", "he", "she", "they", "them",
    "his", "her", "their", "not", "no", "so", "than", "then", "there", "here",
    "which", "who", "whom", "into", "over", "after", "before", "have", "has",
    "had", "will", "would", "could", "should", "may", "might", "do", "did",
    "does", "we", "you", "i", "my", "our", "your",
}

PART_MIN = 700
PART_MAX = 1200
TARGET_SUMMARY_CHARS = 4200
MIN_PARTS = 3
MAX_PARTS = 6


_ARABIC_RANGES = (
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
)
_LATIN_PAREN_RE = re.compile(r"[\(\[\{【]\s*[A-Za-z][^\)\]\}】]{0,80}[\)\]\}】]")
_TRAILING_LATIN_RE = re.compile(r"\s*[-–—:]\s*[A-Za-z][A-Za-z0-9 ,.'’\-:]{2,}$")


def _is_arabic_char(ch: str) -> bool:
    o = ord(ch)
    return any(a <= o <= b for a, b in _ARABIC_RANGES)


def _letter_counts(text: str) -> tuple[int, int]:
    ar = 0
    lat = 0
    for c in text or "":
        if _is_arabic_char(c):
            ar += 1
        elif c.isalpha():
            lat += 1
    return ar, lat


def strip_latin_title_extras(text: str) -> str:
    """Drop Latin parentheticals / trailing English so Arabic titles are not rejected."""
    t = text or ""
    t = _LATIN_PAREN_RE.sub("", t)
    t = _TRAILING_LATIN_RE.sub("", t)
    return t.strip(" \t-–—:،,")


def title_reject_reason(text: str) -> str:
    """Human-readable why a title failed the Arabic acceptance check (no secrets)."""
    if not (text or "").strip():
        return "empty"
    ar, lat = _letter_counts(text)
    total = ar + lat
    ratio = (ar / total) if total else 0.0
    if ar == 0:
        return f"no_arabic_letters latin={lat} chars={len(text)}"
    if lat > ar or ratio < 0.5:
        return f"latin_heavy arabic_ratio={ratio:.2f} ar={ar} lat={lat} chars={len(text)}"
    return "ok"


def is_arabic(text: str) -> bool:
    letters = [c for c in text if c.isalpha() or _is_arabic_char(c)]
    if len(letters) < 20:
        return False
    ar = sum(1 for c in letters if _is_arabic_char(c))
    return (ar / len(letters)) >= 0.35


def title_is_arabic(text: str) -> bool:
    """True when a (possibly short) title is already predominantly Arabic script.

    Counts Arabic presentation forms as Arabic. Latin parentheticals should be
    stripped with strip_latin_title_extras before this check when parsing Gemini.
    """
    ar, lat = _letter_counts(text)
    total = ar + lat
    if ar == 0 or total == 0:
        return False
    return ar >= lat and (ar / total) >= 0.5


def title_is_latin_heavy(text: str) -> bool:
    """True when a title has little/no Arabic and still needs translation."""
    cleaned = strip_latin_title_extras(text)
    return not title_is_arabic(cleaned or text)


def sanitize_for_speech(text: str) -> str:
    """Strip asterisks and markdown TTS would voice. Keep Arabic punctuation."""
    if not text:
        return ""
    s = text.replace("\r\n", "\n").replace("\r", "\n")
    # Leading markdown headers: drop the hashes, keep the words.
    s = re.sub(r"^#{1,6}\s+", "", s, flags=re.M)
    s = s.replace("`", "")
    s = re.sub(r"__([^_\n]+)__", r"\1", s)
    s = re.sub(r"_([^_\n]+)_", r"\1", s)
    # Asterisks must never reach Telegram or TTS (edge-tts would say "asterisk").
    s = s.replace("*", " ")
    s = s.replace("_", " ")
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r" *\n *", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _sentences(text: str) -> list[str]:
    parts = [p.strip() for p in SENT_SPLIT.split(text) if p and p.strip()]
    out: list[str] = []
    buf = ""
    for p in parts:
        cand = (buf + " " + p).strip() if buf else p
        letters = sum(1 for c in cand if c.isalpha() or "\u0600" <= c <= "\u06FF")
        if letters < 18:
            buf = cand
            continue
        out.append(cand)
        buf = ""
    if buf and len(buf) > 20:
        out.append(buf)
    return out


def extractive_summary(text: str, target_chars: int = TARGET_SUMMARY_CHARS) -> str:
    sents = _sentences(text)
    if not sents:
        return text[:target_chars].strip()
    prefer_arabic = is_arabic(text)

    freqs: Counter[str] = Counter()
    for sent in sents:
        for w in WORD_RE.findall(sent.lower()):
            if w in STOPWORDS or len(w) < 3:
                continue
            freqs[w] += 1
    if not freqs:
        joined = " ".join(sents)
        return joined[:target_chars].strip()

    max_f = max(freqs.values())
    scored: list[tuple[float, int, str]] = []
    for i, sent in enumerate(sents):
        words = [w for w in WORD_RE.findall(sent.lower()) if w not in STOPWORDS and len(w) >= 3]
        if not words:
            continue
        n = len(sent)
        ar = sum(1 for c in sent if "\u0600" <= c <= "\u06FF")
        digits = sum(1 for c in sent if c.isdigit() or "٠" <= c <= "٩")
        tokens = sent.split()
        short = sum(1 for t in tokens if len(re.sub(r"\W+", "", t, flags=re.UNICODE)) <= 1) if tokens else 0
        # Drop OCR junk: too few letters, too many digits, too many 1-char tokens.
        if prefer_arabic and ar / max(n, 1) < 0.42:
            continue
        if not prefer_arabic and ar / max(n, 1) > 0.5:
            continue
        if digits / max(n, 1) > 0.12:
            continue
        if tokens and short / len(tokens) > 0.35:
            continue
        score = sum(freqs[w] / max_f for w in words) / len(words)
        if 70 <= n <= 380:
            score *= 1.3
        elif n < 45:
            score *= 0.5
        elif n > 700:
            score *= 0.7
        avg_w = sum(len(w) for w in words) / max(len(words), 1)
        if 3.5 <= avg_w <= 8:
            score *= 1.15
        scored.append((score, i, sent))

    if not scored:
        joined = " ".join(sents)
        return joined[:target_chars].strip()

    scored.sort(key=lambda t: t[0], reverse=True)
    chosen: list[tuple[int, str]] = []
    total = 0
    for _score, i, sent in scored:
        if total >= target_chars and len(chosen) >= 8:
            break
        if len(chosen) >= 28:
            break
        chosen.append((i, sent))
        total += len(sent) + 1
    chosen.sort(key=lambda t: t[0])
    summary = " ".join(s for _, s in chosen)
    return summary.strip()


def _translate_chunk(text: str, translator_name: str) -> str:
    from deep_translator import GoogleTranslator, MyMemoryTranslator

    if translator_name == "google":
        return GoogleTranslator(source="auto", target="ar").translate(text)
    return MyMemoryTranslator(source="english", target="arabic").translate(text)


def translate_to_arabic(text: str) -> str:
    if not text.strip():
        return text
    if is_arabic(text):
        return text
    chunks: list[str] = []
    buf = text
    # Google unofficial endpoint is happiest under ~4500 chars.
    size = 4200
    while buf:
        piece = buf[:size]
        if len(buf) > size:
            cut = max(piece.rfind(". "), piece.rfind("? "), piece.rfind("! "), piece.rfind("\n"))
            if cut >= size * 0.5:
                piece = piece[: cut + 1]
        chunks.append(piece.strip())
        buf = buf[len(piece) :].lstrip()

    translated: list[str] = []
    for piece in chunks:
        last_err: Exception | None = None
        for backend in ("google", "mymemory"):
            try:
                out = _translate_chunk(piece, backend)
                if out and out.strip():
                    translated.append(out.strip())
                    last_err = None
                    break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                log.warning("translate via %s failed: %s", backend, exc)
        if last_err is not None and not translated:
            raise RuntimeError("Arabic translation failed") from last_err
        elif last_err is not None:
            log.warning("chunk translation failed, skipping chunk")
    joined = "\n".join(translated).strip()
    if not is_arabic(joined):
        raise RuntimeError("translation result was not Arabic")
    return joined



GEMINI_MODELS = (
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-flash-latest",
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash",
)
RETRYABLE_HTTP = {429, 500, 502, 503, 504}
# Normal transport retries; 429 uses RATE_LIMIT_BACKOFFS instead.
RETRY_BACKOFFS = (3.0, 8.0, 15.0)
RATE_LIMIT_BACKOFFS = (20.0, 45.0, 90.0)
# Space out successive Gemini calls so chunked summaries do not trip free-tier RPM.
MIN_CALL_GAP_SEC = 12.0
GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
GEMINI_MAX_INPUT_CHARS = 80000
GEMINI_MIN_GOOD_CHARS = 2500

_gemini_model: str | None = None
_gemini_last_status: int | None = None
_gemini_last_error: str = ""
_gemini_key_index: int = 0
_gemini_last_call_at: float = 0.0


class GeminiError(RuntimeError):
    def __init__(self, status: int, snippet: str, message: str = "") -> None:
        self.status = status
        self.snippet = snippet
        super().__init__(message or f"Gemini HTTP {status}: {snippet}")


class GeminiProhibitedError(GeminiError):
    """Gemini blocked the prompt as PROHIBITED_CONTENT."""


class GeminiAuthError(GeminiError):
    pass


class SummarizeError(RuntimeError):
    """User-facing failure: do not send a fake extractive summary."""

    def __init__(self, message: str = "تعذّر إعداد الملخص") -> None:
        super().__init__(message)


def _redact_secrets(text: str) -> str:
    """Strip API keys / tokens from a snippet before logging or printing."""
    out = text or ""
    for env_name in (
        "GEMINI_API_KEY",
        "GEMINI_STORY_API_KEY",
        "GEMINI_TASHKEEL_API_KEY",
        "GEMINI_API_KEY_2",
        "GEMINI_API_KEY_3",
        "COHERE_API_KEY",
        "TELEGRAM_BOT_TOKEN",
        "OPENAI_API_KEY",
        "DEEPSEEK_API_KEY",
    ):
        val = (os.getenv(env_name) or "").strip()
        if val:
            out = out.replace(val, "[redacted]")
    out = re.sub(r"(?i)(key=)[^&\s\"']+", r"\1[redacted]", out)
    out = re.sub(r"(?i)(x-goog-api-key[\"'\s:=]+)[^\s\"']+", r"\1[redacted]", out)
    out = re.sub(r"AIza[0-9A-Za-z_\-]{10,}", "[redacted]", out)
    return out


def _body_snippet(resp: httpx.Response, limit: int = 280) -> str:
    raw = (resp.text or "")[:limit]
    return _redact_secrets(raw).replace("\n", " ").strip()


# Separate key pools: story (extract/rewrite) vs tashkeel (vocalize/split).
_gemini_key_index_by_purpose: dict[str, int] = {"story": 0, "tashkeel": 0}
_gemini_model_by_purpose: dict[str, str | None] = {"story": None, "tashkeel": None}
_gemini_purpose: str = "story"


def _gemini_api_keys(purpose: str | None = None) -> list[str]:
    """Return API keys for a purpose without mixing quotas when possible."""
    purpose = (purpose or _gemini_purpose or "story").strip().lower()
    if purpose not in {"story", "tashkeel"}:
        purpose = "story"
    keys: list[str] = []

    def _add(name: str) -> None:
        val = (os.getenv(name) or "").strip()
        if val and val not in keys:
            keys.append(val)

    if purpose == "tashkeel":
        # New tashkeel key first, then the previously configured key, then story keys.
        for name in (
            "GEMINI_TASHKEEL_API_KEY",
            "GEMINI_API_KEY",
            "GEMINI_STORY_API_KEY",
            "GEMINI_API_KEY_2",
            "GEMINI_API_KEY_3",
        ):
            _add(name)
        extra = (os.getenv("GEMINI_API_KEYS") or "").strip()
        if extra:
            for part in extra.replace(";", ",").split(","):
                val = part.strip()
                if val and val not in keys:
                    keys.append(val)
        return keys

    # story / default — new story key first, keep previous GEMINI_API_KEY in the pool
    for name in (
        "GEMINI_STORY_API_KEY",
        "GEMINI_API_KEY",
        "GEMINI_API_KEY_2",
        "GEMINI_API_KEY_3",
    ):
        _add(name)
    extra = (os.getenv("GEMINI_API_KEYS") or "").strip()
    if extra:
        for part in extra.replace(";", ",").split(","):
            val = part.strip()
            if val and val not in keys:
                keys.append(val)
    return keys


def _gemini_api_key(purpose: str | None = None) -> str:
    purpose = (purpose or _gemini_purpose or "story").strip().lower()
    if purpose not in _gemini_key_index_by_purpose:
        purpose = "story"
    keys = _gemini_api_keys(purpose)
    if not keys:
        return ""
    idx = _gemini_key_index_by_purpose.get(purpose, 0)
    if idx >= len(keys):
        idx = 0
        _gemini_key_index_by_purpose[purpose] = 0
    return keys[idx]


def _rotate_gemini_key(reason: str = "", purpose: str | None = None) -> bool:
    """Move to the next API key in the active purpose pool. Return True if rotated."""
    global _gemini_model
    purpose = (purpose or _gemini_purpose or "story").strip().lower()
    if purpose not in _gemini_key_index_by_purpose:
        purpose = "story"
    keys = _gemini_api_keys(purpose)
    if len(keys) <= 1:
        return False
    prev = _gemini_key_index_by_purpose.get(purpose, 0)
    nxt = (prev + 1) % len(keys)
    _gemini_key_index_by_purpose[purpose] = nxt
    _gemini_model_by_purpose[purpose] = None
    _gemini_model = None
    log.warning(
        "gemini rotating %s API key %s -> %s (%s)",
        purpose, prev + 1, nxt + 1, reason or "rate/auth",
    )
    return True


def _pace_gemini_calls() -> None:
    """Wait so consecutive Gemini requests are not fired back-to-back."""
    global _gemini_last_call_at
    now = time.monotonic()
    wait = MIN_CALL_GAP_SEC - (now - _gemini_last_call_at)
    if _gemini_last_call_at > 0 and wait > 0:
        log.info("gemini pacing sleep %.1fs before next request", wait)
        time.sleep(wait)
    _gemini_last_call_at = time.monotonic()


def _gemini_prompt(text: str, title: str, language: str, kind: str = "full", author: str = "") -> str:
    author = (author or "").strip() or "مؤلف غير مذكور"
    src_title = (title or "").strip() or "بدون عنوان"
    if title_is_arabic(src_title):
        title_rule = (
            "سطر العنوان: أبقِ العنوان العربي كما ورد في المصدر دون إعادة ترجمة إلى لغة أخرى.\n"
        )
        title_out = f"العنوان: {src_title}"
    else:
        title_rule = (
            "سطر العنوان (السطر 1): ترجم العنوان إلى عربية أدبية مهنية فصيحة. "
            "لا تُبقِ العنوان الإنجليزي ولا أي حروف لاتينية في هذا السطر. "
            "لا تعكس ترجمة عنوان عربي الأصل.\n"
        )
        title_out = "العنوان: (ترجمة عربية أدبية للعنوان بلا حروف لاتينية)"
    title_rule += (
        "سطر المؤلف (السطر 2): إن كان للمؤلف رسم عربي شائع (مثل روبرت لويس ستيفنسون) فاستخدمه؛ "
        "وإلا فانقل الاسم إلى الحروف العربية نقلاً صوتياً ليُنطَق في التحويل الصوتي. "
        "لا حروف لاتينية في سطري العنوان والمؤلف إن أمكن.\n"
        "جسم الملخص بالعربية الفصحى الأدبية فقط.\n"
    )
    kind_note = {
        "full": "",
        "first": "هذا النصف الأول من الكتاب. لخّصه تلخيصاً شاملاً دقيقاً دون اختزال مخلّ.\n",
        "second": "هذا النصف الثاني من الكتاب. لخّصه تلخيصاً شاملاً دقيقاً دون اختزال مخلّ.\n",
        "merge": (
            "أمامك ملخصان جزئيّان لنصفي الكتاب. ادمجهما في مقال عربي واحد متماسك "
            "بأسلوب أدبي فصيح، مع إزالة التكرار وصقل الصياغة واستكمال الحبكة أو الحجّة.\n"
        ),
    }.get(kind, "")
    return (
        "أنت ملخّص أدبي محترف. اكتب ملخصاً عربياً فصيحاً شاملاً وكاملاً للكتاب التالي.\n"
        "يجب أن يغطي الملخص: المقدمة والفرضية، بنية العمل وترتيب أقسامه، "
        "الحبكة أو الحجّة كاملة مع نقاط التحوّل، الشخصيات أو الأفكار الرئيسة، "
        "الثيمات، والنهاية أو الخلاصة كما وردت في النص.\n"
        "الأسلوب: مقال متّصل متأنٍّ بجودة أدبية عالية، حوالي 1200 إلى 2200 كلمة عربية. "
        "لا تضف معلومات من خارج النص المعطى.\n"
        "FORBIDDEN: لا تكتب مقدّمات ميتا مثل «ملخص الكتاب» أو «في هذا الملخص» أو «سأقوم بتلخيص». "
        "جملة حبكتية مثل «هذه الرواية تدور» مقبولة إن كانت سرداً لا تعليقاً على فعل التلخيص. "
        "FORBIDDEN: لا تكتب خواتيم مثل «في الختام» أو «خلاصة القول» أو «وهكذا نرى» أو «نسأل الله». "
        "FORBIDDEN: لا تسمّ أجزاء («الجزء 1 من 6» أو «يتبع»).\n"
        "FORBIDDEN: يُمنع منعاً باتاً استخدام رمز النجمة * أو ** أو أي ماركداون "
        "(غامق/مائل/تعداد بنجمة) أو __ أو _مائل_ أو علامات الخلفية ` أو عناوين تبدأ بـ #. "
        "اكتب نصاً عادياً بلا تمييز بصري.\n"
        "التشكيل الصوتي: ضع حركات قليلة وحاسمة فقط على الكلمات التي يُخطئ القارئ الآلي في نطقها "
        "(مثل إِنَّ مقابل أَنْ، التنوين في جِدّاً وأيضاً، حركة المبني للمجهول مثل يُقال). "
        "لا تشكّل كل حرف في كل كلمة، ولا تبطئ النص بكثرة الحركات.\n"
        f"{title_rule}{kind_note}"
        "الناتج يجب أن يبدأ حرفياً بالسطرين التاليين ثم سطر فارغ ثم جسم الملخص مباشرة دون أي مقدّمة أخرى:\n"
        f"{title_out}\n"
        "المؤلف: (اسم المؤلف بالعربية فقط)\n\n"
        f"عنوان المصدر: {src_title}\n"
        f"مؤلف المصدر: {author}\n\n"
        f"النص:\n{text}"
    )


def _parse_gemini_response(data: dict) -> tuple[str, str]:
    cands = data.get("candidates") or []
    if not cands:
        feedback = data.get("promptFeedback") or {}
        reason = str(feedback.get("blockReason") or "no_candidates")
        return "", reason
    cand = cands[0]
    reason = str(cand.get("finishReason") or "")
    parts = ((cand.get("content") or {}).get("parts") or [])
    text = "".join((p.get("text") or "") for p in parts).strip()
    return text, reason


def _gemini_post(model: str, prompt: str, api_key: str) -> httpx.Response:
    url = GEMINI_ENDPOINT.format(model=model)
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": 16384},
    }
    timeout = httpx.Timeout(connect=30.0, read=240.0, write=60.0, pool=30.0)
    with httpx.Client(timeout=timeout) as client:
        return client.post(
            url,
            headers={
                "x-goog-api-key": api_key,
                "Content-Type": "application/json",
            },
            json=payload,
        )


def _retryable_http(status: int, snippet: str) -> bool:
    if status in RETRYABLE_HTTP:
        return True
    low = (snippet or "").lower()
    return any(
        tok in low
        for tok in (
            "unavailable",
            "resource_exhausted",
            "high demand",
            "try again later",
            "overloaded",
            "temporarily",
        )
    )


def _gemini_generate(prompt: str, purpose: str = "story") -> tuple[str, str]:
    """Call Gemini; cache the first working model. Returns (text, finish_reason).

    purpose: "story" (extract/rewrite) or "tashkeel" (vocalize/split) — separate API keys.
    On 429/503/UNAVAILABLE/timeout: retry the same model 3 times (backoff 2s, 5s),
    then try the next model. Do not raise until every model is exhausted.
    """
    global _gemini_model, _gemini_last_status, _gemini_last_error, _gemini_purpose
    purpose = (purpose or "story").strip().lower()
    if purpose not in {"story", "tashkeel"}:
        purpose = "story"
    _gemini_purpose = purpose
    api_key = _gemini_api_key(purpose)
    _gemini_last_status = None
    _gemini_last_error = ""
    if not api_key:
        _gemini_last_error = f"GEMINI key missing for purpose={purpose}"
        raise GeminiError(0, _gemini_last_error)

    cached = _gemini_model_by_purpose.get(purpose) or _gemini_model
    if cached:
        order = [cached] + [m for m in GEMINI_MODELS if m != cached]
    else:
        order = list(GEMINI_MODELS)

    last_err: GeminiError | None = None
    attempts = 1 + len(RETRY_BACKOFFS)
    for model in order:
        for attempt in range(1, attempts + 1):
            try:
                _pace_gemini_calls()
                api_key = _gemini_api_key()
                resp = _gemini_post(model, prompt, api_key)
            except httpx.TimeoutException as exc:
                snip = _redact_secrets(str(exc))[:240]
                log.warning(
                    "gemini timeout model=%s attempt=%s/%s: %s",
                    model, attempt, attempts, snip,
                )
                last_err = GeminiError(0, f"timeout: {snip}")
                if attempt < attempts:
                    delay = RETRY_BACKOFFS[attempt - 1]
                    log.info("gemini retry model=%s after %.0fs (timeout)", model, delay)
                    time.sleep(delay)
                    continue
                break
            except Exception as exc:  # noqa: BLE001
                snip = _redact_secrets(str(exc))[:240]
                log.warning(
                    "gemini request error model=%s attempt=%s/%s: %s",
                    model, attempt, attempts, snip,
                )
                last_err = GeminiError(0, snip)
                if attempt < attempts:
                    delay = RETRY_BACKOFFS[attempt - 1]
                    log.info("gemini retry model=%s after %.0fs (transport)", model, delay)
                    time.sleep(delay)
                    continue
                break

            _gemini_last_status = resp.status_code
            snippet = _body_snippet(resp)
            if resp.status_code == 404:
                log.warning("gemini model 404 model=%s body=%s", model, snippet)
                last_err = GeminiError(404, snippet)
                if _gemini_model == model:
                    _gemini_model = None
                break
            if resp.status_code in {401, 403}:
                _gemini_last_error = snippet
                log.warning("gemini auth/permission HTTP %s body=%s", resp.status_code, snippet)
                if _rotate_gemini_key(f"HTTP {resp.status_code}"):
                    api_key = _gemini_api_key()
                    continue
                raise GeminiAuthError(resp.status_code, snippet)
            if resp.status_code >= 400:
                log.warning(
                    "gemini HTTP %s model=%s attempt=%s/%s body=%s",
                    resp.status_code, model, attempt, attempts, snippet,
                )
                last_err = GeminiError(resp.status_code, snippet)
                low = snippet.lower()
                if "not found" in low or "not supported" in low:
                    if _gemini_model == model:
                        _gemini_model = None
                    break
                if _retryable_http(resp.status_code, snippet) and attempt < attempts:
                    if resp.status_code == 429:
                        delay = RATE_LIMIT_BACKOFFS[min(attempt - 1, len(RATE_LIMIT_BACKOFFS) - 1)]
                        _rotate_gemini_key("HTTP 429")
                        api_key = _gemini_api_key()
                    else:
                        delay = RETRY_BACKOFFS[min(attempt - 1, len(RETRY_BACKOFFS) - 1)]
                    log.info(
                        "gemini retry model=%s after %.0fs (HTTP %s)",
                        model, delay, resp.status_code,
                    )
                    time.sleep(delay)
                    continue
                log.warning(
                    "gemini giving up model=%s after %s attempts; trying next model",
                    model, attempt,
                )
                break

            try:
                data = resp.json()
            except Exception as exc:  # noqa: BLE001
                snip = _redact_secrets(str(exc))[:200]
                log.warning("gemini JSON parse failed model=%s: %s", model, snip)
                last_err = GeminiError(resp.status_code, snip)
                if attempt < attempts:
                    time.sleep(RETRY_BACKOFFS[attempt - 1])
                    continue
                break

            text, reason = _parse_gemini_response(data)
            if not text:
                log.warning(
                    "gemini empty response model=%s reason=%s attempt=%s/%s body=%s",
                    model, reason, attempt, attempts, snippet,
                )
                reason_u = (reason or "").upper()
                if "PROHIBITED" in reason_u or "BLOCKLIST" in reason_u or "SAFETY" in reason_u:
                    # Content policy block — retrying the same prompt rarely helps.
                    last_err = GeminiProhibitedError(resp.status_code, f"empty reason={reason}")
                    break
                last_err = GeminiError(resp.status_code, f"empty reason={reason}")
                if _retryable_http(resp.status_code, snippet + " " + reason) and attempt < attempts:
                    time.sleep(RETRY_BACKOFFS[attempt - 1])
                    continue
                if attempt < attempts:
                    time.sleep(RETRY_BACKOFFS[attempt - 1])
                    continue
                break

            _gemini_model = model
            _gemini_model_by_purpose[_gemini_purpose] = model
            _gemini_last_error = ""
            log.info("gemini ok model=%s chars=%s finish=%s", model, len(text), reason or "STOP")
            return text, reason or "STOP"

    _gemini_last_error = str(last_err) if last_err else "all models failed"
    if last_err:
        raise last_err
    raise GeminiError(0, "all Gemini models failed")


def _split_halves(text: str) -> tuple[str, str]:
    chunks = _split_chunks(text, 2)
    if len(chunks) == 1:
        mid = max(1, len(text) // 2)
        return text[:mid].strip(), text[mid:].strip()
    return chunks[0], chunks[1]


def _split_chunks(text: str, n: int) -> list[str]:
    """Split a long book into n chunks on paragraph/sentence boundaries."""
    text = (text or "").strip()
    if n <= 1 or len(text) < 4000:
        return [text] if text else []
    bounds = [0]
    for i in range(1, n):
        target = len(text) * i // n
        window_lo = int(len(text) * (i / n) - len(text) * 0.08)
        window_hi = int(len(text) * (i / n) + len(text) * 0.08)
        window_lo = max(bounds[-1] + 1, window_lo)
        window_hi = min(len(text) - 1, max(window_lo + 1, window_hi))
        cut = text.rfind("\n\n", window_lo, window_hi)
        if cut < window_lo:
            cut = text.rfind(". ", window_lo, window_hi)
        if cut < window_lo:
            cut = target
        else:
            cut = cut + 1
        if cut <= bounds[-1]:
            cut = min(len(text), bounds[-1] + max(1, (len(text) - bounds[-1]) // (n - i + 1)))
        bounds.append(cut)
    bounds.append(len(text))
    return [text[a:b].strip() for a, b in zip(bounds, bounds[1:]) if text[a:b].strip()]


def _large_excerpt(text: str, max_chars: int = GEMINI_MAX_INPUT_CHARS) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    try:
        from books import representative_text

        return representative_text(text, max_chars=max_chars)
    except Exception:
        third = max_chars // 3
        mid = len(text) // 2
        return "\n\n".join([text[:third], text[mid : mid + third], text[-third:]])


def _output_truncated(text: str, reason: str, expected_long: bool) -> bool:
    if (reason or "").upper() == "MAX_TOKENS":
        return True
    if expected_long and len(text) < GEMINI_MIN_GOOD_CHARS:
        return True
    return False


def _merge_partials(partials: list[str], title: str, language: str, author: str) -> str:
    merged_src = "\n\n".join(
        f"الملخص الجزئي {i + 1}:\n{p}" for i, p in enumerate(partials)
    )
    out, _reason = _gemini_generate(
        _gemini_prompt(merged_src, title, language, kind="merge", author=author)
    )
    return out


def _gemini_chunk_summary(source: str, title: str, language: str, author: str) -> str:
    n = 3 if len(source) > 200_000 else 2
    log.info("gemini long-book path chars=%s chunks=%s", len(source), n)
    chunks = [_large_excerpt(c) for c in _split_chunks(source, n)]
    if not chunks:
        chunks = [_large_excerpt(source)]
    partials: list[str] = []
    for i, chunk in enumerate(chunks):
        kind = "first" if i == 0 else "second"
        piece, _reason = _gemini_generate(
            _gemini_prompt(chunk, title, language, kind=kind, author=author)
        )
        partials.append(piece)
    try:
        return _merge_partials(partials, title, language, author)
    except GeminiError as exc:
        log.warning("gemini merge failed: HTTP %s %s", exc.status, exc.snippet)
        joined = "\n\n".join(partials)
        if joined and is_arabic(joined):
            return joined
        raise


def _gemini_summary(text: str, title: str, language: str, author: str = "") -> str | None:
    """Professional Arabic summary via Gemini. None if unavailable or failed."""
    if not _gemini_api_key():
        return None
    source = (text or "").strip()
    if not source:
        return None
    try:
        if len(source) > GEMINI_MAX_INPUT_CHARS:
            # Always read the book in 2–3 chunks; never a single 80k sample.
            out = _gemini_chunk_summary(source, title, language, author)
        else:
            excerpt = _large_excerpt(source)
            expected_long = len(excerpt) >= 8000
            prompt = _gemini_prompt(excerpt, title, language, kind="full", author=author)
            out, reason = _gemini_generate(prompt)
            if _output_truncated(out, reason, expected_long) and len(source) > 25000:
                log.info(
                    "gemini output truncated (reason=%s chars=%s); splitting",
                    reason, len(out),
                )
                out = _gemini_chunk_summary(source, title, language, author)
    except GeminiError as exc:
        log.warning("gemini summary failed: HTTP %s %s", exc.status, exc.snippet)
        return None
    except Exception as exc:  # noqa: BLE001
        log.warning("gemini summary failed: %s", _redact_secrets(str(exc))[:240])
        return None

    _, _, body_for_check = parse_summary_meta(out)
    check = body_for_check or out
    if not out or not is_arabic(check):
        log.warning("gemini result not Arabic or empty (chars=%s)", len(out or ""))
        return None
    return out


def _openai_summary(text: str, title: str) -> str | None:
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        return None
    excerpt = text[:12000]
    payload = {
        "model": os.getenv("OPENAI_MODEL") or "gpt-4o-mini",
        "temperature": 0.4,
        "messages": [
            {
                "role": "system",
                "content": (
                    "لخّص مقطع الكتاب التالي بالعربية الفصحى الواضحة، "
                    "في حوالي 900 إلى 1200 كلمة، بأسلوب سردي متسلسل يناسب القراءة الصوتية. "
                    "لا تضف معلومات من خارج المقطع. أعد النص العربي فقط."
                ),
            },
            {
                "role": "user",
                "content": f"عنوان الكتاب: {title}\n\nالمقطع:\n{excerpt}",
            },
        ],
    }
    try:
        with httpx.Client(timeout=90.0) as client:
            r = client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=payload,
            )
            r.raise_for_status()
            data = r.json()
        content = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
        if content and is_arabic(content) and len(content) >= 800:
            return content
    except Exception as exc:  # noqa: BLE001
        log.warning("OpenAI summary failed, falling back to extractive: %s", exc)
    return None


def summarize_installment(text: str, title: str = "", language: str = "ara", author: str = "") -> str:
    if not _gemini_api_key():
        log.warning("GEMINI_API_KEY missing; extractive+translate last-last resort")
        fallback_src = text
        if len(text) > GEMINI_MAX_INPUT_CHARS:
            fallback_src = _large_excerpt(text)
        extracted = extractive_summary(fallback_src)
        arabic = translate_to_arabic(extracted)
        if len(arabic) < PART_MIN * MIN_PARTS:
            extracted = extractive_summary(fallback_src, target_chars=TARGET_SUMMARY_CHARS * 2)
            arabic = translate_to_arabic(extracted)
        return arabic
    gemini = _gemini_summary(text, title, language, author=author)
    if gemini:
        return gemini
    log.warning("Gemini summary failed after retries; not using extractive fallback")
    status = _gemini_last_status
    if status == 429:
        raise SummarizeError(
            "وصلت لحد Gemini المجاني. انتظر دقيقة ثم أرسل /now مرة ثانية."
        )
    raise SummarizeError(
        "تعذّر إعداد الملخص الآن. انتظر قليلاً ثم أعد المحاولة عبر /now."
    )


def _split_units(text: str) -> list[str]:
    units = [u.strip() for u in SENT_SPLIT.split(text) if u and u.strip()]
    return units or [text.strip()]


def split_parts(text: str, min_parts: int = MIN_PARTS, max_parts: int = MAX_PARTS) -> list[str]:
    """Split Arabic summary into 3–6 bare text chunks of ~700–1200 characters."""
    text = strip_part_labels(text)
    text = sanitize_for_speech(text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    n = len(text)
    # Choose a part count that keeps each piece in the 700–1200 band when possible.
    target_count = min(max_parts, max(min_parts, round(n / 950)))
    target_count = max(1, min(max_parts, target_count))
    target_len = max(PART_MIN, min(PART_MAX, n // target_count if target_count else n))

    units = _split_units(text)
    parts: list[str] = []
    buf = ""
    for unit in units:
        trial = (buf + " " + unit).strip() if buf else unit
        if len(trial) <= PART_MAX and (not buf or len(trial) < target_len or len(buf) < PART_MIN):
            buf = trial
            continue
        if buf:
            parts.append(buf)
        if len(unit) > PART_MAX:
            # Hard-wrap oversized units on spaces.
            words = unit.split()
            chunk = ""
            for w in words:
                t = (chunk + " " + w).strip()
                if len(t) > PART_MAX and chunk:
                    parts.append(chunk)
                    chunk = w
                else:
                    chunk = t
            buf = chunk
        else:
            buf = unit
    if buf:
        parts.append(buf)

    # Merge tiny tails.
    merged: list[str] = []
    for p in parts:
        if merged and (len(p) < PART_MIN * 0.55 or (len(merged[-1]) + 1 + len(p) <= PART_MAX and len(merged[-1]) < PART_MIN)):
            merged[-1] = (merged[-1] + " " + p).strip()
        else:
            merged.append(p)

    # If we have more than max_parts, merge neighbors.
    while len(merged) > max_parts:
        # Merge the two shortest adjacent pair.
        best_i = 0
        best_len = 10**9
        for i in range(len(merged) - 1):
            s = len(merged[i]) + len(merged[i + 1])
            if s < best_len:
                best_len = s
                best_i = i
        merged[best_i] = (merged[best_i] + " " + merged[best_i + 1]).strip()
        del merged[best_i + 1]

    # If fewer than min_parts and text is long enough, rebalance by character windows.
    if len(merged) < min_parts and n >= PART_MIN * 2:
        merged = _rebalance(text, min(min_parts, max(2, n // PART_MIN)))

    # Rebalance if the tail is much shorter than the target band.
    if merged and len(merged) >= 2 and len(merged[-1]) < PART_MIN * 0.75:
        merged = _rebalance(" ".join(merged), len(merged) if len(merged[-1]) >= 400 else max(min_parts, len(merged) - 1))

    return [p.strip() for p in merged if p.strip()]


def _rebalance(text: str, count: int) -> list[str]:
    units = _split_units(text)
    total = sum(len(u) for u in units) or len(text)
    goal = total / count
    parts: list[str] = []
    buf = ""
    for unit in units:
        trial = (buf + " " + unit).strip() if buf else unit
        if buf and len(buf) >= goal * 0.8 and len(parts) < count - 1:
            parts.append(buf)
            buf = unit
        else:
            buf = trial
    if buf:
        parts.append(buf)
    return parts


TITLE_LINE_RE = re.compile(r"^\s*العنوان\s*[:：]\s*(.+?)\s*$")
AUTHOR_LINE_RE = re.compile(r"^\s*المؤلف\s*[:：]\s*(.+?)\s*$")
PART_LABEL_RE = re.compile(
    r"^[\s«]*الجزء\s+\d+\s+من\s+\d+[^\n»]*»?\s*$",
    re.M,
)
YATBA_RE = re.compile(r"^\s*يتبع\.?\s*$", re.M)
MD_HEADING_RE = re.compile(r"^#{1,6}\s+.*$", re.M)


def strip_part_labels(text: str) -> str:
    """Drop leftover part headers, 'يتبع', and markdown headings."""
    text = text or ""
    text = PART_LABEL_RE.sub("", text)
    text = re.sub(
        r"«الجزء\s+\d+\s+من\s+\d+[^»]*»",
        "",
        text,
    )
    text = YATBA_RE.sub("", text)
    text = MD_HEADING_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_summary_meta(text: str) -> tuple[str | None, str | None, str]:
    """If Gemini prepended العنوان/المؤلف, return those and the remaining body."""
    raw = (text or "").strip()
    if not raw:
        return None, None, ""
    lines = raw.splitlines()
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    title = None
    author = None
    if i < len(lines):
        m = TITLE_LINE_RE.match(lines[i])
        if m:
            title = m.group(1).strip() or None
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            if not title and i < len(lines) and not AUTHOR_LINE_RE.match(lines[i]):
                title = lines[i].strip() or None
                if title:
                    i += 1
                    while i < len(lines) and not lines[i].strip():
                        i += 1
        elif re.fullmatch(r"العنوان\s*[:：]?\s*", lines[i].strip()):
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            if i < len(lines):
                title = lines[i].strip() or None
                i += 1
                while i < len(lines) and not lines[i].strip():
                    i += 1
        if i < len(lines):
            m2 = AUTHOR_LINE_RE.match(lines[i])
            if m2:
                author = m2.group(1).strip() or None
                i += 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    body = "\n".join(lines[i:]).strip()
    return title, author, body


def _extract_title_candidate(out: str) -> str | None:
    """Parse a Gemini title-translation reply. Handles العنوان: on its own line."""
    out = sanitize_for_speech(out or "")
    if not out:
        return None
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    if not lines:
        return None
    for i, line in enumerate(lines):
        m = TITLE_LINE_RE.match(line)
        if m:
            cand = sanitize_for_speech(m.group(1).strip())
            if cand:
                return cand
            if i + 1 < len(lines):
                nxt = sanitize_for_speech(re.sub(r"^العنوان\s*[:：]\s*", "", lines[i + 1]).strip())
                if nxt:
                    return nxt
        if re.fullmatch(r"العنوان\s*[:：]?\s*", line) and i + 1 < len(lines):
            nxt = sanitize_for_speech(lines[i + 1])
            if nxt:
                return nxt
    first = re.sub(r"^العنوان\s*[:：]\s*", "", lines[0]).strip()
    first = sanitize_for_speech(first)
    if not first and len(lines) > 1:
        first = sanitize_for_speech(lines[1])
    return first or None


def _gemini_translate_title(title: str) -> str | None:
    """Tiny Gemini call: Arabic translation of a book title. None on failure."""
    title = (title or "").strip()
    if not title:
        return None
    if not _gemini_api_key():
        return None
    prompt = (
        "ترجم عنوان الكتاب التالي إلى عربية أدبية مهنية فصيحة. "
        "إن كان العنوان عربياً أصلاً فأعده كما هو دون تغيير. "
        "أرجع سطراً واحداً يبدأ حرفياً بـ العنوان: ثم الترجمة فقط. "
        "ممنوع استخدام * أو ماركداون أو حروف لاتينية في الناتج.\n"
        f"العنوان: {title}"
    )
    try:
        out, _reason = _gemini_generate(prompt)
    except Exception as exc:  # noqa: BLE001
        log.warning("gemini title translate failed: %s", _redact_secrets(str(exc))[:200])
        return None
    cand = _extract_title_candidate(out or "")
    if cand:
        return cand
    log.warning(
        "title translation parse failed chars=%s preview=%s",
        len(out or ""),
        sanitize_for_speech(out or "")[:80],
    )
    return None


def _gemini_translate_title_strict(title: str) -> str | None:
    """Second-chance tiny prompt: one Arabic line, no Latin."""
    title = (title or "").strip()
    if not title or not _gemini_api_key():
        return None
    prompt = f"ترجم هذا العنوان فقط إلى العربية الفصحى، سطر واحد: {title}"
    try:
        out, _reason = _gemini_generate(prompt)
    except Exception as exc:  # noqa: BLE001
        log.warning("gemini strict title translate failed: %s", _redact_secrets(str(exc))[:200])
        return None
    cand = _extract_title_candidate(out or "")
    if cand:
        return cand
    first = sanitize_for_speech((out or "").splitlines()[0] if (out or "").strip() else "")
    return first or None


def _accept_arabic_title(raw: str | None, via: str, original: str) -> str | None:
    if not raw:
        log.warning("title translation empty via=%s original=%s", via, original[:80])
        return None
    cand = sanitize_for_speech(raw)
    cleaned = strip_latin_title_extras(cand) or cand
    if title_is_arabic(cleaned):
        log.info("title translated via=%s result=%s", via, cleaned[:80])
        return cleaned
    reason = title_reject_reason(cleaned)
    log.warning(
        "title translation rejected via=%s reason=%s sample=%s",
        via, reason, cleaned[:80],
    )
    return None


def ensure_arabic_title(title: str) -> str:
    """Keep Arabic titles; otherwise translate via Gemini. Never keep English."""
    title = sanitize_for_speech((title or "").strip())
    if not title:
        return title
    cleaned = strip_latin_title_extras(title) or title
    if title_is_arabic(cleaned):
        return cleaned
    got = _accept_arabic_title(_gemini_translate_title(title), "gemini", title)
    if got:
        return got
    got = _accept_arabic_title(_gemini_translate_title(title), "gemini_retry", title)
    if got:
        return got
    got = _accept_arabic_title(_gemini_translate_title_strict(title), "gemini_strict", title)
    if got:
        return got
    log.warning(
        "title translation failed after retries; not keeping Latin title=%s",
        title[:80],
    )
    if _gemini_last_status == 429:
        raise SummarizeError(
            "وصلت لحد Gemini المجاني. انتظر دقيقة ثم أرسل /now مرة ثانية."
        )
    raise SummarizeError(
        "تعذّر إعداد الملخص الآن. انتظر قليلاً ثم أعد المحاولة عبر /now."
    )


def parts_from_summary(
    summary: str,
    fallback_title: str,
    fallback_author: str = "",
) -> tuple[str, str, list[str]]:
    """Parse title/author if present, then split the remaining body into bare chunks."""
    cleaned = strip_part_labels(summary or "")
    parsed_title, parsed_author, body = parse_summary_meta(cleaned)
    if not body:
        body = cleaned
    body = sanitize_for_speech(strip_part_labels(body))
    title = sanitize_for_speech((parsed_title or "").strip() or (fallback_title or "").strip())
    author = sanitize_for_speech(
        (parsed_author or "").strip() or (fallback_author or "").strip() or "مؤلف غير مذكور"
    )
    title = ensure_arabic_title(title)
    parts = split_parts(body)
    if len(parts) < 2 and body:
        mid = max(1, len(body) // 2)
        cut = body.rfind(" ", 0, mid) or mid
        parts = [p for p in (body[:cut].strip(), body[cut:].strip()) if p]
    parts = [sanitize_for_speech(strip_part_labels(p)) for p in parts if p and p.strip()]
    return title, author, parts


def outgoing_text_messages(title: str, author: str, parts: list[str]) -> list[str]:
    """First Telegram text is title+author+body; later messages are body only."""
    title = sanitize_for_speech((title or "").strip())
    author = sanitize_for_speech((author or "").strip() or "مؤلف غير مذكور")
    clean_parts = [
        sanitize_for_speech(strip_part_labels(p))
        for p in (parts or [])
        if (p or "").strip()
    ]
    if not clean_parts:
        return [f"{title}\n{author}"]
    first = f"{title}\n{author}\n\n{clean_parts[0]}"
    return [first, *clean_parts[1:]]


def format_part(part_text: str, index: int = 0, total: int = 0, book_title: str = "") -> str:
    """Return bare part text. Part headers are no longer used."""
    return sanitize_for_speech(strip_part_labels(part_text or ""))


def build_arabic_parts(installment_text: str, book_title: str, author: str = "") -> list[str]:
    summary = summarize_installment(installment_text, title=book_title, author=author)
    _title, _author, parts = parts_from_summary(summary, book_title, author)
    return parts


def build_daily_parts(
    book_text: str,
    book_title: str,
    language: str,
    author: str = "",
) -> tuple[str, str, list[str]]:
    """Whole-book Arabic summary split into bare Telegram parts.

    Returns (title, author, parts). Parts have no «الجزء» headers.
    """
    # Pass the full book so long texts take the 2–3 chunk Gemini path.
    summary = summarize_installment(
        book_text, title=book_title, language=language, author=author
    )
    return parts_from_summary(summary, book_title, author)




_TASHKEEL_RE = re.compile(r"[\u064B-\u0652\u0670\u0640]")  # harakat + tatweel
_PUNCT_RE = re.compile(
    r"[\s\d"
    r"\u060C\u061B\u061F\u06D4"  # Arabic comma/semicolon/question/full stop
    r".,!?;:\"'()\[\]{}<>«»…—–\-_/@#$%^&*+=~`\\|]"
)



_MORAL_STARTS = (
    "والعبرة", "العبرة", "وهكذا نتعلم", "وفي هذا درس", "والدرس المستفاد",
    "خلاصة القول", "نستنتج أن", "فليعتبر", "فليكن هذا درسا",
    "والحكمة من", "الحكمة من ذلك",
)


def _strip_moral_tail(text: str) -> str:
    """Remove a trailing moral/lesson paragraph if present."""
    body = (text or "").strip()
    if not body:
        return body
    # Prefer cutting from a blank-line-separated closing moral block
    paras = re.split(r"\n\s*\n", body)
    if len(paras) >= 2:
        last = paras[-1].strip()
        if any(last.startswith(s) for s in _MORAL_STARTS):
            return "\n\n".join(paras[:-1]).rstrip()
    # Same-paragraph / last-sentence moral
    for s in _MORAL_STARTS:
        idx = body.rfind(s)
        if idx > len(body) * 0.55:
            return body[:idx].rstrip(" \n،,")
    return body





def bare_letter_count(text: str) -> int:
    """Count letters only — exclude tashkeel, punctuation, digits, and whitespace."""
    if not text:
        return 0
    s = _TASHKEEL_RE.sub("", text)
    s = _PUNCT_RE.sub("", s)
    return sum(1 for c in s if c.isalpha() or ("\u0600" <= c <= "\u06FF"))


def trim_to_bare_letters(text: str, limit: int = 7800) -> str:
    """Trim text so bare_letter_count <= limit, preferring a sentence end."""
    if bare_letter_count(text) <= limit:
        return text.strip()
    words = text.split()
    out: list[str] = []
    for w in words:
        trial = (" ".join(out + [w])).strip()
        if bare_letter_count(trial) > limit and out:
            break
        out.append(w)
    body = " ".join(out).rstrip()
    for sep in ("۔", ".", "؟", "!", "…"):
        cut = body.rfind(sep)
        if cut > len(body) * 0.65:
            body = body[: cut + 1]
            break
    return body.strip()


def _folk_story_prompt(
    source_text: str,
    title: str,
    source_lang: str,
    author: str = "",
    collection_title: str = "",
) -> str:
    """Build Gemini prompt: English jestbook/merry tale → Arabic nawadir-style narration."""
    lang_names = {
        "ara": "العربية",
        "eng": "الإنجليزية",
        "rus": "الروسية",
        "urd": "الأردية",
    }
    src = lang_names.get(source_lang, source_lang)
    author = (author or "").strip() or "مؤلف غير مذكور"
    title = (title or "").strip() or "نادرة"
    book = (collection_title or "").strip() or "كتاب نوادر إنجليزي"
    return (
        "أنت راوي نوادر وحكايات اجتماعية طريفة. المادة من كتب إنجليزية قديمة من نوع Jestbooks "
        "(نوادر / حكايات مرحة / طرف اجتماعية / فابليو / حكايات زواج وقضاة ونساء) وليست تاريخ ملوك وحروب.\n"
        "مهمتك: خذ النادرة/الحكاية في النص المصدري من أولها إلى آخرها، وترجمها وأعد صياغتها بالعربية الفصحى الأدبية "
        "بأسلوب قريب من نوادر العرب والبخلاء. المجالات تشمل: نساء، قضاة ومحامين، متزوجين وعشاق، "
        "حيلة وحمق وبخل، ومواقف ذات طابع جنسي أو جريء إن وُجدت في الأصل.\n"
        "أمانة النقل أولاً: انقل المعنى والأحداث والشخصيات والنهاية كما في المصدر دون تحريف أو تليين أو حذف "
        "للمشاهد الجريئة أو الجنسية، ودون اختراع مشاهد لم تكن في الأصل، ودون تحويلها إلى وعظ أخلاقي.\n"
        "ابدأ من بداية الموقف الأصلي (تعريف الشخصيات والمكان والسبب) ولا تقفز إلى الوسط. وسّع المشهد والحوار "
        "ليصلح للاستماع نحو 9 دقائق، مع الحفاظ على روح النكتة والعقدة والنهاية كما في الأصل "
        "(لا تختلق نهاية مختلفة ولا تحوّلها إلى درس تاريخي).\n"
        "الأسلوب: أنيق سلس، خفيف الظل، صور حية، بلا عامية ثقيلة وبلا ميلودراما صاخبة. "
        "إن كان الأصل جريئاً فكن واضحاً بأدب سردي دون ابتذال زائد ودون رقابة تحذف المعنى.\n"
        "العنوان يجب أن يسمّي الشخصية أو الموقف الطريف بوضوح (مثل «البخيل والضيف» أو «القاضي والشاهدة» أو «الزوجة والحيلة»). "
        "ممنوع منعاً باتاً أن يكون العنوان اسم الكتاب أو ترجمة لاسم المجموعة "
        f"(مثل ترجمة «{book}» أو «نوادر إنجليزية» أو «طرائف جمعت من مصادر»).\n"
        "إذا كان المقطع مقدمة أو فهرساً بلا نادرة: أجب فقط بالسطر: رفض: مادة غير سردية\n"
        "ممنوع ديباجة طويلة بعد العنوان، وممنوع عبرة وعظية ختامية من طراز «وهكذا نتعلم».\n"
        "بعد انتهاء أحداث النادرة مباشرة: اختم بخاتمة قصيرة جداً (جملة أو جملتين) توحي أن القصة انتهت "
        "(مثل «وهنا تنتهي النادرة.» أو «وهكذا انقضت الحكاية.»). ممنوع أن تكون الخاتمة درساً أخلاقياً.\n"
        "احكِ سرداً متصلاً كقطعة واحدة كاملة من البداية للنهاية. ممنوع «جزء» أو «فصل» أو «تتمة» أو أرقام تقسيم داخل النص.\n"
        "أصلح تشويش الـOCR، وترجم الأسماء إلى صيغ عربية مألوفة إن أمكن.\n"
        f"لغة المصدر: {src}. الناتج بالعربية فقط.\n"
        "الطول المستهدف: نحو 9 دقائق رواية صوتية. "
        "حدّ الأحرف: حوالي 7200 إلى 7800 حرفاً من حروف الكلمات فقط "
        "(الترقيم والمسافات لا تُحسب). لا تتجاوز 7800 حرفاً صافياً.\n"
        "مهم جداً: اكتب بدون أي علامات تشكيل. نص عادي غير مشكّل.\n"
        "الترقيم: استخدم ترقيمًا عربيًا واضحاً.\n"
        "اختم بانتهاء الموقف ثم الخاتمة القصيرة التي تعلن انتهاء القصة (لا «يتبع»).\n"
        "FORBIDDEN: لا تشكيل، لا ميتا، لا ماركداون، لا نجوم *، لا عناوين #، لا لاتينية في جسم السرد.\n"
        "الناتج يبدأ حرفياً بالأسطر التالية ثم سطر فارغ ثم عنوان النادرة ثم السرد من البداية:\n"
        "العنوان: (عنوان عربي أدبي قصير للنادرة — ليس اسم الكتاب)\n"
        "الأصل: (اسم النادرة/الحكاية بالإنجليزية كما في المصدر إن وُجد، وإلا أقرب صياغة)\n"
        "الكتاب: (اسم الكتاب/المجموعة بالعربية)\n"
        "المؤلف: (اسم المؤلف إن وُجد)\n"
        "ثم سطر فارغ، ثم سطر عنوان النادرة وحده، ثم أحداث النادرة من أولها مباشرة.\n\n"
        f"عنوان النادرة المقترح من المصدر: {title}\n"
        f"اسم الكتاب/المجموعة: {book}\n"
        f"المؤلف: {author}\n\n"
        f"النص المصدري:\n{source_text}"
    )


_ENDING_HINTS = (
    "وهكذا", "فكان ما كان", "انتهت", "تمت الحكاية", "تمت القصة",
    "تنتهي النادرة", "انقضت الحكاية", "وهنا تنتهي",
    "وعاشا", "وعاشوا", "إلى يومنا", "وانتهت", "فخلصت", "نجا",
    "تزوّج", "تزوج", "ودخلوا", "الحمد", "الخاتمة",
)


def _story_looks_incomplete(text: str, finish_reason: str = "") -> bool:
    """True if the tale likely stopped early."""
    body = (text or "").strip()
    if not body:
        return True
    if (finish_reason or "").upper() == "MAX_TOKENS":
        return True
    words = body.split()
    if bare_letter_count(body) < 3000 or len(words) < 280:
        return True
    tail = body[-280:]
    # Ends mid-sentence / mid-word
    if body[-1] not in ".!?؟۔…»\"'”":
        # allow if last chars are Arabic letter then we likely truncated
        if body[-1].isalnum() or "\u0600" <= body[-1] <= "\u06FF" or body[-1] in "،,":
            return True
    # Very short and no ending hint
    low = tail.replace("ً", "").replace("ٌ", "").replace("ٍ", "")
    if bare_letter_count(body) < 4500 and not any(h in low for h in _ENDING_HINTS):
        # still OK if ends with clear sentence period and decent length
        if bare_letter_count(body) < 3800:
            return True
    return False


def _continue_folk_story(so_far: str, pass_no: int) -> tuple[str, str]:
    cont_prompt = (
        f"هذه تكملة رقم {pass_no} لنفس النادرة/الحكاية المرحة.\n"
        "أكمل من حيث توقّف النص حرفياً، بنفس الأسلوب الأدبي المشوّق وبدون أي تشكيل.\n"
        "لا تُعِد العنوان ولا المقدّمة ولا تلخّص ما سبق.\n"
        "أكمل النادرة حتى تنتهي. أبقِ حروف الكلمات تحت 7800 (الترقيم والمسافات لا تُحسب؛ ممنوع التشكيل؛ حوالي 9 دقائق كحد أقصى). "
        "لا مقدمة ولا عبرة وعظية ولا جزء/فصل/تتمة. إذا اقتربت من النهاية فأكمل الحدث ثم اختم بجملة قصيرة تعلن انتهاء القصة.\n"
        "لا ماركداون ولا نجوم ولا كلمة «يتبع».\n\n"
        f"النص حتى الآن:\n{so_far}"
    )
    more, reason = _gemini_generate(cont_prompt)
    return (more or "").strip(), reason or ""



def _split_exactly_three(text: str) -> list[str]:
    """Split body into exactly 3 sentence-aware chunks (~3 min speech each)."""
    text = sanitize_for_speech(strip_part_labels(text or "")).strip()
    if not text:
        return []
    units = _split_units(text)
    if len(units) < 3:
        # hard character thirds on word boundaries
        words = text.split()
        if len(words) < 3:
            return [text]
        n = len(words)
        cuts = [0, n // 3, (2 * n) // 3, n]
        return [" ".join(words[cuts[i]:cuts[i + 1]]).strip() for i in range(3) if words[cuts[i]:cuts[i + 1]]]
    total = sum(len(u) for u in units) or 1
    goal = total / 3
    parts: list[str] = []
    buf = ""
    for unit in units:
        trial = (buf + " " + unit).strip() if buf else unit
        if buf and len(buf) >= goal * 0.85 and len(parts) < 2:
            parts.append(buf)
            buf = unit
        else:
            buf = trial
    if buf:
        parts.append(buf)
    while len(parts) > 3:
        parts[-2] = (parts[-2] + " " + parts[-1]).strip()
        parts.pop()
    while len(parts) < 3 and parts:
        # split the longest part
        i = max(range(len(parts)), key=lambda k: len(parts[k]))
        words = parts[i].split()
        if len(words) < 4:
            break
        mid = len(words) // 2
        parts[i:i + 1] = [" ".join(words[:mid]).strip(), " ".join(words[mid:]).strip()]
    # Telegram safety
    safe: list[str] = []
    for p in parts[:3]:
        if len(p) <= 4000:
            safe.append(p)
            continue
        # rare: keep first 3900 at sentence
        cut = p.rfind(". ", 0, 3900)
        if cut < 2000:
            cut = 3900
        safe.append(p[: cut + 1].strip())
    while len(safe) < 3 and safe:
        safe.append(safe[-1])
    return [sanitize_for_speech(x) for x in safe[:3] if x]


def build_folk_story(
    source_text: str,
    title: str,
    source_lang: str = "ara",
    author: str = "",
    collection_title: str = "",
) -> tuple[str, str, list[str], str]:
    """Produce Arabic jest/nawadir story (translated/rewritten, no tashkeel), split for Telegram.

    Returns (title, source_label, parts, full_text).
    Uses multi-pass continuation until the tale has a clear ending (or max passes).
    """
    raw = (source_text or "").strip()
    if not raw:
        raise SummarizeError("نص الحكاية المصدر فارغ")
    if len(raw) > GEMINI_MAX_INPUT_CHARS:
        raw = raw[:GEMINI_MAX_INPUT_CHARS]
    prompt = _folk_story_prompt(
        raw, title, source_lang, author=author, collection_title=collection_title,
    )
    if not _gemini_api_key():
        raise SummarizeError("مفتاح Gemini غير متوفر")

    try:
        out, reason = _gemini_generate(prompt)
    except GeminiProhibitedError:
        raise
    except GeminiError as exc:
        log.warning("folk story gemini failed: %s", _redact_secrets(str(exc))[:200])
        # Preserve prohibited signal if nested in message
        if "PROHIBITED" in str(exc).upper():
            raise GeminiProhibitedError(getattr(exc, "status", 0) or 0, str(exc)) from exc
        raise SummarizeError(
            "تعذّر إعداد الحكاية الآن. انتظر قليلاً ثم أعد المحاولة عبر /now."
        ) from exc

    max_passes = 4
    for pass_no in range(1, max_passes + 1):
        bare_now = bare_letter_count(out or "")
        words_now = len((out or "").split())
        if bare_now >= 7000 and not _story_looks_incomplete(out or "", ""):
            break
        if not _story_looks_incomplete(out or "", reason or "") and bare_now >= 6200:
            break
        log.info(
            "folk story incomplete pass=%s bare=%s words=%s finish=%s — continuing",
            pass_no, bare_now, words_now, reason,
        )
        try:
            more, reason2 = _continue_folk_story(out or "", pass_no)
        except Exception as cont_exc:  # noqa: BLE001
            log.warning(
                "folk story continuation failed pass=%s: %s",
                pass_no, _redact_secrets(str(cont_exc))[:160],
            )
            break
        if not more:
            break
        # Avoid duplicating a long overlapping prefix
        overlap = (out or "")[-120:].strip()
        if overlap and more.startswith(overlap[:40]):
            more = more[len(overlap):].lstrip() if more.startswith(overlap) else more
        out = ((out or "").rstrip() + "\n\n" + more).strip()
        reason = f"{reason}+{reason2}"
        if bare_letter_count(out or "") >= 7800:
            log.info(
                "folk story stopped continuing at bare=%s (7800/~9min cap)",
                bare_letter_count(out or ""),
            )
            break

    cleaned = sanitize_for_speech(strip_part_labels(out or ""))
    if not cleaned or len(cleaned) < 400:
        raise SummarizeError("ناتج الحكاية قصير جداً أو فارغ")
    lines = cleaned.splitlines()
    story_title = (title or "").strip() or "نادرة"
    source_label = (collection_title or author or "").strip() or "نوادر إنجليزية"
    original_title = (title or "").strip()
    body_start = 0
    meta_keys = {
        "العنوان": "title",
        "الأصل": "original",
        "المصدر": "source",
        "الكتاب": "book",
        "المؤلف": "author",
    }
    while body_start < len(lines):
        raw_ln = lines[body_start].strip()
        if not raw_ln:
            body_start += 1
            # blank after meta block ends meta scan if we already saw title
            if body_start > 0 and story_title:
                # peek: if next is not a meta line, stop
                if body_start < len(lines):
                    peek = lines[body_start].strip()
                    if peek and not re.match(r"^(العنوان|الأصل|المصدر|الكتاب|المؤلف)\s*[:：]", peek):
                        break
                continue
            continue
        mmeta = re.match(r"^(العنوان|الأصل|المصدر|الكتاب|المؤلف)\s*[:：]\s*(.+?)\s*$", raw_ln)
        if not mmeta:
            break
        key, val = mmeta.group(1), sanitize_for_speech(mmeta.group(2).strip())
        if key == "العنوان" and val:
            story_title = val
        elif key == "الأصل" and val:
            original_title = val
        elif key in {"المصدر", "الكتاب"} and val:
            source_label = val
        elif key == "المؤلف" and val:
            if source_label and source_label != val:
                source_label = f"{source_label} — {val}"
            else:
                source_label = val
        body_start += 1
    while body_start < len(lines) and not lines[body_start].strip():
        body_start += 1
    # If the body opens by repeating the Arabic title line, drop that duplicate
    if body_start < len(lines) and story_title and lines[body_start].strip() == story_title:
        body_start += 1
        while body_start < len(lines) and not lines[body_start].strip():
            body_start += 1
    body = "\n".join(lines[body_start:]).strip() or cleaned
    body = sanitize_for_speech(strip_part_labels(body))
    body = _TASHKEEL_RE.sub("", body)
    body = body.replace("\u0640", "")  # tatweel
    # Drop moral/epilogue tails if model added them
    body = _strip_moral_tail(body)
    bare_before = bare_letter_count(body)
    if bare_before > 7800:
        body = trim_to_bare_letters(body, 7800)
        log.info(
            "folk story soft-trimmed bare %s -> %s (cap 7800 ~9min excl. punct)",
            bare_before, bare_letter_count(body),
        )
    story_title = ensure_arabic_title(story_title) if story_title else "نادرة"
    # Reject titles that are clearly the book/collection name
    bookish_ar = ("نوادر إنجليز", "طرائف إنجليز", "جمعت من مصادر", "مجموعة من مصادر", "مئة حكاية", "كتاب النوادر")
    st_low = (story_title or "")
    coll = (collection_title or "")
    if any(b in st_low for b in bookish_ar) or (coll and coll[:20] and coll[:20].lower() in st_low.lower()):
        # Prefer a character/situation title from the first body sentence
        first_sent = re.split(r"(?<=[.!?؟۔])\s+", body[:220])
        guess = (first_sent[0].strip() if first_sent else "")[:60]
        if len(guess) >= 8:
            story_title = guess
            log.warning("replaced bookish Arabic title with opener guess")
    # Ensure spoken/sent body opens with the tale title then the tale
    body_stripped = body.lstrip()
    title_line = (story_title or "").strip()
    if title_line and not body_stripped.startswith(title_line):
        body = f"{title_line}\n\n{body_stripped}"
    # Stash attribution for callers via source_label enrichment
    if original_title and original_title not in source_label:
        source_label = f"{source_label} | أصل: {original_title}" if source_label else f"أصل: {original_title}"
    parts = _split_exactly_three(body)
    safe: list[str] = []
    for p in parts:
        if len(p) <= 4000:
            safe.append(p)
            continue
        chunk = ""
        for w in p.split():
            t = (chunk + " " + w).strip()
            if len(t) > 3900 and chunk:
                safe.append(chunk)
                chunk = w
            else:
                chunk = t
        if chunk:
            safe.append(chunk)
    parts = [sanitize_for_speech(p) for p in safe if p and p.strip()]
    if len(parts) != 3:
        parts = _split_exactly_three(body)
    if not parts:
        raise SummarizeError("تعذّر تقسيم نص الحكاية")
    full = "\n\n".join(parts)
    incomplete = _story_looks_incomplete(body, reason or "")
    log.info(
        "folk story ready title=%s parts=%s words=%s bare=%s finish=%s incomplete=%s",
        story_title[:60], len(parts), len(body.split()), bare_letter_count(body), reason, incomplete,
    )
    return story_title, source_label, parts, full


TWEET_MAX_CHARS = 240


def split_into_tweets(text: str, max_chars: int = TWEET_MAX_CHARS) -> list[str]:
    """Split Arabic text into short tweet-sized chunks at sentence/word boundaries."""
    body = sanitize_for_speech(text or "").strip()
    if not body:
        return []
    if len(body) <= max_chars:
        return [body]
    sents = _sentences(body) or [body]
    tweets: list[str] = []
    buf = ""
    for sent in sents:
        sent = sent.strip()
        if not sent:
            continue
        if len(sent) > max_chars:
            if buf:
                tweets.append(buf.strip())
                buf = ""
            words = sent.split()
            chunk = ""
            for w in words:
                trial = (chunk + " " + w).strip()
                if len(trial) > max_chars and chunk:
                    tweets.append(chunk)
                    chunk = w
                else:
                    chunk = trial
            if chunk:
                tweets.append(chunk)
            continue
        trial = (buf + " " + sent).strip() if buf else sent
        if len(trial) > max_chars and buf:
            tweets.append(buf.strip())
            buf = sent
        else:
            buf = trial
    if buf.strip():
        tweets.append(buf.strip())
    return [t for t in tweets if t]


def format_tweets_file(
    tweets: list[str],
    title: str = "",
    original_title: str = "",
    collection_title: str = "",
    author: str = "",
) -> str:
    """Short tweets for the delivered .txt file (no numbering) with source attribution."""
    lines: list[str] = []
    t = (title or "").strip()
    if t:
        lines.append(t)
        lines.append("")
    orig = (original_title or "").strip()
    book = (collection_title or "").strip()
    auth = (author or "").strip()
    if orig:
        lines.append(f"الأصل: {orig}")
    if book:
        lines.append(f"الكتاب: {book}")
    if auth:
        lines.append(f"المؤلف: {auth}")
    if orig or book or auth:
        lines.append("")
    for tw in tweets or []:
        s = (tw or "").strip()
        if not s:
            continue
        # Avoid repeating the Arabic title as the first tweet
        if t and s == t:
            continue
        lines.append(s)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def tweets_to_tts_parts(tweets: list[str], target_parts: int = 3) -> list[str]:
    """Merge short tweets into a few TTS segments (keep vocalization)."""
    clean = [sanitize_for_speech(t) for t in (tweets or []) if sanitize_for_speech(t)]
    if not clean:
        return []
    if len(clean) <= target_parts:
        return clean
    total = sum(len(t) for t in clean) or 1
    goal = total / max(1, target_parts)
    parts: list[str] = []
    buf: list[str] = []
    size = 0
    for tw in clean:
        if buf and size + len(tw) >= goal * 0.9 and len(parts) < target_parts - 1:
            parts.append("\n\n".join(buf))
            buf = [tw]
            size = len(tw)
        else:
            buf.append(tw)
            size += len(tw)
    if buf:
        parts.append("\n\n".join(buf))
    return parts


def _chunk_for_vocalize(text: str, max_chars: int = 1400) -> list[str]:
    """Split plain Arabic into mid-size chunks so Gemini tashkeel does not hit MAX_TOKENS."""
    body = (text or "").strip()
    if not body:
        return []
    if len(body) <= max_chars:
        return [body]
    sents = _sentences(body) or [body]
    chunks: list[str] = []
    buf = ""
    for sent in sents:
        sent = sent.strip()
        if not sent:
            continue
        trial = (buf + " " + sent).strip() if buf else sent
        if len(trial) > max_chars and buf:
            chunks.append(buf.strip())
            buf = sent
        else:
            buf = trial
    if buf.strip():
        chunks.append(buf.strip())
    return chunks


def _parse_vocalize_chunks(cleaned: str) -> list[str]:
    cleaned = sanitize_for_speech(cleaned or "")
    if not cleaned:
        return []
    if "---" in cleaned:
        chunks = [c.strip() for c in re.split(r"\n?\s*---\s*\n?", cleaned) if c.strip()]
    else:
        chunks = [c.strip() for c in re.split(r"\n\s*\n+", cleaned) if c.strip()]
    tweets: list[str] = []
    for c in chunks:
        c = re.sub(r"^\d+[\).\-\u060c]\s*", "", c).strip()
        if not c:
            continue
        if len(c) <= TWEET_MAX_CHARS + 40:
            tweets.append(c)
        else:
            tweets.extend(split_into_tweets(c, TWEET_MAX_CHARS))
    return tweets


def _vocalize_one_chunk(plain_chunk: str, title: str, chunk_i: int, chunk_n: int) -> list[str]:
    prompt = (
        "أنت مدقق لغة عربية ومشكّل نصوص للرواية الصوتية.\n"
        f"هذه القطعة {chunk_i}/{chunk_n} من نص أطول.\n"
        "المطلوب:\n"
        "1) أعد كتابة القطعة بتشكيل عربي كامل واضح (فتحة وضمة وكسرة وتنوين وشدة وهمزة حيث يلزم) "
        "بدون تغيير المعنى أو الأحداث أو الأسماء، ودون حذف أي جملة.\n"
        "2) اقسم الناتج إلى تغريدات قصيرة متتابعة، كل تغريدة جملة أو جملتين، "
        f"وبحد أقصى حوالي {TWEET_MAX_CHARS} حرفاً للتغريدة الواحدة (مع التشكيل).\n"
        "القواعد:\n"
        "- أكمل القطعة كاملة حتى آخر كلمة؛ ممنوع الاختصار.\n"
        "- لا مقدمة ولا تعليق ولا عبرة.\n"
        "- لا ماركداون ولا نجوم ولا عناوين #.\n"
        "- افصل كل تغريدة بسطر فيه ثلاثة شرطات فقط: ---\n"
        "- لا ترقّم التغريدات.\n"
        f"عنوان مساعد: {title or '—'}\n\n"
        f"القطعة:\n{plain_chunk}"
    )
    try:
        out, reason = _gemini_generate(prompt, purpose="tashkeel")
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "vocalize chunk %s/%s failed: %s",
            chunk_i, chunk_n, _redact_secrets(str(exc))[:160],
        )
        return split_into_tweets(plain_chunk)

    cleaned = sanitize_for_speech(out or "")
    tweets = _parse_vocalize_chunks(cleaned)
    if not tweets:
        tweets = split_into_tweets(cleaned if cleaned else plain_chunk)

    # If Gemini truncated this chunk, continue from the unfinished tail.
    bare_in = bare_letter_count(plain_chunk)
    bare_out = bare_letter_count("\n".join(tweets))
    if reason == "MAX_TOKENS" or (bare_in and bare_out < int(bare_in * 0.85)):
        log.warning(
            "vocalize chunk %s/%s short/truncated bare_in=%s bare_out=%s finish=%s — local fill",
            chunk_i, chunk_n, bare_in, bare_out, reason,
        )
        # Keep what we have (vocalized), append local tweets for uncovered plain tail.
        # Approximate: if coverage low, append remaining plain via local split of whole chunk
        # only the missing portion by comparing stripped letters ratio — safest is append
        # local split of plain_chunk sentences not clearly present.
        if bare_out < int(bare_in * 0.5):
            # Too broken — use full local for this chunk
            return split_into_tweets(plain_chunk)
        # Partial: keep vocalized head, append local split of last ~uncovered fraction
        # Heuristic: take last (1 - ratio) of plain_chunk
        ratio = (bare_out / bare_in) if bare_in else 1.0
        cut = int(len(plain_chunk) * min(0.95, max(0.4, ratio)))
        # snap to sentence near cut
        tail = plain_chunk[cut:].lstrip()
        if len(tail) > 40:
            tweets.extend(split_into_tweets(tail))

    log.info(
        "vocalize chunk %s/%s tweets=%s bare_in=%s bare_out=%s finish=%s",
        chunk_i, chunk_n, len(tweets), bare_in, bare_letter_count("\n".join(tweets)), reason,
    )
    return tweets




def _cohere_api_key() -> str:
    return (os.getenv("COHERE_API_KEY") or "").strip()


def _cohere_chat(prompt: str, *, max_tokens: int = 8192, temperature: float = 0.35) -> str:
    """Call Cohere Chat API v2. Raises SummarizeError on failure."""
    import json
    import urllib.error
    import urllib.request

    api_key = _cohere_api_key()
    if not api_key:
        raise SummarizeError("COHERE_API_KEY missing")
    models = [
        (os.getenv("COHERE_REVIEW_MODEL") or "").strip() or "command-a-03-2025",
        "command-r-plus",
        "command-r",
        "command",
    ]
    # dedupe
    seen = set()
    order = []
    for m in models:
        if m and m not in seen:
            seen.add(m)
            order.append(m)
    last_err = ""
    for model in order:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            "https://api.cohere.com/v2/chat",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raw = ""
            try:
                raw = exc.read().decode("utf-8", errors="replace")[:400]
            except Exception:
                pass
            last_err = _redact_secrets(f"HTTP {exc.code}: {raw}")
            log.warning("cohere HTTP %s model=%s body=%s", exc.code, model, last_err[:200])
            continue
        except Exception as exc:  # noqa: BLE001
            last_err = _redact_secrets(str(exc))[:240]
            log.warning("cohere request failed model=%s: %s", model, last_err)
            continue
        msg = body.get("message") or {}
        content = msg.get("content") or []
        text = ""
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text += part.get("text") or ""
                elif isinstance(part, str):
                    text += part
        elif isinstance(content, str):
            text = content
        text = (text or body.get("text") or "").strip()
        if text:
            log.info("cohere ok model=%s chars=%s", model, len(text))
            return text
        last_err = "empty response"
        log.warning("cohere empty model=%s", model)
    raise SummarizeError(f"تعذّر إخراج المراجعة عبر Cohere: {last_err}")


def review_finalize_story(text: str, title: str = "", source_label: str = "") -> tuple[str, str]:
    """Cohere review pass: attractive title + professional final Arabic body.

    Returns (final_title, final_body). Does not add a code-made closing; the model must.
    """
    raw = sanitize_for_speech(text or "").strip()
    if not raw:
        return (title or "نادرة", "")
    if not _cohere_api_key():
        log.warning("COHERE_API_KEY missing; skipping review finalize")
        return ((title or "نادرة").strip(), raw)

    # Drop leading duplicate title line for cleaner review input
    body = raw
    t0 = (title or "").strip()
    if t0 and body.lstrip().startswith(t0):
        body = body.lstrip()[len(t0):].lstrip()

    prompt = (
        "أنت محرر أدبي عربي محترف لنوادر وحكايات مرحة (نساء، قضاة، متزوجون، ومواقف جريئة إن وُجدت).\n"
        "المطلوب: راجع النص التالي، حسّن سلاسته وإيقاعه دون تغيير المعنى أو الأحداث أو الشخصيات، "
        "واختر عنواناً جذاباً قصيراً يسمّي الموقف أو الشخصية (ممنوع اسم الكتاب أو عنوان عام).\n"
        "أمانة: لا تحذف ولا تليّن المشاهد الجريئة أو الجنسية إن وُجدت، ولا تختلق مشاهد جديدة.\n"
        "أخرج إخراجاً نهائياً جاهزاً للقراءة والاستماع:\n"
        "- العربية الفصحى الأدبية الواضحة، بدون تشكيل.\n"
        "- بدون ماركداون وبدون نجوم وبدون عناوين #.\n"
        "- بدون عبرة وعظية ختامية؛ لكن اختم بجملة قصيرة من عندك توحي أن القصة انتهت فعلاً "
        "(مثل «وهنا تنتهي النادرة.») بعد اكتمال الأحداث.\n"
        "- لا تلخّص ولا تحذف مشاهد مهمة؛ أبقِ الطول قريباً من الأصل.\n"
        "- لا لاتينية في الجسم.\n"
        "صيغة الناتج حرفياً:\n"
        "العنوان: (عنوان جذاب)\n"
        "ثم سطر فارغ\n"
        "ثم نص القصة الكامل من البداية للنهاية مع الخاتمة القصيرة.\n\n"
        f"عنوان سابق مقترح: {t0 or '—'}\n"
        f"المصدر: {(source_label or '').strip() or '—'}\n\n"
        f"النص:\n{body}"
    )
    try:
        out = _cohere_chat(prompt, max_tokens=10000, temperature=0.35)
    except Exception as exc:  # noqa: BLE001
        log.warning("review_finalize_story failed: %s", _redact_secrets(str(exc))[:200])
        return (t0 or "نادرة", raw)

    cleaned = sanitize_for_speech(out)
    lines = cleaned.splitlines()
    final_title = t0 or "نادرة"
    start = 0
    if lines:
        m = TITLE_LINE_RE.match(lines[0].strip())
        if m:
            parsed = sanitize_for_speech(m.group(1).strip())
            if parsed and len(parsed) >= 4:
                final_title = parsed
            start = 1
            while start < len(lines) and not lines[start].strip():
                start += 1
        elif len(lines[0].strip()) <= 80 and not lines[0].strip().endswith((".", "۔", "!", "؟")):
            # bare title line
            cand = lines[0].strip()
            if cand and len(cand) >= 4:
                final_title = cand
                start = 1
                while start < len(lines) and not lines[start].strip():
                    start += 1
    final_body = "\n".join(lines[start:]).strip() or cleaned
    if final_title and final_body.lstrip().startswith(final_title):
        final_body = final_body.lstrip()[len(final_title):].lstrip()
    # Re-attach title at top for downstream consistency
    if final_title and not final_body.lstrip().startswith(final_title):
        final_body = f"{final_title}\n\n{final_body}"
    final_title = ensure_arabic_title(final_title) if final_title else "نادرة"
    log.info(
        "cohere review ready title=%s chars=%s",
        final_title[:60], len(final_body),
    )
    return final_title, final_body


def vocalize_and_split_tweets(text: str, title: str = "") -> list[str]:
    """Gemini: full Arabic tashkeel + split into short tweets. Chunked to avoid truncation."""
    raw = sanitize_for_speech(text or "").strip()
    if not raw:
        return []
    plain = _TASHKEEL_RE.sub("", raw).replace("\u0640", "")
    if not (_gemini_api_key("tashkeel") or _gemini_api_key("story")):
        return split_into_tweets(plain)

    pieces = _chunk_for_vocalize(plain, max_chars=1400)
    if not pieces:
        return split_into_tweets(plain)

    all_tweets: list[str] = []
    for i, piece in enumerate(pieces, 1):
        all_tweets.extend(_vocalize_one_chunk(piece, title, i, len(pieces)))

    if not all_tweets:
        all_tweets = split_into_tweets(plain)

    joined = "\n".join(all_tweets)
    bare_in = bare_letter_count(plain)
    bare_out = bare_letter_count(joined)
    marks = len(_TASHKEEL_RE.findall(joined))
    log.info(
        "vocalize done chunks=%s tweets=%s chars=%s bare_in=%s bare_out=%s tashkeel_marks=%s",
        len(pieces), len(all_tweets), len(joined), bare_in, bare_out, marks,
    )
    # Final safety: if still severely short, fall back to full local split of original
    if bare_in and bare_out < int(bare_in * 0.75):
        log.warning(
            "vocalize coverage too low bare_out/bare_in=%s/%s — fallback local split",
            bare_out, bare_in,
        )
        return split_into_tweets(plain)
    return all_tweets

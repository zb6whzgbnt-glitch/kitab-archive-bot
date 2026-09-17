"""Tiny unit tests: Arabic titles, no asterisks, no «الجزء» headers."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import summarize
import tts


def _body() -> str:
    # ~3k Arabic-ish chars so split_parts yields multiple chunks.
    para = (
        "هذه الرواية تدور حول أسرة تعيش في مدينة ساحلية، وتتتابع فيها الأحداث "
        "بين الحب والخيانة والعمل اليومي حتى تتضح مصائر الشخصيات واحدة تلو الأخرى. "
    )
    return para * 40


def main() -> int:
    fake = (
        "العنوان: ألف ليلة وليلة\n"
        "المؤلف: مؤلف مجهول\n\n"
        "«الجزء 1 من 6 — Alf Layla wa Layla, Cairo ألف ليلة وليلة طبعة القاهرة»\n\n"
        + _body()
        + "\nيتبع\n"
    )
    title, author, parts = summarize.parts_from_summary(
        fake, fallback_title="FALLBACK", fallback_author="FALLBACK_AUTHOR"
    )
    assert title == "ألف ليلة وليلة", title
    assert summarize.title_is_arabic(title), title
    assert author == "مؤلف مجهول", author
    assert parts, "expected parts"
    joined = "\n".join(parts)
    assert "الجزء" not in joined, joined[:200]
    assert "يتبع" not in joined
    assert "*" not in joined

    formatted = summarize.format_part(parts[0], 1, len(parts), title)
    assert not formatted.startswith("«الجزء"), formatted[:80]
    assert "الجزء" not in formatted[:40]
    assert "*" not in formatted

    msgs = summarize.outgoing_text_messages(title, author, parts)
    assert msgs[0].startswith("ألف ليلة وليلة\nمؤلف مجهول\n\n"), msgs[0][:80]
    for later in msgs[1:]:
        assert "الجزء" not in later
        assert not later.startswith("ألف ليلة")
        assert "يتبع" not in later
        assert "*" not in later
    for m in msgs:
        assert "*" not in m

    # Missing author from Gemini and metadata.
    title2, author2, parts2 = summarize.parts_from_summary(
        _body(), fallback_title="كتاب بلا مؤلف", fallback_author=""
    )
    assert title2 == "كتاب بلا مؤلف"
    assert author2 == "مؤلف غير مذكور"
    msgs2 = summarize.outgoing_text_messages(title2, author2, parts2)
    assert msgs2[0].startswith("كتاب بلا مؤلف\nمؤلف غير مذكور\n\n")

    # Arabic-origin titles stay as-is (do not reverse-translate).
    assert summarize.title_is_arabic("كليلة ودمنة")
    assert summarize.title_is_arabic("ألف ليلة وليلة")
    assert not summarize.title_is_latin_heavy("كليلة ودمنة")
    assert summarize.title_is_latin_heavy("Treasure Island")
    assert summarize.title_is_latin_heavy("Alf Layla wa Layla")
    assert summarize.ensure_arabic_title("كليلة ودمنة") == "كليلة ودمنة"

    # Latin-heavy Gemini title is translated via one extra call (mocked).
    original_translate = summarize._gemini_translate_title
    summarize._gemini_translate_title = lambda t: "جزيرة الكنز"
    try:
        fake_en = (
            "العنوان: Treasure Island\n"
            "المؤلف: روبرت لويس ستيفنسون\n\n"
            + _body()
        )
        t_en, a_en, p_en = summarize.parts_from_summary(
            fake_en, fallback_title="Treasure Island", fallback_author="Stevenson"
        )
        assert t_en == "جزيرة الكنز", t_en
        assert summarize.title_is_arabic(t_en)
        assert a_en == "روبرت لويس ستيفنسون"
        assert p_en
    finally:
        summarize._gemini_translate_title = original_translate

    # Mixed Arabic+Latin parenthetical must not be rejected as latin-heavy.
    mixed = "هكذا تكلم زرادشت (Thus Spoke Zarathustra)"
    stripped = summarize.strip_latin_title_extras(mixed)
    assert stripped == "هكذا تكلم زرادشت", stripped
    assert summarize.title_is_arabic(stripped)
    assert summarize.ensure_arabic_title(mixed) == "هكذا تكلم زرادشت"
    # 57-char Gemini reply (العنوان: + Arabic + Latin paren) must parse as Arabic.
    gemini57 = "العنوان: هكذا تكلم زرادشت (Thus Spoke Zarathustra)"
    cand57 = summarize._extract_title_candidate(gemini57)
    accepted = summarize._accept_arabic_title(cand57, "unit", "Thus Spoke Zarathustra")
    assert accepted == "هكذا تكلم زرادشت", (len(gemini57), cand57, accepted)

    # If translation fails, do not keep English.
    summarize._gemini_translate_title = lambda t: None
    summarize._gemini_translate_title_strict = lambda t: None
    try:
        raised = False
        try:
            summarize.ensure_arabic_title("Treasure Island")
        except summarize.SummarizeError:
            raised = True
        assert raised, "expected SummarizeError when title stays Latin"
    finally:
        summarize._gemini_translate_title = original_translate

    # Gemini prompt: translate English titles; do not keep Latin on line 1.
    prompt = summarize._gemini_prompt(
        "Once upon a time", "Treasure Island", "eng", author="Robert Louis Stevenson"
    )
    assert "يحتفظ بالعنوان الإنجليزي" not in prompt
    assert "ترجم العنوان" in prompt
    assert "روبرت لويس ستيفنسون" in prompt
    assert "FORBIDDEN" in prompt
    assert "*" in prompt  # instruction names the forbidden character
    ara_prompt = summarize._gemini_prompt(
        "كان يا ما كان", "كليلة ودمنة", "ara", author="ابن المقفع"
    )
    assert "أبقِ العنوان العربي" in ara_prompt
    assert "ترجم العنوان" not in ara_prompt

    # Sanitizer: zero asterisks; markdown stripped; Arabic punctuation kept.
    dirty = "*عنوان* و **كلمة** و __مهم__ و _مائل_ و `رمز`.\n# رأس\nنص، واضح؟"
    cleaned = summarize.sanitize_for_speech(dirty)
    assert "*" not in cleaned, cleaned
    assert "عنوان" in cleaned
    assert "كلمة" in cleaned
    assert "مهم" in cleaned
    assert "مائل" in cleaned
    assert "رمز" in cleaned
    assert "رأس" in cleaned
    assert "#" not in cleaned
    assert "`" not in cleaned
    assert "،" in cleaned
    assert "؟" in cleaned
    assert "_" not in cleaned

    dirty_parts_src = "*عنوان* يبدأ النص و **كلمة** مهمة في السرد. " + _body()
    split = summarize.split_parts(dirty_parts_src)
    assert split
    for p in split:
        assert "*" not in p, p[:80]

    msgs_dirty = summarize.outgoing_text_messages(
        "*عنوان*", "**كلمة**", ["**كلمة** في النص، واضح؟"]
    )
    for m in msgs_dirty:
        assert "*" not in m, m
        assert "_" not in m
    assert "عنوان" in msgs_dirty[0]
    assert "كلمة" in msgs_dirty[0]
    assert "،" in msgs_dirty[0]
    assert "؟" in msgs_dirty[0]

    # TTS input path is clean (sanitize before synthesize).
    script = tts.speech_text(
        ["*عنوان* في البداية و **كلمة** بعد ذلك. " + _body()[:400]],
        title="*كنز*",
        author="**مؤلف**",
    )
    assert "*" not in script, script[:120]
    assert "_" not in script
    assert "عنوان" in script
    assert "كلمة" in script
    assert "كنز" in script
    assert "مؤلف" in script

    # 503 must retry same model then the next, not raise immediately.
    original_post = summarize._gemini_post
    original_key = summarize._gemini_api_key
    original_sleep = summarize.time.sleep
    original_backoffs = summarize.RETRY_BACKOFFS
    original_model = summarize._gemini_model
    summarize._gemini_model = None
    summarize.RETRY_BACKOFFS = (0.0, 0.0)
    summarize.time.sleep = lambda s: None
    summarize._gemini_api_key = lambda: "dummy-key"

    class FakeResp:
        def __init__(self, status, payload=None, body=""):
            self.status_code = status
            self._payload = payload or {}
            self.text = body or '{"error":{"code":503,"status":"UNAVAILABLE"}}'

        def json(self):
            return self._payload

    posts = []

    def fake_post(model, prompt, api_key):
        posts.append(model)
        # First model: 3 retryable 503s, then next model succeeds.
        if len(posts) <= 3:
            return FakeResp(503, body='{"error":{"status":"UNAVAILABLE"}}')
        arabic = "هذا نص عربي طويل بما يكفي لاجتياز الفحص. " * 8
        return FakeResp(
            200,
            payload={"candidates": [{"content": {"parts": [{"text": arabic}]}, "finishReason": "STOP"}]},
            body="ok",
        )

    summarize._gemini_post = fake_post
    try:
        out, reason = summarize._gemini_generate("prompt")
        assert out and "عربي" in out
        assert reason == "STOP"
        assert len(posts) == 4, posts
        assert posts[:3] == ["gemini-3.6-flash"] * 3, posts
        assert posts[3] == "gemini-2.5-flash", posts
    finally:
        summarize._gemini_post = original_post
        summarize._gemini_api_key = original_key
        summarize.time.sleep = original_sleep
        summarize.RETRY_BACKOFFS = original_backoffs
        summarize._gemini_model = original_model

    print("test_parts ok")
    print(f"parts={len(parts)} first_starts_with_title=yes later_have_part_header=no arabic_title=yes sanitizer_strips_star=yes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

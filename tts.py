"""Generate MP3 audio via Gemini TTS only (fixed narrator voice)."""

from __future__ import annotations

import os

import asyncio
import base64
import logging
import re
import subprocess
import tempfile
import wave
from pathlib import Path

import httpx

log = logging.getLogger("kitab.tts")

ROOT = Path(__file__).resolve().parent
AUDIO_DIR = ROOT / "data" / "audio"

# Gemini TTS (male narrator voices)
GEMINI_TTS_MODELS = (
    "gemini-2.5-flash-preview-tts",
    "gemini-2.5-pro-preview-tts",
    "gemini-3.1-flash-tts-preview",
)
GEMINI_TTS_VOICE = "Charon"  # fixed Gemini narrator — never switch
GEMINI_TTS_VOICES = (GEMINI_TTS_VOICE,)
GEMINI_TTS_STYLE = (
    "Narrator: Charon only. Same fixed voice and tone on every chunk. "
    "اقرأ بأسلوب راوٍ أدبي راقٍ ثابت النبرة، صوت واضح هادئ متزن، "
    "سرد فصيح أنيق بدون مبالغة انفعالية، بنفس الإيقاع عبر كل المقاطع."
)
GEMINI_TTS_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
# Soft char budget per TTS request (API often limits ~2000–4000).
TTS_CHUNK_CHARS = 5200  # ~3 min Arabic with tashkeel in one Gemini TTS call
PCM_RATE = 24000
PCM_CHANNELS = 1
PCM_SAMPLE_WIDTH = 2  # 16-bit

# edge-tts fallback (male)
EDGE_FALLBACK_VOICE = "ar-SA-HamedNeural"
EDGE_SPEECH_RATE = "+0%"

# Fish Audio TTS (primary for story delivery)
FISH_TTS_ENDPOINT = "https://api.fish.audio/v1/tts"
FISH_TTS_MODEL = "s2.1-pro-free"
FISH_VOICE_MALE = "29a87b9798c649f8926ed514b078a286"  # راوي قصص عربي
FISH_VOICE_FEMALE = "e0ff395a389f447ea62641274a1832b5"  # راوية قصص
FISH_CHUNK_CHARS = 2200
FISH_MAX_CHUNKS = 8
FISH_HTTP_TIMEOUT = 90.0

_HARAKA = re.compile(r"[\u064B-\u0652]")
_AR_TOKEN = re.compile(r"[\u0600-\u06FF]+")

_EXACT = {
    "جدا": "جِدّاً",
    "أيضا": "أَيضاً",
    "ايضا": "أَيضاً",
    "حقا": "حَقّاً",
    "دائما": "دائِماً",
    "أبدا": "أَبَداً",
    "ابدا": "أَبَداً",
    "مثلا": "مَثَلاً",
    "خصوصا": "خُصوصاً",
    "غالبا": "غالِباً",
    "تقريبا": "تَقريباً",
    "طبعا": "طَبعاً",
    "قطعا": "قَطعاً",
    "أساسا": "أَساساً",
    "نسبيا": "نِسبياً",
    "إذن": "إِذَنْ",
    "اذا": "إِذا",
    "إذا": "إِذا",
    "لكن": "لَكِنَّ",
    "لكنه": "لَكِنَّهُ",
    "إنها": "إِنَّها",
    "إنه": "إِنَّهُ",
    "إنهم": "إِنَّهُم",
    "لأنه": "لِأَنَّهُ",
    "لأنها": "لِأَنَّها",
    "لانه": "لِأَنَّهُ",
    "لانها": "لِأَنَّها",
    "يقال": "يُقال",
    "تعد": "تُعَدّ",
    "يعد": "يُعَدّ",
}


def selective_tashkeel(text: str) -> str:
    """Light post-pass diacritics when Gemini text is under-vowelled."""
    if not text:
        return text
    tokens = re.findall(r"\S+|\s+", text)

    def bare(w: str) -> str:
        return _HARAKA.sub("", w)

    out: list[str] = []
    prev_bare = ""
    for tok in tokens:
        if tok.isspace() or not _AR_TOKEN.fullmatch(_HARAKA.sub("", tok)):
            out.append(tok)
            if not tok.isspace():
                prev_bare = bare(tok)
            continue
        b = bare(tok)
        if _HARAKA.search(tok):
            out.append(tok)
            prev_bare = b
            continue
        prefix = ""
        core = b
        if len(core) > 2 and core[0] in "وفبكل":
            prefix, core = core[0], core[1:]
        if core in _EXACT:
            out.append(prefix + _EXACT[core])
            prev_bare = b
            continue
        verbish = prev_bare in {
            "قال", "قالت", "يقول", "يمكن", "يجب", "ينبغي", "يريد", "أراد",
            "نرى", "نعلم", "ظن", "يعتقد", "يفترض", "أكد", "أوضح",
        }
        if b in {"إن", "ان"} and not verbish:
            out.append("إِنَّ")
        elif b in {"أن"} or (b == "ان" and verbish):
            out.append("أَنْ")
        else:
            out.append(tok)
        prev_bare = b
    return "".join(out)


def _safe_name(stem: str) -> str:
    stem = re.sub(r"[^\w\u0600-\u06FF.-]+", "_", stem, flags=re.UNICODE)
    return (stem.strip("._") or "part")[:80]


def _run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


def _chunk_text(text: str, limit: int = TTS_CHUNK_CHARS) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining.strip())
            break
        cut = remaining.rfind("\n", 0, limit)
        if cut < limit // 3:
            cut = remaining.rfind(" ", 0, limit)
        if cut < limit // 3:
            cut = limit
        piece = remaining[:cut].strip()
        if piece:
            chunks.append(piece)
        remaining = remaining[cut:].strip()
    return chunks


def _parse_mime_rate(mime: str) -> int:
    m = re.search(r"rate\s*=\s*(\d+)", mime or "", re.I)
    if m:
        return int(m.group(1))
    return PCM_RATE


def _pcm_to_wav_bytes(pcm: bytes, rate: int = PCM_RATE) -> bytes:
    import io

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(PCM_CHANNELS)
        wf.setsampwidth(PCM_SAMPLE_WIDTH)
        wf.setframerate(rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def _ffmpeg_to_mp3(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".partial.mp3")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-codec:a", "libmp3lame", "-q:a", "4",
        str(tmp),
    ]
    subprocess.run(cmd, check=True)
    tmp.replace(dest)


def _concat_mp3s(paths: list[Path], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if len(paths) == 1:
        if paths[0].resolve() != dest.resolve():
            dest.write_bytes(paths[0].read_bytes())
        return
    with tempfile.TemporaryDirectory(prefix="folk_tts_") as td:
        td_path = Path(td)
        list_file = td_path / "list.txt"
        lines = []
        for i, p in enumerate(paths):
            # Re-encode each to consistent mp3 then concat demuxer.
            fixed = td_path / f"seg_{i:03d}.mp3"
            _ffmpeg_to_mp3(p, fixed)
            lines.append(f"file '{fixed}'")
        list_file.write_text("\n".join(lines), encoding="utf-8")
        tmp = dest.with_suffix(".partial.mp3")
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(list_file),
            "-c", "copy", str(tmp),
        ]
        subprocess.run(cmd, check=True)
        tmp.replace(dest)


def _extract_inline_audio(data: dict) -> tuple[bytes, str]:
    cands = data.get("candidates") or []
    if not cands:
        feedback = data.get("promptFeedback") or {}
        raise RuntimeError(f"no TTS candidates: {feedback.get('blockReason') or 'empty'}")
    parts = ((cands[0].get("content") or {}).get("parts") or [])
    for part in parts:
        inline = part.get("inlineData") or part.get("inline_data") or {}
        b64 = inline.get("data")
        mime = inline.get("mimeType") or inline.get("mime_type") or ""
        if b64:
            return base64.b64decode(b64), mime
    raise RuntimeError("TTS response had no inline audio data")


def _gemini_tts_once(text: str, model: str, voice: str, api_key: str) -> tuple[bytes, str]:
    from summarize import _pace_gemini_calls, _redact_secrets

    prompt = f"{GEMINI_TTS_STYLE}\n\n{text}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {"voiceName": voice}
                }
            },
        },
    }
    url = GEMINI_TTS_ENDPOINT.format(model=model)
    _pace_gemini_calls()
    timeout = httpx.Timeout(connect=30.0, read=300.0, write=60.0, pool=30.0)
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(
            url,
            headers={
                "x-goog-api-key": api_key,
                "Content-Type": "application/json",
            },
            json=payload,
        )
    if resp.status_code == 404:
        raise FileNotFoundError(f"TTS model 404: {model}")
    if resp.status_code >= 400:
        snip = _redact_secrets((resp.text or "")[:240])
        raise RuntimeError(f"TTS HTTP {resp.status_code}: {snip}")
    return _extract_inline_audio(resp.json())


def _write_pcm_segment(pcm: bytes, mime: str, dest_wav: Path) -> Path:
    rate = _parse_mime_rate(mime)
    # Some responses are already WAV/MP3 wrapped.
    low = (mime or "").lower()
    if "mpeg" in low or "mp3" in low:
        dest_mp3 = dest_wav.with_suffix(".mp3")
        dest_mp3.write_bytes(pcm)
        return dest_mp3
    if "wav" in low:
        dest_wav.write_bytes(pcm)
        return dest_wav
    dest_wav.write_bytes(_pcm_to_wav_bytes(pcm, rate=rate))
    return dest_wav


def gemini_text_to_mp3(text: str, dest: Path) -> Path:
    """Primary path: Gemini TTS → PCM/WAV → MP3 (chunk + concat if needed)."""
    from summarize import _gemini_api_key, _rotate_gemini_key, _redact_secrets

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    chunks = _chunk_text(text, TTS_CHUNK_CHARS)
    if not chunks:
        raise RuntimeError("empty TTS text")

    api_key = _gemini_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY missing")

    models_ok: list[str] = []
    models_404: list[str] = []
    last_err: Exception | None = None
    working_model: str | None = None
    working_voice: str = GEMINI_TTS_VOICE

    with tempfile.TemporaryDirectory(prefix="gemini_tts_") as td:
        td_path = Path(td)
        seg_paths: list[Path] = []
        for ci, chunk in enumerate(chunks):
            pcm = None
            mime = ""
            tried_keys = 0
            while tried_keys < 3:
                order = list(GEMINI_TTS_MODELS)
                if working_model and working_model in order:
                    order = [working_model] + [m for m in order if m != working_model]
                voices = list(GEMINI_TTS_VOICES)
                if working_voice and working_voice in voices:
                    voices = [working_voice] + [v for v in voices if v != working_voice]
                got = False
                for model in order:
                    if model in models_404:
                        continue
                    for voice in voices:
                        try:
                            api_key = _gemini_api_key()
                            pcm, mime = _gemini_tts_once(chunk, model, voice, api_key)
                            working_model = model
                            working_voice = voice
                            if model not in models_ok:
                                models_ok.append(model)
                            got = True
                            log.info(
                                "gemini tts ok model=%s voice=%s chunk=%s/%s pcm=%s",
                                model, voice, ci + 1, len(chunks), len(pcm),
                            )
                            break
                        except FileNotFoundError:
                            models_404.append(model)
                            log.warning("gemini tts model 404 skip=%s", model)
                            continue
                        except Exception as exc:  # noqa: BLE001
                            last_err = exc
                            msg = _redact_secrets(str(exc))[:200]
                            log.warning(
                                "gemini tts fail model=%s voice=%s: %s",
                                model, voice, msg,
                            )
                            if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc).upper():
                                if _rotate_gemini_key("tts 429"):
                                    break  # break voice loop to retry with new key
                            continue
                    if got:
                        break
                if got:
                    break
                tried_keys += 1
                if not _rotate_gemini_key("tts retry"):
                    break
            if not pcm:
                raise RuntimeError(
                    f"Gemini TTS failed all models; 404={models_404}; last={last_err}"
                ) from last_err
            raw_path = td_path / f"raw_{ci:03d}"
            written = _write_pcm_segment(pcm, mime, raw_path.with_suffix(".wav"))
            seg_mp3 = td_path / f"seg_{ci:03d}.mp3"
            if written.suffix.lower() == ".mp3":
                seg_mp3.write_bytes(written.read_bytes())
            else:
                _ffmpeg_to_mp3(written, seg_mp3)
            seg_paths.append(seg_mp3)
        _concat_mp3s(seg_paths, dest)

    if models_404:
        log.info("gemini tts skipped 404 models: %s", ",".join(models_404))
    if not (dest.exists() and dest.stat().st_size > 500):
        raise RuntimeError("Gemini TTS produced empty MP3")
    log.info(
        "gemini tts done bytes=%s model=%s voice=%s path=%s",
        dest.stat().st_size, working_model, working_voice, dest,
    )
    return dest


async def _edge_synthesize(text: str, dest: Path, voice: str) -> None:
    import edge_tts

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".partial.mp3")
    communicate = edge_tts.Communicate(text, voice, rate=EDGE_SPEECH_RATE)
    await communicate.save(str(tmp))
    tmp.replace(dest)


def edge_text_to_mp3(text: str, dest: Path, voice: str | None = None) -> Path:
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    v = voice or EDGE_FALLBACK_VOICE
    _run_async(_edge_synthesize(text, dest_path, v))
    if not (dest_path.exists() and dest_path.stat().st_size > 500):
        raise RuntimeError("edge-tts produced an empty file")
    log.info("edge-tts ok voice=%s bytes=%s path=%s", v, dest_path.stat().st_size, dest_path)
    return dest_path



def _fish_api_key() -> str:
    """Load Fish Audio API key from env (Railway) or local box-secrets card."""
    import json
    import os as _os
    from pathlib import Path as _P

    v = (_os.getenv("FISH_AUDIO_API_KEY") or _os.getenv("FISH_API_KEY") or "").strip()
    if v:
        return v
    for p in (
        _P("/home/box/agent-data/box-secrets.json"),
        _P("/home/box/sand-data/box-secrets.json"),
        ROOT.parent / "box-secrets.json",
    ):
        try:
            if not p.exists():
                continue
            card = (json.loads(p.read_text(encoding="utf-8")).get("card") or {})
            for name in ("FISH_AUDIO_API_KEY", "FISH_API_KEY"):
                vv = str(card.get(name) or "").strip()
                if vv:
                    return vv
        except Exception:
            continue
    return ""


def fish_reference_id(gender: str | None = None) -> str:
    g = (gender or "male").strip().lower()
    if g.startswith("f"):
        return FISH_VOICE_FEMALE
    return FISH_VOICE_MALE


def fish_text_to_mp3(
    text: str,
    dest: Path,
    gender: str | None = None,
    reference_id: str | None = None,
) -> Path:
    """Synthesize MP3 via Fish Audio; raises on failure."""
    import httpx

    key = _fish_api_key()
    if not key:
        raise RuntimeError("FISH_AUDIO_API_KEY missing")
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    ref = (reference_id or fish_reference_id(gender)).strip()
    raw = (text or "").strip()
    if not raw:
        raise RuntimeError("empty text for Fish TTS")

    chunks = _chunk_text(raw, FISH_CHUNK_CHARS)
    if not chunks:
        raise RuntimeError("no Fish TTS chunks")
    if len(chunks) > FISH_MAX_CHUNKS:
        # Prefer fewer longer calls; hard-cap to avoid hanging the bot
        merged: list[str] = []
        cur = ""
        target = max(1, (len(chunks) + FISH_MAX_CHUNKS - 1) // FISH_MAX_CHUNKS)
        for i, ch in enumerate(chunks):
            cur = (cur + " " + ch).strip() if cur else ch
            if (i + 1) % target == 0:
                merged.append(cur)
                cur = ""
        if cur:
            merged.append(cur)
        chunks = merged[:FISH_MAX_CHUNKS]
        log.warning("fish tts merged to %s chunks (cap=%s)", len(chunks), FISH_MAX_CHUNKS)

    pcm_parts: list[bytes] = []
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "model": FISH_TTS_MODEL,
    }
    with httpx.Client(timeout=FISH_HTTP_TIMEOUT) as client:
        for i, chunk in enumerate(chunks):
            resp = client.post(
                FISH_TTS_ENDPOINT,
                headers=headers,
                json={"text": chunk, "reference_id": ref, "format": "mp3"},
            )
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"Fish TTS HTTP {resp.status_code}: {(resp.text or '')[:180]}"
                )
            if len(resp.content) < 500:
                raise RuntimeError("Fish TTS returned empty audio")
            pcm_parts.append(resp.content)
            log.info(
                "fish tts ok gender=%s ref=%s chunk=%s/%s bytes=%s",
                gender or "male",
                ref[:8],
                i + 1,
                len(chunks),
                len(resp.content),
            )

    if len(pcm_parts) == 1:
        dest_path.write_bytes(pcm_parts[0])
    else:
        # Concatenate MP3 segments via ffmpeg
        with tempfile.TemporaryDirectory(prefix="fish_tts_") as td:
            td_path = Path(td)
            files: list[Path] = []
            for i, blob in enumerate(pcm_parts):
                fp = td_path / f"p{i:02d}.mp3"
                fp.write_bytes(blob)
                files.append(fp)
            _concat_mp3s(files, dest_path)
    if not (dest_path.exists() and dest_path.stat().st_size > 500):
        raise RuntimeError("Fish TTS produced empty file")
    return dest_path


def text_to_mp3(
    text: str,
    dest: Path | str,
    voice: str | None = None,
    engine: str = "gemini",
    gender: str | None = None,
) -> Path:
    """Write `text` as MP3 via Fish / Gemini / Edge."""
    from summarize import sanitize_for_speech

    engine = (engine or "gemini").strip().lower()
    if engine in {"edge", "microsoft", "edge-tts", "ms"}:
        engine = "edge"
    elif engine in {"fish", "fishaudio", "fish-audio"}:
        engine = "fish"
    elif engine not in {"gemini", "fish", "edge"}:
        engine = "gemini"

    text = selective_tashkeel(sanitize_for_speech(text or ""))
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if engine == "fish":
        out = fish_text_to_mp3(text, dest_path, gender=gender)
        text_to_mp3._last_backend = f"fish:{(gender or 'male')}"  # type: ignore[attr-defined]
        return out

    if engine == "edge":
        v = voice or EDGE_FALLBACK_VOICE
        out = edge_text_to_mp3(text, dest_path, voice=v)
        text_to_mp3._last_backend = f"edge:{v}"  # type: ignore[attr-defined]
        return out

    # Gemini Charon
    out = gemini_text_to_mp3(text, dest_path)
    text_to_mp3._last_backend = f"gemini:{GEMINI_TTS_VOICE}"  # type: ignore[attr-defined]
    return out


def part_mp3_path(identifier: str, installment_index: int, part_index: int) -> Path:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    ident = _safe_name(identifier)
    return AUDIO_DIR / f"{ident}_i{installment_index:03d}_p{part_index:02d}.mp3"


def combined_mp3_path(identifier: str, slot: str, ymd: str) -> Path:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    ident = _safe_name(identifier)
    return AUDIO_DIR / f"{ymd}_{slot}_{ident}.mp3"


def story_mp3_path(identifier: str, story_index: int, stamp: str) -> Path:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    ident = _safe_name(identifier)
    return AUDIO_DIR / f"{stamp}_folk_{ident}_s{story_index:04d}.mp3"


def speech_text(
    parts: list[str],
    title: str = "",
    author: str = "",
) -> str:
    """Join title and parts into the string that TTS will speak."""
    from summarize import sanitize_for_speech, strip_part_labels

    def _clean(value: str) -> str:
        return sanitize_for_speech(strip_part_labels(value or ""))

    chunks: list[str] = []
    t = _clean(title)
    if t:
        chunks.append(t)
    for p in parts:
        piece = _clean(p)
        if piece:
            chunks.append(piece)
    return sanitize_for_speech("\n\n".join(chunks))


def parts_to_mp3(
    parts: list[str],
    dest: Path | str,
    voice: str | None = None,
    title: str = "",
    author: str = "",
    engine: str = "gemini",
    gender: str | None = None,
) -> Path:
    """Speak story as one MP3: each text part = one TTS segment (aim 3×~3 min)."""
    from summarize import sanitize_for_speech, strip_part_labels

    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    def _clean(value: str) -> str:
        return selective_tashkeel(sanitize_for_speech(strip_part_labels(value or "")))

    segments: list[str] = []
    t = _clean(title)
    cleaned_parts = [_clean(p) for p in parts if _clean(p)]
    if not cleaned_parts:
        raise RuntimeError("no story text for TTS")
    # Prepend title to first segment (no separate intro clip).
    if t:
        cleaned_parts[0] = f"{t}.\n\n{cleaned_parts[0]}"
    segments = cleaned_parts[:3] if len(cleaned_parts) >= 3 else cleaned_parts

    # One Gemini/edge call per segment when possible, then concat.
    with tempfile.TemporaryDirectory(prefix="folk_parts_tts_") as td:
        td_path = Path(td)
        seg_files: list[Path] = []
        backends: list[str] = []
        for i, seg in enumerate(segments):
            out_i = td_path / f"part_{i:02d}.mp3"
            text_to_mp3(seg, out_i, voice=voice, engine=engine, gender=gender)
            backends.append(getattr(text_to_mp3, "_last_backend", "?"))
            seg_files.append(out_i)
        if len(seg_files) == 1:
            dest_path.write_bytes(seg_files[0].read_bytes())
        else:
            _concat_mp3s(seg_files, dest_path)
        text_to_mp3._last_backend = "+".join(backends)  # type: ignore[attr-defined]
        log.info(
            "parts_to_mp3 segments=%s backends=%s bytes=%s",
            len(seg_files), backends, dest_path.stat().st_size if dest_path.exists() else 0,
        )
    return dest_path


def last_tts_backend() -> str:
    """Hint for reporting; set by text_to_mp3 callers via module attr."""
    return getattr(text_to_mp3, "_last_backend", "unknown")

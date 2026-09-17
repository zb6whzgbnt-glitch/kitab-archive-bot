"""Static 1920x1080 Arabic title card + MP3 -> MP4 (no extra motion)."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont, features

from summarize import sanitize_for_speech

log = logging.getLogger("kitab.video")

ROOT = Path(__file__).resolve().parent
FONTS_DIR = ROOT / "fonts"
VIDEO_DIR = ROOT / "data" / "video"
FFMPEG = "/usr/bin/ffmpeg"

W, H = 1920, 1080

# Deep navy / charcoal + cream / gold
BG_TOP = (11, 18, 32)
BG_BOT = (18, 18, 26)
CREAM = (244, 235, 208)
GOLD = (212, 175, 55)
GOLD_SOFT = (201, 168, 96)
RULE = (184, 149, 58)
FRAME = (168, 138, 62)

_AR_FONT_PATH: Path | None = None
_LAT_FONT_PATH: Path | None = None
# Pillow with libraqm already does Arabic joining + bidi; pre-shaping would reverse it.
_HAS_RAQM = bool(features.check("raqm"))
_RTL_KW: dict = {"direction": "rtl", "language": "ar"} if _HAS_RAQM else {}
_NOTDEF_SIG: dict[tuple[str, int], bytes] = {}

_ARABIC_RANGES = (
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
)


def _is_arabic_char(ch: str) -> bool:
    o = ord(ch)
    return any(a <= o <= b for a, b in _ARABIC_RANGES)


def _pick_existing(candidates: list[Path]) -> Path | None:
    for p in candidates:
        if p.is_file():
            return p
    return None


def _pick_arabic_font() -> Path:
    global _AR_FONT_PATH
    if _AR_FONT_PATH and _AR_FONT_PATH.is_file():
        return _AR_FONT_PATH
    found = _pick_existing(
        [
            FONTS_DIR / "NotoNaskhArabic-Bold.ttf",
            FONTS_DIR / "NotoNaskhArabic-Regular.ttf",
            FONTS_DIR / "Amiri-Bold.ttf",
            FONTS_DIR / "Amiri-Regular.ttf",
        ]
    )
    if not found:
        raise FileNotFoundError("No Arabic TTF font found for the title card")
    _AR_FONT_PATH = found
    log.info("title-card arabic-font=%s", found)
    return found


def _pick_latin_font() -> Path:
    global _LAT_FONT_PATH
    if _LAT_FONT_PATH and _LAT_FONT_PATH.is_file():
        return _LAT_FONT_PATH
    found = _pick_existing(
        [
            FONTS_DIR / "NotoSans-Bold.ttf",
            FONTS_DIR / "NotoSans-Regular.ttf",
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        ]
    )
    if not found:
        raise FileNotFoundError("No Latin TTF font found for the title card")
    _LAT_FONT_PATH = found
    log.info("title-card latin-font=%s", found)
    return found


def font_used() -> Path:
    """Primary (Arabic) card font; Latin companion is resolved separately."""
    return _pick_arabic_font()


def fonts_used() -> tuple[Path, Path]:
    return _pick_arabic_font(), _pick_latin_font()


def _ar_font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(_pick_arabic_font()), size=size)


def _lat_font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(_pick_latin_font()), size=size)


def _notdef_sig(font: ImageFont.FreeTypeFont) -> bytes:
    key = (str(getattr(font, "path", "")), int(getattr(font, "size", 0) or 0))
    cached = _NOTDEF_SIG.get(key)
    if cached is not None:
        return cached
    sig = bytes(font.getmask("\uFFFE"))
    _NOTDEF_SIG[key] = sig
    return sig


def _has_glyph(font: ImageFont.FreeTypeFont, ch: str) -> bool:
    """True when the font has a real glyph, not .notdef tofu."""
    if not ch or ch.isspace():
        return True
    try:
        mask = font.getmask(ch)
        data = bytes(mask)
    except Exception:
        return False
    if not data:
        return float(font.getlength(ch) or 0) > 0
    if not any(data):
        return float(font.getlength(ch) or 0) > 0
    return data != _notdef_sig(font)


def _font_for_char(
    ch: str,
    ar_font: ImageFont.FreeTypeFont,
    lat_font: ImageFont.FreeTypeFont,
) -> ImageFont.FreeTypeFont | None:
    if ch.isspace():
        return lat_font
    primary = ar_font if _is_arabic_char(ch) else lat_font
    if _has_glyph(primary, ch):
        return primary
    other = lat_font if primary is ar_font else ar_font
    if _has_glyph(other, ch):
        return other
    log.warning("no glyph for U+%04X %r; skipping to avoid tofu", ord(ch), ch)
    return None


def _char_script(ch: str) -> str:
    if _is_arabic_char(ch):
        return "ar"
    if ch.isalpha() or ch.isdigit():
        return "latn"
    if ch.isspace():
        return "sp"
    return "punct"


def _script_runs(text: str) -> list[tuple[str, str]]:
    """Split into Arabic vs Latin/digit runs; keep inter-script spaces as spacers."""
    if not text:
        return []
    raw: list[tuple[str, str]] = []
    for ch in text:
        s = _char_script(ch)
        if raw and raw[-1][0] == s:
            raw[-1] = (s, raw[-1][1] + ch)
        else:
            raw.append((s, ch))

    letter = {"ar", "latn"}
    n = len(raw)
    kinds = [s for s, _ in raw]

    def nearest_letter(idx: int, direction: int) -> int | None:
        rng = range(idx - 1, -1, -1) if direction < 0 else range(idx + 1, n)
        for j in rng:
            if kinds[j] in letter:
                return j
        return None

    absorb = [None] * n
    for i, (s, _t) in enumerate(raw):
        if s in letter:
            continue
        left = nearest_letter(i, -1)
        right = nearest_letter(i, 1)
        if left is not None and right is not None:
            if kinds[left] == kinds[right]:
                absorb[i] = left
            else:
                absorb[i] = None
        elif left is not None:
            absorb[i] = left
        elif right is not None:
            absorb[i] = right

    pieces = [(s, t) for s, t in raw]
    for i in range(n):
        j = absorb[i]
        if j is None:
            continue
        if j < i:
            pieces[j] = (pieces[j][0], pieces[j][1] + pieces[i][1])
            pieces[i] = (pieces[i][0], "")
        else:
            pieces[j] = (pieces[j][0], pieces[i][1] + pieces[j][1])
            pieces[i] = (pieces[i][0], "")

    out: list[tuple[str, str]] = []
    for s, t in pieces:
        if not t:
            continue
        if s in letter:
            script = s
        elif s == "sp":
            script = "sp"
        else:
            script = "latn"
        if out and out[-1][0] == script:
            out[-1] = (script, out[-1][1] + t)
        else:
            out.append((script, t))
    return out


def _glyph_runs(
    text: str,
    ar_font: ImageFont.FreeTypeFont,
    lat_font: ImageFont.FreeTypeFont,
) -> list[tuple[str, str, ImageFont.FreeTypeFont]]:
    out: list[tuple[str, str, ImageFont.FreeTypeFont]] = []
    for script, piece in _script_runs(text):
        run_font = ar_font if script == "ar" else lat_font
        for ch in piece:
            if ch.isspace() or script == "sp":
                font = run_font
                tag = script if script == "ar" else "sp"
            else:
                font = _font_for_char(ch, ar_font, lat_font)
                if font is None:
                    continue
                tag = "ar" if font is ar_font else "latn"
            if out and out[-1][0] == tag:
                out[-1] = (tag, out[-1][1] + ch, font)
            else:
                out.append((tag, ch, font))
    return out


def _shape(text: str) -> str:
    """Prepare Arabic for drawing.

    With libraqm, pass logical text (Pillow shapes + bidi).
    Without it, reshape presentation forms and apply python-bidi.
    """
    if not text:
        return ""
    if _HAS_RAQM:
        return text
    import arabic_reshaper

    try:
        from bidi.algorithm import get_display
    except ImportError:  # pragma: no cover
        from bidi import get_display

    reshaped = arabic_reshaper.reshape(text)
    try:
        return get_display(reshaped, base_dir="R")
    except TypeError:
        return get_display(reshaped)


def _run_width(script: str, text: str, font: ImageFont.FreeTypeFont) -> int:
    if not text:
        return 0
    if script == "ar":
        shaped = _shape(text)
        kw = _RTL_KW
    else:
        shaped = text
        kw = {}
    bbox = font.getbbox(shaped, **kw)
    return max(0, bbox[2] - bbox[0])


def _mixed_width(text: str, size: int) -> int:
    ar_f, lat_f = _ar_font(size), _lat_font(size)
    return sum(_run_width(s, t, f) for s, t, f in _glyph_runs(text, ar_f, lat_f))


def _mixed_height(text: str, size: int) -> int:
    ar_f, lat_f = _ar_font(size), _lat_font(size)
    sample = text or "أA"
    h = 1
    runs = _glyph_runs(sample, ar_f, lat_f)
    if not runs:
        bbox = ar_f.getbbox(_shape("أ"), **_RTL_KW)
        return max(1, bbox[3] - bbox[1])
    for s, t, f in runs:
        kw = _RTL_KW if s == "ar" else {}
        drawn = _shape(t) if s == "ar" else t
        bbox = f.getbbox(drawn or "أ", **kw)
        h = max(h, bbox[3] - bbox[1])
    ascent, descent = ar_f.getmetrics()
    return max(h, ascent + descent)


def _visual_runs(
    text: str, size: int
) -> list[tuple[str, str, ImageFont.FreeTypeFont]]:
    ar_f, lat_f = _ar_font(size), _lat_font(size)
    runs = _glyph_runs(text, ar_f, lat_f)
    # Keep Latin words in LTR order as one block; reverse only Arabic vs Latin blocks.
    # Spaces that sit between Arabic and Latin stay as their own run so they remain
    # between the two blocks after RTL reversal.
    coalesced: list[tuple[str, str, ImageFont.FreeTypeFont]] = []
    for s, t, f in runs:
        if s == "ar":
            coalesced.append((s, t, f))
        elif s == "sp":
            if coalesced and coalesced[-1][0] == "latn":
                ps, pt, pf = coalesced[-1]
                coalesced[-1] = (ps, pt + t, pf)
            else:
                coalesced.append((s, t, f))
        else:
            if coalesced and coalesced[-1][0] == "latn":
                ps, pt, pf = coalesced[-1]
                coalesced[-1] = (ps, pt + t, pf)
            else:
                coalesced.append((s, t, f))
    runs = coalesced
    has_ar = any(s == "ar" and any(_is_arabic_char(c) for c in t) for s, t, _f in runs)
    has_lat = any(s == "latn" and any(c.isalpha() or c.isdigit() for c in t) for s, t, _f in runs)
    if has_ar and has_lat:
        runs = list(reversed(runs))
    return runs


def _draw_mixed_line(
    draw: ImageDraw.ImageDraw,
    cx: int,
    y_mid: int,
    text: str,
    size: int,
    fill: tuple[int, int, int],
) -> None:
    runs = _visual_runs(text, size)
    if not runs:
        return
    widths = [_run_width(s, t, f) for s, t, f in runs]
    total = sum(widths)
    x = float(cx) - total / 2.0
    for (s, t, f), w in zip(runs, widths):
        if s == "ar":
            draw.text((x, y_mid), _shape(t), font=f, fill=fill, anchor="lm", **_RTL_KW)
        else:
            draw.text((x, y_mid), t, font=f, fill=fill, anchor="lm")
        x += w


def _width(text: str, size: int) -> int:
    return _mixed_width(text, size)


def _height(text: str, size: int) -> int:
    return _mixed_height(text, size)


def _wrap(text: str, size: int, max_width: int, max_lines: int) -> list[str]:
    words = (text or "").split()
    if not words:
        return [""]
    lines: list[str] = []
    i = 0
    while i < len(words):
        left = max_lines - len(lines)
        if left <= 1:
            lines.append(" ".join(words[i:]))
            break
        cur = words[i]
        i += 1
        while i < len(words):
            trial = f"{cur} {words[i]}"
            if _width(trial, size) <= max_width:
                cur = trial
                i += 1
            else:
                break
        lines.append(cur)
    return lines


def _fit(
    text: str,
    max_width: int,
    max_lines: int,
    max_size: int,
    min_size: int,
) -> tuple[int, list[str]]:
    size = max_size
    while size >= min_size:
        lines = _wrap(text, size, max_width, max_lines)
        if all(_width(ln, size) <= max_width for ln in lines):
            return size, lines
        size -= 4
    return min_size, _wrap(text, min_size, max_width, max_lines)


def _gradient() -> Image.Image:
    top = Image.new("RGB", (W, H), BG_TOP)
    bot = Image.new("RGB", (W, H), BG_BOT)
    mask = Image.linear_gradient("L").resize((W, H))
    return Image.composite(bot, top, mask)


def _draw_rule(draw: ImageDraw.ImageDraw, y: int, half_w: int = 260) -> None:
    cx = W // 2
    draw.line([(cx - half_w, y), (cx + half_w, y)], fill=RULE, width=2)
    r = 6
    draw.polygon(
        [(cx, y - r), (cx + r, y), (cx, y + r), (cx - r, y)],
        fill=GOLD,
    )


def render_card(title: str, author: str, dest: Path | str | None = None) -> Path:
    """Render a 1920x1080 Arabic title card PNG. Returns the path."""
    title = sanitize_for_speech(title or "")
    author = sanitize_for_speech(author or "") or "مؤلف غير مذكور"
    if dest is None:
        VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        dest_path = VIDEO_DIR / "card.png"
    else:
        dest_path = Path(dest)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

    ar_path, lat_path = fonts_used()
    log.info(
        "title-card fonts arabic=%s latin=%s raqm=%s",
        ar_path.name,
        lat_path.name,
        _HAS_RAQM,
    )

    img = _gradient()
    # Soft vignette
    vig = Image.new("L", (W, H), 0)
    vd = ImageDraw.Draw(vig)
    vd.ellipse((-200, -180, W + 200, H + 180), fill=255)
    vig = vig.filter(ImageFilter.GaussianBlur(90))
    shade = Image.new("RGB", (W, H), (6, 8, 14))
    img = Image.composite(img, shade, vig)

    draw = ImageDraw.Draw(img)
    # Double gold frame
    draw.rounded_rectangle([72, 54, W - 72, H - 54], radius=28, outline=FRAME, width=3)
    draw.rounded_rectangle([86, 68, W - 86, H - 68], radius=20, outline=GOLD_SOFT, width=1)

    max_text_w = W - 280  # ~1640
    label = "ملخص كتاب"
    author_line = f"للكاتب {author}"

    label_size = 64
    title_size, title_lines = _fit(
        title, max_text_w, max_lines=3, max_size=132, min_size=56
    )
    author_size, author_lines = _fit(
        author_line, max_text_w, max_lines=2, max_size=52, min_size=36
    )

    label_h = _height(label, label_size)
    title_gap = max(18, int(title_size * 0.28))
    title_hs = [_height(ln, title_size) for ln in title_lines]
    title_block = sum(title_hs) + title_gap * (len(title_lines) - 1)
    author_gap = 10
    author_hs = [_height(ln, author_size) for ln in author_lines]
    author_block = sum(author_hs) + author_gap * (len(author_lines) - 1)

    rule_pad = 36
    stack_h = (
        label_h
        + rule_pad
        + 12
        + rule_pad
        + title_block
        + 48
        + author_block
    )
    y = (H - stack_h) // 2

    cx = W // 2
    _draw_mixed_line(draw, cx, y + label_h // 2, label, label_size, GOLD)
    y += label_h + rule_pad
    _draw_rule(draw, y)
    y += rule_pad + 12

    for ln, lh in zip(title_lines, title_hs):
        _draw_mixed_line(draw, cx, y + lh // 2, ln, title_size, CREAM)
        y += lh + title_gap
    y -= title_gap
    y += 48

    for ln, lh in zip(author_lines, author_hs):
        _draw_mixed_line(draw, cx, y + lh // 2, ln, author_size, GOLD_SOFT)
        y += lh + author_gap

    img.save(dest_path, "PNG", optimize=True)
    log.info(
        "wrote title card %s arabic=%s latin=%s",
        dest_path,
        ar_path.name,
        lat_path.name,
    )
    return dest_path


def image_plus_audio_mp4(png: Path | str, mp3: Path | str, dest: Path | str) -> Path:
    """Mux a still PNG with an MP3 into a 1920x1080 H.264/AAC MP4."""
    png_p, mp3_p, dest_p = Path(png), Path(mp3), Path(dest)
    if not png_p.is_file():
        raise FileNotFoundError(f"missing card png: {png_p}")
    if not mp3_p.is_file():
        raise FileNotFoundError(f"missing audio: {mp3_p}")
    dest_p.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest_p.with_suffix(".partial.mp4")
    cmd = [
        FFMPEG,
        "-y",
        "-loop",
        "1",
        "-framerate",
        "1",
        "-i",
        str(png_p),
        "-i",
        str(mp3_p),
        "-c:v",
        "libx264",
        "-tune",
        "stillimage",
        "-pix_fmt",
        "yuv420p",
        "-vf",
        "scale=1920:1080",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        "-movflags",
        "+faststart",
        str(tmp),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not tmp.is_file() or tmp.stat().st_size < 1000:
        err = (proc.stderr or proc.stdout or "")[-1200:]
        raise RuntimeError(f"ffmpeg mux failed (code={proc.returncode}): {err}")
    tmp.replace(dest_p)
    log.info("wrote mp4 %s bytes=%s", dest_p, dest_p.stat().st_size)
    return dest_p


def ensure_mp4(
    title: str,
    author: str,
    mp3: Path | str,
    dest: Path | str | None = None,
) -> Path:
    """Build (or reuse) an MP4 next to the MP3 from a title card + TTS audio."""
    mp3_p = Path(mp3)
    dest_p = Path(dest) if dest else mp3_p.with_suffix(".mp4")
    if (
        dest_p.is_file()
        and dest_p.stat().st_size > 1000
        and dest_p.stat().st_mtime >= mp3_p.stat().st_mtime
    ):
        return dest_p
    png = dest_p.with_suffix(".png")
    render_card(title, author, dest=png)
    return image_plus_audio_mp4(png, mp3_p, dest_p)

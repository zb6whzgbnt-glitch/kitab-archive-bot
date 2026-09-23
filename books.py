"""Search, download, and slice public-domain books from archive.org."""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from urllib.parse import quote

import httpx

log = logging.getLogger("kitab.books")

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data" / "books"
IA_SEARCH = "https://archive.org/advancedsearch.php"
IA_META = "https://archive.org/metadata/{ident}"
IA_DOWNLOAD = "https://archive.org/download/{ident}/{name}"

HEADERS = {
    "User-Agent": "KitabArchiveBot/1.0 (public-domain educational summaries)",
    "Accept": "application/json, text/plain, */*",
}

# Mission: English jestbooks / merry tales / anecdotes (incl. women, judges, marriage, bawdy facetiae)
# → Arabic nawadir-style delivery; faithful tone, no bowdlerizing.
CONTENT_MODE = "jestbooks"
SOLE_SOURCE_TITLE = ""
SOLE_SOURCE_AUTHOR = ""
SOLE_SOURCE_IDENTIFIERS: list[str] = []

# Preferred public-domain jestbooks / merry tales / facetiae / anecdotes / fabliaux.
PREFERRED_JESTBOOKS_EN = [
    # Core English jestbooks & merry tales
    "englishjestsanda49370gut",      # English Jests and Anecdotes (Gutenberg — clean text)
    "thejestbookthech20352gut",      # The Jest Book
    "jestbookchoicest00lemo",        # The Jest Book (choicest anecdotes)
    "joemillerscomple00mill",        # Joe Miller's complete jest book
    "hundredmerrytale00hazl",        # A Hundred Merry Tales
    "cu31924013155068",
    "thehundredmerrytalesorsh",
    "picturebookofmer00londiala",
    "apicturebookofme43599gut",
    # Judges / lawyers / bench & bar anecdotes
    "cu31924021264209",              # Legal facetiae : satirical and humorous
    "westminsterhall00roscgoog",     # Westminster Hall anecdotes of bar & bench
    "lawandlawyersas01polsgoog",     # Law and Lawyers gossip & anecdotes
    "cu31924018791883",              # Bench and bar in California (anecdotes)
    "oldcourthouserem00cart",        # Old court house reminiscences & anecdotes
    # Women / marriage / lovers (literary public-domain tale collections)
    "decameron01boccuoft",           # Boccaccio — The Decameron
    "storiesboccacci00boccgoog",     # Stories of Boccaccio (The Decameron)
    "heptameronoftale00marguoft",    # Heptameron of Margaret of Navarre
    "cu31924082463211",              # The Heptameron
    # Bawdy / sexual-tone facetiae & fabliaux (faithful literary jests, not modern porn)
    "facetiaeorjocos00bracgoog",     # Poggio Facetiae / jocose tales
    "facetiaecantabri00gooc",        # Facetiae cantabrigienses
    "fabliauxortalesa03legriala",    # Fabliaux or tales
    "fabliauxortalesa01legriala",
    "fabliauxetcontes00tarsiala",
    "facetiaemusarumd01mennuoft",    # Facetiae: Musarum deliciae
    "cu31924013117167",              # Merry tales of Gotham
    "merrytalesofthre00paul",
    "englishjestsanec00edin",
    "englishjestsanec00lond",
    "hundredmerrytale0000zall",
]
PREFERRED_ARABIC: list[str] = []
PREFERRED_ENGLISH = list(PREFERRED_JESTBOOKS_EN)
PREFERRED_RUSSIAN: list[str] = []
PREFERRED_URDU: list[str] = []
PREFERRED_BY_LANG = {
    "ara": [],
    "eng": PREFERRED_ENGLISH,
    "rus": PREFERRED_RUSSIAN,
    "urd": PREFERRED_URDU,
}
PREFERRED_IDENTIFIERS = PREFERRED_ENGLISH
PREFERRED_HISTORY_EN = PREFERRED_JESTBOOKS_EN  # legacy alias

SEED_QUERIES_AR: list[str] = []
SEED_QUERIES_EN = [
    # 1) Jest books / merry tales / conceits
    (
        '(title:jestbook OR title:"jest book" OR title:jests OR title:"merry tales" '
        'OR title:"merry conceits" OR title:"quick answers" OR title:"pleasant tales" '
        'OR title:"pleasant conceits" OR title:"witty tales" OR title:"comic tales") '
        "AND mediatype:texts AND (language:eng OR language:English) "
        "AND (collection:gutenberg OR format:DjVuTXT OR format:Text) "
        "AND downloads:[20 TO *]"
    ),
    # 2) Facetiae / anecdotes / wit
    (
        '(title:facetiae OR title:facetiæ OR title:anecdotes OR title:"wit and humor" '
        'OR title:"wit and humour" OR title:"humorous anecdotes" OR creator:"Joe Miller") '
        "AND mediatype:texts AND (language:eng OR language:English) "
        "AND (collection:gutenberg OR format:DjVuTXT OR format:Text) "
        "AND downloads:[20 TO *]"
    ),
    # 3) Coffee-house / city jests / fabliaux / eulenspiegel
    (
        '(title:"coffee-house" OR title:"coffee house" OR title:"oxford jests" '
        'OR title:"london jests" OR title:fabliaux OR title:eulenspiegel '
        'OR title:"ship of fools" OR title:"english rogue" OR title:gotham) '
        "AND mediatype:texts AND (language:eng OR language:English OR language:fre OR language:French) "
        "AND (collection:gutenberg OR format:DjVuTXT OR format:Text) "
        "AND downloads:[15 TO *]"
    ),
    # 4) Gutenberg merry/jest/anecdote sweep
    (
        "collection:gutenberg AND mediatype:texts AND (language:eng OR language:English) "
        'AND (title:jests OR title:"merry tales" OR title:anecdotes OR title:"jest book" '
        'OR title:facetiae OR title:fabliaux OR title:eulenspiegel) '
        "AND downloads:[10 TO *]"
    ),
    # 5) Women / wives / marriage / lovers (literary tale collections)
    (
        '(title:heptameron OR title:decameron OR title:"droll stories" '
        'OR title:"comforts of matrimony" OR title:"wives and husbands" '
        'OR (title:anecdotes AND (title:women OR title:wives OR title:ladies)) '
        'OR title:"conjugal") '
        "AND mediatype:texts AND (language:eng OR language:English) "
        "AND (collection:gutenberg OR format:DjVuTXT OR format:Text) "
        "AND downloads:[20 TO *]"
    ),
    # 6) Judges / lawyers / bench & bar anecdotes
    (
        '(title:"legal facetiae" OR title:"law and lawyers" OR title:"westminster hall" '
        'OR (title:anecdotes AND (title:bench OR title:bar OR title:lawyer OR title:lawyers '
        'OR title:judge OR title:judges OR title:court)) '
        'OR title:"bench and bar") '
        "AND mediatype:texts AND (language:eng OR language:English) "
        "AND (format:DjVuTXT OR format:Text OR collection:gutenberg) "
        "AND downloads:[10 TO *]"
    ),
    # 7) Bawdy / sexual-tone facetiae & fabliaux (public-domain literary, not modern erotica)
    (
        '(title:facetiae OR title:facetiæ OR title:fabliaux OR title:poggio '
        'OR title:scoggin OR title:"mery talys" OR title:"quick answers" '
        'OR title:"musarum deliciae") '
        "AND mediatype:texts AND (language:eng OR language:English) "
        "AND (collection:gutenberg OR format:DjVuTXT OR format:Text) "
        "AND downloads:[15 TO *]"
    ),
]
SEED_QUERIES_RU: list[str] = []
SEED_QUERIES_UR: list[str] = []
SEED_QUERIES_BY_LANG = {
    "ara": SEED_QUERIES_AR,
    "eng": SEED_QUERIES_EN,
    "rus": SEED_QUERIES_RU,
    "urd": SEED_QUERIES_UR,
}
SEED_QUERIES = list(SEED_QUERIES_EN)
ALLOW_ARCHIVE_SEARCH = True
SOURCE_LANGUAGES = ("eng",)

SKIP_TITLE_RE = re.compile(
    r"(quran|qur.?an|mushaf|مصحف|القرآن|قرآن|textbook|nizami|nct0|"
    r"color.?coded|tajw[ie]d|pearl.?a journal|"
    r"kaczynski|anarchist library|descriptive catalogue|"
    r"deaths.?jest.?book|fool.?s tragedy|"
    r"treatise on the law|law of husband and wife|handbook of husband|"
    r"ashraf ali|thanvi|f\*\*\* my wife|erotic cuckold|"
    r"gift to husband and wife)",
    re.I,
)

CHAPTER_RE = re.compile(
    r"(?m)^(?:"
    r"الفصل\s+\S+"
    r"|الباب\s+\S+"
    r"|الليلة\s+(?:\S+\s*){0,4}"
    r"|القسم\s+\S+"
    r"|الحكاية\s+\S+"
    r"|حكاية\s+\S+"
    r"|Chapter\s+\d+"
    r"|CHAPTER\s+[IVXLC\d]+"
    r"|BOOK\s+[IVXLC\d]+"
    r"|Night\s+\d+"
    r"|THE\s+\d+(?:ST|ND|RD|TH)?\s+NIGHT"
    r"|Сказка\s+\S+"
    r"|СКАЗКА\s+\S+"
    r"|Tale\s+\d+"
    r"|Story\s+\d+"
    r").{0,100}$"
)

# OCR / mid-line night markers in Alf Layla editions
NIGHT_INLINE_RE = re.compile(
    r"(?:"
    r"\(\s*وف[يى]?\s*ال?ليلة"
    r"|وف\s*الليلة"
    r"|وفي\s*ليلة"
    r"|الليلة\s+[٠-٩\d]+"
    r"|Night\s+\d+"
    r")"
)

STORY_TARGET_WORDS = 1200  # richer slice so Gemini can expand to a medium tale


WORD_RE = re.compile(r"[\w\u0600-\u06FF]+", re.UNICODE)
ARABIC_RE = re.compile(r"[\u0600-\u06FF]")

MIN_BOOK_CHARS = 8000
MIN_ARABIC_RATIO = 0.12
INSTALLMENT_WORDS = 2800


def _client() -> httpx.Client:
    return httpx.Client(headers=HEADERS, timeout=60.0, follow_redirects=True)


def _get_json(client: httpx.Client, url: str, params: dict | list | None = None) -> dict:
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            r = client.get(url, params=params)
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            log.warning("JSON fetch failed (%s) %s", attempt + 1, url)
    raise RuntimeError(f"Failed to fetch JSON: {url}") from last_exc


def search_archive(query: str, rows: int = 25) -> list[dict]:
    params = [
        ("q", query),
        ("fl[]", "identifier"),
        ("fl[]", "title"),
        ("fl[]", "creator"),
        ("fl[]", "subject"),
        ("fl[]", "language"),
        ("fl[]", "year"),
        ("fl[]", "downloads"),
        ("rows", str(rows)),
        ("page", "1"),
        ("output", "json"),
        ("sort[]", "downloads desc"),
    ]
    with _client() as client:
        data = _get_json(client, IA_SEARCH, params=params)
    docs = (data.get("response") or {}).get("docs") or []
    log.info("archive.org search numFound=%s returned=%s", (data.get("response") or {}).get("numFound"), len(docs))
    return docs


def get_metadata(identifier: str) -> dict:
    with _client() as client:
        return _get_json(client, IA_META.format(ident=quote(identifier, safe="")))


def _is_text_file(entry: dict) -> bool:
    name = (entry.get("name") or "").lower()
    fmt = (entry.get("format") or "").lower()
    if name.endswith("_djvu.txt"):
        return True
    if fmt in {"djvutxt", "full text"}:
        return True
    if name.endswith(".txt") and fmt in {"text", "djvutxt", "full text", ""}:
        return True
    if name.endswith(".txt") and "djvu" not in name and "gutenberg" in name:
        return True
    if name.endswith(".txt"):
        return True
    return False



def _volume_text_files(metadata: dict) -> list[dict]:
    """For the sole multi-volume item, return G1..G5 DjVuTXT files in order."""
    files = metadata.get("files") or []
    vols: list[tuple[int, dict]] = []
    for f in files:
        name = str(f.get("name") or "")
        m = re.search(r"Asatter_Sha3byh-G([1-5]).*_djvu\.txt$", name, re.I)
        if not m:
            m = re.search(r"Asatter_Sha3byh-G([1-5])\.txt$", name, re.I)
        if m and _is_text_file(f):
            vols.append((int(m.group(1)), f))
    vols.sort(key=lambda t: t[0])
    # unique by volume number
    seen: set[int] = set()
    out: list[dict] = []
    for num, f in vols:
        if num in seen:
            continue
        seen.add(num)
        out.append(f)
    return out


def pick_text_file(metadata: dict) -> dict | None:
    files = metadata.get("files") or []
    candidates = [f for f in files if _is_text_file(f)]
    if not candidates:
        return None

    def score(entry: dict) -> tuple:
        name = (entry.get("name") or "").lower()
        fmt = (entry.get("format") or "").lower()
        size = int(entry.get("size") or 0)
        pref = 0
        if name.endswith("_djvu.txt") or fmt == "djvutxt":
            pref += 100
        if name.endswith(".txt"):
            pref += 40
        if "gutenberg" in name or name.endswith("-0.txt") or name.endswith("_djvu.txt"):
            pref += 10
        # Prefer reasonably sized plain text, skip tiny stubs and huge dumps first.
        size_score = min(size, 5_000_000)
        return (pref, size_score)

    candidates.sort(key=score, reverse=True)
    return candidates[0]


def download_text_file(identifier: str, filename: str) -> str:
    url = IA_DOWNLOAD.format(ident=quote(identifier, safe=""), name=quote(filename))
    last_exc: Exception | None = None
    with _client() as client:
        for attempt in range(3):
            try:
                r = client.get(url)
                r.raise_for_status()
                data = r.content
                for enc in ("utf-8", "utf-8-sig", "cp1256", "latin-1"):
                    try:
                        return data.decode(enc)
                    except UnicodeDecodeError:
                        continue
                return data.decode("utf-8", errors="replace")
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                log.warning("download failed (%s) %s", attempt + 1, url)
    raise RuntimeError(f"Failed to download {identifier}/{filename}") from last_exc


RUNNING_HEADER_RE = re.compile(
    r"تهذيب\s+إسلام[ىي]\s+لقصص\s+كليلة\s+ودمن[ةه]",
    re.I,
)
FOOTNOTE_RE = re.compile(r"[\(][\d٠-٩IVXLC/\s،,.-]{1,12}[\)]")


def _line_is_noise(line: str) -> bool:
    letters = sum(1 for c in line if c.isalpha() or "\u0600" <= c <= "\u06FF")
    if letters < 8:
        return True
    if letters / max(len(line), 1) < 0.38:
        return True
    if re.fullmatch(r"[\d\W_]{0,16}", line):
        return True
    tokens = line.split()
    if tokens:
        short = sum(1 for t in tokens if len(re.sub(r"\W+", "", t, flags=re.UNICODE)) <= 1)
        if short / len(tokens) > 0.4:
            return True
    digits = sum(1 for c in line if c.isdigit() or "٠" <= c <= "٩")
    if digits >= 10 and digits / max(len(line), 1) > 0.18:
        return True
    return False


def clean_ocr_text(text: str) -> str:
    text = text.replace("\ufeff", "").replace("\x0c", "\n")
    text = unicodedata.normalize("NFC", text)
    text = RUNNING_HEADER_RE.sub(" ", text)
    text = FOOTNOTE_RE.sub(" ", text)
    kept: list[str] = []
    for raw in text.splitlines():
        line = re.sub(r"[ \t]+", " ", raw).strip()
        if not line:
            if kept and kept[-1] != "":
                kept.append("")
            continue
        if _line_is_noise(line):
            continue
        kept.append(line)
    cleaned = "\n".join(kept)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r" {2,}", " ", cleaned)
    return cleaned.strip()


def arabic_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha() or "\u0600" <= c <= "\u06FF"]
    if not letters:
        return 0.0
    ar = sum(1 for c in letters if "\u0600" <= c <= "\u06FF")
    return ar / len(letters)


def cyrillic_count(text: str) -> int:
    return sum(1 for c in text if "\u0400" <= c <= "\u04FF")


def arabic_script_count(text: str) -> int:
    # Arabic + presentation forms (covers Urdu Nastaliq range partially via Arabic block)
    return sum(
        1 for c in text
        if ("\u0600" <= c <= "\u06FF") or ("\u0750" <= c <= "\u077F")
        or ("\uFB50" <= c <= "\uFDFF") or ("\uFE70" <= c <= "\uFEFF")
    )


def is_usable_text(text: str) -> bool:
    if len(text) < MIN_BOOK_CHARS:
        return False
    words = WORD_RE.findall(text)
    if len(words) < 800:
        return False
    ratio = arabic_ratio(text)
    latin = sum(1 for c in text if ("A" <= c <= "Z") or ("a" <= c <= "z"))
    cyr = cyrillic_count(text)
    if ratio >= MIN_ARABIC_RATIO:
        return True
    if latin >= 4000:
        return True
    if cyr >= 3000:
        return True
    if arabic_script_count(text) >= 4000:
        return True
    return False


def cache_path(identifier: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", identifier)[:120]
    return CACHE_DIR / f"{safe}.txt"


def load_book_text(identifier: str, metadata: dict | None = None) -> tuple[str, str, dict]:
    """Return (cleaned_text, title, metadata). Raises if unusable."""
    path = cache_path(identifier)
    meta = metadata or get_metadata(identifier)
    md = meta.get("metadata") or {}
    if identifier in SOLE_SOURCE_IDENTIFIERS:
        title = SOLE_SOURCE_TITLE
    else:
        title = str(md.get("title") or identifier).strip()
    if path.exists() and path.stat().st_size > MIN_BOOK_CHARS:
        text = path.read_text(encoding="utf-8", errors="replace")
        if is_usable_text(text):
            return text, title, meta

    raw_parts: list[str] = []
    if identifier in SOLE_SOURCE_IDENTIFIERS:
        vol_files = _volume_text_files(meta)
        if vol_files:
            for f in vol_files:
                log.info("downloading sole-source volume %s/%s", identifier, f.get("name"))
                raw_parts.append(download_text_file(identifier, f["name"]))
        else:
            file_entry = pick_text_file(meta)
            if not file_entry:
                raise ValueError(f"no extractable text for {identifier}")
            raw_parts.append(download_text_file(identifier, file_entry["name"]))
    else:
        file_entry = pick_text_file(meta)
        if not file_entry:
            raise ValueError(f"no extractable text for {identifier}")
        raw_parts.append(download_text_file(identifier, file_entry["name"]))

    text = clean_ocr_text("\n\n".join(raw_parts))
    if not is_usable_text(text):
        raise ValueError(f"text unusable for {identifier} (len={len(text)})")
    path.write_text(text, encoding="utf-8")
    return text, title, meta


def _word_chunks(text: str, target_words: int = INSTALLMENT_WORDS) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    i = 0
    n = len(words)
    while i < n:
        j = min(n, i + target_words)
        # Stretch to a sentence-ish boundary when possible.
        if j < n:
            joined = " ".join(words[i:j])
            extra = 0
            while j + extra < n and extra < 80:
                w = words[j + extra]
                joined_extra = joined if extra == 0 else joined + " " + " ".join(words[j : j + extra + 1])
                if re.search(r"[.!?؟۔]$", w) or w.endswith((".", "!", "?", "؟", "۔")):
                    j = j + extra + 1
                    break
                extra += 1
                joined = joined_extra
        chunk = " ".join(words[i:j]).strip()
        if len(chunk) > 400:
            chunks.append(chunk)
        i = j
    return chunks


def slice_installments(text: str) -> list[str]:
    """Split a book into installment slices (chapters if possible, else word chunks)."""
    matches = list(CHAPTER_RE.finditer(text))
    if len(matches) >= 4:
        starts = [m.start() for m in matches]
        starts.append(len(text))
        # Keep preamble only if long enough to stand alone.
        parts: list[str] = []
        if starts[0] > 1500:
            pre = text[: starts[0]].strip()
            if len(pre) > 800:
                parts.append(pre)
        for a, b in zip(starts, starts[1:]):
            piece = text[a:b].strip()
            if len(piece) > 600:
                parts.append(piece)
        # Merge tiny leftover chapters into neighbors.
        merged: list[str] = []
        for p in parts:
            if merged and len(p.split()) < 400:
                merged[-1] = merged[-1] + "\n\n" + p
            else:
                merged.append(p)
        # If chapters are huge, sub-chunk them.
        final: list[str] = []
        for p in merged:
            if len(p.split()) > INSTALLMENT_WORDS * 1.6:
                final.extend(_word_chunks(p))
            else:
                final.append(p)
        if len(final) >= 2:
            return final
    chunks = _word_chunks(text)
    return chunks or [text]




# عناوين حكايات الجهيمان الشائعة في OCR
JUHAYMAN_TITLE_RE = re.compile(
    r"(?m)^\s*((?:سالفة|سبحونة|سباحونة)\s+[^\n]{3,90}?)\s*$"
)
_JUHAYMAN_TITLE_BAD = re.compile(
    r"(ومعناها|فقالت|قصتها|من سوالف|التي |كما أن|وهو |في خلوة|واستعدت|فهي من|"
    r"جديدة لم|فأحب أن|تكون حقيقة|مرة ثانية|إذا كنتم|فاجلسوا|أصحاب|"
    r"لم تسمعوها|اسمها|أسمها|وهي سبحونة)",
    re.I,
)


def _clean_juhayman_title(raw: str) -> str:
    cand = re.sub(r"\s+", " ", (raw or "")).strip(" .…:-–—!؟?،,")
    # drop trailing OCR digits / junk tokens
    cand = re.sub(r"[\d٠-٩]+", " ", cand)
    cand = re.sub(r"\s+", " ", cand).strip()
    if len(cand) < 6 or _JUHAYMAN_TITLE_BAD.search(cand):
        return ""
    for sep in (".", "،", ":", "…"):
        if sep in cand[8:]:
            cand = cand.split(sep, 1)[0].strip()
    # keep only if starts with marker and has a bit of substance
    if not re.match(r"^(?:سالفة|سبحونة|سباحونة)\s+\S+", cand):
        return ""
    if len(cand.split()) < 2:
        return ""
    if len(cand) > 80:
        cand = cand[:80].rstrip()
    return cand



_BOOKISH_TITLE_RE = re.compile(
    r"(?i)\b("
    r"jests?|anecdotes?|merry\s+tales?|facetiae|jest\s*book|collected|various|"
    r"complete\s+jest|choicest|wit\s+and\s+humou?r"
    r")\b"
)


def extract_jest_title(text: str, collection_title: str = "") -> str:
    """Best-effort title of a single jest/anecdote inside a slice (not the book title)."""
    body = (text or "").strip()
    if not body:
        return ""
    coll = (collection_title or "").strip().lower()
    bookish = {
        "contents", "preface", "introduction", "advertisement", "to the reader",
        "index", "title page", "frontispiece",
    }
    for ln in body.splitlines()[:20]:
        s = re.sub(r"\s+", " ", ln.strip(" \t.\u2014-"))
        if not s or len(s) < 6 or len(s) > 90:
            continue
        low = s.lower()
        if low in bookish:
            continue
        if coll and (low == coll or low in coll or coll in low):
            continue
        if _BOOKISH_TITLE_RE.search(s) and len(s.split()) >= 5:
            # Likely the collection name repeated as a heading
            continue
        # Numbered jests: "XII. The Parson and ..." / "42. Of a ..."
        m = re.match(r"^(?:[IVXLCDM]+|\d+)[.)\]\s:-]+(.+)$", s, re.I)
        if m and len(m.group(1).strip()) >= 6:
            cand = m.group(1).strip()
            if not _BOOKISH_TITLE_RE.search(cand):
                return cand[:80]
        # Title-case / short heading line
        letters = sum(c.isalpha() for c in s)
        if letters < 6:
            continue
        if s.endswith((".", "!", "?", ";", ":")) and len(s.split()) > 14:
            continue  # prose sentence
        if s[:1].isupper() and letters / max(1, len(s)) > 0.55:
            return s[:80]
    # Fallback: first short sentence fragment
    first = re.split(r"(?<=[.!?])\s+", body[:400])
    if first:
        frag = first[0].strip()
        if 12 <= len(frag) <= 90 and not _BOOKISH_TITLE_RE.search(frag):
            return frag[:80]
    return ""


def extract_history_episode_title(text: str) -> str | None:
    """Best-effort chapter/episode title from an English history slice."""
    import re as _re
    for ln in (text or "").splitlines()[:25]:
        s = ln.strip()
        if not s or len(s) > 90 or len(s) < 6:
            continue
        low = s.lower()
        if any(x in low for x in ("contents", "chapter list", "illustrated", "translated", "volume")):
            continue
        # Life of X / Book X / The Battle of ...
        if _re.match(r"^(life of|the life of|book\s+[ivxlc\d]+|chapter\s+\d+|the battle of|history of)\b", low):
            return s[:80]
        if s.isupper() and 6 <= len(s) <= 60 and sum(c.isalpha() for c in s) >= 5:
            return s.title()[:80]
    return None


def extract_juhayman_title(text: str) -> str:
    """Best-effort original tale title from a story slice (سالفة/سبحونة)."""
    body = (text or "").strip()
    if not body:
        return ""
    head = "\n".join(body.splitlines()[:50])
    for m in JUHAYMAN_TITLE_RE.finditer(head):
        cand = _clean_juhayman_title(m.group(1))
        if cand:
            return cand
    m = re.search(r"((?:سالفة|سبحونة|سباحونة)\s+[^\n.،:]{3,70})", body[:1200])
    if m:
        cand = _clean_juhayman_title(m.group(1))
        if cand:
            return cand
    return ""


def _juhayman_title_cut_points(text: str) -> list[int]:
    """High-confidence title starts for splitting the الجهيمان corpus."""
    cuts: list[int] = []
    for m in JUHAYMAN_TITLE_RE.finditer(text):
        cand = _clean_juhayman_title(m.group(1))
        if not cand:
            continue
        pos = m.start()
        if cuts and pos - cuts[-1] < 1200:
            continue
        cuts.append(pos)
    return cuts



_HISTORY_CHAPTER_RE = __import__("re").compile(
    r"(?m)^(?:\s*)(?:"
    r"LIFE OF [A-Z][A-Za-z \-']{2,40}"
    r"|THE LIFE OF [A-Z][A-Za-z \-']{2,40}"
    r"|BOOK\s+[IVXLCDM]+\b"
    r"|CHAPTER\s+[IVXLCDM\d]+\b"
    r"|BOOK\s+\d+\b"
    r"|CHAPTER\s+\d+\b"
    r")\s*$"
)


def slice_history_chapters(text: str) -> list[str]:
    """Split English history books on Life/Book/Chapter headings when possible."""
    raw = text or ""
    matches = list(_HISTORY_CHAPTER_RE.finditer(raw))
    if len(matches) < 2:
        return []
    parts: list[str] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        chunk = raw[start:end].strip()
        if len(chunk.split()) >= 150:
            parts.append(chunk)
    return parts


def slice_stories(text: str) -> list[str]:
    """Split a collection into short standalone stories/nights/chapters."""
    text = (text or "").strip()
    if not text:
        return []
    # Prefer English history Life/Book/Chapter headings when present.
    hist = slice_history_chapters(text)
    if len(hist) >= 2:
        final_h: list[str] = []
        for p in hist:
            wc = len(p.split())
            if wc > STORY_TARGET_WORDS * 2:
                final_h.extend(_word_chunks(p, STORY_TARGET_WORDS))
            else:
                final_h.append(p)
        if len(final_h) >= 2:
            return final_h
    # Prefer chapter/night headings.
    matches = list(CHAPTER_RE.finditer(text))
    if len(matches) < 3:
        # Fall back to inline night markers (OCR Alf Layla).
        matches = list(NIGHT_INLINE_RE.finditer(text))
    if len(matches) >= 3:
        starts = [m.start() for m in matches]
        starts.append(len(text))
        parts: list[str] = []
        if starts[0] > 1200:
            pre = text[: starts[0]].strip()
            if len(pre.split()) > 200:
                parts.append(pre)
        for a, b in zip(starts, starts[1:]):
            piece = text[a:b].strip()
            if len(piece.split()) >= 120:
                parts.append(piece)
        # Sub-chunk huge nights; merge tiny ones.
        final: list[str] = []
        for p in parts:
            wc = len(p.split())
            if wc > STORY_TARGET_WORDS * 2:
                final.extend(_word_chunks(p, STORY_TARGET_WORDS))
            elif final and wc < 180:
                final[-1] = final[-1] + "\n\n" + p
            else:
                final.append(p)
        if len(final) >= 2:
            return final
    chunks = _word_chunks(text, STORY_TARGET_WORDS)
    return chunks or [text]


def story_id(identifier: str, story_index: int) -> str:
    return f"{identifier}#{int(story_index)}"


def _title_ok(title: str) -> bool:
    return not bool(SKIP_TITLE_RE.search(title or ""))


def _candidate_from_doc(doc: dict) -> dict:
    ident = str(doc.get("identifier") or "")
    title = str(doc.get("title") or ident)
    return {
        "identifier": ident,
        "title": title,
        "author": _join_meta_field(doc.get("creator")),
        "raw": doc,
    }


GENRES: list[tuple[str, str]] = [
    ("fiction", "رواية"),
    ("short", "قصص قصيرة"),
    ("history", "تاريخ"),
    ("sociology", "علم اجتماع"),
    ("philosophy", "فلسفة"),
    ("science", "علوم"),
    ("poetry", "شعر"),
    ("bio", "سيرة"),
    ("politics", "سياسة"),
    ("travel", "رحلات"),
    ("any", "أي مجال"),
]
GENRE_LABELS = {gid: label for gid, label in GENRES}

GENRE_SUBJECT = {
    "fiction": "subject:(fiction OR novel OR novels OR رواية OR روايات)",
    "short": 'subject:("short stories" OR "short story" OR قصص OR قصة)',
    "history": "subject:(history OR تاريخ OR historical)",
    "sociology": 'subject:(sociology OR "social sciences" OR اجتماع OR اجتماعي)',
    "philosophy": "subject:(philosophy OR فلسفة)",
    "science": "subject:(science OR علوم OR scientific) NOT fiction",
    "poetry": "subject:(poetry OR poems OR شعر OR ديوان)",
    "bio": "subject:(biography OR autobiograph* OR سيرة OR تراجم OR أعلام)",
    "politics": 'subject:(politics OR "political science" OR سياسة)',
    "travel": "subject:(travel OR voyages OR رحلات OR رحلة)",
}

GENRE_KEYWORDS = {
    "fiction": ("fiction", "novel", "novels", "رواية", "روايات"),
    "short": ("short stories", "short story", "قصص", "قصة"),
    "history": ("history", "تاريخ", "historical"),
    "sociology": ("sociology", "social sciences", "اجتماع", "اجتماعي"),
    "philosophy": ("philosophy", "فلسفة"),
    "science": ("science", "علوم", "scientific"),
    "poetry": ("poetry", "poems", "شعر", "ديوان"),
    "bio": ("biography", "autobiograph", "سيرة", "تراجم", "أعلام"),
    "politics": ("politics", "political science", "سياسة"),
    "travel": ("travel", "voyages", "رحلات", "رحلة"),
}


def genre_label(genre_id: str | None) -> str:
    return GENRE_LABELS.get(genre_id or "any", "أي مجال")


def _join_meta_field(val: object) -> str:
    if val is None:
        return ""
    if isinstance(val, (list, tuple)):
        parts = [str(x).strip() for x in val if str(x).strip()]
        return ", ".join(parts)
    return str(val).strip()


def creator_from_metadata(meta: dict | None) -> str:
    md = (meta or {}).get("metadata") if isinstance(meta, dict) else None
    if not isinstance(md, dict):
        md = meta if isinstance(meta, dict) else {}
    return _join_meta_field(md.get("creator"))


def _lang_filter(language: str) -> str:
    if language == "eng":
        return "(language:eng OR language:English) AND mediatype:texts"
    if language == "rus":
        return "(language:rus OR language:Russian OR language:ru) AND mediatype:texts"
    if language == "urd":
        return "(language:urd OR language:Urdu OR language:ur) AND mediatype:texts"
    return "(language:ara OR language:Arabic) AND mediatype:texts"


def _genre_query(language: str, genre: str) -> str:
    subject = GENRE_SUBJECT.get(genre) or ""
    lang_f = _lang_filter(language)
    fmt = "(format:DjVuTXT OR collection:gutenberg)"
    if subject:
        return f"{lang_f} AND {subject} AND {fmt}"
    return f"{lang_f} AND {fmt}"


def _prefer_genre(cands: list[dict], genre: str) -> list[dict]:
    keys = GENRE_KEYWORDS.get(genre) or ()
    if not keys:
        return list(cands)

    def score(c: dict) -> int:
        raw = c.get("raw") or {}
        subj = raw.get("subject")
        if isinstance(subj, list):
            subj_s = " ".join(str(s) for s in subj)
        else:
            subj_s = str(subj or "")
        blob = " ".join(
            [str(c.get("title") or ""), str(c.get("author") or ""), subj_s]
        ).lower()
        return sum(1 for k in keys if k.lower() in blob)

    return sorted(cands, key=score, reverse=True)


def list_candidates(used: set[str] | None = None, language: str = "ara", genre: str | None = None) -> list[dict]:
    used = used or set()
    seen: set[str] = set()
    out: list[dict] = []
    preferred = PREFERRED_BY_LANG.get(language) or PREFERRED_ARABIC
    queries = SEED_QUERIES_BY_LANG.get(language) or SEED_QUERIES_AR
    genre_id = (genre or "").strip() or None
    if genre_id == "any":
        genre_id = None

    def add(ident: str, title: str, raw: dict | None = None) -> None:
        if not ident or ident in seen or ident in used:
            return
        if not _title_ok(title):
            return
        seen.add(ident)
        author = _join_meta_field((raw or {}).get("creator")) if raw else ""
        out.append({"identifier": ident, "title": title, "author": author, "raw": raw or {}})

    for ident in preferred:
        add(ident, ident)

    if not ALLOW_ARCHIVE_SEARCH:
        return out

    search_queries = [_genre_query(language, genre_id)] if genre_id else list(queries)
    rows = 25 if genre_id else 20
    for query in search_queries:
        try:
            docs = search_archive(query, rows=rows)
        except Exception as exc:  # noqa: BLE001
            log.warning("search failed: %s", exc)
            continue
        for doc in docs:
            ident = str(doc.get("identifier") or "")
            title = str(doc.get("title") or ident)
            add(ident, title, doc)
        if len(out) >= 12:
            break
    return out


def pick_next_book(used: set[str] | None = None, language: str = "ara", genre: str | None = None) -> dict:
    """Return {identifier, title, author, text, installments, metadata} for the next usable book."""
    errors: list[str] = []
    tried: set[str] = set()
    genre_id = (genre or "").strip() or None
    if genre_id == "any":
        genre_id = None

    def try_candidate(cand: dict) -> dict | None:
        ident = cand["identifier"]
        if ident in tried:
            return None
        tried.add(ident)
        meta = get_metadata(ident)
        title = str((meta.get("metadata") or {}).get("title") or cand["title"] or ident)
        if not _title_ok(title):
            raise ValueError("title skipped")
        text, title, meta = load_book_text(ident, metadata=meta)
        ratio = arabic_ratio(text)
        cyr = cyrillic_count(text)
        if language == "ara" and ratio < MIN_ARABIC_RATIO:
            raise ValueError(f"not Arabic enough ratio={ratio:.3f}")
        if language == "eng" and ratio >= 0.25 and cyr < 500:
            # English folk tales should be mostly Latin.
            latin = sum(1 for c in text if ("A" <= c <= "Z") or ("a" <= c <= "z"))
            if latin < 3000:
                raise ValueError(f"not English enough ar_ratio={ratio:.3f}")
        if language == "rus" and cyr < 2000:
            raise ValueError(f"not Russian enough cyr={cyr}")
        if language == "urd" and arabic_script_count(text) < 2000:
            raise ValueError(f"not Urdu/Arabic-script enough")
        installments = slice_installments(text)
        if len(installments) < 1:
            raise ValueError("no installments")
        author = creator_from_metadata(meta) or cand.get("author") or ""
        log.info(
            "picked book lang=%s genre=%s %s (%s) installments=%s chars=%s",
            language, genre_id or "any", ident, title[:60], len(installments), len(text),
        )
        return {
            "identifier": ident,
            "title": title,
            "author": author,
            "text": text,
            "installments": installments,
            "metadata": meta,
            "arabic_ratio": ratio,
            "language": language,
            "genre": genre_id,
        }

    def run_list(cands: list[dict]) -> dict | None:
        for cand in cands:
            ident = cand["identifier"]
            try:
                got = try_candidate(cand)
            except Exception as exc:  # noqa: BLE001
                msg = f"{ident}: {exc}"
                errors.append(msg)
                log.warning("skip book %s", msg)
                continue
            if got:
                return got
        return None

    found = run_list(list_candidates(used, language=language, genre=genre_id))
    if found:
        return found
    if genre_id:
        log.warning(
            "no usable book for lang=%s genre=%s; falling back to broader query",
            language,
            genre_id,
        )
        fallback = list_candidates(used, language=language, genre=None)
        fallback = _prefer_genre(fallback, genre_id)
        found = run_list(fallback)
        if found:
            return found
    raise RuntimeError("No usable archive.org book found. Tried: " + "; ".join(errors[:8]))


def representative_text(text: str, max_chars: int = 80000) -> str:
    """Take beginning + middle + end so a whole-book summary has coverage."""
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    third = max_chars // 3
    mid = len(text) // 2
    start = text[:third]
    middle = text[mid: mid + third]
    end = text[-third:]
    return "\n\n".join([start, middle, end])


def get_book(identifier: str) -> dict:
    text, title, meta = load_book_text(identifier)
    installments = slice_installments(text)
    return {
        "identifier": identifier,
        "title": title,
        "author": creator_from_metadata(meta),
        "text": text,
        "installments": installments,
        "metadata": meta,
        "arabic_ratio": arabic_ratio(text),
    }



_FRONT_MATTER_MARKERS = (
    "contents", "table of contents", "list of illustrations", "printed by",
    "copyright", "all rights reserved", "university press", "not to be taken",
    "the branch libraries", "new york public library", "preface", "introduction",
    "translator's preface", "advertisement", "published by", "editorial note",
    "to the reader", "bibliographical",
)

_JEST_SIGNAL = (
    "jest", "merry", "tale", "anecdote", "fool", "wife", "husband", "priest",
    "friar", "merchant", "landlord", "neighbour", "neighbor", "laughed", "laugh",
    "said he", "said she", "answered", "reply", "witty", "knave", "miser",
    "drunk", "inn", "tavern", "coffee", "gentleman", "lady", "servant",
    "once", "fellow", "rogue", "cuckold", "parson", "farmer",
)


def is_usable_jest_slice(text: str, index: int = 0) -> bool:
    """Reject OCR front matter/TOC/editorial notes; keep merry-tale narrative."""
    raw = (text or "").strip()
    if not raw:
        return False
    words = raw.split()
    if len(words) < 80:
        return False
    low = raw.lower()
    editorial = (
        "editor", "preface", "introduction", "bibliographical", "collation",
        "verso", "folio", "hazlitt", "corrigenda", "errata", "apparatus",
        "notes to", "variant reading", "our text has been corrected",
        "project gutenberg", "gutenberg ebook", "gutenberg license",
        "start of this project gutenberg", "end of the project gutenberg",
        "produced by", "transcriber",
    )
    if sum(1 for m in editorial if m in low) >= 2:
        return False
    if any(x in low for x in ("project gutenberg", "gutenberg ebook", "gutenberg license")):
        return False
    # Skip scholarly Latin notes / critical apparatus masquerading as tales
    latinish = sum(1 for m in ("quod ", "cum ", "inquit", "narratvr", "vxorem", "fol.", "speculum") if m in low)
    if latinish >= 3:
        return False
    hits = sum(1 for m in _FRONT_MATTER_MARKERS if m in low)
    if hits >= 2:
        return False
    if index <= 2 and hits >= 1:
        return False
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    short_num = sum(1 for ln in lines[:40] if len(ln) < 40 and any(ch.isdigit() for ch in ln))
    if short_num >= 12 and index <= 3:
        return False
    # Need social/humor narrative signal
    sig = sum(1 for m in _JEST_SIGNAL if m in low)
    if sig < 3:
        return False
    # Prefer slices that look like spoken anecdote (dialogue markers)
    dialogue = sum(low.count(x) for x in ("said ", " quoth", "answered", "replied", "he said", "she said"))
    if dialogue < 1 and sig < 5:
        return False
    return True


# Back-compat alias
def is_usable_history_slice(text: str, index: int = 0) -> bool:
    return is_usable_jest_slice(text, index=index)




def pick_next_story(used: set[str] | None = None, language: str = "eng") -> dict:
    """Return one unused English jestbook / merry-tale slice.

    `used` holds keys like "identifier#index" from state.used_stories().
    Anti-repeat: never return a key already in `used`.
    """
    if language == "ara":
        language = "eng"  # sources must be non-Arabic
    used = set(used or [])
    errors: list[str] = []
    # Collect candidate books for this language (book-level used = all indices consumed is rare;
    # we skip a book only if we cannot find any free story index).
    book_used: set[str] = set()
    # Prefer collections still having free story slots — try preferred first.
    cands = list_candidates(used=book_used, language=language, genre=None)
    # Also force preferred ids even if search fails.
    preferred = PREFERRED_BY_LANG.get(language) or []
    seen_ids = {c["identifier"] for c in cands}
    for ident in preferred:
        if ident not in seen_ids:
            cands.insert(0, {"identifier": ident, "title": ident, "author": "", "raw": {}})
            seen_ids.add(ident)

    for cand in cands:
        ident = cand["identifier"]
        try:
            meta = get_metadata(ident)
            title = str((meta.get("metadata") or {}).get("title") or cand["title"] or ident)
            if not _title_ok(title):
                raise ValueError("title skipped")
            text, title, meta = load_book_text(ident, metadata=meta)
            # Language sanity (best-effort)
            ratio = arabic_ratio(text)
            cyr = cyrillic_count(text)
            if language == "ara" and ratio < MIN_ARABIC_RATIO:
                raise ValueError(f"not Arabic enough ratio={ratio:.3f}")
            if language == "rus" and cyr < 1500:
                raise ValueError(f"not Russian enough cyr={cyr}")
            stories = slice_stories(text)
            if not stories:
                raise ValueError("no stories")
            for idx, story_text in enumerate(stories):
                key = story_id(ident, idx)
                if key in used:
                    continue
                # Skip absurdly tiny / front-matter / non-history slices
                if len(story_text.split()) < 80:
                    continue
                if not is_usable_jest_slice(story_text, index=idx):
                    log.info("skip non-jest/front-matter slice %s#%s", ident, idx)
                    continue
                author = creator_from_metadata(meta) or cand.get("author") or ""
                story_title = (
                    extract_jest_title(story_text, collection_title=title)
                    or extract_history_episode_title(story_text)
                    or ""
                )
                if story_title and story_title.lower() in {title.lower(), ident.lower()}:
                    story_title = ""
                if not story_title:
                    # Last resort: short opening phrase, never the book title alone
                    opener = " ".join(story_text.split()[:12]).strip()
                    story_title = (opener[:70] + "…") if len(opener) > 70 else (opener or f"Jest {idx+1}")
                log.info(
                    "picked story lang=%s %s#%s title=%s book=%s words=%s",
                    language, ident, idx, story_title[:50], title[:40], len(story_text.split()),
                )
                return {
                    "identifier": ident,
                    "title": story_title,
                    "collection_title": title,
                    "author": author,
                    "text": story_text,
                    "story_index": idx,
                    "language": language,
                    "story_key": key,
                    "metadata": meta,
                }
            log.info("all story slices used for %s (%s slices)", ident, len(stories))
        except Exception as exc:  # noqa: BLE001
            msg = f"{ident}: {exc}"
            errors.append(msg)
            log.warning("skip story source %s", msg)
            continue

    raise RuntimeError(
        "No unused jestbook tale found on archive.org. Tried: "
        + "; ".join(errors[:10])
    )

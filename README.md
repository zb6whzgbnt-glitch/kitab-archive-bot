# بوت أرشيف الكتاب

بوت تيليجرام يرسل مرتين يومياً (08:00 و 20:00 بتوقيت الرياض، كل أيام الأسبوع) حلقة ملخّصة بالعربية من كتاب في الملك العام على [أرشيف الإنترنت](https://archive.org)، مقسّمة على عدة رسائل، ثم يُرسَل فيديو 1920×1080 واحد (بطاقة عنوان + صوت TTS) عبر `sendVideo` — لا صوت منفصل.

يعمل **بدون مفتاح ذكاء اصطناعي مدفوع** (تلخيص استخراجي + ترجمة إلى العربية). عند ضبط `GEMINI_API_KEY` تُكتب الملخصات عبر Gemini بأسلوب عربي احترافي.

## التشغيل

```bash
cd /workspace/kitab-archive-bot
```

1. ضع توكن البوت من [@BotFather](https://t.me/BotFather) في ملف `.env`:

```bash
TELEGRAM_BOT_TOKEN=ضع_التوكن_هنا
```

أو انسخ المثال:

```bash
cp .env.example .env
# ثم حرّر .env
```

2. ثبّت الاعتمادات (مرة واحدة):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. شغّل البوت (استطلاع طويل + الجدولة):

```bash
python -m bot
# أو
python bot.py
```

إذا كان التوكن فارغاً أو مرفوضاً من تيليجرام (401) يتوقف البرنامج برسالة واضحة **دون** إرسال أي رسالة.

4. في تيليجرام أرسل `/start` للاشتراك.

5. للتجربة الفورية أرسل `/now` ليصلك الملخص التالي نصاً ثم فيديو.

## الأوامر

| الأمر | الوظيفة |
|---|---|
| `/start` | اشتراك + رسالة ترحيب |
| `/stop` | إلغاء الاشتراك |
| `/status` | الكتاب الحالي والتقدم |
| `/next` | إرسال الحلقة التالية لهذه المحادثة (للاختبار) |

## تجربة بدون تيليجرام (أرشيف + تلخيص + فيديو)

```bash
source .venv/bin/activate
python dry_run.py
```

يُحفظ نموذج صوتي في `data/audio/sample.mp3` ونص الأجزاء في `data/audio/sample_parts.txt`.

## دوكر (اختياري)

```bash
docker compose up --build
```

ضع التوكن في `.env` قبل التشغيل.

## ملاحظات

- الحالة تُحفظ في `data/state.json` (معرّفات المحادثات، الكتاب الحالي، رقم الحلقة).
- الملفات الصوتية/الفيديو في `data/audio/` و `data/video/` والنصوص المخزّنة في `data/books/` وملف `.env` كلها خارج git.
- الفيديو: بطاقة عنوان عربية 1920×1080 + صوت `ar-SA-ZariyahNeural` (احتياطي `ar-SA-HamedNeural`) عبر `edge-tts` ثم `ffmpeg`. يُرسَل كفيديو واحد، لا ملف MP3 منفصل.


## GitHub Actions hosting

Scheduled delivery (Asia/Riyadh): **08:00 / 12:00 / 16:00 / 20:00** via `.github/workflows/deliver-story.yml`.

Manual run: GitHub → Actions → **Deliver story** → Run workflow.

Required repository secrets:
- `TELEGRAM_BOT_TOKEN`
- `GEMINI_API_KEY` (legacy fallback)
- `GEMINI_STORY_API_KEY`
- `GEMINI_TASHKEEL_API_KEY`
- `GEMINI_API_KEY_2` (optional)
- `BOOTSTRAP_SUBSCRIBERS` (e.g. `8581434211,8415608677`)

Notes:
- Actions is **not** always-on: Telegram commands like `/now` and `/gemini` do not work on this hosting mode.
- `data/state.json` is committed after each run so used-stories / subscribers persist.
- Stop any Railway/local polling instance to avoid Telegram `getUpdates` Conflict.

الاستضافة على GitHub Actions: قصص مجدولة أربع مرات يومياً + تشغيل يدوي. أوامر تيليجرام التفاعلية غير متاحة في هذا الوضع.

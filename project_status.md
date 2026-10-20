# AI Visualizer — Project Status
_Last reviewed: October 19, 2026_

---

## What This Project Is

A Python pipeline that takes a narration script (TXT or Markdown) plus an optional audio file (MP3 or WAV) and produces:
- Semantically grouped scenes with real timestamps
- AI-generated image prompts per scene (via Gemini)
- Numbered images per scene (via Stability AI, Replicate, or placeholder)
- CSV and JSON exports of the full scene data
- A Gradio web UI to drive the whole process interactively

---

## Current File Structure

```
ai_visualizer/
  __init__.py           version = "0.1.0"
  __main__.py           entry point → cli.main()
  scene_parser.py       script → Scene list
  timestamp_mapper.py   audio duration → proportional timestamps
  exporter.py           Scene list → CSV + JSON
  gemini_client.py      Gemini API → image_prompt per scene
  image_generator.py    image_prompt → numbered PNGs
  cli.py                argparse CLI (analyze + ui subcommands)
  ui.py                 Gradio 3-tab web interface

tests/
  test_scene_parser.py      6 tests
  test_timestamp_mapper.py  5 tests
  test_exporter.py          6 tests
  test_image_generator.py   5 tests

outputs/                    last real run against sample_script.md
  scenes.csv
  scenes.json
  scenes_with_images.csv
  images/scene_001.png … scene_008.png

sample_script.md            "The Rise of AI" — 6-section test script
requirements.txt
.env.example
.gitignore
README.md
```

---

## Phase-by-Phase Status

### Phase 1 — Script → Scenes → Timestamps → CSV/JSON ✅ COMPLETE

**What works:**
- `scene_parser.py` handles TXT and Markdown
- Three grouping strategies working and tested:
  1. Explicit markers (`##`, `---`, `[SCENE]`, etc.) force a new scene
  2. Blank lines (paragraph breaks) flush the current scene
  3. Soft sentence-count cap (5 sentences) catches walls of text with no breaks
- Markdown formatting (bold, italic, inline code) is stripped before export
- `timestamp_mapper.py` reads audio duration via mutagen (fast, no decoding) with pydub as fallback
- Timestamps are distributed proportionally by word count — no ASR needed
- Fallback mode estimates timestamps at 130 wpm when no audio is provided
- `exporter.py` writes CSV and JSON with all scene fields
- Full CLI pipeline: `python -m ai_visualizer analyze --script … --audio …`

**Known limitation:**
- Timestamps are proportional estimates, not word-aligned. Good for previewing; not frame-accurate for final video assembly. Whisper forced-alignment would improve this but is not needed yet.

**Verified output (sample_script.md, no audio):**
```
Scene 1:  [0.0s –   1.8s]  "The Rise of AI"         (title-only scene — known issue, see below)
Scene 2:  [1.8s –  18.5s]  "The Quiet Beginning"
Scene 3:  [18.5s – 36.0s]  "The First Wave"
Scene 4:  [36.0s – 51.2s]  "The Long Winter"
Scene 5:  [51.2s – 68.3s]  "The Neural Revolution"
Scene 6:  [68.3s – 82.6s]  "The Data Explosion"
Scene 7:  [82.6s – 84.5s]  "Where We Stand Today"   (title-only scene — known issue)
Scene 8:  [84.5s – 102.0s] "Today, AI writes code…"
```

---

### Phase 2 — Gemini Image Prompt Generation ✅ WIRED, UNTESTED WITH REAL KEY

**What works:**
- `gemini_client.py` loads key from `.env` only — never hardcoded
- `is_available()` check guards all calls: returns `False` if key is missing or placeholder
- Uses `gemini-1.5-flash` by default (cheapest/fastest)
- Skips scenes that already have an `image_prompt` (safe to re-run)
- Per-scene error handling: one failure doesn't abort the batch
- 1-second polite delay between calls to avoid rate limits
- `enhance_single()` exists for UI retry flow
- CLI: add `--gemini` flag to any `analyze` run
- UI Tab 2 has an "Enhance with Gemini" button

**Not yet tested:**
- No `.env` file exists yet (only `.env.example`) — real API call has not been made
- The Gemini SDK (`google-generativeai`) is listed in `requirements.txt` but not confirmed installed

**Risk:**
- `google-generativeai==0.5.4` is pinned to a version from mid-2024. The current SDK may have a different interface. Needs a quick import check before first real use.

---

### Phase 3 — Image Generation ✅ PLACEHOLDER COMPLETE, REAL BACKENDS UNTESTED

**What works:**
- Job queue iterates scenes, skips scenes without prompts, skips existing files
- `placeholder` backend creates real labeled PNGs (1024×576) using Pillow — no API needed
- Placeholder includes: scene number, title, prompt text, timestamp
- `stability` backend code written (Stability AI SDXL v1.0 endpoint)
- `replicate` backend code written (SDXL via Replicate)
- All images saved as `scene_NNN.png` (zero-padded, 3 digits)
- Auto-exports `scenes_with_images.csv` after generation
- CLI fallback: if `--images` is passed without `--gemini`, scene text is used as the prompt
- 8 placeholder images generated and confirmed working

**Not yet tested:**
- `stability` backend: endpoint URL hardcoded to SDXL v1.0. Stability AI has since released newer models (SD3, SDXL Turbo). The URL may still work but is worth confirming.
- `replicate` backend: model hash pinned (`39ed52f2a…`). Pinned hashes are stable on Replicate so this should be fine.
- No error-retry logic. If one image fails mid-batch, it logs and continues, but there's no automatic retry or resume-from-failure.

---

### Phase 4 — Gradio Web UI ✅ BUILT, NOT LAUNCHED YET

**What exists (3 tabs):**

| Tab | Name | What it does |
|-----|------|-------------|
| 1 | Upload & Analyze | Upload script + optional audio → scene table + CSV/JSON paths |
| 2 | Review & Enhance | Editable scene table, Gemini enhance button, save edits back to `_SCENES` |
| 3 | Generate Images | Backend picker, generate button, image gallery |

**Structural issues to be aware of before first launch:**

1. **Global mutable state** — `_SCENES` is a module-level list. This works fine for a single user but will cause session collisions if two users run the UI at the same time. Fine for local use; not safe for shared/hosted deployment.

2. **Temp dir is session-scoped** — `_TMP` is created once at import time. If the UI is restarted, the path changes. Downloads/links from a previous session won't work after restart.

3. **Tab 2 is disconnected from Tab 1's table** — Tab 1 shows `scenes_table_1`, Tab 2 has its own `scenes_table_2`. They both read from `_SCENES` but they're separate Gradio components. A user who edits the table in Tab 2 won't see their edits reflected back unless they click "Save Edits" first.

4. **No download buttons** — Users can see CSV/JSON paths in the status text but can't download files directly from the UI. They have to find them manually in the temp directory.

5. **`gradio` not confirmed installed** — It's in `requirements.txt` but installation was done piecemeal. Needs `pip install gradio` before first launch.

---

## Tests

**22 tests, all passing. Zero API keys required to run them.**

```
tests/test_scene_parser.py       6/6  ✅
tests/test_timestamp_mapper.py   5/5  ✅
tests/test_exporter.py           6/6  ✅
tests/test_image_generator.py    5/5  ✅
```

Run with: `python -m pytest tests/ -v`

One bug was caught and fixed during initial test run:
- Soft-cap sentence counter only fired per-line, not per-sentence within a line. Walls of text (no newlines) were never split. Fixed by iterating sentences individually.

**No tests exist for:**
- `gemini_client.py` (requires live API key — appropriate to skip in CI)
- `ui.py` (no UI tests of any kind)
- CLI integration (no end-to-end subprocess tests)
- Real audio timestamp mapping (requires an actual audio file)

---

## Known Issues / Bugs

| # | Severity | Description |
|---|----------|-------------|
| 1 | Low | Section title lines (`## The Quiet Beginning`) with no body text become their own 1–3 word scenes (e.g., Scene 1 "The Rise of AI", Scene 7 "Where We Stand Today"). These short title-only scenes get proportionally tiny timestamps and waste an image slot. They should be merged into the following scene or filtered by minimum word count. `MIN_WORDS_FOR_SCENE = 4` partially helps but "The Rise of AI" is exactly 4 words and still passes. |
| 2 | Low | `requirements.txt` lists `nltk==3.8.1` and `spacy==3.7.4` but neither is imported anywhere in the codebase. They were likely planned for smarter NLP-based scene splitting. Leaving them in adds ~500MB of unnecessary download. |
| 3 | Low | `requirements.txt` pins `python-dotenv==1.0.1` but `1.0.1` doesn't exist — the actual version installed was `1.2.4`. Minor version mismatch won't cause issues but `pip install -r requirements.txt` may warn or fail strictly. |
| 4 | Medium | The Stability AI endpoint URL is pinned to SDXL v1.0 (`stable-diffusion-xl-1024-v1-0`). Stability AI has changed its API structure since then. Worth confirming before spending credits. |
| 5 | Low | No `.env` file exists. The project cannot reach Phase 2 or real Phase 3 backends until one is created from `.env.example`. |

---

## Dependencies Status

| Package | In requirements.txt | Confirmed installed | Used |
|---------|--------------------|--------------------|------|
| python-dotenv | ✅ (wrong version) | ✅ 1.2.4 | ✅ |
| pydub | ✅ | ✅ | ✅ |
| mutagen | ✅ | ✅ | ✅ |
| nltk | ✅ | ❓ | ❌ not used |
| spacy | ✅ | ❓ | ❌ not used |
| google-generativeai | ✅ | ❓ | ✅ (Phase 2) |
| Pillow | ✅ | ✅ | ✅ |
| requests | ✅ | ✅ (pre-existing) | ✅ |
| gradio | ✅ | ❓ | ✅ (Phase 4) |
| pandas | ✅ | ✅ (pre-existing) | ✅ |
| tqdm | ✅ | ✅ (pre-existing) | ❌ imported in req but not used in code |

---

## What's Missing (Not Built)

These were discussed but not implemented:

1. **Video assembly** — No module combines images + audio into a video file (e.g., via `moviepy` or `ffmpeg`). The pipeline stops at "images + timestamps in a CSV."

2. **Word-level timestamp alignment** — Timestamps are proportional estimates. Whisper or a forced-alignment tool (e.g., `whisper-timestamped`, `aeneas`) would give per-word accuracy.

3. **Session persistence / project save-load** — No way to save a session and resume it later. Every UI session starts fresh.

4. **Download buttons in UI** — Can't download CSV/JSON/images from the Gradio interface. Files only exist in a temp directory.

5. **Retry / resume image generation** — If a batch of 20 images fails at #15, there's no way to resume from #15. The `skip_existing` flag partially handles this at the file level but the session state is lost on restart.

6. **Real audio + script test** — The pipeline has been tested with `sample_script.md` and estimated timestamps only. It has not yet been tested with a real ElevenLabs MP3 to confirm proportional timestamps feel accurate.

---

## Recommended Next Steps (Priority Order)

1. **Test with a real ElevenLabs audio file** before anything else. Upload your narration MP3 and run Phase 1. Confirm the proportional timestamps feel close enough, or decide if word-alignment is worth adding.

2. **Fix the title-only scene problem** (Issue #1). Merge short title-only scenes (< ~6 words, no body text) into the following scene instead of creating a standalone scene.

3. **Create `.env` and test Gemini** (Phase 2). Copy `.env.example` to `.env`, add your key, run `python -m ai_visualizer analyze --script sample_script.md --gemini`, and verify the image prompts look good before generating real images.

4. **Confirm the Stability AI endpoint** before spending credits. A quick `curl` test or checking their current docs will tell you if the pinned URL still works.

5. **Clean up `requirements.txt`** — Remove `nltk`, `spacy`, and `tqdm` (unused), fix the `python-dotenv` version to `1.2.4`.

6. **Add download buttons to the UI** so outputs are accessible without digging through temp directories.

7. **Add video assembly** as a new module if the goal is a finished video. `moviepy` can combine the numbered images with the audio using the timestamps from the CSV.

---

## How to Run Right Now

```bash
# Install what's needed
pip install python-dotenv pydub mutagen Pillow pandas pytest

# Run all tests (no API keys needed)
python -m pytest tests/ -v

# Phase 1 — script only
python -m ai_visualizer analyze --script sample_script.md

# Phase 1 — with audio (your ElevenLabs MP3)
python -m ai_visualizer analyze --script your_script.txt --audio your_audio.mp3

# Phase 1 + placeholder images (safe, no credits)
python -m ai_visualizer analyze --script sample_script.md --images

# Launch the web UI (requires gradio)
pip install gradio
python -m ai_visualizer ui
```

# AI Visualizer

Automate scenes, timestamps, and images from a narration script and audio file.

## Phases

| Phase | What it does |
|-------|-------------|
| 1 | Parse TXT/MD script + MP3/WAV audio → scenes CSV & JSON with real timestamps |
| 2 | Gemini API enriches each scene with a cinematic image prompt |
| 3 | Queue image jobs → numbered PNGs → final CSV with image paths |
| 4 | Gradio web UI — upload, analyze, review, generate |

## Quick start

```bash
pip install -r requirements.txt
```

### Phase 1 — Script only (estimated timestamps)
```bash
python -m ai_visualizer analyze --script sample_script.md
```

### Phase 1 — Script + audio (real timestamps)
```bash
python -m ai_visualizer analyze --script my_script.txt --audio narration.mp3
```

### Phase 1 + 3 — With placeholder images (no credits)
```bash
python -m ai_visualizer analyze --script my_script.txt --audio narration.mp3 --images
```

### Phase 1 + 2 + 3 — Full pipeline with Gemini prompts + real images
```bash
# 1. Copy .env.example to .env and set GEMINI_API_KEY and IMAGE_BACKEND
python -m ai_visualizer analyze \
  --script my_script.txt \
  --audio narration.mp3 \
  --gemini \
  --images \
  --backend stability
```

### Phase 4 — Web UI
```bash
python -m ai_visualizer ui
# Open http://localhost:7860
```

## Scene grouping

Scenes are grouped **by meaning, not by sentence count**. A new scene starts when:
- An explicit marker is found (`##`, `---`, `[SCENE]`, etc.)
- A blank line separates paragraphs
- A sentence-count soft cap is reached (default: 5 sentences)

## API keys

Copy `.env.example` to `.env` and fill in your keys. Keys are never stored in code.

```
GEMINI_API_KEY=...
IMAGE_BACKEND=stability      # or replicate or placeholder
STABILITY_API_KEY=...
REPLICATE_API_TOKEN=...
```

## Testing

```bash
python -m pytest tests/ -v
```

22 tests, all passing. No API keys required.

## Output structure

```
outputs/
  scenes.csv                 # Phase 1 output
  scenes.json                # Phase 1 output
  scenes_gemini.csv          # Phase 2 output (if --gemini)
  scenes_with_images.csv     # Phase 3 output (includes image paths)
  images/
    scene_001.png
    scene_002.png
    ...
```

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

### Phase 1 + 3 — With mock images (no credits)
```bash
python -m ai_visualizer analyze --script my_script.txt --audio narration.mp3 --images
```

### Generate images from a planned scene file
```bash
# Validate the plan and show the work without writing images or calling a backend.
python -m ai_visualizer generate-images --plan scene_plan.json --backend mock --dry-run

# Generate labeled mock images; no credentials, network, or GPU required.
python -m ai_visualizer generate-images --plan scene_plan.json --backend mock
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

## Scene-plan contract

`generate-images` reads a UTF-8 JSON array from `scene_plan.json`. Each scene
is an object with these fields:

| Field | Type | Purpose |
|-------|------|---------|
| `index` | integer | 1-based stable scene ID |
| `title` | string | Short scene title |
| `text` | string | Original scene text |
| `start_time`, `end_time` | number | Scene timestamps in seconds |
| `image_prompt` | string | Prompt sent to the selected image backend |
| `image_path` | string | Existing planner value; generation does not rewrite the plan |
| `description` | string | Planner-generated scene description |
| `filename_suggestion` | string | Stable PNG filename, normally `scene_NNN.png` |
| `segment_ids` | array of integers | Transcript segments represented by the scene |
| `word_count` | integer | Planner word count |

The planner also writes a CSV with these scene columns; `segment_ids` is
semicolon-separated in CSV. Image generation only reads the scene plan. It
leaves all scene text and timestamps unchanged and writes images and a
separate manifest.

## Image backends

Copy `.env.example` to `.env`. `IMAGE_BACKEND` selects exactly one backend;
generation errors do not trigger a fallback.

```
IMAGE_BACKEND=mock           # mock | api | local
GEMINI_API_KEY=...           # only needed for api
GEMINI_IMAGE_MODEL=gemini-nano-banana-2.1
COMFYUI_BASE_URL=http://127.0.0.1:8188
COMFYUI_WORKFLOW=C:\path\to\workflow_api.json
COMFYUI_MODEL=               # optional model override for a single loader node
```

The Gemini image backend uses the official Gemini Interactions REST API and
the configurable `GEMINI_IMAGE_MODEL`. The key is sent in an HTTP header, not
logged or placed in frontend code. No API request is made by setup, dry-run,
or tests.

For the local backend, start ComfyUI yourself and export a workflow in **API
format**. Put `{{prompt}}` in the positive text input. The workflow must
include a `SaveImage` node and one supported model loader
(`CheckpointLoaderSimple`, `CheckpointLoader`, or `UNETLoader`). Before
submitting a job, the backend checks ComfyUI connectivity, workflow node
availability, and the selected checkpoint/diffusion model. Aspect ratios are
applied to supported empty-latent nodes. Use `COMFYUI_MODEL` only when the
workflow has exactly one supported model loader. Models and ComfyUI are
installed and managed separately; the application does not download models
or assume any particular GPU/VRAM capacity.

The mock backend writes clearly labeled PNGs that are not AI-generated.
All backends use stable scene filenames, a resumable manifest, bounded retries,
and concurrency of one by default. Increase concurrency with `--concurrency`
only if the selected backend and hardware can safely handle it.

One-image commands (replace paths/settings as needed):

```bash
# First configure GEMINI_API_KEY in .env. This explicitly makes one real API request.
python -m ai_visualizer generate-images --plan scene_plan.json --backend api --scenes 1 --force --retries 0

# Configure a reachable ComfyUI server and API-format workflow in .env first.
python -m ai_visualizer generate-images --plan scene_plan.json --backend local --scenes 1 --force --retries 0
```

## Testing

```bash
python -m pytest tests -v
```

The suite uses mocked network responses and the mock backend; no API keys,
network calls, ComfyUI server, or GPU inference are required.

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
  image_manifest.json        # scene status, backend/model, prompt, output and errors
```

`generate-images` defaults to `outputs/images/` and writes its manifest to
`outputs/image_manifest.json`. The scene-plan JSON and CSV, including their
timestamps and scene contents, are inputs to this stage and are not rewritten.

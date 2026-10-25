# AI Visualizer

Automate scenes, timestamps, and images from a narration script and audio file.

## Phases

| Phase | What it does |
|-------|-------------|
| 1 | Parse TXT/MD script + MP3/WAV audio → scenes CSV & JSON with real timestamps |
| 2 | Gemini API enriches each scene with a cinematic image prompt |
| 3 | Queue image jobs → format-correct scene images → final CSV with image paths |
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

### Narrato Studio
```bash
python -m ai_visualizer ui
# Open http://localhost:7860
```

Studio loads `scene_plan.json` and the existing image manifest, then runs the
same `image_runner` pipeline as the CLI. Select **local** for a configured
ComfyUI server or **api** for Gemini. API generation requires an explicit
confirmation in the UI. Jobs run in a background worker; cancellation stops
queued scenes and waits for active backend calls to finish. Non-secret UI
preferences are saved to the current user's Narrato settings directory.
Existing image-generation callbacks are exposed in API docs. Gallery and file
payload schemas are excluded because they contain JSON Schema boolean forms
unsupported by the project's pinned Gradio client; manual project callbacks
are UI-only.

The shared **New Project** screen starts with narration audio and its script.
It accepts readable MP3/WAV audio and a UTF-8 TXT/Markdown file or pasted script
text; pasted text is saved as `script.txt`. Enter a project name, review the
audio and script, create the project, then explicitly choose **Assisted Visual
Creation** or **Automatic Visual Creation**. Projects are stored under
`projects/<safe-project-name>/` with stable `voiceover.*` and `script.*` files,
`project.json`, and `markers.json`. The schema-v1 project format is retained;
new projects add optional name, creation timestamp/state, and selected-mode
metadata, while existing projects without those fields remain loadable.
Duplicate project folders are not overwritten. No image generation starts
during project setup.

The **Assisted Visual Creation** workspace keeps existing project loading and
manual visual-marker tools. Load a saved project by entering its folder and
selecting **Load saved project**. The audio player supports playback and
scrubbing. Script paragraphs are available as navigable segments with their
own editable marker boundaries and image prompts; prompt edits are saved to
that segment's marker without changing unrelated markers.

Segment times are proportional estimates based on script word counts, not
word-level speech alignment. Adjust the marker boundaries while listening to
the audio. Selecting or navigating segments, editing a prompt, and loading a
project do not generate images. **Generate Image** and **Retry / regenerate this
segment** run only the selected segment through the configured local ComfyUI
workflow. The prompt is combined with Studio's quality prompts; output versions
are stored under the project `images/` folder and the selected marker's image
path is saved back into `markers.json`. Regeneration requires confirmation and
creates a new version rather than overwriting the previous image. Generation
progress and the selected segment's image preview appear in the workspace. This
feature has unit coverage with a mocked backend; live generation still depends
on a reachable ComfyUI service and a valid local workflow/model configuration.

The **Automatic Visual Creation** workspace reuses the selected project's
script, narration, and `markers.json`; it does not ask for the assets again.
On entry, Narrato creates or loads a reviewable plan at
`automatic/scene_plan.json` and associates scenes with the audio timeline using
proportional word-count timing estimates. These are estimates, not speech
alignment. Initial prompts are deterministic `[FALLBACK]` prompts derived from
scene text; inspect them in the scene table. The project's existing marker
prompts are authoritative during generation, including prompts edited in the
Assisted workspace. Loading the project or switching modes never starts image
generation.

Review the scene text, estimated times, prompts, statuses, and existing images,
then explicitly click **Start Automatic Generation**. Scenes are submitted
sequentially through the existing image runner and selected backend. New images
are saved under `automatic/images/`, with progress and results recorded in
`automatic/image_manifest.json` and corresponding image references persisted to
the project's existing `markers.json`; the scene gallery shows saved marker
images alongside the scene table. Successful scenes are skipped on later
starts unless replacement is explicitly enabled. Failed scenes can be retried
individually. Cancellation stops queued work at a safe point; an in-progress
backend call may finish before the worker stops. Generation requires a
configured ComfyUI API-format workflow for local runs, or configured Gemini
credentials and explicit confirmation for API runs. Automated tests mock
generation; live ComfyUI generation has not been verified as part of this
workflow.

The **Check backend readiness** button performs ComfyUI connectivity, workflow,
and model checks for local generation. For Gemini it only verifies local
configuration and displays the model; it does not make an API request.
Existing files are not replaced unless **Explicitly replace existing output
files** is checked. The gallery previews valid generated images; placeholder
images are labeled as such. Studio does not assemble a video.

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

Set backend configuration in the project-root `.env` file. Keep that file
private; `IMAGE_BACKEND` selects exactly one backend and generation errors do
not trigger a fallback.

```
IMAGE_BACKEND=local          # mock | api | local
GEMINI_API_KEY=...           # only needed for api
GEMINI_IMAGE_MODEL=gemini-nano-banana-2.1
COMFYUI_URL=http://127.0.0.1:8188
COMFYUI_WORKFLOW=C:\path\to\workflow_api.json  # set to your exported workflow
COMFYUI_MODEL=v1-5-pruned-emaonly.safetensors  # optional single-loader override
# Optional only when workflow node detection is ambiguous:
COMFYUI_POSITIVE_NODE_ID=
COMFYUI_NEGATIVE_NODE_ID=
```

`COMFYUI_BASE_URL` remains supported as a compatibility alias when
`COMFYUI_URL` is not set.

The Gemini image backend uses the official Gemini Interactions REST API and
the configurable `GEMINI_IMAGE_MODEL`. The key is sent in an HTTP header, not
logged or placed in frontend code. No API request is made by setup, dry-run,
or tests.

For the local backend, start ComfyUI yourself and export a workflow in **API
format** (in ComfyUI, use **Save (API Format)**). Put `{{prompt}}` in the
positive `CLIPTextEncode` text input and `{{negative_prompt}}` in the negative
one. Alternatively, Narrato detects the text nodes connected to the sampler's
`positive` and `negative` inputs. If either is ambiguous, set
`COMFYUI_POSITIVE_NODE_ID` or `COMFYUI_NEGATIVE_NODE_ID` to that workflow's
node ID; the Studio readiness check validates the choice. The workflow must
include a `SaveImage` node and one supported model loader
(`CheckpointLoaderSimple`, `CheckpointLoader`, or `UNETLoader`). Before
submitting a job, the backend checks ComfyUI connectivity, workflow node
availability, and the selected checkpoint/diffusion model. Aspect ratios are
applied to supported empty-latent nodes only when selected; **Workflow
dimensions** preserves the exported width/height and all checkpoint, sampler,
scheduler, steps, CFG, and seed settings. An explicit 3:2 override uses
768 × 512. Use `COMFYUI_MODEL` only when the workflow has exactly one supported
model loader. Models and ComfyUI are installed and managed separately; the
application does not download models or assume any particular GPU/VRAM capacity.

For the Phase 5 validation run, the configured workflow JSON existed and was
accepted by Narrato's configuration parser, but ComfyUI at
`http://127.0.0.1:8188` refused the `/system_stats` preflight connection. No
live image was generated, and workflow-node/model readiness could not be
confirmed against the running ComfyUI service. Start ComfyUI at the configured
`COMFYUI_URL` (or update that URL), then use **Check backend readiness**. A
successful HTTP preflight alone does not prove image generation; verify one
disposable scene's output image and marker/manifest persistence before relying
on a production run.

Narrato composes the scene prompt with reusable cinematic positive and negative
quality prompts. These are editable in Studio and saved with the other
non-secret Studio settings. Effective prompts are recorded in
`image_manifest.json`. Gemini receives negative guidance as text; ComfyUI
workflows use a dedicated negative text node.

The mock backend writes clearly labeled PNGs that are not AI-generated.
All backends use stable scene filenames, a resumable manifest, bounded retries,
and concurrency of one by default. Increase concurrency with `--concurrency`
only if the selected backend and hardware can safely handle it.

One-image commands (replace paths/settings as needed):

```bash
# First configure GEMINI_API_KEY in .env. This explicitly makes one real API request.
python -m ai_visualizer generate-images --plan scene_plan.json --output outputs/api_pilot --manifest outputs/api_pilot_manifest.json --backend api --scenes 1 --force --retries 0

# Configure COMFYUI_URL and the actual exported COMFYUI_WORKFLOW path first.
python -m ai_visualizer generate-images --plan scene_plan.json --output outputs/local_pilot --manifest outputs/local_pilot_manifest.json --backend local --scenes 1 --retries 0
```

From PowerShell, launch Studio with:

```powershell
Set-Location 'C:\Users\Anish\Desktop\AI visualizer'
python -m ai_visualizer ui
```

Select **Selected scenes**, choose one scene for a pilot, run **Check backend
readiness**, then start generation. Existing images are not overwritten unless
**Explicitly replace existing output files** is checked.

## Testing

```bash
python -m pytest tests -q
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
    scene_001.png            # mock and ComfyUI backends
    scene_001.jpg            # Gemini API backend
    ...
  image_manifest.json        # scene status, backend/model, prompt, output and errors
```

`generate-images` defaults to `outputs/images/` and writes its manifest to
`outputs/image_manifest.json`. The scene-plan JSON and CSV, including their
timestamps and scene contents, are inputs to this stage and are not rewritten.
The Gemini API returns JPEG, so those outputs and manifest filenames use
`.jpg`; mock and ComfyUI outputs remain PNG.

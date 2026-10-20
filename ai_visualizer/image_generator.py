"""
image_generator.py
------------------
Phase 3 – Image job queue.

Reads image prompts from scenes, queues generation jobs, saves numbered
images (scene_001.png, scene_002.png …), and exports a final CSV.

Supported backends (set IMAGE_BACKEND in .env):
  - mock : clearly labeled test image, no API or GPU
  - api  : Gemini image generation using GEMINI_API_KEY
  - local: ComfyUI local workflow and model
  - placeholder, stability, replicate: legacy compatibility options

The placeholder backend lets you test the entire pipeline end-to-end
without spending any image credits.
"""

from __future__ import annotations

import os
import json
import tempfile
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from .scene_parser import Scene
from .exporter import to_csv

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _output_dir(base: Path) -> Path:
    d = base / "images"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _image_filename(scene: Scene) -> str:
    return f"scene_{scene.index:03d}.png"


# ---------------------------------------------------------------------------
# Placeholder backend (no credits consumed)
# ---------------------------------------------------------------------------

def _generate_placeholder(scene: Scene, out_dir: Path) -> Path:
    """Create a simple labeled PNG using Pillow."""
    from PIL import Image, ImageDraw, ImageFont  # type: ignore

    img = Image.new("RGB", (1024, 576), color=(30, 30, 50))
    draw = ImageDraw.Draw(img)

    # Title
    title = f"Scene {scene.index}: {scene.title}"
    draw.text((30, 30), title, fill=(200, 200, 255))

    # Wrapped prompt text
    prompt = scene.image_prompt or scene.text
    lines: List[str] = []
    words = prompt.split()
    line: List[str] = []
    for word in words:
        line.append(word)
        if len(" ".join(line)) > 80:
            lines.append(" ".join(line[:-1]))
            line = [word]
    lines.append(" ".join(line))

    y = 80
    for ln in lines[:12]:  # max 12 lines
        draw.text((30, y), ln, fill=(180, 180, 180))
        y += 24

    # Timestamp
    ts = f"⏱  {scene.start_time:.1f}s – {scene.end_time:.1f}s"
    draw.text((30, 520), ts, fill=(100, 200, 100))

    out_path = out_dir / _image_filename(scene)
    img.save(str(out_path))
    return out_path


# ---------------------------------------------------------------------------
# Stability AI backend
# ---------------------------------------------------------------------------

def _generate_stability(scene: Scene, out_dir: Path) -> Path:
    import requests  # type: ignore

    api_key = os.getenv("STABILITY_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("STABILITY_API_KEY not set in .env")

    url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "text_prompts": [{"text": scene.image_prompt, "weight": 1.0}],
        "cfg_scale": 7,
        "height": 576,
        "width": 1024,
        "steps": 30,
        "samples": 1,
    }

    resp = requests.post(url, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    import base64
    image_data = base64.b64decode(data["artifacts"][0]["base64"])

    out_path = out_dir / _image_filename(scene)
    out_path.write_bytes(image_data)
    return out_path


# ---------------------------------------------------------------------------
# Replicate backend
# ---------------------------------------------------------------------------

def _generate_replicate(scene: Scene, out_dir: Path) -> Path:
    import replicate  # type: ignore
    import requests   # type: ignore

    token = os.getenv("REPLICATE_API_TOKEN", "").strip()
    if not token:
        raise RuntimeError("REPLICATE_API_TOKEN not set in .env")

    os.environ["REPLICATE_API_TOKEN"] = token
    output = replicate.run(
        "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
        input={"prompt": scene.image_prompt, "width": 1024, "height": 576},
    )
    image_url = output[0] if isinstance(output, list) else output

    resp = requests.get(image_url, timeout=60)
    resp.raise_for_status()

    out_path = out_dir / _image_filename(scene)
    out_path.write_bytes(resp.content)
    return out_path


# ---------------------------------------------------------------------------
# Job queue
# ---------------------------------------------------------------------------

_BACKENDS = {
    "placeholder": _generate_placeholder,
    "stability": _generate_stability,
    "replicate": _generate_replicate,
}


def generate_images(
    scenes: List[Scene],
    output_dir: str | Path = "outputs",
    backend: str | None = None,
    delay_seconds: float = 2.0,
    skip_existing: bool = True,
) -> List[Scene]:
    """
    Queue image generation for all scenes that have an image_prompt set.

    Parameters
    ----------
    scenes       : list of Scene objects (should have image_prompt set)
    output_dir   : root output folder
    backend      : 'mock' | 'api' | 'local' (legacy options remain supported)
                   Defaults to IMAGE_BACKEND env var, then 'mock'.
    delay_seconds: pause between API calls
    skip_existing: skip scenes whose image_path already points to a file

    Returns the same list with image_path fields populated.
    """
    from .image_runner import RunConfig, run_image_generation

    base_dir = Path(output_dir)
    out_dir = _output_dir(base_dir)
    scene_plan = []
    for scene in scenes:
        scene_data = scene.to_dict()
        scene_data["filename_suggestion"] = _image_filename(scene)
        scene_plan.append(scene_data)

    base_dir.mkdir(parents=True, exist_ok=True)
    temporary_plan = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        prefix=".image-plan-",
        dir=base_dir,
        delete=False,
    )
    plan_path = Path(temporary_plan.name)
    try:
        with temporary_plan:
            json.dump(scene_plan, temporary_plan, indent=2, ensure_ascii=False)
        results = run_image_generation(RunConfig(
            scene_plan_path=plan_path,
            output_dir=out_dir,
            manifest_path=base_dir / "image_manifest.json",
            backend=backend,
            delay_seconds=delay_seconds,
            force=not skip_existing,
        ))
        results_by_index = {result.index: result for result in results}
        for scene in scenes:
            result = results_by_index[scene.index]
            scene.image_path = (
                result.image_path if result.status in {"ok", "skipped"} else ""
            )
        csv_path = to_csv(scenes, base_dir / "scenes_with_images.csv")
        print(f"\n[ImageGen] Export → {csv_path}")
    finally:
        try:
            plan_path.unlink()
        except FileNotFoundError:
            pass

    return scenes

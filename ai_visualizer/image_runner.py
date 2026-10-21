"""
image_runner.py
---------------
Image-generation stage for the SceneFlow / AI Visualizer pipeline.

Reads scene_plan.json (produced by scene_planner.py), validates every
scene, then generates one image per scene using a pluggable backend.

Design principles
-----------------
- Independent of scene-planning: operates entirely from the JSON file.
- Resume-safe: skips scenes whose output file already exists (unless
  --force is passed).
- Retry-safe: failed scenes are logged in the manifest and can be retried
  on the next run.
- Rate-limited: a configurable delay between API calls prevents 429 errors.
- Dry-run: validates every scene and prints a plan without making any
  API calls or writing any files.
- Manifest: writes image_manifest.json after each scene so partial runs
  are inspectable.
- Backend-agnostic: mock, Gemini API, and ComfyUI share ImageBackend.
- No secrets in output: API keys are never written to logs or manifests.

Built-in backends
-----------------
  mock : visibly labeled local PNG, no API key or GPU
  api  : Gemini image generation using GEMINI_API_KEY
  local: ComfyUI API using a configured workflow and local model

Legacy placeholder, Stability, and Replicate adapters remain for the original
CLI pipeline; new SceneFlow runs should use mock, api, or local.

The [FALLBACK] prefix on prompts is stripped before sending to real APIs
so it does not pollute image-generation results.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from threading import Event
from typing import Callable, Dict, List, Optional, Any

from dotenv import load_dotenv

from .image_backends import (
    BackendConfigurationError,
    ImageBackend,
    ImageOptions,
    MockBackend,
    create_backend,
    validate_options,
)
from .image_prompts import (
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_POSITIVE_PROMPT,
    compose_prompt,
)

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

# A legacy/test backend callable: (scene_dict, out_dir) -> Path
BackendFn = Callable[[Dict[str, Any], Path], Path]


@dataclass
class SceneResult:
    """Per-scene outcome written into the manifest."""
    index: int
    title: str
    filename: str
    image_path: str       # absolute or relative path, empty on failure
    prompt_used: str      # exact prompt sent (after stripping [FALLBACK])
    status: str           # "ok" | "skipped" | "error" | "dry_run"
    error: str = ""
    backend: str = ""
    is_placeholder: bool = False
    model: str = ""
    negative_prompt_used: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data.update({
            "scene_id": str(self.index),
            "prompt": self.prompt_used,
            "path": self.image_path,
        })
        return data


# ---------------------------------------------------------------------------
# Prompt cleaning
# ---------------------------------------------------------------------------

_FALLBACK_PREFIX = "[FALLBACK] "


def _clean_prompt(raw: str) -> str:
    """Strip the [FALLBACK] label before sending to a real API."""
    if raw.startswith(_FALLBACK_PREFIX):
        return raw[len(_FALLBACK_PREFIX):]
    return raw


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class ScenePlanError(ValueError):
    pass


def validate_scene_plan(scenes: Any) -> List[str]:
    """
    Validate a list of scene dicts loaded from scene_plan.json.
    Returns a list of error strings (empty means valid).
    """
    errors: List[str] = []

    if not isinstance(scenes, list):
        return ["Scene plan must be a JSON array."]
    if not scenes:
        errors.append("Scene plan is empty.")
        return errors

    seen_indexes: set = set()
    seen_filenames: set = set()
    prev_end: Optional[float] = None

    for i, s in enumerate(scenes):
        if not isinstance(s, dict):
            errors.append(f"Scene[{i}] must be an object.")
            continue
        loc = f"Scene[{i}] (index={s.get('index', '?')})"

        # Required fields
        for f_name in ("index", "title", "text", "start_time", "end_time",
                       "image_prompt", "filename_suggestion"):
            if f_name not in s:
                errors.append(f"{loc}: missing field '{f_name}'")

        idx = s.get("index")
        if idx is not None:
            if not isinstance(idx, int) or isinstance(idx, bool) or idx < 1:
                errors.append(f"{loc}: index must be a positive integer")
            elif idx in seen_indexes:
                errors.append(f"{loc}: duplicate index {idx}")
            else:
                seen_indexes.add(idx)

        for field_name in ("title", "text", "image_prompt"):
            if field_name in s and not isinstance(s[field_name], str):
                errors.append(f"{loc}: {field_name} must be a string")
        if "negative_prompt" in s and not isinstance(s["negative_prompt"], str):
            errors.append(f"{loc}: negative_prompt must be a string")

        st = s.get("start_time")
        et = s.get("end_time")
        if st is not None and et is not None:
            if (
                not isinstance(st, (int, float))
                or isinstance(st, bool)
                or not isinstance(et, (int, float))
                or isinstance(et, bool)
                or not math.isfinite(st)
                or not math.isfinite(et)
            ):
                errors.append(f"{loc}: start_time/end_time must be numbers")
            elif et <= st:
                errors.append(f"{loc}: end_time ({et}) must be > start_time ({st})")
            elif prev_end is not None and st < prev_end - 0.01:
                errors.append(
                    f"{loc}: start_time ({st}) overlaps previous scene end ({prev_end})"
                )
            prev_end = et

        fname = s.get("filename_suggestion", "")
        if not isinstance(fname, str):
            errors.append(f"{loc}: filename_suggestion must be a string")
            fname = ""
        if fname and not fname.endswith(".png"):
            errors.append(f"{loc}: filename_suggestion '{fname}' must end with .png")
        if fname and (Path(fname).name != fname or fname in {".", ".."}):
            errors.append(f"{loc}: filename_suggestion must be a filename, not a path")
        if fname and fname in seen_filenames:
            errors.append(f"{loc}: duplicate filename_suggestion '{fname}'")
        if fname:
            seen_filenames.add(fname)

    return errors


# ---------------------------------------------------------------------------
# Backends — thin wrappers that accept dicts instead of Scene objects
# ---------------------------------------------------------------------------

def _backend_placeholder(scene: Dict[str, Any], out_dir: Path) -> Path:
    """Generate a labeled placeholder PNG using Pillow. No API needed."""
    from PIL import Image, ImageDraw  # type: ignore

    prompt = _clean_prompt(scene.get("image_prompt", "") or scene.get("text", ""))
    title = f"Scene {scene['index']}: {scene['title']}"
    st, et = scene.get("start_time", 0), scene.get("end_time", 0)

    img = Image.new("RGB", (1024, 576), color=(30, 30, 50))
    draw = ImageDraw.Draw(img)

    # Mark clearly as placeholder
    draw.text((30, 12), "[PLACEHOLDER — not AI-generated]", fill=(255, 180, 50))
    draw.text((30, 32), title, fill=(200, 200, 255))

    # Word-wrap prompt text
    words = prompt.split()
    lines, line = [], []
    for word in words:
        line.append(word)
        if len(" ".join(line)) > 80:
            lines.append(" ".join(line[:-1]))
            line = [word]
    lines.append(" ".join(line))
    y = 76
    for ln in lines[:11]:
        draw.text((30, y), ln, fill=(180, 180, 180))
        y += 24

    draw.text((30, 520), f"⏱  {st:.1f}s – {et:.1f}s", fill=(100, 200, 100))

    out_path = out_dir / scene["filename_suggestion"]
    img.save(str(out_path))
    return out_path


def _backend_stability(scene: Dict[str, Any], out_dir: Path) -> Path:
    """Call Stability AI SDXL. Requires STABILITY_API_KEY."""
    import requests, base64  # type: ignore

    api_key = os.getenv("STABILITY_API_KEY", "").strip()
    if not api_key or api_key == "your_stability_api_key_here":
        raise RuntimeError("STABILITY_API_KEY is not configured in .env")

    prompt = _clean_prompt(scene.get("image_prompt", "") or scene.get("text", ""))
    if not prompt:
        raise ValueError(f"Scene {scene['index']} has no image_prompt")

    url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "text_prompts": [
            {"text": prompt, "weight": 1.0},
            *(
                [{"text": scene["negative_prompt"], "weight": -1.0}]
                if scene.get("negative_prompt")
                else []
            ),
        ],
        "cfg_scale": 7,
        "height": 576,
        "width": 1024,
        "steps": 30,
        "samples": 1,
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=90)
    resp.raise_for_status()
    data = resp.json()
    image_bytes = base64.b64decode(data["artifacts"][0]["base64"])

    out_path = out_dir / scene["filename_suggestion"]
    out_path.write_bytes(image_bytes)
    return out_path


def _backend_replicate(scene: Dict[str, Any], out_dir: Path) -> Path:
    """Call Replicate SDXL. Requires REPLICATE_API_TOKEN."""
    import replicate, requests  # type: ignore

    token = os.getenv("REPLICATE_API_TOKEN", "").strip()
    if not token or token == "your_replicate_token_here":
        raise RuntimeError("REPLICATE_API_TOKEN is not configured in .env")

    prompt = _clean_prompt(scene.get("image_prompt", "") or scene.get("text", ""))
    if not prompt:
        raise ValueError(f"Scene {scene['index']} has no image_prompt")

    os.environ["REPLICATE_API_TOKEN"] = token
    output = replicate.run(
        "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
        input={
            "prompt": prompt,
            "negative_prompt": scene.get("negative_prompt", ""),
            "width": 1024,
            "height": 576,
        },
    )
    image_url = output[0] if isinstance(output, list) else output
    resp = requests.get(image_url, timeout=90)
    resp.raise_for_status()

    out_path = out_dir / scene["filename_suggestion"]
    out_path.write_bytes(resp.content)
    return out_path


BACKENDS: Dict[str, BackendFn] = {
    "placeholder": _backend_placeholder,
    "stability": _backend_stability,
    "replicate": _backend_replicate,
}

# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

MANIFEST_FILENAME = "image_manifest.json"


def load_manifest(path: Path) -> Dict[int, SceneResult]:
    """Load an existing manifest, keyed by scene index. Returns {} if absent."""
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Image manifest must contain a JSON array: {path}")
    allowed = set(SceneResult.__dataclass_fields__)
    out: Dict[int, SceneResult] = {}
    for position, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"Image manifest entry {position} must be an object.")
        normalized = dict(entry)
        if "index" not in normalized and "scene_id" in normalized:
            normalized["index"] = int(normalized["scene_id"])
        if "image_path" not in normalized and "path" in normalized:
            normalized["image_path"] = normalized["path"]
        if "prompt_used" not in normalized and "prompt" in normalized:
            normalized["prompt_used"] = normalized["prompt"]
        try:
            result = SceneResult(**{
                key: value for key, value in normalized.items() if key in allowed
            })
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid image manifest entry {position}: {exc}") from exc
        out[result.index] = result
    return out


def save_manifest(results: List[SceneResult], path: Path) -> None:
    """Write the manifest atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(
        json.dumps([r.to_dict() for r in results], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_path.replace(path)


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    """All settings for one image-generation run."""
    scene_plan_path: Path = Path("scene_plan.json")
    output_dir: Path = Path("outputs/images")
    manifest_path: Path = Path("outputs/image_manifest.json")
    backend: Optional[str] = None
    delay_seconds: float = 2.0
    max_retries: int = 2
    retry_delay_seconds: float = 5.0
    force: bool = False          # ignore skip_existing
    dry_run: bool = False        # validate only, no API calls
    scene_indexes: Optional[List[int]] = None  # None = all scenes
    aspect_ratio: Optional[str] = None
    positive_quality_prompt: str = DEFAULT_POSITIVE_PROMPT
    negative_quality_prompt: str = DEFAULT_NEGATIVE_PROMPT
    concurrency: int = 1
    cancel_event: Optional[Event] = None
    progress_callback: Optional[
        Callable[[str, int, Optional[SceneResult], int, int], None]
    ] = None
    backend_instance: Optional[ImageBackend] = None


def _backend_mock(scene: Dict[str, Any], out_dir: Path) -> Path:
    """Compatibility adapter around the common mock-backend contract."""
    output_path = out_dir / scene["filename_suggestion"]
    result = MockBackend().generate(
        str(scene["index"]),
        _clean_prompt(scene.get("image_prompt", "")),
        output_path,
        ImageOptions(),
    )
    if not result.success or result.output_path is None:
        raise RuntimeError(result.error or "Mock backend did not return an image.")
    return result.output_path


_DEFAULT_MOCK_BACKEND = _backend_mock
BACKENDS["mock"] = _backend_mock


def _validate_generated_image(path: Path, backend_name: str) -> None:
    expected_formats = {
        "api": ("JPEG", ".jpg"),
        "local": ("PNG", ".png"),
        "mock": ("PNG", ".png"),
        "placeholder": ("PNG", ".png"),
    }
    expected = expected_formats.get(backend_name)
    if expected is None:
        return

    expected_format, expected_extension = expected
    if path.suffix.lower() != expected_extension:
        raise ValueError(
            f"Backend {backend_name!r} must save {expected_format} images "
            f"with a {expected_extension} extension."
        )
    try:
        from PIL import Image

        with Image.open(path) as image:
            actual_format = image.format
            image.verify()
    except (OSError, ValueError) as exc:
        raise ValueError(f"Backend {backend_name!r} produced an invalid image: {exc}") from exc
    if actual_format != expected_format:
        raise ValueError(
            f"Backend {backend_name!r} must return {expected_format} image data; "
            f"received {actual_format or 'unknown format'}."
        )


def _is_valid_generated_image(path: Path, backend_name: str) -> bool:
    try:
        _validate_generated_image(path, backend_name)
    except (OSError, ValueError):
        return False
    return True


def _generate_one(
    scene: Dict[str, Any],
    filename: str,
    prompt: str,
    negative_prompt: str,
    backend_name: str,
    backend_fn: BackendFn,
    backend_instance: Any,
    output_dir: Path,
    options: ImageOptions,
    max_retries: int,
    retry_delay_seconds: float,
) -> SceneResult:
    index = scene["index"]
    output_path = output_dir / filename
    last_error = ""
    for attempt in range(max_retries + 1):
        try:
            if not prompt.strip():
                raise ValueError("image_prompt is empty")
            if backend_instance is not None:
                generated = backend_instance.generate(
                    str(index), prompt, output_path,
                    ImageOptions(options.aspect_ratio, negative_prompt),
                )
                if not generated.success or generated.output_path is None:
                    raise RuntimeError(generated.error or "Backend returned no output image.")
                saved_path = generated.output_path
                model = generated.model_identifier
            elif backend_name == "mock" and backend_fn is _DEFAULT_MOCK_BACKEND:
                generated = MockBackend().generate(
                    str(index), prompt, output_path,
                    ImageOptions(options.aspect_ratio, negative_prompt),
                )
                if not generated.success or generated.output_path is None:
                    raise RuntimeError(generated.error or "Mock backend returned no output image.")
                saved_path = generated.output_path
                model = generated.model_identifier
            else:
                backend_scene = dict(scene)
                backend_scene["image_prompt"] = prompt
                backend_scene["negative_prompt"] = negative_prompt
                saved_path = backend_fn(backend_scene, output_dir)
                model = ""
            saved_path = Path(saved_path)
            if not saved_path.is_file():
                raise RuntimeError(f"Backend reported success but image is missing: {saved_path}")
            if saved_path.name != filename:
                raise ValueError(
                    f"Backend {backend_name!r} returned {saved_path.name!r}; "
                    f"expected stable scene filename {filename!r}."
                )
            _validate_generated_image(saved_path, backend_name)
            return SceneResult(
                index=index, title=scene["title"], filename=filename,
                image_path=str(saved_path), prompt_used=prompt, status="ok",
                backend=backend_name, is_placeholder=backend_name in {"mock", "placeholder"},
                model=model,
                negative_prompt_used=negative_prompt,
            )
        except Exception as exc:
            last_error = str(exc)
            if attempt < max_retries:
                print(
                    f"  Retry scene {index:03d}, attempt {attempt + 1}: "
                    f"{last_error[:80]}"
                )
                time.sleep(retry_delay_seconds)
    return SceneResult(
        index=index, title=scene["title"], filename=filename,
        image_path="", prompt_used=prompt, status="error", error=last_error,
        backend=backend_name, is_placeholder=backend_name in {"mock", "placeholder"},
        model=getattr(backend_instance, "model_identifier", ""),
        negative_prompt_used=negative_prompt,
    )


def run_image_generation(cfg: RunConfig) -> List[SceneResult]:
    """
    Main entry point for image generation.

    1. Load and validate scene_plan.json.
    2. Load existing manifest (for resume).
    3. For each scene: skip / dry-run / generate with retry.
    4. Write manifest after every scene.
    5. Return list of SceneResult objects.
    """
    # ── Load scene plan ──────────────────────────────────────────────────
    if not cfg.scene_plan_path.exists():
        raise FileNotFoundError(f"Scene plan not found: {cfg.scene_plan_path}")

    all_scenes: Any = json.loads(
        cfg.scene_plan_path.read_text(encoding="utf-8")
    )

    # ── Validate ─────────────────────────────────────────────────────────
    errors = validate_scene_plan(all_scenes)
    if errors:
        raise ScenePlanError(
            "Scene plan validation failed:\n" + "\n".join(f"  • {e}" for e in errors)
        )

    # ── Filter scenes ─────────────────────────────────────────────────────
    if cfg.scene_indexes:
        idx_set = set(cfg.scene_indexes)
        scenes = [s for s in all_scenes if s["index"] in idx_set]
        missing_indexes = idx_set - {s["index"] for s in scenes}
        if missing_indexes:
            raise ValueError(f"Scene indexes not found in plan: {sorted(missing_indexes)}")
    else:
        scenes = all_scenes

    # ── Select backend ────────────────────────────────────────────────────
    backend_name = (cfg.backend or os.getenv("IMAGE_BACKEND", "mock")).strip().lower()
    if backend_name not in BACKENDS and backend_name not in {"api", "local"}:
        raise ValueError(
            f"Unknown backend '{backend_name}'. Available: "
            f"{sorted(set(BACKENDS) | {'api', 'local'})}"
        )
    cfg.backend = backend_name
    if cfg.max_retries < 0 or cfg.max_retries > 10:
        raise ValueError("max_retries must be between 0 and 10.")
    if cfg.concurrency < 1 or cfg.concurrency > 8:
        raise ValueError("concurrency must be between 1 and 8.")
    if cfg.delay_seconds < 0 or cfg.retry_delay_seconds < 0:
        raise ValueError("delay values must not be negative.")
    aspect_ratio = cfg.aspect_ratio
    if backend_name != "local" and aspect_ratio is None:
        aspect_ratio = "16:9"
    options = ImageOptions(aspect_ratio=aspect_ratio)
    validate_options(options)
    if backend_name in {"stability", "replicate"} and aspect_ratio != "16:9":
        raise BackendConfigurationError(
            f"Backend {backend_name!r} only supports the existing 16:9 image dimensions."
        )

    backend_fn = BACKENDS.get(backend_name)
    output_extension = ".jpg" if backend_name == "api" else ".png"
    backend_instance = cfg.backend_instance

    # ── Setup output dir ──────────────────────────────────────────────────
    if not cfg.dry_run:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load existing manifest (resume support) ───────────────────────────
    existing = load_manifest(cfg.manifest_path)

    # Build working results list (seed from existing manifest)
    results: List[SceneResult] = []

    # ── Dry-run header ────────────────────────────────────────────────────
    if cfg.dry_run:
        print(f"DRY RUN - backend={backend_name}  scenes={len(scenes)}")
        print(f"Output dir : {cfg.output_dir}")
        print(f"Manifest   : {cfg.manifest_path}")
        print()
        print(f"{'Sc':>3}  {'Filename':<20}  {'Prompt (first 70 chars)'}")
        print("-" * 100)

    # ── Process each scene ────────────────────────────────────────────────
    pending: List[tuple[Dict[str, Any], str, str, str]] = []
    conflicts: List[SceneResult] = []
    completed_count = 0

    def notify_complete(result: SceneResult) -> None:
        nonlocal completed_count
        completed_count += 1
        if cfg.progress_callback:
            cfg.progress_callback(
                "complete", result.index, result, completed_count, len(scenes)
            )

    for i, scene in enumerate(scenes):
        idx = scene["index"]
        suggested_name = scene.get("filename_suggestion") or f"scene_{idx:03d}.png"
        fname = Path(suggested_name).with_suffix(output_extension).name
        raw_prompt = scene.get("image_prompt", "").strip()
        scene_prompt = _clean_prompt(raw_prompt)
        clean = (
            compose_prompt(cfg.positive_quality_prompt, scene_prompt)
            if scene_prompt
            else ""
        )
        negative_prompt = compose_prompt(
            cfg.negative_quality_prompt,
            _clean_prompt(scene.get("negative_prompt", "")),
        )
        out_path = cfg.output_dir / fname

        # ── Dry-run ───────────────────────────────────────────────────────
        if cfg.dry_run:
            was_fallback = raw_prompt.startswith(_FALLBACK_PREFIX)
            flag = " [FALLBACK prompt]" if was_fallback else ""
            exists_flag = " [FILE EXISTS]" if out_path.exists() else ""
            print(f"{idx:>3}  {fname:<20}  {clean[:70]}{flag}{exists_flag}")
            results.append(SceneResult(
                index=idx, title=scene["title"], filename=fname,
                image_path="", prompt_used=clean,
                status="dry_run", backend=backend_name,
                is_placeholder=backend_name in {"mock", "placeholder"},
                model=getattr(backend_instance, "model_identifier", ""),
                negative_prompt_used=negative_prompt,
            ))
            notify_complete(results[-1])
            continue

        # ── Skip if already done (resume) ────────────────────────────────
        prev = existing.get(idx)
        if (not cfg.force and prev and prev.status == "ok"
                and prev.backend == backend_name
                and prev.filename == fname
                and prev.is_placeholder == (backend_name in {"mock", "placeholder"})
                and prev.image_path
                and Path(prev.image_path).resolve() == out_path.resolve()
                and Path(prev.image_path).is_file()
                and _is_valid_generated_image(Path(prev.image_path), backend_name)):
            print(f"  Scene {idx:03d} - skipped (exists in manifest)")
            results.append(SceneResult(
                index=idx, title=scene["title"], filename=fname,
                image_path=prev.image_path, prompt_used=prev.prompt_used,
                status="skipped", backend=prev.backend,
                is_placeholder=prev.is_placeholder,
                model=prev.model,
                negative_prompt_used=prev.negative_prompt_used,
            ))
            notify_complete(results[-1])
            continue

        if (not cfg.force and out_path.exists() and idx not in existing
                and backend_name not in {"api", "local"}):
            # File exists but manifest entry absent — treat as done
            print(f"  Scene {idx:03d} - skipped (file exists)")
            results.append(SceneResult(
                index=idx, title=scene["title"], filename=fname,
                image_path=str(out_path), prompt_used=clean,
                status="skipped", backend="unknown",
                is_placeholder=False, model="unknown",
                negative_prompt_used=negative_prompt,
            ))
            save_manifest(results + [existing[k] for k in existing
                                     if k not in {r.index for r in results}],
                          cfg.manifest_path)
            notify_complete(results[-1])
            continue
        if not cfg.force and out_path.exists() and backend_name in {"api", "local"}:
            previous_backend = existing[idx].backend if idx in existing else "unknown"
            conflicts.append(SceneResult(
                index=idx,
                title=scene["title"],
                filename=fname,
                image_path=str(out_path),
                prompt_used=clean,
                status="error",
                error=(
                    f"Output already exists at {out_path} (manifest backend: "
                    f"{previous_backend}). Choose another output folder or explicitly "
                    "enable replacement."
                ),
                backend=backend_name,
                model="",
                negative_prompt_used=negative_prompt,
            ))
            continue
        pending.append((scene, fname, clean, negative_prompt))

    if pending or conflicts:
        if backend_instance is None and backend_name in {"api", "local"}:
            if pending:
                backend_instance = create_backend(backend_name)

        def persist_result(result: SceneResult) -> None:
            results.append(result)
            print(
                f"  [{len(results)}/{len(scenes)}] Scene {result.index:03d} - "
                f"{'saved: ' + Path(result.image_path).name if result.status == 'ok' else 'ERROR: ' + result.error}"
            )
            all_results = results + [
                existing[key] for key in existing
                if key not in {item.index for item in results}
            ]
            save_manifest(all_results, cfg.manifest_path)
            notify_complete(result)

        if pending:
            with ThreadPoolExecutor(max_workers=cfg.concurrency) as executor:
                futures = {}
                next_job = 0

                def submit_next() -> bool:
                    nonlocal next_job
                    if next_job >= len(pending):
                        return False
                    if (
                        next_job > 0
                        and backend_name not in {"mock", "placeholder"}
                        and cfg.delay_seconds
                    ):
                        if cfg.cancel_event:
                            if cfg.cancel_event.wait(cfg.delay_seconds):
                                return False
                        else:
                            time.sleep(cfg.delay_seconds)
                    scene, filename, prompt, negative_prompt = pending[next_job]
                    next_job += 1
                    future = executor.submit(
                        _generate_one,
                        scene, filename, prompt, negative_prompt, backend_name, backend_fn,
                        backend_instance, cfg.output_dir, options, cfg.max_retries,
                        cfg.retry_delay_seconds,
                    )
                    futures[future] = scene["index"]
                    if cfg.progress_callback:
                        cfg.progress_callback(
                            "running", scene["index"], None, completed_count, len(scenes)
                        )
                    return True

                for _ in range(min(cfg.concurrency, len(pending))):
                    if not submit_next():
                        break

                while futures:
                    done, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        futures.pop(future)
                        persist_result(future.result())
                    while (
                        not (cfg.cancel_event and cfg.cancel_event.is_set())
                        and len(futures) < cfg.concurrency
                        and submit_next()
                    ):
                        continue

                if cfg.cancel_event and cfg.cancel_event.is_set():
                    for scene, filename, prompt, negative_prompt in pending[next_job:]:
                        persist_result(SceneResult(
                            index=scene["index"],
                            title=scene["title"],
                            filename=filename,
                            image_path="",
                            prompt_used=prompt,
                            status="cancelled",
                            error="Cancelled before this scene started.",
                            backend=backend_name,
                            is_placeholder=backend_name in {"mock", "placeholder"},
                            model=getattr(backend_instance, "model_identifier", ""),
                            negative_prompt_used=negative_prompt,
                        ))

        for conflict in conflicts:
            persist_result(conflict)

    results.sort(key=lambda result: result.index)

    # ── Summary ───────────────────────────────────────────────────────────
    if not cfg.dry_run:
        ok = sum(1 for r in results if r.status == "ok")
        skipped = sum(1 for r in results if r.status in ("skipped",))
        errors = sum(1 for r in results if r.status == "error")
        print()
        print(f"Done. {ok} generated, {skipped} skipped, {errors} failed.")
        print(f"Manifest: {cfg.manifest_path}")

    return results

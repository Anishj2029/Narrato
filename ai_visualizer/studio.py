"""Gradio-based Narrato Studio for scene-plan image generation."""

from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import quote, urlparse

from .image_backends import (
    BackendConfigurationError,
    ComfyUIBackend,
    GeminiApiBackend,
    DEFAULT_GEMINI_MODEL,
    SUPPORTED_ASPECT_RATIOS,
    _redact_google_message,
    create_backend,
)
from .image_runner import (
    RunConfig,
    ScenePlanError,
    SceneResult,
    load_manifest,
    run_image_generation,
    validate_scene_plan,
)
from .image_prompts import (
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_POSITIVE_PROMPT,
    DEFAULT_THEME,
    get_theme_names,
    get_theme_prompts,
)

_SETTINGS_PATH = Path(os.getenv("APPDATA", str(Path.home()))) / "Narrato" / "studio_settings.json"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_JOB_LOCK = threading.RLock()
_JOB: Dict[str, Any] = {
    "status": "idle",
    "backend": "",
    "current_scene": None,
    "total": 0,
    "completed": 0,
    "scene_status": {},
    "results": [],
    "error": "",
    "cancel_event": None,
    "thread": None,
    "started_at": None,
}


@dataclass(frozen=True)
class StudioSettings:
    plan_path: str = str(_PROJECT_ROOT / "scene_plan.json")
    output_dir: str = str(_PROJECT_ROOT / "outputs" / "images_v2")
    manifest_path: str = str(_PROJECT_ROOT / "outputs" / "image_manifest.json")
    backend: str = "local"
    comfyui_url: str = ""
    workflow_path: str = ""
    aspect_ratio: str = "Workflow dimensions"
    selection_mode: str = "All scenes"
    selected_scenes: tuple[str, ...] = ()
    concurrency: int = 1
    delay_seconds: float = 2.0
    retries: int = 2
    positive_quality_prompt: str = DEFAULT_POSITIVE_PROMPT
    negative_quality_prompt: str = DEFAULT_NEGATIVE_PROMPT
    theme: str = DEFAULT_THEME


def load_settings(path: Path = _SETTINGS_PATH) -> StudioSettings:
    """Load non-secret UI preferences; credentials are never persisted here."""
    if not path.is_file():
        return StudioSettings(
            comfyui_url=(
                os.getenv("COMFYUI_URL")
                or os.getenv("COMFYUI_BASE_URL")
                or "http://127.0.0.1:8188"
            ),
            workflow_path=os.getenv("COMFYUI_WORKFLOW", ""),
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Studio settings must be a JSON object: {path}")
    safe = {
        key: value
        for key, value in raw.items()
        if key in StudioSettings.__dataclass_fields__
    }
    if "selected_scenes" in safe:
        safe["selected_scenes"] = tuple(safe["selected_scenes"])
    try:
        return StudioSettings(**safe)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid Narrato Studio settings: {exc}") from exc


def save_settings(settings: StudioSettings, path: Path = _SETTINGS_PATH) -> None:
    """Persist UI preferences while explicitly excluding secret-like fields."""
    values = asdict(settings)
    values["selected_scenes"] = list(settings.selected_scenes)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(values, indent=2), encoding="utf-8")
    temporary.replace(path)


def _project_path(path: str | Path) -> Path:
    """Resolve relative Studio paths from the repository root."""
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = _PROJECT_ROOT / resolved
    return resolved.resolve()


def validate_studio_settings(
    backend: str,
    plan_path: str,
    output_dir: str,
    manifest_path: str,
    aspect_ratio: str,
    concurrency: int,
    delay_seconds: float,
    retries: int,
    comfyui_url: str = "",
    workflow_path: str = "",
) -> None:
    if backend not in {"api", "local"}:
        raise ValueError("Choose either the local or API backend.")
    if not plan_path.strip() or not output_dir.strip() or not manifest_path.strip():
        raise ValueError("Plan, output, and manifest paths are required.")
    if aspect_ratio != "Workflow dimensions" and aspect_ratio not in SUPPORTED_ASPECT_RATIOS:
        raise ValueError(f"Unsupported aspect ratio: {aspect_ratio}")
    try:
        concurrency_value = float(concurrency)
        retry_value = float(retries)
        delay_value = float(delay_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError("Concurrency, delay, and retries must be numeric values.") from exc
    if not math.isfinite(concurrency_value) or not concurrency_value.is_integer():
        raise ValueError("Concurrency must be a whole number between 1 and 8.")
    if not 1 <= concurrency_value <= 8:
        raise ValueError("Concurrency must be between 1 and 8.")
    if not math.isfinite(delay_value) or not 0 <= delay_value <= 3600:
        raise ValueError("Delay must be between 0 and 3600 seconds.")
    if not math.isfinite(retry_value) or not retry_value.is_integer():
        raise ValueError("Retries must be a whole number between 0 and 10.")
    if not 0 <= retry_value <= 10:
        raise ValueError("Retries must be between 0 and 10.")
    if backend == "local":
        if not comfyui_url.strip():
            raise ValueError("Enter the ComfyUI server URL.")
        parsed_url = urlparse(comfyui_url.strip())
        if (
            parsed_url.username
            or parsed_url.password
            or parsed_url.query
            or parsed_url.fragment
        ):
            raise ValueError(
                "ComfyUI URL must not contain credentials, query parameters, or fragments."
            )
        if not workflow_path.strip():
            raise ValueError("Enter the ComfyUI API-format workflow path.")
        ComfyUIBackend(
            base_url=comfyui_url.strip(),
            workflow_path=workflow_path.strip(),
        )


def _read_plan(plan_path: str) -> List[Dict[str, Any]]:
    path = _project_path(plan_path)
    if not path.is_file():
        raise FileNotFoundError(f"Scene plan not found: {path}")
    try:
        scenes = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Scene plan is not valid JSON: {exc}") from exc
    errors = validate_scene_plan(scenes)
    if errors:
        raise ScenePlanError("Scene plan validation failed: " + "; ".join(errors))
    return scenes


def _plan_status(plan_path: str, output_dir: str, manifest_path: str):
    try:
        scenes = _read_plan(plan_path)
    except Exception as exc:
        return (
            f"**Plan unavailable:** {exc}",
            [],
            [],
            _counts_markdown(0, 0, 0, 0),
            [],
            [],
        )

    try:
        manifest = load_manifest(Path(manifest_path).expanduser())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return (
            f"**Manifest unavailable:** {exc}",
            [],
            [],
            _counts_markdown(0, len(scenes), 0, 0),
            [str(scene["index"]) for scene in scenes],
            [],
        )
    current_status = _snapshot_job()
    status_by_scene = dict(current_status["scene_status"])
    rows = []
    gallery = []
    failed_ids = []
    generated = skipped = failed = pending = 0
    for scene in scenes:
        index = scene["index"]
        entry = manifest.get(index)
        job_status = status_by_scene.get(index)
        if job_status in {"queued", "running", "cancelled"}:
            status = job_status
            backend = current_status["backend"] or (entry.backend if entry else "")
            image_path = entry.image_path if entry else ""
            error = _safe_diagnostic(entry.error) if entry else ""
            placeholder = entry.is_placeholder if entry else False
        elif entry and entry.status in {"ok", "skipped"}:
            status = (
                "placeholder"
                if entry.is_placeholder
                else "skipped" if entry.status == "skipped" else "completed"
            )
            backend = entry.backend
            image_path = entry.image_path
            error = ""
            placeholder = entry.is_placeholder
        elif entry and entry.status == "error":
            status = "failed"
            backend = entry.backend
            image_path = entry.image_path
            error = _safe_diagnostic(entry.error)
            placeholder = entry.is_placeholder
        else:
            status = "pending"
            backend = ""
            image_path = ""
            error = ""
            placeholder = False

        valid_image = _valid_preview(image_path) if image_path else False
        if status in {"completed", "skipped", "placeholder"} and not valid_image:
            status = "failed"
            error = "The manifest image is missing or is not a valid image file."

        if status == "completed":
            generated += 1
        elif status == "skipped":
            skipped += 1
        elif status == "failed":
            failed += 1
            failed_ids.append(str(index))
        elif status == "pending":
            pending += 1
        elif status in {"queued", "running", "cancelled", "placeholder"}:
            pending += 1

        if valid_image and status in {"completed", "skipped"} and not placeholder:
            gallery.append((
                image_path,
                f"Scene {index:03d} · {scene['title']} · {backend}",
            ))
        elif valid_image and placeholder:
            gallery.append((
                image_path,
                f"Scene {index:03d} · {scene['title']} · PLACEHOLDER ({backend})",
            ))
        rows.append([
            index,
            scene["title"],
            f"{scene['start_time']:.1f}–{scene['end_time']:.1f}s",
            status,
            backend,
            image_path,
            error,
        ])

    runtime_error = current_status["error"]
    note = (
        f"**Plan ready** · {len(scenes)} scenes · `{_project_path(plan_path)}`"
        if not runtime_error
        else f"**Last job error:** {runtime_error}"
    )
    return (
        note,
        rows,
        gallery,
        _counts_markdown(generated, pending, skipped, failed),
        [str(scene["index"]) for scene in scenes],
        failed_ids,
    )


def _valid_preview(image_path: str) -> bool:
    try:
        from PIL import Image

        with Image.open(image_path) as image:
            image.verify()
        return True
    except (OSError, ValueError):
        return False


def _counts_markdown(generated: int, pending: int, skipped: int, failed: int) -> str:
    return (
        f"**Generated** {generated}　·　**Pending** {pending}　·　"
        f"**Skipped** {skipped}　·　**Failed** {failed}"
    )


def _safe_diagnostic(message: str) -> str:
    return _redact_google_message(message, os.getenv("GEMINI_API_KEY", ""))


def _snapshot_job() -> Dict[str, Any]:
    with _JOB_LOCK:
        return {
            "status": _JOB["status"],
            "backend": _JOB["backend"],
            "current_scene": _JOB["current_scene"],
            "total": _JOB["total"],
            "completed": _JOB["completed"],
            "scene_status": dict(_JOB["scene_status"]),
            "results": list(_JOB["results"]),
            "error": _JOB["error"],
            "started_at": _JOB["started_at"],
        }


def _progress_callback(
    event: str,
    index: int,
    result: Optional[SceneResult],
    completed: int,
    total: int,
) -> None:
    with _JOB_LOCK:
        if event == "running":
            _JOB["current_scene"] = index
            _JOB["scene_status"][index] = "running"
        elif result is not None:
            _JOB["scene_status"][index] = (
                "failed" if result.status == "error" else result.status
            )
            _JOB["results"] = [
                item for item in _JOB["results"] if item.index != index
            ] + [result]
        _JOB["completed"] = completed
        _JOB["total"] = total


def start_generation(
    backend: str,
    plan_path: str,
    output_dir: str,
    manifest_path: str,
    aspect_ratio: str,
    selection_mode: str,
    selected_scenes: Optional[Sequence[str]],
    concurrency: int,
    delay_seconds: float,
    retries: int,
    confirm_api: bool,
    replace_existing: bool,
    comfyui_url: str = "",
    workflow_path: str = "",
    positive_quality_prompt: str = DEFAULT_POSITIVE_PROMPT,
    negative_quality_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    theme: str = DEFAULT_THEME,
) -> str:
    """Validate and start one non-blocking image-generation job."""
    plan_path = str(_project_path(plan_path))
    output_dir = str(_project_path(output_dir))
    manifest_path = str(_project_path(manifest_path))
    if not isinstance(positive_quality_prompt, str) or not isinstance(
        negative_quality_prompt, str
    ):
        raise ValueError("Quality prompts must be text.")
    get_theme_prompts(theme)
    validate_studio_settings(
        backend,
        plan_path,
        output_dir,
        manifest_path,
        aspect_ratio,
        concurrency,
        delay_seconds,
        retries,
        comfyui_url,
        workflow_path,
    )
    scenes = _read_plan(plan_path)
    if backend == "api" and not confirm_api:
        raise ValueError(
            "Confirm that API generation may consume quota or incur charges before starting."
        )

    existing_manifest = load_manifest(Path(manifest_path).expanduser())
    if selection_mode not in {"All scenes", "Selected scenes", "Failed scenes only"}:
        raise ValueError("Choose all scenes, selected scenes, or failed scenes only.")
    if selection_mode == "Failed scenes only":
        indexes = sorted(
            index for index, result in existing_manifest.items()
            if result.status == "error"
        )
        if not indexes:
            raise ValueError("There are no failed scenes in the selected manifest.")
    elif selection_mode == "Selected scenes":
        try:
            indexes = sorted({int(item) for item in (selected_scenes or [])})
        except (TypeError, ValueError) as exc:
            raise ValueError("Select at least one valid scene.") from exc
        if not indexes:
            raise ValueError("Select at least one scene.")
    else:
        indexes = None

    if indexes is not None:
        plan_indexes = {scene["index"] for scene in scenes}
        missing = set(indexes) - plan_indexes
        if missing:
            raise ValueError(f"Selected scenes are not in this plan: {sorted(missing)}")

    backend_instance = None
    if backend == "local":
        backend_instance = ComfyUIBackend(
            base_url=comfyui_url.strip(),
            workflow_path=workflow_path.strip(),
        )
    else:
        backend_instance = create_backend("api")

    settings = StudioSettings(
        plan_path=plan_path,
        output_dir=output_dir,
        manifest_path=manifest_path,
        backend=backend,
        comfyui_url=comfyui_url,
        workflow_path=workflow_path,
        aspect_ratio=aspect_ratio,
        selection_mode=selection_mode,
        selected_scenes=tuple(str(index) for index in (selected_scenes or [])),
        concurrency=int(concurrency),
        delay_seconds=float(delay_seconds),
        retries=int(retries),
        positive_quality_prompt=positive_quality_prompt,
        negative_quality_prompt=negative_quality_prompt,
        theme=theme,
    )
    save_settings(settings)

    cancel_event = threading.Event()
    with _JOB_LOCK:
        running_thread = _JOB.get("thread")
        if (
            _JOB.get("status") in {"starting", "running", "cancelling"}
            or (running_thread is not None and running_thread.is_alive())
        ):
            raise RuntimeError("A generation job is already running.")
        selected = indexes or [scene["index"] for scene in scenes]
        _JOB.update({
            "status": "starting",
            "backend": backend,
            "output_dir": output_dir,
            "current_scene": None,
            "total": len(selected),
            "completed": 0,
            "scene_status": {index: "queued" for index in selected},
            "results": [],
            "error": "",
            "cancel_event": cancel_event,
            "thread": None,
            "started_at": time.time(),
        })

    def worker() -> None:
        try:
            config = RunConfig(
                scene_plan_path=Path(plan_path).expanduser(),
                output_dir=Path(output_dir).expanduser(),
                manifest_path=Path(manifest_path).expanduser(),
                backend=backend,
                delay_seconds=float(delay_seconds),
                max_retries=int(retries),
                force=bool(replace_existing),
                scene_indexes=indexes,
                aspect_ratio=(
                    None if aspect_ratio == "Workflow dimensions" else aspect_ratio
                ),
                positive_quality_prompt=positive_quality_prompt,
                negative_quality_prompt=negative_quality_prompt,
                theme=theme,
                concurrency=int(concurrency),
                cancel_event=cancel_event,
                progress_callback=_progress_callback,
                backend_instance=backend_instance,
            )
            run_image_generation(config)
        except Exception as exc:
            with _JOB_LOCK:
                _JOB["error"] = _safe_diagnostic(str(exc))
        finally:
            with _JOB_LOCK:
                _JOB["status"] = (
                    "cancelled"
                    if cancel_event.is_set()
                    else "failed" if _JOB["error"] else "completed"
                )
                _JOB["current_scene"] = None

    thread = threading.Thread(target=worker, name="narrato-image-generation", daemon=True)
    with _JOB_LOCK:
        _JOB["thread"] = thread
        _JOB["status"] = "running"
        thread.start()
    return f"Started **{backend}** generation for {len(selected)} scene(s)."


def cancel_generation() -> str:
    with _JOB_LOCK:
        event = _JOB.get("cancel_event")
        thread = _JOB.get("thread")
        if event is None or thread is None or not thread.is_alive():
            return "No generation job is currently running."
        event.set()
        _JOB["status"] = "cancelling"
    return "Cancellation requested. In-progress backend calls will finish; queued scenes will not start."


def check_backend_readiness(
    backend: str, comfyui_url: str = "", workflow_path: str = ""
) -> str:
    """Check local ComfyUI connectivity or API configuration without an API call."""
    if backend == "api":
        try:
            api_backend = GeminiApiBackend()
        except BackendConfigurationError as exc:
            return f"**Not ready:** {exc}"
        return (
            f"**Configured** · Gemini model `{api_backend.model_identifier}`. "
            "No API request was made; quota and billing still need to be available."
        )
    if backend != "local":
        return "**Not ready:** Select local or API."
    try:
        parsed_url = urlparse(comfyui_url.strip())
        if (
            parsed_url.username
            or parsed_url.password
            or parsed_url.query
            or parsed_url.fragment
        ):
            raise ValueError(
                "ComfyUI URL must not contain credentials, query parameters, or fragments."
            )
        local_backend = ComfyUIBackend(
            base_url=comfyui_url.strip(),
            workflow_path=workflow_path.strip(),
        )
        local_backend.preflight()
        return f"**Ready** · ComfyUI model: `{local_backend.model_identifier}`"
    except Exception as exc:
        return f"**Not ready:** {_safe_diagnostic(str(exc))}"


def _refresh_files(manifest_path: str, plan_path: str, output_dir: str):
    manifest = _project_path(manifest_path)
    csv_path = _project_path(plan_path).with_suffix(".csv")
    folder = _project_path(plan_path).parent
    resolved_output_dir = _project_path(output_dir)
    status = _snapshot_job()
    if status.get("results"):
        image_paths = [result.image_path for result in status["results"] if result.image_path]
        if image_paths:
            resolved_output_dir = Path(image_paths[0]).parent
    resolved_output_dir = resolved_output_dir.resolve()
    folder_link = f"file:///{quote(str(resolved_output_dir).replace(os.sep, '/'), safe='/:')}"
    return (
        str(manifest) if manifest.is_file() else None,
        str(csv_path) if csv_path.is_file() else None,
        f"**Output folder:** [{resolved_output_dir}]({folder_link})  \n"
        f"**Scene-plan folder:** `{folder.resolve()}`",
    )


def refresh_dashboard(plan_path: str, output_dir: str, manifest_path: str):
    plan_info, rows, gallery, counts, scene_ids, failed_ids = _plan_status(
        plan_path, output_dir, manifest_path
    )
    snapshot = _snapshot_job()
    if snapshot["total"]:
        current = (
            f"Scene {snapshot['current_scene']:03d}"
            if snapshot["current_scene"] is not None
            else "—"
        )
        progress = (
            f"**Job:** {snapshot['status']} · **Backend:** {snapshot['backend']} · "
            f"**Progress:** {snapshot['completed']}/{snapshot['total']} complete · "
            f"**Current:** {current}"
        )
    else:
        progress = "**Job:** idle"
    manifest_file, csv_file, paths = _refresh_files(
        manifest_path, plan_path, output_dir
    )
    return (
        plan_info,
        counts,
        progress,
        rows,
        gallery,
        gr_update(choices=scene_ids),
        gr_update(choices=failed_ids, value=None),
        manifest_file,
        csv_file,
        paths,
    )


def gr_update(**kwargs):
    """Late-bind Gradio so non-UI functions remain testable without the package."""
    import gradio as gr

    return gr.update(**kwargs)


def build_ui():
    import gradio as gr

    class _NoPyiComponentMeta(type(gr.Gallery)):
        def __new__(metaclass, name, bases, attributes, **kwargs):
            return type.__new__(metaclass, name, bases, attributes)

        def __init__(cls, name, bases, attributes, **kwargs):
            type.__init__(cls, name, bases, attributes)

    class _StudioGallery(gr.Gallery, metaclass=_NoPyiComponentMeta):
        @property
        def skip_api(self):
            return True

        def get_block_name(self) -> str:
            return "gallery"

        def get_block_class(self) -> str:
            return "gallery"

    class _StudioFile(gr.File, metaclass=_NoPyiComponentMeta):
        @property
        def skip_api(self):
            return True

        def get_block_name(self) -> str:
            return "file"

        def get_block_class(self) -> str:
            return "file"

    settings_warning = ""
    try:
        settings = load_settings()
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        settings = StudioSettings(
            comfyui_url=(
                os.getenv("COMFYUI_URL")
                or os.getenv("COMFYUI_BASE_URL")
                or "http://127.0.0.1:8188"
            ),
            workflow_path=os.getenv("COMFYUI_WORKFLOW", ""),
        )
        settings_warning = f"Studio preferences could not be loaded: {exc}"
    css = """
    :root { --narrato-bg: #0b1020; --narrato-panel: #111a2e; }
    .gradio-container { max-width: 1500px !important; background: var(--narrato-bg) !important; }
    .narrato-hero { padding: 26px 30px; border: 1px solid #283650; border-radius: 18px;
      background: linear-gradient(120deg, #141f38, #101525 70%); margin-bottom: 16px; }
    .narrato-hero h1 { margin: 0; font-size: 30px; }
    .narrato-muted { color: #9aa9c2; }
    .narrato-card { border: 1px solid #26334c; border-radius: 14px; background: #111a2e; }
    """
    with gr.Blocks(
        title="Narrato Studio",
        theme=gr.themes.Base(primary_hue="violet", neutral_hue="slate"),
        css=css,
    ) as demo:
        gr.HTML(
            '<div class="narrato-hero"><h1>Narrato Studio</h1>'
            '<div class="narrato-muted">Plan, generate, monitor, and review scene imagery.</div></div>'
        )
        if settings_warning:
            gr.Markdown(f"**Settings warning:** {settings_warning}")
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=330, elem_classes=["narrato-card"]):
                gr.Markdown("### Project")
                plan_path = gr.Textbox(label="Scene plan JSON", value=settings.plan_path)
                output_dir = gr.Textbox(label="Image output folder", value=settings.output_dir)
                manifest_path = gr.Textbox(
                    label="Manifest JSON", value=settings.manifest_path
                )
                refresh_btn = gr.Button("Load / refresh plan")
                plan_info = gr.Markdown("Load a scene plan to view its scenes.")

                gr.Markdown("### Backend")
                backend = gr.Radio(
                    choices=["local", "api"],
                    value=settings.backend if settings.backend in {"local", "api"} else "local",
                    label="Image backend",
                )
                with gr.Group(visible=settings.backend != "api") as local_settings:
                    comfyui_url = gr.Textbox(
                        label="ComfyUI URL",
                        value=settings.comfyui_url or "http://127.0.0.1:8188",
                    )
                    workflow_path = gr.Textbox(
                        label="API-format workflow JSON",
                        value=settings.workflow_path,
                    )
                with gr.Group(visible=settings.backend == "api") as api_settings:
                    model_name = (
                        os.getenv("GEMINI_IMAGE_MODEL") or DEFAULT_GEMINI_MODEL
                    )
                    api_status = (
                        "API key configured (value hidden)."
                        if os.getenv("GEMINI_API_KEY", "").strip()
                        else "API key not configured. Set GEMINI_API_KEY in .env."
                    )
                    gr.Markdown(
                        f"**Model:** `{model_name}`  \n{api_status}  \n"
                        "API quota or billing restrictions can prevent generation."
                    )
                ready_btn = gr.Button("Check backend readiness")
                readiness = gr.Markdown()

                gr.Markdown("### Generation")
                aspect_ratio = gr.Dropdown(
                    choices=["Workflow dimensions", *sorted(SUPPORTED_ASPECT_RATIOS)],
                    value=settings.aspect_ratio,
                    label="Aspect ratio override",
                )
                visual_theme = gr.Dropdown(
                    choices=get_theme_names(),
                    value=settings.theme if settings.theme in get_theme_names() else DEFAULT_THEME,
                    label="Visual theme",
                    info="Select a theme to set permanent positive and negative quality prompts.",
                )
                previous_theme = gr.State(
                    value=(
                        settings.theme
                        if settings.theme in get_theme_names()
                        else DEFAULT_THEME
                    )
                )
                positive_quality_prompt = gr.TextArea(
                    label="Permanent positive quality prompt",
                    value=settings.positive_quality_prompt,
                    lines=4,
                )
                negative_quality_prompt = gr.TextArea(
                    label="Permanent negative quality prompt",
                    value=settings.negative_quality_prompt,
                    lines=4,
                )
                selection_mode = gr.Radio(
                    choices=["All scenes", "Selected scenes", "Failed scenes only"],
                    value=settings.selection_mode,
                    label="Scenes to generate",
                )
                selected_scenes = gr.CheckboxGroup(
                    choices=list(settings.selected_scenes),
                    value=list(settings.selected_scenes),
                    label="Selected scene IDs",
                    visible=settings.selection_mode == "Selected scenes",
                )
                failed_scene = gr.Dropdown(
                    choices=[],
                    label="Retry one failed scene",
                )
                with gr.Row():
                    concurrency = gr.Slider(
                        minimum=1, maximum=8, step=1,
                        value=settings.concurrency, label="Concurrency",
                    )
                    retries = gr.Slider(
                        minimum=0, maximum=10, step=1,
                        value=settings.retries, label="Retries",
                    )
                delay = gr.Number(
                    label="Delay between submissions (seconds)",
                    value=settings.delay_seconds,
                    minimum=0,
                )
                confirm_api = gr.Checkbox(
                    label="I understand API generation may consume quota or incur charges",
                    value=False,
                    visible=settings.backend == "api",
                )
                replace_existing = gr.Checkbox(
                    label="Explicitly replace existing output files",
                    value=False,
                )
                with gr.Row():
                    start_btn = gr.Button("Start generation", variant="primary")
                    cancel_btn = gr.Button("Cancel", variant="stop")
                retry_btn = gr.Button("Retry selected failed scene")
                action_status = gr.Markdown()

            with gr.Column(scale=2, elem_classes=["narrato-card"]):
                gr.Markdown("### Run status")
                counts = gr.Markdown(_counts_markdown(0, 0, 0, 0))
                progress = gr.Markdown("**Job:** idle")
                gr.Markdown("### Scene gallery")
                gallery = _StudioGallery(
                    label="Generated images · click a thumbnail to preview",
                    columns=3,
                    height=520,
                    object_fit="contain",
                    preview=True,
                )
                gr.Markdown("### Scene details")
                scene_table = gr.Dataframe(
                    headers=[
                        "Scene", "Title", "Time", "Status", "Backend", "Image path", "Error",
                    ],
                    datatype=["number", "str", "str", "str", "str", "str", "str"],
                    interactive=False,
                    wrap=True,
                )
                with gr.Row():
                    manifest_file = _StudioFile(
                        label="Download manifest", interactive=False
                    )
                    csv_file = _StudioFile(
                        label="Download scene-plan CSV", interactive=False
                    )
                paths = gr.Markdown()

        def backend_visibility(value):
            is_api = value == "api"
            return (
                gr.update(visible=not is_api),
                gr.update(visible=is_api),
                gr.update(visible=is_api),
            )

        backend.change(
            backend_visibility,
            inputs=[backend],
            outputs=[local_settings, api_settings, confirm_api],
        )
        selection_mode.change(
            lambda value: gr.update(visible=value == "Selected scenes"),
            inputs=[selection_mode],
            outputs=[selected_scenes],
        )

        def apply_theme(theme_name, positive_current, negative_current, old_theme):
            old_positive, old_negative = get_theme_prompts(old_theme)
            positive, negative = get_theme_prompts(theme_name)
            return (
                positive if positive_current == old_positive else positive_current,
                negative if negative_current == old_negative else negative_current,
                theme_name,
            )

        visual_theme.change(
            apply_theme,
            inputs=[
                visual_theme,
                positive_quality_prompt,
                negative_quality_prompt,
                previous_theme,
            ],
            outputs=[
                positive_quality_prompt,
                negative_quality_prompt,
                previous_theme,
            ],
        )
        refresh_btn.click(
            refresh_dashboard,
            inputs=[plan_path, output_dir, manifest_path],
            outputs=[
                plan_info, counts, progress, scene_table, gallery,
                selected_scenes, failed_scene, manifest_file, csv_file, paths,
            ],
        )
        ready_btn.click(
            check_backend_readiness,
            inputs=[backend, comfyui_url, workflow_path],
            outputs=[readiness],
        )

        start_inputs = [
            backend,
            plan_path,
            output_dir,
            manifest_path,
            aspect_ratio,
            selection_mode,
            selected_scenes,
            concurrency,
            delay,
            retries,
            confirm_api,
            replace_existing,
            comfyui_url,
            workflow_path,
            positive_quality_prompt,
            negative_quality_prompt,
            visual_theme,
        ]

        def start_handler(*values):
            try:
                return start_generation(*values)
            except Exception as exc:
                return f"**Could not start:** {exc}"

        start_btn.click(
            start_handler,
            inputs=start_inputs,
            outputs=[action_status],
        ).then(
            refresh_dashboard,
            inputs=[plan_path, output_dir, manifest_path],
            outputs=[
                plan_info, counts, progress, scene_table, gallery,
                selected_scenes, failed_scene, manifest_file, csv_file, paths,
            ],
        )
        cancel_btn.click(
            cancel_generation, outputs=[action_status]
        )

        def retry_one(*values):
            (
                backend_value, plan_value, output_value, manifest_value, ratio,
                mode, chosen, concurrency_value, delay_value, retries_value,
                api_confirmed, replace, url, workflow, positive_prompt,
                negative_prompt, theme_name, failed_index,
            ) = values
            try:
                return start_generation(
                    backend_value, plan_value, output_value, manifest_value, ratio,
                    "Selected scenes", [failed_index] if failed_index else [],
                    concurrency_value, delay_value, retries_value,
                    api_confirmed, replace, url, workflow,
                    positive_prompt, negative_prompt, theme_name,
                )
            except Exception as exc:
                return f"**Could not retry scene:** {exc}"

        retry_btn.click(
            retry_one,
            inputs=start_inputs + [failed_scene],
            outputs=[action_status],
        )

        demo.load(
            refresh_dashboard,
            inputs=[plan_path, output_dir, manifest_path],
            outputs=[
                plan_info, counts, progress, scene_table, gallery,
                selected_scenes, failed_scene, manifest_file, csv_file, paths,
            ],
            every=2,
        )

    return demo


def launch(share: bool = False, port: int = 7860) -> None:
    build_ui().launch(
        share=share,
        server_name="127.0.0.1",
        server_port=port,
    )

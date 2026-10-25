"""Gradio-based Narrato Studio for scene-plan image generation."""

from __future__ import annotations

import filecmp
import hashlib
import json
import logging
import math
import os
import re
import shutil
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence
from urllib.parse import quote, urlparse

from .image_backends import (
    BackendConfigurationError,
    ComfyUIBackend,
    GeminiApiBackend,
    DEFAULT_GEMINI_MODEL,
    ImageOptions,
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
    compose_prompt,
)
from .markers import (
    VisualMarker,
    create_marker,
    delete_marker,
    edit_marker,
    load_markers,
    save_markers,
)
from .scene_parser import Scene, parse_script_text
from .scene_planner import plan_script_scenes
from .timestamp_mapper import assign_timestamps, get_audio_duration

logger = logging.getLogger(__name__)
_SETTINGS_PATH = Path(os.getenv("APPDATA", str(Path.home()))) / "Narrato" / "studio_settings.json"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_JOB_LOCK = threading.RLock()
_AUTOMATIC_PLAN_LOCK = threading.RLock()
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
    "plan_path": None,
    "project_dir": None,
}
_MARKER_JOB_LOCK = threading.RLock()
_MARKER_JOB: Dict[str, Any] = {
    "status": "idle",
    "marker_id": None,
    "project_dir": None,
    "progress": 0,
    "message": "",
    "image_path": None,
    "error": "",
    "thread": None,
}


def _log_studio_callback(
    action: str,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorate(callback: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(callback)
        def logged_callback(*args, **kwargs):
            logger.info("Studio callback started: %s", action)
            try:
                result = callback(*args, **kwargs)
            except Exception as exc:
                _log_exception(f"Studio callback failed: {action}", exc)
                raise
            logger.info("Studio callback completed: %s", action)
            return result

        return logged_callback

    return decorate


def _log_exception(action: str, exc: Exception) -> None:
    message = _safe_diagnostic(str(exc))
    safe_exception = RuntimeError(message)
    logger.error(
        "%s (%s): %s",
        action,
        type(exc).__name__,
        message,
        exc_info=(RuntimeError, safe_exception, exc.__traceback__),
    )


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


def _copy_project_asset(source: str | Path, destination: Path) -> None:
    source_path = Path(source)
    if not source_path.is_file():
        raise FileNotFoundError(f"Project asset not found: {source_path}")
    if destination.exists():
        if not destination.is_file() or not filecmp.cmp(
            source_path, destination, shallow=False
        ):
            raise FileExistsError(
                f"Project asset already exists and differs: {destination}. "
                "Choose another project folder; existing assets are never replaced."
            )
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with source_path.open("rb") as source_file:
            with destination.open("xb") as destination_file:
                created = True
                shutil.copyfileobj(source_file, destination_file)
                destination_file.flush()
                os.fsync(destination_file.fileno())
    except Exception:
        if created:
            destination.unlink(missing_ok=True)
        raise


def save_project_assets(
    project_dir: str | Path,
    audio_file: str | Path,
    script_file: str | Path,
    *,
    project_name: Optional[str] = None,
    created_at: Optional[str] = None,
) -> dict[str, Any]:
    """Create or reopen a project without replacing existing assets or markers."""
    if not str(project_dir).strip():
        raise ValueError("Choose a project folder.")
    if audio_file is None or not str(audio_file).strip():
        raise ValueError("Upload the voice-over audio file.")
    if script_file is None or not str(script_file).strip():
        raise ValueError("Upload the matching script file.")
    project_path = _project_path(project_dir)
    if project_path == _PROJECT_ROOT:
        raise ValueError("Choose a project subfolder, not the repository root.")
    audio_source = Path(audio_file)
    script_source = Path(script_file)
    if not audio_source.is_file():
        raise FileNotFoundError(f"Project asset not found: {audio_source}")
    if not script_source.is_file():
        raise FileNotFoundError(f"Project asset not found: {script_source}")
    audio_suffix = audio_source.suffix.lower()
    script_suffix = script_source.suffix.lower()
    if audio_suffix not in {".mp3", ".wav"}:
        raise ValueError("Voice-over audio must be an MP3 or WAV file.")
    if script_suffix not in {".txt", ".md"}:
        raise ValueError("The matching script must be a TXT or Markdown file.")

    audio_name = f"voiceover{audio_suffix}"
    script_name = f"script{script_suffix}"
    metadata_path = project_path / "project.json"
    expected = {
        "schema_version": 1,
        "audio_file": audio_name,
        "script_file": script_name,
        "markers_file": "markers.json",
    }
    metadata_to_write = dict(expected)
    if project_name is not None:
        cleaned_name = project_name.strip()
        if not cleaned_name:
            raise ValueError("Project name cannot be empty.")
        metadata_to_write["project_name"] = cleaned_name
        metadata_to_write["creation_state"] = "created"
        metadata_to_write["created_at"] = (
            created_at or datetime.now(timezone.utc).isoformat()
        )
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Project metadata is invalid: {exc}") from exc
        if not isinstance(metadata, dict) or any(
            metadata.get(key) != value for key, value in expected.items()
        ):
            raise ValueError(
                f"Project metadata conflicts with the selected assets: {metadata_path}"
            )

    project_path.mkdir(parents=True, exist_ok=True)
    for source_path, asset_name in (
        (audio_source, audio_name),
        (script_source, script_name),
    ):
        destination = project_path / asset_name
        if destination.exists() and (
            not destination.is_file()
            or not filecmp.cmp(source_path, destination, shallow=False)
        ):
            raise FileExistsError(
                f"Project asset already exists and differs: {destination}. "
                "Choose another project folder; existing assets are never replaced."
            )
    _copy_project_asset(audio_source, project_path / audio_name)
    _copy_project_asset(script_source, project_path / script_name)
    if not metadata_path.exists():
        _write_project_metadata(metadata_path, metadata_to_write)

    markers_path = project_path / expected["markers_file"]
    if markers_path.exists():
        markers = load_markers(markers_path)
    else:
        save_markers(markers_path, [])
        markers = []
    return {
        "project_dir": project_path,
        "audio_path": project_path / audio_name,
        "script_path": project_path / script_name,
        "markers_path": markers_path,
        "markers": markers,
        "metadata": (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.is_file()
            else metadata_to_write
        ),
    }


def _write_project_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(metadata, temporary_file, indent=2)
            temporary_file.write("\n")
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def load_project_workspace(project_dir: str | Path) -> dict[str, Any]:
    """Load project assets and existing markers from a saved project folder."""
    if not str(project_dir).strip():
        raise ValueError("Enter the saved project folder.")
    project_path = _project_path(project_dir)
    metadata_path = project_path / "project.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Saved Narrato project not found: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Project metadata is invalid: {exc}") from exc
    expected_fields = {
        "schema_version", "audio_file", "script_file", "markers_file",
    }
    optional_fields = {
        "project_name", "created_at", "creation_state", "selected_mode",
    }
    if (
        not isinstance(metadata, dict)
        or not expected_fields.issubset(metadata)
        or set(metadata) - expected_fields - optional_fields
        or metadata.get("schema_version") != 1
    ):
        raise ValueError(f"Unsupported or invalid Narrato project: {metadata_path}")
    if (
        "project_name" in metadata
        and (
            not isinstance(metadata["project_name"], str)
            or not metadata["project_name"].strip()
        )
    ):
        raise ValueError(f"Invalid project name in metadata: {metadata_path}")
    if "created_at" in metadata and (
        not isinstance(metadata["created_at"], str)
        or not metadata["created_at"].strip()
    ):
        raise ValueError(f"Invalid creation timestamp in metadata: {metadata_path}")
    if metadata.get("creation_state") not in {
        None, "created", "workflow_selected",
    }:
        raise ValueError(f"Invalid project creation state: {metadata_path}")
    if metadata.get("selected_mode") not in {
        None, *_PROJECT_MODES,
    }:
        raise ValueError(f"Invalid selected workflow in metadata: {metadata_path}")

    asset_paths = {}
    for key in ("audio_file", "script_file", "markers_file"):
        relative_path = metadata[key]
        if not isinstance(relative_path, str) or Path(relative_path).is_absolute():
            raise ValueError(f"Invalid project {key} path.")
        asset_path = (project_path / relative_path).resolve()
        if asset_path.parent != project_path.resolve():
            raise ValueError(f"Project {key} path must stay inside the project folder.")
        if not asset_path.is_file():
            raise FileNotFoundError(f"Project asset not found: {asset_path}")
        asset_paths[key] = asset_path

    audio_path = asset_paths["audio_file"]
    script_path = asset_paths["script_file"]
    if audio_path.suffix.lower() not in {".mp3", ".wav"}:
        raise ValueError("Saved voice-over audio must be an MP3 or WAV file.")
    if script_path.suffix.lower() not in {".txt", ".md"}:
        raise ValueError("Saved script must be a TXT or Markdown file.")
    return {
        "project_dir": project_path,
        "project_name": metadata.get("project_name", project_path.name),
        "selected_mode": metadata.get("selected_mode"),
        "metadata": metadata,
        "audio_path": audio_path,
        "script_path": script_path,
        "markers_path": asset_paths["markers_file"],
        "markers": load_markers(asset_paths["markers_file"]),
    }


_PROJECT_MODES = {
    "Assisted Visual Creation": "assisted",
    "Automatic Visual Creation": "automatic",
}


def _uploaded_path(upload: Any, label: str) -> Path:
    if upload is None:
        raise ValueError(f"Choose the {label} file.")
    if isinstance(upload, (str, Path)):
        value = upload
    else:
        value = getattr(upload, "path", None) or getattr(upload, "name", None)
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"Choose the {label} file.")
    return Path(value)


def _safe_project_folder_name(project_name: str) -> str:
    display_name = project_name.strip()
    if not display_name:
        raise ValueError("Enter a project name.")
    if len(display_name) > 100:
        raise ValueError("Project names must be 100 characters or fewer.")
    folder_name = re.sub(r"[^A-Za-z0-9._-]+", "_", display_name).strip(" ._")
    if not folder_name:
        raise ValueError("Project name must contain letters or numbers.")
    if len(folder_name) > 80:
        folder_name = folder_name[:80].rstrip(" ._")
    reserved_names = {
        "CON", "PRN", "AUX", "NUL",
        *(f"COM{number}" for number in range(1, 10)),
        *(f"LPT{number}" for number in range(1, 10)),
    }
    if folder_name.split(".", 1)[0].upper() in reserved_names:
        folder_name = f"{folder_name}_project"
    return folder_name


def create_project(
    project_name: str,
    audio_file: Any,
    script_file: Any = None,
    pasted_script: str = "",
    *,
    projects_root: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Create a new schema-v1 project atomically from an audio and script."""
    folder_name = _safe_project_folder_name(project_name)
    audio_path = _uploaded_path(audio_file, "audio")
    if audio_path.suffix.lower() not in {".mp3", ".wav"}:
        raise ValueError("Audio must be an MP3 or WAV file.")
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    try:
        get_audio_duration(audio_path)
    except Exception as exc:
        raise ValueError(
            "Narrato could not read this audio file. Choose a valid MP3 or WAV."
        ) from exc

    script_path: Optional[Path] = None
    script_text = pasted_script if isinstance(pasted_script, str) else ""
    if script_file is not None and (
        not isinstance(script_file, (str, Path)) or str(script_file).strip()
    ):
        if script_text.strip():
            raise ValueError("Choose a script file or paste script text, not both.")
        script_path = _uploaded_path(script_file, "script")
        if script_path.suffix.lower() not in {".txt", ".md"}:
            raise ValueError("Script files must be TXT or Markdown.")
        if not script_path.is_file():
            raise FileNotFoundError(f"Script file not found: {script_path}")
        try:
            script_text = script_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ValueError(f"Could not read the script as UTF-8: {exc}") from exc
    elif not script_text.strip():
        raise ValueError("Upload a TXT/Markdown script or paste script text.")
    if not script_text.strip():
        raise ValueError("The script is empty. Add script text before creating a project.")

    root = (
        Path(projects_root).expanduser().resolve()
        if projects_root is not None
        else (_PROJECT_ROOT / "projects").resolve()
    )
    root.mkdir(parents=True, exist_ok=True)
    project_path = root / folder_name
    if project_path.exists():
        raise FileExistsError(
            f"A project named “{project_name.strip()}” already exists. "
            "Choose another name or open the existing project."
        )

    staging_dir = Path(tempfile.mkdtemp(prefix=f".{folder_name}.", dir=root))
    try:
        if script_path is None:
            script_source = staging_dir / "script.txt"
            script_source.write_text(script_text, encoding="utf-8")
        else:
            script_source = script_path

        created_at = datetime.now(timezone.utc).isoformat()
        project = save_project_assets(
            staging_dir,
            audio_path,
            script_source,
            project_name=project_name.strip(),
            created_at=created_at,
        )
        project["metadata"]["creation_state"] = "created"
        _write_project_metadata(staging_dir / "project.json", project["metadata"])
        os.rename(staging_dir, project_path)
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise

    return load_project_workspace(project_path)


def select_project_mode(project_dir: str | Path, mode: str) -> str:
    """Persist an explicit workflow choice and return its workspace tab ID."""
    if mode not in _PROJECT_MODES:
        raise ValueError("Choose Assisted or Automatic Visual Creation.")
    project = load_project_workspace(project_dir)
    metadata = dict(project["metadata"])
    metadata["selected_mode"] = mode
    metadata["creation_state"] = "workflow_selected"
    _write_project_metadata(project["project_dir"] / "project.json", metadata)
    return _PROJECT_MODES[mode]


def preview_new_project_inputs(
    audio_file: Any,
    script_file: Any,
    pasted_script: str,
) -> tuple[str, Optional[str], str, Optional[str], str, str]:
    """Prepare uploaded audio and script details for review in the new-project form."""
    audio_path = None
    audio_name = ""
    audio_status = "Choose an MP3 or WAV narration file."
    if audio_file is not None:
        try:
            path = _uploaded_path(audio_file, "audio")
            if path.suffix.lower() not in {".mp3", ".wav"}:
                raise ValueError("Audio must be an MP3 or WAV file.")
            if not path.is_file():
                raise FileNotFoundError(f"Audio file not found: {path}")
            audio_path = str(path)
            audio_name = path.name
            try:
                duration = get_audio_duration(path)
                audio_status = f"Audio duration: {duration:.2f} seconds."
            except Exception:
                audio_status = "Audio selected; duration could not be determined."
        except (FileNotFoundError, OSError, ValueError) as exc:
            audio_status = str(exc)

    script_name = ""
    script_text = ""
    script_status = ""
    if script_file is not None:
        try:
            path = _uploaded_path(script_file, "script")
            if path.suffix.lower() not in {".txt", ".md"}:
                raise ValueError("Script files must be TXT or Markdown.")
            if pasted_script.strip():
                raise ValueError("Choose a script file or paste script text, not both.")
            script_text = path.read_text(encoding="utf-8")
            if not script_text.strip():
                raise ValueError("The selected script file is empty.")
            script_name = path.name
        except (FileNotFoundError, OSError, UnicodeError, ValueError) as exc:
            script_status = str(exc)
    elif isinstance(pasted_script, str) and pasted_script.strip():
        script_name = "Pasted script"
        script_text = pasted_script
    else:
        script_status = "Upload a TXT/Markdown script or paste script text."

    status = " ".join(message for message in (audio_status, script_status) if message)
    return status, audio_path, audio_name, audio_status, script_name, script_text


def create_project_callback(
    project_name: str,
    audio_file: Any,
    script_file: Any,
    pasted_script: str,
):
    try:
        project = create_project(
            project_name,
            audio_file,
            script_file,
            pasted_script,
        )
        return (
            f"**Project created:** {project['project_name']}  \n"
            f"Project folder: `{project['project_dir']}`. Choose a workflow to continue.",
            str(project["project_dir"]),
            gr_update(visible=True),
            gr_update(interactive=False),
        )
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        logger.warning("Project creation rejected: %s", _safe_diagnostic(str(exc)))
        return (
            f"**Could not create project:** {exc}",
            None,
            gr_update(visible=False),
            gr_update(interactive=True),
        )


def route_project_to_mode(project_dir: str | Path, mode: str):
    tab_index = select_project_mode(project_dir, mode)
    project = load_project_workspace(project_dir)
    return project, tab_index


def _marker_table_rows(markers: Sequence[VisualMarker]) -> list[list[Any]]:
    ordered = sorted(markers, key=lambda marker: (marker.start_time, marker.marker_id))
    return [
        [
            marker.marker_id,
            marker.start_time,
            "" if marker.end_time is None else marker.end_time,
            marker.image_prompt,
            marker.image_path or "",
        ]
        for marker in ordered
    ]


def _project_markers_path(project_dir: str | Path) -> Path:
    return load_project_workspace(project_dir)["markers_path"]


def add_project_marker(
    project_dir: str | Path,
    start_time: float,
    end_time: Optional[float] = None,
    image_prompt: str = "",
) -> VisualMarker:
    markers_path = _project_markers_path(project_dir)
    markers = load_markers(markers_path)
    marker = create_marker(markers, start_time, end_time, image_prompt)
    save_markers(markers_path, markers)
    return marker


def update_project_marker(
    project_dir: str | Path,
    marker_id: str,
    start_time: float,
    end_time: Optional[float],
    image_prompt: str,
) -> VisualMarker:
    markers_path = _project_markers_path(project_dir)
    markers = load_markers(markers_path)
    marker = edit_marker(
        markers,
        marker_id,
        start_time=start_time,
        end_time=end_time,
        image_prompt=image_prompt,
    )
    save_markers(markers_path, markers)
    return marker


def _assisted_marker_id(scene: Scene) -> str:
    segment_hash = hashlib.sha256(scene.text.encode("utf-8")).hexdigest()[:12]
    return f"assisted-{scene.index:04d}-{segment_hash}"


def _assisted_segments(project: dict[str, Any]) -> list[dict[str, Any]]:
    script_text = project["script_path"].read_text(encoding="utf-8")
    scenes = parse_script_text(script_text)
    if not scenes:
        raise ValueError("No readable script segments were found in this project.")
    assign_timestamps(scenes, project["audio_path"])

    markers = load_markers(project["markers_path"])
    markers_by_id = {marker.marker_id: marker for marker in markers}
    changed = False
    for scene in scenes:
        marker_id = _assisted_marker_id(scene)
        if marker_id not in markers_by_id:
            marker = VisualMarker(
                marker_id=marker_id,
                start_time=scene.start_time,
                end_time=scene.end_time,
            )
            markers.append(marker)
            markers_by_id[marker_id] = marker
            changed = True
    if changed:
        save_markers(project["markers_path"], markers)

    with _MARKER_JOB_LOCK:
        job = dict(_MARKER_JOB)
    segments = []
    for scene in scenes:
        marker_id = _assisted_marker_id(scene)
        marker = markers_by_id[marker_id]
        image_path = _marker_image_path(project["project_dir"], marker.image_path)
        if job["marker_id"] == marker_id and job["status"] in {"starting", "running"}:
            status = "Generating"
        elif job["marker_id"] == marker_id and job["status"] == "failed":
            status = "Failed"
        elif image_path is not None:
            status = "Image saved"
        elif marker.image_prompt.strip():
            status = "Prompt saved"
        else:
            status = "No image"
        segments.append(
            {
                "segment_id": marker_id,
                "index": scene.index,
                "title": scene.title,
                "text": scene.text,
                "estimated_start": scene.start_time,
                "estimated_end": scene.end_time,
                "marker": marker,
                "image_path": str(image_path) if image_path else None,
                "status": status,
            }
        )
    return segments


def load_assisted_workspace(project_dir: str | Path) -> dict[str, Any]:
    """Load script segments, estimated timings, and their persisted visual markers."""
    project = load_project_workspace(project_dir)
    project["segments"] = _assisted_segments(project)
    project["markers"] = load_markers(project["markers_path"])
    project["audio_duration"] = get_audio_duration(project["audio_path"])
    project["timing_note"] = (
        "Segment times are proportional estimates based on script word counts, "
        "not word-level audio alignment. Scrub the audio player and adjust marker "
        "boundaries to match the narration."
    )
    return project


def _automatic_project_paths(project_dir: str | Path) -> dict[str, Path]:
    project_path = _project_path(project_dir)
    automatic_dir = project_path / "automatic"
    return {
        "project_dir": project_path,
        "plan_path": automatic_dir / "scene_plan.json",
        "output_dir": automatic_dir / "images",
        "manifest_path": automatic_dir / "image_manifest.json",
    }


def _write_automatic_plan(path: Path, scenes: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(list(scenes), indent=2, ensure_ascii=False) + "\n"
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(serialized)
        with _AUTOMATIC_PLAN_LOCK:
            if path.exists():
                raise FileExistsError(
                    f"Automatic scene plan already exists and was not replaced: {path}"
                )
            os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _automatic_marker_ids(
    project: dict[str, Any], scenes: Sequence[dict[str, Any]]
) -> dict[int, str]:
    segments = _assisted_segments(project)
    segment_ids = {
        segment["index"]: segment["segment_id"] for segment in segments
    }
    marker_ids = {}
    for scene in scenes:
        index = scene["index"]
        marker_id = segment_ids.get(index)
        if marker_id is None:
            raise ValueError(
                f"Automatic scene {index} cannot be matched to a project marker."
            )
        marker_ids[index] = marker_id
    return marker_ids


def _automatic_scene_rows(project_dir: str | Path) -> list[list[Any]]:
    project = load_project_workspace(project_dir)
    paths = _automatic_project_paths(project["project_dir"])
    if not paths["plan_path"].is_file():
        return []
    scenes = _read_plan(str(paths["plan_path"]))
    marker_ids = _automatic_marker_ids(project, scenes)
    markers = {
        marker.marker_id: marker for marker in load_markers(project["markers_path"])
    }
    manifest = load_manifest(paths["manifest_path"])
    with _JOB_LOCK:
        job_matches = _JOB.get("plan_path") == str(paths["plan_path"])
        job_status = dict(_JOB.get("scene_status", {})) if job_matches else {}
    rows = []
    for scene in scenes:
        index = scene["index"]
        marker = markers[marker_ids[index]]
        image_path = _marker_image_path(project["project_dir"], marker.image_path)
        result = manifest.get(index)
        if image_path is not None:
            status = "Completed"
            shown_image = str(image_path)
            error = ""
        elif job_status.get(index) in {"queued", "running"}:
            status = (
                "Queued" if job_status[index] == "queued" else "Generating"
            )
            shown_image = ""
            error = ""
        elif result is not None and result.status == "error":
            status = "Failed"
            shown_image = ""
            error = _safe_diagnostic(result.error)
        elif (
            result is not None
            and result.status in {"ok", "skipped"}
            and result.image_path
            and Path(result.image_path).is_file()
        ):
            status = "Completed"
            shown_image = result.image_path
            error = ""
        elif job_status.get(index) == "cancelled":
            status = "Cancelled"
            shown_image = ""
            error = "Cancelled before this scene started."
        else:
            status = "Planned"
            shown_image = ""
            error = ""
        rows.append(
            [
                index,
                scene["title"],
                scene["text"],
                f"{scene['start_time']:.2f}–{scene['end_time']:.2f}s (estimated)",
                marker.image_prompt or scene["image_prompt"],
                status,
                shown_image,
                error,
            ]
        )
    return rows


def _automatic_scene_gallery(project_dir: str | Path) -> list[tuple[str, str]]:
    project = load_project_workspace(project_dir)
    paths = _automatic_project_paths(project["project_dir"])
    if not paths["plan_path"].is_file():
        return []
    scenes = _read_plan(str(paths["plan_path"]))
    marker_ids = _automatic_marker_ids(project, scenes)
    markers = {
        marker.marker_id: marker for marker in load_markers(project["markers_path"])
    }
    manifest = load_manifest(paths["manifest_path"])
    images = []
    for scene in scenes:
        marker = markers[marker_ids[scene["index"]]]
        image_path = _marker_image_path(project["project_dir"], marker.image_path)
        if image_path is None:
            result = manifest.get(scene["index"])
            if result is not None and result.image_path:
                candidate = Path(result.image_path).resolve()
                try:
                    candidate.relative_to(project["project_dir"])
                except ValueError:
                    continue
                image_path = candidate if candidate.is_file() else None
        if image_path is not None:
            images.append(
                (str(image_path), f"Scene {scene['index']}: {scene['title']}")
            )
    return images


def load_automatic_project(project_dir: str | Path) -> dict[str, Any]:
    """Load or create a reviewable project-local visual plan without generating."""
    project = load_project_workspace(project_dir)
    paths = _automatic_project_paths(project["project_dir"])
    with _AUTOMATIC_PLAN_LOCK:
        if paths["plan_path"].exists():
            scenes = _read_plan(str(paths["plan_path"]))
        else:
            segments = _assisted_segments(project)
            segment_by_index = {
                segment["index"]: segment for segment in segments
            }
            planned = plan_script_scenes(
                project["script_path"].read_text(encoding="utf-8"),
                project["audio_path"],
            )
            scenes = []
            markers = load_markers(project["markers_path"])
            markers_by_id = {marker.marker_id: marker for marker in markers}
            changed = False
            for scene in planned:
                segment = segment_by_index[scene.index]
                marker = markers_by_id[segment["segment_id"]]
                if marker.image_prompt.strip():
                    scene.image_prompt = marker.image_prompt
                else:
                    marker.image_prompt = scene.image_prompt
                    changed = True
                scene_data = scene.to_dict()
                scene_data["marker_id"] = marker.marker_id
                scenes.append(scene_data)
            if changed:
                save_markers(project["markers_path"], markers)
            _write_automatic_plan(paths["plan_path"], scenes)
    marker_ids = _automatic_marker_ids(project, scenes)
    project.update(
        {
            "automatic_paths": paths,
            "automatic_scenes": scenes,
            "automatic_marker_ids": marker_ids,
            "audio_duration": get_audio_duration(project["audio_path"]),
            "timing_note": (
                "Scene times are proportional estimates based on script word "
                "counts, not word-level audio alignment."
            ),
        }
    )
    return project


def load_automatic_project_callback(project_dir: str | Path):
    try:
        project = load_automatic_project(project_dir)
        scenes = project["automatic_scenes"]
        scene_choices = [
            (f"{scene['index']:03d} · {scene['title']}", str(scene["index"]))
            for scene in scenes
        ]
        rows = _automatic_scene_rows(project["project_dir"])
        return (
            str(project["automatic_paths"]["plan_path"]),
            str(project["automatic_paths"]["output_dir"]),
            str(project["automatic_paths"]["manifest_path"]),
            (
                f"**{len(scenes)} scenes planned for "
                f"{project['project_name']}** · Review the scenes and prompts "
                "below, then explicitly start generation."
            ),
            project["timing_note"],
            rows,
            _automatic_scene_gallery(project["project_dir"]),
            gr_update(choices=scene_choices, value=[value for _, value in scene_choices]),
            gr_update(choices=[], value=None),
        )
    except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
        logger.warning(
            "Automatic project planning rejected: %s", _safe_diagnostic(str(exc))
        )
        return (
            gr_update(),
            gr_update(),
            gr_update(),
            f"**Could not load automatic project:** {exc}",
            gr_update(),
            gr_update(),
            gr_update(),
            gr_update(),
            gr_update(),
        )


def _assisted_timing_label(segment: dict[str, Any]) -> str:
    marker = segment["marker"]
    end_time = (
        marker.end_time
        if marker.end_time is not None
        else segment["estimated_end"]
    )
    matches_estimate = (
        marker.end_time is not None
        and math.isclose(
            marker.start_time, segment["estimated_start"], rel_tol=0, abs_tol=0.001
        )
        and math.isclose(
            marker.end_time, segment["estimated_end"], rel_tol=0, abs_tol=0.001
        )
    )
    if matches_estimate:
        timing_source = "estimated"
    elif marker.end_time is None:
        timing_source = "marker start · estimated end"
    else:
        timing_source = "marker"
    return (
        f"{marker.start_time:.2f}–{end_time:.2f}s ({timing_source})"
    )


def assisted_segment_rows(project_dir: str | Path) -> list[list[Any]]:
    segments = load_assisted_workspace(project_dir)["segments"]
    return [
        [
            segment["index"],
            segment["title"],
            _assisted_timing_label(segment),
            segment["status"],
        ]
        for segment in segments
    ]


def assisted_segment_choices(project_dir: str | Path):
    segments = load_assisted_workspace(project_dir)["segments"]
    return [
        (
            f"{segment['index']:02d} · {segment['title']} · "
            f"{_assisted_timing_label(segment)}",
            segment["segment_id"],
        )
        for segment in segments
    ]


def select_assisted_segment(
    project_dir: str | Path, segment_id: Optional[str]
) -> tuple[str, Optional[float], Optional[float], str, Optional[str], str]:
    if not segment_id:
        return "", None, None, "", None, "Select a script segment."
    project = load_assisted_workspace(project_dir)
    segment = next(
        (
            item for item in project["segments"]
            if item["segment_id"] == segment_id
        ),
        None,
    )
    if segment is None:
        marker = next(
            (item for item in project["markers"] if item.marker_id == segment_id),
            None,
        )
        if marker is None:
            raise KeyError(f"Assisted script segment not found: {segment_id}")
        image_path = _marker_image_path(project["project_dir"], marker.image_path)
        return (
            "",
            marker.start_time,
            marker.end_time,
            marker.image_prompt,
            str(image_path) if image_path else None,
            f"Manual marker · {marker.start_time:.2f}s",
        )
    marker = segment["marker"]
    return (
        segment["text"],
        marker.start_time,
        marker.end_time,
        marker.image_prompt,
        segment["image_path"],
        f"**Segment {segment['index']}** · {_assisted_timing_label(segment)} · "
        f"{segment['status']}",
    )


def select_assisted_segment_ui(project_dir: str | Path, segment_id: Optional[str]):
    if not segment_id:
        return tuple(gr_update() for _ in range(8))
    values = select_assisted_segment(project_dir, segment_id)
    markers = load_project_workspace(project_dir)["markers"]
    marker_choices = [
        (
            f"{marker.start_time:.3f}s · {marker.marker_id[:8]}",
            marker.marker_id,
        )
        for marker in sorted(
            markers, key=lambda item: (item.start_time, item.marker_id)
        )
    ]
    return (
        gr_update(choices=marker_choices, value=segment_id),
        *values,
        values[4],
    )


def select_assisted_marker_ui(project_dir: str | Path, marker_id: Optional[str]):
    if not marker_id:
        return tuple(gr_update() for _ in range(8))
    values = select_assisted_segment(project_dir, marker_id)
    project = load_assisted_workspace(project_dir)
    segment_choices = [
        (
            f"{segment['index']:02d} · {segment['title']} · "
            f"{_assisted_timing_label(segment)}",
            segment["segment_id"],
        )
        for segment in project["segments"]
    ]
    selected_segment_id = next(
        (
            segment["segment_id"]
            for segment in project["segments"]
            if segment["segment_id"] == marker_id
        ),
        None,
    )
    return (
        gr_update(choices=segment_choices, value=selected_segment_id),
        *values,
        values[4],
    )


def save_assisted_segment(
    project_dir: str | Path,
    segment_id: str,
    start_time: Optional[float],
    end_time: Optional[float],
    image_prompt: str,
) -> VisualMarker:
    if not segment_id:
        raise ValueError("Select a script segment before saving its prompt.")
    if start_time is None:
        raise ValueError("Enter a segment start time.")
    project = load_project_workspace(project_dir)
    scenes = parse_script_text(project["script_path"].read_text(encoding="utf-8"))
    if not any(_assisted_marker_id(scene) == segment_id for scene in scenes):
        raise KeyError(f"Script segment not found: {segment_id}")
    return update_project_marker(
        project_dir,
        segment_id,
        start_time,
        end_time,
        image_prompt,
    )


def navigate_assisted_segment(
    project_dir: str | Path, selected_segment_id: Optional[str], direction: int
) -> tuple[
    Any, Any, str, Optional[float], Optional[float], str, Optional[str], str
]:
    project = load_assisted_workspace(project_dir)
    choices = [
        (
            f"{segment['index']:02d} · {segment['title']} · "
            f"{_assisted_timing_label(segment)}",
            segment["segment_id"],
        )
        for segment in project["segments"]
    ]
    segment_ids = [value for _, value in choices]
    if not segment_ids:
        raise ValueError("This project has no script segments to navigate.")
    if direction not in {-1, 1}:
        raise ValueError("Segment navigation direction must be previous or next.")
    if selected_segment_id not in segment_ids:
        selected_index = 0 if direction > 0 else len(segment_ids) - 1
    else:
        selected_index = segment_ids.index(selected_segment_id)
        selected_index = (selected_index + direction) % len(segment_ids)
    selected_id = segment_ids[selected_index]
    marker_choices = [
        (
            f"{marker.start_time:.3f}s · {marker.marker_id[:8]}",
            marker.marker_id,
        )
        for marker in sorted(
            project["markers"], key=lambda item: (item.start_time, item.marker_id)
        )
    ]
    return (
        gr_update(choices=choices, value=selected_id),
        gr_update(choices=marker_choices, value=selected_id),
        *select_assisted_segment(project_dir, selected_id),
    )


def remove_project_marker(project_dir: str | Path, marker_id: str) -> VisualMarker:
    markers_path = _project_markers_path(project_dir)
    markers = load_markers(markers_path)
    marker = delete_marker(markers, marker_id)
    save_markers(markers_path, markers)
    return marker


def _marker_image_path(project_dir: str | Path, stored_path: Optional[str]) -> Optional[Path]:
    if not stored_path:
        return None
    project_path = _project_path(project_dir)
    image_path = Path(stored_path)
    if not image_path.is_absolute():
        image_path = (project_path / image_path).resolve()
        if project_path not in image_path.parents:
            raise ValueError("Marker image path must stay inside its project folder.")
    return image_path if image_path.is_file() else None


def _next_marker_image_path(project_dir: Path, marker_id: str) -> Path:
    image_dir = project_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    stable_id = hashlib.sha256(marker_id.encode("utf-8")).hexdigest()
    version = 1
    while True:
        output_path = image_dir / f"marker_{stable_id}_v{version:03d}.png"
        if not output_path.exists():
            return output_path
        version += 1


def _validate_marker_image(path: Path) -> None:
    if not path.is_file():
        raise RuntimeError(f"ComfyUI did not create the marker image: {path}")
    try:
        from PIL import Image

        with Image.open(path) as image:
            if image.format != "PNG":
                raise ValueError("Expected a PNG image.")
            image.verify()
    except (OSError, ValueError) as exc:
        raise ValueError(f"Generated marker image is invalid: {exc}") from exc


def _set_marker_job(**changes: Any) -> None:
    with _MARKER_JOB_LOCK:
        _MARKER_JOB.update(changes)


def _marker_generation_worker(
    project_path: Path,
    marker_id: str,
    output_path: Path,
    prompt: str,
    negative_prompt: str,
    backend: ComfyUIBackend,
    retries: int,
    aspect_ratio: Optional[str],
) -> None:
    try:
        logger.info("Marker image generation started for marker %s", marker_id)
        _set_marker_job(
            status="running",
            progress=10,
            message="Checking ComfyUI connection, workflow, and model…",
        )
        backend.preflight()
        last_error = ""
        result = None
        for attempt in range(retries + 1):
            _set_marker_job(
                progress=25 + int(60 * attempt / max(retries + 1, 1)),
                message=(
                    f"Generating selected marker image "
                    f"(attempt {attempt + 1} of {retries + 1})…"
                ),
            )
            result = backend.generate(
                marker_id,
                prompt,
                output_path,
                ImageOptions(
                    aspect_ratio=aspect_ratio,
                    negative_prompt=negative_prompt,
                ),
            )
            if result.success and result.output_path is not None:
                if Path(result.output_path).resolve() != output_path.resolve():
                    last_error = (
                        "ComfyUI returned an image path different from the selected "
                        "marker output path."
                    )
                    if attempt < retries:
                        time.sleep(1)
                    continue
                try:
                    _validate_marker_image(output_path)
                except (OSError, ValueError, RuntimeError) as exc:
                    last_error = str(exc)
                else:
                    break
            else:
                last_error = result.error or "ComfyUI did not return an image."
            if attempt < retries:
                time.sleep(1)
        else:
            raise RuntimeError(last_error or "Marker image generation failed.")

        relative_image_path = output_path.resolve().relative_to(
            project_path.resolve()
        ).as_posix()
        markers_path = project_path / "markers.json"
        markers = load_markers(markers_path)
        edit_marker(markers, marker_id, image_path=relative_image_path)
        save_markers(markers_path, markers)
        _set_marker_job(
            status="completed",
            progress=100,
            message=f"Image generated for marker `{marker_id}`.",
            image_path=str(output_path),
            error="",
        )
        logger.info("Marker image generation completed for marker %s", marker_id)
    except Exception as exc:
        _log_exception(
            f"Marker image generation failed for marker {marker_id}", exc
        )
        _set_marker_job(
            status="failed",
            progress=100,
            message=f"**Image generation failed:** {exc}",
            error=str(exc),
        )


def start_marker_generation(
    project_dir: str | Path,
    marker_id: Optional[str],
    comfyui_url: str,
    workflow_path: str,
    positive_quality_prompt: str,
    negative_quality_prompt: str,
    confirm_regeneration: bool = False,
    retries: int = 2,
    aspect_ratio: Optional[str] = None,
) -> str:
    """Start one selected marker's local ComfyUI image job."""
    if not marker_id:
        raise ValueError("Select a manual marker before generating an image.")
    if not isinstance(positive_quality_prompt, str) or not isinstance(
        negative_quality_prompt, str
    ):
        raise ValueError("Quality prompts must be text.")
    try:
        retry_count = int(retries)
    except (TypeError, ValueError) as exc:
        raise ValueError("Retries must be a whole number between 0 and 10.") from exc
    if retry_count != retries or not 0 <= retry_count <= 10:
        raise ValueError("Retries must be a whole number between 0 and 10.")
    if aspect_ratio == "Workflow dimensions":
        aspect_ratio = None
    if aspect_ratio is not None and aspect_ratio not in SUPPORTED_ASPECT_RATIOS:
        raise ValueError(f"Unsupported aspect ratio: {aspect_ratio}")

    project = load_project_workspace(project_dir)
    marker = next(
        (item for item in project["markers"] if item.marker_id == marker_id),
        None,
    )
    if marker is None:
        raise KeyError(f"Marker not found: {marker_id}")
    if not marker.image_prompt.strip():
        raise ValueError("Add a custom image prompt to this marker before generation.")
    previous_image = _marker_image_path(project["project_dir"], marker.image_path)
    stable_id = hashlib.sha256(marker_id.encode("utf-8")).hexdigest()
    prior_versions = list(
        (project["project_dir"] / "images").glob(f"marker_{stable_id}_v*.png")
    )
    if (previous_image or prior_versions) and not confirm_regeneration:
        raise ValueError(
            "This marker already has a generated image. Confirm regeneration to "
            "create a new version."
        )

    backend = ComfyUIBackend(
        base_url=comfyui_url.strip(),
        workflow_path=workflow_path.strip(),
    )
    prompt = compose_prompt(positive_quality_prompt, marker.image_prompt.strip())
    negative_prompt = compose_prompt(negative_quality_prompt, "")
    project_path = project["project_dir"]
    with _MARKER_JOB_LOCK:
        running_thread = _MARKER_JOB.get("thread")
        if (
            _MARKER_JOB["status"] in {"starting", "running"}
            or running_thread is not None and running_thread.is_alive()
        ):
            raise RuntimeError("A manual marker image job is already running.")
        output_path = _next_marker_image_path(project_path, marker_id)
        _MARKER_JOB.update({
            "status": "starting",
            "marker_id": marker_id,
            "project_dir": str(project_path),
            "progress": 0,
            "message": "Preparing selected marker generation…",
            "image_path": None,
            "error": "",
            "thread": None,
        })
        thread = threading.Thread(
            target=_marker_generation_worker,
            args=(
                project_path,
                marker_id,
                output_path,
                prompt,
                negative_prompt,
                backend,
                retry_count,
                aspect_ratio,
            ),
            name="narrato-marker-image-generation",
            daemon=True,
        )
        _MARKER_JOB["thread"] = thread
        thread.start()
    return f"Started local image generation for marker `{marker_id}`."


def marker_generation_status() -> tuple[str, Any, Any]:
    with _MARKER_JOB_LOCK:
        status = _MARKER_JOB["status"]
        progress = _MARKER_JOB["progress"]
        message = _MARKER_JOB["message"] or "**Marker image job:** idle"
        image_path = _MARKER_JOB["image_path"]
    return message, gr_update(value=progress), image_path


@_log_studio_callback("generate_selected_marker")
def start_marker_generation_callback(
    project_dir: str,
    marker_id: Optional[str],
    comfyui_url: str,
    workflow_path: str,
    positive_quality_prompt: str,
    negative_quality_prompt: str,
    confirm_regeneration: bool,
    retries: int,
    aspect_ratio: str,
) -> tuple[str, bool]:
    try:
        message = start_marker_generation(
            project_dir,
            marker_id,
            comfyui_url,
            workflow_path,
            positive_quality_prompt,
            negative_quality_prompt,
            confirm_regeneration,
            retries,
            aspect_ratio,
        )
        return message, False
    except Exception as exc:
        diagnostic = _safe_diagnostic(str(exc))
        if isinstance(
            exc,
            (BackendConfigurationError, FileNotFoundError, KeyError, OSError, ValueError),
        ):
            logger.warning("Marker image generation could not start: %s", diagnostic)
        else:
            _log_exception("Marker image generation could not start unexpectedly", exc)
        message = f"**Could not start marker image generation:** {exc}"
        _set_marker_job(
            status="failed",
            progress=100,
            message=message,
            image_path=None,
            error=str(exc),
        )
        return message, False


def assisted_workspace_ui_state(project_dir: str | Path):
    project = load_assisted_workspace(project_dir)
    segments = project["segments"]
    choices = [
        (
            f"{segment['index']:02d} · {segment['title']} · "
            f"{_assisted_timing_label(segment)}",
            segment["segment_id"],
        )
        for segment in segments
    ]
    selected_id = choices[0][1] if choices else None
    selected = (
        select_assisted_segment(project_dir, selected_id)
        if selected_id
        else ("", None, None, "", None, "No script segments are available.")
    )
    markers = sorted(
        project["markers"], key=lambda marker: (marker.start_time, marker.marker_id)
    )
    return (
        [
            [
                segment["index"],
                segment["title"],
                _assisted_timing_label(segment),
                segment["status"],
            ]
            for segment in segments
        ],
        gr_update(choices=choices, value=selected_id),
        gr_update(
            choices=[
                (
                    f"{marker.start_time:.3f}s · {marker.marker_id[:8]}",
                    marker.marker_id,
                )
                for marker in markers
            ],
            value=selected_id,
        ),
        *selected,
        project["timing_note"],
    )


def load_assisted_workspace_callback(project_dir: str | Path):
    try:
        return assisted_workspace_ui_state(project_dir)
    except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
        logger.warning(
            "Assisted workspace load failed: %s", _safe_diagnostic(str(exc))
        )
        return (
            [],
            gr_update(choices=[], value=None),
            gr_update(choices=[], value=None),
            "",
            None,
            None,
            "",
            None,
            f"**Could not load assisted project:** {exc}",
            "",
        )


def save_assisted_segment_callback(
    project_dir: str,
    segment_id: Optional[str],
    start_time: Optional[float],
    end_time: Optional[float],
    image_prompt: str,
):
    try:
        marker = save_assisted_segment(
            project_dir, segment_id or "", start_time, end_time, image_prompt
        )
        project = load_assisted_workspace(project_dir)
        segment = next(
            (
                item for item in project["segments"]
                if item["segment_id"] == marker.marker_id
            ),
            None,
        )
        if segment is None:
            raise KeyError(f"Script segment not found: {marker.marker_id}")
        return (
            f"**Saved prompt and segment boundaries** for segment "
            f"{segment['index']}. No image was generated.",
            [
                [
                    item["index"],
                    item["title"],
                    _assisted_timing_label(item),
                    item["status"],
                ]
                for item in project["segments"]
            ],
            _marker_image_path(project_dir, marker.image_path),
            f"**Segment {segment['index']}** · {segment['status']}",
        )
    except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
        logger.warning(
            "Assisted segment save rejected: %s", _safe_diagnostic(str(exc))
        )
        return f"**Could not save segment:** {exc}", gr_update(), gr_update(), gr_update()


def autosave_assisted_segment_callback(
    project_dir: str,
    segment_id: Optional[str],
    start_time: Optional[float],
    end_time: Optional[float],
    image_prompt: str,
):
    return save_assisted_segment_callback(
        project_dir, segment_id, start_time, end_time, image_prompt
    )


def generate_assisted_segment_callback(
    project_dir: str,
    segment_id: Optional[str],
    start_time: Optional[float],
    end_time: Optional[float],
    image_prompt: str,
    comfyui_url: str,
    workflow_path: str,
    positive_quality_prompt: str,
    negative_quality_prompt: str,
    confirm_regeneration: bool,
    retries: int,
    aspect_ratio: str,
    *,
    retry: bool = False,
):
    try:
        marker = save_assisted_segment(
            project_dir, segment_id or "", start_time, end_time, image_prompt
        )
        message = start_marker_generation(
            project_dir,
            marker.marker_id,
            comfyui_url,
            workflow_path,
            positive_quality_prompt,
            negative_quality_prompt,
            confirm_regeneration or retry,
            retries,
            aspect_ratio,
        )
        confirmation = False
        image_path = _marker_image_path(project_dir, marker.image_path)
        with _MARKER_JOB_LOCK:
            is_running = (
                _MARKER_JOB.get("marker_id") == marker.marker_id
                and _MARKER_JOB["status"] in {"starting", "running"}
            )
        return (
            message,
            confirmation,
            str(image_path) if image_path else None,
            gr_update(interactive=not is_running),
            gr_update(interactive=not is_running),
        )
    except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as exc:
        logger.warning(
            "Assisted image generation rejected: %s", _safe_diagnostic(str(exc))
        )
        message = f"**Could not start image generation:** {exc}"
        with _MARKER_JOB_LOCK:
            running = _MARKER_JOB["status"] in {"starting", "running"}
        if not running:
            _set_marker_job(
                status="failed",
                progress=100,
                message=message,
                error=str(exc),
            )
        return (
            message,
            False,
            None,
            gr_update(interactive=True),
            gr_update(interactive=True),
        )


def refresh_assisted_segment_status(
    project_dir: str | Path, selected_segment_id: Optional[str]
):
    with _MARKER_JOB_LOCK:
        job = dict(_MARKER_JOB)
    active_job = job["status"] in {"starting", "running"}
    job_project_matches = job["project_dir"] == str(_project_path(project_dir))
    if (
        job["status"] not in {"starting", "running", "completed", "failed"}
        or not job_project_matches
        or job["marker_id"] != selected_segment_id
    ):
        return (
            gr_update(),
            gr_update(),
            gr_update(),
            gr_update(),
            gr_update(interactive=not active_job),
            gr_update(interactive=not active_job),
        )

    project = load_assisted_workspace(project_dir)
    segments = project["segments"]
    rows = [
        [
            segment["index"],
            segment["title"],
            _assisted_timing_label(segment),
            segment["status"],
        ]
        for segment in segments
    ]
    segment = next(
        (
            item for item in segments
            if item["segment_id"] == selected_segment_id
        ),
        None,
    )
    if segment and job["marker_id"] == selected_segment_id:
        message = job["message"] or f"**{segment['status']}**"
        running = job["status"] in {"starting", "running"}
        return (
            rows,
            message,
            gr_update(value=job["progress"]),
            segment["image_path"],
            gr_update(interactive=not running),
            gr_update(interactive=not running),
        )
    if segment:
        message = f"**Segment {segment['index']}** · {segment['status']}"
        return (
            rows,
            message,
            gr_update(value=0),
            segment["image_path"],
            gr_update(interactive=True),
            gr_update(interactive=True),
        )
    return (
        rows,
        gr_update(),
        gr_update(),
        gr_update(),
        gr_update(interactive=True),
        gr_update(interactive=True),
    )


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
    status_by_scene = (
        dict(current_status["scene_status"])
        if current_status["plan_path"] == str(_project_path(plan_path))
        else {}
    )
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
    sanitized = _redact_google_message(message, os.getenv("GEMINI_API_KEY", ""))
    return re.sub(r"(?i)(https?://)[^/\s@]+@", r"\1[redacted]@", sanitized)


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
            "plan_path": _JOB.get("plan_path"),
            "project_dir": _JOB.get("project_dir"),
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


def _persist_automatic_scene_result(
    project_dir: str | Path,
    plan_path: str | Path,
    scene_index: int,
    image_path: str | Path,
) -> None:
    project = load_project_workspace(project_dir)
    paths = _automatic_project_paths(project["project_dir"])
    if _project_path(plan_path) != paths["plan_path"]:
        raise ValueError("Automatic generation plan does not belong to this project.")
    plan_scenes = _read_plan(str(paths["plan_path"]))
    scene = next(
        (item for item in plan_scenes if item["index"] == scene_index), None
    )
    if scene is None:
        raise KeyError(f"Automatic scene {scene_index} is not in the project plan.")
    marker_id = _automatic_marker_ids(project, [scene])[scene_index]
    marker_path = Path(image_path).resolve()
    try:
        relative_image_path = marker_path.relative_to(project["project_dir"]).as_posix()
    except ValueError as exc:
        raise ValueError("Generated scene image must stay inside its project folder.") from exc
    markers = load_markers(project["markers_path"])
    marker = next(item for item in markers if item.marker_id == marker_id)
    changes = {}
    if not marker.image_path or marker.image_path != relative_image_path:
        changes["image_path"] = relative_image_path
    if not marker.image_prompt.strip() and scene["image_prompt"].strip():
        changes["image_prompt"] = scene["image_prompt"]
    if changes:
        edit_marker(markers, marker_id, **changes)
        save_markers(project["markers_path"], markers)


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
    project_dir: Optional[str | Path] = None,
) -> str:
    """Validate and start one non-blocking image-generation job."""
    plan_path = str(_project_path(plan_path))
    output_dir = str(_project_path(output_dir))
    manifest_path = str(_project_path(manifest_path))
    automatic_project = None
    if project_dir is not None:
        automatic_project = _automatic_project_paths(project_dir)
        if (
            _project_path(plan_path) != automatic_project["plan_path"]
            or _project_path(output_dir) != automatic_project["output_dir"]
            or _project_path(manifest_path) != automatic_project["manifest_path"]
        ):
            raise ValueError(
                "Automatic plan, image, and manifest paths must stay in the "
                "selected project's automatic folder."
            )
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
    if automatic_project is not None:
        project = load_project_workspace(automatic_project["project_dir"])
        marker_ids = _automatic_marker_ids(project, scenes)
        markers = {
            marker.marker_id: marker for marker in load_markers(project["markers_path"])
        }
        for scene in scenes:
            marker = markers[marker_ids[scene["index"]]]
            if marker.image_prompt.strip():
                scene["image_prompt"] = marker.image_prompt
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
            "plan_path": plan_path,
            "project_dir": (
                str(automatic_project["project_dir"])
                if automatic_project is not None
                else None
            ),
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
        temporary_plan_path: Optional[Path] = None
        try:
            def progress_callback(event, index, result, completed, total):
                _progress_callback(event, index, result, completed, total)
                if (
                    automatic_project is not None
                    and result is not None
                    and result.status in {"ok", "skipped"}
                    and result.image_path
                ):
                    _persist_automatic_scene_result(
                        automatic_project["project_dir"],
                        plan_path,
                        index,
                        result.image_path,
                    )

            generation_plan_path = Path(plan_path).expanduser()
            if automatic_project is not None:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=automatic_project["project_dir"] / "automatic",
                    prefix=".generation-plan.",
                    suffix=".json",
                    delete=False,
                ) as temporary_plan:
                    temporary_plan_path = Path(temporary_plan.name)
                    json.dump(scenes, temporary_plan, indent=2, ensure_ascii=False)
                    temporary_plan.write("\n")
                generation_plan_path = temporary_plan_path

            config = RunConfig(
                scene_plan_path=generation_plan_path,
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
                progress_callback=progress_callback,
                backend_instance=backend_instance,
            )
            run_image_generation(config)
        except Exception as exc:
            with _JOB_LOCK:
                _JOB["error"] = _safe_diagnostic(str(exc))
        finally:
            if temporary_plan_path is not None:
                try:
                    temporary_plan_path.unlink(missing_ok=True)
                except OSError as exc:
                    logger.warning(
                        "Could not remove temporary generation plan: %s",
                        _safe_diagnostic(str(exc)),
                    )
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


def start_automatic_generation(
    project_dir: str | Path,
    backend: str,
    aspect_ratio: str,
    selection_mode: str,
    selected_scenes: Optional[Sequence[str]],
    delay_seconds: float,
    retries: int,
    confirm_api: bool,
    replace_existing: bool,
    comfyui_url: str,
    workflow_path: str,
    positive_quality_prompt: str,
    negative_quality_prompt: str,
    theme: str,
    retry_scene: Optional[str] = None,
) -> str:
    project = load_automatic_project(project_dir)
    paths = project["automatic_paths"]
    scenes = project["automatic_scenes"]
    manifest = load_manifest(paths["manifest_path"])
    marker_ids = project["automatic_marker_ids"]
    markers = {
        marker.marker_id: marker for marker in load_markers(project["markers_path"])
    }

    if retry_scene:
        try:
            retry_index = int(retry_scene)
        except (TypeError, ValueError) as exc:
            raise ValueError("Choose a valid failed scene to retry.") from exc
        failed_result = manifest.get(retry_index)
        if failed_result is None or failed_result.status != "error":
            raise ValueError("Only a scene recorded as failed can be retried here.")
        selected_indexes = [retry_index]
    elif selection_mode == "Failed scenes only":
        selected_indexes = sorted(
            index for index, result in manifest.items()
            if result.status == "error"
        )
        if not selected_indexes:
            raise ValueError("There are no failed automatic scenes to retry.")
    elif selection_mode == "Selected scenes":
        try:
            selected_indexes = sorted({int(value) for value in selected_scenes or []})
        except (TypeError, ValueError) as exc:
            raise ValueError("Select valid scene numbers before starting.") from exc
        if not selected_indexes:
            raise ValueError("Select at least one planned scene.")
    elif selection_mode == "All scenes":
        selected_indexes = [scene["index"] for scene in scenes]
    else:
        raise ValueError("Choose all scenes, selected scenes, or failed scenes only.")

    scene_by_index = {scene["index"]: scene for scene in scenes}
    unknown = set(selected_indexes) - set(scene_by_index)
    if unknown:
        raise ValueError(f"Selected scenes are not in this project plan: {sorted(unknown)}")
    if not replace_existing and not retry_scene:
        pending_indexes = []
        for index in selected_indexes:
            marker = markers[marker_ids[index]]
            marker_image = _marker_image_path(
                project["project_dir"], marker.image_path
            )
            result = manifest.get(index)
            manifest_image = (
                Path(result.image_path)
                if result is not None and result.image_path
                else None
            )
            manifest_image_exists = (
                manifest_image is not None and manifest_image.is_file()
            )
            if marker_image is None and not (
                result is not None
                and result.status in {"ok", "skipped"}
                and manifest_image_exists
            ):
                pending_indexes.append(index)
        selected_indexes = pending_indexes
    if not selected_indexes:
        return "All selected scenes already have saved images; nothing was regenerated."

    return start_generation(
        backend=backend,
        plan_path=str(paths["plan_path"]),
        output_dir=str(paths["output_dir"]),
        manifest_path=str(paths["manifest_path"]),
        aspect_ratio=aspect_ratio,
        selection_mode="Selected scenes",
        selected_scenes=[str(index) for index in selected_indexes],
        concurrency=1,
        delay_seconds=delay_seconds,
        retries=retries,
        confirm_api=confirm_api,
        replace_existing=replace_existing,
        comfyui_url=comfyui_url,
        workflow_path=workflow_path,
        positive_quality_prompt=positive_quality_prompt,
        negative_quality_prompt=negative_quality_prompt,
        theme=theme,
        project_dir=project["project_dir"],
    )


def automatic_generation_callback(
    project_dir: str,
    backend: str,
    aspect_ratio: str,
    selection_mode: str,
    selected_scenes: Optional[Sequence[str]],
    delay_seconds: float,
    retries: int,
    confirm_api: bool,
    replace_existing: bool,
    comfyui_url: str,
    workflow_path: str,
    positive_quality_prompt: str,
    negative_quality_prompt: str,
    theme: str,
) -> str:
    try:
        return start_automatic_generation(
            project_dir,
            backend,
            aspect_ratio,
            selection_mode,
            selected_scenes,
            delay_seconds,
            retries,
            confirm_api,
            replace_existing,
            comfyui_url,
            workflow_path,
            positive_quality_prompt,
            negative_quality_prompt,
            theme,
        )
    except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as exc:
        logger.warning(
            "Automatic generation rejected: %s", _safe_diagnostic(str(exc))
        )
        return f"**Could not start automatic generation:** {exc}"


def retry_automatic_scene_callback(
    project_dir: str,
    backend: str,
    aspect_ratio: str,
    delay_seconds: float,
    retries: int,
    confirm_api: bool,
    comfyui_url: str,
    workflow_path: str,
    positive_quality_prompt: str,
    negative_quality_prompt: str,
    theme: str,
    scene_index: Optional[str],
) -> str:
    try:
        return start_automatic_generation(
            project_dir,
            backend,
            aspect_ratio,
            "Selected scenes",
            [scene_index] if scene_index else [],
            delay_seconds,
            retries,
            confirm_api,
            False,
            comfyui_url,
            workflow_path,
            positive_quality_prompt,
            negative_quality_prompt,
            theme,
            retry_scene=scene_index,
        )
    except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as exc:
        logger.warning(
            "Automatic scene retry rejected: %s", _safe_diagnostic(str(exc))
        )
        return f"**Could not retry automatic scene:** {exc}"


def refresh_automatic_scene_table(project_dir: str | Path):
    if not str(project_dir).strip():
        return gr_update(), gr_update(), gr_update()
    try:
        return (
            _automatic_scene_rows(project_dir),
            gr_update(),
            _automatic_scene_gallery(project_dir),
        )
    except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
        logger.warning(
            "Automatic scene status refresh failed: %s", _safe_diagnostic(str(exc))
        )
        return gr_update(), f"**Could not refresh scene status:** {exc}", gr_update()


def cancel_generation() -> str:
    with _JOB_LOCK:
        event = _JOB.get("cancel_event")
        thread = _JOB.get("thread")
        if event is None or thread is None or not thread.is_alive():
            return "No generation job is currently running."
        event.set()
        _JOB["status"] = "cancelling"
    return "Cancellation requested. In-progress backend calls will finish; queued scenes will not start."


@_log_studio_callback("backend_readiness")
def check_backend_readiness(
    backend: str, comfyui_url: str = "", workflow_path: str = ""
) -> str:
    """Check local ComfyUI connectivity or API configuration without an API call."""
    if backend == "api":
        try:
            api_backend = GeminiApiBackend()
        except BackendConfigurationError as exc:
            logger.warning("Backend readiness check failed: %s", _safe_diagnostic(str(exc)))
            return f"**Not ready:** {_safe_diagnostic(str(exc))}"
        return (
            f"**Configured** · Gemini model `{api_backend.model_identifier}`. "
            "No API request was made; quota and billing still need to be available."
        )
    if backend != "local":
        logger.warning("Backend readiness check rejected unsupported backend")
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
        logger.warning(
            "Backend readiness check failed: %s", _safe_diagnostic(str(exc))
        )
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


def _project_ui_state(project: dict[str, Any], status: str):
    audio_path = project["audio_path"]
    duration = get_audio_duration(audio_path)
    script_text = project["script_path"].read_text(encoding="utf-8")
    markers = project["markers"]
    return (
        status,
        str(audio_path),
        f"**Audio duration:** {duration:.3f} seconds. Timestamps are entered manually; "
        "no word-level alignment is performed.",
        script_text,
        _marker_table_rows(markers),
        gr_update(
            choices=[
                (f"{marker.start_time:.3f}s · {marker.marker_id[:8]}", marker.marker_id)
                for marker in sorted(
                    markers, key=lambda item: (item.start_time, item.marker_id)
                )
            ],
            value=None,
        ),
        None,
        None,
        "",
        None,
    )


@_log_studio_callback("save_project")
def save_project_callback(project_dir: str, audio_file: str, script_file: str):
    try:
        project = save_project_assets(project_dir, audio_file, script_file)
        status = f"**Project saved:** `{project['project_dir']}`"
        return _project_ui_state(project, status)
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        logger.warning("Project save rejected: %s", _safe_diagnostic(str(exc)))
        return (
            f"**Could not save project:** {exc}",
            None,
            "",
            "",
            [],
            gr_update(choices=[], value=None),
            None,
            None,
            "",
            None,
        )


@_log_studio_callback("load_project")
def load_project_callback(project_dir: str):
    try:
        project = load_project_workspace(project_dir)
        status = f"**Project loaded:** `{project['project_dir']}`"
        return _project_ui_state(project, status)
    except (FileNotFoundError, OSError, ValueError) as exc:
        logger.warning("Project load rejected: %s", _safe_diagnostic(str(exc)))
        return (
            f"**Could not load project:** {exc}",
            None,
            "",
            "",
            [],
            gr_update(choices=[], value=None),
            None,
            None,
            "",
            None,
        )


def select_project_marker(project_dir: str, marker_id: Optional[str]):
    if not marker_id:
        return None, None, "", None
    marker = next(
        (
            item for item in load_markers(_project_markers_path(project_dir))
            if item.marker_id == marker_id
        ),
        None,
    )
    if marker is None:
        raise KeyError(f"Marker not found: {marker_id}")
    return (
        marker.start_time,
        None if marker.end_time is None else marker.end_time,
        marker.image_prompt,
        str(_marker_image_path(project_dir, marker.image_path) or "") or None,
    )


def _marker_update_ui(project_dir: str, status: str, selected_id: Optional[str]):
    markers = load_markers(_project_markers_path(project_dir))
    ordered = sorted(markers, key=lambda item: (item.start_time, item.marker_id))
    fields = (
        select_project_marker(project_dir, selected_id)
        if selected_id
        else (None, None, "", None)
    )
    return (
        status,
        _marker_table_rows(markers),
        gr_update(
            choices=[
                (f"{marker.start_time:.3f}s · {marker.marker_id[:8]}", marker.marker_id)
                for marker in ordered
            ],
            value=selected_id,
        ),
        *fields,
    )


@_log_studio_callback("add_marker")
def add_marker_callback(
    project_dir: str, start_time: Optional[float], end_time: Optional[float],
    image_prompt: str,
):
    try:
        if start_time is None:
            raise ValueError("Enter a marker start time.")
        marker = add_project_marker(project_dir, start_time, end_time, image_prompt)
        return _marker_update_ui(
            project_dir, f"**Marker added:** `{marker.marker_id}`", marker.marker_id
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        logger.warning("Marker add rejected: %s", _safe_diagnostic(str(exc)))
        return (
            f"**Could not add marker:** {exc}",
            gr_update(),
            gr_update(),
            gr_update(),
            gr_update(),
            gr_update(),
            gr_update(),
        )


@_log_studio_callback("save_marker")
def save_marker_callback(
    project_dir: str, marker_id: Optional[str], start_time: Optional[float],
    end_time: Optional[float], image_prompt: str,
):
    try:
        if not marker_id:
            raise ValueError("Select a marker to edit.")
        if start_time is None:
            raise ValueError("Enter a marker start time.")
        marker = update_project_marker(
            project_dir, marker_id, start_time, end_time, image_prompt
        )
        return _marker_update_ui(
            project_dir, f"**Marker saved:** `{marker.marker_id}`", marker.marker_id
        )
    except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
        logger.warning("Marker save rejected: %s", _safe_diagnostic(str(exc)))
        return (
            f"**Could not save marker:** {exc}",
            gr_update(),
            gr_update(),
            gr_update(),
            gr_update(),
            gr_update(),
            gr_update(),
        )


@_log_studio_callback("delete_marker")
def delete_marker_callback(project_dir: str, marker_id: Optional[str]):
    try:
        if not marker_id:
            raise ValueError("Select a marker to delete.")
        remove_project_marker(project_dir, marker_id)
        return _marker_update_ui(
            project_dir, f"**Marker deleted:** `{marker_id}`", None
        )
    except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
        logger.warning("Marker delete rejected: %s", _safe_diagnostic(str(exc)))
        return (
            f"**Could not delete marker:** {exc}",
            gr_update(),
            gr_update(),
            gr_update(),
            gr_update(),
            gr_update(),
            gr_update(),
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
    :root {
        --narrato-bg: #F5F1E8;
        --narrato-surface: #FCFAF5;
        --narrato-secondary: #EAE3D7;
        --narrato-primary: #718067;
        --narrato-active: #485442;
        --narrato-secondary-accent: #80664F;
        --narrato-text: #38362F;
        --narrato-muted: #777367;
        --narrato-border: #E1D8C9;
        --narrato-shadow: rgba(56, 54, 47, 0.08);
    }
    body, .gradio-container {
        background: var(--narrato-bg) !important;
        color: var(--narrato-text) !important;
    }
    .gradio-container { max-width: 1500px !important; }
    .narrato-hero {
        padding: 28px 32px; border: 1px solid var(--narrato-border); border-radius: 18px;
        background: var(--narrato-surface); box-shadow: 0 12px 28px var(--narrato-shadow);
        margin-bottom: 18px;
    }
    .narrato-hero h1 {
        margin: 0 0 8px; font-size: 32px; line-height: 1.1; color: var(--narrato-text);
    }
    .narrato-subtitle { color: var(--narrato-muted); font-size: 16px; }
    .narrato-card {
        border: 1px solid var(--narrato-border); border-radius: 16px; background: var(--narrato-surface);
        box-shadow: 0 8px 18px var(--narrato-shadow);
    }
    .narrato-mode-card {
        padding: 18px 20px; border: 1px solid var(--narrato-border); border-radius: 18px;
        background: linear-gradient(135deg, rgba(113, 128, 103, 0.10), rgba(128, 102, 79, 0.08));
        min-height: 140px;
    }
    .narrato-mode-card h3 {
        margin: 0 0 8px; font-size: 1.1rem; color: var(--narrato-text);
    }
    .narrato-mode-card p {
        margin: 0; color: var(--narrato-muted); line-height: 1.5;
    }
    """
    with gr.Blocks(
        title="Narrato Studio",
        theme=gr.themes.Soft(primary_hue="green", neutral_hue="stone"),
        css=css,
    ) as demo:
        gr.HTML(
            '<div class="narrato-hero"><h1>Narrato</h1>'
            '<div class="narrato-subtitle">Two visual workflows. One shared audio-and-script project.</div></div>'
        )
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                gr.HTML(
                    '<div class="narrato-mode-card"><h3>Assisted Visual Creation</h3>'
                    '<p>Choose each moment, write your prompt, and create visuals at your own pace.</p></div>'
                )
            with gr.Column(scale=1):
                gr.HTML(
                    '<div class="narrato-mode-card"><h3>Automatic Visual Creation</h3>'
                    '<p>Let Narrato plan scenes and generate visuals from your narration.</p></div>'
                )
        with gr.Row():
            new_project_nav_button = gr.Button("New Project", size="sm")
            for label in ["Recent Projects", "Continue Working", "Projects", "Settings"]:
                gr.Button(label, size="sm")
        if settings_warning:
            gr.Markdown(f"**Settings warning:** {settings_warning}")
        with gr.Tabs(selected="new-project") as tabs:
            with gr.Tab("New Project", id="new-project"):
                with gr.Group(elem_classes=["narrato-card"]):
                    gr.Markdown("### Start a project")
                    gr.Markdown(
                        "Every Narrato project begins with your narration and its script. "
                        "Supported audio: MP3 and WAV. Script: TXT, Markdown, or pasted text."
                    )
                    project_name_input = gr.Textbox(
                        label="Project name",
                        placeholder="For example, The history of flight",
                        max_lines=1,
                    )
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=1):
                            new_audio_upload = _StudioFile(
                                label="Narration audio (.mp3 or .wav)",
                                file_types=[".mp3", ".wav"],
                                type="filepath",
                                interactive=True,
                            )
                            new_audio_filename = gr.Textbox(
                                label="Selected audio", interactive=False,
                            )
                            new_audio_preview = gr.Audio(
                                label="Audio preview",
                                type="filepath",
                                interactive=False,
                                show_download_button=False,
                            )
                            new_audio_status = gr.Markdown(
                                "Choose an MP3 or WAV narration file."
                            )
                        with gr.Column(scale=1):
                            new_script_upload = _StudioFile(
                                label="Script file (.txt or .md)",
                                file_types=[".txt", ".md"],
                                type="filepath",
                                interactive=True,
                            )
                            new_script_filename = gr.Textbox(
                                label="Selected script", interactive=False,
                            )
                            pasted_script_input = gr.TextArea(
                                label="Or paste script text",
                                placeholder="Paste or review your script here.",
                                lines=8,
                            )
                            script_review = gr.TextArea(
                                label="Script review",
                                lines=8,
                                interactive=False,
                            )
                    new_project_input_status = gr.Markdown()
                    project_create_status = gr.Markdown()
                    create_project_button = gr.Button(
                        "Create Project", variant="primary",
                    )
                    created_project_dir = gr.State(value=None)
                    with gr.Group(visible=False) as workflow_choices:
                        gr.Markdown("### Choose how to create your visuals")
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=1, elem_classes=["narrato-mode-card"]):
                                gr.Markdown(
                                    "### Assisted Visual Creation\n"
                                    "Choose each script segment, write your own prompt, "
                                    "and generate visuals one at a time."
                                )
                                assisted_mode_button = gr.Button(
                                    "Choose Assisted Visual Creation",
                                    variant="primary",
                                )
                            with gr.Column(scale=1, elem_classes=["narrato-mode-card"]):
                                gr.Markdown(
                                    "### Automatic Visual Creation\n"
                                    "Let Narrato plan scenes and generate visuals from "
                                    "your script and narration."
                                )
                                automatic_mode_button = gr.Button(
                                    "Choose Automatic Visual Creation",
                                    variant="primary",
                                )
                    workflow_entry_status = gr.Markdown()

                new_audio_upload.change(
                    preview_new_project_inputs,
                    inputs=[
                        new_audio_upload,
                        new_script_upload,
                        pasted_script_input,
                    ],
                    outputs=[
                        new_project_input_status,
                        new_audio_preview,
                        new_audio_filename,
                        new_audio_status,
                        new_script_filename,
                        script_review,
                    ],
                    api_name=False,
                )
                new_script_upload.change(
                    preview_new_project_inputs,
                    inputs=[
                        new_audio_upload,
                        new_script_upload,
                        pasted_script_input,
                    ],
                    outputs=[
                        new_project_input_status,
                        new_audio_preview,
                        new_audio_filename,
                        new_audio_status,
                        new_script_filename,
                        script_review,
                    ],
                    api_name=False,
                )
                pasted_script_input.change(
                    preview_new_project_inputs,
                    inputs=[
                        new_audio_upload,
                        new_script_upload,
                        pasted_script_input,
                    ],
                    outputs=[
                        new_project_input_status,
                        new_audio_preview,
                        new_audio_filename,
                        new_audio_status,
                        new_script_filename,
                        script_review,
                    ],
                    api_name=False,
                )

                def reset_project_creation(_value):
                    return (
                        gr_update(interactive=True),
                        gr_update(visible=False),
                        None,
                        "",
                        "",
                    )

                for input_component in (
                    project_name_input,
                    new_audio_upload,
                    new_script_upload,
                    pasted_script_input,
                ):
                    input_component.change(
                        reset_project_creation,
                        inputs=[input_component],
                        outputs=[
                            create_project_button,
                            workflow_choices,
                            created_project_dir,
                            project_create_status,
                            workflow_entry_status,
                        ],
                        api_name=False,
                    )
                create_project_button.click(
                    create_project_callback,
                    inputs=[
                        project_name_input,
                        new_audio_upload,
                        new_script_upload,
                        pasted_script_input,
                    ],
                    outputs=[
                        project_create_status,
                        created_project_dir,
                        workflow_choices,
                        create_project_button,
                    ],
                    api_name=False,
                )
            with gr.Tab("Assisted Visual Creation", id="assisted") as assisted_tab:
                with gr.Group(elem_classes=["narrato-card"]):
                    gr.Markdown("### Shared project setup")
                    gr.Markdown(
                        "Create or reload a project folder containing stable copies of the "
                        "voice-over and script. The audio player provides native playback and "
                        "scrubbing; marker times are manual, with no word-level alignment."
                    )
                    with gr.Row():
                        project_dir = gr.Textbox(
                            label="Project folder",
                            value=str(_PROJECT_ROOT / "projects" / "new_project"),
                            scale=2,
                        )
                        audio_upload = _StudioFile(
                            label="Voice-over audio (.mp3 or .wav)",
                            file_types=[".mp3", ".wav"],
                            type="filepath",
                            interactive=True,
                            scale=1,
                        )
                        script_upload = _StudioFile(
                            label="Matching script (.txt or .md)",
                            file_types=[".txt", ".md"],
                            type="filepath",
                            interactive=True,
                            scale=1,
                        )
                    with gr.Row():
                        save_project_btn = gr.Button("Save assets / create project")
                        load_project_btn = gr.Button("Load saved project")
                    project_status = gr.Markdown()
                    with gr.Row():
                        audio_player = gr.Audio(
                            label="Voice-over playback",
                            type="filepath",
                            interactive=False,
                            show_download_button=True,
                            scale=2,
                        )
                        audio_duration = gr.Markdown(
                            "Audio duration will appear when a project loads."
                        )
                    assisted_timing_note = gr.Markdown(
                        "Segment times are estimates; this project has no word-level "
                        "audio alignment."
                    )
                    script_preview = gr.TextArea(
                        label="Saved matching script",
                        lines=6,
                        interactive=False,
                    )
                    with gr.Row():
                        with gr.Column(scale=1):
                            assisted_segment_selector = gr.Dropdown(
                                choices=[],
                                label="Script segments",
                                interactive=True,
                            )
                            with gr.Row():
                                previous_segment_btn = gr.Button("Previous segment")
                                next_segment_btn = gr.Button("Next segment")
                            assisted_segments_table = gr.Dataframe(
                                headers=["Segment", "Title", "Time", "Status"],
                                datatype=["number", "str", "str", "str"],
                                interactive=False,
                                wrap=True,
                                label="Segment progress",
                            )
                            assisted_segment_text = gr.TextArea(
                                label="Selected script segment",
                                lines=6,
                                interactive=False,
                            )
                            assisted_segment_status = gr.Markdown(
                                "Load a project to browse its script segments."
                            )
                            marker_selector = gr.Dropdown(
                                choices=[],
                                label="Manual markers",
                                interactive=True,
                            )
                            marker_table = gr.Dataframe(
                                headers=[
                                    "Marker ID", "Start (s)", "End (s)", "Image prompt",
                                    "Generated image path",
                                ],
                                datatype=["str", "number", "str", "str", "str"],
                                interactive=False,
                                wrap=True,
                            )
                        with gr.Column(scale=1):
                            marker_start = gr.Number(
                                label="Segment start time (seconds)",
                                minimum=0,
                                value=None,
                            )
                            marker_end = gr.Number(
                                label="Segment end time (seconds, optional)",
                                minimum=0,
                                value=None,
                            )
                            marker_prompt = gr.TextArea(
                                label="Image prompt · saved to this segment",
                                lines=4,
                            )
                            assisted_image_preview = gr.Image(
                                label="Selected segment image",
                                type="filepath",
                                interactive=False,
                            )
                            with gr.Row():
                                add_marker_btn = gr.Button("Add marker")
                                save_marker_btn = gr.Button(
                                    "Save prompt & segment timing",
                                )
                                delete_marker_btn = gr.Button("Delete marker", variant="stop")
                            marker_status = gr.Markdown()
                            marker_regeneration_confirm = gr.Checkbox(
                                label="Confirm replacing an existing segment image",
                                value=False,
                            )
                            marker_generate_btn = gr.Button(
                                "Generate Image",
                                variant="primary",
                            )
                            assisted_retry_btn = gr.Button(
                                "Retry / regenerate this segment",
                            )
                            marker_image_preview = gr.Image(
                                label="Manual marker image preview",
                                type="filepath",
                                interactive=False,
                            )
                            marker_generation_status_text = gr.Markdown(
                                "Image generation is idle."
                            )
                            marker_generation_progress = gr.Slider(
                                minimum=0,
                                maximum=100,
                                step=1,
                                value=0,
                                label="Selected marker generation progress",
                                interactive=False,
                            )

                    project_outputs = [
                        project_status,
                        audio_player,
                        audio_duration,
                        script_preview,
                        marker_table,
                        marker_selector,
                        marker_start,
                        marker_end,
                        marker_prompt,
                        marker_image_preview,
                    ]
                    save_project_btn.click(
                        save_project_callback,
                        inputs=[project_dir, audio_upload, script_upload],
                        outputs=project_outputs,
                        api_name=False,
                    )
                    load_project_btn.click(
                        load_project_callback,
                        inputs=[project_dir],
                        outputs=project_outputs,
                        api_name=False,
                    ).then(
                        load_assisted_workspace_callback,
                        inputs=[project_dir],
                        outputs=[
                            assisted_segments_table,
                            assisted_segment_selector,
                            marker_selector,
                            assisted_segment_text,
                            marker_start,
                            marker_end,
                            marker_prompt,
                            assisted_image_preview,
                            assisted_segment_status,
                            assisted_timing_note,
                        ],
                        api_name=False,
                    )
                    save_project_btn.click(
                        load_assisted_workspace_callback,
                        inputs=[project_dir],
                        outputs=[
                            assisted_segments_table,
                            assisted_segment_selector,
                            marker_selector,
                            assisted_segment_text,
                            marker_start,
                            marker_end,
                            marker_prompt,
                            assisted_image_preview,
                            assisted_segment_status,
                            assisted_timing_note,
                        ],
                        api_name=False,
                    )
                    marker_selector.input(
                        select_assisted_marker_ui,
                        inputs=[project_dir, marker_selector],
                        outputs=[
                            assisted_segment_selector,
                            assisted_segment_text,
                            marker_start,
                            marker_end,
                            marker_prompt,
                            assisted_image_preview,
                            assisted_segment_status,
                            marker_image_preview,
                        ],
                        api_name=False,
                    )
                    assisted_segment_selector.input(
                        select_assisted_segment_ui,
                        inputs=[project_dir, assisted_segment_selector],
                        outputs=[
                            marker_selector,
                            assisted_segment_text,
                            marker_start,
                            marker_end,
                            marker_prompt,
                            assisted_image_preview,
                            assisted_segment_status,
                            marker_image_preview,
                        ],
                        api_name=False,
                    )
                    for navigation_button, step in (
                        (previous_segment_btn, -1),
                        (next_segment_btn, 1),
                    ):
                        def navigation_callback(direction):
                            return lambda project_path, selected_id: (
                                navigate_assisted_segment(
                                    project_path, selected_id, direction
                                )
                            )

                        navigation_button.click(
                            navigation_callback(step),
                            inputs=[project_dir, assisted_segment_selector],
                            outputs=[
                                assisted_segment_selector,
                                marker_selector,
                                assisted_segment_text,
                                marker_start,
                                marker_end,
                                marker_prompt,
                                assisted_image_preview,
                                assisted_segment_status,
                            ],
                            api_name=False,
                        )
                    marker_outputs = [
                        marker_status,
                        marker_table,
                        marker_selector,
                        marker_start,
                        marker_end,
                        marker_prompt,
                        marker_image_preview,
                    ]
                    add_marker_btn.click(
                        add_marker_callback,
                        inputs=[project_dir, marker_start, marker_end, marker_prompt],
                        outputs=marker_outputs,
                        api_name=False,
                    )
                    save_marker_btn.click(
                        save_marker_callback,
                        inputs=[
                            project_dir, marker_selector, marker_start, marker_end,
                            marker_prompt,
                        ],
                        outputs=marker_outputs,
                        api_name=False,
                    ).then(
                        refresh_assisted_segment_status,
                        inputs=[project_dir, marker_selector],
                        outputs=[
                            assisted_segments_table,
                            assisted_segment_status,
                            marker_generation_progress,
                            assisted_image_preview,
                            marker_generate_btn,
                            assisted_retry_btn,
                        ],
                        api_name=False,
                    )
                    for edit_component in (
                        marker_start,
                        marker_end,
                        marker_prompt,
                    ):
                        edit_component.input(
                            autosave_assisted_segment_callback,
                            inputs=[
                                project_dir,
                                assisted_segment_selector,
                                marker_start,
                                marker_end,
                                marker_prompt,
                            ],
                            outputs=[
                                marker_status,
                                assisted_segments_table,
                                assisted_image_preview,
                                assisted_segment_status,
                            ],
                            api_name=False,
                        )
                    delete_marker_btn.click(
                        delete_marker_callback,
                        inputs=[project_dir, marker_selector],
                        outputs=marker_outputs,
                        api_name=False,
                    )
                    assisted_tab.select(
                        load_assisted_workspace_callback,
                        inputs=[project_dir],
                        outputs=[
                            assisted_segments_table,
                            assisted_segment_selector,
                            marker_selector,
                            assisted_segment_text,
                            marker_start,
                            marker_end,
                            marker_prompt,
                            assisted_image_preview,
                            assisted_segment_status,
                            assisted_timing_note,
                        ],
                        api_name=False,
                    )
            with gr.Tab("Automatic Visual Creation", id="automatic") as automatic_tab:
                with gr.Group(elem_classes=["narrato-card"]):
                    gr.Markdown("### Project selected for this workflow")
                    automatic_project_dir = gr.Textbox(
                        label="Project folder",
                        interactive=False,
                    )
                    with gr.Row(equal_height=True):
                        automatic_audio_preview = gr.Audio(
                            label="Project narration",
                            type="filepath",
                            interactive=False,
                            show_download_button=False,
                        )
                        automatic_script_preview = gr.TextArea(
                            label="Project script",
                            lines=5,
                            interactive=False,
                        )
                    automatic_timing_note = gr.Markdown(
                        "Scene timings are estimates, not speech alignment."
                    )
                    automatic_plan_status = gr.Markdown(
                        "Choose Automatic Visual Creation from a project to plan its scenes."
                    )
                    automatic_scene_table = gr.Dataframe(
                        headers=[
                            "Scene",
                            "Title",
                            "Script text",
                            "Estimated time",
                            "Image prompt",
                            "Status",
                            "Image",
                            "Error",
                        ],
                        datatype=[
                            "number", "str", "str", "str",
                            "str", "str", "str", "str",
                        ],
                        interactive=False,
                        wrap=True,
                        label="Automatic scene plan · review before generation",
                    )
                    automatic_project_gallery = _StudioGallery(
                        label="Saved images by scene",
                        columns=3,
                        height=300,
                        object_fit="contain",
                        preview=True,
                    )
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=330, elem_classes=["narrato-card"]):
                        gr.Markdown("### Project")
                        plan_path = gr.Textbox(
                            label="Project scene plan", value=settings.plan_path,
                            interactive=False,
                        )
                        output_dir = gr.Textbox(
                            label="Project image folder", value=settings.output_dir,
                            interactive=False,
                        )
                        manifest_path = gr.Textbox(
                            label="Project manifest", value=settings.manifest_path,
                            interactive=False,
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
                            retries = gr.Slider(
                                minimum=0, maximum=10, step=1,
                                value=settings.retries, label="Retries",
                            )
                            gr.Markdown("Scenes are processed sequentially.")
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
                            start_btn = gr.Button(
                                "Start Automatic Generation", variant="primary"
                            )
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

        def open_new_project_mode(project_dir_value, mode):
            try:
                project, tab_index = route_project_to_mode(project_dir_value, mode)
                project_state = _project_ui_state(
                    project,
                    f"**Project ready for {mode}:** `{project['project_dir']}`",
                )
                return (
                    f"Opening **{mode}** with the selected project.",
                    gr.update(selected=tab_index),
                    str(project["project_dir"]),
                    *project_state,
                    str(project["project_dir"]),
                    str(project["audio_path"]),
                    project["script_path"].read_text(encoding="utf-8"),
                )
            except (FileNotFoundError, OSError, ValueError) as exc:
                logger.warning(
                    "Project workflow selection failed: %s", _safe_diagnostic(str(exc))
                )
                empty_updates = tuple(gr.update() for _ in range(14))
                return (
                    f"**Could not open workflow:** {exc}",
                    gr.update(),
                    *empty_updates,
                )

        mode_outputs = [
            workflow_entry_status,
            tabs,
            project_dir,
            *project_outputs,
            automatic_project_dir,
            automatic_audio_preview,
            automatic_script_preview,
        ]
        new_project_nav_button.click(
            lambda: gr.update(selected="new-project"),
            outputs=[tabs],
            api_name=False,
        )
        assisted_mode_button.click(
            lambda project_path: open_new_project_mode(
                project_path, "Assisted Visual Creation"
            ),
            inputs=[created_project_dir],
            outputs=mode_outputs,
            api_name=False,
        ).then(
            load_assisted_workspace_callback,
            inputs=[project_dir],
            outputs=[
                assisted_segments_table,
                assisted_segment_selector,
                marker_selector,
                assisted_segment_text,
                marker_start,
                marker_end,
                marker_prompt,
                assisted_image_preview,
                assisted_segment_status,
                assisted_timing_note,
            ],
            api_name=False,
        )
        automatic_mode_button.click(
            lambda project_path: open_new_project_mode(
                project_path, "Automatic Visual Creation"
            ),
            inputs=[created_project_dir],
            outputs=mode_outputs,
            api_name=False,
        ).then(
            load_automatic_project_callback,
            inputs=[automatic_project_dir],
            outputs=[
                plan_path,
                output_dir,
                manifest_path,
                automatic_plan_status,
                automatic_timing_note,
                automatic_scene_table,
                automatic_project_gallery,
                selected_scenes,
                failed_scene,
            ],
            api_name=False,
        ).then(
            refresh_dashboard,
            inputs=[plan_path, output_dir, manifest_path],
            outputs=[
                plan_info, counts, progress, scene_table, gallery,
                selected_scenes, failed_scene, manifest_file, csv_file, paths,
            ],
            api_name=False,
        )
        automatic_tab.select(
            load_automatic_project_callback,
            inputs=[automatic_project_dir],
            outputs=[
                plan_path,
                output_dir,
                manifest_path,
                automatic_plan_status,
                automatic_timing_note,
                automatic_scene_table,
                automatic_project_gallery,
                selected_scenes,
                failed_scene,
            ],
            api_name=False,
        )

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
        ).then(
            refresh_automatic_scene_table,
            inputs=[automatic_project_dir],
            outputs=[
                automatic_scene_table,
                automatic_plan_status,
                automatic_project_gallery,
            ],
            api_name=False,
        )
        ready_btn.click(
            check_backend_readiness,
            inputs=[backend, comfyui_url, workflow_path],
            outputs=[readiness],
        )
        assisted_generation_inputs = [
            project_dir,
            marker_selector,
            marker_start,
            marker_end,
            marker_prompt,
            comfyui_url,
            workflow_path,
            positive_quality_prompt,
            negative_quality_prompt,
            marker_regeneration_confirm,
            retries,
            aspect_ratio,
        ]
        assisted_generation_outputs = [
            marker_generation_status_text,
            marker_regeneration_confirm,
            assisted_image_preview,
            marker_generate_btn,
            assisted_retry_btn,
        ]
        marker_generate_btn.click(
            generate_assisted_segment_callback,
            inputs=assisted_generation_inputs,
            outputs=assisted_generation_outputs,
            api_name=False,
        )
        assisted_retry_btn.click(
            lambda *values: generate_assisted_segment_callback(*values, retry=True),
            inputs=assisted_generation_inputs,
            outputs=assisted_generation_outputs,
            api_name=False,
        )
        automatic_start_inputs = [
            automatic_project_dir,
            backend,
            aspect_ratio,
            selection_mode,
            selected_scenes,
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
        start_btn.click(
            automatic_generation_callback,
            inputs=automatic_start_inputs,
            outputs=[action_status],
        ).then(
            refresh_dashboard,
            inputs=[plan_path, output_dir, manifest_path],
            outputs=[
                plan_info, counts, progress, scene_table, gallery,
                selected_scenes, failed_scene, manifest_file, csv_file, paths,
            ],
            api_name=False,
        ).then(
            refresh_automatic_scene_table,
            inputs=[automatic_project_dir],
            outputs=[
                automatic_scene_table,
                automatic_plan_status,
                automatic_project_gallery,
            ],
            api_name=False,
        )
        cancel_btn.click(
            cancel_generation, outputs=[action_status]
        )

        retry_btn.click(
            retry_automatic_scene_callback,
            inputs=[
                automatic_project_dir,
                backend,
                aspect_ratio,
                delay,
                retries,
                confirm_api,
                comfyui_url,
                workflow_path,
                positive_quality_prompt,
                negative_quality_prompt,
                visual_theme,
                failed_scene,
            ],
            outputs=[action_status],
            api_name=False,
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
        demo.load(
            marker_generation_status,
            outputs=[
                marker_generation_status_text,
                marker_generation_progress,
                marker_image_preview,
            ],
            every=1,
            api_name=False,
        )
        demo.load(
            refresh_assisted_segment_status,
            inputs=[project_dir, marker_selector],
            outputs=[
                assisted_segments_table,
                assisted_segment_status,
                marker_generation_progress,
                assisted_image_preview,
                marker_generate_btn,
                assisted_retry_btn,
            ],
            every=1,
            api_name=False,
        )
        demo.load(
            refresh_automatic_scene_table,
            inputs=[automatic_project_dir],
            outputs=[
                automatic_scene_table,
                automatic_plan_status,
                automatic_project_gallery,
            ],
            every=2,
            api_name=False,
        )

    return demo


def launch(share: bool = False, port: int = 7860) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.setLevel(logging.INFO)
    build_ui().launch(
        share=share,
        server_name="127.0.0.1",
        server_port=port,
        show_error=True,
    )

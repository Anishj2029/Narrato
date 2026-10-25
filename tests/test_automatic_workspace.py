from __future__ import annotations

import json
import threading
import wave
from pathlib import Path

import pytest
from PIL import Image

from ai_visualizer import studio
from ai_visualizer.image_backends import GenerationResult
from ai_visualizer.markers import edit_marker, load_markers, save_markers


def _write_audio(path: Path) -> Path:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes(b"\0\0" * 24000)
    return path


@pytest.fixture
def automatic_project(tmp_path):
    audio = _write_audio(tmp_path / "voice.wav")
    script = tmp_path / "script.md"
    script.write_text(
        "## Opening\n"
        "A quiet lighthouse stands beside the northern sea.\n\n"
        "## Harbor\n"
        "A fishing boat returns safely to the harbor at sunrise.",
        encoding="utf-8",
    )
    return studio.save_project_assets(tmp_path / "automatic-project", audio, script)[
        "project_dir"
    ]


@pytest.fixture(autouse=True)
def reset_generation_job(monkeypatch):
    monkeypatch.setattr(studio, "save_settings", lambda settings: None)
    with studio._JOB_LOCK:
        studio._JOB.update(
            {
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
        )
    yield
    thread = studio._JOB.get("thread")
    if thread is not None and thread.is_alive():
        studio.cancel_generation()
        thread.join(timeout=3)


def _fake_backend(monkeypatch, outcomes=None, started=None, release=None):
    calls = []
    remaining_outcomes = list(outcomes or [])

    class FakeBackend:
        model_identifier = "test-model"

        def __init__(self, base_url, workflow_path):
            calls.append(("configured", base_url, workflow_path))

        def generate(self, marker_id, prompt, output_path, options):
            calls.append(("generate", marker_id, prompt, output_path, options))
            if started is not None:
                started.set()
            if release is not None:
                release.wait(timeout=5)
            outcome = remaining_outcomes.pop(0) if remaining_outcomes else "success"
            if outcome != "success":
                return GenerationResult(None, "local", "test-model", False, outcome)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), color=(90, 100, 70)).save(output_path)
            return GenerationResult(output_path, "local", "test-model", True)

    monkeypatch.setattr(studio, "ComfyUIBackend", FakeBackend)
    return calls


def _start(project_dir, **overrides):
    values = {
        "project_dir": str(project_dir),
        "backend": "local",
        "aspect_ratio": "Workflow dimensions",
        "selection_mode": "All scenes",
        "selected_scenes": ["1", "2"],
        "delay_seconds": 0,
        "retries": 0,
        "confirm_api": False,
        "replace_existing": False,
        "comfyui_url": "http://127.0.0.1:8188",
        "workflow_path": "workflow.json",
        "positive_quality_prompt": "Quality",
        "negative_quality_prompt": "Negative",
        "theme": "Default",
    }
    values.update(overrides)
    return studio.automatic_generation_callback(**values)


def _wait_for_job():
    thread = studio._JOB.get("thread")
    assert thread is not None
    thread.join(timeout=5)
    assert not thread.is_alive()


def test_automatic_plan_uses_project_script_and_estimated_audio_timing(
    automatic_project,
):
    project = studio.load_automatic_project(automatic_project)
    scenes = project["automatic_scenes"]

    assert len(scenes) == 2
    assert scenes[0]["text"].startswith("Opening")
    assert scenes[0]["start_time"] == pytest.approx(0)
    assert scenes[-1]["end_time"] == pytest.approx(3)
    assert all(scene["image_prompt"].startswith("[FALLBACK]") for scene in scenes)
    assert "proportional estimates" in project["timing_note"]
    assert project["automatic_paths"]["plan_path"].is_file()
    assert not project["automatic_paths"]["manifest_path"].exists()
    assert not project["automatic_paths"]["output_dir"].exists()


def test_existing_plan_and_markers_are_reused_without_overwriting(
    automatic_project,
):
    original = studio.load_automatic_project(automatic_project)
    plan_path = original["automatic_paths"]["plan_path"]
    plan_bytes = plan_path.read_bytes()
    first_marker_id = original["automatic_marker_ids"][1]
    markers_path = automatic_project / "markers.json"
    markers = load_markers(markers_path)
    edit_marker(markers, first_marker_id, image_prompt="Keep this custom prompt")
    save_markers(markers_path, markers)

    reopened = studio.load_automatic_project(automatic_project)

    assert plan_path.read_bytes() == plan_bytes
    assert reopened["automatic_marker_ids"][1] == first_marker_id
    assert next(
        marker for marker in reopened["markers"] if marker.marker_id == first_marker_id
    ).image_prompt == "Keep this custom prompt"
    assert len(reopened["markers"]) == 2


def test_loading_automatic_project_never_starts_image_generation(
    automatic_project, monkeypatch,
):
    calls = _fake_backend(monkeypatch)

    loaded = studio.load_automatic_project_callback(automatic_project)

    assert len(loaded[5]) == 2
    assert all(row[5] == "Planned" for row in loaded[5])
    assert studio._JOB["status"] == "idle"
    assert not [call for call in calls if call[0] == "generate"]


def test_generation_requires_explicit_start_and_persists_each_image(
    automatic_project, monkeypatch,
):
    calls = _fake_backend(monkeypatch)
    studio.load_automatic_project(automatic_project)
    assert studio._JOB["status"] == "idle"
    assert not [call for call in calls if call[0] == "generate"]

    response = _start(automatic_project)
    _wait_for_job()
    reopened = studio.load_automatic_project(automatic_project)
    markers = {marker.marker_id: marker for marker in load_markers(
        automatic_project / "markers.json"
    )}

    assert response.startswith("Started **local** generation")
    assert len([call for call in calls if call[0] == "generate"]) == 2
    for scene_index, marker_id in reopened["automatic_marker_ids"].items():
        marker = markers[marker_id]
        assert marker.image_path == (
            f"automatic/images/scene_{scene_index:03d}.png"
        )
        assert (automatic_project / marker.image_path).is_file()
    assert all(row[5] == "Completed" for row in studio._automatic_scene_rows(
        automatic_project
    ))


def test_generation_uses_persisted_marker_prompt(automatic_project, monkeypatch):
    project = studio.load_automatic_project(automatic_project)
    marker_id = project["automatic_marker_ids"][1]
    markers_path = automatic_project / "markers.json"
    markers = load_markers(markers_path)
    edit_marker(markers, marker_id, image_prompt="Custom marker prompt")
    save_markers(markers_path, markers)
    calls = _fake_backend(monkeypatch)

    _start(automatic_project, selected_scenes=["1"])
    _wait_for_job()

    generation_call = next(call for call in calls if call[0] == "generate")
    assert "Custom marker prompt" in generation_call[2]
    assert not list((automatic_project / "automatic").glob(".generation-plan.*"))


def test_repeated_start_is_rejected_while_automatic_job_is_running(
    automatic_project, monkeypatch,
):
    started = threading.Event()
    release = threading.Event()
    calls = _fake_backend(monkeypatch, started=started, release=release)

    first_response = _start(automatic_project)
    assert first_response.startswith("Started")
    assert started.wait(timeout=3)
    second_response = _start(automatic_project)
    release.set()
    _wait_for_job()

    assert "already running" in second_response
    assert len([call for call in calls if call[0] == "generate"]) == 2


def test_scene_progress_and_backend_errors_are_reported(automatic_project, monkeypatch):
    started = threading.Event()
    release = threading.Event()
    _fake_backend(
        monkeypatch,
        outcomes=["ComfyUI is unavailable"],
        started=started,
        release=release,
    )

    _start(automatic_project, selected_scenes=["1"])
    assert started.wait(timeout=3)
    in_progress = studio._automatic_scene_rows(automatic_project)
    release.set()
    _wait_for_job()
    final_rows = studio._automatic_scene_rows(automatic_project)

    assert in_progress[0][5] == "Generating"
    assert final_rows[0][5] == "Failed"
    assert "ComfyUI is unavailable" in final_rows[0][7]


def test_retry_targets_only_failed_scene_and_keeps_completed_image(
    automatic_project, monkeypatch,
):
    calls = _fake_backend(
        monkeypatch,
        outcomes=["success", "Temporary ComfyUI failure", "success"],
    )

    _start(automatic_project)
    _wait_for_job()
    plan = studio.load_automatic_project(automatic_project)
    first_marker = next(
        marker for marker in load_markers(automatic_project / "markers.json")
        if marker.marker_id == plan["automatic_marker_ids"][1]
    )
    first_image = first_marker.image_path

    retry_message = studio.retry_automatic_scene_callback(
        str(automatic_project),
        "local",
        "Workflow dimensions",
        0,
        0,
        False,
        "http://127.0.0.1:8188",
        "workflow.json",
        "Quality",
        "Negative",
        "Default",
        "2",
    )
    _wait_for_job()
    generated_scene_ids = [
        call[1] for call in calls if call[0] == "generate"
    ]
    rows = studio._automatic_scene_rows(automatic_project)

    assert retry_message.startswith("Started")
    assert generated_scene_ids == ["1", "2", "2"]
    assert rows[0][5] == "Completed"
    assert rows[1][5] == "Completed"
    assert first_image == "automatic/images/scene_001.png"
    no_op_message = _start(automatic_project)
    assert "nothing was regenerated" in no_op_message
    assert len(generated_scene_ids) == 3
    assert len([call for call in calls if call[0] == "generate"]) == 3


def test_existing_marker_images_are_shown_and_not_regenerated(
    automatic_project, monkeypatch,
):
    project = studio.load_automatic_project(automatic_project)
    marker_id = project["automatic_marker_ids"][1]
    image_path = automatic_project / "images" / "preexisting.png"
    image_path.parent.mkdir()
    Image.new("RGB", (8, 8)).save(image_path)
    markers = load_markers(automatic_project / "markers.json")
    edit_marker(markers, marker_id, image_path="images/preexisting.png")
    save_markers(automatic_project / "markers.json", markers)
    calls = _fake_backend(monkeypatch)

    rows = studio._automatic_scene_rows(automatic_project)
    gallery = studio._automatic_scene_gallery(automatic_project)
    response = _start(automatic_project)
    _wait_for_job()

    assert rows[0][5] == "Completed"
    assert rows[0][6] == str(image_path)
    assert gallery[0][0] == str(image_path)
    assert gallery[0][1].startswith("Scene 1: Opening")
    assert response.startswith("Started")
    assert [call[1] for call in calls if call[0] == "generate"] == ["2"]


def test_comfyui_configuration_failure_is_useful_and_preserves_plan(
    automatic_project, monkeypatch,
):
    studio.load_automatic_project(automatic_project)

    class MissingWorkflowBackend:
        def __init__(self, base_url, workflow_path):
            raise studio.BackendConfigurationError("Workflow JSON was not found.")

    monkeypatch.setattr(studio, "ComfyUIBackend", MissingWorkflowBackend)

    response = _start(automatic_project)

    assert "Workflow JSON was not found" in response
    assert studio._JOB["status"] == "idle"
    assert studio._automatic_project_paths(automatic_project)["plan_path"].is_file()


def test_duplicate_generation_does_not_replace_existing_plan(
    automatic_project,
):
    plan = studio.load_automatic_project(automatic_project)["automatic_paths"][
        "plan_path"
    ]
    contents = json.loads(plan.read_text(encoding="utf-8"))

    with pytest.raises(FileExistsError, match="not replaced"):
        studio._write_automatic_plan(plan, contents)

    assert json.loads(plan.read_text(encoding="utf-8")) == contents

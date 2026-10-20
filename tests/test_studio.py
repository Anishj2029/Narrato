from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
from PIL import Image

from ai_visualizer.image_runner import SceneResult, save_manifest
from ai_visualizer import studio


def _scene(index: int = 1):
    return {
        "index": index,
        "title": f"Scene {index}",
        "text": "A scene description.",
        "start_time": (index - 1) * 10.0,
        "end_time": index * 10.0,
        "image_prompt": "A cinematic scene.",
        "image_path": "",
        "description": "A scene.",
        "filename_suggestion": f"scene_{index:03d}.png",
        "segment_ids": [index],
        "word_count": 3,
    }


def _write_plan(path: Path, scenes=None):
    path.write_text(json.dumps(scenes or [_scene()]), encoding="utf-8")


def _reset_job():
    with studio._JOB_LOCK:
        studio._JOB.update({
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
        })


@pytest.fixture(autouse=True)
def clean_job():
    _reset_job()
    yield
    worker = studio._JOB.get("thread")
    if worker and worker.is_alive():
        studio.cancel_generation()
        worker.join(timeout=2)
    _reset_job()


def test_settings_persist_non_secret_fields_only(tmp_path):
    path = tmp_path / "studio.json"
    settings = studio.StudioSettings(
        backend="local",
        comfyui_url="http://127.0.0.1:8188",
        workflow_path="workflow.json",
        selected_scenes=("1", "3"),
    )

    studio.save_settings(settings, path)
    loaded = studio.load_settings(path)
    stored = json.loads(path.read_text(encoding="utf-8"))

    assert loaded == settings
    assert "api_key" not in json.dumps(stored).lower()
    assert stored["selected_scenes"] == ["1", "3"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"backend": "mock"}, "either the local or API"),
        ({"aspect_ratio": "17:9"}, "Unsupported aspect ratio"),
        ({"concurrency": 0}, "Concurrency"),
        ({"concurrency": 1.5}, "whole number"),
        ({"retries": 11}, "Retries"),
    ],
)
def test_settings_validation_rejects_invalid_values(kwargs, message):
    values = {
        "backend": "api",
        "plan_path": "scene_plan.json",
        "output_dir": "outputs/images",
        "manifest_path": "outputs/manifest.json",
        "aspect_ratio": "16:9",
        "concurrency": 1,
        "delay_seconds": 0,
        "retries": 0,
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        studio.validate_studio_settings(**values)


def test_local_settings_require_workflow_and_reject_url_credentials(tmp_path):
    with pytest.raises(ValueError, match="workflow path"):
        studio.validate_studio_settings(
            "local", "plan.json", "images", "manifest.json", "16:9",
            1, 0, 0, "http://127.0.0.1:8188", "",
        )
    workflow = tmp_path / "workflow.json"
    workflow.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="must not contain credentials"):
        studio.validate_studio_settings(
            "local", "plan.json", "images", "manifest.json", "16:9",
            1, 0, 0, "http://user:password@localhost:8188", str(workflow),
        )


def test_missing_or_invalid_scene_plan_is_reported(tmp_path):
    info, rows, gallery, counts, scene_ids, failed_ids = studio._plan_status(
        str(tmp_path / "missing.json"),
        str(tmp_path / "images"),
        str(tmp_path / "manifest.json"),
    )

    assert "Plan unavailable" in info
    assert rows == []
    assert gallery == []
    assert scene_ids == []
    assert failed_ids == []


def test_studio_builds_with_project_gradio_dependency():
    import gradio as gr

    app = studio.build_ui()

    assert isinstance(app, gr.Blocks)


def test_dashboard_uses_manifest_statuses_and_handles_invalid_preview(tmp_path):
    plan = tmp_path / "scene_plan.json"
    scenes = [_scene(1), _scene(2), _scene(3)]
    _write_plan(plan, scenes)
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    valid_image = image_dir / "scene_001.jpg"
    Image.new("RGB", (2, 2), (20, 30, 40)).save(valid_image, format="JPEG")
    placeholder_image = image_dir / "scene_002.png"
    Image.new("RGB", (2, 2), (40, 30, 20)).save(placeholder_image, format="PNG")
    invalid_image = image_dir / "invalid.png"
    invalid_image.write_bytes(b"not an image")
    manifest = tmp_path / "manifest.json"
    save_manifest([
        SceneResult(
            index=1, title="Scene 1", filename=valid_image.name,
            image_path=str(valid_image), prompt_used="p", status="ok",
            backend="api", model="test-model",
        ),
        SceneResult(
            index=2, title="Scene 2", filename=placeholder_image.name,
            image_path=str(placeholder_image), prompt_used="p", status="ok",
            backend="mock", is_placeholder=True,
        ),
        SceneResult(
            index=3, title="Scene 3", filename="scene_003.png",
            image_path="", prompt_used="p", status="error",
            error="server unavailable", backend="local",
        ),
    ], manifest)

    info, rows, gallery, counts, scene_ids, failed_ids = studio._plan_status(
        str(plan), str(image_dir), str(manifest)
    )

    assert "3 scenes" in info
    assert len(rows) == 3
    assert rows[1][3] == "placeholder"
    assert rows[2][3] == "failed"
    assert len(gallery) == 2
    assert "PLACEHOLDER" in gallery[1][1]
    assert "generated" in counts.lower()
    assert "Pending** 1" in counts
    assert "Failed** 1" in counts
    assert scene_ids == ["1", "2", "3"]
    assert failed_ids == ["3"]
    assert studio._valid_preview(str(valid_image))
    assert not studio._valid_preview(str(invalid_image))
    assert not studio._valid_preview(str(tmp_path / "missing.png"))


def test_dashboard_marks_corrupt_successful_output_as_failed(tmp_path):
    plan = tmp_path / "plan.json"
    _write_plan(plan)
    bad_image = tmp_path / "scene_001.png"
    bad_image.write_bytes(b"corrupt")
    manifest = tmp_path / "manifest.json"
    save_manifest([SceneResult(
        index=1, title="Scene 1", filename=bad_image.name,
        image_path=str(bad_image), prompt_used="p", status="ok",
        backend="local",
    )], manifest)

    _, rows, gallery, counts, _, failed_ids = studio._plan_status(
        str(plan), str(tmp_path), str(manifest)
    )

    assert rows[0][3] == "failed"
    assert "not a valid image" in rows[0][6]
    assert gallery == []
    assert "Failed** 1" in counts
    assert failed_ids == ["1"]


def test_api_readiness_checks_configuration_without_request(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-never-displayed")
    monkeypatch.setenv("GEMINI_IMAGE_MODEL", "test-image-model")
    from ai_visualizer import image_backends

    monkeypatch.setattr(
        image_backends.requests,
        "post",
        lambda *args, **kwargs: pytest.fail("readiness must not call the API"),
    )
    status = studio.check_backend_readiness("api")

    assert "test-image-model" in status
    assert "configured" in status.lower()
    assert "test-key-never-displayed" not in status
    assert "No API request was made" in status


def test_start_generation_runs_in_background_and_reports_status(
    tmp_path, monkeypatch
):
    plan = tmp_path / "scene_plan.json"
    _write_plan(plan)
    workflow = tmp_path / "workflow.json"
    workflow.write_text("{}", encoding="utf-8")

    class LocalBackend:
        model_identifier = "test-comfy-model"

        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(studio, "ComfyUIBackend", LocalBackend)
    monkeypatch.setattr(studio, "save_settings", lambda settings: None)

    def fake_run(config):
        config.progress_callback("running", 1, None, 0, 1)
        result = SceneResult(
            index=1, title="Scene 1", filename="scene_001.png",
            image_path="", prompt_used="p", status="skipped", backend="local",
        )
        config.progress_callback("complete", 1, result, 1, 1)
        return [result]

    monkeypatch.setattr(studio, "run_image_generation", fake_run)
    status = studio.start_generation(
        "local",
        str(plan),
        str(tmp_path / "images"),
        str(tmp_path / "manifest.json"),
        "16:9",
        "All scenes",
        [],
        1,
        0,
        0,
        False,
        False,
        "http://127.0.0.1:8188",
        str(workflow),
    )
    worker = studio._JOB["thread"]
    worker.join(timeout=2)
    snapshot = studio._snapshot_job()

    assert "Started" in status
    assert snapshot["status"] == "completed"
    assert snapshot["completed"] == 1
    assert snapshot["scene_status"][1] == "skipped"
    assert snapshot["results"][0].status == "skipped"


def test_background_job_exception_is_exposed_as_failed_status(tmp_path, monkeypatch):
    plan = tmp_path / "scene_plan.json"
    _write_plan(plan)
    workflow = tmp_path / "workflow.json"
    workflow.write_text("{}", encoding="utf-8")

    class LocalBackend:
        model_identifier = "test-comfy-model"

        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(studio, "ComfyUIBackend", LocalBackend)
    monkeypatch.setattr(studio, "save_settings", lambda settings: None)

    def fail_run(config):
        raise RuntimeError("test runner failure")

    monkeypatch.setattr(studio, "run_image_generation", fail_run)
    studio.start_generation(
        "local",
        str(plan),
        str(tmp_path / "images"),
        str(tmp_path / "manifest.json"),
        "16:9",
        "All scenes",
        [],
        1,
        0,
        0,
        False,
        False,
        "http://127.0.0.1:8188",
        str(workflow),
    )
    worker = studio._JOB["thread"]
    worker.join(timeout=2)
    snapshot = studio._snapshot_job()

    assert snapshot["status"] == "failed"
    assert snapshot["error"] == "test runner failure"


def test_api_batch_requires_confirmation_and_key(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    plan = tmp_path / "plan.json"
    _write_plan(plan)
    with pytest.raises(ValueError, match="Confirm"):
        studio.start_generation(
            "api", str(plan), str(tmp_path / "images"),
            str(tmp_path / "manifest.json"), "16:9", "All scenes", [],
            1, 0, 0, False, False,
        )
    with pytest.raises(studio.BackendConfigurationError, match="GEMINI_API_KEY"):
        studio.start_generation(
            "api", str(plan), str(tmp_path / "images"),
            str(tmp_path / "manifest.json"), "16:9", "All scenes", [],
            1, 0, 0, True, False,
        )


def test_cancel_generation_requests_cooperative_cancel():
    event = threading.Event()
    done = threading.Event()

    def running_job():
        event.wait(timeout=2)
        done.set()

    thread = threading.Thread(target=running_job)
    thread.start()
    with studio._JOB_LOCK:
        studio._JOB["thread"] = thread
        studio._JOB["cancel_event"] = event
        studio._JOB["status"] = "running"

    message = studio.cancel_generation()
    thread.join(timeout=2)

    assert "Cancellation requested" in message
    assert event.is_set()
    assert done.is_set()

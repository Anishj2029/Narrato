from __future__ import annotations

import inspect
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
        positive_quality_prompt="Film look",
        negative_quality_prompt="Blurry",
    )

    studio.save_settings(settings, path)
    loaded = studio.load_settings(path)
    stored = json.loads(path.read_text(encoding="utf-8"))

    assert loaded == settings
    assert "api_key" not in json.dumps(stored).lower()
    assert stored["selected_scenes"] == ["1", "3"]
    assert loaded.positive_quality_prompt == "Film look"
    assert loaded.negative_quality_prompt == "Blurry"


def test_workflow_dimensions_are_a_valid_studio_setting():
    studio.validate_studio_settings(
        "api",
        "plan.json",
        "images",
        "manifest.json",
        "Workflow dimensions",
        1,
        0,
        0,
    )


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
    assert "Scene plan not found" in info
    assert str(tmp_path / "missing.json") in info


def test_missing_relative_scene_plan_reports_project_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    missing_name = "narrato-missing-scene-plan.json"

    with pytest.raises(FileNotFoundError) as exc_info:
        studio._read_plan(missing_name)

    assert str(studio._PROJECT_ROOT / missing_name) in str(exc_info.value)


def test_relative_scene_plan_resolves_from_project_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    scenes = studio._read_plan("scene_plan.json")

    assert len(scenes) == len(json.loads(
        (studio._PROJECT_ROOT / "scene_plan.json").read_text(encoding="utf-8")
    ))


def test_refresh_callback_loads_relative_scene_plan_from_project_root(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    app = studio.build_ui()
    refresh = next(
        callback.fn
        for dependency, callback in zip(app.config["dependencies"], app.fns)
        if dependency["api_name"] == "refresh_dashboard"
    )

    result = refresh("scene_plan.json", "outputs/images", "outputs/manifest.json")

    assert result[0].startswith("**Plan ready**")
    assert len(result[3]) == len(studio._read_plan("scene_plan.json"))
    scenes = studio._read_plan("scene_plan.json")
    assert result[5]["choices"] == [str(scene["index"]) for scene in scenes]
    assert result[6]["choices"] == []


def test_refresh_dashboard_returns_file_paths_accepted_by_gradio(tmp_path):
    import gradio as gr

    plan = tmp_path / "scene_plan.json"
    _write_plan(plan)
    csv_path = plan.with_suffix(".csv")
    csv_path.write_text("index,title\n1,Scene 1\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]", encoding="utf-8")
    output_dir = tmp_path / "images"

    outputs = studio.refresh_dashboard(str(plan), str(output_dir), str(manifest))

    assert len(outputs) == 10
    assert outputs[0].startswith("**Plan ready**")
    assert outputs[7] == str(manifest)
    assert outputs[8] == str(csv_path)
    manifest_output = gr.File().postprocess(outputs[7])
    csv_output = gr.File().postprocess(outputs[8])
    assert manifest_output.path == str(manifest)
    assert csv_output.path == str(csv_path)


def test_studio_builds_with_project_gradio_dependency():
    import gradio as gr

    app = studio.build_ui()

    assert isinstance(app, gr.Blocks)


def test_studio_api_info_excludes_filedata_schemas_without_hiding_callbacks():
    app = studio.build_ui()
    config = app.get_config_file()
    file_components = [
        component
        for component in config["components"]
        if component["type"] in {"gallery", "file"}
    ]
    assert len(file_components) == 3
    assert all(component["skip_api"] for component in file_components)

    api_info = app.get_api_info()

    assert "/refresh_dashboard" in api_info["named_endpoints"]
    assert "/refresh_dashboard_1" in api_info["named_endpoints"]
    assert "/continuous_coro" in api_info["named_endpoints"]
    assert len(
        api_info["named_endpoints"]["/refresh_dashboard"]["parameters"]
    ) == 3
    assert len(
        api_info["named_endpoints"]["/refresh_dashboard"]["returns"]
    ) == 7
    assert len(api_info["named_endpoints"]["/start_handler"]["parameters"]) == 16
    assert len(api_info["named_endpoints"]["/retry_one"]["parameters"]) == 17
    assert len(api_info["named_endpoints"]["/continuous_coro"]["parameters"]) == 3
    assert all(
        dep["show_api"]
        for dep in config["dependencies"]
    )


def test_studio_event_callback_arities_match_registered_dependencies():
    app = studio.build_ui()
    expected_counts = {
        "backend_visibility": (1, 3),
        "lambda": (1, 1),
        "refresh_dashboard": (3, 10),
        "check_backend_readiness": (3, 1),
        "start_handler": (16, 1),
        "refresh_dashboard_1": (3, 10),
        "cancel_generation": (0, 1),
        "retry_one": (17, 1),
        "continuous_coro": (3, 10),
    }

    assert len(app.config["dependencies"]) == len(expected_counts)
    for dependency, callback in zip(app.config["dependencies"], app.fns):
        name = dependency["api_name"]
        input_count, output_count = expected_counts[name]
        assert len(dependency["inputs"]) == input_count
        assert len(dependency["outputs"]) == output_count
        assert len(callback.inputs) == input_count
        assert len(callback.outputs) == output_count

        parameters = inspect.signature(callback.fn).parameters.values()
        positional_count = sum(
            parameter.kind
            in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
            for parameter in parameters
        )
        accepts_varargs = any(
            parameter.kind == parameter.VAR_POSITIONAL
            for parameter in parameters
        )
        assert accepts_varargs or positional_count == input_count


def test_start_and_retry_callbacks_forward_all_registered_inputs(monkeypatch):
    app = studio.build_ui()
    callbacks = {
        dependency["api_name"]: callback.fn
        for dependency, callback in zip(app.config["dependencies"], app.fns)
    }
    calls = []

    def record_start_generation(*args):
        calls.append(args)
        return "Started"

    monkeypatch.setattr(studio, "start_generation", record_start_generation)
    start_values = [
        "local",
        "scene_plan.json",
        "outputs/images",
        "outputs/manifest.json",
        "Workflow dimensions",
        "All scenes",
        [],
        2,
        0.5,
        1,
        False,
        False,
        "http://127.0.0.1:8188",
        "workflow.json",
        "positive prompt",
        "negative prompt",
    ]

    assert callbacks["start_handler"](*start_values) == "Started"
    assert calls[-1] == tuple(start_values)

    assert callbacks["retry_one"](*start_values, "7") == "Started"
    retry_values = start_values.copy()
    retry_values[5] = "Selected scenes"
    retry_values[6] = ["7"]
    assert calls[-1] == tuple(retry_values)


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


def test_local_readiness_reports_server_unavailable(tmp_path, monkeypatch):
    workflow = tmp_path / "workflow.json"
    workflow.write_text(
        '{"1": {"class_type": "SaveImage", "inputs": {}}}',
        encoding="utf-8",
    )
    from ai_visualizer import image_backends

    monkeypatch.setattr(
        image_backends.requests,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            image_backends.requests.ConnectionError("connection refused")
        ),
    )
    status = studio.check_backend_readiness(
        "local", "http://127.0.0.1:8188", str(workflow)
    )

    assert "Not ready" in status
    assert "ComfyUI request failed during preflight" in status


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

    received_configs = []

    def fake_run(config):
        received_configs.append(config)
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
        "Film look",
        "No artifacts",
    )
    worker = studio._JOB["thread"]
    worker.join(timeout=2)
    snapshot = studio._snapshot_job()

    assert "Started" in status
    assert snapshot["status"] == "completed"
    assert snapshot["completed"] == 1
    assert snapshot["scene_status"][1] == "skipped"
    assert snapshot["results"][0].status == "skipped"
    assert received_configs[0].positive_quality_prompt == "Film look"
    assert received_configs[0].negative_quality_prompt == "No artifacts"


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

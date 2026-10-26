from __future__ import annotations

import inspect
import json
import threading
import wave
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

from ai_visualizer.image_backends import GenerationResult
from ai_visualizer.image_prompts import (
    CINEMATIC_CONCEPT_ART_NEGATIVE,
    CINEMATIC_CONCEPT_ART_POSITIVE,
    DEFAULT_THEME,
    compose_prompt,
)
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
    with studio._MARKER_JOB_LOCK:
        studio._MARKER_JOB.update({
            "status": "idle",
            "marker_id": None,
            "project_dir": None,
            "progress": 0,
            "message": "",
            "image_path": None,
            "error": "",
            "thread": None,
        })


def _write_test_audio(path: Path):
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes(b"\0\0" * 8000)


def _create_marker_project(tmp_path: Path, image_prompt: str = "Selected visual"):
    audio = tmp_path / "voice.wav"
    script = tmp_path / "script.md"
    _write_test_audio(audio)
    script.write_text("The matching script.", encoding="utf-8")
    project_dir = tmp_path / "project"
    studio.save_project_assets(project_dir, audio, script)
    selected = studio.add_project_marker(
        project_dir, 2.25, 6.5, image_prompt
    )
    other = studio.add_project_marker(
        project_dir, 8.0, 9.0, "Do not generate this marker"
    )
    return project_dir, selected, other


def _fake_marker_backend(monkeypatch, outcomes=None):
    outcomes = list(outcomes or ["success"])
    calls = []

    class FakeBackend:
        def __init__(self, base_url, workflow_path):
            calls.append(("configured", base_url, workflow_path))
            self.model_identifier = "test-model"

        def preflight(self):
            calls.append(("preflight",))

        def generate(self, marker_id, prompt, output_path, options):
            calls.append(("generate", marker_id, prompt, output_path, options))
            outcome = outcomes.pop(0) if outcomes else "success"
            if isinstance(outcome, Exception):
                raise outcome
            if outcome != "success":
                return GenerationResult(
                    None, "local", "test-model", False, str(outcome)
                )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), color=(30, 60, 90)).save(output_path)
            return GenerationResult(output_path, "local", "test-model", True)

    monkeypatch.setattr(studio, "ComfyUIBackend", FakeBackend)
    return calls


@pytest.fixture(autouse=True)
def clean_job():
    _reset_job()
    yield
    worker = studio._JOB.get("thread")
    if worker and worker.is_alive():
        studio.cancel_generation()
        worker.join(timeout=2)
    marker_worker = studio._MARKER_JOB.get("thread")
    if marker_worker and marker_worker.is_alive():
        marker_worker.join(timeout=2)
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


def test_settings_without_theme_load_with_default_theme(tmp_path):
    path = tmp_path / "older-studio.json"
    path.write_text(
        json.dumps({"backend": "api", "positive_quality_prompt": "Film look"}),
        encoding="utf-8",
    )

    settings = studio.load_settings(path)

    assert settings.theme == DEFAULT_THEME
    assert settings.positive_quality_prompt == "Film look"


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


def test_project_assets_persist_and_saved_project_reloads(tmp_path):
    audio = tmp_path / "upload.mp3"
    script = tmp_path / "upload.md"
    audio.write_bytes(b"voice-over audio")
    script.write_text("# Narration\nA matching script.", encoding="utf-8")
    project_dir = tmp_path / "project"

    project = studio.save_project_assets(project_dir, audio, script)
    marker = studio.add_project_marker(
        project_dir, 1.25, 3.5, "A manually selected visual"
    )
    studio.save_project_assets(project_dir, audio, script)
    audio.unlink()
    script.unlink()

    loaded = studio.load_project_workspace(project_dir)

    assert loaded["audio_path"].read_bytes() == b"voice-over audio"
    assert loaded["script_path"].read_text(encoding="utf-8") == (
        "# Narration\nA matching script."
    )
    assert loaded["markers_path"] == project["markers_path"]
    assert loaded["markers"][0].to_dict() == marker.to_dict()
    assert (project_dir / "project.json").is_file()


def test_saving_project_never_overwrites_different_existing_assets(tmp_path):
    audio = tmp_path / "voice.wav"
    script = tmp_path / "script.txt"
    audio.write_bytes(b"original audio")
    script.write_text("original script", encoding="utf-8")
    project_dir = tmp_path / "project"
    studio.save_project_assets(project_dir, audio, script)

    audio.write_bytes(b"replacement audio")

    with pytest.raises(FileExistsError, match="never replaced"):
        studio.save_project_assets(project_dir, audio, script)

    assert (project_dir / "voiceover.wav").read_bytes() == b"original audio"
    assert (project_dir / "script.txt").read_text(encoding="utf-8") == (
        "original script"
    )


def test_project_marker_callbacks_add_edit_and_delete_without_losing_id(tmp_path):
    audio = tmp_path / "voice.wav"
    script = tmp_path / "script.txt"
    audio.write_bytes(b"audio")
    script.write_text("script", encoding="utf-8")
    project_dir = tmp_path / "project"
    studio.save_project_assets(project_dir, audio, script)

    added = studio.add_marker_callback(
        str(project_dir), 2.0, None, "Opening visual"
    )
    assert added[0].startswith("**Marker added:**")
    marker_id = studio.load_project_workspace(project_dir)["markers"][0].marker_id
    edited = studio.save_marker_callback(
        str(project_dir), marker_id, 2.5, 4.0, "Edited visual"
    )
    saved_marker = studio.load_project_workspace(project_dir)["markers"][0]

    assert edited[0].startswith("**Marker saved:**")
    assert saved_marker.marker_id == marker_id
    assert saved_marker.start_time == 2.5
    assert saved_marker.end_time == 4.0
    assert saved_marker.image_prompt == "Edited visual"

    deleted = studio.delete_marker_callback(str(project_dir), marker_id)

    assert deleted[0].startswith("**Marker deleted:**")
    assert studio.load_project_workspace(project_dir)["markers"] == []


def test_marker_callback_logs_validation_failure_and_returns_status(
    tmp_path, caplog
):
    with caplog.at_level("INFO", logger=studio.__name__):
        result = studio.add_marker_callback(
            str(tmp_path / "project"), None, None, "Unpersisted prompt"
        )

    assert result[0] == "**Could not add marker:** Enter a marker start time."
    assert "Studio callback started: add_marker" in caplog.text
    assert "Marker add rejected: Enter a marker start time." in caplog.text
    assert "Studio callback completed: add_marker" in caplog.text
    assert "Unpersisted prompt" not in caplog.text


def test_launch_enables_gradio_error_display(monkeypatch):
    launch_options = {}

    class FakeBlocks:
        def launch(self, **kwargs):
            launch_options.update(kwargs)

    monkeypatch.setattr(studio, "build_ui", FakeBlocks)

    studio.launch(port=7861)

    assert launch_options["show_error"] is True
    assert launch_options["server_port"] == 7861


def test_safe_diagnostic_redacts_api_keys_and_url_credentials(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "private-api-key")

    diagnostic = studio._safe_diagnostic(
        "Request to https://user:password@localhost failed with private-api-key"
    )

    assert "private-api-key" not in diagnostic
    assert "user:password" not in diagnostic
    assert "[redacted]" in diagnostic


def test_marker_generation_uses_only_selected_marker_and_persists_image(
    tmp_path, monkeypatch
):
    project_dir, selected, other = _create_marker_project(tmp_path)
    calls = _fake_marker_backend(monkeypatch)
    plan_path = tmp_path / "scene_plan.json"
    original_plan = '[{"index":1,"image_prompt":"automatic plan"}]\n'
    plan_path.write_text(original_plan, encoding="utf-8")
    positive_quality = "Film still, coherent composition"
    negative_quality = "Blurry, distorted"

    message = studio.start_marker_generation(
        project_dir,
        selected.marker_id,
        "http://127.0.0.1:8188",
        "workflow.json",
        positive_quality,
        negative_quality,
        retries=0,
    )
    worker = studio._MARKER_JOB["thread"]
    worker.join(timeout=3)

    assert message.startswith("Started local image generation")
    assert studio._MARKER_JOB["status"] == "completed"
    generation_calls = [call for call in calls if call[0] == "generate"]
    assert len(generation_calls) == 1
    _, called_marker_id, prompt, output_path, options = generation_calls[0]
    assert called_marker_id == selected.marker_id
    assert prompt == compose_prompt(positive_quality, selected.image_prompt)
    assert options.negative_prompt == compose_prompt(negative_quality, "")
    assert output_path.parent == project_dir / "images"
    assert output_path.name.startswith("marker_")
    assert output_path.name.endswith("_v001.png")
    assert studio._MARKER_JOB["image_path"] == str(output_path)

    reloaded = studio.load_project_workspace(project_dir)["markers"]
    updated = next(marker for marker in reloaded if marker.marker_id == selected.marker_id)
    untouched = next(marker for marker in reloaded if marker.marker_id == other.marker_id)
    assert updated.image_path == "images/" + output_path.name
    assert (project_dir / updated.image_path).is_file()
    assert updated.start_time == selected.start_time
    assert updated.end_time == selected.end_time
    assert updated.image_prompt == selected.image_prompt
    assert untouched.to_dict() == other.to_dict()
    assert plan_path.read_text(encoding="utf-8") == original_plan
    assert studio.marker_generation_status()[0].startswith("Image generated")


def test_marker_generation_requires_explicit_regeneration_and_versions_files(
    tmp_path, monkeypatch
):
    project_dir, marker, _ = _create_marker_project(tmp_path)
    calls = _fake_marker_backend(monkeypatch)
    arguments = (
        project_dir,
        marker.marker_id,
        "http://127.0.0.1:8188",
        "workflow.json",
        "Quality",
        "Negative",
    )

    studio.start_marker_generation(*arguments, retries=0)
    studio._MARKER_JOB["thread"].join(timeout=3)
    first_image = studio.load_project_workspace(project_dir)["markers"][0].image_path

    with pytest.raises(ValueError, match="Confirm regeneration"):
        studio.start_marker_generation(*arguments, retries=0)
    assert len([call for call in calls if call[0] == "generate"]) == 1

    studio.start_marker_generation(
        *arguments, confirm_regeneration=True, retries=0
    )
    studio._MARKER_JOB["thread"].join(timeout=3)
    second_image = studio.load_project_workspace(project_dir)["markers"][0].image_path

    assert first_image != second_image
    assert first_image.endswith("_v001.png")
    assert second_image.endswith("_v002.png")
    assert (project_dir / first_image).is_file()
    assert (project_dir / second_image).is_file()
    assert len([call for call in calls if call[0] == "generate"]) == 2


def test_marker_generation_rejects_missing_prompt_or_invalid_marker(
    tmp_path, monkeypatch
):
    project_dir, marker, _ = _create_marker_project(tmp_path, image_prompt="")
    calls = _fake_marker_backend(monkeypatch)
    args = (
        project_dir,
        marker.marker_id,
        "http://127.0.0.1:8188",
        "workflow.json",
        "",
        "",
    )

    with pytest.raises(ValueError, match="custom image prompt"):
        studio.start_marker_generation(*args)
    with pytest.raises(KeyError, match="Marker not found"):
        studio.start_marker_generation(
            project_dir, "missing-marker", *args[2:]
        )
    assert not [call for call in calls if call[0] == "generate"]


def test_marker_generation_failure_preserves_markers_and_reports_error(
    tmp_path, monkeypatch, caplog
):
    project_dir, selected, other = _create_marker_project(tmp_path)
    _fake_marker_backend(monkeypatch, outcomes=["ComfyUI unavailable"])

    with caplog.at_level("ERROR", logger=studio.__name__):
        studio.start_marker_generation(
            project_dir,
            selected.marker_id,
            "http://127.0.0.1:8188",
            "workflow.json",
            "Quality",
            "Negative",
            retries=0,
        )
        studio._MARKER_JOB["thread"].join(timeout=3)

    assert studio._MARKER_JOB["status"] == "failed"
    assert "ComfyUI unavailable" in studio.marker_generation_status()[0]
    assert any(record.exc_info for record in caplog.records)
    assert "Marker image generation failed" in caplog.text
    assert selected.image_prompt not in caplog.text
    markers = studio.load_project_workspace(project_dir)["markers"]
    assert markers[0].to_dict() == selected.to_dict()
    assert markers[1].to_dict() == other.to_dict()


def test_marker_generation_retries_backend_errors(tmp_path, monkeypatch):
    project_dir, selected, _ = _create_marker_project(tmp_path)
    calls = _fake_marker_backend(
        monkeypatch, outcomes=["temporary failure", "success"]
    )

    studio.start_marker_generation(
        project_dir,
        selected.marker_id,
        "http://127.0.0.1:8188",
        "workflow.json",
        "Quality",
        "Negative",
        retries=1,
    )
    studio._MARKER_JOB["thread"].join(timeout=4)

    assert studio._MARKER_JOB["status"] == "completed"
    assert len([call for call in calls if call[0] == "generate"]) == 2


def test_marker_generation_callback_surfaces_missing_comfyui_configuration(
    tmp_path, monkeypatch
):
    project_dir, marker, _ = _create_marker_project(tmp_path)

    def unavailable_backend(**kwargs):
        raise RuntimeError("ComfyUI workflow not found")

    monkeypatch.setattr(studio, "ComfyUIBackend", unavailable_backend)

    message = studio.start_marker_generation_callback(
        str(project_dir),
        marker.marker_id,
        "http://127.0.0.1:8188",
        "missing.json",
        "Quality",
        "Negative",
        False,
        0,
        "Workflow dimensions",
    )

    assert "ComfyUI workflow not found" in message[0]
    assert message[1] is False
    assert studio.marker_generation_status()[0] == message[0]


def test_load_project_callback_returns_audio_and_chronological_markers(tmp_path):
    audio = tmp_path / "voice.wav"
    script = tmp_path / "script.txt"
    _write_test_audio(audio)
    script.write_text("Loaded script", encoding="utf-8")
    project_dir = tmp_path / "project"
    studio.save_project_assets(project_dir, audio, script)
    later = studio.add_project_marker(project_dir, 8, image_prompt="Later")
    earlier = studio.add_project_marker(project_dir, 2, image_prompt="Earlier")

    result = studio.load_project_callback(str(project_dir))

    assert result[0].startswith("**Project loaded:**")
    assert Path(result[1]) == project_dir / "voiceover.wav"
    assert "1.000 seconds" in result[2]
    assert result[3] == "Loaded script"
    assert [row[0] for row in result[4]] == [earlier.marker_id, later.marker_id]
    assert result[5]["choices"][0][1] == earlier.marker_id


def test_automatic_scene_planning_does_not_modify_manual_markers(tmp_path):
    from ai_visualizer.scene_planner import plan_scenes, save_json

    audio = tmp_path / "voice.wav"
    script = tmp_path / "script.txt"
    audio.write_bytes(b"audio")
    script.write_text("script", encoding="utf-8")
    project_dir = tmp_path / "project"
    studio.save_project_assets(project_dir, audio, script)
    studio.add_project_marker(
        project_dir, 2.25, 6.5, "Keep this independently authored prompt"
    )
    marker_file = project_dir / "markers.json"
    markers_before = marker_file.read_bytes()
    transcript = tmp_path / "transcript.json"
    transcript.write_text(
        json.dumps({
            "audio_duration_seconds": 20.0,
            "segments": [{
                "id": 1,
                "start": 0.0,
                "end": 20.0,
                "text": "A short narration segment for automatic scene planning.",
                "words": [],
            }],
        }),
        encoding="utf-8",
    )
    scene_plan = tmp_path / "scene_plan.json"

    save_json(plan_scenes(transcript), scene_plan)

    assert scene_plan.is_file()
    assert marker_file.read_bytes() == markers_before


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
    assert len(file_components) == 8
    assert all(component["skip_api"] for component in file_components)

    api_info = app.get_api_info()

    assert "/refresh_dashboard" in api_info["named_endpoints"]
    assert "/automatic_generation_callback" in api_info["named_endpoints"]
    assert "/continuous_coro" in api_info["named_endpoints"]
    assert len(
        api_info["named_endpoints"]["/refresh_dashboard"]["parameters"]
    ) == 3
    assert len(
        api_info["named_endpoints"]["/refresh_dashboard"]["returns"]
    ) == 7
    assert len(
        api_info["named_endpoints"]["/automatic_generation_callback"]["parameters"]
    ) == 14
    assert len(api_info["named_endpoints"]["/continuous_coro"]["parameters"]) == 3
    assert all(
        dep["show_api"]
        for dep in config["dependencies"]
        if dep["api_name"] is not False
    )
    assert sum(not dep["show_api"] for dep in config["dependencies"]) == 40


def test_studio_event_callback_arities_match_registered_dependencies():
    app = studio.build_ui()
    expected_callbacks = Counter(
        [
            ("preview_new_project_inputs", 3, 6),
            ("preview_new_project_inputs", 3, 6),
            ("preview_new_project_inputs", 3, 6),
            ("reset_project_creation", 1, 5),
            ("reset_project_creation", 1, 5),
            ("reset_project_creation", 1, 5),
            ("reset_project_creation", 1, 5),
            ("create_project_callback", 4, 4),
            ("autosave_assisted_segment_callback", 5, 4),
            ("autosave_assisted_segment_callback", 5, 4),
            ("autosave_assisted_segment_callback", 5, 4),
            ("generate_assisted_segment_callback", 12, 5),
            ("load_assisted_workspace_callback", 1, 10),
            ("load_assisted_workspace_callback", 1, 10),
            ("load_assisted_workspace_callback", 1, 10),
            ("load_assisted_workspace_callback", 1, 10),
            ("load_automatic_project_callback", 1, 9),
            ("load_automatic_project_callback", 1, 9),
            ("refresh_automatic_scene_table", 1, 3),
            ("refresh_automatic_scene_table", 1, 3),
            ("refresh_assisted_segment_status", 2, 6),
            ("select_assisted_marker_ui", 2, 8),
            ("select_assisted_segment_ui", 2, 8),
            ("save_project_callback", 3, 10),
            ("load_project_callback", 1, 10),
            ("add_marker_callback", 4, 7),
            ("save_marker_callback", 5, 7),
            ("delete_marker_callback", 2, 7),
            ("<lambda>", 0, 1),
            ("<lambda>", 1, 16),
            ("<lambda>", 1, 16),
            ("backend_visibility", 1, 3),
            ("lambda", 1, 1),
            ("apply_theme", 4, 3),
            ("refresh_dashboard", 3, 10),
            ("refresh_dashboard", 3, 10),
            ("refresh_dashboard", 3, 10),
            ("check_backend_readiness", 3, 1),
            ("automatic_generation_callback", 14, 1),
            ("cancel_generation", 0, 1),
            ("retry_automatic_scene_callback", 12, 1),
            ("continuous_coro", 3, 10),
            ("marker_generation_status", 0, 3),
            ("continuous_coro", 2, 6),
            ("continuous_coro", 1, 3),
            ("<lambda>", 2, 8),
            ("<lambda>", 2, 8),
            ("<lambda>", 12, 5),
        ]
    )
    actual_callbacks = Counter()

    for dependency, callback in zip(app.config["dependencies"], app.fns):
        name = (
            callback.fn.__name__
            if dependency["api_name"] is False
            else dependency["api_name"]
        )
        if name == "continuous_coro" and not dependency["inputs"]:
            name = "marker_generation_status"
        input_count = len(dependency["inputs"])
        output_count = len(dependency["outputs"])
        actual_callbacks[(name, input_count, output_count)] += 1
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

    assert actual_callbacks == expected_callbacks


def test_assisted_generate_and_retry_use_the_script_segment_selector():
    app = studio.build_ui()
    component_labels = {
        component["id"]: component["props"].get("label")
        for component in app.config["components"]
    }
    generation_dependencies = [
        dependency
        for dependency, callback in zip(app.config["dependencies"], app.fns)
        if callback.fn is studio.generate_assisted_segment_callback
        or (
            callback.fn.__name__ == "<lambda>"
            and len(dependency["inputs"]) == 12
            and len(dependency["outputs"]) == 5
        )
    ]

    assert len(generation_dependencies) == 2
    assert all(
        component_labels[dependency["inputs"][1]] == "Script segments"
        for dependency in generation_dependencies
    )


def test_automatic_start_and_retry_callbacks_forward_registered_inputs(monkeypatch):
    app = studio.build_ui()
    callbacks = {
        dependency["api_name"]: callback.fn
        for dependency, callback in zip(app.config["dependencies"], app.fns)
    }
    retry_callback = next(
        callback.fn
        for dependency, callback in zip(app.config["dependencies"], app.fns)
        if callback.fn is studio.retry_automatic_scene_callback
    )
    calls = []

    def record_automatic_generation(*args, **kwargs):
        calls.append((args, kwargs))
        return "Started"

    monkeypatch.setattr(
        studio, "start_automatic_generation", record_automatic_generation
    )
    start_values = [
        "C:\\projects\\sample",
        "local",
        "Workflow dimensions",
        "All scenes",
        [],
        0.5,
        1,
        False,
        False,
        "http://127.0.0.1:8188",
        "workflow.json",
        "positive prompt",
        "negative prompt",
        "Cinematic Concept Art",
    ]

    assert callbacks["automatic_generation_callback"](*start_values) == "Started"
    assert calls[-1] == (tuple(start_values), {})

    retry_values = [
        start_values[0],
        start_values[1],
        start_values[2],
        start_values[5],
        start_values[6],
        start_values[7],
        start_values[9],
        start_values[10],
        start_values[11],
        start_values[12],
        start_values[13],
        "7",
    ]
    assert retry_callback(*retry_values) == "Started"
    assert calls[-1] == (
        (
            start_values[0],
            start_values[1],
            start_values[2],
            "Selected scenes",
            ["7"],
            start_values[5],
            start_values[6],
            start_values[7],
            False,
            start_values[9],
            start_values[10],
            start_values[11],
            start_values[12],
            start_values[1 + 12],
        ),
        {"retry_scene": "7"},
    )


def test_theme_change_updates_theme_defaults_but_preserves_prompt_edits():
    app = studio.build_ui()
    callback = next(
        callback.fn
        for dependency, callback in zip(app.config["dependencies"], app.fns)
        if dependency["api_name"] == "apply_theme"
    )

    positive, negative, theme = callback(
        "Cinematic Concept Art",
        studio.DEFAULT_POSITIVE_PROMPT,
        studio.DEFAULT_NEGATIVE_PROMPT,
        DEFAULT_THEME,
    )

    assert positive == CINEMATIC_CONCEPT_ART_POSITIVE
    assert negative == CINEMATIC_CONCEPT_ART_NEGATIVE
    assert theme == "Cinematic Concept Art"

    positive, negative, theme = callback(
        DEFAULT_THEME,
        "My edited positive prompt",
        CINEMATIC_CONCEPT_ART_NEGATIVE,
        "Cinematic Concept Art",
    )

    assert positive == "My edited positive prompt"
    assert negative == studio.DEFAULT_NEGATIVE_PROMPT
    assert theme == DEFAULT_THEME


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
    saved_settings = []
    monkeypatch.setattr(studio, "save_settings", saved_settings.append)

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
        "Cinematic Concept Art",
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
    assert received_configs[0].theme == "Cinematic Concept Art"
    assert saved_settings[0].theme == "Cinematic Concept Art"


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


def test_start_generation_rejects_unknown_theme_before_launch(tmp_path, monkeypatch):
    plan = tmp_path / "plan.json"
    _write_plan(plan)

    with pytest.raises(ValueError, match="Unknown visual theme"):
        studio.start_generation(
            "api", str(plan), str(tmp_path / "images"),
            str(tmp_path / "manifest.json"), "16:9", "All scenes", [],
            1, 0, 0, True, False, theme="Unknown",
        )

    assert studio._JOB["status"] == "idle"


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

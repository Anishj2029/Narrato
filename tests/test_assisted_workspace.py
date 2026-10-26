from __future__ import annotations

import json
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
        audio.writeframes(b"\0\0" * 16000)
    return path


@pytest.fixture
def assisted_project(tmp_path):
    audio = _write_audio(tmp_path / "voice.wav")
    script = tmp_path / "script.md"
    script.write_text(
        "## Opening scene\n"
        "A quiet lighthouse stands beside the cold northern sea.\n\n"
        "## Second scene\n"
        "A fishing boat returns safely to the harbor at sunrise.",
        encoding="utf-8",
    )
    project = studio.save_project_assets(tmp_path / "project", audio, script)
    return project["project_dir"]


@pytest.fixture(autouse=True)
def reset_marker_job():
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
    yield
    thread = studio._MARKER_JOB.get("thread")
    if thread is not None and thread.is_alive():
        thread.join(timeout=3)


def _fake_backend(monkeypatch, result="success", outcomes=None):
    calls = []
    pending_outcomes = list(outcomes or [])

    class FakeBackend:
        model_identifier = "test-model"

        def __init__(self, base_url, workflow_path):
            calls.append(("configured", base_url, workflow_path))

        def preflight(self):
            calls.append(("preflight",))

        def generate(self, marker_id, prompt, output_path, options):
            calls.append(("generate", marker_id, prompt, output_path, options))
            outcome = pending_outcomes.pop(0) if pending_outcomes else result
            if outcome != "success":
                return GenerationResult(None, "local", "test-model", False, outcome)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), color=(60, 90, 70)).save(output_path)
            return GenerationResult(output_path, "local", "test-model", True)

    monkeypatch.setattr(studio, "ComfyUIBackend", FakeBackend)
    return calls


def test_existing_project_loads_with_segments_and_existing_markers(assisted_project):
    manual = studio.add_project_marker(
        assisted_project, 1.5, 2.0, "Existing manual marker"
    )

    loaded = studio.load_assisted_workspace(assisted_project)
    ui_state = studio.load_assisted_workspace_callback(assisted_project)

    assert len(loaded["segments"]) == 2
    assert loaded["audio_path"].name == "voiceover.wav"
    assert loaded["script_path"].name == "script.md"
    assert "proportional estimates" in loaded["timing_note"]
    assert manual.marker_id in {item.marker_id for item in loaded["markers"]}
    assert len(ui_state[0]) == 2
    assert len(ui_state[1]["choices"]) == 2
    assert ui_state[3] == loaded["segments"][0]["text"]


def test_assisted_segments_use_estimated_word_count_timing_and_keep_manual_markers(
    assisted_project,
):
    manual = studio.add_project_marker(assisted_project, 0.15, 0.25, "Keep me")
    loaded = studio.load_assisted_workspace(assisted_project)

    assert loaded["audio_duration"] == pytest.approx(2.0)
    assert loaded["segments"][0]["estimated_start"] == pytest.approx(0.0)
    assert loaded["segments"][-1]["estimated_end"] == pytest.approx(2.0)
    assert all(
        row[2].endswith("(estimated)")
        for row in studio.assisted_segment_rows(assisted_project)
    )
    markers = studio.load_project_workspace(assisted_project)["markers"]
    assert next(item for item in markers if item.marker_id == manual.marker_id).image_prompt == (
        "Keep me"
    )


def test_selecting_a_segment_only_loads_context_and_never_generates(
    assisted_project, monkeypatch
):
    calls = _fake_backend(monkeypatch)
    loaded = studio.load_assisted_workspace(assisted_project)
    selected = studio.select_assisted_segment(
        assisted_project, loaded["segments"][1]["segment_id"]
    )

    assert selected[0] == loaded["segments"][1]["text"]
    assert selected[1] == loaded["segments"][1]["marker"].start_time
    assert "estimated" in selected[5]
    assert not [call for call in calls if call[0] == "generate"]
    assert studio._MARKER_JOB["status"] == "idle"


def test_empty_selection_refreshes_do_not_clear_the_active_segment(
    assisted_project,
):
    segment_id = studio.load_assisted_workspace(assisted_project)["segments"][0][
        "segment_id"
    ]
    segment_selection = studio.select_assisted_segment_ui(assisted_project, None)
    marker_selection = studio.select_assisted_marker_ui(assisted_project, None)
    status_refresh = studio.refresh_assisted_segment_status(assisted_project, None)
    idle_refresh = studio.refresh_assisted_segment_status(
        assisted_project, segment_id
    )

    assert segment_selection[6] == studio.gr_update()
    assert marker_selection[5] == studio.gr_update()
    assert status_refresh[1] == studio.gr_update()
    assert idle_refresh[0] == studio.gr_update()


def test_editing_a_prompt_persists_only_on_the_selected_segment(assisted_project):
    loaded = studio.load_assisted_workspace(assisted_project)
    first_id, second_id = (
        segment["segment_id"] for segment in loaded["segments"]
    )

    studio.save_assisted_segment(
        assisted_project, first_id, 0.1, 0.8, "A detailed lighthouse prompt"
    )
    reloaded = studio.load_assisted_workspace(assisted_project)

    assert reloaded["segments"][0]["marker"].image_prompt == (
        "A detailed lighthouse prompt"
    )
    assert reloaded["segments"][0]["marker"].start_time == 0.1
    assert reloaded["segments"][0]["marker"].end_time == 0.8
    assert reloaded["segments"][1]["marker"].image_prompt == ""
    assert reloaded["segments"][1]["marker"].start_time == pytest.approx(
        reloaded["segments"][1]["estimated_start"]
    )
    rows = studio.assisted_segment_rows(assisted_project)
    assert rows[0][2] == "0.10–0.80s (marker)"
    assert rows[1][2].endswith("(estimated)")


def test_autosave_persists_prompt_and_timing_for_a_valid_segment(
    assisted_project,
):
    segment = studio.load_assisted_workspace(assisted_project)["segments"][0]

    response = studio.autosave_assisted_segment_callback(
        str(assisted_project),
        segment["segment_id"],
        0.2,
        0.9,
        "Autosaved lighthouse prompt",
    )
    reloaded = studio.load_assisted_workspace(assisted_project)
    saved = reloaded["segments"][0]["marker"]

    assert response[0].startswith("**Saved prompt and segment boundaries**")
    assert saved.start_time == 0.2
    assert saved.end_time == 0.9
    assert saved.image_prompt == "Autosaved lighthouse prompt"


@pytest.mark.parametrize("selection", ["stale", "manual"])
def test_autosave_rejects_missing_or_non_segment_ids_without_changing_data(
    assisted_project, selection
):
    if selection == "manual":
        selected_id = studio.add_project_marker(
            assisted_project, 0.2, image_prompt="Manual marker prompt"
        ).marker_id
    else:
        selected_id = "assisted-9999-stale"
    markers_path = assisted_project / "markers.json"
    before = markers_path.read_bytes()

    response = studio.autosave_assisted_segment_callback(
        str(assisted_project), selected_id, 0.3, 1.1, "Must not be saved"
    )

    assert response[0].startswith("**Could not save segment:**")
    assert "Script segment not found" in response[0]
    assert markers_path.read_bytes() == before


def test_navigation_then_autosave_targets_the_newly_displayed_segment(
    assisted_project,
):
    segments = studio.load_assisted_workspace(assisted_project)["segments"]
    navigation = studio.navigate_assisted_segment(
        assisted_project, segments[0]["segment_id"], 1
    )
    selected_id = navigation[0]["value"]

    response = studio.autosave_assisted_segment_callback(
        str(assisted_project), selected_id, 1.1, 1.9, "Second scene prompt"
    )
    reloaded = studio.load_assisted_workspace(assisted_project)["segments"]

    assert selected_id == segments[1]["segment_id"]
    assert navigation[1]["value"] == selected_id
    assert response[0].startswith("**Saved prompt and segment boundaries**")
    assert reloaded[0]["marker"].image_prompt == ""
    assert reloaded[1]["marker"].image_prompt == "Second scene prompt"
    assert reloaded[1]["marker"].start_time == 1.1
    assert reloaded[1]["marker"].end_time == 1.9


def test_generate_targets_only_the_selected_segment_and_reloads_image(
    assisted_project, monkeypatch
):
    calls = _fake_backend(monkeypatch)
    first, second = studio.load_assisted_workspace(assisted_project)["segments"]

    response = studio.generate_assisted_segment_callback(
        str(assisted_project),
        first["segment_id"],
        first["marker"].start_time,
        first["marker"].end_time,
        "A single selected visual",
        "http://127.0.0.1:8188",
        "workflow.json",
        "Quality",
        "Negative",
        False,
        0,
        "Workflow dimensions",
    )
    studio._MARKER_JOB["thread"].join(timeout=3)
    reloaded = studio.load_assisted_workspace(assisted_project)

    generated_calls = [call for call in calls if call[0] == "generate"]
    assert response[0].startswith("Started local image generation")
    assert len(generated_calls) == 1
    assert generated_calls[0][1] == first["segment_id"]
    assert generated_calls[0][2].endswith("A single selected visual")
    assert reloaded["segments"][0]["status"] == "Image saved"
    assert reloaded["segments"][0]["image_path"]
    assert Path(reloaded["segments"][0]["image_path"]).is_file()
    assert reloaded["segments"][1]["status"] == "No image"
    assert reloaded["segments"][1]["marker"].image_path is None


@pytest.mark.parametrize("selection", ["stale", "manual"])
def test_generation_rejects_stale_or_manual_marker_ids_without_mutating_project(
    assisted_project, monkeypatch, selection
):
    calls = _fake_backend(monkeypatch)
    studio.load_assisted_workspace(assisted_project)
    if selection == "manual":
        selected_id = studio.add_project_marker(
            assisted_project, 0.2, image_prompt="Manual marker prompt"
        ).marker_id
    else:
        selected_id = "5e8d6d5a-01b6-4305-bf4f-7e58c1a5e57f"
    markers_path = assisted_project / "markers.json"
    before = markers_path.read_bytes()

    response = studio.generate_assisted_segment_callback(
        str(assisted_project),
        selected_id,
        0.2,
        0.8,
        "Must not be saved",
        "http://127.0.0.1:8188",
        "workflow.json",
        "Quality",
        "Negative",
        False,
        0,
        "Workflow dimensions",
    )

    assert response[0].startswith("**Could not start image generation:**")
    assert "Script segment not found" in response[0]
    assert not [call for call in calls if call[0] == "generate"]
    assert markers_path.read_bytes() == before


def test_navigation_then_generation_targets_the_newly_selected_segment(
    assisted_project, monkeypatch
):
    calls = _fake_backend(monkeypatch)
    first, second = studio.load_assisted_workspace(assisted_project)["segments"]
    navigation = studio.navigate_assisted_segment(
        assisted_project, first["segment_id"], 1
    )
    selected_id = navigation[0]["value"]

    response = studio.generate_assisted_segment_callback(
        str(assisted_project),
        selected_id,
        second["marker"].start_time,
        second["marker"].end_time,
        "Second scene after navigation",
        "http://127.0.0.1:8188",
        "workflow.json",
        "Quality",
        "Negative",
        False,
        0,
        "Workflow dimensions",
    )
    studio._MARKER_JOB["thread"].join(timeout=3)
    reloaded = studio.load_assisted_workspace(assisted_project)
    generated = [call for call in calls if call[0] == "generate"]

    assert selected_id == second["segment_id"]
    assert response[0].startswith("Started local image generation")
    assert len(generated) == 1
    assert generated[0][1] == second["segment_id"]
    assert reloaded["segments"][0]["marker"].image_path is None
    assert reloaded["segments"][1]["marker"].image_path
    assert Path(reloaded["segments"][1]["image_path"]).is_file()


def test_editing_prompt_then_generating_uses_that_segment_and_prompt(
    assisted_project, monkeypatch
):
    calls = _fake_backend(monkeypatch)
    first, second = studio.load_assisted_workspace(assisted_project)["segments"]
    prompt = "Edited prompt for the first scene"
    studio.autosave_assisted_segment_callback(
        str(assisted_project),
        first["segment_id"],
        first["marker"].start_time,
        first["marker"].end_time,
        prompt,
    )

    response = studio.generate_assisted_segment_callback(
        str(assisted_project),
        first["segment_id"],
        first["marker"].start_time,
        first["marker"].end_time,
        prompt,
        "http://127.0.0.1:8188",
        "workflow.json",
        "Quality",
        "Negative",
        False,
        0,
        "Workflow dimensions",
    )
    studio._MARKER_JOB["thread"].join(timeout=3)
    generated = [call for call in calls if call[0] == "generate"]
    reloaded = studio.load_assisted_workspace(assisted_project)

    assert response[0].startswith("Started local image generation")
    assert len(generated) == 1
    assert generated[0][1] == first["segment_id"]
    assert generated[0][2].endswith(prompt)
    assert reloaded["segments"][0]["marker"].image_prompt == prompt
    assert reloaded["segments"][0]["marker"].image_path
    assert reloaded["segments"][1]["marker"].image_path is None


def test_retry_after_failed_generation_persists_image_to_same_segment(
    assisted_project, monkeypatch
):
    calls = _fake_backend(
        monkeypatch, outcomes=["ComfyUI unavailable", "success"]
    )
    segment = studio.load_assisted_workspace(assisted_project)["segments"][0]
    arguments = (
        str(assisted_project),
        segment["segment_id"],
        segment["marker"].start_time,
        segment["marker"].end_time,
        "Retry this selected scene",
        "http://127.0.0.1:8188",
        "workflow.json",
        "Quality",
        "Negative",
        False,
        0,
        "Workflow dimensions",
    )

    first_response = studio.generate_assisted_segment_callback(*arguments)
    studio._MARKER_JOB["thread"].join(timeout=3)
    assert studio._MARKER_JOB["status"] == "failed"

    retry_response = studio.generate_assisted_segment_callback(
        *arguments, retry=True
    )
    studio._MARKER_JOB["thread"].join(timeout=3)
    generated = [call for call in calls if call[0] == "generate"]
    reloaded = studio.load_assisted_workspace(assisted_project)

    assert first_response[0].startswith("Started local image generation")
    assert retry_response[0].startswith("Started local image generation")
    assert [call[1] for call in generated] == [
        segment["segment_id"],
        segment["segment_id"],
    ]
    assert reloaded["segments"][0]["marker"].image_path
    assert Path(reloaded["segments"][0]["image_path"]).is_file()
    assert reloaded["segments"][1]["marker"].image_path is None


def test_failed_generation_status_is_visible_and_does_not_save_bad_image(
    assisted_project, monkeypatch
):
    _fake_backend(monkeypatch, result="ComfyUI unavailable")
    segment = studio.load_assisted_workspace(assisted_project)["segments"][0]

    studio.generate_assisted_segment_callback(
        str(assisted_project),
        segment["segment_id"],
        segment["marker"].start_time,
        segment["marker"].end_time,
        "A saved prompt despite backend failure",
        "http://127.0.0.1:8188",
        "workflow.json",
        "Quality",
        "Negative",
        False,
        0,
        "Workflow dimensions",
    )
    studio._MARKER_JOB["thread"].join(timeout=3)
    refreshed = studio.refresh_assisted_segment_status(
        assisted_project, segment["segment_id"]
    )
    reloaded = studio.load_assisted_workspace(assisted_project)

    assert studio._MARKER_JOB["status"] == "failed"
    assert "ComfyUI unavailable" in refreshed[1]
    assert reloaded["segments"][0]["status"] == "Failed"
    assert reloaded["segments"][0]["marker"].image_path is None
    assert reloaded["segments"][0]["marker"].image_prompt == (
        "A saved prompt despite backend failure"
    )


def test_previous_and_next_segment_navigation_wraps(assisted_project):
    segments = studio.load_assisted_workspace(assisted_project)["segments"]

    next_values = studio.navigate_assisted_segment(
        assisted_project, segments[0]["segment_id"], 1
    )
    previous_values = studio.navigate_assisted_segment(
        assisted_project, segments[0]["segment_id"], -1
    )

    assert next_values[0]["value"] == segments[1]["segment_id"]
    assert next_values[1]["value"] == segments[1]["segment_id"]
    assert next_values[2] == segments[1]["text"]
    assert previous_values[0]["value"] == segments[1]["segment_id"]
    assert previous_values[2] == segments[1]["text"]


def test_legacy_project_metadata_and_existing_marker_images_remain_compatible(
    assisted_project,
):
    project_file = assisted_project / "project.json"
    metadata = json.loads(project_file.read_text(encoding="utf-8"))
    for field in ("project_name", "created_at", "creation_state", "selected_mode"):
        metadata.pop(field, None)
    project_file.write_text(json.dumps(metadata), encoding="utf-8")
    existing_marker = studio.add_project_marker(
        assisted_project, 0.5, image_prompt="Legacy marker prompt"
    )
    image = assisted_project / "images" / "existing.png"
    image.parent.mkdir(exist_ok=True)
    Image.new("RGB", (4, 4)).save(image)
    markers = load_markers(assisted_project / "markers.json")
    edit_marker(
        markers,
        existing_marker.marker_id,
        image_path="images/existing.png",
    )
    save_markers(assisted_project / "markers.json", markers)

    loaded = studio.load_assisted_workspace(assisted_project)

    assert loaded["metadata"]["schema_version"] == 1
    assert loaded["project_name"] == assisted_project.name
    assert loaded["segments"]
    assert studio._marker_image_path(assisted_project, "images/existing.png") == image
    assert sum(not item.marker_id.startswith("assisted-") for item in loaded["markers"]) == 1

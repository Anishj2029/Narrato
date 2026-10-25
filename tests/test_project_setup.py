from __future__ import annotations

import json
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_visualizer import studio


def _write_audio(path: Path) -> Path:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes(b"\0\0" * 8000)
    return path


@pytest.fixture
def project_inputs(tmp_path):
    audio = _write_audio(tmp_path / "narration.wav")
    script = tmp_path / "script.txt"
    script.write_text("A complete narration script.", encoding="utf-8")
    return audio, script, tmp_path / "projects"


@pytest.mark.parametrize(
    ("filename", "contents"),
    [
        ("script.txt", "A text script."),
        ("script.md", "# A markdown script"),
    ],
)
def test_create_project_accepts_txt_and_markdown(
    tmp_path, project_inputs, filename, contents
):
    audio, _, projects_root = project_inputs
    script = tmp_path / filename
    script.write_text(contents, encoding="utf-8")

    project = studio.create_project(
        "Test Story", audio, script, projects_root=projects_root
    )

    assert project["project_name"] == "Test Story"
    assert project["audio_path"].name == "voiceover.wav"
    assert project["script_path"].name == f"script{script.suffix}"
    assert project["script_path"].read_text(encoding="utf-8") == contents
    assert project["markers_path"].is_file()


def test_create_project_accepts_pasted_script(project_inputs):
    audio, _, projects_root = project_inputs

    project = studio.create_project(
        "Pasted project",
        audio,
        pasted_script="Script entered in the form.",
        projects_root=projects_root,
    )

    assert project["script_path"].name == "script.txt"
    assert project["script_path"].read_text(encoding="utf-8") == (
        "Script entered in the form."
    )


@pytest.mark.parametrize("upload_shape", ["filepath", "gradio_file_data"])
def test_project_creation_callback_accepts_uploaded_script_and_reloads_dropdowns(
    tmp_path, project_inputs, monkeypatch, upload_shape
):
    audio, script, _ = project_inputs
    script.write_text(
        "## Opening scene\nA complete narration script.", encoding="utf-8"
    )
    root = tmp_path / "workspace"
    monkeypatch.setattr(studio, "_PROJECT_ROOT", root)
    script_upload = (
        str(script)
        if upload_shape == "filepath"
        else SimpleNamespace(path=str(script))
    )

    response = studio.create_project_callback(
        "Uploaded script", audio, script_upload, ""
    )

    project_dir = Path(response[1])
    assert response[0].startswith("**Project created:**")
    assert project_dir == root / "projects" / "Uploaded_script"
    assert (project_dir / "script.txt").read_text(encoding="utf-8") == (
        "## Opening scene\nA complete narration script."
    )
    studio.load_project_callback(str(project_dir))
    workspace_state = studio.load_assisted_workspace_callback(project_dir)
    segment_dropdown = workspace_state[1]
    marker_dropdown = workspace_state[2]

    assert segment_dropdown["value"] in {
        value for _, value in segment_dropdown["choices"]
    }
    assert marker_dropdown["value"] in {
        value for _, value in marker_dropdown["choices"]
    }
    assert studio.load_project_workspace(project_dir)["project_dir"] == project_dir


def test_project_creation_callback_accepts_pasted_script(
    tmp_path, project_inputs, monkeypatch
):
    audio, _, _ = project_inputs
    monkeypatch.setattr(studio, "_PROJECT_ROOT", tmp_path / "workspace")

    response = studio.create_project_callback(
        "Pasted callback project", audio, None, "Script pasted into the form."
    )

    project_dir = Path(response[1])
    assert response[0].startswith("**Project created:**")
    assert (project_dir / "script.txt").read_text(encoding="utf-8") == (
        "Script pasted into the form."
    )


@pytest.mark.parametrize(
    ("audio_value", "script_value", "message"),
    [
        (None, "script.txt", "Choose the audio file"),
        ("narration.wav", None, "Upload a TXT/Markdown script"),
    ],
)
def test_project_creation_callback_reports_missing_inputs_without_creating_project(
    tmp_path, project_inputs, monkeypatch, audio_value, script_value, message
):
    _, script, _ = project_inputs
    audio = audio_value
    if audio_value == "narration.wav":
        audio = _write_audio(tmp_path / audio_value)
    if script_value == "script.txt":
        script_value = str(script)
    root = tmp_path / "workspace"
    monkeypatch.setattr(studio, "_PROJECT_ROOT", root)

    response = studio.create_project_callback(
        "Incomplete project", audio, script_value, ""
    )

    assert message in response[0]
    assert response[1] is None
    assert not (root / "projects").exists()


def test_new_project_preview_shows_file_names_duration_and_script(project_inputs):
    audio, script, _ = project_inputs

    status, audio_path, audio_name, duration, script_name, script_text = (
        studio.preview_new_project_inputs(audio, script, "")
    )

    assert audio_path == str(audio)
    assert audio_name == audio.name
    assert "1.00 seconds" in duration
    assert script_name == script.name
    assert script_text == "A complete narration script."
    assert "Upload a TXT/Markdown script" not in status


def test_create_project_rejects_missing_audio(project_inputs):
    _, script, projects_root = project_inputs

    with pytest.raises(ValueError, match="audio file"):
        studio.create_project(
            "Missing audio", None, script, projects_root=projects_root
        )


def test_create_project_rejects_missing_script(project_inputs):
    audio, _, projects_root = project_inputs

    with pytest.raises(ValueError, match="Upload a TXT/Markdown script"):
        studio.create_project(
            "Missing script", audio, projects_root=projects_root
        )


def test_create_project_rejects_empty_script(project_inputs, tmp_path):
    audio, _, projects_root = project_inputs
    empty_script = tmp_path / "empty.txt"
    empty_script.write_text(" \n\t", encoding="utf-8")

    with pytest.raises(ValueError, match="script is empty"):
        studio.create_project(
            "Empty script", audio, empty_script, projects_root=projects_root
        )


@pytest.mark.parametrize(
    ("audio_suffix", "script_suffix", "message"),
    [
        (".ogg", ".txt", "MP3 or WAV"),
        (".wav", ".rtf", "TXT or Markdown"),
    ],
)
def test_create_project_rejects_unsupported_file_types(
    tmp_path, project_inputs, audio_suffix, script_suffix, message
):
    valid_audio, _, projects_root = project_inputs
    audio = valid_audio
    if audio_suffix == ".ogg":
        audio = tmp_path / f"voice{audio_suffix}"
        audio.write_bytes(b"not audio")
    script = tmp_path / f"script{script_suffix}"
    script.write_text("Some text.", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        studio.create_project(
            "Unsupported file", audio, script, projects_root=projects_root
        )


def test_create_project_uses_safe_windows_folder_name(project_inputs):
    audio, script, projects_root = project_inputs

    project = studio.create_project(
        "../Story: one. ",
        audio,
        script,
        projects_root=projects_root,
    )

    assert project["project_dir"] == projects_root.resolve() / "Story_one"
    assert project["project_dir"].parent == projects_root.resolve()


def test_create_project_persists_metadata_and_reopens(project_inputs):
    audio, script, projects_root = project_inputs

    created = studio.create_project(
        "Metadata Story", audio, script, projects_root=projects_root
    )
    metadata = json.loads(
        (created["project_dir"] / "project.json").read_text(encoding="utf-8")
    )
    reopened = studio.load_project_workspace(created["project_dir"])

    assert metadata["schema_version"] == 1
    assert metadata["project_name"] == "Metadata Story"
    assert metadata["creation_state"] == "created"
    assert metadata["created_at"]
    assert metadata["audio_file"] == "voiceover.wav"
    assert metadata["script_file"] == "script.txt"
    assert reopened["project_name"] == "Metadata Story"
    assert reopened["script_path"].read_text(encoding="utf-8") == (
        "A complete narration script."
    )
    assert reopened["markers"] == []


@pytest.mark.parametrize(
    ("mode", "tab_index"),
    [
        ("Assisted Visual Creation", "assisted"),
        ("Automatic Visual Creation", "automatic"),
    ],
)
def test_project_mode_selection_persists_and_routes(
    project_inputs, mode, tab_index
):
    audio, script, projects_root = project_inputs
    project = studio.create_project(
        "Mode Story", audio, script, projects_root=projects_root
    )

    reopened, selected_tab = studio.route_project_to_mode(
        project["project_dir"], mode
    )

    assert selected_tab == tab_index
    assert reopened["selected_mode"] == mode
    assert reopened["metadata"]["creation_state"] == "workflow_selected"


def test_project_creation_does_not_generate_images(project_inputs):
    audio, script, projects_root = project_inputs

    project = studio.create_project(
        "No automatic images", audio, script, projects_root=projects_root
    )

    assert not (project["project_dir"] / "images").exists()
    assert studio._snapshot_job()["total"] == 0


def test_create_project_rejects_duplicate_name_without_overwriting(project_inputs):
    audio, script, projects_root = project_inputs
    project = studio.create_project(
        "Repeated", audio, script, projects_root=projects_root
    )
    before = (project["project_dir"] / "project.json").read_bytes()

    with pytest.raises(FileExistsError, match="already exists"):
        studio.create_project(
            "Repeated", audio, script, projects_root=projects_root
        )

    assert (project["project_dir"] / "project.json").read_bytes() == before
    assert not list(projects_root.glob(".Repeated.*"))


def test_create_project_cleans_staging_files_after_persistence_failure(
    project_inputs, monkeypatch
):
    audio, script, projects_root = project_inputs
    save_assets = studio.save_project_assets

    def fail_after_saving(*args, **kwargs):
        save_assets(*args, **kwargs)
        raise OSError("simulated persistence failure")

    monkeypatch.setattr(studio, "save_project_assets", fail_after_saving)

    with pytest.raises(OSError, match="simulated persistence failure"):
        studio.create_project(
            "Failed project", audio, script, projects_root=projects_root
        )

    assert list(projects_root.iterdir()) == []

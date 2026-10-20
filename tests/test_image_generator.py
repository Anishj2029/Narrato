"""
Tests for image_generator – placeholder backend only (no API credits).
"""

import tempfile
from pathlib import Path

import pytest
from ai_visualizer.scene_parser import parse_script_text
from ai_visualizer.timestamp_mapper import assign_timestamps_no_audio
from ai_visualizer.image_generator import generate_images


SCRIPT = """
# Dawn

The sun rises slowly over the mountain. The world holds its breath.

# Midday

Heat shimmers on the road. Travelers seek shade under ancient trees.
"""


def _get_scenes():
    scenes = parse_script_text(SCRIPT)
    assign_timestamps_no_audio(scenes)
    # Give each scene a prompt so generation isn't skipped
    for s in scenes:
        s.image_prompt = s.text
    return scenes


def test_placeholder_creates_images():
    scenes = _get_scenes()
    with tempfile.TemporaryDirectory() as tmp:
        generate_images(scenes, output_dir=tmp, backend="placeholder")
        for s in scenes:
            assert s.image_path, f"Scene {s.index} missing image_path"
            assert Path(s.image_path).exists(), f"Image file missing: {s.image_path}"


def test_placeholder_images_are_png():
    scenes = _get_scenes()
    with tempfile.TemporaryDirectory() as tmp:
        generate_images(scenes, output_dir=tmp, backend="placeholder")
        for s in scenes:
            assert s.image_path.endswith(".png")


def test_image_filenames_are_numbered():
    scenes = _get_scenes()
    with tempfile.TemporaryDirectory() as tmp:
        generate_images(scenes, output_dir=tmp, backend="placeholder")
        for s in scenes:
            fname = Path(s.image_path).name
            assert fname == f"scene_{s.index:03d}.png", f"Unexpected filename: {fname}"


def test_skip_existing():
    scenes = _get_scenes()
    with tempfile.TemporaryDirectory() as tmp:
        # First run
        generate_images(scenes, output_dir=tmp, backend="placeholder")
        mtimes = {s.index: Path(s.image_path).stat().st_mtime for s in scenes}

        # Reset image_path to force re-check
        for s in scenes:
            s.image_path = ""

        # Second run with skip_existing=True (default)
        generate_images(scenes, output_dir=tmp, backend="placeholder", skip_existing=True)
        for s in scenes:
            new_mtime = Path(s.image_path).stat().st_mtime
            assert new_mtime == mtimes[s.index], "File should not be regenerated"


def test_csv_exported_after_generation():
    scenes = _get_scenes()
    with tempfile.TemporaryDirectory() as tmp:
        generate_images(scenes, output_dir=tmp, backend="placeholder")
        csv_path = Path(tmp) / "scenes_with_images.csv"
        assert csv_path.exists(), "CSV should be exported after generation"

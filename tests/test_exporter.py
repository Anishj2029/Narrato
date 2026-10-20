"""
Tests for exporter – writes to tmp dirs, no external deps.
"""

import csv
import json
import tempfile
from pathlib import Path

import pytest
from ai_visualizer.scene_parser import parse_script_text
from ai_visualizer.timestamp_mapper import assign_timestamps_no_audio
from ai_visualizer.exporter import to_csv, to_json, load_json


SCRIPT = """
# Act One

The hero begins an ordinary life, unaware of what is coming.

# Act Two

Everything changes. A call to adventure arrives and cannot be ignored.

# Act Three

The hero returns transformed, having faced the darkness and survived.
"""


def _get_scenes():
    scenes = parse_script_text(SCRIPT)
    return assign_timestamps_no_audio(scenes)


def test_csv_creates_file():
    scenes = _get_scenes()
    with tempfile.TemporaryDirectory() as tmp:
        path = to_csv(scenes, Path(tmp) / "scenes.csv")
        assert path.exists()


def test_csv_has_correct_rows():
    scenes = _get_scenes()
    with tempfile.TemporaryDirectory() as tmp:
        path = to_csv(scenes, Path(tmp) / "scenes.csv")
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == len(scenes)


def test_csv_headers():
    scenes = _get_scenes()
    with tempfile.TemporaryDirectory() as tmp:
        path = to_csv(scenes, Path(tmp) / "scenes.csv")
        with open(path, newline="", encoding="utf-8") as f:
            headers = next(csv.reader(f))
        expected = ["index", "title", "text", "start_time", "end_time",
                    "image_prompt", "image_path"]
        assert headers == expected


def test_json_creates_file():
    scenes = _get_scenes()
    with tempfile.TemporaryDirectory() as tmp:
        path = to_json(scenes, Path(tmp) / "scenes.json")
        assert path.exists()


def test_json_is_valid():
    scenes = _get_scenes()
    with tempfile.TemporaryDirectory() as tmp:
        path = to_json(scenes, Path(tmp) / "scenes.json")
        data = load_json(path)
        assert isinstance(data, list)
        assert len(data) == len(scenes)
        assert data[0]["index"] == 1


def test_roundtrip_timestamps():
    scenes = _get_scenes()
    with tempfile.TemporaryDirectory() as tmp:
        path = to_json(scenes, Path(tmp) / "scenes.json")
        data = load_json(path)
        for original, loaded in zip(scenes, data):
            assert abs(original.start_time - loaded["start_time"]) < 0.01
            assert abs(original.end_time - loaded["end_time"]) < 0.01

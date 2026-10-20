"""
Tests for timestamp_mapper – no audio file needed (uses no-audio fallback).
"""

import pytest
from ai_visualizer.scene_parser import parse_script_text
from ai_visualizer.timestamp_mapper import assign_timestamps_no_audio


SCRIPT = """
The first act sets the stage.

The second act raises the stakes. Things get complicated for everyone involved.

The third act resolves everything in a satisfying way.
"""


def _get_scenes():
    return parse_script_text(SCRIPT)


def test_timestamps_are_monotonic():
    scenes = assign_timestamps_no_audio(_get_scenes())
    for i in range(len(scenes) - 1):
        assert scenes[i].end_time <= scenes[i + 1].start_time + 0.01


def test_start_time_first_scene():
    scenes = assign_timestamps_no_audio(_get_scenes())
    assert scenes[0].start_time == 0.0


def test_end_time_increases():
    scenes = assign_timestamps_no_audio(_get_scenes())
    for s in scenes:
        assert s.end_time > s.start_time


def test_longer_scenes_get_more_time():
    scenes = assign_timestamps_no_audio(_get_scenes())
    # Scene 2 has more words than scene 1 and should be longer
    durations = [s.end_time - s.start_time for s in scenes]
    assert durations[1] > durations[0], "Longer text should get more time"


def test_custom_wpm():
    scenes_slow = assign_timestamps_no_audio(_get_scenes(), words_per_minute=60)
    scenes_fast = assign_timestamps_no_audio(parse_script_text(SCRIPT), words_per_minute=200)
    # Slower WPM → longer total duration
    total_slow = scenes_slow[-1].end_time
    total_fast = scenes_fast[-1].end_time
    assert total_slow > total_fast

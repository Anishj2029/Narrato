"""
Tests for scene_parser – run without any API keys or audio files.
"""

import pytest
from ai_visualizer.scene_parser import parse_script_text, Scene


SIMPLE_SCRIPT = """
# The Beginning

The universe started with a bang. Everything came from a single point of infinite density.

## The First Stars

After millions of years of darkness, the first stars ignited. Their light crossed the void for the first time.

## Life Emerges

On a small rocky planet, chemistry became biology. Simple molecules assembled into something that could replicate itself.
"""

PARAGRAPH_SCRIPT = """
The morning fog rolled in from the sea, covering the harbor in a soft grey blanket.
Fishermen prepared their boats in near silence, each knowing the ritual by heart.

By midday the fog had lifted. The water sparkled under a full sun.
Children ran along the dock, daring each other to jump in.

Evening brought the boats back, heavy with the day's catch.
The smell of salt and fish filled the streets as families gathered for dinner.
"""

SOFT_CAP_SCRIPT = " ".join([f"Sentence number {i}." for i in range(20)])


def test_marker_based_scenes():
    scenes = parse_script_text(SIMPLE_SCRIPT)
    assert len(scenes) == 3, f"Expected 3 scenes, got {len(scenes)}"
    assert scenes[0].index == 1
    assert scenes[1].index == 2
    assert scenes[2].index == 3


def test_scene_titles_are_short():
    scenes = parse_script_text(SIMPLE_SCRIPT)
    for s in scenes:
        words = s.title.replace("…", "").split()
        assert len(words) <= 7, f"Title too long: {s.title!r}"


def test_paragraph_grouping():
    scenes = parse_script_text(PARAGRAPH_SCRIPT)
    # Three blank-line separated paragraphs → 3 scenes
    assert len(scenes) == 3, f"Expected 3 paragraph scenes, got {len(scenes)}"


def test_soft_cap_splits_long_runs():
    scenes = parse_script_text(SOFT_CAP_SCRIPT)
    # 20 sentences should be split into multiple scenes
    assert len(scenes) > 1, "Long run of sentences should be split"


def test_scene_text_not_empty():
    scenes = parse_script_text(SIMPLE_SCRIPT)
    for s in scenes:
        assert s.text.strip(), f"Scene {s.index} has empty text"


def test_to_dict_keys():
    scenes = parse_script_text(SIMPLE_SCRIPT)
    d = scenes[0].to_dict()
    expected = {"index", "title", "text", "start_time", "end_time",
                "image_prompt", "image_path"}
    assert set(d.keys()) == expected

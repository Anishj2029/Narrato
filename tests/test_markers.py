"""Tests for the independent manual visual-marker data model."""

import csv
import json
import math

import pytest

from ai_visualizer.markers import (
    VisualMarker,
    create_marker,
    delete_marker,
    edit_marker,
    export_markers_csv,
    export_markers_json,
    load_markers,
    save_markers,
    validate_markers,
)


def test_create_marker_appends_with_unique_stable_id():
    markers = []

    first = create_marker(markers, 1.25, image_prompt="First visual")
    second = create_marker(markers, 1.25, image_prompt="Layered visual")

    assert markers == [first, second]
    assert first.marker_id
    assert second.marker_id
    assert first.marker_id != second.marker_id
    assert first.end_time is None


def test_edit_marker_preserves_id_and_updates_values():
    markers = [VisualMarker(start_time=2, end_time=5, image_prompt="Old")]
    marker_id = markers[0].marker_id

    updated = edit_marker(
        markers,
        marker_id,
        start_time=3,
        end_time=6,
        image_prompt="New",
        image_path="images/marker.png",
    )

    assert updated is markers[0]
    assert updated.marker_id == marker_id
    assert updated.start_time == 3
    assert updated.end_time == 6
    assert updated.image_prompt == "New"
    assert updated.image_path == "images/marker.png"


def test_invalid_edit_does_not_change_marker():
    markers = [VisualMarker(start_time=2, end_time=5, image_prompt="Original")]
    before = markers[0].to_dict()

    with pytest.raises(ValueError, match="greater than"):
        edit_marker(markers, markers[0].marker_id, end_time=2)

    assert markers[0].to_dict() == before


def test_edit_cannot_change_stable_id():
    markers = [VisualMarker(start_time=2)]

    with pytest.raises(ValueError, match="Cannot edit"):
        edit_marker(markers, markers[0].marker_id, marker_id="replacement")


def test_delete_marker_removes_and_returns_marker():
    markers = [VisualMarker(start_time=1), VisualMarker(start_time=3)]
    deleted = markers[0]

    result = delete_marker(markers, deleted.marker_id)

    assert result is deleted
    assert markers == [markers[0]]
    assert all(marker.marker_id != deleted.marker_id for marker in markers)


@pytest.mark.parametrize(
    ("start_time", "end_time", "message"),
    [
        (-0.01, None, "non-negative"),
        (math.inf, None, "finite"),
        (math.nan, None, "finite"),
        (True, None, "non-negative"),
        (2, 2, "greater than"),
        (3, 2, "greater than"),
        (1, math.inf, "finite"),
        (1, math.nan, "finite"),
        (1, True, "finite"),
    ],
)
def test_invalid_timestamps_are_rejected(start_time, end_time, message):
    with pytest.raises(ValueError, match=message):
        VisualMarker(start_time=start_time, end_time=end_time)


def test_overlapping_markers_are_allowed():
    markers = [
        VisualMarker(start_time=1, end_time=5),
        VisualMarker(start_time=3, end_time=7),
    ]

    validate_markers(markers)


def test_duplicate_marker_ids_are_rejected():
    marker_id = "same-id"
    markers = [
        VisualMarker(start_time=1, marker_id=marker_id),
        VisualMarker(start_time=2, marker_id=marker_id),
    ]

    with pytest.raises(ValueError, match="Duplicate marker_id"):
        validate_markers(markers)


def test_save_load_round_trip_preserves_all_marker_data(tmp_path):
    markers = [
        VisualMarker(
            marker_id="opening-shot",
            start_time=1.234567,
            end_time=4.5,
            image_prompt="A carefully chosen prompt",
            image_path="outputs/custom-image.png",
        ),
        VisualMarker(
            marker_id="continuing-overlay",
            start_time=3.25,
            image_prompt="An overlapping marker",
        ),
    ]
    path = tmp_path / "project" / "markers.json"

    assert save_markers(path, markers) == path
    loaded = load_markers(path)

    assert [marker.to_dict() for marker in loaded] == [
        marker.to_dict() for marker in markers
    ]


def test_load_markers_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_markers(tmp_path / "missing-markers.json")


def test_save_markers_does_not_modify_scene_plan(tmp_path):
    scene_plan = tmp_path / "scene_plan.json"
    original_scene_plan = '[{"index":1,"image_prompt":"automatic"}]\n'
    scene_plan.write_text(original_scene_plan, encoding="utf-8")

    save_markers(tmp_path / "markers.json", [VisualMarker(start_time=2)])

    assert scene_plan.read_text(encoding="utf-8") == original_scene_plan


def test_json_export_contains_marker_fields(tmp_path):
    markers = [
        VisualMarker(
            marker_id="marker-json",
            start_time=0,
            image_prompt="Prompt",
            image_path="frame.png",
        )
    ]
    path = export_markers_json(markers, tmp_path / "exports" / "markers.json")

    data = json.loads(path.read_text(encoding="utf-8"))

    assert data == [markers[0].to_dict()]
    assert data[0]["end_time"] is None


def test_csv_export_contains_marker_fields_and_optional_values(tmp_path):
    markers = [
        VisualMarker(
            marker_id="marker-csv",
            start_time=0.5,
            end_time=2.75,
            image_prompt="Prompt, with comma",
            image_path="frame.png",
        ),
        VisualMarker(marker_id="marker-open-ended", start_time=4),
    ]
    path = export_markers_csv(markers, tmp_path / "exports" / "markers.csv")

    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    assert reader.fieldnames == [
        "marker_id", "start_time", "end_time", "image_prompt", "image_path",
    ]
    assert len(rows) == 2
    assert rows[0] == {
        "marker_id": "marker-csv",
        "start_time": "0.5",
        "end_time": "2.75",
        "image_prompt": "Prompt, with comma",
        "image_path": "frame.png",
    }
    assert rows[1]["marker_id"] == "marker-open-ended"
    assert rows[1]["end_time"] == ""
    assert rows[1]["image_path"] == ""


def test_save_rejects_invalid_collection_without_replacing_existing_file(tmp_path):
    path = tmp_path / "markers.json"
    original_content = '[{"marker_id":"existing"}]\n'
    path.write_text(original_content, encoding="utf-8")
    duplicate_markers = [
        VisualMarker(start_time=0, marker_id="duplicate"),
        VisualMarker(start_time=1, marker_id="duplicate"),
    ]

    with pytest.raises(ValueError, match="Duplicate marker_id"):
        save_markers(path, duplicate_markers)

    assert path.read_text(encoding="utf-8") == original_content

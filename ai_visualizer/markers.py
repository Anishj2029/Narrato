"""Data model, persistence, and exports for manual visual markers."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional


@dataclass
class VisualMarker:
    """A manually authored visual cue on the narration timeline."""

    start_time: float
    marker_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    end_time: Optional[float] = None
    image_prompt: str = ""
    image_path: Optional[str] = None

    def __post_init__(self) -> None:
        validate_marker(self)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_marker(marker: VisualMarker) -> None:
    """Raise ValueError when a marker has invalid fields or timestamps."""
    if not isinstance(marker.marker_id, str) or not marker.marker_id.strip():
        raise ValueError("marker_id must be a non-empty string.")
    if (
        not isinstance(marker.start_time, (int, float))
        or isinstance(marker.start_time, bool)
        or not math.isfinite(marker.start_time)
        or marker.start_time < 0
    ):
        raise ValueError("start_time must be a finite, non-negative number.")
    if marker.end_time is not None:
        if (
            not isinstance(marker.end_time, (int, float))
            or isinstance(marker.end_time, bool)
            or not math.isfinite(marker.end_time)
        ):
            raise ValueError("end_time must be a finite number or None.")
        if marker.end_time <= marker.start_time:
            raise ValueError("end_time must be greater than start_time.")
    if not isinstance(marker.image_prompt, str):
        raise ValueError("image_prompt must be a string.")
    if marker.image_path is not None and not isinstance(marker.image_path, str):
        raise ValueError("image_path must be a string or None.")


def validate_markers(markers: Iterable[VisualMarker]) -> None:
    """Validate each marker and ensure IDs are unique; overlaps are allowed."""
    seen_ids: set[str] = set()
    for marker in markers:
        if not isinstance(marker, VisualMarker):
            raise ValueError("markers must contain only VisualMarker instances.")
        validate_marker(marker)
        if marker.marker_id in seen_ids:
            raise ValueError(f"Duplicate marker_id: {marker.marker_id}")
        seen_ids.add(marker.marker_id)


def create_marker(
    markers: list[VisualMarker],
    start_time: float,
    end_time: Optional[float] = None,
    image_prompt: str = "",
    image_path: Optional[str] = None,
) -> VisualMarker:
    """Create and append a marker with a newly generated stable ID."""
    marker = VisualMarker(
        start_time=start_time,
        end_time=end_time,
        image_prompt=image_prompt,
        image_path=image_path,
    )
    validate_markers([*markers, marker])
    markers.append(marker)
    return marker


def edit_marker(
    markers: list[VisualMarker], target_marker_id: str, **changes: Any
) -> VisualMarker:
    """Edit marker fields in place, retaining its stable ID."""
    allowed_fields = {"start_time", "end_time", "image_prompt", "image_path"}
    unexpected = set(changes) - allowed_fields
    if unexpected:
        raise ValueError(f"Cannot edit marker field(s): {sorted(unexpected)}")

    marker = _find_marker(markers, target_marker_id)
    updated = VisualMarker(
        marker_id=marker.marker_id,
        start_time=changes.get("start_time", marker.start_time),
        end_time=changes.get("end_time", marker.end_time),
        image_prompt=changes.get("image_prompt", marker.image_prompt),
        image_path=changes.get("image_path", marker.image_path),
    )
    for field_name in allowed_fields:
        setattr(marker, field_name, getattr(updated, field_name))
    return marker


def delete_marker(markers: list[VisualMarker], marker_id: str) -> VisualMarker:
    """Remove and return a marker by ID."""
    marker = _find_marker(markers, marker_id)
    markers.remove(marker)
    return marker


def _find_marker(markers: list[VisualMarker], marker_id: str) -> VisualMarker:
    for marker in markers:
        if marker.marker_id == marker_id:
            return marker
    raise KeyError(f"Marker not found: {marker_id}")


def _markers_from_data(data: Any) -> list[VisualMarker]:
    if not isinstance(data, list):
        raise ValueError("Marker data must be a JSON array.")
    markers = []
    allowed_fields = set(VisualMarker.__dataclass_fields__)
    for position, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Marker at position {position} must be an object.")
        if "marker_id" not in item:
            raise ValueError(f"Marker at position {position} is missing 'marker_id'.")
        if set(item) - allowed_fields:
            raise ValueError(f"Marker at position {position} has unknown fields.")
        try:
            markers.append(VisualMarker(**item))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid marker at position {position}: {exc}") from exc
    validate_markers(markers)
    return markers


def load_markers(path: str | Path) -> list[VisualMarker]:
    """Load markers from their dedicated project JSON file."""
    marker_path = Path(path)
    data = json.loads(marker_path.read_text(encoding="utf-8"))
    return _markers_from_data(data)


def _atomic_write(path: Path, write_content: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            write_content(temporary_file)
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return path


def save_markers(path: str | Path, markers: Iterable[VisualMarker]) -> Path:
    """Atomically save markers to a dedicated JSON file."""
    marker_list = list(markers)
    validate_markers(marker_list)
    serialized = json.dumps(
        [marker.to_dict() for marker in marker_list], indent=2, ensure_ascii=False
    )
    return _atomic_write(
        Path(path), lambda file: file.write(serialized)
    )


def export_markers_json(
    markers: Iterable[VisualMarker], output_path: str | Path
) -> Path:
    """Export manual markers as a JSON array without changing scene exports."""
    return save_markers(output_path, markers)


def export_markers_csv(
    markers: Iterable[VisualMarker], output_path: str | Path
) -> Path:
    """Atomically export manual markers as CSV."""
    marker_list = list(markers)
    validate_markers(marker_list)
    fieldnames = [
        "marker_id", "start_time", "end_time", "image_prompt", "image_path",
    ]

    def write_csv(file: Any) -> None:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for marker in marker_list:
            writer.writerow(marker.to_dict())

    return _atomic_write(Path(output_path), write_csv)

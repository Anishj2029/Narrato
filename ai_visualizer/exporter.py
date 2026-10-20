"""
exporter.py
-----------
Save scenes to CSV and JSON.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List

from .scene_parser import Scene


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def to_csv(scenes: List[Scene], output_path: str | Path) -> Path:
    """Write scenes to a CSV file. Returns the resolved path."""
    path = Path(output_path)
    _ensure_dir(path)

    fieldnames = ["index", "title", "text", "start_time", "end_time",
                  "image_prompt", "image_path"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for scene in scenes:
            writer.writerow(scene.to_dict())

    return path


def to_json(scenes: List[Scene], output_path: str | Path) -> Path:
    """Write scenes to a JSON file. Returns the resolved path."""
    path = Path(output_path)
    _ensure_dir(path)

    data = [s.to_dict() for s in scenes]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return path


def load_json(input_path: str | Path) -> List[dict]:
    """Load previously exported scenes from JSON."""
    path = Path(input_path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)

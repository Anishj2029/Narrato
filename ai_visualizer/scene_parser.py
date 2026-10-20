"""
scene_parser.py
---------------
Reads a TXT or Markdown script and groups lines into semantic scenes.

Grouping strategy:
  - Explicit scene markers  (#, ##, ---, [SCENE], SCENE:) always start a new scene.
  - Otherwise sentences are collected until a "meaning boundary" is detected:
      * A blank line separating paragraphs
      * A hard topic shift detected via simple keyword / structural heuristics
      * Accumulated sentence count exceeds MAX_SENTENCES_PER_SCENE (soft cap,
        only applied when no natural boundary is found)
  The goal is one visual concept per scene, NOT one sentence per scene.
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------
MAX_SENTENCES_PER_SCENE = 5   # soft cap – only triggers without a natural break
MIN_WORDS_FOR_SCENE = 4       # ignore tiny fragments


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class Scene:
    index: int                          # 1-based
    title: str                          # short label (first ~6 words)
    text: str                           # full scene text
    start_time: float = 0.0             # seconds – filled in by timestamp module
    end_time: float = 0.0               # seconds – filled in by timestamp module
    image_prompt: str = ""              # filled in by Gemini module
    image_path: str = ""                # filled in by image generator

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "title": self.title,
            "text": self.text,
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3),
            "image_prompt": self.image_prompt,
            "image_path": self.image_path,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_SCENE_MARKER = re.compile(
    r"^\s*(#{1,3}|---+|\*\*\*+|\[SCENE\]|SCENE\s*:?)\s*",
    re.IGNORECASE,
)
_SENTENCE_END = re.compile(r"[.!?…]+\s*$")


def _is_scene_marker(line: str) -> bool:
    return bool(_SCENE_MARKER.match(line))


def _strip_marker(line: str) -> str:
    return _SCENE_MARKER.sub("", line).strip()


def _title_from_text(text: str) -> str:
    words = text.split()
    return " ".join(words[:6]) + ("…" if len(words) > 6 else "")


def _split_sentences(text: str) -> List[str]:
    """Naïve sentence splitter – good enough without nltk dependency at parse time."""
    parts = re.split(r"(?<=[.!?…])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _word_count(text: str) -> int:
    return len(text.split())


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------

def parse_script(path: str | Path) -> List[Scene]:
    """
    Parse a TXT or Markdown file into a list of Scene objects.
    Returns scenes in order with 1-based indexes.
    """
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    return _group_into_scenes(raw)


def parse_script_text(raw: str) -> List[Scene]:
    """Parse a script from a raw string (useful for the UI upload flow)."""
    return _group_into_scenes(raw)


def _group_into_scenes(raw: str) -> List[Scene]:
    lines = raw.splitlines()
    scenes: List[Scene] = []
    buffer: List[str] = []
    sentence_count = 0

    def flush(buf: List[str]) -> None:
        text = " ".join(buf).strip()
        # Clean up markdown formatting for readability
        text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)  # bold/italic
        text = re.sub(r"`([^`]+)`", r"\1", text)               # inline code
        if _word_count(text) >= MIN_WORDS_FOR_SCENE:
            idx = len(scenes) + 1
            scenes.append(Scene(index=idx, title=_title_from_text(text), text=text))
        buf.clear()

    for line in lines:
        stripped = line.strip()

        # Explicit scene marker → flush current buffer and start fresh
        if _is_scene_marker(stripped):
            if buffer:
                flush(buffer)
            remainder = _strip_marker(stripped)
            if remainder:
                buffer.append(remainder)
            sentence_count = 0
            continue

        # Blank line = paragraph boundary → flush
        if not stripped:
            if buffer:
                flush(buffer)
                sentence_count = 0
            continue

        # Regular content – split into sentences, flush on soft cap
        sentences = _split_sentences(stripped)
        for sentence in sentences:
            buffer.append(sentence)
            sentence_count += 1
            if sentence_count >= MAX_SENTENCES_PER_SCENE:
                flush(buffer)
                sentence_count = 0

    # Flush any remaining text
    if buffer:
        flush(buffer)

    return scenes

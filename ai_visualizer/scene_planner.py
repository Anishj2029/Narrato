"""
scene_planner.py
----------------
Phase 1.5 — Convert a Whisper transcript into a scene plan with
accurate, word-level-grounded timestamps and image prompts.

This module is INDEPENDENT of the script-parsing pipeline.  It reads
transcript_segments.json (produced by transcribe.py / faster-whisper)
and groups Whisper segments into meaningful visual scenes.

Grouping strategy (deterministic — no LLM required)
----------------------------------------------------
1. Hard boundary: a silence gap >= PAUSE_HARD_THRESHOLD seconds between
   consecutive Whisper segments forces a new scene.
2. Soft cap:      if a scene accumulates >= TARGET_SCENE_WORDS words,
   the *next* hard-ish pause (>= PAUSE_SOFT_THRESHOLD) also breaks.
3. Minimum scene: scenes shorter than MIN_SCENE_DURATION seconds are
   merged forward into the next scene (avoids 1-sentence scenes).
4. Final scene:   the last scene always absorbs whatever is left,
   regardless of the minimum-duration rule.

All timestamps come directly from the Whisper word/segment data.
They are NOT re-estimated proportionally.

Optional LLM enhancement
-------------------------
If a Gemini key is configured (.env), call enhance_scenes() on the
resulting Scene objects just like the rest of the pipeline does.
That step is handled externally — this module only produces the plan.

Fallback descriptions
---------------------
When Gemini is not available, image_prompt is generated deterministically
from the scene text using keyword extraction.  All fallback prompts are
prefixed with "[FALLBACK] " so they can be filtered or replaced later.

Output fields per scene (extends the existing Scene dataclass)
--------------------------------------------------------------
index, title, text, start_time, end_time,     — standard Scene fields
image_prompt, image_path,                      — standard Scene fields
segment_ids, word_count,                       — planner extras (stored in JSON)
description, filename_suggestion               — planner extras (stored in JSON)
"""

from __future__ import annotations

import json
import re
import string
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

from .scene_parser import parse_script_text
from .timestamp_mapper import assign_timestamps

# ---------------------------------------------------------------------------
# Configurable constants
# ---------------------------------------------------------------------------
TARGET_SCENE_WORDS: int = 70      # aim for scenes of roughly this many words
MIN_SCENE_DURATION: float = 15.0  # merge scenes shorter than this (seconds)
PAUSE_HARD_THRESHOLD: float = 0.8 # silence >= this ALWAYS breaks a scene
PAUSE_SOFT_THRESHOLD: float = 0.5 # silence >= this breaks when over word cap
MIN_WORDS_FOR_SCENE: int = 8      # drop scenes with fewer words than this


# ---------------------------------------------------------------------------
# Extended scene dataclass
# ---------------------------------------------------------------------------
@dataclass
class PlannedScene:
    index: int
    title: str
    text: str
    start_time: float
    end_time: float
    image_prompt: str = ""
    image_path: str = ""
    # Planner-specific metadata
    description: str = ""
    filename_suggestion: str = ""
    segment_ids: List[int] = field(default_factory=list)
    word_count: int = 0

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Full dict for JSON export (superset of Scene.to_dict)."""
        return {
            "index": self.index,
            "title": self.title,
            "text": self.text,
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3),
            "image_prompt": self.image_prompt,
            "image_path": self.image_path,
            "description": self.description,
            "filename_suggestion": self.filename_suggestion,
            "segment_ids": self.segment_ids,
            "word_count": self.word_count,
        }

    def to_csv_dict(self) -> Dict[str, Any]:
        """Flat dict for CSV — only the fields the existing exporter uses,
        plus the planner extras as extra columns."""
        return {
            "index": self.index,
            "title": self.title,
            "text": self.text,
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3),
            "image_prompt": self.image_prompt,
            "image_path": self.image_path,
            "description": self.description,
            "filename_suggestion": self.filename_suggestion,
            "word_count": self.word_count,
            "segment_ids": ";".join(str(x) for x in self.segment_ids),
        }

    # Compatibility shim so existing code that calls .to_dict() on a
    # generic Scene still works:
    def to_scene_dict(self) -> Dict[str, Any]:
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
# Transcript loading
# ---------------------------------------------------------------------------

def load_transcript(path: str | Path) -> Dict[str, Any]:
    """Load transcript_segments.json produced by transcribe.py."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Transcript not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if "segments" not in data:
        raise ValueError("JSON does not contain a 'segments' key.")
    return data


# ---------------------------------------------------------------------------
# Scene boundary detection
# ---------------------------------------------------------------------------

def _pause_between(seg_a: Dict, seg_b: Dict) -> float:
    """Return the silence gap in seconds between two adjacent segments."""
    return round(seg_b["start"] - seg_a["end"], 3)


def _compute_boundaries(
    segments: List[Dict],
    target_words: int = TARGET_SCENE_WORDS,
    pause_hard: float = PAUSE_HARD_THRESHOLD,
    pause_soft: float = PAUSE_SOFT_THRESHOLD,
) -> List[int]:
    """
    Return a list of segment indexes at which a new scene should start.
    Index 0 is always included (the first scene starts at segment 0).
    """
    boundaries = [0]
    cumulative_words = 0
    over_cap = False

    for i in range(1, len(segments)):
        prev = segments[i - 1]
        curr = segments[i]
        pause = _pause_between(prev, curr)
        seg_words = len(curr["text"].split())

        cumulative_words += len(prev["text"].split())

        if cumulative_words >= target_words:
            over_cap = True

        # Hard break regardless of word count
        if pause >= pause_hard:
            boundaries.append(i)
            cumulative_words = 0
            over_cap = False
            continue

        # Soft break: over word cap AND a notable pause
        if over_cap and pause >= pause_soft:
            boundaries.append(i)
            cumulative_words = 0
            over_cap = False
            continue

    return boundaries


# ---------------------------------------------------------------------------
# Scene assembly
# ---------------------------------------------------------------------------

def _assemble_scenes(
    segments: List[Dict],
    boundaries: List[int],
    audio_duration: float,
) -> List[PlannedScene]:
    """
    Group segments into PlannedScene objects according to boundary indexes.
    start_time / end_time come directly from Whisper segment timestamps.
    """
    scenes: List[PlannedScene] = []
    boundary_set = set(boundaries)

    for b_idx, start_seg_i in enumerate(boundaries):
        end_seg_i = boundaries[b_idx + 1] - 1 if b_idx + 1 < len(boundaries) else len(segments) - 1
        group = segments[start_seg_i: end_seg_i + 1]

        if not group:
            continue

        # Timestamps: first word start → last word end (word-level precision)
        start_time = _word_start(group[0])
        end_time = _word_end(group[-1])

        # Clamp last scene to audio duration
        if b_idx == len(boundaries) - 1:
            end_time = audio_duration

        text = " ".join(seg["text"].strip() for seg in group)
        word_count = sum(len(seg["text"].split()) for seg in group)
        seg_ids = [seg["id"] for seg in group]
        title = _make_title(text)

        scene = PlannedScene(
            index=len(scenes) + 1,
            title=title,
            text=text,
            start_time=start_time,
            end_time=end_time,
            word_count=word_count,
            segment_ids=seg_ids,
        )
        scenes.append(scene)

    return scenes


def _word_start(segment: Dict) -> float:
    """Get the start time of the first word in a segment (word-level precision)."""
    words = segment.get("words", [])
    if words:
        return words[0]["start"]
    return segment["start"]


def _word_end(segment: Dict) -> float:
    """Get the end time of the last word in a segment (word-level precision)."""
    words = segment.get("words", [])
    if words:
        return words[-1]["end"]
    return segment["end"]


# ---------------------------------------------------------------------------
# Post-processing: merge short scenes
# ---------------------------------------------------------------------------

def _merge_short_scenes(
    scenes: List[PlannedScene],
    min_duration: float = MIN_SCENE_DURATION,
    min_words: int = MIN_WORDS_FOR_SCENE,
) -> List[PlannedScene]:
    """
    Merge scenes that are too short into the following adequate scene.

    Algorithm:
    - Walk forward collecting scenes into a "pending" accumulator.
    - As soon as the accumulator reaches min_duration AND min_words,
      flush it as one scene and start a new accumulator.
    - Any leftover accumulator at the end is flushed as-is (last scene
      is never dropped, even if it doesn't meet the minimums).
    """
    if not scenes:
        return scenes

    result: List[PlannedScene] = []

    # Accumulator: list of scenes to be merged into one
    acc: List[PlannedScene] = []

    for scene in scenes:
        acc.append(scene)
        acc_duration = acc[-1].end_time - acc[0].start_time
        acc_words = sum(s.word_count for s in acc)

        if acc_duration >= min_duration and acc_words >= min_words:
            result.append(_fuse(acc))
            acc = []

    # Flush any remaining accumulator (always becomes a scene)
    if acc:
        if result:
            # Merge the leftover tail into the last flushed scene
            last = result[-1]
            tail = _fuse(acc)
            fused = PlannedScene(
                index=0,
                title=last.title,  # keep the leading scene's title
                text=last.text + " " + tail.text,
                start_time=last.start_time,
                end_time=tail.end_time,
                word_count=last.word_count + tail.word_count,
                segment_ids=last.segment_ids + tail.segment_ids,
            )
            result[-1] = fused
        else:
            result.append(_fuse(acc))

    # Re-index
    for i, s in enumerate(result):
        s.index = i + 1
        s.filename_suggestion = f"scene_{s.index:03d}.png"

    return result


def _fuse(acc: List[PlannedScene]) -> PlannedScene:
    """Combine a list of PlannedScene objects into one."""
    if len(acc) == 1:
        return acc[0]
    combined_text = " ".join(s.text for s in acc)
    return PlannedScene(
        index=0,
        title=_make_title(acc[0].text),   # title from first scene in group
        text=combined_text,
        start_time=acc[0].start_time,
        end_time=acc[-1].end_time,
        word_count=sum(s.word_count for s in acc),
        segment_ids=[sid for s in acc for sid in s.segment_ids],
    )


# ---------------------------------------------------------------------------
# Title and description generation (deterministic)
# ---------------------------------------------------------------------------

_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "as", "by", "from", "is", "it", "its", "this", "that",
    "these", "those", "be", "been", "are", "was", "were", "have", "has",
    "had", "do", "does", "did", "will", "would", "could", "should", "may",
    "might", "can", "not", "no", "so", "if", "we", "you", "i", "me", "my",
    "our", "your", "they", "them", "their", "he", "she", "his", "her",
    "who", "what", "when", "where", "why", "how", "all", "some", "more",
    "also", "just", "up", "about", "now", "then", "even", "still", "only",
}

# High-value content keywords for this specific transcript's domain
_DOMAIN_KEYWORDS = {
    "family", "friends", "partner", "relationship", "relationships",
    "childhood", "attachment", "personality", "behavior", "identity",
    "love", "honest", "vulnerable", "trust", "influence", "emotions",
    "conflict", "disagreement", "growth", "patterns", "self",
    "person", "people", "child", "boy", "classroom", "quiet", "confident",
    "psychologist", "bowlby", "ainsworth", "responsiveness",
}


def _make_title(text: str) -> str:
    """Extract a short title from the first meaningful sentence."""
    clean = re.sub(r"[^\w\s']", " ", text.lower())
    # First sentence
    sentences = re.split(r"[.!?]", text)
    first = sentences[0].strip() if sentences else text
    words = first.split()
    # Take first 6-8 words, skip leading function words
    start = 0
    while start < len(words) and words[start].lower() in {"and", "but", "or",
                                                            "now", "so", "yet",
                                                            "because", "also"}:
        start += 1
    title_words = words[start: start + 7]
    title = " ".join(title_words)
    if len(words) > start + 7:
        title += "…"
    return title if title else "Scene"


def _extract_keywords(text: str, n: int = 6) -> List[str]:
    """Return the n most content-rich words from text."""
    tokens = re.findall(r"[a-z']+", text.lower())
    # Prioritise domain keywords, then long non-stopword tokens
    domain = [t for t in tokens if t in _DOMAIN_KEYWORDS]
    content = [t for t in tokens if t not in _STOP_WORDS
               and len(t) > 4 and t not in domain]
    combined = domain + content
    # Deduplicate preserving order
    seen = set()
    unique = []
    for w in combined:
        if w not in seen:
            seen.add(w)
            unique.append(w)
    return unique[:n]


def _fallback_image_prompt(scene: PlannedScene) -> str:
    """
    Deterministic image prompt from scene text.
    Always prefixed with '[FALLBACK] ' so it can be identified and replaced.
    """
    keywords = _extract_keywords(scene.text)
    kw_str = ", ".join(keywords) if keywords else scene.title
    # Pull the most vivid sentence (longest one that isn't a question)
    sentences = [s.strip() for s in re.split(r"[.!?]", scene.text)
                 if s.strip() and not s.strip().endswith("?")]
    if sentences:
        vivid = max(sentences, key=len)[:120]
    else:
        vivid = scene.text[:120]
    return (
        f"[FALLBACK] Cinematic illustration: {vivid}. "
        f"Key themes: {kw_str}. "
        f"Warm, thoughtful, documentary style."
    )


def _fallback_description(scene: PlannedScene) -> str:
    """One-sentence visual description from keywords."""
    keywords = _extract_keywords(scene.text, n=4)
    kw = ", ".join(keywords) if keywords else "human connection"
    dur = round(scene.end_time - scene.start_time, 1)
    return (
        f"[FALLBACK] A {dur}s scene about {kw}. "
        f"Characters maintain visual consistency with adjacent scenes."
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def plan_scenes(
    transcript_path: str | Path = "transcript_segments.json",
    target_words: int = TARGET_SCENE_WORDS,
    min_duration: float = MIN_SCENE_DURATION,
    pause_hard: float = PAUSE_HARD_THRESHOLD,
    pause_soft: float = PAUSE_SOFT_THRESHOLD,
    min_words: int = MIN_WORDS_FOR_SCENE,
) -> List[PlannedScene]:
    """
    Read a Whisper transcript and return a list of PlannedScene objects.

    All timestamps are grounded in Whisper word-level data.
    Image prompts use the deterministic fallback (no LLM calls here).
    Call ai_visualizer.gemini_client.enhance_scenes() afterwards to
    replace fallback prompts with Gemini-generated ones.

    Parameters
    ----------
    transcript_path : path to transcript_segments.json
    target_words    : soft word cap per scene (default 80)
    min_duration    : scenes shorter than this are merged (default 15s)
    pause_hard      : gap >= this always breaks a scene (default 1.5s)
    pause_soft      : gap >= this breaks when over word cap (default 0.7s)
    min_words       : scenes with fewer words are dropped/merged (default 8)
    """
    data = load_transcript(transcript_path)
    segments = data["segments"]
    audio_duration = data["audio_duration_seconds"]

    if not segments:
        return []

    # Step 1: find break points
    boundaries = _compute_boundaries(
        segments,
        target_words=target_words,
        pause_hard=pause_hard,
        pause_soft=pause_soft,
    )

    # Step 2: assemble raw scenes
    scenes = _assemble_scenes(segments, boundaries, audio_duration)

    # Step 3: merge short scenes
    scenes = _merge_short_scenes(scenes, min_duration=min_duration, min_words=min_words)

    # Step 4: add fallback descriptions and prompts
    for scene in scenes:
        scene.description = _fallback_description(scene)
        scene.image_prompt = _fallback_image_prompt(scene)
        scene.filename_suggestion = f"scene_{scene.index:03d}.png"

    return scenes


def plan_script_scenes(
    script_text: str, audio_path: str | Path
) -> List[PlannedScene]:
    """Create an estimated-time visual plan from a Narrato project script."""
    parsed_scenes = parse_script_text(script_text)
    if not parsed_scenes:
        raise ValueError("No readable script scenes were found in this project.")
    assign_timestamps(parsed_scenes, audio_path)

    planned_scenes = []
    for scene in parsed_scenes:
        planned = PlannedScene(
            index=scene.index,
            title=scene.title,
            text=scene.text,
            start_time=scene.start_time,
            end_time=scene.end_time,
            word_count=len(scene.text.split()),
            filename_suggestion=f"scene_{scene.index:03d}.png",
        )
        planned.description = _fallback_description(planned)
        planned.image_prompt = _fallback_image_prompt(planned)
        planned_scenes.append(planned)
    return planned_scenes


# ---------------------------------------------------------------------------
# Export helpers (mirrors exporter.py for PlannedScene)
# ---------------------------------------------------------------------------

import csv as _csv


def save_json(scenes: List[PlannedScene], path: str | Path) -> Path:
    """Save scene plan as JSON. Includes all planner-specific fields."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = [s.to_dict() for s in scenes]
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def save_csv(scenes: List[PlannedScene], path: str | Path) -> Path:
    """Save scene plan as CSV. Flat representation including planner extras."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "index", "title", "text", "start_time", "end_time",
        "image_prompt", "image_path", "description",
        "filename_suggestion", "word_count", "segment_ids",
    ]
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for scene in scenes:
            writer.writerow(scene.to_csv_dict())
    return out


def coverage_report(scenes: List[PlannedScene], total_words: int) -> Dict[str, Any]:
    """Return a summary dict for validation/reporting."""
    covered = sum(s.word_count for s in scenes)
    durations = [s.end_time - s.start_time for s in scenes]
    return {
        "total_scenes": len(scenes),
        "total_words_covered": covered,
        "total_words_in_transcript": total_words,
        "coverage_pct": round(covered / total_words * 100, 1) if total_words else 0,
        "min_scene_duration_s": round(min(durations), 3) if durations else 0,
        "max_scene_duration_s": round(max(durations), 3) if durations else 0,
        "avg_scene_duration_s": round(sum(durations) / len(durations), 1) if durations else 0,
        "total_audio_duration_s": scenes[-1].end_time if scenes else 0,
    }

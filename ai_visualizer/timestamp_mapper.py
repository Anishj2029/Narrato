"""
timestamp_mapper.py
-------------------
Maps scenes to real timestamps derived from an MP3 or WAV audio file.

Strategy (no speech-recognition needed):
  1. Get total audio duration.
  2. Count total words in the script.
  3. Distribute time proportionally by word count per scene.
     This gives good-enough timestamps without an ASR API call.

If you later want word-level alignment, swap in whisper-timestamped or
a forced-alignment library – the Scene dataclass fields stay the same.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from .scene_parser import Scene


# ---------------------------------------------------------------------------
# Audio duration helpers
# ---------------------------------------------------------------------------

def _duration_mutagen(path: Path) -> float:
    """Use mutagen (fast, no decoding) to get duration in seconds."""
    from mutagen import File as MutagenFile  # type: ignore
    audio = MutagenFile(str(path))
    if audio is None:
        raise ValueError(f"mutagen could not read: {path}")
    return float(audio.info.length)


def _duration_pydub(path: Path) -> float:
    """Fallback: use pydub (decodes the file)."""
    from pydub import AudioSegment  # type: ignore
    seg = AudioSegment.from_file(str(path))
    return len(seg) / 1000.0


def get_audio_duration(audio_path: str | Path) -> float:
    """Return duration in seconds. Tries mutagen first, falls back to pydub."""
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    try:
        return _duration_mutagen(path)
    except Exception:
        return _duration_pydub(path)


# ---------------------------------------------------------------------------
# Timestamp assignment
# ---------------------------------------------------------------------------

def assign_timestamps(scenes: List[Scene], audio_path: str | Path) -> List[Scene]:
    """
    Assign start_time / end_time to each scene proportionally by word count.

    Parameters
    ----------
    scenes     : list of Scene objects (already parsed)
    audio_path : path to the MP3 or WAV file

    Returns the same list (mutated in place) for convenience.
    """
    duration = get_audio_duration(audio_path)

    word_counts = [len(s.text.split()) for s in scenes]
    total_words = sum(word_counts)

    if total_words == 0:
        # Edge case: distribute evenly
        per_scene = duration / max(len(scenes), 1)
        for i, scene in enumerate(scenes):
            scene.start_time = round(i * per_scene, 3)
            scene.end_time = round((i + 1) * per_scene, 3)
        return scenes

    cursor = 0.0
    for scene, wc in zip(scenes, word_counts):
        scene_duration = duration * (wc / total_words)
        scene.start_time = round(cursor, 3)
        scene.end_time = round(cursor + scene_duration, 3)
        cursor += scene_duration

    # Make sure the last scene ends exactly at the audio end
    if scenes:
        scenes[-1].end_time = round(duration, 3)

    return scenes


def assign_timestamps_no_audio(scenes: List[Scene], words_per_minute: int = 130) -> List[Scene]:
    """
    Fallback: estimate timestamps from average reading/speech speed when no
    audio file is provided.

    Default 130 wpm ≈ typical narration pace.
    """
    cursor = 0.0
    for scene in scenes:
        wc = len(scene.text.split())
        scene_duration = (wc / words_per_minute) * 60.0
        scene.start_time = round(cursor, 3)
        scene.end_time = round(cursor + scene_duration, 3)
        cursor += scene_duration
    return scenes

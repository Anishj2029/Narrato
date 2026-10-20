"""
tests/test_scene_planner.py
---------------------------
Tests for the scene_planner module.

All tests are deterministic and require no API keys, no audio playback,
and no LLM calls. They use either the real transcript_segments.json
(integration tests) or small in-memory fixture data (unit tests).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import List, Dict, Any

import pytest

from ai_visualizer.scene_planner import (
    PlannedScene,
    plan_scenes,
    save_json,
    save_csv,
    coverage_report,
    _compute_boundaries,
    _assemble_scenes,
    _merge_short_scenes,
    _make_title,
    _fallback_image_prompt,
    _fallback_description,
    load_transcript,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

REAL_TRANSCRIPT = Path("transcript_segments.json")

# Minimal synthetic transcript for unit tests (no real audio needed)
def _make_transcript(segments: List[Dict]) -> Dict[str, Any]:
    """Build a synthetic transcript dict matching the real schema."""
    all_words = []
    for seg in segments:
        all_words.extend(seg.get("words", []))
    return {
        "audio_file": "test.mp3",
        "model": "medium",
        "device": "cpu",
        "audio_duration_seconds": segments[-1]["end"] if segments else 0.0,
        "transcription_time_seconds": 1.0,
        "real_time_factor": 0.5,
        "language": "en",
        "language_probability": 1.0,
        "total_segments": len(segments),
        "total_words": len(all_words),
        "segments": segments,
    }


def _seg(id: int, start: float, end: float, text: str) -> Dict:
    """Create a minimal segment with word-level timing."""
    words_raw = text.split()
    if not words_raw:
        return {"id": id, "start": start, "end": end, "text": text,
                "avg_logprob": -0.1, "no_speech_prob": 0.01, "words": []}
    step = (end - start) / len(words_raw)
    words = [
        {"word": f" {w}", "start": round(start + i * step, 3),
         "end": round(start + (i + 1) * step, 3), "probability": 0.99}
        for i, w in enumerate(words_raw)
    ]
    return {
        "id": id, "start": start, "end": end, "text": text,
        "avg_logprob": -0.1, "no_speech_prob": 0.01, "words": words,
    }


# A synthetic transcript with three natural topic blocks separated by 2s pauses
SYNTH_SEGS_3BLOCKS = [
    # Block 1: childhood / identity intro (0–30s)
    _seg(1,  0.0,  3.0, "Have you ever noticed something strange about yourself."),
    _seg(2,  3.5,  6.5, "Around your family you behave one way."),
    _seg(3,  7.0, 10.0, "Around your friends you become a completely different person."),
    _seg(4, 10.5, 14.0, "Which one of these is the real you."),
    _seg(5, 14.5, 18.0, "Perhaps the answer is not as simple as we think."),
    _seg(6, 18.5, 22.0, "You see when I was younger I used to think every person had one personality."),
    _seg(7, 22.5, 26.0, "But the more you observe people the more you begin to wonder."),
    # 2.0s pause here (hard boundary)
    _seg(8, 28.0, 32.0, "Imagine a young boy sitting in a classroom with plenty of ideas."),
    _seg(9, 32.5, 35.5, "But whenever he says something his classmates laugh at him."),
    _seg(10, 36.0, 39.0, "So slowly he becomes quiet and stops sharing his thoughts."),
    _seg(11, 39.5, 43.0, "Years pass and he meets new people who listen to him carefully."),
    _seg(12, 43.5, 47.0, "Suddenly he begins talking more and becomes confident again."),
    # 2.0s pause here (hard boundary)
    _seg(13, 49.0, 52.0, "Research in personality psychology gives us an interesting clue."),
    _seg(14, 52.5, 56.0, "People can behave differently in different situations while stable."),
    _seg(15, 56.5, 60.0, "In other words you can be the same person and still act differently."),
    _seg(16, 60.5, 64.0, "Depending on where you are whom you are with and how you feel."),
]


@pytest.fixture
def synth_3block_path(tmp_path):
    data = _make_transcript(SYNTH_SEGS_3BLOCKS)
    p = tmp_path / "synth.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


@pytest.fixture
def real_transcript_path():
    if not REAL_TRANSCRIPT.exists():
        pytest.skip("transcript_segments.json not present")
    return REAL_TRANSCRIPT


# ---------------------------------------------------------------------------
# Unit tests: boundary detection
# ---------------------------------------------------------------------------

class TestBoundaryDetection:

    def test_hard_pause_always_breaks(self):
        """A gap >= 1.5s must always produce a boundary."""
        segs = [
            _seg(1, 0.0, 3.0, "First sentence here."),
            _seg(2, 5.0, 8.0, "After a 2s gap."),   # 2s pause
            _seg(3, 8.5, 11.0, "Third sentence."),
        ]
        boundaries = _compute_boundaries(segs, target_words=999,
                                         pause_hard=1.5, pause_soft=0.7)
        assert 0 in boundaries
        assert 1 in boundaries   # seg index 1 starts after the 2s pause

    def test_no_boundary_on_small_gaps(self):
        """Small pauses (< soft threshold) never break even if over word cap."""
        segs = [_seg(i, i * 3.0, i * 3.0 + 2.5,
                     "word " * 20) for i in range(5)]
        boundaries = _compute_boundaries(segs, target_words=10,
                                         pause_hard=1.5, pause_soft=0.7)
        # All gaps are 0.5s — under soft threshold — so only index 0
        assert boundaries == [0]

    def test_soft_boundary_triggers_when_over_cap(self):
        """After accumulating >= target_words, a pause >= soft threshold breaks."""
        # seg0 = 5 words (under target=10), seg1 = 8 words (cumulative 13 >= 10)
        # pause between seg1 and seg2 is 0.8s >= soft 0.7 → break at index 2
        segs = [
            _seg(1, 0.0, 3.0, "one two three four five"),        # 5 words
            _seg(2, 3.5, 6.5, "six seven eight nine ten eleven twelve thirteen"),  # 8 words
            _seg(3, 7.3, 10.0, "second topic starts here now"),  # 0.8s gap
            _seg(4, 10.5, 13.0, "fourth segment more content"),
        ]
        boundaries = _compute_boundaries(segs, target_words=10,
                                         pause_hard=1.5, pause_soft=0.7)
        assert 0 in boundaries
        assert 2 in boundaries   # break fires at seg index 2 (after cumwords >= target)

    def test_first_boundary_always_zero(self):
        segs = [_seg(1, 0.0, 3.0, "Hello world."),
                _seg(2, 3.5, 6.0, "Goodbye world.")]
        b = _compute_boundaries(segs)
        assert b[0] == 0

    def test_single_segment_returns_only_zero(self):
        segs = [_seg(1, 0.0, 5.0, "Only one segment here.")]
        b = _compute_boundaries(segs)
        assert b == [0]


# ---------------------------------------------------------------------------
# Unit tests: scene assembly
# ---------------------------------------------------------------------------

class TestSceneAssembly:

    def test_timestamps_come_from_whisper_not_proportional(self):
        """start_time must equal the first word start of the first segment."""
        segs = [
            _seg(1, 0.0, 3.0, "First words spoken here."),
            _seg(2, 5.0, 8.0, "After a pause."),
        ]
        scenes = _assemble_scenes(segs, [0, 1], audio_duration=8.0)
        # First scene starts at word start of seg 1 (word_start = 0.0)
        assert scenes[0].start_time == 0.0
        # Second scene starts at word start of seg 2 (5.0)
        assert scenes[1].start_time == 5.0

    def test_last_scene_ends_at_audio_duration(self):
        segs = [_seg(1, 0.0, 3.0, "Test."), _seg(2, 4.0, 7.0, "End.")]
        audio_dur = 8.5
        scenes = _assemble_scenes(segs, [0], audio_duration=audio_dur)
        assert scenes[-1].end_time == audio_dur

    def test_segment_ids_are_tracked(self):
        segs = [
            _seg(1, 0.0, 3.0, "A."), _seg(2, 3.5, 6.0, "B."),
            _seg(3, 8.0, 11.0, "C."),  # hard boundary here
        ]
        boundaries = [0, 2]
        scenes = _assemble_scenes(segs, boundaries, audio_duration=11.0)
        assert 1 in scenes[0].segment_ids
        assert 2 in scenes[0].segment_ids
        assert 3 in scenes[1].segment_ids

    def test_word_count_is_correct(self):
        segs = [
            _seg(1, 0.0, 3.0, "one two three four five"),
            _seg(2, 3.5, 6.0, "six seven eight"),
        ]
        scenes = _assemble_scenes(segs, [0], audio_duration=6.0)
        assert scenes[0].word_count == 8

    def test_scenes_are_sequential_no_overlap(self):
        segs = [_seg(i, i * 3.0, i * 3.0 + 2.5, f"Segment {i} text.") for i in range(6)]
        boundaries = [0, 2, 4]
        scenes = _assemble_scenes(segs, boundaries, audio_duration=17.5)
        for i in range(len(scenes) - 1):
            assert scenes[i].end_time <= scenes[i + 1].start_time + 0.01, \
                f"Scene {scenes[i].index} overlaps Scene {scenes[i+1].index}"


# ---------------------------------------------------------------------------
# Unit tests: scene merging
# ---------------------------------------------------------------------------

class TestMergeShortScenes:

    def _make_scene(self, idx, start, end, text):
        wc = len(text.split())
        s = PlannedScene(
            index=idx, title=_make_title(text), text=text,
            start_time=start, end_time=end, word_count=wc,
            segment_ids=[idx],
        )
        return s

    def test_short_scene_merged_into_next(self):
        scenes = [
            self._make_scene(1, 0.0, 5.0,
                "Very short fragment."),                                  # 5s < min 15s, 3 words < 8
            self._make_scene(2, 5.0, 35.0,
                "A longer scene here with more than eight words of content."),  # 30s, 12 words
            self._make_scene(3, 35.0, 65.0,
                "Another long scene with enough words to stand alone here."),   # 30s, 11 words
        ]
        merged = _merge_short_scenes(scenes, min_duration=15.0, min_words=8)
        # Scene 1 (short) must be merged into scene 2; scene 3 stays separate
        assert len(merged) == 2, f"Expected 2 scenes after merge, got {len(merged)}: {[s.text[:40] for s in merged]}"
        assert merged[0].start_time == 0.0
        assert merged[0].end_time == 35.0
        assert merged[1].start_time == 35.0
        assert merged[1].end_time == 65.0

    def test_adequate_scenes_not_merged(self):
        scenes = [
            self._make_scene(1, 0.0, 20.0, "This is a sufficiently long scene with many words."),
            self._make_scene(2, 20.0, 45.0, "Another adequate scene with plenty of content here."),
        ]
        merged = _merge_short_scenes(scenes, min_duration=15.0, min_words=8)
        assert len(merged) == 2

    def test_single_scene_not_merged(self):
        scenes = [self._make_scene(1, 0.0, 5.0, "Tiny.")]
        merged = _merge_short_scenes(scenes, min_duration=15.0, min_words=8)
        assert len(merged) == 1

    def test_indexes_resequenced_after_merge(self):
        scenes = [
            self._make_scene(1, 0.0, 3.0, "Short one."),
            self._make_scene(2, 3.0, 25.0, "Long scene two content."),
            self._make_scene(3, 25.0, 50.0, "Long scene three content."),
        ]
        merged = _merge_short_scenes(scenes, min_duration=15.0, min_words=8)
        for expected_idx, s in enumerate(merged, start=1):
            assert s.index == expected_idx, f"Expected index {expected_idx}, got {s.index}"


# ---------------------------------------------------------------------------
# Unit tests: title and prompt generation
# ---------------------------------------------------------------------------

class TestTitleAndPrompt:

    def test_title_reasonable_length(self):
        text = "Have you ever noticed something strange about yourself? " * 3
        scene = PlannedScene(1, "", text, 0, 10, word_count=len(text.split()))
        title = _make_title(text)
        assert 1 <= len(title.split()) <= 10

    def test_title_skips_leading_connectors(self):
        title = _make_title("And this is a test sentence about something.")
        # Should not start with "And"
        assert not title.lower().startswith("and ")

    def test_fallback_prompt_has_prefix(self):
        scene = PlannedScene(1, "Test Title", "Some scene text here.",
                              0.0, 10.0, word_count=5)
        prompt = _fallback_image_prompt(scene)
        assert prompt.startswith("[FALLBACK]")

    def test_fallback_description_has_prefix(self):
        scene = PlannedScene(1, "Test", "Some scene text.", 0.0, 20.0, word_count=5)
        desc = _fallback_description(scene)
        assert desc.startswith("[FALLBACK]")

    def test_fallback_prompt_not_empty(self):
        scene = PlannedScene(1, "Test", "Family friends identity relationship love.",
                              0.0, 30.0, word_count=5)
        prompt = _fallback_image_prompt(scene)
        assert len(prompt) > 20


# ---------------------------------------------------------------------------
# Integration tests: synthetic 3-block transcript
# ---------------------------------------------------------------------------

class TestSyntheticTranscript:

    def test_produces_three_scenes(self, synth_3block_path):
        scenes = plan_scenes(synth_3block_path)
        assert len(scenes) == 3, f"Expected 3 scenes, got {len(scenes)}: {[s.title for s in scenes]}"

    def test_all_words_covered(self, synth_3block_path):
        """Sum of scene word counts should equal total transcript words."""
        data = json.loads(synth_3block_path.read_text())
        total = data["total_words"]
        scenes = plan_scenes(synth_3block_path)
        covered = sum(s.word_count for s in scenes)
        assert covered == total, f"Covered {covered} words, expected {total}"

    def test_no_timestamp_overlap(self, synth_3block_path):
        scenes = plan_scenes(synth_3block_path)
        for i in range(len(scenes) - 1):
            a, b = scenes[i], scenes[i + 1]
            assert a.end_time <= b.start_time + 0.001, \
                f"Overlap: scene {a.index} ends {a.end_time} > scene {b.index} starts {b.start_time}"

    def test_timestamps_monotonic(self, synth_3block_path):
        scenes = plan_scenes(synth_3block_path)
        times = [(s.start_time, s.end_time) for s in scenes]
        for i in range(len(times) - 1):
            assert times[i][1] <= times[i + 1][0] + 0.001

    def test_all_end_times_exceed_start_times(self, synth_3block_path):
        scenes = plan_scenes(synth_3block_path)
        for s in scenes:
            assert s.end_time > s.start_time, \
                f"Scene {s.index}: end_time {s.end_time} <= start_time {s.start_time}"

    def test_indexes_sequential(self, synth_3block_path):
        scenes = plan_scenes(synth_3block_path)
        for i, s in enumerate(scenes, start=1):
            assert s.index == i

    def test_all_prompts_are_fallback(self, synth_3block_path):
        """Without Gemini, all prompts should be fallback-labelled."""
        scenes = plan_scenes(synth_3block_path)
        for s in scenes:
            assert s.image_prompt.startswith("[FALLBACK]"), \
                f"Scene {s.index} prompt not labelled: {s.image_prompt[:50]}"

    def test_filename_suggestions_numbered(self, synth_3block_path):
        scenes = plan_scenes(synth_3block_path)
        for s in scenes:
            assert s.filename_suggestion == f"scene_{s.index:03d}.png"


# ---------------------------------------------------------------------------
# Integration tests: export round-trip
# ---------------------------------------------------------------------------

class TestExport:

    def test_json_roundtrip(self, synth_3block_path, tmp_path):
        scenes = plan_scenes(synth_3block_path)
        out = tmp_path / "scene_plan.json"
        save_json(scenes, out)
        assert out.exists()
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert len(loaded) == len(scenes)
        assert loaded[0]["index"] == 1
        assert "start_time" in loaded[0]
        assert "segment_ids" in loaded[0]
        assert "description" in loaded[0]

    def test_csv_roundtrip(self, synth_3block_path, tmp_path):
        import csv
        scenes = plan_scenes(synth_3block_path)
        out = tmp_path / "scene_plan.csv"
        save_csv(scenes, out)
        assert out.exists()
        with open(out, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == len(scenes)
        assert "index" in rows[0]
        assert "start_time" in rows[0]
        assert "description" in rows[0]
        assert "filename_suggestion" in rows[0]

    def test_json_timestamps_are_floats(self, synth_3block_path, tmp_path):
        scenes = plan_scenes(synth_3block_path)
        out = tmp_path / "scene_plan.json"
        save_json(scenes, out)
        loaded = json.loads(out.read_text())
        for row in loaded:
            assert isinstance(row["start_time"], float)
            assert isinstance(row["end_time"], float)


# ---------------------------------------------------------------------------
# Integration tests: invalid inputs
# ---------------------------------------------------------------------------

class TestInvalidInputs:

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            plan_scenes(tmp_path / "nonexistent.json")

    def test_missing_segments_key_raises(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"audio_file": "x.mp3"}))
        with pytest.raises(ValueError, match="segments"):
            load_transcript(bad)

    def test_empty_segments_returns_empty(self, tmp_path):
        empty = tmp_path / "empty.json"
        data = _make_transcript([])
        data["audio_duration_seconds"] = 0.0
        empty.write_text(json.dumps(data))
        scenes = plan_scenes(empty)
        assert scenes == []

    def test_overlapping_timestamps_not_produced(self, tmp_path):
        """Planner must never output overlapping scene timestamps."""
        segs = [_seg(i, i * 4.0, i * 4.0 + 3.5, f"Segment {i} text sentence.") for i in range(10)]
        data = _make_transcript(segs)
        p = tmp_path / "t.json"
        p.write_text(json.dumps(data))
        scenes = plan_scenes(p, min_duration=0.0)
        for i in range(len(scenes) - 1):
            a, b = scenes[i], scenes[i + 1]
            assert a.end_time <= b.start_time + 0.001, \
                f"Overlap: scene {a.index} [{a.start_time},{a.end_time}] and " \
                f"scene {b.index} [{b.start_time},{b.end_time}]"


# ---------------------------------------------------------------------------
# Integration tests: real transcript (skipped if file absent)
# ---------------------------------------------------------------------------

class TestRealTranscript:

    def test_scene_count_reasonable(self, real_transcript_path):
        """Expect 8–20 scenes for an 11-min narration at ~70 target words."""
        scenes = plan_scenes(real_transcript_path)
        assert 8 <= len(scenes) <= 20, \
            f"Expected 8–20 scenes, got {len(scenes)}: {[s.title for s in scenes]}"

    def test_no_scene_overlap(self, real_transcript_path):
        scenes = plan_scenes(real_transcript_path)
        for i in range(len(scenes) - 1):
            a, b = scenes[i], scenes[i + 1]
            assert a.end_time <= b.start_time + 0.001, \
                f"Overlap at scene {a.index}→{b.index}"

    def test_first_scene_starts_near_zero(self, real_transcript_path):
        scenes = plan_scenes(real_transcript_path)
        assert scenes[0].start_time < 1.0, \
            f"First scene starts at {scenes[0].start_time}s, expected < 1s"

    def test_last_scene_ends_at_audio_duration(self, real_transcript_path):
        data = json.loads(Path(real_transcript_path).read_text())
        audio_dur = data["audio_duration_seconds"]
        scenes = plan_scenes(real_transcript_path)
        assert abs(scenes[-1].end_time - audio_dur) < 0.1, \
            f"Last scene ends at {scenes[-1].end_time}s, audio is {audio_dur}s"

    def test_word_coverage_above_95_pct(self, real_transcript_path):
        """Words covered by scenes should be >= 95% of transcript words."""
        data = json.loads(Path(real_transcript_path).read_text())
        total = data["total_words"]
        scenes = plan_scenes(real_transcript_path)
        covered = sum(s.word_count for s in scenes)
        pct = covered / total * 100
        assert pct >= 95.0, f"Only {pct:.1f}% of words covered ({covered}/{total})"

    def test_timestamps_grounded_not_proportional(self, real_transcript_path):
        """
        Verify that scene start_times come from Whisper segment timestamps,
        not from proportional word-count estimation.

        Method: for each scene, find the first Whisper segment assigned to it
        and confirm the scene's start_time matches that segment's actual start
        (from the word-level data) within 50ms.  A proportional system would
        assign a different value calculated from audio_duration * (cumwords /
        total_words), which will differ by seconds for most scenes.
        """
        data = json.loads(Path(real_transcript_path).read_text())
        segments = {seg["id"]: seg for seg in data["segments"]}
        scenes = plan_scenes(real_transcript_path)

        for scene in scenes:
            if not scene.segment_ids:
                continue
            first_seg_id = scene.segment_ids[0]
            seg = segments[first_seg_id]
            # Word-level start (what the planner should use)
            words = seg.get("words", [])
            expected_start = words[0]["start"] if words else seg["start"]
            diff = abs(scene.start_time - expected_start)
            assert diff < 0.05, (
                f"Scene {scene.index} start_time={scene.start_time} does not match "
                f"Whisper segment {first_seg_id} word_start={expected_start} "
                f"(diff={diff:.3f}s > 50ms tolerance)"
            )

    def test_min_scene_duration_respected(self, real_transcript_path):
        """No scene should be shorter than MIN_SCENE_DURATION after merge."""
        from ai_visualizer.scene_planner import MIN_SCENE_DURATION
        scenes = plan_scenes(real_transcript_path)
        # All scenes except possibly the very last (absorbs leftovers)
        for s in scenes[:-1]:
            dur = s.end_time - s.start_time
            assert dur >= MIN_SCENE_DURATION - 0.5, \
                f"Scene {s.index} is {dur:.1f}s, below min {MIN_SCENE_DURATION}s"

    def test_coverage_report_structure(self, real_transcript_path):
        data = json.loads(Path(real_transcript_path).read_text())
        scenes = plan_scenes(real_transcript_path)
        report = coverage_report(scenes, data["total_words"])
        required = {"total_scenes", "total_words_covered", "total_words_in_transcript",
                    "coverage_pct", "min_scene_duration_s", "max_scene_duration_s",
                    "avg_scene_duration_s", "total_audio_duration_s"}
        assert required.issubset(report.keys())
        assert report["coverage_pct"] >= 95.0

# Real Audio Validation Report
**AI Visualizer — Audio Processing Pipeline**
_Validated: October 19, 2026_

---

## 1. Environment

| Item | Value |
|------|-------|
| Python | 3.10.9 |
| Platform | Windows 11 (win32) |
| mutagen | 1.48.1 |
| pydub | 0.25.1 |
| ffmpeg | **NOT INSTALLED** |
| pytest | 9.1.1 |
| Audio file | `narration.mp3` (copied from Downloads) |
| Script file | `sample_script.md` |

---

## 2. Test Suite Results

All 22 existing unit tests passed before any real-audio testing was performed.

```
tests/test_exporter.py::test_csv_creates_file             PASSED
tests/test_exporter.py::test_csv_has_correct_rows         PASSED
tests/test_exporter.py::test_csv_headers                  PASSED
tests/test_exporter.py::test_json_creates_file            PASSED
tests/test_exporter.py::test_json_is_valid                PASSED
tests/test_exporter.py::test_roundtrip_timestamps         PASSED
tests/test_image_generator.py::test_placeholder_creates_images  PASSED
tests/test_image_generator.py::test_placeholder_images_are_png  PASSED
tests/test_image_generator.py::test_image_filenames_are_numbered PASSED
tests/test_image_generator.py::test_skip_existing         PASSED
tests/test_image_generator.py::test_csv_exported_after_generation PASSED
tests/test_scene_parser.py::test_marker_based_scenes      PASSED
tests/test_scene_parser.py::test_scene_titles_are_short   PASSED
tests/test_scene_parser.py::test_paragraph_grouping       PASSED
tests/test_scene_parser.py::test_soft_cap_splits_long_runs PASSED
tests/test_scene_parser.py::test_scene_text_not_empty     PASSED
tests/test_scene_parser.py::test_to_dict_keys             PASSED
tests/test_timestamp_mapper.py::test_timestamps_are_monotonic    PASSED
tests/test_timestamp_mapper.py::test_start_time_first_scene      PASSED
tests/test_timestamp_mapper.py::test_end_time_increases          PASSED
tests/test_timestamp_mapper.py::test_longer_scenes_get_more_time PASSED
tests/test_timestamp_mapper.py::test_custom_wpm           PASSED

RESULT: 22 passed, 0 failed  (0.33s)
```

No regressions. The pipeline is mechanically sound.

---

## 3. Audio File Inspection

The ElevenLabs file was located in the user's Downloads folder and copied to the project root as `narration.mp3`.

| Property | Value |
|----------|-------|
| Original filename | `ElevenLabs_2026-10-19T13_18_09_Spuds Oxley - Grandpa _pvc_sp76_s83_sb85_v4.mp3` |
| Duration | **711.549 seconds (11 minutes 51.5 seconds)** |
| File size | 11,134.5 KB (~10.9 MB) |
| Bitrate | 128,000 bps (128 kbps) |
| Sample rate | 44,100 Hz |
| Channels | 1 (mono) |
| Tags | TSSE, GEOB:c2pa manifest store |
| Duration source | mutagen (MP3 header, no decoding required) — confirmed reliable |
| pydub silence analysis | **Could not run — ffmpeg not installed** |

The mutagen duration reading worked correctly and matches what would be expected from file size and bitrate: `11,134.5 KB × 8 / 128 kbps ≈ 696s` (within ~2% of 711.5s, consistent with VBR padding and ID3 overhead).

---

## 4. Script/Audio Mismatch — Critical Finding

**The provided script does not match the provided audio.**

| Metric | Value |
|--------|-------|
| `sample_script.md` word count | 243 words |
| Audio duration | 711.549 seconds |
| Implied narration speed if matched | **20.5 words per minute** |
| Normal narration speed | 120–160 wpm |
| Expected word count to fill 711.5s | ~1,541–1,897 words |
| Words unaccounted for | ~1,298–1,654 |

At 20.5 wpm, the narrator would be speaking roughly one word every three seconds. Human speech does not operate at this rate. The conclusion is definitive: **`sample_script.md` is the wrong script for this audio file.**

The audio file was named with the voice character "Spuds Oxley - Grandpa," which is an ElevenLabs voice preset. The `sample_script.md` contains a tech-history narration titled "The Rise of AI." These are unrelated content.

**Impact on all downstream results:** Every timestamp generated in this validation run is numerically correct as a mathematical proportion, but semantically meaningless — they do not correspond to what the narrator is actually saying at any given moment in the audio.

---

## 5. Timestamp Method Clarification

The codebase implements **two distinct timestamp methods**. Both are proportional estimates. Neither is word-level alignment.

### Method A — No-Audio Estimation (`assign_timestamps_no_audio`)

Estimates scene durations using a fixed words-per-minute rate (default 130 wpm). Does not read the audio file at all.

```
Total estimated duration: 102.0s (1.7 min) for the sample script
Deviation from actual audio: 609.5 seconds (85.7% error)
```

This method is only useful for rough previewing when no audio is available.

### Method B — Proportional Audio (`assign_timestamps`)

Reads the actual audio duration via mutagen, then distributes that duration proportionally across scenes based on each scene's word count share of total words.

```
Formula: scene_duration = audio_duration × (scene_words / total_words)
Total duration: 711.549s (exact match to audio)
```

This method correctly anchors the total duration to the real audio. However, it still distributes time by word count ratio, not by actual speech position.

### What Neither Method Does

Neither method performs **word-level or forced alignment** — that is, neither reads the audio signal to determine where specific words are actually spoken. Achieving that requires:

- **OpenAI Whisper** with `word_timestamps=True`
- **whisper-timestamped** (third-party wrapper)
- **aeneas** (forced alignment tool)

None of these are implemented or imported in the current codebase.

---

## 6. Timestamp Accuracy Findings (With Matched Script, Hypothetically)

Assuming the correct matching script were provided, here is what the proportional timestamps would look like. These figures come from the real audio run with `sample_script.md`:

| Scene | Words | % of Total | Start | End | Duration | Exp. @ 130 wpm |
|-------|-------|-----------|-------|-----|----------|----------------|
| 1 | 4 | 1.8% | 0.000s | 12.879s | 12.879s | 1.8s |
| 2 | 36 | 16.3% | 12.879s | 128.787s | 115.908s | 16.6s |
| 3 | 38 | 17.2% | 128.787s | 251.135s | 122.348s | 17.5s |
| 4 | 33 | 14.9% | 251.135s | 357.385s | 106.250s | 15.2s |
| 5 | 37 | 16.7% | 357.385s | 476.513s | 119.128s | 17.1s |
| 6 | 31 | 14.0% | 476.513s | 576.323s | 99.810s | 14.3s |
| 7 | 4 | 1.8% | 576.323s | 589.202s | 12.879s | 1.8s |
| 8 | 38 | 17.2% | 589.202s | 711.549s | 122.347s | 17.5s |

**Proportionality verification:** All 8 scenes pass. The ratio `scene_duration / audio_duration` equals `scene_words / total_words` to within 0.00000111 (floating-point rounding only).

**Boundary verification:**
- First scene starts at exactly 0.0s ✅
- Last scene ends at exactly 711.549s ✅
- Zero gaps between all scene boundaries ✅

**The math is correct. The input data is wrong.**

---

## 7. Scene Segmentation Findings

### 7.1 Scene Count

The parser produced 8 scenes from 6 content sections. The extra 2 scenes are caused by the title-only bug.

### 7.2 Title-Only Scene Bug (Confirmed)

Two scenes contain only the section heading text, with no body content. These are generated because the `## Section Title` marker flushes any existing buffer, then the heading text itself is left in a new buffer that gets flushed as a standalone scene when the first real sentence arrives.

| Scene | Words | Text | Duration (real audio run) |
|-------|-------|------|--------------------------|
| Scene 1 | 4 | "The Rise of AI" | 12.9s |
| Scene 7 | 4 | "Where We Stand Today" | 12.9s |

`MIN_WORDS_FOR_SCENE = 4` is the current guard. "The Rise of AI" is exactly 4 words — it passes the filter and becomes a scene. These two scenes consume 25.8 seconds of audio time (3.6% of the total) without representing any real visual content.

### 7.3 Scenes 3 and 8 Are Identical Word Count

Scenes 3 and 8 both have 38 words and receive identical durations (122.3s each). This is expected behavior — it is not a bug, just a coincidence in the sample script.

### 7.4 Word Count Discrepancy

The script has 243 raw words but the parser only registers 221 words across scenes. The 22-word gap comes from:
- Markdown heading lines that are stripped by `_SCENE_MARKER` regex before entering the word count
- Heading text for title-only scenes (4 + 4 = 8 words) that get their own scenes
- Minor whitespace tokenization differences

This does not affect proportionality since the proportional calculation uses only the scene word counts as parsed.

---

## 8. Bugs Discovered

### BUG-001 — Title-Only Scenes (Pre-existing, Confirmed)
**Severity:** Medium  
**File:** `ai_visualizer/scene_parser.py`  
**Description:** Section headings with no inline body text create standalone 1–6 word scenes. The `MIN_WORDS_FOR_SCENE = 4` guard is too low — "The Rise of AI" (4 words) passes. These title scenes consume real timestamp slots and would generate wasted images.  
**Evidence:** Scene 1 ("The Rise of AI", 4 words, 12.9s) and Scene 7 ("Where We Stand Today", 4 words, 12.9s).

---

### BUG-002 — Script/Audio Mismatch Is Silent (New Finding)
**Severity:** High  
**File:** `ai_visualizer/timestamp_mapper.py`, `ai_visualizer/cli.py`  
**Description:** When the script and audio are unrelated (completely different content), the pipeline completes silently with no warning. There is no sanity check comparing expected duration (from word count at a typical WPM) to actual audio duration. A 85.7% deviation (102s expected vs 711s actual) is not flagged.  
**Impact:** A user who uploads a wrong audio file gets confident-looking timestamps with no indication anything is wrong.

---

### BUG-003 — ffmpeg Not Available, pydub Fallback Will Fail
**Severity:** Medium  
**File:** `ai_visualizer/timestamp_mapper.py`  
**Description:** The duration function tries mutagen first and falls back to pydub. Mutagen works without ffmpeg. However, pydub requires ffmpeg for decoding. On this machine ffmpeg is not installed. If mutagen fails for any reason (e.g., a malformed or unsupported format), pydub will raise `FileNotFoundError: [WinError 2]` with a confusing message about `ffmpeg`.  
**Evidence:** Confirmed by testing — `from pydub import AudioSegment; AudioSegment.from_mp3(...)` throws `FileNotFoundError` when ffmpeg is absent.

---

### BUG-004 — Unused Dependencies in requirements.txt (Pre-existing, Confirmed)
**Severity:** Low  
**File:** `requirements.txt`  
**Description:** `nltk==3.8.1`, `spacy==3.7.4`, and `tqdm==4.66.4` are listed but not imported anywhere in the codebase. Combined installation size is approximately 500MB+ (spaCy includes large model data). `python-dotenv==1.0.1` is also wrong — the installed version is `1.2.4` and `1.0.1` does not exist on PyPI.

---

### BUG-005 — No Test for Real Audio Timestamp Mapping (New Finding)
**Severity:** Low  
**File:** `tests/test_timestamp_mapper.py`  
**Description:** All timestamp tests use `assign_timestamps_no_audio`. There is no test for `assign_timestamps` (the real audio path). The mutagen read path, the pydub fallback, and the proportional math when using a real file are all untested in the test suite.

---

## 9. Limitations Confirmed

### 9.1 Proportional Timestamps, Not Word-Aligned
The system distributes total audio duration proportionally by word count. This means:
- If scene A has twice the words of scene B, it gets twice the time.
- The actual start/end position of spoken words in the audio is never consulted.
- A long pause, a sound effect, or narrator speed variation will silently distort all scene boundaries.

**This is the expected, documented behavior.** It is not a bug, but it is an important limitation to understand before using timestamps for video editing.

### 9.2 No Silence Detection Without ffmpeg
The pydub silence detection API (`detect_silence`) requires ffmpeg to decode the MP3 before analysis. ffmpeg is not installed. Structural pause detection (which would give better scene boundary hints) cannot run in the current environment.

### 9.3 No Transcript Verification
There is no mechanism to verify that a script matches its audio — not even a rough duration sanity check. This was the most consequential gap discovered in this validation.

---

## 10. Recommended Fixes (Priority Order)

### P1 — Add a Duration Sanity Check (BUG-002)
**File:** `ai_visualizer/timestamp_mapper.py`  
Add a check in `assign_timestamps` that computes the expected duration from the script's word count at 120–160 wpm and warns if the actual audio duration falls outside a 2× tolerance:

```python
expected_min = total_words / 160 * 60  # fastest reasonable narration
expected_max = total_words / 80 * 60   # slowest reasonable narration
if not (expected_min <= duration <= expected_max):
    import warnings
    warnings.warn(
        f"Audio duration ({duration:.0f}s) is outside the expected range "
        f"({expected_min:.0f}s–{expected_max:.0f}s) for {total_words} words. "
        f"The script and audio may not match.",
        UserWarning,
        stacklevel=2,
    )
```

---

### P2 — Fix Title-Only Scenes (BUG-001)
**File:** `ai_visualizer/scene_parser.py`  
Raise `MIN_WORDS_FOR_SCENE` from 4 to 8. Alternatively, detect when a flushed buffer contains only the text of a heading marker and merge it forward into the next scene rather than creating a standalone scene:

```python
# In _group_into_scenes, when flushing after a scene marker:
# If the buffer is only the remainder text from the heading line
# and word count < 8, store it as a "pending prefix" to prepend
# to the next scene instead of creating a standalone scene.
```

Raising the constant to 8 is the simpler fix:
```python
MIN_WORDS_FOR_SCENE = 8  # was 4
```

---

### P3 — Supply the Correct Matching Script
**Action:** Not a code fix — a content fix.  
The audio file `narration.mp3` ("Spuds Oxley - Grandpa" voice, 11m 51s) needs its matching transcript. Either:
- Export the transcript from ElevenLabs (available in the ElevenLabs dashboard)
- Run OpenAI Whisper on the audio to generate a transcript: `whisper narration.mp3 --output_format txt`

Until the correct script is provided, no timestamp result from this audio is meaningful.

---

### P4 — Add a Real-Audio Test (BUG-005)
**File:** `tests/test_timestamp_mapper.py`  
Add a test that creates a minimal real MP3 (or mocks the mutagen call) and verifies that `assign_timestamps` produces timestamps that sum to the audio duration:

```python
def test_real_audio_timestamps_sum_to_duration(tmp_path):
    # Use mutagen mock or a tiny synthetic MP3
    # Verify: scenes[-1].end_time == audio_duration
    # Verify: proportionality holds
```

---

### P5 — Improve the ffmpeg Error Message (BUG-003)
**File:** `ai_visualizer/timestamp_mapper.py`  
Wrap the pydub fallback to catch `FileNotFoundError` and re-raise with a clear message:

```python
def _duration_pydub(path: Path) -> float:
    try:
        from pydub import AudioSegment
        seg = AudioSegment.from_file(str(path))
        return len(seg) / 1000.0
    except FileNotFoundError:
        raise RuntimeError(
            "pydub requires ffmpeg to decode audio. "
            "Install ffmpeg: https://ffmpeg.org/download.html"
        ) from None
```

---

### P6 — Clean Up requirements.txt (BUG-004)
Remove unused packages and fix the version pin:

```diff
- nltk==3.8.1
- spacy==3.7.4
- tqdm==4.66.4
- python-dotenv==1.0.1
+ python-dotenv==1.2.4
```

---

### P7 — Add Word-Level Alignment (Future Enhancement)
When frame-accurate timestamps are needed for video editing, add optional Whisper alignment:

```
pip install openai-whisper
```

```python
# In timestamp_mapper.py — new optional function
def assign_timestamps_whisper(scenes, audio_path):
    import whisper
    model = whisper.load_model("base")
    result = model.transcribe(str(audio_path), word_timestamps=True)
    # align scene text to whisper word segments...
```

This is optional. The proportional method is sufficient for storyboard previewing; Whisper is needed for final video sync.

---

## 11. Files Changed During This Validation

| File | Action | Reason |
|------|--------|--------|
| `narration.mp3` | **Created** (copied from Downloads) | Real audio file needed for testing |
| `outputs_real_audio/scenes.csv` | **Created** | Output from real-audio pipeline run |
| `outputs_real_audio/scenes.json` | **Created** | Output from real-audio pipeline run |
| `_analyze_timestamps.py` | **Created** (temporary) | Analysis script — can be deleted |
| `_compare_timestamps.py` | **Created** (temporary) | Comparison script — can be deleted |
| `REAL_AUDIO_VALIDATION.md` | **Created** | This report |

**No existing source files were modified.**  
The project architecture is unchanged.

---

## 12. Summary

The audio processing pipeline is **mechanically correct** — mutagen reads duration accurately, timestamps are distributed proportionally with no gaps or overflows, and all 22 unit tests pass. However, this validation exposed three significant issues:

1. **The script and audio are unrelated.** `sample_script.md` (243 words, ~1.7 min) was tested against an 11-minute ElevenLabs recording with entirely different content. No meaningful timestamp validation can be done until the matching transcript is provided.

2. **Timestamps are proportional estimates, not speech-aligned.** The system distributes time by word-count ratio. It does not read the audio signal. This is documented but worth stating explicitly: timestamps will drift whenever narration speed varies across scenes.

3. **There is no sanity check for script/audio mismatch.** The pipeline succeeded silently despite an 85.7% duration deviation. A simple WPM-range check would catch this class of error immediately.

The highest-value next action is: **obtain the transcript that matches `narration.mp3`**, re-run Phase 1, and then evaluate whether proportional timestamps are close enough for your use case or whether Whisper alignment is needed.

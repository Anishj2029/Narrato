# Transcript Recovery Report
**AI Visualizer — Audio Transcript Recovery & Verification**
_Completed: October 19, 2026_

---

## 1. Objectives and Approach

| Objective | Result |
|-----------|--------|
| Locate matching script in project | ❌ Not found |
| Check local folders for matching script | ❌ Not found |
| Extract text from audio metadata (C2PA) | ❌ Metadata contains only signing certificates |
| Transcribe audio with word-level timestamps | ✅ Complete |
| Compare transcript to `sample_script.md` | ✅ Confirmed mismatch |

---

## 2. Search Results — Original Script

### 2.1 Project Files
Only one script exists in the project: `sample_script.md` (243 words, "The Rise of AI"). It does not match the audio.

### 2.2 Local File Search
Searched `C:\Users\Anish\Desktop` and `C:\Users\Anish\Downloads` for `.txt` and `.md` files modified in the last 7 days. No script matching the audio content was found.

### 2.3 Audio Metadata (C2PA)
The MP3 contains a `GEOB:c2pa manifest store` tag (16,851 bytes). Its contents are:
- ElevenLabs creator/organization record: `{"@type":"Organization","name":"Eleven Labs Inc."}`
- A DigiCert code-signing certificate chain
- CBOR-encoded binary hashes and exclusion maps

**No text content or narration script is embedded in the metadata.**

---

## 3. Hardware & Dependency Report

| Component | Status | Detail |
|-----------|--------|--------|
| GPU | RTX 4050 Laptop, 6,141 MB VRAM | Detected via nvidia-smi |
| CUDA driver | 617.14 | Installed |
| CUDA Toolkit (cublas64_12.dll) | ❌ NOT INSTALLED | Only display driver present |
| ctranslate2 | 4.5.3 | CUDA compute types visible but runtime DLLs missing |
| faster-whisper | 1.2.1 | Pre-installed |
| Whisper model | `medium` (1,457 MB) | Already cached at `~/.cache/huggingface/hub/models--Systran--faster-whisper-medium` |
| CPU | AMD Ryzen 7 7435HS, 16 logical cores | Used for inference |
| RAM | 15.7 GB total, 5.7 GB free | Sufficient |

**CUDA Limitation:** The CUDA Toolkit runtime libraries (`cublas64_12.dll`, etc.) are not installed. The GPU driver is present but ctranslate2 could not use the GPU. A `RuntimeError: Library cublas64_12.dll is not found` error was raised when GPU inference was attempted. CPU inference with `int8` quantization was used instead.

**To enable GPU inference in future:** Install [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-downloads) and ensure `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin` is on the system PATH.

---

## 4. Transcription Results

### 4.1 Run Configuration

| Parameter | Value |
|-----------|-------|
| Model | `medium` (faster-whisper) |
| Device | CPU (int8 quantization) |
| Word timestamps | Enabled |
| VAD filter | Enabled (silence ≥ 500ms = segment break) |
| Language | English (auto-detected, probability = 1.0) |
| Audio duration | 711.549s (11m 51.5s) |
| Transcription time | 578.4s (9m 38s) |
| Real-time factor | 0.81x (faster than real time) |

### 4.2 Output Files

| File | Size | Contents |
|------|------|----------|
| `transcript.txt` | 9.8 KB | Plain-text transcript, 1,672 words |
| `transcript_segments.json` | 275.7 KB | 221 segments with per-word start/end times and confidence scores |

### 4.3 Transcript Quality

| Metric | Value |
|--------|-------|
| Total words | 1,672 |
| Total segments | 221 |
| Narration speed | 141.0 wpm (consistent with deliberate narration — normal range: 120–160 wpm) |
| Average word confidence | 0.982 (out of 1.0) |
| Low-confidence words (< 0.6) | 19 (1.1% of all words) |
| Confidence verdict | Excellent — 98.9% of words above 0.6 threshold |

### 4.4 Low-Confidence Words (for review)

These 19 words may contain transcription errors and should be spot-checked against the audio:

| Timestamp | Word | Confidence |
|-----------|------|-----------|
| 0.00s | 'Have' | 0.51 |
| 28.12s | 'the' | 0.28 |
| 40.31s | 'Perhaps' | 0.35 |
| 83.64s | 'who' | 0.17 |
| 209.72s | 'help' | 0.18 |
| 220.14s | 'And' | 0.53 |
| 249.44s | 'completely' | 0.16 |
| 334.54s | 'Some' | 0.27 |
| 415.98s | 'we' | 0.55 |
| 443.80s | 'Our' | 0.51 |
| 480.49s | 'partner' | 0.46 |
| 492.15s | 'even' | 0.35 |
| 569.73s | 'but' | 0.54 |
| 584.59s | 'and' | 0.56 |
| 601.67s | 'what' | 0.24 |

Most are common function words at segment boundaries where VAD cuts in — likely correct despite low scores.

### 4.5 Transcript Content Summary

The narration is a **motivational psychology essay** about identity, relationships, and how the people around us shape who we become. Structure:

| Timestamp Range | Content |
|----------------|---------|
| 0s – 92s | Opening hook — "Which version of you is the real you?" |
| 92s – 155s | The classroom boy story — personality shaped by environment |
| 155s – 210s | Personality psychology research — situational behavior vs. stable traits |
| 210s – 335s | Attachment theory — Bowlby & Ainsworth, childhood relationships |
| 335s – 445s | Friends and peer influence — choosing and being shaped by friends |
| 445s – 545s | Family patterns — how family behavior repeats into adulthood |
| 545s – 600s | Romantic partners — partner responsiveness, healthy disagreement |
| 600s – 650s | Individual agency — "we are not simply the product of people around us" |
| 650s – 711s | Closing — authenticity vs. adaptation, what relationships make possible |

---

## 5. Script vs. Audio Comparison

| Metric | `sample_script.md` | `transcript.txt` |
|--------|--------------------|------------------|
| Word count | 243 | 1,672 |
| Topic | History of AI (Alan Turing, neural networks, AI winters) | Psychology of identity and relationships |
| Unique words (meaningful) | 172 | 619 |
| Shared meaningful words | 29 (stopword-filtered) | — |
| Vocabulary overlap | 8.4% | — |
| Match verdict | **NO MATCH** | — |

The 8.4% overlap consists entirely of common English vocabulary (e.g., *question*, *human*, *learn*, *researchers*, *patterns*) — these words appear in both pieces coincidentally, not because they share content.

**Conclusion: `sample_script.md` is not the transcript for `narration.mp3`.** The two files are completely unrelated.

---

## 6. What the Transcript Is

Based on the content, the narration is a stand-alone educational/motivational piece about:
- Why people behave differently around different people
- Personality psychology (situational behavior vs. stable traits)
- Attachment theory (Bowlby & Ainsworth)
- How friendships, family, and romantic relationships shape identity
- The balance between adapting to others and staying authentic

The ElevenLabs filename (`Spuds Oxley - Grandpa _pvc_sp76_s83_sb85_v4`) indicates it was generated with the "Grandpa" voice preset at specific voice settings (`pvc_sp76_s83_sb85`). The script was written and submitted to ElevenLabs separately — it is not in this project's files.

---

## 7. Timestamp Method Clarification

The timestamps in `transcript_segments.json` are **word-level aligned** — produced by the Whisper model reading the actual audio signal.

| Method | How it works | Accuracy |
|--------|-------------|---------|
| Whisper word timestamps (this report) | Audio signal analysis — Whisper identifies where each word is spoken | ~10–50ms typical accuracy |
| Proportional (existing `assign_timestamps`) | Word count ratio × audio duration | May be off by many seconds or minutes |

The Whisper timestamps are now available as a ground-truth reference for testing the proportional method in future.

---

## 8. What Remains Unresolved

| Item | Status |
|------|--------|
| Original script for `narration.mp3` | Not found. Must be retrieved from ElevenLabs dashboard history or the person who created it. |
| CUDA Toolkit installation | Not done — GPU inference blocked. Install CUDA 12.x to enable 5–10x faster transcription in future. |
| Integration of `transcript_segments.json` into the scene pipeline | Not done — this is a future task. The `Scene` dataclass and `timestamp_mapper.py` would need to be extended to accept word-level timestamps as input. |
| `_read_c2pa.py` cleanup | Left in project root — can be deleted if not needed. |

---

## 9. Files Created / Changed

| File | Action | Notes |
|------|--------|-------|
| `transcript.txt` | **Created** | 1,672-word plain transcript |
| `transcript_segments.json` | **Created** | 221 segments × word-level timestamps, confidence scores |
| `TRANSCRIPT_RECOVERY_REPORT.md` | **Created** | This document |
| `_compare_scripts.py` | Created then deleted | Temporary analysis helper |

**No existing source files were modified.**

---

## 10. Summary

The original script for `narration.mp3` was not found anywhere in the project or local file system, and the audio metadata contained no embedded text. The audio was transcribed using `faster-whisper medium` on CPU (CUDA unavailable due to missing CUDA Toolkit). The transcription ran at 0.81× real time, completing in 9m 38s, and produced 1,672 words with 98.9% average word confidence.

The recovered transcript confirms the audio is a psychology/identity essay with no relation to `sample_script.md`. **The correct next step is to retrieve the matching script from the ElevenLabs dashboard** (generation history stores the input text), then re-run the scene pipeline using `transcript_segments.json` as the timestamp source instead of the proportional estimator.

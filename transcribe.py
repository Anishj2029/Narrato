"""
transcribe.py
-------------
Transcribe narration.mp3 using faster-whisper with word-level timestamps.

Model choice rationale:
  - Hardware: RTX 4050 Laptop (6GB VRAM), AMD Ryzen 7 7435HS, 16GB RAM
  - 'medium' (1.5GB) gives excellent accuracy on clear narration audio
    and fits easily in 6GB VRAM with float16 precision.
  - 'large-v3' (3GB) is available as a fallback if accuracy is insufficient.
  - Device: CUDA (GPU) — significantly faster than CPU for an 11-min file.

Outputs:
  - transcript.txt         plain text transcript
  - transcript_segments.json  word-level timestamps per segment

Usage:
  python transcribe.py
  python transcribe.py --model large-v3
  python transcribe.py --device cpu   (fallback if GPU fails)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

AUDIO_FILE = "narration.mp3"
TRANSCRIPT_TXT = "transcript.txt"
TRANSCRIPT_JSON = "transcript_segments.json"


def run(model_name: str = "medium", device: str = "cuda", audio: str = AUDIO_FILE) -> None:
    audio_path = Path(audio)
    if not audio_path.exists():
        print(f"ERROR: Audio file not found: {audio_path}")
        sys.exit(1)

    print(f"Audio file : {audio_path} ({audio_path.stat().st_size / 1024:.0f} KB)")
    print(f"Model      : {model_name}")
    print(f"Device     : {device}")
    print(f"Timestamps : word-level")
    print()

    # Import here so errors are surfaced cleanly
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("ERROR: faster-whisper is not installed.")
        print("  pip install faster-whisper")
        sys.exit(1)

    # Load model
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"Loading model '{model_name}' on {device} ({compute_type})...")
    t0 = time.time()
    try:
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
    except Exception as exc:
        print(f"ERROR loading model: {exc}")
        if device == "cuda":
            print("Hint: try --device cpu as a fallback.")
        sys.exit(1)
    print(f"Model loaded in {time.time()-t0:.1f}s")
    print()

    # Transcribe with word timestamps
    print(f"Transcribing {audio_path.name} ...")
    t1 = time.time()
    segments, info = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        language="en",
        beam_size=5,
        vad_filter=True,               # skip silence with VAD
        vad_parameters=dict(
            min_silence_duration_ms=500,   # gaps > 0.5s treated as segment breaks
        ),
    )

    # Collect all segments (generator — must iterate to run inference)
    all_segments = []
    all_words = []
    full_text_parts = []

    print("Processing segments...")
    for seg in segments:
        seg_dict = {
            "id": seg.id,
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
            "avg_logprob": round(seg.avg_logprob, 4),
            "no_speech_prob": round(seg.no_speech_prob, 4),
            "words": [],
        }
        if seg.words:
            for w in seg.words:
                word_dict = {
                    "word": w.word,
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                    "probability": round(w.probability, 4),
                }
                seg_dict["words"].append(word_dict)
                all_words.append(word_dict)

        all_segments.append(seg_dict)
        full_text_parts.append(seg.text.strip())
        print(f"  [{seg.start:7.2f}s – {seg.end:7.2f}s]  {seg.text.strip()[:80]}")

    elapsed = time.time() - t1
    audio_dur = info.duration
    rtf = elapsed / audio_dur  # real-time factor

    print()
    print(f"Transcription complete in {elapsed:.1f}s")
    print(f"Audio duration detected : {audio_dur:.3f}s ({int(audio_dur//60)}m {audio_dur%60:.1f}s)")
    print(f"Real-time factor        : {rtf:.2f}x  (1.0 = real time, <1.0 = faster)")
    print(f"Segments                : {len(all_segments)}")
    print(f"Words (with timestamps) : {len(all_words)}")
    print()

    # ---- Save plain text transcript ----
    full_text = "\n".join(full_text_parts)
    Path(TRANSCRIPT_TXT).write_text(full_text, encoding="utf-8")
    print(f"Saved: {TRANSCRIPT_TXT}  ({len(full_text)} chars, {len(full_text.split())} words)")

    # ---- Save structured JSON ----
    output = {
        "audio_file": str(audio_path),
        "model": model_name,
        "device": device,
        "audio_duration_seconds": round(audio_dur, 3),
        "transcription_time_seconds": round(elapsed, 1),
        "real_time_factor": round(rtf, 3),
        "language": info.language,
        "language_probability": round(info.language_probability, 4),
        "total_segments": len(all_segments),
        "total_words": len(all_words),
        "segments": all_segments,
    }
    Path(TRANSCRIPT_JSON).write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved: {TRANSCRIPT_JSON}  ({len(all_segments)} segments, {len(all_words)} words)")
    print()

    # ---- Quick stats ----
    word_count = len(full_text.split())
    wpm = word_count / (audio_dur / 60) if audio_dur > 0 else 0
    avg_conf = sum(w["probability"] for w in all_words) / len(all_words) if all_words else 0
    low_conf = [w for w in all_words if w["probability"] < 0.6]

    print("=== Transcript Statistics ===")
    print(f"Word count       : {word_count}")
    print(f"Narration speed  : {wpm:.1f} wpm")
    print(f"Avg word conf.   : {avg_conf:.3f}  (1.0 = certain)")
    print(f"Low-conf words   : {len(low_conf)}  (< 0.6 probability)")
    if low_conf:
        print("  Low-confidence words:")
        for w in low_conf[:15]:
            print(f"    {w['start']:.2f}s  '{w['word'].strip()}'  ({w['probability']:.2f})")

    print()
    print("=== Sample (first 300 chars) ===")
    print(full_text[:300])
    print()
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe narration.mp3 with faster-whisper")
    parser.add_argument("--model", default="medium", choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--audio", default=AUDIO_FILE)
    args = parser.parse_args()
    run(model_name=args.model, device=args.device, audio=args.audio)


if __name__ == "__main__":
    main()

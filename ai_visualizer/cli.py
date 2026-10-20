"""
cli.py
------
Command-line interface for AI Visualizer.

Usage examples
--------------
# Phase 1 – parse only
python -m ai_visualizer.cli analyze --script my_script.txt

# Phase 1 – with audio
python -m ai_visualizer.cli analyze --script my_script.txt --audio narration.mp3

# Phase 2 – enhance with Gemini
python -m ai_visualizer.cli analyze --script my_script.txt --audio narration.mp3 --gemini

# Phase 3 – also generate images (placeholder by default)
python -m ai_visualizer.cli analyze --script my_script.txt --audio narration.mp3 --gemini --images

# Phase 4 – launch the web UI
python -m ai_visualizer.cli ui
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def cmd_analyze(args: argparse.Namespace) -> None:
    from .scene_parser import parse_script
    from .timestamp_mapper import assign_timestamps, assign_timestamps_no_audio
    from .exporter import to_csv, to_json

    script_path = Path(args.script)
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        sys.exit(1)

    print(f"📖 Parsing: {script_path.name}")
    scenes = parse_script(script_path)
    print(f"   → {len(scenes)} scenes detected")

    if args.audio:
        audio_path = Path(args.audio)
        if not audio_path.exists():
            print(f"❌ Audio not found: {audio_path}")
            sys.exit(1)
        print(f"🎵 Mapping timestamps from: {audio_path.name}")
        assign_timestamps(scenes, audio_path)
    else:
        print("⚠  No audio provided – estimating timestamps at 130 wpm")
        assign_timestamps_no_audio(scenes)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = to_csv(scenes, out_dir / "scenes.csv")
    json_path = to_json(scenes, out_dir / "scenes.json")
    print(f"✅ CSV  → {csv_path}")
    print(f"✅ JSON → {json_path}")

    # Phase 2 – Gemini
    if args.gemini:
        from .gemini_client import is_available, enhance_scenes
        if not is_available():
            print("⚠  Gemini API key not configured – skipping enhancement")
        else:
            print("✨ Enhancing scenes with Gemini…")
            enhance_scenes(scenes)
            csv_path = to_csv(scenes, out_dir / "scenes_gemini.csv")
            json_path = to_json(scenes, out_dir / "scenes_gemini.json")
            print(f"✅ Gemini CSV  → {csv_path}")
            print(f"✅ Gemini JSON → {json_path}")

    # Phase 3 – Images
    if args.images:
        from .image_generator import generate_images
        # Use scene text as prompt fallback when Gemini hasn't run
        for s in scenes:
            if not s.image_prompt:
                s.image_prompt = s.text
        print(f"🖼  Generating images (backend: {args.backend})…")
        generate_images(scenes, output_dir=str(out_dir), backend=args.backend)

    # Print scene summary
    print("\n── Scene Summary ──────────────────────────────────────")
    for s in scenes:
        ts = f"[{s.start_time:.1f}s – {s.end_time:.1f}s]"
        prompt_preview = f" | prompt: {s.image_prompt[:50]}…" if s.image_prompt else ""
        print(f"  {s.index:02d}. {ts}  {s.title}{prompt_preview}")


def cmd_ui(args: argparse.Namespace) -> None:
    from .ui import launch
    print("🚀 Launching AI Visualizer UI…")
    launch(share=args.share, port=args.port)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ai-visualizer",
        description="Automate scenes, timestamps, and images from a script.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # analyze subcommand
    p_analyze = sub.add_parser("analyze", help="Parse script and assign timestamps")
    p_analyze.add_argument("--script", required=True, help="Path to .txt or .md script")
    p_analyze.add_argument("--audio", default=None, help="Path to .mp3 or .wav audio")
    p_analyze.add_argument("--output", default="outputs", help="Output directory (default: outputs)")
    p_analyze.add_argument("--gemini", action="store_true", help="Enhance scenes with Gemini")
    p_analyze.add_argument("--images", action="store_true", help="Generate images")
    p_analyze.add_argument(
        "--backend",
        default="placeholder",
        choices=["placeholder", "stability", "replicate"],
        help="Image backend (default: placeholder)",
    )

    # ui subcommand
    p_ui = sub.add_parser("ui", help="Launch the web interface")
    p_ui.add_argument("--share", action="store_true", help="Create a public Gradio link")
    p_ui.add_argument("--port", type=int, default=7860, help="Port (default: 7860)")

    args = parser.parse_args()

    if args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "ui":
        cmd_ui(args)


if __name__ == "__main__":
    main()

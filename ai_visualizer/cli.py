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


def cmd_generate_images(args: argparse.Namespace) -> None:
    """Generate images from an existing scene_plan.json."""
    from pathlib import Path
    from .image_runner import RunConfig, run_image_generation

    cfg = RunConfig(
        scene_plan_path=Path(args.plan),
        output_dir=Path(args.output),
        manifest_path=(
            Path(args.manifest)
            if args.manifest
            else Path(args.output).parent / "image_manifest.json"
        ),
        backend=args.backend,
        delay_seconds=args.delay,
        max_retries=args.retries,
        force=args.force,
        dry_run=args.dry_run,
        scene_indexes=[int(x) for x in args.scenes.split(",")] if args.scenes else None,
        aspect_ratio=args.aspect_ratio,
        concurrency=args.concurrency,
    )

    if cfg.dry_run:
        print("Dry run - no images will be generated.\n")
    else:
        print(f"Generating images (backend: {cfg.backend})...\n")

    results = run_image_generation(cfg)

    if not cfg.dry_run:
        ok = sum(1 for r in results if r.status == "ok")
        skip = sum(1 for r in results if r.status == "skipped")
        fail = sum(1 for r in results if r.status == "error")
        print(f"\nDone: {ok} generated, {skip} skipped, {fail} failed")
        print(f"Manifest: {cfg.manifest_path}")


def cmd_ui(args: argparse.Namespace) -> None:
    from .studio import launch
    print("Launching Narrato Studio...")
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
        default=None,
        choices=["mock", "api", "local", "placeholder", "stability", "replicate"],
        help="Image backend (default: IMAGE_BACKEND, then mock)",
    )

    # generate-images subcommand
    p_gen = sub.add_parser("generate-images", help="Generate images from a scene_plan.json")
    p_gen.add_argument("--plan",     default="scene_plan.json",
                       help="Path to scene_plan.json (default: scene_plan.json)")
    p_gen.add_argument("--output",   default="outputs/images",
                       help="Output directory for images (default: outputs/images)")
    p_gen.add_argument("--backend",  default=None,
                       choices=["mock", "api", "local", "placeholder", "stability", "replicate"],
                       help="Image backend (default: IMAGE_BACKEND, then mock)")
    p_gen.add_argument("--manifest", default=None,
                       help="Manifest path (default: outputs/image_manifest.json)")
    p_gen.add_argument("--aspect-ratio", default="16:9",
                       help="Image aspect ratio (default: 16:9)")
    p_gen.add_argument("--concurrency", type=int, default=1,
                       help="Maximum concurrent image jobs (default: 1)")
    p_gen.add_argument("--delay",    type=float, default=2.0,
                       help="Seconds between API calls (default: 2.0)")
    p_gen.add_argument("--retries",  type=int,   default=2,
                       help="Max retries per scene on failure (default: 2)")
    p_gen.add_argument("--force",    action="store_true",
                       help="Regenerate images even if they already exist")
    p_gen.add_argument("--dry-run",  action="store_true",
                       help="Validate and print plan without making API calls")
    p_gen.add_argument("--scenes",   default=None,
                       help="Comma-separated scene indexes to process (e.g. 1,3,5)")

    # ui subcommand
    p_ui = sub.add_parser("ui", help="Launch the web interface")
    p_ui.add_argument("--share", action="store_true", help="Create a public Gradio link")
    p_ui.add_argument("--port", type=int, default=7860, help="Port (default: 7860)")

    args = parser.parse_args()

    if args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "generate-images":
        cmd_generate_images(args)
    elif args.command == "ui":
        cmd_ui(args)


if __name__ == "__main__":
    main()

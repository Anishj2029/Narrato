"""
ui.py
-----
Phase 4 – Simple Gradio interface.

Tabs:
  1. Upload & Analyze  – script + audio → scenes CSV/JSON
  2. Review Scenes     – table view, edit image prompts
  3. Generate Images   – queue image jobs, show gallery

Run: python -m ai_visualizer.ui
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import List, Tuple

import gradio as gr
import pandas as pd

from .scene_parser import parse_script_text, Scene
from .timestamp_mapper import assign_timestamps, assign_timestamps_no_audio
from .exporter import to_csv, to_json
from .gemini_client import is_available as gemini_available, enhance_scenes
from .image_generator import generate_images

# Temp working dir for this session
_TMP = Path(tempfile.mkdtemp(prefix="ai_visualizer_"))
_SCENES: List[Scene] = []  # session state (simple global for single-user UI)


# ---------------------------------------------------------------------------
# Tab 1 – Upload & Analyze
# ---------------------------------------------------------------------------

def analyze(script_file, audio_file) -> Tuple[str, str, pd.DataFrame]:
    """Parse script, assign timestamps, export CSV + JSON."""
    global _SCENES

    if script_file is None:
        return "❌ Please upload a script file.", "", pd.DataFrame()

    # Read script
    script_path = Path(script_file.name)
    raw_text = script_path.read_text(encoding="utf-8")

    _SCENES = parse_script_text(raw_text)

    if not _SCENES:
        return "❌ No scenes detected. Check your script format.", "", pd.DataFrame()

    # Assign timestamps
    if audio_file is not None:
        assign_timestamps(_SCENES, audio_file.name)
        ts_note = "✅ Real timestamps from audio"
    else:
        assign_timestamps_no_audio(_SCENES)
        ts_note = "⚠ Estimated timestamps (no audio)"

    # Export
    csv_path = _TMP / "scenes.csv"
    json_path = _TMP / "scenes.json"
    to_csv(_SCENES, csv_path)
    to_json(_SCENES, json_path)

    summary = (
        f"✅ Found **{len(_SCENES)} scenes**  |  {ts_note}\n\n"
        f"📄 CSV: `{csv_path}`\n"
        f"📄 JSON: `{json_path}`"
    )

    df = _scenes_to_df(_SCENES)
    return summary, str(json_path), df


# ---------------------------------------------------------------------------
# Tab 2 – Review & Enhance
# ---------------------------------------------------------------------------

def enhance_with_gemini() -> Tuple[str, pd.DataFrame]:
    global _SCENES

    if not _SCENES:
        return "❌ No scenes loaded. Go to Upload & Analyze first.", pd.DataFrame()

    if not gemini_available():
        return (
            "❌ Gemini API key not configured.\n"
            "Add GEMINI_API_KEY to your .env file.",
            _scenes_to_df(_SCENES),
        )

    try:
        enhance_scenes(_SCENES)
        status = f"✅ Gemini enhanced {len(_SCENES)} scenes."
    except Exception as exc:
        status = f"❌ Gemini error: {exc}"

    return status, _scenes_to_df(_SCENES)


def save_edits(edited_df: pd.DataFrame) -> str:
    """Sync edits from the dataframe back into _SCENES."""
    global _SCENES

    if _SCENES and edited_df is not None:
        for _, row in edited_df.iterrows():
            idx = int(row["index"])
            for scene in _SCENES:
                if scene.index == idx:
                    scene.image_prompt = str(row.get("image_prompt", ""))
                    break
        # Re-export
        to_csv(_SCENES, _TMP / "scenes_edited.csv")
        to_json(_SCENES, _TMP / "scenes_edited.json")
        return "✅ Edits saved."
    return "⚠ Nothing to save."


# ---------------------------------------------------------------------------
# Tab 3 – Generate Images
# ---------------------------------------------------------------------------

def generate(backend_choice: str) -> Tuple[str, List]:
    global _SCENES

    if not _SCENES:
        return "❌ No scenes loaded.", []

    missing_prompts = [s for s in _SCENES if not s.image_prompt]
    if missing_prompts:
        # Use scene text as fallback prompt
        for s in missing_prompts:
            s.image_prompt = s.text

    out_dir = str(_TMP / "outputs")
    generate_images(_SCENES, output_dir=out_dir, backend=backend_choice)

    images = [s.image_path for s in _SCENES if s.image_path and Path(s.image_path).exists()]
    status = f"✅ Generated {len(images)} / {len(_SCENES)} images  →  {out_dir}"
    manifest_path = Path(out_dir) / "image_manifest.json"
    if manifest_path.exists():
        entries = json.loads(manifest_path.read_text(encoding="utf-8"))
        failures = [
            f"Scene {entry['index']}: {entry['error']}"
            for entry in entries
            if entry.get("status") == "error"
        ]
        if failures:
            status += "\n\nErrors:\n" + "\n".join(failures)

    # Re-export CSV with image paths
    to_csv(_SCENES, _TMP / "scenes_final.csv")

    return status, images


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _scenes_to_df(scenes: List[Scene]) -> pd.DataFrame:
    rows = [
        {
            "index": s.index,
            "title": s.title,
            "start": f"{s.start_time:.1f}s",
            "end": f"{s.end_time:.1f}s",
            "text": s.text[:120] + ("…" if len(s.text) > 120 else ""),
            "image_prompt": s.image_prompt,
        }
        for s in scenes
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="AI Visualizer", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎬 AI Visualizer\nAutomate scenes, timestamps, and images from your script.")

        # ---- Tab 1 ----
        with gr.Tab("📤 Upload & Analyze"):
            with gr.Row():
                script_upload = gr.File(label="Script (.txt or .md)", file_types=[".txt", ".md"])
                audio_upload = gr.File(label="Audio (.mp3 or .wav) — optional",
                                       file_types=[".mp3", ".wav"])
            analyze_btn = gr.Button("Analyze", variant="primary")
            analyze_status = gr.Markdown()
            json_output = gr.Textbox(label="JSON output path", interactive=False)
            scenes_table_1 = gr.DataFrame(label="Scenes preview")

            analyze_btn.click(
                analyze,
                inputs=[script_upload, audio_upload],
                outputs=[analyze_status, json_output, scenes_table_1],
            )

        # ---- Tab 2 ----
        with gr.Tab("✏️ Review & Enhance"):
            with gr.Row():
                enhance_btn = gr.Button("Enhance with Gemini", variant="primary")
                save_btn = gr.Button("Save Edits")
            review_status = gr.Markdown()
            scenes_table_2 = gr.DataFrame(
                label="Scenes (edit image_prompt column freely)",
                interactive=True,
            )

            enhance_btn.click(
                enhance_with_gemini,
                outputs=[review_status, scenes_table_2],
            )
            save_btn.click(
                save_edits,
                inputs=[scenes_table_2],
                outputs=[review_status],
            )

        # ---- Tab 3 ----
        with gr.Tab("🖼️ Generate Images"):
            backend_radio = gr.Radio(
                choices=["mock", "api", "local"],
                value="mock",
                label="Image backend",
                info="Use 'mock' to test without API calls or local inference.",
            )
            gen_btn = gr.Button("Generate", variant="primary")
            gen_status = gr.Markdown()
            gallery = gr.Gallery(label="Generated images", columns=3, height="auto")

            gen_btn.click(
                generate,
                inputs=[backend_radio],
                outputs=[gen_status, gallery],
            )

    return demo


def launch(share: bool = False, port: int = 7860) -> None:
    ui = build_ui()
    ui.launch(share=share, server_port=port)


if __name__ == "__main__":
    launch()

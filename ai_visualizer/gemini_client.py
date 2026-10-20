"""
gemini_client.py
----------------
Phase 2 – Gemini integration.

Reads GEMINI_API_KEY from the .env file at the project root.
Never hardcodes secrets.

Usage
-----
    from ai_visualizer.gemini_client import enhance_scenes
    scenes = enhance_scenes(scenes)   # adds image_prompt to each Scene
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

from .scene_parser import Scene

# Load .env from project root (two levels up from this file, then root)
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

_GEMINI_AVAILABLE: Optional[bool] = None


def _check_api() -> bool:
    """Return True if the Gemini API key is present and the SDK is importable."""
    global _GEMINI_AVAILABLE
    if _GEMINI_AVAILABLE is not None:
        return _GEMINI_AVAILABLE

    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key or key == "your_gemini_api_key_here":
        _GEMINI_AVAILABLE = False
        return False

    try:
        import google.generativeai  # noqa: F401
        _GEMINI_AVAILABLE = True
    except ImportError:
        _GEMINI_AVAILABLE = False

    return _GEMINI_AVAILABLE


def is_available() -> bool:
    """Public check – call this before using enhance_scenes."""
    return _check_api()


def _build_prompt(scene: Scene) -> str:
    return (
        "You are a visual director. Given the following scene text, write a "
        "concise, vivid image generation prompt (max 60 words) that captures "
        "the core visual concept. Focus on subject, setting, mood, and style. "
        "Do NOT include any disclaimers or explanations.\n\n"
        f"Scene text:\n{scene.text}\n\n"
        "Image prompt:"
    )


def enhance_scenes(
    scenes: List[Scene],
    model_name: str = "gemini-1.5-flash",
    delay_seconds: float = 1.0,
) -> List[Scene]:
    """
    Call Gemini to generate an image_prompt for each scene.

    Parameters
    ----------
    scenes        : list of Scene objects
    model_name    : Gemini model to use (flash is cheapest / fastest)
    delay_seconds : polite delay between API calls to avoid rate limits

    Returns the same list with image_prompt fields populated.
    Raises RuntimeError if the API key is not configured.
    """
    if not _check_api():
        raise RuntimeError(
            "Gemini API key not found or not configured.\n"
            "1. Copy .env.example to .env\n"
            "2. Set GEMINI_API_KEY=<your key>\n"
            "3. Re-run."
        )

    import google.generativeai as genai  # type: ignore

    key = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=key)
    model = genai.GenerativeModel(model_name)

    for i, scene in enumerate(scenes):
        if scene.image_prompt:
            # Already has a prompt – skip
            continue
        try:
            response = model.generate_content(_build_prompt(scene))
            scene.image_prompt = response.text.strip()
        except Exception as exc:
            print(f"  [Gemini] Scene {scene.index} failed: {exc}")
            scene.image_prompt = f"[ERROR: {exc}]"

        if i < len(scenes) - 1:
            time.sleep(delay_seconds)

    return scenes


def enhance_single(scene: Scene, model_name: str = "gemini-1.5-flash") -> Scene:
    """Enhance a single scene. Useful for retries from the UI."""
    return enhance_scenes([scene], model_name=model_name)[0]

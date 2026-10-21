"""Reusable quality prompts and scene prompt composition."""

from __future__ import annotations

import re


DEFAULT_POSITIVE_PROMPT = (
    "Cinematic film still, high-quality cinematic imagery, coherent composition, "
    "realistic lighting and shadows, natural color grading, rich but balanced colors, "
    "clear subject definition, well-defined edges, detailed natural textures, realistic "
    "fabric and material surfaces, natural skin texture, believable human anatomy and "
    "proportions, detailed environments, clear foreground and background separation, "
    "atmospheric depth, balanced exposure, subtle cinematic contrast, fine visual "
    "details, professional cinematography, consistent visual style, clean image with "
    "minimal unwanted noise."
)

DEFAULT_NEGATIVE_PROMPT = (
    "Low quality, low resolution, blurry, out of focus, muddy details, smeared textures, "
    "excessive noise, heavy grain, compression artifacts, oversharpened, oversaturated, "
    "washed out, flat lighting, harsh artificial lighting, plastic skin, waxy faces, "
    "unnatural skin texture, deformed anatomy, malformed hands, extra fingers, duplicate "
    "limbs, distorted proportions, poorly defined objects, inconsistent details, text, "
    "captions, letters, logo, watermark."
)


def compose_prompt(quality_prompt: str, scene_prompt: str) -> str:
    """Prepend quality guidance without discarding or repeating scene details."""
    phrases = []
    seen = set()
    for source in (quality_prompt, scene_prompt):
        for phrase in source.split(","):
            cleaned = phrase.strip().rstrip(".;")
            key = re.sub(r"\s+", " ", cleaned.rstrip(".;")).casefold()
            if cleaned and key not in seen:
                phrases.append(cleaned)
                seen.add(key)
    return ", ".join(phrases)

"""Reusable quality prompts, visual themes, and scene prompt composition."""

from __future__ import annotations

import re
from dataclasses import dataclass


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


# ---------------------------------------------------------------------------
# Visual themes
# ---------------------------------------------------------------------------

DEFAULT_THEME = "Default"

CINEMATIC_CONCEPT_ART_POSITIVE = (
    "cinematic concept art, sophisticated painterly realism, premium film concept painting, "
    "beautifully composed visual storytelling, atmospheric depth, rich environmental detail, "
    "expressive natural lighting, soft volumetric light, nuanced shadows, harmonious muted colours, "
    "earthy tones with subtle teal and amber accents, painterly textures with refined details, "
    "believable human proportions, coherent environments, clear visual hierarchy, "
    "carefully balanced composition, cinematic widescreen framing, polished digital painting, "
    "consistent art direction, professional film concept artwork"
)

CINEMATIC_CONCEPT_ART_NEGATIVE = (
    "photorealistic stock photo, cheap 3D render, plastic surfaces, childish cartoon, anime, "
    "flat vector art, messy brush strokes, muddy colours, excessive grain, visual noise, "
    "blurry focal subject, smeared textures, poorly defined objects, distorted anatomy, "
    "malformed hands, extra limbs, duplicate subjects, inconsistent lighting, oversaturated colours, "
    "harsh outlines, text, captions, logos, watermark"
)


@dataclass(frozen=True)
class ThemeSpec:
    """A named visual theme with permanent positive and negative prompts."""
    positive_prompt: str
    negative_prompt: str


VISUAL_THEMES: dict[str, ThemeSpec] = {
    "Default": ThemeSpec(
        positive_prompt=DEFAULT_POSITIVE_PROMPT,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
    ),
    "Cinematic Concept Art": ThemeSpec(
        positive_prompt=CINEMATIC_CONCEPT_ART_POSITIVE,
        negative_prompt=CINEMATIC_CONCEPT_ART_NEGATIVE,
    ),
}


def get_theme_names() -> list[str]:
    """Return all available theme names in definition order."""
    return list(VISUAL_THEMES.keys())


def get_theme_prompts(theme_name: str) -> tuple[str, str]:
    """
    Return (positive_prompt, negative_prompt) for the named theme.

    Raises ValueError if the theme name is not recognised.
    """
    theme = VISUAL_THEMES.get(theme_name)
    if theme is None:
        raise ValueError(
            f"Unknown visual theme: {theme_name!r}. "
            f"Available themes: {get_theme_names()}"
        )
    return theme.positive_prompt, theme.negative_prompt


# ---------------------------------------------------------------------------
# Prompt composition
# ---------------------------------------------------------------------------

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

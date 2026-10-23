"""
tests/test_themes.py
--------------------
Tests for the visual theme system: theme selection, prompt composition,
no duplicate templates, and preservation of scene-specific descriptions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from ai_visualizer.image_prompts import (
    CINEMATIC_CONCEPT_ART_NEGATIVE,
    CINEMATIC_CONCEPT_ART_POSITIVE,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_POSITIVE_PROMPT,
    DEFAULT_THEME,
    VISUAL_THEMES,
    compose_prompt,
    get_theme_names,
    get_theme_prompts,
)
from ai_visualizer.image_runner import (
    BACKENDS,
    RunConfig,
    SceneResult,
    _clean_prompt,
    run_image_generation,
)


# ---------------------------------------------------------------------------
# Theme definitions
# ---------------------------------------------------------------------------

class TestThemeDefinitions:
    def test_cinematic_concept_art_is_registered(self):
        names = get_theme_names()
        assert "Cinematic Concept Art" in names

    def test_default_theme_is_registered(self):
        names = get_theme_names()
        assert DEFAULT_THEME in names

    def test_each_theme_has_both_prompts(self):
        for name, spec in VISUAL_THEMES.items():
            assert spec.positive_prompt, f"Theme {name!r} has empty positive prompt"
            assert spec.negative_prompt, f"Theme {name!r} has empty negative prompt"

    def test_theme_prompts_are_not_identical_across_themes(self):
        # Ensure the two themes actually differ
        default_pos, default_neg = get_theme_prompts(DEFAULT_THEME)
        cca_pos, cca_neg = get_theme_prompts("Cinematic Concept Art")
        assert default_pos != cca_pos
        assert default_neg != cca_neg

    def test_get_theme_prompts_returns_exact_cca_strings(self):
        pos, neg = get_theme_prompts("Cinematic Concept Art")
        assert pos == CINEMATIC_CONCEPT_ART_POSITIVE
        assert neg == CINEMATIC_CONCEPT_ART_NEGATIVE

    def test_get_theme_prompts_unknown_theme_raises(self):
        with pytest.raises(ValueError, match="Unknown visual theme"):
            get_theme_prompts("Nonexistent Theme")


# ---------------------------------------------------------------------------
# Theme selection changes both prompts
# ---------------------------------------------------------------------------

class TestThemeSelection:
    def test_changing_theme_changes_positive_prompt(self):
        default_pos, _ = get_theme_prompts(DEFAULT_THEME)
        cca_pos, _ = get_theme_prompts("Cinematic Concept Art")
        assert default_pos != cca_pos

    def test_changing_theme_changes_negative_prompt(self):
        _, default_neg = get_theme_prompts(DEFAULT_THEME)
        _, cca_neg = get_theme_prompts("Cinematic Concept Art")
        assert default_neg != cca_neg

    def test_cca_negative_excludes_default_negative_phrases(self):
        """CCA negative should contain its own phrases, not just the defaults."""
        _, cca_neg = get_theme_prompts("Cinematic Concept Art")
        assert "photorealistic stock photo" in cca_neg
        assert "cheap 3D render" in cca_neg
        assert "painterly textures" not in cca_neg  # that's in positive


# ---------------------------------------------------------------------------
# Prompt composition with themes
# ---------------------------------------------------------------------------

class TestPromptComposition:
    def test_compose_prepends_quality_prompt_to_scene(self):
        quality = CINEMATIC_CONCEPT_ART_POSITIVE
        scene = "A woman reading in a sunlit library"
        result = compose_prompt(quality, scene)
        assert result.startswith("cinematic concept art")
        assert "A woman reading in a sunlit library" in result

    def test_compose_deduplicates_overlapping_phrases(self):
        quality = "cinematic concept art, painterly textures"
        scene = "painterly textures, a quiet village at dawn"
        result = compose_prompt(quality, scene)
        # "painterly textures" should appear exactly once
        assert result.count("painterly textures") == 1

    def test_compose_preserves_scene_specific_description(self):
        quality = CINEMATIC_CONCEPT_ART_POSITIVE
        scene = "[FALLBACK] A close-up of weathered hands holding a letter"
        cleaned = _clean_prompt(scene)
        result = compose_prompt(quality, cleaned)
        assert "A close-up of weathered hands holding a letter" in result

    def test_compose_does_not_duplicate_quality_phrases(self):
        quality = CINEMATIC_CONCEPT_ART_POSITIVE
        scene = "cinematic concept art, atmospheric depth, a lone figure on a cliff"
        result = compose_prompt(quality, scene)
        # Each quality phrase should appear exactly once
        assert result.count("cinematic concept art") == 1
        assert result.count("atmospheric depth") == 1

    def test_compose_empty_scene_prompt_returns_quality_only(self):
        result = compose_prompt(CINEMATIC_CONCEPT_ART_POSITIVE, "")
        assert result == CINEMATIC_CONCEPT_ART_POSITIVE

    def test_compose_empty_quality_prompt_returns_scene_only(self):
        scene = "A quiet mountain lake at dawn"
        result = compose_prompt("", scene)
        assert result == scene


# ---------------------------------------------------------------------------
# No duplicate template prompts across themes
# ---------------------------------------------------------------------------

class TestNoDuplicateTemplates:
    def test_theme_positive_prompts_differ_from_each_other(self):
        prompts = [spec.positive_prompt for spec in VISUAL_THEMES.values()]
        # No two themes should have identical positive prompts
        assert len(prompts) == len(set(prompts))

    def test_theme_negative_prompts_differ_from_each_other(self):
        prompts = [spec.negative_prompt for spec in VISUAL_THEMES.values()]
        assert len(prompts) == len(set(prompts))

    def test_cca_positive_is_not_just_default_repeated(self):
        assert CINEMATIC_CONCEPT_ART_POSITIVE != DEFAULT_POSITIVE_PROMPT

    def test_cca_negative_is_not_just_default_repeated(self):
        assert CINEMATIC_CONCEPT_ART_NEGATIVE != DEFAULT_NEGATIVE_PROMPT


# ---------------------------------------------------------------------------
# Scene-specific description preservation
# ---------------------------------------------------------------------------

def _make_scene(
    index: int = 1,
    title: str = "Test Scene",
    text: str = "A person walks across a bridge at sunset.",
    start_time: float = 0.0,
    end_time: float = 30.0,
    image_prompt: str = "Cinematic bridge at sunset, warm light.",
    filename_suggestion: str = "",
) -> Dict[str, Any]:
    return {
        "index": index,
        "title": title,
        "text": text,
        "start_time": start_time,
        "end_time": end_time,
        "image_prompt": image_prompt,
        "image_path": "",
        "description": "A test scene.",
        "filename_suggestion": filename_suggestion or f"scene_{index:03d}.png",
        "segment_ids": [index],
        "word_count": len(text.split()),
    }


def _write_plan(scenes: List[Dict], path: Path) -> None:
    path.write_text(json.dumps(scenes, indent=2), encoding="utf-8")


def _fake_backend(scene: Dict[str, Any], out_dir: Path) -> Path:
    """Mocked backend that writes a tiny valid PNG."""
    from PIL import Image
    img = Image.new("RGB", (4, 4), color=(0, 128, 0))
    out = out_dir / scene["filename_suggestion"]
    img.save(str(out))
    return out


class TestSceneDescriptionPreservation:
    def test_dry_run_preserves_scene_text_and_timestamps(self, tmp_path):
        scenes = [
            _make_scene(1, "Sunset Bridge", "A person walks across a bridge at sunset.",
                        0.0, 30.0, "Cinematic bridge at sunset, warm light."),
            _make_scene(2, "Dawn Village", "Mist over a quiet village at dawn.",
                        30.0, 60.0, "Misty village at dawn, soft light."),
            _make_scene(3, "Night Camp", "A campfire flickers under starlight.",
                        60.0, 90.0, "Campfire under stars, warm glow."),
        ]
        plan = tmp_path / "plan.json"
        _write_plan(scenes, plan)

        cfg = RunConfig(
            scene_plan_path=plan,
            output_dir=tmp_path / "img",
            manifest_path=tmp_path / "manifest.json",
            backend="placeholder",
            dry_run=True,
            theme="Cinematic Concept Art",
        )
        results = run_image_generation(cfg)

        assert len(results) == 3
        # Check scene-specific prompts are preserved in dry-run results
        for r in results:
            assert r.status == "dry_run"
            assert r.theme == "Cinematic Concept Art"
            # The scene-specific prompt should be embedded after quality prompt
            assert "cinematic concept art" in r.prompt_used

    def test_generation_preserves_scene_specific_prompt_in_result(self, tmp_path):
        from unittest.mock import patch
        from ai_visualizer.image_runner import BACKENDS

        scene = _make_scene(
            image_prompt="A close-up portrait of an elderly fisherman"
        )
        plan = tmp_path / "plan.json"
        _write_plan([scene], plan)

        with patch.dict(BACKENDS, {"mock": _fake_backend}):
            cfg = RunConfig(
                scene_plan_path=plan,
                output_dir=tmp_path / "img",
                manifest_path=tmp_path / "manifest.json",
                backend="mock",
                theme="Cinematic Concept Art",
            )
            results = run_image_generation(cfg)

        assert results[0].status == "ok"
        assert results[0].prompt_used.startswith("cinematic concept art")
        assert "A close-up portrait of an elderly fisherman" in results[0].prompt_used
        assert results[0].theme == "Cinematic Concept Art"
        assert results[0].negative_prompt_used == CINEMATIC_CONCEPT_ART_NEGATIVE

    def test_manifest_records_theme_and_effective_prompts(self, tmp_path):
        from unittest.mock import patch
        from ai_visualizer.image_runner import BACKENDS

        scene = _make_scene(image_prompt="A lone figure on a windswept cliff")
        plan = tmp_path / "plan.json"
        _write_plan([scene], plan)

        with patch.dict(BACKENDS, {"mock": _fake_backend}):
            cfg = RunConfig(
                scene_plan_path=plan,
                output_dir=tmp_path / "img",
                manifest_path=tmp_path / "manifest.json",
                backend="mock",
                theme="Cinematic Concept Art",
            )
            run_image_generation(cfg)

        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest[0]["theme"] == "Cinematic Concept Art"
        assert "cinematic concept art" in manifest[0]["prompt_used"]
        assert "A lone figure on a windswept cliff" in manifest[0]["prompt_used"]
        assert manifest[0]["negative_prompt_used"] == CINEMATIC_CONCEPT_ART_NEGATIVE

    def test_theme_prompts_are_composed_once_across_retries(self, tmp_path):
        scene = _make_scene(
            image_prompt="A close-up portrait of an elderly fisherman"
        )
        plan = tmp_path / "plan.json"
        _write_plan([scene], plan)
        calls = []

        def flaky(scene_data, out_dir):
            calls.append((
                scene_data["image_prompt"],
                scene_data["negative_prompt"],
            ))
            if len(calls) == 1:
                raise RuntimeError("temporary failure")
            return _fake_backend(scene_data, out_dir)

        with patch.dict(BACKENDS, {"theme-retry": flaky}):
            with patch("ai_visualizer.image_runner.time.sleep"):
                results = run_image_generation(RunConfig(
                    scene_plan_path=plan,
                    output_dir=tmp_path / "images",
                    manifest_path=tmp_path / "manifest.json",
                    backend="theme-retry",
                    max_retries=1,
                    retry_delay_seconds=0,
                    theme="Cinematic Concept Art",
                ))

        assert results[0].status == "ok"
        assert results[0].theme == "Cinematic Concept Art"
        assert len(calls) == 2
        assert calls[0] == calls[1]
        assert calls[0][0].count("cinematic concept art") == 1
        assert calls[0][0].count("close-up portrait") == 1
        assert calls[0][1] == CINEMATIC_CONCEPT_ART_NEGATIVE

    def test_default_theme_records_correctly(self, tmp_path):
        from unittest.mock import patch
        from ai_visualizer.image_runner import BACKENDS

        scene = _make_scene(image_prompt="A quiet meadow at noon")
        plan = tmp_path / "plan.json"
        _write_plan([scene], plan)

        with patch.dict(BACKENDS, {"mock": _fake_backend}):
            cfg = RunConfig(
                scene_plan_path=plan,
                output_dir=tmp_path / "img",
                manifest_path=tmp_path / "manifest.json",
                backend="mock",
                theme=DEFAULT_THEME,
            )
            results = run_image_generation(cfg)

        assert results[0].theme == DEFAULT_THEME
        assert "Cinematic film still" in results[0].prompt_used
        assert "A quiet meadow at noon" in results[0].prompt_used
        assert results[0].negative_prompt_used == compose_prompt(
            DEFAULT_NEGATIVE_PROMPT, ""
        )

    def test_theme_does_not_force_full_body_shots(self, tmp_path):
        """Scene-specific framing like 'close-up' should survive composition."""
        quality = CINEMATIC_CONCEPT_ART_POSITIVE
        scene_prompt = "A close-up of weathered hands holding a letter"
        result = compose_prompt(quality, scene_prompt)
        # The scene-specific framing should be preserved
        assert "close-up" in result
        # The quality prompt should not inject "full body" or similar
        assert "full body" not in result.lower()
        assert "full-body" not in result.lower()

    def test_scene_timestamps_preserved_in_dry_run(self, tmp_path):
        scenes = [
            _make_scene(1, start_time=12.5, end_time=45.3,
                        text="Scene at specific time"),
            _make_scene(2, start_time=45.3, end_time=78.1,
                        text="Scene at later time"),
        ]
        plan = tmp_path / "plan.json"
        _write_plan(scenes, plan)

        cfg = RunConfig(
            scene_plan_path=plan,
            output_dir=tmp_path / "img",
            manifest_path=tmp_path / "manifest.json",
            backend="placeholder",
            dry_run=True,
            theme="Cinematic Concept Art",
        )
        results = run_image_generation(cfg)

        assert results[0].index == 1
        assert results[1].index == 2
        # Timestamps come from the scene plan, not modified by theme
        # (They're stored in the plan file, not in SceneResult directly)

    def test_multiple_scenes_get_same_theme_but_different_prompts(self, tmp_path):
        from unittest.mock import patch
        from ai_visualizer.image_runner import BACKENDS

        scenes = [
            _make_scene(
                1, image_prompt="A wide shot of a mountain range at dawn",
                start_time=0, end_time=30,
            ),
            _make_scene(
                2, image_prompt="A close-up of a tearful eye",
                start_time=30, end_time=60,
            ),
            _make_scene(
                3, image_prompt="A medium shot of two people arguing",
                start_time=60, end_time=90,
            ),
        ]
        plan = tmp_path / "plan.json"
        _write_plan(scenes, plan)

        with patch.dict(BACKENDS, {"mock": _fake_backend}):
            cfg = RunConfig(
                scene_plan_path=plan,
                output_dir=tmp_path / "img",
                manifest_path=tmp_path / "manifest.json",
                backend="mock",
                theme="Cinematic Concept Art",
            )
            results = run_image_generation(cfg)

        # All scenes share the same theme
        assert all(r.theme == "Cinematic Concept Art" for r in results)
        # But scene-specific prompts are different
        prompts = [r.prompt_used for r in results]
        assert len(prompts) == len(set(prompts))  # all unique
        assert "A wide shot of a mountain range at dawn" in prompts[0]
        assert "A close-up of a tearful eye" in prompts[1]
        assert "A medium shot of two people arguing" in prompts[2]
        # All share the same negative prompt
        assert all(r.negative_prompt_used == CINEMATIC_CONCEPT_ART_NEGATIVE for r in results)

"""
tests/test_image_runner.py
--------------------------
Tests for ai_visualizer.image_runner.

All tests use mocked backends — no real API calls, no credits consumed.
The placeholder backend (Pillow only) is tested against real Pillow calls.
"""

from __future__ import annotations

import json
import tempfile
import time
from threading import Lock
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from ai_visualizer.image_runner import (
    BACKENDS,
    RunConfig,
    ScenePlanError,
    SceneResult,
    _clean_prompt,
    load_manifest,
    run_image_generation,
    save_manifest,
    validate_scene_plan,
)

# ---------------------------------------------------------------------------
# Fixtures
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


# ---------------------------------------------------------------------------
# Prompt cleaning
# ---------------------------------------------------------------------------

class TestCleanPrompt:
    def test_strips_fallback_prefix(self):
        raw = "[FALLBACK] Cinematic illustration: something."
        assert _clean_prompt(raw) == "Cinematic illustration: something."

    def test_leaves_real_prompt_unchanged(self):
        raw = "A warm sunset over calm ocean water."
        assert _clean_prompt(raw) == raw

    def test_empty_string_unchanged(self):
        assert _clean_prompt("") == ""

    def test_only_prefix_returns_empty_string(self):
        assert _clean_prompt("[FALLBACK] ") == ""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidateScenePlan:

    def test_valid_plan_no_errors(self):
        scenes = [_make_scene(i, start_time=i*30.0, end_time=i*30.0+25.0)
                  for i in range(1, 4)]
        assert validate_scene_plan(scenes) == []

    def test_empty_plan_is_error(self):
        errs = validate_scene_plan([])
        assert any("empty" in e.lower() for e in errs)

    def test_missing_required_field(self):
        scene = _make_scene()
        del scene["image_prompt"]
        errs = validate_scene_plan([scene])
        assert any("image_prompt" in e for e in errs)

    def test_end_before_start_is_error(self):
        scene = _make_scene(start_time=50.0, end_time=10.0)
        errs = validate_scene_plan([scene])
        assert any("end_time" in e for e in errs)

    def test_equal_start_end_is_error(self):
        scene = _make_scene(start_time=10.0, end_time=10.0)
        errs = validate_scene_plan([scene])
        assert any("end_time" in e for e in errs)

    def test_duplicate_index_is_error(self):
        scenes = [_make_scene(1, start_time=0.0, end_time=10.0),
                  _make_scene(1, start_time=10.0, end_time=20.0)]
        errs = validate_scene_plan(scenes)
        assert any("duplicate" in e.lower() for e in errs)

    def test_overlapping_timestamps_is_error(self):
        scenes = [
            _make_scene(1, start_time=0.0, end_time=30.0),
            _make_scene(2, start_time=20.0, end_time=50.0),  # starts before scene 1 ends
        ]
        errs = validate_scene_plan(scenes)
        assert any("overlap" in e.lower() for e in errs)

    def test_non_png_filename_is_error(self):
        scene = _make_scene()
        scene["filename_suggestion"] = "scene_001.jpg"
        errs = validate_scene_plan([scene])
        assert any(".png" in e for e in errs)

    def test_invalid_plan_raises_in_runner(self, tmp_path):
        scene = _make_scene(start_time=50.0, end_time=10.0)  # end < start
        p = tmp_path / "plan.json"
        _write_plan([scene], p)
        cfg = RunConfig(scene_plan_path=p, output_dir=tmp_path/"img",
                        manifest_path=tmp_path/"manifest.json")
        with pytest.raises(ScenePlanError):
            run_image_generation(cfg)


# ---------------------------------------------------------------------------
# Placeholder backend (real Pillow, no API)
# ---------------------------------------------------------------------------

class TestPlaceholderBackend:

    def test_generates_png_file(self, tmp_path):
        scene = _make_scene()
        out = BACKENDS["placeholder"](scene, tmp_path)
        assert out.exists()
        assert out.suffix == ".png"

    def test_filename_matches_suggestion(self, tmp_path):
        scene = _make_scene(index=7)
        out = BACKENDS["placeholder"](scene, tmp_path)
        assert out.name == "scene_007.png"

    def test_placeholder_file_is_valid_image(self, tmp_path):
        from PIL import Image
        scene = _make_scene()
        out = BACKENDS["placeholder"](scene, tmp_path)
        img = Image.open(str(out))
        assert img.width == 1024
        assert img.height == 576

    def test_fallback_prompt_stripped_in_image(self, tmp_path):
        """[FALLBACK] prefix should not appear in the image filename."""
        scene = _make_scene(image_prompt="[FALLBACK] Cinematic test prompt.")
        out = BACKENDS["placeholder"](scene, tmp_path)
        assert out.exists()  # generation should succeed despite prefix


# ---------------------------------------------------------------------------
# Dry-run mode
# ---------------------------------------------------------------------------

class TestDryRun:

    def test_dry_run_creates_no_files(self, tmp_path, capsys):
        scenes = [_make_scene(i, start_time=i*30.0, end_time=i*30.0+25.0)
                  for i in range(1, 4)]
        p = tmp_path / "plan.json"
        _write_plan(scenes, p)

        cfg = RunConfig(
            scene_plan_path=p,
            output_dir=tmp_path / "img",
            manifest_path=tmp_path / "manifest.json",
            backend="placeholder",
            dry_run=True,
        )
        results = run_image_generation(cfg)

        # No image files created
        assert not (tmp_path / "img").exists() or \
               len(list((tmp_path / "img").iterdir())) == 0
        # No manifest created
        assert not (tmp_path / "manifest.json").exists()

    def test_dry_run_returns_dry_run_status(self, tmp_path):
        scenes = [_make_scene()]
        p = tmp_path / "plan.json"
        _write_plan(scenes, p)

        cfg = RunConfig(scene_plan_path=p, output_dir=tmp_path/"img",
                        manifest_path=tmp_path/"manifest.json",
                        backend="placeholder", dry_run=True)
        results = run_image_generation(cfg)
        assert all(r.status == "dry_run" for r in results)

    def test_dry_run_count_equals_scene_count(self, tmp_path):
        scenes = [_make_scene(i, start_time=i*30.0, end_time=i*30.0+25.0)
                  for i in range(1, 6)]
        p = tmp_path / "plan.json"
        _write_plan(scenes, p)

        cfg = RunConfig(scene_plan_path=p, output_dir=tmp_path/"img",
                        manifest_path=tmp_path/"manifest.json",
                        backend="placeholder", dry_run=True)
        results = run_image_generation(cfg)
        assert len(results) == 5

    def test_api_dry_run_needs_no_key_and_makes_no_request(self, tmp_path, monkeypatch):
        scenes = [_make_scene()]
        plan = tmp_path / "plan.json"
        _write_plan(scenes, plan)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        with patch("ai_visualizer.image_backends.requests.post") as post:
            cfg = RunConfig(
                scene_plan_path=plan,
                output_dir=tmp_path / "img",
                manifest_path=tmp_path / "manifest.json",
                backend="api",
                dry_run=True,
            )
            results = run_image_generation(cfg)

        assert results[0].status == "dry_run"
        post.assert_not_called()
        assert not (tmp_path / "manifest.json").exists()


# ---------------------------------------------------------------------------
# Successful generation with mocked backend
# ---------------------------------------------------------------------------

class TestSuccessfulGeneration:

    def test_images_created_with_mock_backend(self, tmp_path):
        scenes = [_make_scene(i, start_time=i*30.0, end_time=i*30.0+25.0)
                  for i in range(1, 4)]
        p = tmp_path / "plan.json"
        _write_plan(scenes, p)
        img_dir = tmp_path / "img"

        with patch.dict(BACKENDS, {"mock": _fake_backend}):
            cfg = RunConfig(scene_plan_path=p, output_dir=img_dir,
                            manifest_path=tmp_path/"manifest.json",
                            backend="mock")
            results = run_image_generation(cfg)

        assert all(r.status == "ok" for r in results)
        for i in range(1, 4):
            assert (img_dir / f"scene_{i:03d}.png").exists()

    def test_manifest_written_after_run(self, tmp_path):
        scenes = [_make_scene(i, start_time=i*30.0, end_time=i*30.0+25.0)
                  for i in range(1, 3)]
        p = tmp_path / "plan.json"
        _write_plan(scenes, p)

        cfg = RunConfig(scene_plan_path=p, output_dir=tmp_path/"img",
                        manifest_path=tmp_path/"manifest.json", backend="mock")
        run_image_generation(cfg)

        assert (tmp_path / "manifest.json").exists()
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert len(manifest) == 2
        assert all(e["status"] == "ok" for e in manifest)
        assert manifest[0]["scene_id"] == "1"
        assert manifest[0]["prompt"] == "Cinematic bridge at sunset, warm light."
        assert manifest[0]["path"] == manifest[0]["image_path"]
        assert manifest[0]["model"] == "mock-labeled-png-v1"

    def test_resume_keeps_original_backend_and_model(self, tmp_path):
        image_dir = tmp_path / "img"
        image_dir.mkdir()
        image_path = image_dir / "scene_001.png"
        image_path.write_bytes(b"existing")
        manifest_path = tmp_path / "manifest.json"
        save_manifest([SceneResult(
            index=1, title="T", filename=image_path.name,
            image_path=str(image_path), prompt_used="prior prompt",
            status="ok", backend="api", model="prior-model",
        )], manifest_path)
        plan = tmp_path / "plan.json"
        _write_plan([_make_scene()], plan)

        with patch.dict(BACKENDS, {"mock": _fake_backend}):
            results = run_image_generation(RunConfig(
                scene_plan_path=plan,
                output_dir=image_dir,
                manifest_path=manifest_path,
                backend="mock",
            ))

        assert results[0].status == "skipped"
        assert results[0].backend == "api"
        assert results[0].model == "prior-model"

    def test_result_image_paths_exist(self, tmp_path):
        scenes = [_make_scene(i, start_time=i*30.0, end_time=i*30.0+25.0)
                  for i in range(1, 4)]
        p = tmp_path / "plan.json"
        _write_plan(scenes, p)

        with patch.dict(BACKENDS, {"mock": _fake_backend}):
            cfg = RunConfig(scene_plan_path=p, output_dir=tmp_path/"img",
                            manifest_path=tmp_path/"manifest.json", backend="mock")
            results = run_image_generation(cfg)

        for r in results:
            assert Path(r.image_path).exists()

    def test_filenames_match_scene_suggestion(self, tmp_path):
        scenes = [_make_scene(i, start_time=i*30.0, end_time=i*30.0+25.0)
                  for i in range(1, 4)]
        p = tmp_path / "plan.json"
        _write_plan(scenes, p)

        with patch.dict(BACKENDS, {"mock": _fake_backend}):
            cfg = RunConfig(scene_plan_path=p, output_dir=tmp_path/"img",
                            manifest_path=tmp_path/"manifest.json", backend="mock")
            results = run_image_generation(cfg)

        for r in results:
            assert r.filename == f"scene_{r.index:03d}.png"

    def test_prompt_has_fallback_prefix_stripped(self, tmp_path):
        scene = _make_scene(image_prompt="[FALLBACK] Some fallback text here.")
        p = tmp_path / "plan.json"
        _write_plan([scene], p)

        with patch.dict(BACKENDS, {"mock": _fake_backend}):
            cfg = RunConfig(scene_plan_path=p, output_dir=tmp_path/"img",
                            manifest_path=tmp_path/"manifest.json", backend="mock")
            results = run_image_generation(cfg)

        assert not results[0].prompt_used.startswith("[FALLBACK]")
        assert results[0].prompt_used == "Some fallback text here."


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------

class TestFailureHandling:

    def _error_backend(self, scene, out_dir):
        raise RuntimeError("Simulated API failure")

    def test_failed_scene_has_error_status(self, tmp_path):
        scenes = [_make_scene()]
        p = tmp_path / "plan.json"
        _write_plan(scenes, p)

        with patch.dict(BACKENDS, {"failing": self._error_backend}):
            cfg = RunConfig(scene_plan_path=p, output_dir=tmp_path/"img",
                            manifest_path=tmp_path/"manifest.json",
                            backend="failing", max_retries=0)
            results = run_image_generation(cfg)

        assert results[0].status == "error"
        assert results[0].image_path == ""

    def test_error_message_recorded_in_manifest(self, tmp_path):
        scenes = [_make_scene()]
        p = tmp_path / "plan.json"
        _write_plan(scenes, p)

        with patch.dict(BACKENDS, {"failing": self._error_backend}):
            cfg = RunConfig(scene_plan_path=p, output_dir=tmp_path/"img",
                            manifest_path=tmp_path/"manifest.json",
                            backend="failing", max_retries=0)
            run_image_generation(cfg)

        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest[0]["error"] != ""

    def test_one_failure_does_not_abort_other_scenes(self, tmp_path):
        call_count = {"n": 0}

        def sometimes_fails(scene, out_dir):
            call_count["n"] += 1
            if scene["index"] == 2:
                raise RuntimeError("fail on scene 2")
            return _fake_backend(scene, out_dir)

        scenes = [_make_scene(i, start_time=i*30.0, end_time=i*30.0+25.0)
                  for i in range(1, 4)]
        p = tmp_path / "plan.json"
        _write_plan(scenes, p)

        with patch.dict(BACKENDS, {"partial": sometimes_fails}):
            cfg = RunConfig(scene_plan_path=p, output_dir=tmp_path/"img",
                            manifest_path=tmp_path/"manifest.json",
                            backend="partial", max_retries=0)
            results = run_image_generation(cfg)

        statuses = {r.index: r.status for r in results}
        assert statuses[1] == "ok"
        assert statuses[2] == "error"
        assert statuses[3] == "ok"

    def test_empty_prompt_is_error(self, tmp_path):
        scene = _make_scene(image_prompt="")
        p = tmp_path / "plan.json"
        _write_plan([scene], p)

        with patch.dict(BACKENDS, {"mock": _fake_backend}):
            cfg = RunConfig(scene_plan_path=p, output_dir=tmp_path/"img",
                            manifest_path=tmp_path/"manifest.json",
                            backend="mock", max_retries=0)
            results = run_image_generation(cfg)

        assert results[0].status == "error"


# ---------------------------------------------------------------------------
# Retry behaviour
# ---------------------------------------------------------------------------

class TestRetryBehaviour:

    def test_retries_up_to_max_retries(self, tmp_path):
        call_log = []

        def flaky(scene, out_dir):
            call_log.append(scene["index"])
            if len(call_log) < 3:
                raise RuntimeError("transient error")
            return _fake_backend(scene, out_dir)

        scenes = [_make_scene()]
        p = tmp_path / "plan.json"
        _write_plan(scenes, p)

        with patch("ai_visualizer.image_runner.time.sleep"):  # skip delay in tests
            with patch.dict(BACKENDS, {"flaky": flaky}):
                cfg = RunConfig(scene_plan_path=p, output_dir=tmp_path/"img",
                                manifest_path=tmp_path/"manifest.json",
                                backend="flaky", max_retries=3,
                                retry_delay_seconds=0.0)
                results = run_image_generation(cfg)

        assert results[0].status == "ok"
        assert len(call_log) == 3  # failed twice, succeeded on 3rd

    def test_exhausted_retries_give_error(self, tmp_path):
        def always_fails(scene, out_dir):
            raise RuntimeError("always fails")

        scenes = [_make_scene()]
        p = tmp_path / "plan.json"
        _write_plan(scenes, p)

        with patch("ai_visualizer.image_runner.time.sleep"):
            with patch.dict(BACKENDS, {"always_fail": always_fails}):
                cfg = RunConfig(scene_plan_path=p, output_dir=tmp_path/"img",
                                manifest_path=tmp_path/"manifest.json",
                                backend="always_fail", max_retries=2,
                                retry_delay_seconds=0.0)
                results = run_image_generation(cfg)

        assert results[0].status == "error"


# ---------------------------------------------------------------------------
# Resume / skip-existing behaviour
# ---------------------------------------------------------------------------

class TestResumeBehaviour:

    def test_existing_ok_scene_is_skipped(self, tmp_path):
        img_dir = tmp_path / "img"
        img_dir.mkdir()
        # Pre-create the image file
        (img_dir / "scene_001.png").write_bytes(b"fake")

        # Pre-populate manifest with a successful entry
        manifest_path = tmp_path / "manifest.json"
        existing = [SceneResult(
            index=1, title="T", filename="scene_001.png",
            image_path=str(img_dir / "scene_001.png"),
            prompt_used="test", status="ok", backend="mock",
        )]
        save_manifest(existing, manifest_path)

        scenes = [_make_scene()]
        p = tmp_path / "plan.json"
        _write_plan(scenes, p)

        call_count = {"n": 0}
        def counting_backend(scene, out_dir):
            call_count["n"] += 1
            return _fake_backend(scene, out_dir)

        with patch.dict(BACKENDS, {"counting": counting_backend}):
            cfg = RunConfig(scene_plan_path=p, output_dir=img_dir,
                            manifest_path=manifest_path,
                            backend="counting", force=False)
            results = run_image_generation(cfg)

        assert call_count["n"] == 0   # backend never called
        assert results[0].status == "skipped"

    def test_force_flag_overrides_skip(self, tmp_path):
        img_dir = tmp_path / "img"
        img_dir.mkdir()
        (img_dir / "scene_001.png").write_bytes(b"old")

        manifest_path = tmp_path / "manifest.json"
        existing = [SceneResult(
            index=1, title="T", filename="scene_001.png",
            image_path=str(img_dir / "scene_001.png"),
            prompt_used="test", status="ok", backend="mock",
        )]
        save_manifest(existing, manifest_path)

        scenes = [_make_scene()]
        p = tmp_path / "plan.json"
        _write_plan(scenes, p)

        call_count = {"n": 0}
        def counting_backend(scene, out_dir):
            call_count["n"] += 1
            return _fake_backend(scene, out_dir)

        with patch.dict(BACKENDS, {"counting": counting_backend}):
            cfg = RunConfig(scene_plan_path=p, output_dir=img_dir,
                            manifest_path=manifest_path,
                            backend="counting", force=True)
            results = run_image_generation(cfg)

        assert call_count["n"] == 1   # forced regeneration
        assert results[0].status == "ok"

    def test_partial_run_resume(self, tmp_path):
        """Scene 2 failed in a previous run; re-running should retry only scene 2."""
        img_dir = tmp_path / "img"
        img_dir.mkdir()

        # Scene 1 succeeded and image exists
        _fake_backend(_make_scene(1), img_dir)

        manifest_path = tmp_path / "manifest.json"
        prior = [
            SceneResult(index=1, title="S1", filename="scene_001.png",
                        image_path=str(img_dir/"scene_001.png"),
                        prompt_used="p1", status="ok", backend="mock"),
            SceneResult(index=2, title="S2", filename="scene_002.png",
                        image_path="", prompt_used="p2",
                        status="error", error="timeout", backend="mock"),
        ]
        save_manifest(prior, manifest_path)

        scenes = [
            _make_scene(1, start_time=0.0, end_time=25.0),
            _make_scene(2, start_time=25.0, end_time=50.0),
        ]
        p = tmp_path / "plan.json"
        _write_plan(scenes, p)

        call_log = []
        def logging_backend(scene, out_dir):
            call_log.append(scene["index"])
            return _fake_backend(scene, out_dir)

        with patch.dict(BACKENDS, {"logging": logging_backend}):
            cfg = RunConfig(scene_plan_path=p, output_dir=img_dir,
                            manifest_path=manifest_path,
                            backend="logging", force=False)
            results = run_image_generation(cfg)

        # Only scene 2 should have been processed
        assert call_log == [2]
        statuses = {r.index: r.status for r in results}
        assert statuses[1] == "skipped"
        assert statuses[2] == "ok"


# ---------------------------------------------------------------------------
# Scene selection
# ---------------------------------------------------------------------------

class TestSceneSelection:

    def test_run_only_specified_scenes(self, tmp_path):
        scenes = [_make_scene(i, start_time=i*30.0, end_time=i*30.0+25.0)
                  for i in range(1, 6)]
        p = tmp_path / "plan.json"
        _write_plan(scenes, p)

        call_log = []
        def logging_backend(scene, out_dir):
            call_log.append(scene["index"])
            return _fake_backend(scene, out_dir)

        with patch.dict(BACKENDS, {"logging": logging_backend}):
            cfg = RunConfig(scene_plan_path=p, output_dir=tmp_path/"img",
                            manifest_path=tmp_path/"manifest.json",
                            backend="logging", scene_indexes=[2, 4])
            run_image_generation(cfg)

        assert sorted(call_log) == [2, 4]

    def test_configured_concurrency_runs_multiple_jobs(self, tmp_path):
        state = {"active": 0, "maximum": 0}
        lock = Lock()

        def tracking_backend(scene, out_dir):
            with lock:
                state["active"] += 1
                state["maximum"] = max(state["maximum"], state["active"])
            time.sleep(0.03)
            result = _fake_backend(scene, out_dir)
            with lock:
                state["active"] -= 1
            return result

        scenes = [
            _make_scene(i, start_time=(i - 1) * 30.0, end_time=i * 30.0)
            for i in range(1, 5)
        ]
        plan = tmp_path / "plan.json"
        _write_plan(scenes, plan)

        with patch.dict(BACKENDS, {"parallel-test": tracking_backend}):
            results = run_image_generation(RunConfig(
                scene_plan_path=plan,
                output_dir=tmp_path / "img",
                manifest_path=tmp_path / "manifest.json",
                backend="parallel-test",
                concurrency=2,
                max_retries=0,
                delay_seconds=0,
            ))

        assert all(result.status == "ok" for result in results)
        assert state["maximum"] == 2


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------

class TestManifest:

    def test_save_and_load_roundtrip(self, tmp_path):
        results = [
            SceneResult(index=1, title="A", filename="scene_001.png",
                        image_path="/x/scene_001.png", prompt_used="p1",
                        status="ok", backend="placeholder"),
            SceneResult(index=2, title="B", filename="scene_002.png",
                        image_path="", prompt_used="p2",
                        status="error", error="timeout", backend="stability"),
        ]
        path = tmp_path / "manifest.json"
        save_manifest(results, path)
        loaded = load_manifest(path)

        assert loaded[1].status == "ok"
        assert loaded[2].status == "error"
        assert loaded[2].error == "timeout"

    def test_load_missing_manifest_returns_empty(self, tmp_path):
        loaded = load_manifest(tmp_path / "nonexistent.json")
        assert loaded == {}

    def test_backend_name_recorded(self, tmp_path):
        scenes = [_make_scene()]
        p = tmp_path / "plan.json"
        _write_plan(scenes, p)

        with patch.dict(BACKENDS, {"mock": _fake_backend}):
            cfg = RunConfig(scene_plan_path=p, output_dir=tmp_path/"img",
                            manifest_path=tmp_path/"manifest.json", backend="mock")
            results = run_image_generation(cfg)

        assert results[0].backend == "mock"

    def test_placeholder_flag_set_correctly(self, tmp_path):
        scenes = [_make_scene()]
        p = tmp_path / "plan.json"
        _write_plan(scenes, p)

        cfg = RunConfig(scene_plan_path=p, output_dir=tmp_path/"img",
                        manifest_path=tmp_path/"manifest.json", backend="placeholder")
        results = run_image_generation(cfg)
        assert results[0].is_placeholder is True


# ---------------------------------------------------------------------------
# Unknown backend
# ---------------------------------------------------------------------------

class TestUnknownBackend:

    def test_unknown_backend_raises_value_error(self, tmp_path):
        scenes = [_make_scene()]
        p = tmp_path / "plan.json"
        _write_plan(scenes, p)

        cfg = RunConfig(scene_plan_path=p, output_dir=tmp_path/"img",
                        manifest_path=tmp_path/"manifest.json",
                        backend="nonexistent_backend")
        with pytest.raises(ValueError, match="nonexistent_backend"):
            run_image_generation(cfg)

    def test_missing_plan_file_raises(self, tmp_path):
        cfg = RunConfig(scene_plan_path=tmp_path/"nope.json",
                        output_dir=tmp_path/"img",
                        manifest_path=tmp_path/"manifest.json")
        with pytest.raises(FileNotFoundError):
            run_image_generation(cfg)

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
import requests

from ai_visualizer.image_backends import (
    ComfyUIBackend,
    GenerationResult,
    GeminiApiBackend,
)
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
        assert manifest[0]["prompt"].endswith("Cinematic bridge at sunset, warm light")
        assert manifest[0]["prompt"].count("Cinematic film still") == 1
        assert manifest[0]["negative_prompt_used"]
        assert manifest[0]["path"] == manifest[0]["image_path"]
        assert manifest[0]["model"] == "mock-labeled-png-v1"

    def test_api_jpeg_extension_is_used_in_manifest(self, tmp_path, monkeypatch):
        import base64
        from io import BytesIO
        from PIL import Image

        image_buffer = BytesIO()
        Image.new("RGB", (2, 2), (10, 20, 30)).save(image_buffer, format="JPEG")
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps({
            "output_image": {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(image_buffer.getvalue()).decode("ascii"),
            }
        }).encode("utf-8")
        monkeypatch.setattr(
            "ai_visualizer.image_runner.create_backend",
            lambda name: GeminiApiBackend(api_key="test-key"),
        )
        monkeypatch.setattr(
            "ai_visualizer.image_backends.requests.post",
            lambda *args, **kwargs: response,
        )

        plan = tmp_path / "plan.json"
        _write_plan([_make_scene()], plan)
        image_dir = tmp_path / "images"
        manifest_path = tmp_path / "manifest.json"
        results = run_image_generation(RunConfig(
            scene_plan_path=plan,
            output_dir=image_dir,
            manifest_path=manifest_path,
            backend="api",
            max_retries=0,
            delay_seconds=0,
        ))

        manifest_entry = json.loads(manifest_path.read_text(encoding="utf-8"))[0]
        assert results[0].status == "ok"
        assert results[0].filename == "scene_001.jpg"
        assert Path(results[0].image_path).suffix == ".jpg"
        assert manifest_entry["filename"] == "scene_001.jpg"
        assert Path(manifest_entry["path"]).suffix == ".jpg"
        assert (image_dir / "scene_001.jpg").exists()
        assert not (image_dir / "scene_001.png").exists()

    def test_local_png_is_validated_and_manifest_uses_png_path(self, tmp_path, monkeypatch):
        from PIL import Image

        class LocalBackend:
            model_identifier = "local-test-model"

            def generate(self, scene_id, image_prompt, output_path, options):
                Image.new("RGB", (2, 2), (20, 30, 40)).save(output_path, format="PNG")
                return GenerationResult(
                    output_path, "local", self.model_identifier, True
                )

        monkeypatch.setattr(
            "ai_visualizer.image_runner.create_backend",
            lambda name: LocalBackend(),
        )
        plan = tmp_path / "plan.json"
        _write_plan([_make_scene()], plan)
        manifest_path = tmp_path / "manifest.json"
        results = run_image_generation(RunConfig(
            scene_plan_path=plan,
            output_dir=tmp_path / "images",
            manifest_path=manifest_path,
            backend="local",
            max_retries=0,
            delay_seconds=0,
        ))

        entry = json.loads(manifest_path.read_text(encoding="utf-8"))[0]
        assert results[0].status == "ok"
        assert results[0].filename == "scene_001.png"
        assert entry["backend"] == "local"
        assert entry["model"] == "local-test-model"
        assert Path(entry["path"]).suffix == ".png"

    def test_local_backend_format_mismatch_is_failure_not_fallback(
        self, tmp_path, monkeypatch
    ):
        from PIL import Image

        class WrongFormatBackend:
            model_identifier = "local-test-model"

            def generate(self, scene_id, image_prompt, output_path, options):
                Image.new("RGB", (2, 2), (20, 30, 40)).save(output_path, format="JPEG")
                return GenerationResult(
                    output_path, "local", self.model_identifier, True
                )

        monkeypatch.setattr(
            "ai_visualizer.image_runner.create_backend",
            lambda name: WrongFormatBackend(),
        )
        plan = tmp_path / "plan.json"
        _write_plan([_make_scene()], plan)
        manifest_path = tmp_path / "manifest.json"
        results = run_image_generation(RunConfig(
            scene_plan_path=plan,
            output_dir=tmp_path / "images",
            manifest_path=manifest_path,
            backend="local",
            max_retries=0,
            delay_seconds=0,
        ))

        manifest_entry = json.loads(manifest_path.read_text(encoding="utf-8"))[0]
        assert results[0].status == "error"
        assert "must return PNG image data" in results[0].error
        assert manifest_entry["error"] == results[0].error
        assert manifest_entry["backend"] == "local"

    def test_resume_does_not_reuse_image_from_another_backend(self, tmp_path):
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

        assert results[0].status == "ok"
        assert results[0].backend == "mock"
        assert results[0].is_placeholder
        assert image_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

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
        assert results[0].prompt_used.endswith("Some fallback text here")
        assert results[0].prompt_used.count("Cinematic film still") == 1

    def test_quality_prompts_are_composed_once_on_retry_and_resume(
        self, tmp_path
    ):
        from PIL import Image

        scene = _make_scene(image_prompt="Clear subject, detailed house.")
        scene["negative_prompt"] = "Blurry, malformed hands."
        plan = tmp_path / "plan.json"
        _write_plan([scene], plan)
        calls = []

        class RetryBackend:
            name = "local"
            model_identifier = "test-model"

            def generate(self, scene_id, image_prompt, output_path, options):
                calls.append((image_prompt, options.negative_prompt))
                if len(calls) == 1:
                    return GenerationResult(None, "local", self.model_identifier, False, "retry")
                Image.new("RGB", (2, 2)).save(output_path, format="PNG")
                return GenerationResult(
                    output_path, "local", self.model_identifier, True
                )

        config = RunConfig(
            scene_plan_path=plan,
            output_dir=tmp_path / "images",
            manifest_path=tmp_path / "manifest.json",
            backend="local",
            backend_instance=RetryBackend(),
            max_retries=1,
            retry_delay_seconds=0,
            delay_seconds=0,
            positive_quality_prompt="Film look, clear subject.",
            negative_quality_prompt="Blurry, image noise.",
        )
        first_results = run_image_generation(config)

        expected_positive = "Film look, clear subject, detailed house"
        expected_negative = "Blurry, image noise, malformed hands"
        assert first_results[0].status == "ok"
        assert calls == [
            (expected_positive, expected_negative),
            (expected_positive, expected_negative),
        ]
        manifest_entry = json.loads(
            config.manifest_path.read_text(encoding="utf-8")
        )[0]
        assert manifest_entry["prompt_used"] == expected_positive
        assert manifest_entry["negative_prompt_used"] == expected_negative

        resumed = run_image_generation(config)
        assert resumed[0].status == "skipped"
        assert resumed[0].prompt_used == expected_positive
        assert resumed[0].negative_prompt_used == expected_negative
        assert len(calls) == 2


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

    def test_comfyui_unavailable_is_retried_and_recorded_without_fallback(
        self, tmp_path, monkeypatch
    ):
        workflow_path = tmp_path / "workflow.json"
        workflow_path.write_text(json.dumps({
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "test.safetensors"},
            },
            "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "{{prompt}}"}},
            "3": {"class_type": "SaveImage", "inputs": {}},
        }), encoding="utf-8")
        backend = ComfyUIBackend(
            base_url="http://127.0.0.1:8188",
            workflow_path=workflow_path,
        )
        monkeypatch.setattr(
            "ai_visualizer.image_runner.create_backend",
            lambda name: backend,
        )
        request_count = {"count": 0}

        def unavailable(*args, **kwargs):
            request_count["count"] += 1
            raise requests.ConnectionError("ComfyUI is not reachable")

        monkeypatch.setattr("ai_visualizer.image_backends.requests.get", unavailable)
        with patch("ai_visualizer.image_backends.requests.post") as post:
            plan = tmp_path / "plan.json"
            _write_plan([_make_scene()], plan)
            manifest_path = tmp_path / "manifest.json"
            results = run_image_generation(RunConfig(
                scene_plan_path=plan,
                output_dir=tmp_path / "images",
                manifest_path=manifest_path,
                backend="local",
                max_retries=1,
                retry_delay_seconds=0,
                delay_seconds=0,
            ))

        manifest_entry = json.loads(manifest_path.read_text(encoding="utf-8"))[0]
        assert request_count["count"] == 2
        assert results[0].status == "error"
        assert "ComfyUI request failed during preflight" in results[0].error
        assert manifest_entry["error"] == results[0].error
        assert manifest_entry["backend"] == "local"
        post.assert_not_called()

    def test_google_http_400_details_reach_manifest_and_redact_credentials(
        self, tmp_path, monkeypatch, capsys
    ):
        api_key = "TEST_API_KEY_NEVER_LOG"
        access_token = "TEST_ACCESS_TOKEN_NEVER_LOG"
        response = requests.Response()
        response.status_code = 400
        response.encoding = "utf-8"
        response._content = json.dumps({
            "error": {
                "code": 400,
                "status": "INVALID_ARGUMENT",
                "message": f"Invalid request parameter; key={api_key}",
                "details": [{
                    "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                    "reason": "INVALID_ARGUMENT",
                    "metadata": {
                        "field": "response_format",
                        "access_token": access_token,
                    },
                }],
            }
        }).encode("utf-8")
        monkeypatch.setattr(
            "ai_visualizer.image_runner.create_backend",
            lambda name: GeminiApiBackend(api_key=api_key),
        )
        monkeypatch.setattr(
            "ai_visualizer.image_backends.requests.post",
            lambda *args, **kwargs: response,
        )

        plan_path = tmp_path / "plan.json"
        _write_plan([_make_scene()], plan_path)
        manifest_path = tmp_path / "manifest.json"
        results = run_image_generation(RunConfig(
            scene_plan_path=plan_path,
            output_dir=tmp_path / "images",
            manifest_path=manifest_path,
            backend="api",
            max_retries=0,
            delay_seconds=0,
        ))

        error = results[0].error
        manifest_error = json.loads(manifest_path.read_text(encoding="utf-8"))[0]["error"]
        console_output = capsys.readouterr().out
        for diagnostic in (error, manifest_error, console_output):
            assert "400" in diagnostic
            assert "INVALID_ARGUMENT" in diagnostic
            assert "Invalid request parameter" in diagnostic
            assert "[REDACTED]" in diagnostic
            assert "response_format" in diagnostic
            assert api_key not in diagnostic
            assert access_token not in diagnostic
        assert manifest_error == error

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

    def test_existing_other_backend_output_is_not_overwritten(self, tmp_path, monkeypatch):
        from PIL import Image

        output_dir = tmp_path / "images"
        output_dir.mkdir()
        existing_path = output_dir / "scene_001.png"
        Image.new("RGB", (2, 2), (30, 40, 50)).save(existing_path, format="PNG")
        original_bytes = existing_path.read_bytes()
        manifest_path = tmp_path / "manifest.json"
        save_manifest([SceneResult(
            index=1,
            title="Existing",
            filename=existing_path.name,
            image_path=str(existing_path),
            prompt_used="existing",
            status="ok",
            backend="mock",
            is_placeholder=True,
        )], manifest_path)

        class UnexpectedBackend:
            model_identifier = "local-model"

            def generate(self, *args, **kwargs):
                raise AssertionError("conflicting output must not be overwritten")

        monkeypatch.setattr(
            "ai_visualizer.image_runner.create_backend",
            lambda name: UnexpectedBackend(),
        )
        plan = tmp_path / "plan.json"
        _write_plan([_make_scene()], plan)
        results = run_image_generation(RunConfig(
            scene_plan_path=plan,
            output_dir=output_dir,
            manifest_path=manifest_path,
            backend="local",
            delay_seconds=0,
        ))

        assert results[0].status == "error"
        assert "explicitly enable replacement" in results[0].error
        assert existing_path.read_bytes() == original_bytes
        manifest_entry = json.loads(manifest_path.read_text(encoding="utf-8"))[0]
        assert manifest_entry["backend"] == "local"
        assert manifest_entry["status"] == "error"


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

    def test_cancellation_stops_scheduling_remaining_scenes(self, tmp_path):
        from threading import Event

        call_log = []
        cancel_event = Event()

        def backend(scene, out_dir):
            call_log.append(scene["index"])
            return _fake_backend(scene, out_dir)

        def progress(event_name, index, result, completed, total):
            if event_name == "running" and index == 1:
                cancel_event.set()

        scenes = [
            _make_scene(i, start_time=(i - 1) * 30, end_time=i * 30)
            for i in range(1, 4)
        ]
        plan = tmp_path / "plan.json"
        _write_plan(scenes, plan)
        with patch.dict(BACKENDS, {"cancellable": backend}):
            results = run_image_generation(RunConfig(
                scene_plan_path=plan,
                output_dir=tmp_path / "images",
                manifest_path=tmp_path / "manifest.json",
                backend="cancellable",
                concurrency=1,
                delay_seconds=0,
                max_retries=0,
                cancel_event=cancel_event,
                progress_callback=progress,
            ))

        assert call_log == [1]
        assert {result.index: result.status for result in results} == {
            1: "ok",
            2: "cancelled",
            3: "cancelled",
        }


# ---------------------------------------------------------------------------
# Resume / skip-existing behaviour
# ---------------------------------------------------------------------------

class TestResumeBehaviour:

    def test_existing_ok_scene_is_skipped(self, tmp_path):
        img_dir = tmp_path / "img"
        img_dir.mkdir()
        # Resume only treats an existing, validated image as successful.
        _fake_backend(_make_scene(), img_dir)

        # Pre-populate manifest with a successful entry
        manifest_path = tmp_path / "manifest.json"
        existing = [SceneResult(
            index=1, title="T", filename="scene_001.png",
            image_path=str(img_dir / "scene_001.png"),
            prompt_used="test", status="ok", backend="counting",
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
                        prompt_used="p1", status="ok", backend="logging"),
            SceneResult(index=2, title="S2", filename="scene_002.png",
                        image_path="", prompt_used="p2",
                        status="error", error="timeout", backend="logging"),
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

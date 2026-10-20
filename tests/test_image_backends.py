from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import requests

from ai_visualizer.image_backends import (
    BackendConfigurationError,
    ComfyUIBackend,
    GeminiApiBackend,
    ImageOptions,
    MockBackend,
    create_backend,
)


class _Response:
    def __init__(self, data: Any = None, content: bytes = b"", status: int = 200):
        self._data = data
        self.content = content
        self.status_code = status

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def _png_bytes() -> bytes:
    from PIL import Image
    from io import BytesIO

    buffer = BytesIO()
    Image.new("RGB", (2, 2), (12, 34, 56)).save(buffer, format="PNG")
    return buffer.getvalue()


def _workflow(path: Path) -> Path:
    data = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "test.safetensors"},
        },
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "{{prompt}}"}},
        "3": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 512, "height": 512, "batch_size": 1},
        },
        "4": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "{{filename_prefix}}"},
        },
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_backend_selection_and_invalid_name(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert isinstance(create_backend("mock"), MockBackend)
    with pytest.raises(BackendConfigurationError, match="Unknown image backend"):
        create_backend("other")


def test_api_requires_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(BackendConfigurationError, match="GEMINI_API_KEY"):
        create_backend("api")


def test_mock_backend_creates_distinguishable_png(tmp_path):
    path = tmp_path / "scene_001.png"
    result = MockBackend().generate("1", "A calm lake", path, ImageOptions("16:9"))
    assert result.success
    assert result.backend_name == "mock"
    assert result.model_identifier == "mock-labeled-png-v1"
    assert result.output_path == path
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_api_backend_saves_interaction_image_bytes(tmp_path):
    image_bytes = _png_bytes()
    response = _Response({"outputs": [{
        "type": "image",
        "mime_type": "image/png",
        "data": base64.b64encode(image_bytes).decode("ascii"),
    }]})
    with patch("ai_visualizer.image_backends.requests.post", return_value=response) as post:
        backend = GeminiApiBackend(api_key="test-key", model="test-image-model")
        path = tmp_path / "scene_001.png"
        result = backend.generate("1", "A cinematic lake", path, ImageOptions("16:9"))

    assert result.success
    assert result.output_path == path
    assert result.backend_name == "api"
    assert result.model_identifier == "test-image-model"
    assert path.read_bytes() == image_bytes
    assert post.call_args.kwargs["headers"]["x-goog-api-key"] == "test-key"
    assert post.call_args.kwargs["json"]["response_format"]["aspect_ratio"] == "16:9"


def test_api_backend_rejects_text_only_response(tmp_path):
    with patch(
        "ai_visualizer.image_backends.requests.post",
        return_value=_Response({"outputs": [{"type": "text", "text": "Generated"}]}),
    ):
        result = GeminiApiBackend(api_key="test-key").generate(
            "1", "A scene", tmp_path / "scene_001.png", ImageOptions()
        )
    assert not result.success
    assert "did not contain" in result.error
    assert not (tmp_path / "scene_001.png").exists()


def test_api_backend_reports_http_failure(tmp_path):
    with patch(
        "ai_visualizer.image_backends.requests.post",
        return_value=_Response(status=429),
    ):
        result = GeminiApiBackend(api_key="test-key").generate(
            "1", "A scene", tmp_path / "scene_001.png", ImageOptions()
        )
    assert not result.success
    assert "429" in result.error


def test_api_backend_reports_timeout(tmp_path):
    with patch(
        "ai_visualizer.image_backends.requests.post",
        side_effect=requests.Timeout("timed out"),
    ):
        result = GeminiApiBackend(api_key="test-key").generate(
            "1", "A scene", tmp_path / "scene_001.png", ImageOptions()
        )
    assert not result.success
    assert "timed out" in result.error


def test_unsupported_aspect_ratio_is_explicit_and_makes_no_request(tmp_path):
    with patch("ai_visualizer.image_backends.requests.post") as post:
        result = GeminiApiBackend(api_key="test-key").generate(
            "1", "A scene", tmp_path / "scene_001.png", ImageOptions("7:5")
        )
    assert not result.success
    assert "Unsupported aspect ratio" in result.error
    post.assert_not_called()


def test_local_backend_preflights_submits_polls_and_downloads(tmp_path):
    workflow_path = _workflow(tmp_path / "workflow.json")
    png = _png_bytes()
    object_info = {
        "CheckpointLoaderSimple": {},
        "CLIPTextEncode": {},
        "EmptyLatentImage": {},
        "SaveImage": {},
    }

    def fake_get(url, **kwargs):
        if url.endswith("/system_stats"):
            return _Response({})
        if url.endswith("/object_info"):
            return _Response(object_info)
        if url.endswith("/models/checkpoints"):
            return _Response(["test.safetensors"])
        if url.endswith("/history/job-1"):
            return _Response({"job-1": {"outputs": {
                "4": {"images": [{"filename": "generated.png", "type": "output"}]},
            }}})
        if url.endswith("/view"):
            assert kwargs["params"]["filename"] == "generated.png"
            return _Response(content=png)
        raise AssertionError(f"Unexpected GET {url}")

    with patch("ai_visualizer.image_backends.requests.get", side_effect=fake_get), patch(
        "ai_visualizer.image_backends.requests.post",
        return_value=_Response({"prompt_id": "job-1"}),
    ) as post:
        backend = ComfyUIBackend(
            base_url="http://comfy:8188",
            workflow_path=workflow_path,
            poll_interval=0.001,
        )
        path = tmp_path / "scene_001.png"
        result = backend.generate("1", "A quiet harbor", path, ImageOptions("16:9"))

    assert result.success
    assert result.backend_name == "local"
    assert result.model_identifier == "test.safetensors"
    assert path.read_bytes() == png
    submitted = post.call_args.kwargs["json"]["prompt"]
    assert submitted["2"]["inputs"]["text"] == "A quiet harbor"
    assert submitted["3"]["inputs"]["width"] == 1024
    assert submitted["3"]["inputs"]["height"] == 576
    assert submitted["4"]["inputs"]["filename_prefix"] == "sceneflow_1"


def test_local_backend_rejects_unavailable_model_before_submission(tmp_path):
    workflow_path = _workflow(tmp_path / "workflow.json")
    with patch("ai_visualizer.image_backends.requests.get") as get:
        get.side_effect = [
            _Response({}),
            _Response({
                "CheckpointLoaderSimple": {},
                "CLIPTextEncode": {},
                "EmptyLatentImage": {},
                "SaveImage": {},
            }),
            _Response([]),
        ]
        backend = ComfyUIBackend(workflow_path=workflow_path)
        with pytest.raises(BackendConfigurationError, match="not found"):
            backend.preflight()


def test_local_backend_requires_prompt_placeholder(tmp_path):
    workflow_path = _workflow(tmp_path / "workflow.json")
    data = json.loads(workflow_path.read_text(encoding="utf-8"))
    data["2"]["inputs"]["text"] = "fixed prompt"
    workflow_path.write_text(json.dumps(data), encoding="utf-8")
    with patch("ai_visualizer.image_backends.requests.get") as get:
        get.side_effect = [
            _Response({}),
            _Response({
                "CheckpointLoaderSimple": {},
                "CLIPTextEncode": {},
                "EmptyLatentImage": {},
                "SaveImage": {},
            }),
        ]
        backend = ComfyUIBackend(workflow_path=workflow_path)
        with pytest.raises(BackendConfigurationError, match=r"\{\{prompt\}\}"):
            backend.preflight()


def test_local_backend_reports_job_timeout(tmp_path):
    workflow_path = _workflow(tmp_path / "workflow.json")
    object_info = {
        "CheckpointLoaderSimple": {},
        "CLIPTextEncode": {},
        "EmptyLatentImage": {},
        "SaveImage": {},
    }

    def fake_get(url, **kwargs):
        if url.endswith("/system_stats"):
            return _Response({})
        if url.endswith("/object_info"):
            return _Response(object_info)
        if url.endswith("/models/checkpoints"):
            return _Response(["test.safetensors"])
        if url.endswith("/history/job-timeout"):
            return _Response({"job-timeout": {"outputs": {}}})
        raise AssertionError(f"Unexpected GET {url}")

    with patch("ai_visualizer.image_backends.requests.get", side_effect=fake_get), patch(
        "ai_visualizer.image_backends.requests.post",
        return_value=_Response({"prompt_id": "job-timeout"}),
    ):
        backend = ComfyUIBackend(
            workflow_path=workflow_path,
            timeout=0.01,
            poll_interval=0.001,
        )
        result = backend.generate(
            "1", "A scene", tmp_path / "scene_001.png", ImageOptions("16:9")
        )
    assert not result.success
    assert "timed out" in result.error


def test_local_backend_requires_workflow_configuration(monkeypatch):
    monkeypatch.delenv("COMFYUI_WORKFLOW", raising=False)
    with pytest.raises(BackendConfigurationError, match="COMFYUI_WORKFLOW"):
        create_backend("local")

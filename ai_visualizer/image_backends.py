"""Common image-generation contract and built-in backends."""

from __future__ import annotations

import base64
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol
from urllib.parse import urljoin, urlparse

import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")


SUPPORTED_ASPECT_RATIOS = {
    "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9",
}
DEFAULT_GEMINI_MODEL = "gemini-nano-banana-2.1"
_MODEL_NODE_INPUTS = {
    "CheckpointLoaderSimple": ("ckpt_name", "checkpoints"),
    "CheckpointLoader": ("ckpt_name", "checkpoints"),
    "UNETLoader": ("unet_name", "diffusion_models"),
}


class BackendConfigurationError(ValueError):
    """Raised when the selected backend cannot be configured."""


@dataclass(frozen=True)
class ImageOptions:
    aspect_ratio: Optional[str] = "16:9"


@dataclass(frozen=True)
class GenerationResult:
    output_path: Optional[Path]
    backend_name: str
    model_identifier: str
    success: bool
    error: str = ""


class ImageBackend(Protocol):
    name: str
    model_identifier: str

    def generate(
        self,
        scene_id: str,
        image_prompt: str,
        output_path: Path,
        options: ImageOptions,
    ) -> GenerationResult: ...


def validate_options(options: ImageOptions) -> None:
    if options.aspect_ratio is not None and options.aspect_ratio not in SUPPORTED_ASPECT_RATIOS:
        supported = ", ".join(sorted(SUPPORTED_ASPECT_RATIOS))
        raise ValueError(
            f"Unsupported aspect ratio {options.aspect_ratio!r}. Supported values: {supported}"
        )


def _dimensions(aspect_ratio: str) -> tuple[int, int]:
    width_ratio, height_ratio = (int(part) for part in aspect_ratio.split(":"))
    scale = 1024 / max(width_ratio, height_ratio)
    width = max(256, int(math.floor(width_ratio * scale / 64 + 0.5)) * 64)
    height = max(256, int(math.floor(height_ratio * scale / 64 + 0.5)) * 64)
    return width, height


def _positive_timeout(value: Optional[float], env_name: str, default: float) -> float:
    raw = value if value is not None else os.getenv(env_name, str(default))
    try:
        parsed = float(raw)
    except (TypeError, ValueError) as exc:
        raise BackendConfigurationError(f"{env_name} must be a positive number.") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise BackendConfigurationError(f"{env_name} must be a positive number.")
    return parsed


def _success(path: Path, backend: str, model: str) -> GenerationResult:
    return GenerationResult(path, backend, model, True)


def _failure(backend: str, model: str, error: Exception | str) -> GenerationResult:
    return GenerationResult(None, backend, model, False, str(error))


class MockBackend:
    name = "mock"
    model_identifier = "mock-labeled-png-v1"

    def generate(
        self,
        scene_id: str,
        image_prompt: str,
        output_path: Path,
        options: ImageOptions,
    ) -> GenerationResult:
        try:
            validate_options(options)
            from PIL import Image, ImageDraw

            width, height = _dimensions(options.aspect_ratio or "16:9")
            image = Image.new("RGB", (width, height), (30, 30, 50))
            draw = ImageDraw.Draw(image)
            draw.text((24, 18), "[MOCK IMAGE - not AI-generated]", fill=(255, 180, 50))
            draw.text((24, 48), f"Scene {scene_id}", fill=(200, 200, 255))
            words = image_prompt.split()
            lines: List[str] = []
            line: List[str] = []
            for word in words:
                line.append(word)
                if len(" ".join(line)) > 76:
                    lines.append(" ".join(line[:-1]))
                    line = [word]
            if line:
                lines.append(" ".join(line))
            y = 88
            for text in lines[:12]:
                draw.text((24, y), text, fill=(220, 220, 220))
                y += 24
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path, format="PNG")
            return _success(output_path, self.name, self.model_identifier)
        except Exception as exc:
            return _failure(self.name, self.model_identifier, exc)


class GeminiApiBackend:
    name = "api"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.api_key = (api_key if api_key is not None else os.getenv("GEMINI_API_KEY", "")).strip()
        self.model_identifier = (model or os.getenv("GEMINI_IMAGE_MODEL") or DEFAULT_GEMINI_MODEL).strip()
        self.timeout = _positive_timeout(timeout, "IMAGE_REQUEST_TIMEOUT", 120)
        if not self.api_key or self.api_key == "your_gemini_api_key_here":
            raise BackendConfigurationError(
                "GEMINI_API_KEY is required when IMAGE_BACKEND=api."
            )
        if not self.model_identifier:
            raise BackendConfigurationError("GEMINI_IMAGE_MODEL must not be empty.")

    def generate(
        self,
        scene_id: str,
        image_prompt: str,
        output_path: Path,
        options: ImageOptions,
    ) -> GenerationResult:
        try:
            validate_options(options)
            if not image_prompt.strip():
                raise ValueError("image_prompt is empty")
            payload: Dict[str, Any] = {
                "model": self.model_identifier,
                "input": [{"type": "text", "text": image_prompt}],
            }
            if options.aspect_ratio:
                payload["response_format"] = {
                    "type": "image",
                    "mime_type": "image/png",
                    "aspect_ratio": options.aspect_ratio,
                }
            response = requests.post(
                "https://generativelanguage.googleapis.com/v1beta/interactions",
                headers={
                    "x-goog-api-key": self.api_key,
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            image = data.get("output_image")
            if not isinstance(image, dict):
                outputs = data.get("outputs", [])
                image = next(
                    (
                        item for item in outputs
                        if isinstance(item, dict) and item.get("type") == "image"
                    ),
                    None,
                )
            encoded = image.get("data") if isinstance(image, dict) else None
            if not isinstance(encoded, str) or not encoded:
                raise ValueError("Gemini response did not contain output_image.data")
            image_bytes = base64.b64decode(encoded, validate=True)
            if not image_bytes:
                raise ValueError("Gemini returned empty image data")
            if not image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError("Gemini response was not PNG image data")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(image_bytes)
            return _success(output_path, self.name, self.model_identifier)
        except Exception as exc:
            return _failure(self.name, self.model_identifier, exc)


class ComfyUIBackend:
    name = "local"
    _PROMPT_TOKEN = "{{prompt}}"

    def __init__(
        self,
        base_url: Optional[str] = None,
        workflow_path: Optional[str | Path] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        poll_interval: Optional[float] = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("COMFYUI_BASE_URL", "http://127.0.0.1:8188")).rstrip("/") + "/"
        parsed_url = urlparse(self.base_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise BackendConfigurationError("COMFYUI_BASE_URL must be a valid HTTP(S) URL.")
        configured_workflow = workflow_path or os.getenv("COMFYUI_WORKFLOW", "")
        if not configured_workflow:
            raise BackendConfigurationError(
                "COMFYUI_WORKFLOW must point to a ComfyUI API-format workflow JSON file."
            )
        self.workflow_path = Path(configured_workflow).expanduser()
        if not self.workflow_path.is_file():
            raise BackendConfigurationError(f"ComfyUI workflow not found: {self.workflow_path}")
        try:
            workflow = json.loads(self.workflow_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BackendConfigurationError(f"Could not read ComfyUI workflow: {exc}") from exc
        if not isinstance(workflow, dict) or not workflow:
            raise BackendConfigurationError("ComfyUI workflow must be a non-empty API-format object.")
        self.workflow = workflow
        self.model_override = (model if model is not None else os.getenv("COMFYUI_MODEL", "")).strip()
        self.timeout = _positive_timeout(timeout, "COMFYUI_TIMEOUT_SECONDS", 600)
        self.poll_interval = _positive_timeout(poll_interval, "COMFYUI_POLL_INTERVAL", 2)
        self.model_identifier = self.model_override or self._workflow_model_names()
        self._preflight_done = False

    def _model_nodes(self) -> List[tuple[str, str, str]]:
        found: List[tuple[str, str, str]] = []
        for node_id, node in self.workflow.items():
            if not isinstance(node, dict):
                continue
            class_type = node.get("class_type")
            configured = _MODEL_NODE_INPUTS.get(class_type)
            if configured:
                input_name, folder = configured
                value = node.get("inputs", {}).get(input_name)
                if isinstance(value, str):
                    found.append((node_id, input_name, folder))
        return found

    def _workflow_model_names(self) -> str:
        names = [
            str(self.workflow[node_id]["inputs"][input_name])
            for node_id, input_name, _ in self._model_nodes()
        ]
        return ", ".join(names) if names else "workflow-defined"

    def preflight(self) -> None:
        if self._preflight_done:
            return
        try:
            stats = requests.get(urljoin(self.base_url, "system_stats"), timeout=10)
            stats.raise_for_status()
            object_info_response = requests.get(urljoin(self.base_url, "object_info"), timeout=20)
            object_info_response.raise_for_status()
            object_info = object_info_response.json()
            if not isinstance(object_info, dict):
                raise ValueError("ComfyUI /object_info response was not an object")
            for node in self.workflow.values():
                if not isinstance(node, dict):
                    raise ValueError("Workflow contains an invalid node")
                class_type = node.get("class_type")
                if class_type and class_type not in object_info:
                    raise ValueError(f"ComfyUI workflow node is unavailable: {class_type}")
            if not any(
                isinstance(node, dict) and node.get("class_type") == "SaveImage"
                for node in self.workflow.values()
            ):
                raise ValueError("ComfyUI workflow must contain a SaveImage node")
            if not any(
                isinstance(value, str) and self._PROMPT_TOKEN in value
                for node in self.workflow.values()
                if isinstance(node, dict)
                for value in node.get("inputs", {}).values()
            ):
                raise ValueError(
                    "ComfyUI workflow must include {{prompt}} in the positive text input."
                )

            model_nodes = self._model_nodes()
            if not model_nodes:
                raise ValueError(
                    "ComfyUI workflow must use CheckpointLoaderSimple, CheckpointLoader, "
                    "or UNETLoader so model availability can be verified."
                )
            if self.model_override and len(model_nodes) > 1:
                raise ValueError(
                    "COMFYUI_MODEL cannot override workflows with multiple model-loader nodes."
                )
            model_names: List[str] = []
            for node_id, input_name, folder in model_nodes:
                model = self.model_override or self.workflow[node_id]["inputs"][input_name]
                models_response = requests.get(
                    urljoin(self.base_url, f"models/{folder}"), timeout=20
                )
                models_response.raise_for_status()
                available = models_response.json()
                if not isinstance(available, list) or model not in available:
                    raise ValueError(
                        f"ComfyUI model {model!r} was not found in /models/{folder}."
                    )
                self.workflow[node_id]["inputs"][input_name] = model
                model_names.append(model)
            self.model_identifier = ", ".join(model_names)
            self._preflight_done = True
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            raise BackendConfigurationError(f"ComfyUI preflight failed: {exc}") from exc

    def generate(
        self,
        scene_id: str,
        image_prompt: str,
        output_path: Path,
        options: ImageOptions,
    ) -> GenerationResult:
        try:
            validate_options(options)
            if not image_prompt.strip():
                raise ValueError("image_prompt is empty")
            if not self._preflight_done:
                self.preflight()
            width, height = _dimensions(options.aspect_ratio or "16:9")
            workflow = json.loads(json.dumps(self.workflow))
            prompt_used = False
            for node in workflow.values():
                if not isinstance(node, dict):
                    continue
                inputs = node.get("inputs", {})
                for key, value in list(inputs.items()):
                    if isinstance(value, str) and self._PROMPT_TOKEN in value:
                        prompt_used = True
                    if isinstance(value, str):
                        inputs[key] = value.replace(self._PROMPT_TOKEN, image_prompt)
                        inputs[key] = inputs[key].replace(
                            "{{filename_prefix}}", f"sceneflow_{scene_id}"
                        )
                        inputs[key] = inputs[key].replace("{{model}}", self.model_identifier)
                if node.get("class_type") in {
                    "EmptyLatentImage", "EmptySD3LatentImage", "EmptyLatentImageAdvanced",
                }:
                    if "width" in inputs and "height" in inputs:
                        inputs["width"] = width
                        inputs["height"] = height
            if not prompt_used:
                raise ValueError("ComfyUI workflow does not contain a {{prompt}} placeholder")
            replacements = {
                "{{width}}": str(width),
                "{{height}}": str(height),
            }
            for node in workflow.values():
                if not isinstance(node, dict):
                    continue
                inputs = node.get("inputs", {})
                for key, value in list(inputs.items()):
                    if isinstance(value, str):
                        for token, replacement in replacements.items():
                            inputs[key] = inputs[key].replace(token, replacement)
            if options.aspect_ratio and not any(
                isinstance(node, dict)
                and node.get("class_type") in {
                    "EmptyLatentImage", "EmptySD3LatentImage", "EmptyLatentImageAdvanced",
                }
                and "width" in node.get("inputs", {})
                and "height" in node.get("inputs", {})
                for node in workflow.values()
            ):
                raise ValueError(
                    "ComfyUI workflow cannot apply aspect_ratio; include a supported latent node."
                )
            response = requests.post(
                urljoin(self.base_url, "prompt"),
                json={"prompt": workflow},
                timeout=20,
            )
            response.raise_for_status()
            job = response.json()
            prompt_id = job.get("prompt_id")
            if not prompt_id:
                raise ValueError(f"ComfyUI did not accept the workflow: {job}")
            deadline = time.monotonic() + self.timeout
            history: Dict[str, Any] = {}
            while time.monotonic() < deadline:
                time.sleep(self.poll_interval)
                history_response = requests.get(
                    urljoin(self.base_url, f"history/{prompt_id}"), timeout=20
                )
                history_response.raise_for_status()
                history_data = history_response.json()
                history = history_data.get(prompt_id, {})
                status = history.get("status", {})
                if status.get("status_str") == "error":
                    raise ValueError(f"ComfyUI job failed: {status.get('messages', status)}")
                outputs = history.get("outputs", {})
                images = [
                    image
                    for output in outputs.values()
                    for image in output.get("images", [])
                    if isinstance(image, dict) and image.get("filename")
                ]
                if images:
                    break
            else:
                raise TimeoutError(f"ComfyUI job {prompt_id} timed out")
            image = images[0]
            image_response = requests.get(
                urljoin(self.base_url, "view"),
                params={
                    "filename": image["filename"],
                    "subfolder": image.get("subfolder", ""),
                    "type": image.get("type", "output"),
                },
                timeout=60,
            )
            image_response.raise_for_status()
            if not image_response.content:
                raise ValueError("ComfyUI returned an empty image")
            if not image_response.content.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError("ComfyUI response was not PNG image data")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(image_response.content)
            return _success(output_path, self.name, self.model_identifier)
        except Exception as exc:
            return _failure(self.name, self.model_identifier, exc)


def create_backend(name: Optional[str] = None) -> ImageBackend:
    selected = (name or os.getenv("IMAGE_BACKEND", "mock")).strip().lower()
    if selected in {"mock", "placeholder"}:
        return MockBackend()
    if selected == "api":
        return GeminiApiBackend()
    if selected == "local":
        return ComfyUIBackend()
    raise BackendConfigurationError(
        f"Unknown image backend {selected!r}; choose mock, api, or local."
    )

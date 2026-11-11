"""Local Ollama client. All inference stays on this machine."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.8:27b")
HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")


def chat(
    messages: list[dict[str, str]],
    *,
    model: str,
    temperature: float,
    seed: int,
    num_predict: int = 2200,
) -> str:
    """Return the assistant text. Thinking is disabled so the answer is in content."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": "json",
        "think": False,
        "keep_alive": "30m",
        "options": {
            "temperature": temperature,
            "top_k": 20 if temperature > 0 else 1,
            "top_p": 0.9 if temperature > 0 else 1,
            "seed": seed,
            "num_predict": num_predict,
            "num_ctx": 8192,
            "repeat_penalty": 1,
        },
    }
    request = urllib.request.Request(
        HOST + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if exc.code == 400 and "think" in detail.lower():
            payload.pop("think", None)
            return _post(payload)
        raise RuntimeError(f"Ollama HTTP {exc.code}: {detail[:500]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Could not reach Ollama. Start it and pull the model before running."
        ) from exc
    return _content(body)


def _post(payload: dict[str, Any]) -> str:
    request = urllib.request.Request(
        HOST + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        body = json.loads(response.read().decode("utf-8"))
    return _content(body)


def _content(body: dict[str, Any]) -> str:
    message = body.get("message") or {}
    content = (message.get("content") or "").strip()
    thinking = (message.get("thinking") or "").strip()
    if content:
        return content
    return thinking


_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    match = _JSON_OBJECT.search(text)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None

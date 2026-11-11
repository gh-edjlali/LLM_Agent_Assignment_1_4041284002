"""Local Ollama client. All inference stays on this machine (stdlib only)."""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

# The model name announced for grading can be passed with --model or OLLAMA_MODEL.
# If that exact tag is not installed, resolve_model() picks the closest installed one.
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.6:27b")
_host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").strip().rstrip("/")
if not _host.startswith(("http://", "https://")):
    _host = "http://" + _host
HOST = _host
REQUEST_TIMEOUT = float(os.environ.get("AGENT_LLM_TIMEOUT", "300"))


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _request(path: str, payload: Optional[Dict[str, Any]], timeout: float) -> Dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        HOST + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def installed_models() -> List[str]:
    try:
        body = _request("/api/tags", None, timeout=10)
    except Exception:
        return []
    return [m.get("name", "") for m in body.get("models", []) if m.get("name")]


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9.]", "", name.lower())


def resolve_model(requested: str) -> str:
    """Return `requested` if installed, otherwise the most similar installed model."""
    models = installed_models()
    if not models or requested in models:
        return requested
    if ":" not in requested and f"{requested}:latest" in models:
        return f"{requested}:latest"
    want = _norm(requested)
    want_size = re.search(r"(\d+(?:\.\d+)?)b\b", requested.lower())

    def score(name: str) -> tuple:
        n = name.lower()
        size = re.search(r"(\d+(?:\.\d+)?)b\b", n)
        return (
            _norm(name) == want,
            bool(want_size and size and size.group(1) == want_size.group(1)),
            "qwen" in n and "qwen" in requested.lower(),
            "embed" not in n,
            float(size.group(1)) if size else 0.0,
        )

    best = max(models, key=score)
    _log(f"[llm] model {requested!r} not installed; using {best!r} (installed: {models})")
    return best


def chat(
    messages: List[Dict[str, str]],
    *,
    model: str,
    temperature: float,
    seed: int,
    num_predict: int = 3000,
    json_mode: bool = True,
    timeout: Optional[float] = None,
) -> str:
    """Return the assistant text. Thinking is disabled so the answer is in content."""
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "keep_alive": "30m",
        "options": {
            "temperature": temperature,
            "top_k": 20 if temperature > 0 else 1,
            "top_p": 0.9 if temperature > 0 else 1,
            "seed": seed,
            "num_predict": num_predict,
            "num_ctx": 8192,
        },
    }
    if json_mode:
        payload["format"] = "json"
    timeout = timeout or REQUEST_TIMEOUT
    try:
        body = _request("/api/chat", payload, timeout)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if exc.code == 400 and "think" in detail.lower():
            # Model without a thinking switch: retry once without the flag.
            payload.pop("think", None)
            body = _request("/api/chat", payload, timeout)
        else:
            raise RuntimeError(f"Ollama HTTP {exc.code}: {detail[:500]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {HOST}. Start it and pull the model before running."
        ) from exc
    return _content(body)


def _content(body: Dict[str, Any]) -> str:
    message = body.get("message") or {}
    content = (message.get("content") or "").strip()
    # Some models leak <think> blocks into content; drop them.
    content = re.sub(r"<think>[\s\S]*?</think>", "", content).strip()
    if content:
        return content
    return (message.get("thinking") or "").strip()


_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    for candidate in (text, *(m.group(0) for m in [_JSON_OBJECT.search(text)] if m)):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                data = json.loads(candidate, strict=False)  # tolerate raw newlines in strings
            except json.JSONDecodeError:
                continue
        if isinstance(data, dict):
            return data
    return None

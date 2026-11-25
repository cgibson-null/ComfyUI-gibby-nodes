"""Thin HTTP client for a llama.cpp / llama-swap / Unsloth Studio server.

The servers speak plain OpenAI-shaped JSON over HTTP, so ``requests`` calls
cover everything: list_models, chat_completion, the Unsloth load/unload and
pre-load probe endpoints. llama-swap proxies a llama-server it spins up on
demand, which is why the chat timeout has to outlast a cold model load.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import requests

# A cold llama-swap model swap on a large GGUF takes 30-90s before the first
# token; matches llama-swap's server-side healthCheckTimeout.
DEFAULT_TIMEOUT = 300
# Listing models only reads llama-swap's own config, never a backend process.
MODELS_TIMEOUT = 10
# Unsloth's probes read the GGUF header without touching weights; the room is
# for a validate that resolves a Hub identifier over the network.
PREFLIGHT_TIMEOUT = 60


class LlamaCppError(RuntimeError):
    """Anything that isn't a clean 2xx, carrying the server's own error text."""


def _base(url: str) -> str:
    return url.rstrip("/")


def _auth_headers(api_key: str | None) -> dict[str, str]:
    """Bearer auth when a token is set; no header for an open server."""
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def _raise_for_response(response: requests.Response) -> None:
    if response.ok:
        return
    detail = response.text.strip()
    try:
        payload = response.json()
        # OpenAI's {"error": {"message": ...}} or llama-swap's {"error": "..."}.
        if isinstance(payload, dict):
            err = payload.get("error")
            if isinstance(err, dict):
                detail = err.get("message", detail)
            elif isinstance(err, str):
                detail = err
    except ValueError:
        pass
    raise LlamaCppError(f"HTTP {response.status_code} from {response.url}: {detail}")


def list_models(base_url: str, timeout: int = MODELS_TIMEOUT, api_key: str | None = None) -> list[str]:
    """Return every model id the server (or llama-swap) currently exposes."""
    response = requests.get(
        f"{_base(base_url)}/v1/models",
        timeout=timeout,
        headers=_auth_headers(api_key),
    )
    _raise_for_response(response)
    payload = response.json()
    return [entry["id"] for entry in payload.get("data", [])]


def unload_model(base_url: str, model: str, timeout: int = MODELS_TIMEOUT, api_key: str | None = None) -> None:
    """llama-swap's ``/api/models/unload/:model_id`` - a llama-swap-only endpoint,
    and the only Ollama ``keep_alive`` behaviour that maps to a real call
    (``keep_alive=0``). Positive values have no runtime equivalent; llama-swap
    only exposes a config-time ``ttl``. Callers treat a failure as best-effort."""
    response = requests.post(
        f"{_base(base_url)}/api/models/unload/{model}",
        timeout=timeout,
        headers=_auth_headers(api_key),
    )
    _raise_for_response(response)


def chat_completion(
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    *,
    options: dict[str, Any] | None = None,
    think: bool = False,
    json_mode: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
    api_key: str | None = None,
) -> tuple[dict[str, Any], float]:
    """POST ``/v1/chat/completions``, return ``(parsed JSON response, elapsed_s)``.

    ``options`` is the already-filtered dict from
    :func:`llamacpp_shared.filter_enabled_options`; llama.cpp server extensions
    (top_k, min_p, repeat_penalty, mirostat*, n_predict, reasoning_budget) go
    through as extra top-level fields. ``reasoning_effort`` is the exception - a
    chat-template hint, so it is merged into ``chat_template_kwargs`` instead.
    """
    options = dict(options) if options else {}
    reasoning_effort = options.pop("reasoning_effort", None)

    # enable_thinking is what actually turns reasoning on for hybrid-reasoning
    # models; reasoning_effort is only added when set - templates ignore an
    # absent kwarg fine, no need to send a null.
    template_kwargs: dict[str, Any] = {"enable_thinking": think}
    if reasoning_effort:
        template_kwargs["reasoning_effort"] = reasoning_effort

    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "chat_template_kwargs": template_kwargs,
    }
    if options:
        body.update(options)
    if json_mode:
        body["response_format"] = {"type": "json_object"}

    start = time.monotonic()
    response = requests.post(
        f"{_base(base_url)}/v1/chat/completions",
        json=body,
        timeout=timeout,
        headers=_auth_headers(api_key),
    )
    elapsed_s = time.monotonic() - start
    if response.status_code == 413:
        # A 413 usually comes back with no explanation; the request size is the
        # one number that explains it (the limit itself is a server setting).
        sent = len(response.request.body or b"")
        raise LlamaCppError(
            f"HTTP 413 from {response.url}: this request is {sent / 2**20:.0f} MiB, more than the server "
            "takes. Send less at a time -- a smaller max_image_size, fewer frames_per_video, fewer "
            "reference images -- or raise the server's own limit (llama-server "
            "--limit-request-size; a proxy or server behind it sets its own)."
        )
    _raise_for_response(response)
    return response.json(), elapsed_s


def extract_reply(response: dict[str, Any], think: bool) -> tuple[str, str | None]:
    """``(content, reasoning_content)`` from a chat-completion response. Thinking
    is None when it wasn't requested, even if the server sent the field."""
    message = response["choices"][0]["message"]
    content = message.get("content") or ""
    thinking = message.get("reasoning_content") if think else None
    return content, thinking


def detect_backend(base_url: str, api_key: str | None = None) -> str:
    """'unsloth' when the server answers Unsloth Studio's unauthenticated
    /api/auth/status with {"initialized": ...}, else 'llama'. Probing it is the
    only way to tell them apart - both otherwise speak the same OpenAI surface."""
    try:
        response = requests.get(
            f"{_base(base_url)}/api/auth/status",
            timeout=MODELS_TIMEOUT,
            headers=_auth_headers(api_key),
        )
        if response.ok:
            payload = response.json()
            if isinstance(payload, dict) and "initialized" in payload:
                return "unsloth"
    except (requests.RequestException, ValueError):
        pass
    return "llama"


def list_local_models(base_url: str, api_key: str | None = None) -> list[dict[str, Any]]:
    """Unsloth Studio's local model list (GET /api/models/local). Resolving a
    /v1/models id to an entry's ``path`` is what stops /load from downloading a
    model that is already on disk."""
    response = requests.get(
        f"{_base(base_url)}/api/models/local",
        timeout=MODELS_TIMEOUT,
        headers=_auth_headers(api_key),
    )
    _raise_for_response(response)
    payload = response.json()
    models = payload.get("models", [])
    return [entry for entry in models if isinstance(entry, dict)]


def resolve_local_model(
    base_url: str, model: str, api_key: str | None = None
) -> tuple[str, str] | None:
    """Map a /v1/models id to its local copy's ``(load_identity, chat_identity)``,
    or None when there is no local copy.

    They differ by on-disk layout: a plain folder (LM Studio) uses its ``path``
    for both, but an HF hub cache must chat by the repo ``id`` (the snapshot dir
    404s) while loading from the snapshot dir (a repo-id load re-resolves through
    the Hub and downloads). When they differ the caller has to load explicitly -
    see the ``load != chat`` branch in _maybe_unsloth_load.
    """
    target = (model or "").strip().lower()
    if not target:
        return None
    try:
        entries = list_local_models(base_url, api_key)
    except LlamaCppError:
        return None
    for entry in entries:
        for field in ("model_id", "display_name", "id", "path"):
            value = entry.get(field)
            if isinstance(value, str) and value.strip().lower() == target:
                return _local_identities(entry)
    return None


def _local_identities(entry: dict[str, Any]) -> tuple[str, str] | None:
    path = entry.get("path")
    if not isinstance(path, str) or not path:
        return None
    if "models--" not in path:
        return path, path
    # HF hub cache: the cache top dir is mis-read as a transformers model
    # (HTTP 500, "Should have a model_type key"), so load the active snapshot.
    cache = Path(path)
    rev = None
    refs_main = cache / "refs" / "main"
    if refs_main.is_file():
        rev = refs_main.read_text(encoding="utf-8").strip() or None
    if not rev:
        snapshots = cache / "snapshots"
        if snapshots.is_dir():
            dirs = [d for d in snapshots.iterdir() if d.is_dir()]
            if len(dirs) == 1:
                rev = dirs[0].name
    if not rev:
        return None
    repo_id = entry.get("id")
    if not isinstance(repo_id, str) or not repo_id:
        return None
    return str(cache / "snapshots" / rev), repo_id


def unsloth_preflight(
    base_url: str,
    endpoint: str,
    model_path: str,
    params: dict[str, Any],
    api_key: str | None = None,
) -> dict[str, Any]:
    """POST one of Unsloth Studio's pre-load probes (``"validate"``,
    ``"estimate-memory"``) and return its JSON. Both price a prospective load
    without doing it; estimate-memory answers with the ``context_fitted`` and
    ``gpu_layers`` the caller sends back to /load.

    Anything other than an answer raises: the caller is about to spend a model
    load on the priced command, so a server that will not price it is worth
    knowing about rather than loading blind.
    """
    try:
        response = requests.post(
            f"{_base(base_url)}/api/inference/{endpoint}",
            json={"model_path": model_path, **params},
            timeout=PREFLIGHT_TIMEOUT,
            headers=_auth_headers(api_key),
        )
    except requests.RequestException as exc:
        raise LlamaCppError(f"{endpoint} probe did not answer: {exc}") from exc
    _raise_for_response(response)
    return response.json()


def unsloth_load(
    base_url: str,
    model_path: str,
    params: dict[str, Any],
    timeout: int = DEFAULT_TIMEOUT,
    api_key: str | None = None,
) -> dict[str, Any]:
    """POST Unsloth Studio's ``/api/inference/load``. Blocks until the load settles
    (a cold 27B takes minutes), and no-ops when the model is already loaded with
    matching settings."""
    body = {"model_path": model_path, **params}
    response = requests.post(
        f"{_base(base_url)}/api/inference/load",
        json=body,
        timeout=timeout,
        headers=_auth_headers(api_key),
    )
    _raise_for_response(response)
    return response.json()


def unsloth_unload(
    base_url: str,
    model_path: str,
    timeout: int = DEFAULT_TIMEOUT,
    api_key: str | None = None,
) -> None:
    """POST Unsloth Studio's ``/api/inference/unload`` - the keep_alive=0 path for
    an Unsloth backend. ``model_path`` must name the model the way the server
    loaded it, so callers reuse the load's identity."""
    response = requests.post(
        f"{_base(base_url)}/api/inference/unload",
        json={"model_path": model_path},
        timeout=timeout,
        headers=_auth_headers(api_key),
    )
    _raise_for_response(response)


def describe_speed(response: dict[str, Any], elapsed_s: float) -> str:
    """Human-readable tokens/sec, e.g. ``"41.2 tok/s (187 tokens, server-reported)"``.

    The OpenAI endpoint only guarantees ``usage``, so this prefers a server
    ``timings.predicted_per_second`` when the build has one and otherwise falls
    back to client-side ``completion_tokens / elapsed_s`` - which includes prefill
    and network time, so it reads low on vision requests.
    """
    usage = response.get("usage") or {}
    completion_tokens = usage.get("completion_tokens")

    timings = response.get("timings") or {}
    server_tps = timings.get("predicted_per_second")

    if server_tps:
        tok_count = timings.get("predicted_n", completion_tokens)
        return f"{server_tps:.1f} tok/s ({tok_count} tokens, server-reported)"

    if completion_tokens and elapsed_s > 0:
        tps = completion_tokens / elapsed_s
        return f"{tps:.1f} tok/s ({completion_tokens} tokens, wall-clock incl. prefill)"

    return "tok/s unavailable (no usage/timings in response)"

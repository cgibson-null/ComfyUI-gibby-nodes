"""Thin HTTP client for a llama.cpp / llama-swap OpenAI-compatible server.

Replaces the ``ollama`` Python package used by the original nodes. There is no
official llama.cpp client library worth depending on for this -- the server
speaks plain OpenAI-shaped JSON over HTTP, so a couple of ``requests`` calls
cover everything the enhanced nodes need:

  * ``list_models``            -> GET  {base_url}/v1/models
  * ``chat_completion``        -> POST {base_url}/v1/chat/completions
                                   (returns (response, elapsed_s))
  * ``describe_speed``         -> tokens/sec summary from a chat_completion result

Talks to llama-swap exactly the same way it talks to bare llama-server -- both
implement the same OpenAI-compatible surface, llama-swap just proxies to a
llama-server it spins up on demand. That on-demand spin-up is *why* the
timeout below is generous: the first request after an idle model needs to
wait out a full model load (weights off disk into VRAM), not just an
inference.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import requests

# A cold llama-swap model swap on a large GGUF can take 30-90s (disk I/O +
# VRAM upload) before the first token even starts. Matches the
# `healthCheckTimeout` llama-swap is configured with server-side -- no point
# giving up client-side before the server would.
DEFAULT_TIMEOUT = 300
# Listing models never touches a backend process, just llama-swap's own
# config -- should return in well under a second even while a model is mid-swap.
MODELS_TIMEOUT = 10


class LlamaCppError(RuntimeError):
    """Raised for anything that isn't a clean 2xx from the server.

    Carries the server's own error text when there is one -- llama-swap in
    particular returns useful messages ("no model id could be identified",
    "failed to load config...") that are worth surfacing verbatim rather than
    burying behind a generic "request failed".
    """


def _base(url: str) -> str:
    return url.rstrip("/")


def _auth_headers(api_key: str | None) -> dict[str, str]:
    """Bearer auth for servers that gate the OpenAI API behind a token
    (e.g. Unsloth Studio). Empty/None -> no header, so open llama.cpp and
    llama-swap servers keep working unchanged."""
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def _raise_for_response(response: requests.Response) -> None:
    if response.ok:
        return
    detail = response.text.strip()
    try:
        payload = response.json()
        # OpenAI-style {"error": {"message": "..."}} or llama-swap's own
        # {"error": "..."} shape -- try both before falling back to raw text.
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
    """POST llama-swap's ``/api/models/unload/:model_id`` to free VRAM now.

    This is a llama-swap-only management endpoint -- it does not exist on a
    bare ``llama-server``. It is the one piece of Ollama's per-request
    ``keep_alive`` this pack can actually implement for real: Ollama's
    ``keep_alive=0`` means "unload right after this response," and that maps
    directly onto calling this endpoint once generation finishes. There is no
    equivalent for positive `keep_alive` values (extend/override how long a
    model stays warm) -- llama-swap only exposes that as a *config-time* `ttl`
    per model, not a runtime knob. Raises :class:`LlamaCppError` on failure;
    callers should treat that as best-effort and not fail the whole node run
    over it.
    """
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

    ``elapsed_s`` is wall-clock time around just the ``requests.post`` call --
    feeds :func:`describe_speed`'s wall-clock fallback for tokens/sec when the
    server doesn't hand back its own timings (see there for why both exist).

    ``options`` is the already-filtered dict from
    :func:`llamacpp_shared.filter_enabled_options` -- e.g.
    ``{"temperature": 0.7, "top_k": 40}``. Sampler fields beyond the OpenAI
    standard (top_k, min_p, repeat_penalty, repeat_last_n, mirostat*,
    n_predict, reasoning_budget) are llama.cpp server extensions accepted as
    extra top-level JSON fields; they are not part of the OpenAI spec but
    llama.cpp's server has supported them for a long time.

    ``reasoning_effort`` is the one exception: it isn't a generic top-level
    field, it's a chat-template hint, so if it's present in ``options`` it
    gets pulled out and merged into ``chat_template_kwargs`` alongside
    ``enable_thinking`` instead of being sent top-level.
    """
    options = dict(options) if options else {}
    reasoning_effort = options.pop("reasoning_effort", None)

    # chat_template_kwargs.enable_thinking is what actually turns reasoning
    # on/off for hybrid-reasoning models (proven against this exact stack
    # already). reasoning_format defaults to "auto" server side, which is
    # enough to get reasoning_content split out below without forcing a
    # specific dialect. reasoning_effort only added when set -- most
    # templates ignore an unset/absent kwarg fine, no need to send a null.
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
    _raise_for_response(response)
    return response.json(), elapsed_s


def extract_reply(response: dict[str, Any], think: bool) -> tuple[str, str | None]:
    """Pull ``(content, reasoning_content)`` out of a chat-completion response.

    ``reasoning_content`` is llama.cpp's dedicated field for thinking output
    once the template supports it and thinking was requested -- mirrors
    DeepSeek's API convention. ``None`` when thinking wasn't requested, even
    if the server happened to include the field anyway.
    """
    message = response["choices"][0]["message"]
    content = message.get("content") or ""
    thinking = message.get("reasoning_content") if think else None
    return content, thinking


def detect_backend(base_url: str, api_key: str | None = None) -> str:
    """'unsloth' if the server exposes Unsloth Studio's public auth-status
    endpoint, otherwise 'llama' (bare llama-server or llama-swap).

    GET /api/auth/status is unauthenticated on Unsloth and returns
    {"initialized": ...}; llama.cpp/llama-swap answer 404. Probing it is the
    only reliable way to tell the two apart -- both otherwise speak the same
    OpenAI surface."""
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
    """Return Unsloth Studio's local model list (GET /api/models/local).

    Each entry carries the identifiers the load path needs:
      * ``model_id``  -- the repo-style id (``org/name``) that /v1/models shows
      * ``path``      -- the on-disk directory holding the GGUF
      * ``id``        -- same path for local (lmstudio/hf_cache) models
    Resolving a /v1/models id to its ``path`` is what stops /load from treating
    a locally-present model as a Hub repo and downloading it."""
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
    or None when the model has no local copy (then the caller uses the repo id
    for both and Unsloth downloads it).

    The two identities differ by on-disk layout:
      * plain folder (LM Studio) -- both are the ``path``; a chat with it
        auto-loads from disk, so no explicit load is needed.
      * HF hub cache (``.../models--org--name``) -- chat must use the repo
        ``id`` (the snapshot dir is not a valid chat model, HTTP 404), but a
        repo-id load re-resolves through the Hub and downloads even with a
        complete cache. Load uses the snapshot dir instead, which loads
        straight from disk and picks up the mmproj. The caller must perform
        that load explicitly before chatting -- see the ``load != chat``
        branch in LLMConnect._maybe_unsloth_load.
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
    # (HTTP 500, "Should have a model_type key"), so the loadable dir is the
    # active snapshot under it.
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


def unsloth_load(
    base_url: str,
    model_path: str,
    params: dict[str, Any],
    timeout: int = DEFAULT_TIMEOUT,
    api_key: str | None = None,
) -> dict[str, Any]:
    """POST Unsloth Studio's ``/api/inference/load`` with the model's load
    params (max_seq_length, speculative_type, cache_type_kv, ...).

    Blocks until the load settles -- a 27B GGUF cold load can take a couple
    of minutes, hence the shared DEFAULT_TIMEOUT. Unsloth no-ops the call if
    the model is already loaded with matching settings, so re-running a
    workflow is cheap."""
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
    """POST Unsloth Studio's ``/api/inference/unload`` -- the keep_alive=0
    path for an Unsloth backend (llama-swap uses ``/api/models/unload/:id``).

    ``model_path`` must name the model the way the server loaded it, so
    callers send the same local-path-or-repo-id they used for the load. A
    teardown of a big GGUF can take a while, hence the shared DEFAULT_TIMEOUT.
    """
    response = requests.post(
        f"{_base(base_url)}/api/inference/unload",
        json={"model_path": model_path},
        timeout=timeout,
        headers=_auth_headers(api_key),
    )
    _raise_for_response(response)


def describe_speed(response: dict[str, Any], elapsed_s: float) -> str:
    """Human-readable tokens/sec summary -- proof-of-life more than a precise
    benchmark, e.g. ``"41.2 tok/s (187 tokens, server-reported)"``. The
    caller prints the total wall-clock time separately.

    ``/v1/chat/completions`` only guarantees an OpenAI-shaped ``usage`` block
    (prompt/completion/total token counts) -- confirmed by reading the actual
    response shape, no ``timings`` object the way llama.cpp's own native
    ``/completion`` endpoint gives you. So this prefers the server's own
    ``timings.predicted_per_second`` when a build happens to include it
    (decode-only, the more honest number), and otherwise falls back to
    ``completion_tokens / elapsed_s`` measured client-side around the POST --
    that fallback bundles in prompt/image prefill and network time too, so
    it'll read a little low versus "pure" decode speed, especially on
    vision requests with a big prefill. Either way it's real work being
    measured, not a fake spinner.
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

"""
LLM Connect
-----------
Nodes for talking to a llama.cpp server / llama-swap proxy
(OpenAI-compatible API); Unsloth Studio works too:

- Connectivity: server URL + model selection (dropdown auto-fetched from
  the server, see the /gibby_llm/get_models route in the pack __init__).
- Sampling Options: per-request sampler settings, each field gated by an
  enable_* toggle - only enabled fields are sent.
- Unsloth Load Options: per-load parameters (context size, speculative,
  KV-cache quant, extra flags) folded into the options dict under a
  "load" key; Generate applies them via Unsloth's /load API.
- Generate: single-shot text generation with auto-growing image/video
  reference slots (Autogrow); videos are sampled into still frames, the
  chat API takes stills only. Every run prints a tokens/sec line with the
  time it took. Takes a media pipe (its refs + keyframes are sent, labeled
  as such) and outputs one with the wired refs appended and the generated
  text as its prompt.

The HTTP client, reference collection and option filtering live in
llamacpp_client.py / llamacpp_refs.py / llamacpp_shared.py in this folder.
"""

from __future__ import annotations

from typing import Any

from pprint import pprint

from comfy_api.latest import io

from .llamacpp_client import (
    LlamaCppError,
    chat_completion,
    detect_backend,
    describe_speed,
    extract_reply,
    resolve_local_model,
    unload_model,
    unsloth_load,
    unsloth_unload,
)
from ..media_pipe import slot_order, pipe_add_images
from .llamacpp_refs import MAX_REF_SLOTS, _is_frame_batch, collect_reference_images, encode_image_batch
from .llamacpp_shared import filter_enabled_options

_CONNECTIVITY_TYPE = io.Custom("LLAMACPP_CONNECTIVITY")
_OPTIONS_TYPE = io.Custom("LLAMACPP_OPTIONS")

_CATEGORY = "gibby/llm connect"


def _normalize_stop(options: dict[str, Any] | None) -> dict[str, Any] | None:
    """llama.cpp's `stop` wants a list; the widget is a single STRING field.

    Wraps a non-empty stop string into a one-element list, drops the key
    entirely for an empty string (matches "no stop sequence configured").
    """
    if not options or "stop" not in options:
        return options
    options = dict(options)
    value = options["stop"]
    if isinstance(value, str):
        if value.strip():
            options["stop"] = [value]
        else:
            del options["stop"]
    return options


def _maybe_unload(url: str, model: str, connectivity: dict[str, Any] | None, debug: bool = False) -> None:
    """Best-effort implementation of Ollama's ``keep_alive == 0`` semantics.

    Only the "unload immediately" case has a real server-side call behind it:
    llama-swap's ``POST /api/models/unload/:model_id``, or Unsloth Studio's
    ``POST /api/inference/unload`` -- which must name the model the way the
    server loaded it, so it gets the same local-path-or-repo-id the load used.
    Positive/-1 values can't be implemented the same way (see the
    Connectivity node's keep_alive tooltip). Failures here are swallowed: a
    run that already produced a valid result should not fail because the
    best-effort cleanup afterward didn't work.
    """
    if connectivity is None:
        return
    keep_alive = connectivity.get("keep_alive")
    if keep_alive != 0:
        if debug and keep_alive is not None:
            print(
                f"[Gibby-LLM] keep_alive={keep_alive} has no effect beyond 0 -- "
                f"actual unload timing is the server's configured idle/ttl for {model!r}, "
                "not a per-request setting. See node file header."
            )
        return
    api_key = connectivity.get("api_key") or None
    try:
        if detect_backend(url, api_key) == "unsloth":
            local = resolve_local_model(url, model, api_key)
            target = local[1] if local is not None else model
            unsloth_unload(url, target, api_key=api_key)
        else:
            unload_model(url, model, api_key=api_key)
        if debug:
            print(f"[Gibby-LLM] keep_alive=0 -- unloaded {model!r} now.")
    except LlamaCppError as exc:
        if debug:
            print(f"[Gibby-LLM] keep_alive=0 unload request failed (non-fatal): {exc}")


def _clear_comfy_vram(debug: bool = False) -> None:
    """Free ComfyUI's GPU memory before handing VRAM to the LLM server --
    same recipe as Easy-Use's 'Clean VRAM Used' node: gc, CUDA sync,
    unload_all_models, soft_empty_cache."""
    import gc

    import torch
    import comfy.model_management as mm

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    mm.unload_all_models()
    mm.soft_empty_cache()
    if debug:
        print("[Gibby-LLM] Cleared ComfyUI VRAM.")


def _maybe_unsloth_load(url: str, model: str, api_key: str | None, options: dict[str, Any] | None, debug: bool = False) -> str | None:
    """Resolve the model's Unsloth identity, applying the options' ``load``
    block to it when there is one.

    Only meaningful for Unsloth Studio: it has a ``POST /api/inference/load``
    that takes context / speculative / KV-quant per load. llama.cpp and
    llama-swap bake those into the server's startup ``cmd``, so the block is
    ignored there (a note under debug, not an error) and None is returned so
    the caller keeps its own model id.

    A /v1/models id is a Hub repo id, so sending it to /load -- or to
    /v1/chat/completions, where Unsloth auto-loads an unknown model --
    downloads the model even when a local copy is on disk. If a local copy
    exists (LM Studio / hf cache), load it by its on-disk identity so
    Unsloth serves from disk; fall back to the repo id only when there is no
    local copy. That's why the identity is resolved even when there is no
    ``load`` key to apply -- a plain Connectivity -> Generate chain must not
    trigger a Hub download for a locally-present model.

    HF-cache models are the strict case: their chat identity is the repo id,
    so a chat-triggered auto-load re-resolves through the Hub and downloads.
    When load and chat identity differ, the load is therefore explicit and
    unconditional, not just when a ``load`` block is present. Plain-folder
    models chat by the same path they load under, so their chat auto-loads
    from disk and /load is only needed to apply the block.

    Returns the identity to chat with (local when a local copy exists, else
    the repo id). Unsloth keys model lookup and vision support to the exact
    load string, so a hub repo id can miss a locally-loaded model's mmproj
    (reports is_vision=false, then rejects image input).
    """
    if detect_backend(url, api_key) != "unsloth":
        if debug and (options or {}).get("load") is not None:
            print(
                "[Gibby-LLM] load requested, but the backend is "
                "llama.cpp/llama-swap -- those flags are fixed at server start, "
                "so set them in llama-swap's config.yaml cmd instead. Ignored."
            )
        return

    load = (options or {}).get("load")
    local = resolve_local_model(url, model, api_key)
    if local is None:
        load_target = chat_target = model
    else:
        load_target, chat_target = local

    if load is not None or load_target != chat_target:
        try:
            unsloth_load(url, load_target, load or {}, api_key=api_key)
        except LlamaCppError as exc:
            raise RuntimeError(f"Unsloth load failed: {exc}") from exc

        source = "local" if load_target != model else "hub"
        applied = ", ".join(f"{k}={v}" for k, v in (load or {}).items()) or "server defaults"
        print(f"[Gibby-LLM] unsloth: {model!r} loaded from {source} ({applied})")
    return chat_target


def _describe_keyframe_position(pos) -> str:
    """A human-readable keyframe position: frame index / percentage / last."""
    if isinstance(pos, float):
        return "{}%".format(round(pos * 100))
    if pos == 0:
        return "frame 0"
    if pos == -1:
        return "last frame"
    return "frame {}".format(pos) if pos > 0 else "{} frames from the end".format(-pos)


def _pipe_label(pipe: dict, pipe_refs: list, kf_imgs: list, total_images: int) -> str | None:
    """Tell the model which image numbers are references and which are keyframes."""
    if not pipe or total_images == 0:
        return None
    parts: list[str] = []
    n = 0
    if pipe_refs:
        n += len(pipe_refs)
        parts.append(f"images 1-{n} are references")
    if kf_imgs:
        start, n = n + 1, n + len(kf_imgs)
        parts.append(f"images {start}-{n} are keyframes in video order ({', '.join(_describe_keyframe_position(p) for p in pipe.get('keyframe_positions') or [])})")
    if n < total_images:
        parts.append(f"images {n + 1}-{total_images} are additional references")
    if not parts:
        return None
    return "Reference material: " + "; ".join(parts) + "."


def _content_parts(prompt: str, label: str | None, images_b64: list[str]) -> Any:
    """Build the `content` field for a chat message: plain string if there are
    no images (keeps the request minimal for the common case), or an OpenAI
    content-parts array with the text first, then each image as a data-URI
    `image_url` part."""
    if not images_b64:
        return prompt
    parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    if label:
        parts.append({"type": "text", "text": label})
    for b64 in images_b64:
        parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })
    return parts


class GibbyConnectivity(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GibbyConnectivity",
            display_name="Connectivity",
            category=_CATEGORY,
            search_aliases=["llama", "llama.cpp", "llama-swap", "unsloth", "server", "model", "connectivity"],
            description=(
                "Connection to a llama.cpp server, llama-swap proxy or Unsloth Studio. "
                "The model dropdown is fetched from the server's /v1/models (cached per URL). "
                "keep_alive: only 0 does anything real against llama.cpp/llama-swap -- it "
                "unloads the model immediately after each response via llama-swap's "
                "/api/models/unload endpoint; any other value is accepted for compatibility "
                "with the Ollama version's widget shape, but does not change server-side "
                "timing (that's the model's ttl in llama-swap's config.yaml)."
            ),
            inputs=[
                io.String.Input("url", default="http://127.0.0.1:8080", tooltip=(
                    "Base URL of the llama.cpp server or llama-swap proxy. "
                    "Default points at llama-swap's default port on this box.")),
                io.Combo.Input("model", options=[], default="", tooltip=(
                    "Model id, as reported by GET /v1/models on the server above. "
                    "Behind llama-swap this is one of the friendly names from its "
                    "config.yaml, not a raw file path. Use the Reconnect button if the "
                    "list is empty or stale.")),
                io.Int.Input("keep_alive", default=0, min=-1, max=120, step=1, tooltip=(
                    "Only 0 does anything real against llama.cpp/llama-swap: it unloads "
                    "the model immediately after each response. Any other value is "
                    "accepted for compatibility with the Ollama version's widget shape, "
                    "but does not change server-side timing.")),
                io.Combo.Input("keep_alive_unit", options=["minutes", "hours"], default="minutes", tooltip=(
                    "Kept for UI parity with the Ollama version. Currently unused - "
                    "llama.cpp/llama-swap has nothing that consumes a duration here.")),
                io.String.Input("api_key", default="", tooltip=(
                    "Optional Bearer token for servers that gate the OpenAI API behind "
                    "auth (e.g. Unsloth Studio). Sent as 'Authorization: Bearer <key>' "
                    "on every call. Leave empty for an open llama.cpp / llama-swap server.")),
            ],
            outputs=[
                _CONNECTIVITY_TYPE.Output("connectivity"),
            ],
        )

    @classmethod
    def execute(cls, url, model, keep_alive, keep_alive_unit, api_key=""):
        return io.NodeOutput({
            "url": url,
            "model": model,
            "keep_alive": keep_alive,
            "keep_alive_unit": keep_alive_unit,
            "api_key": api_key or "",
        })

    @classmethod
    def validate_inputs(cls, **kwargs) -> bool:
        # `model` is a combo whose options are fetched from the LLM server at
        # runtime (the /gibby_llm/get_models route), so the schema's options
        # list is empty and the built-in "value in list" check would reject
        # every server-reported model id. The server is the source of truth
        # for which models exist, so accept the value as-is.
        return True


class GibbySamplingOptions(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GibbySamplingOptions",
            display_name="Sampling Options",
            category=_CATEGORY,
            search_aliases=["llama", "sampling", "options", "temperature", "top_p", "seed", "reasoning"],
            description=(
                "Per-request sampling for llama.cpp inference -- temperature, "
                "top_k/top_p/min_p, mirostat, repeat_penalty, stop, n_predict, seed, "
                "reasoning_budget/reasoning_effort. Each field has an enable_* gate; "
                "only enabled fields are sent to the server. NOT here: context size, "
                "MTP/speculative, KV-cache quant, GPU layers, batch size -- those are "
                "fixed when the server process starts (see Unsloth Load Options and "
                "llama-swap's config.yaml cmd)."
            ),
            inputs=[
                io.Boolean.Input("enable_mirostat", default=False),
                io.Int.Input("mirostat", default=0, min=0, max=2, step=1, tooltip=(
                    "0 = disabled, 1 = Mirostat 1.0, 2 = Mirostat 2.0.")),
                io.Boolean.Input("enable_mirostat_eta", default=False),
                io.Float.Input("mirostat_eta", default=0.1, min=0, step=0.1, tooltip=(
                    "Mirostat learning rate.")),
                io.Boolean.Input("enable_mirostat_tau", default=False),
                io.Float.Input("mirostat_tau", default=5.0, min=0, step=0.1, tooltip=(
                    "Mirostat target entropy.")),
                io.Boolean.Input("enable_repeat_last_n", default=False),
                io.Int.Input("repeat_last_n", default=64, min=-1, max=2048, step=1, tooltip=(
                    "How far back to look to prevent repetition. 0 = disabled, -1 = context size.")),
                io.Boolean.Input("enable_repeat_penalty", default=False),
                io.Float.Input("repeat_penalty", default=1.1, min=0, max=2, step=0.05, tooltip=(
                    "Higher penalizes repetition more strongly. 1.0 = no penalty.")),
                io.Boolean.Input("enable_temperature", default=False),
                io.Float.Input("temperature", default=0.8, min=0, max=2, step=0.05, tooltip=(
                    "Higher = more creative/random.")),
                io.Boolean.Input("enable_seed", default=True),
                io.Int.Input("seed", default=0, min=-1, max=2 ** 31 - 1, step=1, tooltip=(
                    "Fixed seed for reproducible output. -1 = random each call.")),
                io.Boolean.Input("enable_stop", default=False),
                io.String.Input("stop", default="", tooltip=(
                    "Generation stops immediately if this string is produced.")),
                io.Boolean.Input("enable_n_predict", default=False),
                io.Int.Input("n_predict", default=-1, min=-1, max=32768, step=1, tooltip=(
                    "Max tokens to generate. -1 = unbounded.")),
                io.Boolean.Input("enable_top_k", default=False),
                io.Int.Input("top_k", default=40, min=0, max=200, step=1, tooltip=(
                    "Sample only from the top K tokens by probability.")),
                io.Boolean.Input("enable_top_p", default=False),
                io.Float.Input("top_p", default=0.9, min=0, max=1, step=0.05, tooltip=(
                    "Nucleus sampling threshold.")),
                io.Boolean.Input("enable_min_p", default=False),
                io.Float.Input("min_p", default=0.0, min=0, max=1, step=0.05, tooltip=(
                    "Minimum token probability, relative to the most likely token.")),
                io.Boolean.Input("enable_reasoning_budget", default=False),
                io.Int.Input("reasoning_budget", default=-1, min=-1, max=32768, step=128, tooltip=(
                    "Hard cap, in tokens, on the hidden thinking phase before the visible "
                    "answer starts. Enforced by llama.cpp itself, so it works for any model "
                    "that does a distinct thinking phase. -1 = unrestricted, 0 = skip thinking "
                    "entirely even with 'think' on. Rough feel: ~512 = quick/shallow, ~2048 = "
                    "moderate, ~8192+ = deep. Only matters if 'think' is also enabled on the "
                    "Generate node. Needs a reasonably recent llama.cpp server build.")),
                io.Boolean.Input("enable_reasoning_effort", default=False),
                io.String.Input("reasoning_effort", default="low", tooltip=(
                    "Free-text hint sent to the model's chat template about how much to "
                    "think. Type any value directly (e.g. low/medium/high).")),
                io.Boolean.Input("debug", default=False, tooltip=(
                    "Print the outgoing request and raw response to the ComfyUI console. "
                    "No effect on the API call itself.")),
                _OPTIONS_TYPE.Input("options", optional=True, tooltip=(
                    "Upstream options to merge with (e.g. an Unsloth Load Options node). "
                    "This node's sampler fields are layered on top, so the output carries "
                    "both -- chain it either direction with Load Options and feed the "
                    "result to Generate.")),
            ],
            outputs=[
                _OPTIONS_TYPE.Output("options"),
            ],
        )

    @classmethod
    def execute(cls, options=None, **kwargs):
        # Layer this node's sampler fields on top of any upstream options (e.g.
        # a Load Options node's "load" block), so the two can be chained in
        # either order and the result carries both.
        merged = dict(options or {})
        merged.update(kwargs)
        if merged.get("debug"):
            print("--- gibby sampling options dump\n")
            pprint(merged)
            print("---------------------------------------------------------")
        return io.NodeOutput(merged)


class GibbyLoadOptions(io.ComfyNode):
    _QUANT_LEVELS = ["default", "f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl"]

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GibbyLoadOptions",
            display_name="Unsloth Load Options",
            category=_CATEGORY,
            search_aliases=["llama", "unsloth", "load", "options", "context", "speculative", "mtp"],
            description=(
                "Holds load-time parameters (context size, speculative/MTP, KV-cache "
                "quant, extra llama-server flags) and folds them into the options dict "
                "under a 'load' key. No connection of its own -- Generate resolves the "
                "Connectivity and applies the load: Unsloth Studio via its /load API "
                "(local copy first, else the Hub repo); llama.cpp/llama-swap ignore it "
                "because those flags are fixed at server start (set them in "
                "llama-swap's config.yaml cmd, one entry per variant)."
            ),
            inputs=[
                io.Int.Input("context_size", default=40960, min=0, max=1048576, step=1024, tooltip=(
                    "Context tokens. 0 = let the server pick its default. Unsloth: "
                    "max_seq_length. Ignored on llama-swap (use -c in config).")),
                io.Combo.Input("speculative", options=["off", "auto", "mtp", "mtp+ngram", "ngram", "dspark", "dflash"],
                                default="auto", tooltip=(
                    "Speculative decoding mode. 'mtp' = multi-token-prediction drafting "
                    "(needs MTP weights in the GGUF). 'auto' = let Unsloth pick. 'off' = none. "
                    "Ignored on llama-swap (use --spec-type in config).")),
                io.Int.Input("spec_draft_n_max", default=0, min=0, max=16, step=1, tooltip=(
                    "Max draft tokens per speculative step (1-16). 0 = server default. "
                    "Ignored when speculative is 'off'.")),
                io.Combo.Input("kv_cache_quant", options=cls._QUANT_LEVELS, default="default", tooltip=(
                    "KV cache dtype for K and V ('q4_0' halves the cache's VRAM vs f16). "
                    "'default' = let the server decide. Ignored on llama-swap (use -ctk/-ctv in config).")),
                io.Combo.Input("draft_kv_cache_quant", options=cls._QUANT_LEVELS, default="default", tooltip=(
                    "KV cache dtype for the speculative draft context. 'default' = server "
                    "default (f16). Only meaningful with speculative on.")),
                io.String.Input("extra_args", default="", tooltip=(
                    "Raw llama-server flags, whitespace-separated, e.g. "
                    "'--jinja --flash-attn -ngl 999'. Unsloth appends them after its own "
                    "flags (last-wins). Ignored on llama-swap.")),
                io.Int.Input("n_batch", default=512, min=0, max=8192, step=64, tooltip=(
                    "Prompt processing batch size (llama-server -b). 0 = server default. "
                    "Ignored on llama-swap (use -b in config).")),
                io.Int.Input("n_ubatch", default=512, min=0, max=8192, step=64, tooltip=(
                    "Attention sub-batch size (llama-server -ub). 0 = server default. "
                    "Ignored on llama-swap (use -ub in config).")),
                _OPTIONS_TYPE.Input("options", optional=True, tooltip=(
                    "Upstream options to merge with (e.g. a Sampling Options node). The "
                    "result carries both the sampler fields and this node's load params, "
                    "ready to feed Generate.")),
            ],
            outputs=[
                _OPTIONS_TYPE.Output("options"),
            ],
        )

    @classmethod
    def execute(cls, context_size, speculative, spec_draft_n_max, kv_cache_quant,
                draft_kv_cache_quant, extra_args, n_batch=0, n_ubatch=0, options=None):
        # Only non-default values are sent to /load, so the server can use its
        # own defaults for the rest. The presence of the "load" key is what
        # tells Generate to attempt a load at all.
        load: dict[str, Any] = {}
        if context_size > 0:
            load["max_seq_length"] = context_size
        if speculative != "off":
            load["speculative_type"] = speculative
        if spec_draft_n_max >= 1:
            load["spec_draft_n_max"] = spec_draft_n_max
        if kv_cache_quant != "default":
            load["cache_type_kv"] = kv_cache_quant
        if draft_kv_cache_quant != "default":
            load["spec_draft_cache_type"] = draft_kv_cache_quant
        extra = [token for token in (extra_args or "").split() if token]
        if extra:
            load["llama_extra_args"] = extra
        if n_batch > 0:
            load["n_batch"] = n_batch
        if n_ubatch > 0:
            load["n_ubatch"] = n_ubatch

        merged = dict(options or {})
        merged["load"] = load
        return io.NodeOutput(merged)


class GibbyGenerate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        images_template = io.Autogrow.TemplateNames(
            input=io.Image.Input("image", optional=True),
            names=[f"images_{i}" for i in range(1, MAX_REF_SLOTS + 1)],
            min=0,
        )
        video_template = io.Autogrow.TemplateNames(
            input=io.MultiType.Input("video", [io.Video, io.Image], optional=True),
            names=[f"video_{i}" for i in range(1, MAX_REF_SLOTS + 1)],
            min=0,
        )
        return io.Schema(
            node_id="GibbyGenerate",
            display_name="Generate",
            category=_CATEGORY,
            search_aliases=["llama", "llama.cpp", "llama-swap", "unsloth", "generate", "chat", "llm", "text"],
            description=(
                "Single-shot text generation via a llama.cpp / llama-swap / Unsloth "
                "server, with auto-growing image and video reference inputs. Videos are "
                "sampled into evenly-spaced frames -- the chat API accepts stills only. "
                "Every run prints a tokens/sec line with the time it took to the "
                "ComfyUI console, so a run in progress isn't a black box."
            ),
            inputs=[
                _CONNECTIVITY_TYPE.Input("connectivity", tooltip=(
                    "Server + model from a Connectivity node.")),
                _OPTIONS_TYPE.Input("options", optional=True, tooltip=(
                    "Sampler config from a Sampling Options node. Covers temperature, "
                    "top_k/top_p/min_p, mirostat, repeat_penalty/repeat_last_n, stop, "
                    "n_predict, seed - plus reasoning_budget/reasoning_effort, which "
                    "control how much the model thinks (only relevant "
                    "when 'think' is also on). Nothing is sent to the server unless "
                    "that field's own enable_<name> checkbox is on in the Options node.")),
                io.Dict.Input("media_pipe", optional=True, tooltip=(
                    "Media pipe - its reference images and keyframes are sent to the "
                    "model before the wired images, labeled as "
                    "such; the output pipe carries the wired images appended "
                    "and the generated text overwrites its prompt.")),
                io.String.Input("system", multiline=True, default="You are an AI artist.", optional=True, tooltip=(
                    "System prompt - sets the role and general behavior of the model.")),
                io.String.Input("prompt", multiline=True, default="Describe the reference material.", optional=True, tooltip=(
                    "User prompt. With several references connected, refer to them by "
                    "position and say in the system prompt how many you are sending.")),
                io.Boolean.Input("think", default=True, optional=True, tooltip=(
                    "Ask the model to reason before answering, if it supports thinking. "
                    "Thinking text comes back on the 'thinking' output. Control how much "
                    "it thinks via reasoning_budget/reasoning_effort on a connected "
                    "Sampling Options node.")),
                io.Boolean.Input("clear_vram", default=True, optional=True, tooltip=(
                    "Call torch.cuda.empty_cache() after generation to free unused CUDA "
                    "memory. Useful when switching between LLM and image generation tasks.")),
                io.Combo.Input("format", options=["text", "json"], default="text", optional=True, tooltip=(
                    "'json' sets response_format to json_object (loose JSON mode, not "
                    "schema-validated).")),
                io.Int.Input("frames_per_video", default=4, min=1, max=MAX_REF_SLOTS * 2, step=1, optional=True, tooltip=(
                    "How many evenly-spaced frames to pull from each connected video. "
                    "llama.cpp's chat API has no video field - clips are sampled into "
                    "stills. First and last frame are always included. Most vision models "
                    "degrade past roughly 8 images total.")),
                io.Int.Input("max_image_size", default=0, min=0, max=4096, step=64, optional=True, tooltip=(
                    "Downscale every reference so its long edge is at most this many "
                    "pixels. 0 = send at original resolution. Useful when a big batch is "
                    "bloating the request.")),
                io.Autogrow.Input("images", template=images_template, optional=True, tooltip=(
                    "Reference image or image batch. Connect one and another empty "
                    "images slot appears. Batches are sent in full, unsampled. Whether "
                    "the connected model actually sees these depends on it having vision "
                    "support in llama.cpp.")),
                io.Autogrow.Input("video", template=video_template, optional=True, tooltip=(
                    "Reference video. Accepts a ComfyUI VIDEO or a raw IMAGE frame batch "
                    "(e.g. VideoHelperSuite Load Video). Connect one and another empty "
                    "video slot appears. Sampled down to 'frames_per_video' "
                    "evenly-spaced frames.")),
            ],
            outputs=[
                io.String.Output("result"),
                io.String.Output("thinking"),
                io.Dict.Output(display_name="media_pipe"),
            ],
        )

    @classmethod
    def execute(cls, connectivity=None, system="You are an AI artist.",
                prompt="Describe the reference material.", think=True,
                clear_vram=True, format="text", frames_per_video=4, max_image_size=0,
                options=None, media_pipe=None, images=None, video=None):
        if connectivity is None:
            raise RuntimeError("Gibby Generate: connect a Connectivity node")
        url = connectivity["url"]
        model = connectivity["model"]
        api_key = connectivity.get("api_key") or None

        debug_print = bool(options is not None and options.get("debug"))

        request_options = _normalize_stop(filter_enabled_options(options))
        # Fixed seed by default: when the connected options carry no
        # enable_seed toggle (Load Options only, or no options at all),
        # send seed=0 instead of letting the server randomize each run.
        if "enable_seed" not in (options or {}):
            request_options = {**(request_options or {}), "seed": 0}

        # Pipe refs and keyframes lead, the wired slots append to them.
        pipe = media_pipe or {}
        pipe_refs = [v for k, v in sorted((pipe.get("ref_images") or {}).items(),
                                           key=lambda kv: slot_order(kv[0])) if v is not None]
        kf_imgs = list(pipe.get("keyframes") or [])

        images_b64 = []
        for image in pipe_refs:
            images_b64.extend(encode_image_batch(image, max_image_size))
        for keyframe in kf_imgs:
            images_b64.extend(encode_image_batch(keyframe, max_image_size))
        slot_b64, _report = collect_reference_images(
            {**(images or {}), **(video or {})},
            frames_per_video=frames_per_video,
            max_image_size=max_image_size,
            debug=debug_print,
        )
        images_b64.extend(slot_b64 or [])

        user_message = {"role": "user", "content": _content_parts(
            prompt, _pipe_label(pipe, pipe_refs, kf_imgs, len(images_b64)), images_b64)}
        messages = [{"role": "system", "content": system}, user_message]

        if debug_print:
            print(f"""
--- gibby generate request:

url: {url}
model: {model}
system: {system}
prompt: {prompt}
images: {len(images_b64)}
think: {think}
options: {request_options}
format: {format}
---------------------------------------------------------
""")

        if clear_vram:
            _clear_comfy_vram(debug_print)

        # Unsloth keys model lookup to the identity it loaded the model
        # under, so a hub repo id can miss a locally-loaded model (and
        # trigger a download) -- resolve the local-first identity first.
        chat_model = _maybe_unsloth_load(url, model, api_key, options, debug_print) or model

        try:
            response, elapsed_s = chat_completion(
                url, chat_model, messages,
                options=request_options,
                think=think,
                json_mode=(format == "json"),
                api_key=api_key,
            )
        except LlamaCppError as exc:
            raise RuntimeError(f"Gibby Generate failed: {exc}") from exc

        if debug_print:
            print("\n--- gibby generate response:")
            pprint(response)
            print("---------------------------------------------------------")

        # Always printed, not just under debug -- this is the "yes, it just
        # did real work" confirmation, not a diagnostic dump.
        speed = describe_speed(response, elapsed_s)
        print(f"[Gibby-LLM] {model!r}: {speed} in {elapsed_s:.2f}s")

        result_text, thinking_text = extract_reply(response, think)

        _maybe_unload(url, model, connectivity, debug_print)

        # The pipe the wired references build: frame-batch videos join it as
        # refs too, native Video objects don't - H3 needs raw frames. The
        # generated text overwrites the prompt, so the pipe carries the LLM's
        # description of the refs forward (an empty reply keeps the old one).
        out_pipe = pipe_add_images(media_pipe, [images[name] for name in sorted(images or {}, key=slot_order)
                                                 if (images or {}).get(name) is not None])
        videos = {name: video[name] for name in sorted(video or {}, key=slot_order)
                  if (video or {}).get(name) is not None and _is_frame_batch(video[name])}
        if videos:
            out_pipe["ref_videos"] = {**(out_pipe.get("ref_videos") or {}), **videos}
        if result_text and result_text.strip():
            out_pipe["prompt"] = result_text

        return io.NodeOutput(result_text, thinking_text, out_pipe)

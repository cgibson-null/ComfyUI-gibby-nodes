"""
LLM Connect: nodes for an OpenAI-compatible server - llama.cpp, llama-swap
or Unsloth Studio. Connectivity, Sampling Options, Unsloth Load Options and
Generate. The HTTP client, reference collection and option filtering live in
the sibling llamacpp_*.py files.
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
    unsloth_preflight,
    unsloth_unload,
)
from ..pipes.media_pipe import iter_numbered_slots, pipe_add_images
from ..context import clear_vram as _clear_vram
from .llamacpp_refs import MAX_REF_SLOTS, _is_frame_batch, collect_reference_images, encode_image_batch
from .llamacpp_shared import filter_enabled_options

_CONNECTIVITY_TYPE = io.Custom("LLAMACPP_CONNECTIVITY")
_OPTIONS_TYPE = io.Custom("LLAMACPP_OPTIONS")

_CATEGORY = "gibby/llm connect"

# keep_context history: (node id, url, model) -> earlier turns as OpenAI message
# dicts (text only; images are always re-sent from the inputs). Both bounds are
# load-bearing: an untrimmed conversation outgrows the server's context window,
# and a deleted node would otherwise leave its key behind.
_MAX_CONVERSATIONS = 32
_MAX_TURNS = 40
_CONVERSATIONS: dict[tuple[str, str, str], list[dict[str, Any]]] = {}


def _stored_conversation(conv_key):
    """The stored turns for this node + server + model, marked the most recent."""
    history = _CONVERSATIONS.pop(conv_key, None)
    if history is None:
        return []
    _CONVERSATIONS[conv_key] = history
    return list(history)


def _store_conversation(conv_key, turns):
    _CONVERSATIONS[conv_key] = turns[-_MAX_TURNS:]
    while len(_CONVERSATIONS) > _MAX_CONVERSATIONS:
        _CONVERSATIONS.pop(next(iter(_CONVERSATIONS)))


def _normalize_stop(options: dict[str, Any] | None) -> dict[str, Any] | None:
    """The widget is a single STRING; llama.cpp's `stop` wants a list, and no
    stop sequence means no key at all."""
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
    """Ollama's ``keep_alive == 0``: unload now, via llama-swap's
    /api/models/unload or Unsloth's /api/inference/unload (which has to name the
    model the way the server loaded it). Other values have no server-side call.
    Failures are swallowed - a good result should not fail over cleanup."""
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


#: The load params the two pre-load probes price, in the names they take there.
#: Anything else in the load block (auto_derived, gpu_layers) is an answer from the
#: probes, not something to ask them about.
_PREFLIGHT_KEYS = ("max_seq_length", "n_parallel", "cache_type_kv",
                   "spec_draft_cache_type", "n_batch", "n_ubatch", "spec_draft_n_max",
                   "speculative_type", "llama_extra_args")


def _price_load(url: str, target: str, load: dict[str, Any], api_key: str | None,
                debug: bool = False) -> dict[str, Any]:
    """Ask Unsloth to price this load (``validate``, then ``estimate-memory``) and
    merge its fitted context and gpu_layers back in.

    /load reloads whenever the settings differ from what is loaded, so sending the
    user's raw context instead of the panel's priced command reloads a model that
    is already up - the "loads for 30 seconds, fails, works on the second run"
    report. A probe that errors raises: the load about to be spent on it has no
    price. An estimate that cannot size the model leaves the user's command alone.
    """
    # One Generate is one reply, so one serving slot: Studio's server-wide default
    # priced 10.3 GiB of KV cache for a 27B at 20k where one slot needs 5.2.
    priced = dict(load)
    priced.setdefault("n_parallel", 1)

    probe = {key: value for key, value in priced.items() if key in _PREFLIGHT_KEYS}
    if "max_seq_length" in probe:
        # estimate-memory prices the context under its own --ctx-size name.
        probe["n_ctx"] = probe["max_seq_length"]

    check = unsloth_preflight(url, "validate", target, probe, api_key=api_key)
    if not check.get("valid", True):
        raise RuntimeError(f"Unsloth load refused: {target!r}: {check.get('message') or 'not a loadable model'}")

    estimate = unsloth_preflight(url, "estimate-memory", target, probe, api_key=api_key)
    if not estimate.get("available", False):
        return priced

    if debug:
        gpu = (estimate.get("gpu_bytes") or 0) / 2**30
        total = (estimate.get("total_bytes") or 0) / 2**30
        print(
            f"[Gibby-LLM] unsloth: estimate says {gpu:.1f} GiB GPU / {total:.1f} GiB total at "
            f"{estimate.get('n_ctx')} ctx, {estimate.get('n_parallel')} slot(s), "
            f"{estimate.get('gpu_layers')} gpu_layers"
        )

    fitted = estimate.get("context_fitted")
    requested = priced.get("max_seq_length", 0)
    if isinstance(fitted, int) and fitted > 0 and requested and fitted < requested:
        priced["max_seq_length"] = fitted
        priced["max_seq_length_auto_derived"] = True
        if debug:
            print(f"[Gibby-LLM] unsloth: context {requested} -> {fitted} (Unsloth's own estimate for this load)")
    layers = estimate.get("gpu_layers")
    if isinstance(layers, int) and layers > 0:
        priced["gpu_layers"] = layers
    return priced


def _maybe_unsloth_load(url: str, model: str, api_key: str | None, options: dict[str, Any] | None, debug: bool = False) -> str | None:
    """Resolve the model's Unsloth identity and apply the options' ``load`` block
    to it. Returns the identity to chat with, or None to keep the caller's id.

    Unsloth keys model lookup and vision support to the exact load string, and a
    /v1/models repo id makes it download from the Hub even when a local copy is on
    disk - so a locally-present model must be loaded and chatted by its on-disk
    identity. Where the two identities differ (HF-cache models) the load has to be
    explicit, or the chat auto-load re-downloads.

    llama.cpp and llama-swap fix these flags at server start, so the block is
    ignored there and None is returned.
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
            load = _price_load(url, load_target, load or {}, api_key, debug)
            unsloth_load(url, load_target, load, api_key=api_key)
        except LlamaCppError as exc:
            raise RuntimeError(f"Unsloth load failed: {exc}") from exc

        source = "local" if load_target != model else "hub"
        applied = ", ".join(f"{k}={v}" for k, v in load.items()) or "server defaults"
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


def _content_parts(prompt: str, label: str | None, image_urls: list[str]) -> Any:
    """The chat message's content: a plain string when there are no images, else
    an OpenAI content-parts array with the text first."""
    if not image_urls:
        return prompt
    parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    if label:
        parts.append({"type": "text", "text": label})
    for url in image_urls:
        parts.append({
            "type": "image_url",
            "image_url": {"url": url},
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
                "Connection to a llama.cpp server, llama-swap proxy or Unsloth Studio; the model list is read "
                "from the server's /v1/models."
            ),
            inputs=[
                io.String.Input("url", default="http://127.0.0.1:8080", tooltip=(
                    "Base URL of the server. Default is llama-swap's default port on this box.")),
                io.Combo.Input("model", options=[], default="", tooltip=(
                    "Model id as the server reports it. Behind llama-swap that is the friendly "
                    "name from config.yaml, not a file path. Reconnect if the list is stale.")),
                io.Int.Input("keep_alive", default=0, min=-1, max=120, step=1, tooltip=(
                    "0 unloads the model right after each response. Any other value is accepted "
                    "for Ollama widget parity but does nothing - timing is llama-swap's own ttl.")),
                io.Combo.Input("keep_alive_unit", options=["minutes", "hours"], default="minutes", tooltip=(
                    "Unused - kept for UI parity with the Ollama version.")),
                io.String.Input("api_key", default="", tooltip=(
                    "Bearer token for servers that gate the API behind auth (Unsloth Studio). "
                    "Empty for an open llama.cpp / llama-swap server.")),
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
        # The combo's options come from the live server (/gibby_llm/get_models), so
        # the schema list is empty and the server is the source of truth.
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
                "Per-request sampling for llama.cpp inference. Each field has an enable_* gate; only enabled "
                "fields are sent. Load-time settings are not here - see Unsloth Load Options."
            ),
            inputs=[
                io.Boolean.Input("enable_mirostat", default=False),
                io.Int.Input("mirostat", default=0, min=0, max=2, step=1, tooltip=(
                    "0 = off, 1 = Mirostat 1.0, 2 = Mirostat 2.0")),
                io.Boolean.Input("enable_mirostat_eta", default=False),
                io.Float.Input("mirostat_eta", default=0.1, min=0, step=0.1),
                io.Boolean.Input("enable_mirostat_tau", default=False),
                io.Float.Input("mirostat_tau", default=5.0, min=0, step=0.1),
                io.Boolean.Input("enable_repeat_last_n", default=False),
                io.Int.Input("repeat_last_n", default=64, min=-1, max=2048, step=1, tooltip=(
                    "How far back to look. 0 = disabled, -1 = context size")),
                io.Boolean.Input("enable_repeat_penalty", default=False),
                io.Float.Input("repeat_penalty", default=1.1, min=0, max=2, step=0.05, tooltip=(
                    "1.0 = no penalty")),
                io.Boolean.Input("enable_temperature", default=False),
                io.Float.Input("temperature", default=0.8, min=0, max=2, step=0.05),
                io.Boolean.Input("enable_seed", default=True),
                io.Int.Input("seed", default=0, min=-1, max=2 ** 31 - 1, step=1, tooltip=(
                    "-1 = random each call")),
                io.Boolean.Input("enable_stop", default=False),
                io.String.Input("stop", default=""),
                io.Boolean.Input("enable_n_predict", default=False),
                io.Int.Input("n_predict", default=-1, min=-1, max=32768, step=1, tooltip=(
                    "-1 = unbounded")),
                io.Boolean.Input("enable_top_k", default=False),
                io.Int.Input("top_k", default=40, min=0, max=200, step=1),
                io.Boolean.Input("enable_top_p", default=False),
                io.Float.Input("top_p", default=0.9, min=0, max=1, step=0.05),
                io.Boolean.Input("enable_min_p", default=False),
                io.Float.Input("min_p", default=0.0, min=0, max=1, step=0.05, tooltip=(
                    "Minimum token probability, relative to the most likely token.")),
                io.Boolean.Input("enable_reasoning_budget", default=False),
                io.Int.Input("reasoning_budget", default=-1, min=-1, max=32768, step=128, tooltip=(
                    "Token cap on the hidden thinking phase, enforced by llama.cpp. "
                    "-1 = unrestricted, 0 = no thinking. Needs 'think' on.")),
                io.Boolean.Input("enable_reasoning_effort", default=False),
                io.String.Input("reasoning_effort", default="low", tooltip=(
                    "Free-text hint to the chat template about how much to think, e.g. "
                    "low/medium/high.")),
                io.Boolean.Input("debug", default=False),
                _OPTIONS_TYPE.Input("options", optional=True, tooltip=(
                    "Upstream options to merge with (e.g. Unsloth Load Options). This node's "
                    "fields win, so chain it either direction.")),
            ],
            outputs=[
                _OPTIONS_TYPE.Output("options"),
            ],
        )

    @classmethod
    def execute(cls, options=None, **kwargs):
        # Layer this node's fields over any upstream options so the two chain either way.
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
                "Load-time parameters folded into the options dict under a 'load' key. "
                "Unsloth Studio applies them through its /load API; llama.cpp and llama-swap "
                "fix them at server start and ignore them."
            ),
            inputs=[
                io.Int.Input("context_size", default=0, min=0, max=1048576, step=1024, tooltip=(
                    "Context tokens; 0 = the server's default, which Unsloth fits to free "
                    "VRAM. Ignored on llama-swap (use -c in config).")),
                io.Combo.Input("speculative", options=["off", "auto", "mtp", "mtp+ngram", "ngram", "dspark", "dflash"],
                                default="auto", tooltip=(
                    "Speculative decoding mode; 'mtp' needs MTP weights in the GGUF, 'auto' "
                    "lets Unsloth pick. Ignored on llama-swap (use --spec-type in config).")),
                io.Int.Input("spec_draft_n_max", default=0, min=0, max=16, step=1, tooltip=(
                    "Max draft tokens per speculative step; 0 = server default. "
                    "Ignored when speculative is 'off'.")),
                io.Combo.Input("kv_cache_quant", options=cls._QUANT_LEVELS, default="default", tooltip=(
                    "KV cache dtype for K and V ('q4_0' halves it vs f16). "
                    "Ignored on llama-swap (use -ctk/-ctv in config).")),
                io.Combo.Input("draft_kv_cache_quant", options=cls._QUANT_LEVELS, default="default", tooltip=(
                    "KV cache dtype for the speculative draft context; needs speculative on.")),
                io.String.Input("extra_args", default="", tooltip=(
                    "Raw llama-server flags, whitespace-separated, e.g. "
                    "'--jinja --flash-attn -ngl 999'. Unsloth appends them (last-wins).")),
                io.Int.Input("n_batch", default=0, min=0, max=8192, step=64, tooltip=(
                    "Prompt processing batch size (llama-server -b); 0 = llama.cpp's 2048. "
                    "Ignored on llama-swap (use -b in config).")),
                io.Int.Input("n_ubatch", default=0, min=0, max=8192, step=64, tooltip=(
                    "Attention sub-batch size (llama-server -ub); 0 = the server default, "
                    "which Unsloth raises above the 512 llama.cpp aborts on. "
                    "Ignored on llama-swap (use -ub in config).")),
                _OPTIONS_TYPE.Input("options", optional=True, tooltip=(
                    "Upstream options to merge with (e.g. Sampling Options).")),
            ],
            outputs=[
                _OPTIONS_TYPE.Output("options"),
            ],
        )

    @classmethod
    def execute(cls, context_size, speculative, spec_draft_n_max, kv_cache_quant,
                draft_kv_cache_quant, extra_args, n_batch=0, n_ubatch=0, options=None):
        # Only non-default values go to /load so the server keeps its own defaults;
        # every default here is the "ask the server" value. The presence of the
        # "load" key is what tells Generate to attempt a load at all.
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
                "Single-shot text generation with auto-growing image and video references. Videos are sampled "
                "into evenly-spaced frames - the chat API takes stills only."
            ),
            inputs=[
                _CONNECTIVITY_TYPE.Input("connectivity", tooltip=(
                    "Server + model from a Connectivity node.")),
                _OPTIONS_TYPE.Input("options", optional=True, tooltip=(
                    "Sampler config from a Sampling Options node; a field is only sent when "
                    "its own enable_<name> checkbox is on there.")),
                io.Dict.Input("media_pipe", optional=True, tooltip=(
                    "Media pipe: its reference images and keyframes are sent before the wired "
                    "images, labeled as such. The output pipe has the wired images appended "
                    "and the generated text as its prompt.")),
                io.String.Input("system", multiline=True, default="You are an AI artist.", optional=True),
                io.String.Input("prompt", multiline=True, default="Describe the reference material.", optional=True, tooltip=(
                    "User prompt. With several references, refer to them by position and say "
                    "in the system prompt how many you are sending.")),
                io.Boolean.Input("think", default=True, optional=True, tooltip=(
                    "Ask the model to reason first, if it supports it; the thinking text "
                    "comes back on the 'thinking' output.")),
                io.Boolean.Input("keep_context", default=False, optional=True, tooltip=(
                    "Carry the previous turns (text only) into the next run so the model "
                    "continues the conversation. Kept per node and per server + model; "
                    "switching it off clears the history.")),
                io.Boolean.Input("clear_vram", default=True, optional=True, tooltip=(
                    "Free unused CUDA memory before the request, e.g. when alternating "
                    "LLM and image runs.")),
                io.Combo.Input("format", options=["text", "json"], default="text", optional=True, tooltip=(
                    "'json' sets response_format to json_object - loose JSON mode, "
                    "not schema-validated.")),
                io.Int.Input("frames_per_video", default=4, min=1, max=MAX_REF_SLOTS * 2, step=1, optional=True, tooltip=(
                    "Evenly-spaced frames pulled from each connected video (first and last "
                    "always included). Most vision models degrade past roughly 8 images total.")),
                io.Int.Input("max_image_size", default=0, min=0, max=4096, step=64, optional=True, tooltip=(
                    "Downscale every reference so its long edge is at most this many pixels; "
                    "0 = original resolution.")),
                io.Autogrow.Input("images", template=images_template, optional=True, tooltip=(
                    "Reference image or image batch, sent in full and unsampled. A new slot "
                    "appears when you connect one.")),
                io.Autogrow.Input("video", template=video_template, optional=True, tooltip=(
                    "Reference video: a ComfyUI VIDEO or a raw IMAGE frame batch (e.g. VHS "
                    "Load Video), sampled down to 'frames_per_video' frames.")),
            ],
            outputs=[
                io.String.Output("result"),
                io.String.Output("thinking"),
                io.Dict.Output(display_name="media_pipe"),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, connectivity=None, system="You are an AI artist.",
                prompt="Describe the reference material.", think=True,
                keep_context=False, clear_vram=True, format="text", frames_per_video=4,
                max_image_size=0, options=None, media_pipe=None, images=None, video=None):
        if connectivity is None:
            raise RuntimeError("Gibby Generate: connect a Connectivity node")
        url = connectivity["url"]
        model = connectivity["model"]
        api_key = connectivity.get("api_key") or None

        debug_print = bool(options is not None and options.get("debug"))

        request_options = _normalize_stop(filter_enabled_options(options))
        # No enable_seed toggle in the options (Load Options only, or none at all):
        # send seed=0 rather than let the server randomize each run.
        if "enable_seed" not in (options or {}):
            request_options = {**(request_options or {}), "seed": 0}

        # Pipe refs and keyframes lead, the wired slots append to them.
        pipe = media_pipe or {}
        pipe_refs = [value for _index, _name, value in iter_numbered_slots(pipe.get("ref_images") or {}, "ref_image")]
        kf_imgs = list(pipe.get("keyframes") or [])

        image_urls: list[str] = []
        for image in pipe_refs:
            image_urls.extend(encode_image_batch(image, max_image_size))
        for keyframe in kf_imgs:
            image_urls.extend(encode_image_batch(keyframe, max_image_size))
        slot_urls, _report = collect_reference_images(
            {**(images or {}), **(video or {})},
            frames_per_video=frames_per_video,
            max_image_size=max_image_size,
            debug=debug_print,
        )
        image_urls.extend(slot_urls or [])

        user_message = {"role": "user", "content": _content_parts(
            prompt, _pipe_label(pipe, pipe_refs, kf_imgs, len(image_urls)), image_urls)}

        # keep_context: prepend this node's earlier turns; off drops the history.
        node_id = getattr(getattr(cls, "hidden", None), "unique_id", None) or "generate"
        conv_key = (node_id, url, model)
        if keep_context:
            history = _stored_conversation(conv_key)
        else:
            history = []
            _CONVERSATIONS.pop(conv_key, None)
        messages = [{"role": "system", "content": system}, *history, user_message]

        if debug_print:
            print(f"""
--- gibby generate request:

url: {url}
model: {model}
system: {system}
prompt: {prompt}
images: {len(image_urls)}
think: {think}
previous turns: {len(history)}
options: {request_options}
format: {format}
---------------------------------------------------------
""")

        if clear_vram:
            _clear_vram("before the LLM server", verbose=debug_print)

        # Unsloth keys model lookup to the identity it loaded under, so a hub repo
        # id can miss a locally-loaded model (and re-download it).
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

        # Always printed: the "it did real work" confirmation, not a diagnostic.
        speed = describe_speed(response, elapsed_s)
        print(f"[Gibby-LLM] {model!r}: {speed} in {elapsed_s:.2f}s")

        result_text, thinking_text = extract_reply(response, think)

        # Store the turn only after a success, so a failure never corrupts the
        # conversation. The thinking text is not part of the dialogue.
        if keep_context:
            turn: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
            if result_text:
                turn.append({"role": "assistant", "content": result_text})
            _store_conversation(conv_key, [*history, *turn])

        _maybe_unload(url, model, connectivity, debug_print)

        # Frame-batch videos join the pipe as refs too, native Video objects don't -
        # H3 needs raw frames. The generated text becomes the pipe's prompt.
        out_pipe = pipe_add_images(media_pipe, [value for _index, _name, value in iter_numbered_slots(images or {}, "images")])
        videos = {name: value for _index, name, value in iter_numbered_slots(video or {}, "video")
                  if _is_frame_batch(value)}
        if videos:
            out_pipe["ref_videos"] = {**(out_pipe.get("ref_videos") or {}), **videos}
        if result_text and result_text.strip():
            out_pipe["prompt"] = result_text

        return io.NodeOutput(result_text, thinking_text, out_pipe)

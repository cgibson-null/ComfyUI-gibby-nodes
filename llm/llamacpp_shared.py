"""Small shared pieces: sampler-option filtering.

Ported from ``ollama_shared.py`` in the Ollama-Enhanced pack.
"""

from __future__ import annotations

from typing import Any


# Sampler knobs exposed by GibbySamplingOptions, gated by an
# `enable_<name>` flag each -- same UX as the Ollama pack's Options node.
#
# Dropped versus the Ollama version, and why:
#   num_ctx  -- llama.cpp fixes context size at server start (`-c` flag /
#               llama-swap's per-model `cmd`). There is no per-request
#               equivalent; sending one would just be silently ignored.
#   tfs_z    -- tail-free sampling was removed from llama.cpp itself. Ollama
#               still lists it, but forwarding it to a llama.cpp server would
#               either be ignored or (on strict builds) rejected as unknown.
#
# Renamed versus the Ollama version:
#   num_predict -> n_predict. llama.cpp's server has accepted `max_tokens` for
#               OpenAI compatibility since day one, but its own extension
#               field for "max new tokens, -1 = unbounded" has always been
#               `n_predict`. -1 behaves the same as Ollama's -1.
#
# New, with no Ollama equivalent at all -- Ollama's API has no concept of a
# reasoning phase separate from the answer, so there was nothing to port:
#   reasoning_budget -- llama.cpp server-enforced token cap on the hidden
#               thinking phase (see --reasoning-budget upstream). Works for
#               any model that does a distinct thinking phase, regardless of
#               whether that model was specifically trained around the
#               concept of "effort levels." Only meaningful when the
#               Generate node's own `think` toggle is also on.
#   reasoning_effort -- passed through to the chat template as a hint
#               (low/medium/high). Only understood by models actually
#               trained for it (gpt-oss, StepFun, as of this writing) --
#               silently ignored by everything else. reasoning_budget is the
#               one that actually works universally; this one is along for
#               the ride for when it's supported.
_OPTION_ENABLERS = (
    "enable_mirostat",
    "enable_mirostat_eta",
    "enable_mirostat_tau",
    "enable_repeat_last_n",
    "enable_repeat_penalty",
    "enable_temperature",
    "enable_seed",
    "enable_stop",
    "enable_n_predict",
    "enable_top_k",
    "enable_top_p",
    "enable_min_p",
    "enable_reasoning_budget",
    "enable_reasoning_effort",
)


def filter_enabled_options(options: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return only the sampler options whose matching ``enable_*`` flag is True."""
    if not options:
        return None
    out: dict[str, Any] = {}
    for enabler in _OPTION_ENABLERS:
        if options.get(enabler, False):
            key = enabler.replace("enable_", "")
            out[key] = options[key]
    return out or None

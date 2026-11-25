"""Sampler-option filtering shared by the LLM Connect nodes."""

from __future__ import annotations

from typing import Any


# Sampler knobs exposed by GibbySamplingOptions, each gated by an enable_<name>
# flag. Deliberately absent: num_ctx (llama.cpp fixes the context at server
# start, so a per-request value would be ignored) and tfs_z (removed from
# llama.cpp itself). reasoning_effort is only a chat-template hint that most
# models ignore; reasoning_budget is the one llama.cpp actually enforces.
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

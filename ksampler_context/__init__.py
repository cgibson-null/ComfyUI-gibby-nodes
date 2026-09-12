"""
KSampler (Context) node
-----------------------
A sampler that pulls most parameters from a CONTEXT object, with optional
overrides for model, latent, image, mask, sampler, and sigmas. Designed to work
seamlessly with the Context node's on-demand generation features.

Key behaviors:
- Uses context values unless overridden by inputs or widgets
- Encodes images to latents automatically (with mask support)
- With an image present, samples with refiner steps instead of base steps
- Handles start_step/end_step as offsets from total steps when negative
- Automatically uses Flux2Scheduler for Flux2 models
- Updates context after sampling (removes latent/mask, stores decoded image)
- Options are applied in list order: crop-inpaint and iterative upscale run
  before evaluate, tiled VAE settings apply to encode/decode, clear VRAM
  options free VRAM at the start/end of the execution and after sampling /
  encode-decode (like Clean VRAM Used), lora travel retunes the lora
  strengths per step, prompt travel swaps the positive/negative prompts per
  step (Forge-style [before:after:step] groups, encoded with the context clip);
  the options output returns the (updated) options for feeding back in
"""

import gc
import os
import time

import folder_paths
import torch
import comfy.samplers
import comfy.sample
import comfy.model_management
import comfy.float
import comfy.lora
import comfy.lora_convert
import comfy.model_patcher
import latent_preview
import comfy.model_base
from nodes import VAEDecode, CLIPTextEncode, ConditioningZeroOut
from comfy_api.latest import io

try:
    from comfy_extras.nodes_audio import vae_decode_audio
except ImportError:
    vae_decode_audio = None

try:
    from comfy_extras.nodes_flux import Flux2Scheduler, get_schedule
except ImportError:
    Flux2Scheduler = None
    get_schedule = None

import comfy.utils as _comfy_utils
from nodes import VAEEncode, VAEDecode, VAEDecodeTiled, VAEEncodeTiled, SetLatentNoiseMask
from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
from comfy_extras.nodes_post_processing import ColorTransfer
from ..context import _CONTEXT_TYPE, GibbyContext
from ..crop_inpaint_options import _KSAMPLER_OPTIONS_TYPE


def _find_option(options_list, opt_type):
    if options_list is None:
        return None
    for o in options_list:
        if o.get("type") == opt_type:
            return o
    return None


def _clear_vram(event, verbose=False):
    """Free VRAM like Easy-Use's Clean VRAM Used: gc, CUDA sync, unload all models, empty the cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()
    if verbose:
        print(f"Gibby KSampler (Context): Clear VRAM: {event}")


def _tiled_settings(options_list):
    """Tiled VAE settings from the options list: (tiled, tile_size, overlap, temporal_size, temporal_overlap)."""
    opts = _find_option(options_list, "tiled_vae")
    if opts is None:
        return False, 512, 64, 64, 8
    return True, opts.get("tile_size", 512), opts.get("overlap", 64), opts.get("temporal_size", 64), opts.get("temporal_overlap", 8)


def _image_dims(image):
    """(w, h, length) from a channels-last image tensor; length is None for stills."""
    if image is None:
        return None, None, None
    if image.dim() == 4:
        return image.shape[2], image.shape[1], None
    if image.dim() == 5:
        return image.shape[3], image.shape[2], image.shape[1]
    return None, None, None


def _latent_downscale(vae, channels):
    """(w, h) spatial downscale ratio of a VAE, guessed from latent channels when no VAE is given."""
    r = getattr(vae, "downscale_ratio", None) if vae is not None else None
    if r is not None:
        if isinstance(r, (tuple, list)):
            return int(r[2]), int(r[1])
        return int(r), int(r)
    d = 16 if channels == 128 else 8
    return d, d


def _latent_dims(latent, vae=None):
    """(w, h, length) in pixels from a latent dict's samples tensor; length is None for stills."""
    if latent is None:
        return None, None, None
    s = latent["samples"]
    if s.dim() == 4:
        lw, lh, length = s.shape[3], s.shape[2], None
    elif s.dim() == 5:
        lw, lh, length = s.shape[4], s.shape[3], s.shape[2]
    else:
        return None, None, None
    dw, dh = _latent_downscale(vae, s.shape[1])
    return lw * dw, lh * dh, length


def _resolve_step_range(steps_value, start_step, end_step):
    """Actual (start, end) step indices per the start/end step rules, clamped to [0, steps_value]."""
    def resolve(v):
        val = round(steps_value * v) if abs(v) < 1 else int(round(v))
        actual = steps_value + val if val < 0 else val
        return max(0, min(steps_value, int(actual)))
    return resolve(start_step), resolve(end_step)


def _load_lora_state(lora_name):
    """The lora's state dict from disk, via the LoraLoader path; None when the file is missing."""
    path = folder_paths.get_full_path("loras", lora_name)
    if not path or not os.path.isfile(path):
        return None
    return _comfy_utils.load_torch_file(path, safe_load=True)


def _load_lora_into_model(patcher, lora, strength):
    """Apply a lora's model patches at the given strength. Silent about the lora's clip
    keys (loaded separately) - against the model alone they are expected to not match."""
    key_map = comfy.lora.model_lora_keys_unet(patcher.model, {})
    lora = comfy.lora_convert.convert_lora(lora)
    loaded = comfy.lora.load_lora(lora, key_map, log_missing=False)
    new_patcher = patcher.clone()
    new_patcher.add_patches(loaded, strength)
    return new_patcher


def _load_lora_into_clip(clip, lora, strength):
    """Apply a lora's clip patches at the given strength. Silent about the lora's model
    keys (loaded separately) - against the clip alone they are expected to not match."""
    key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, {})
    lora = comfy.lora_convert.convert_lora(lora)
    loaded = comfy.lora.load_lora(lora, key_map, log_missing=False)
    new_clip = clip.clone()
    new_clip.add_patches(loaded, strength)
    return new_clip


def _travel_step_strength(lora, step):
    """The lora's model strength at a step index: its stack strength scaled by the linear
    ramp from start_str (at start) to end_str (at the last active step, end-1); 0.0
    outside [start, end)."""
    if not (lora["start"] <= step < lora["end"]):
        return 0.0
    last = lora["end"] - 1
    if last == lora["start"]:
        factor = lora["start_str"]
    else:
        factor = lora["start_str"] + (lora["end_str"] - lora["start_str"]) * (step - lora["start"]) / (last - lora["start"])
    return lora["strength"] * factor


def _prepare_lora_travels(model_obj, travel_opts, steps_value):
    """Apply every travel's lora stack to a clone of the model at zero strength and
    record each lora's patch entries, so the per-step callback can retune them.
    Returns None when there is nothing to travel."""
    if not travel_opts or not steps_value or steps_value <= 0:
        return None

    patcher = model_obj.clone()
    loras = []
    for opt in travel_opts:
        start, end = _resolve_step_range(steps_value, opt.get("start_step", 0.0), opt.get("end_step", 100.0))
        if end <= start:
            continue
        start_str = opt.get("start_str", 0.6)
        end_str = opt.get("end_str", 1.0)
        for item in opt.get("lora_stack") or []:
            if not item or len(item) < 3 or item[0] in (None, "None"):
                continue
            name, strength = item[0], item[1]
            if not strength:
                continue
            lora = _load_lora_state(name)
            if lora is None:
                continue
            before = {k: len(v) for k, v in patcher.patches.items()}
            patcher = _load_lora_into_model(patcher, lora, 0.0)
            entries = []
            for k, v in patcher.patches.items():
                extra = len(v) - before.get(k, 0)
                if extra > 0:
                    entries.extend((k, i) for i in range(len(v) - extra, len(v)))
            if entries:
                loras.append({
                    "name": name,
                    "strength": strength,
                    "start": start,
                    "end": end,
                    "start_str": start_str,
                    "end_str": end_str,
                    "entries": entries,
                    "current": 0.0,
                })
    if not loras:
        return None
    return {
        "patcher": patcher,
        "loras": loras,
        "steps": steps_value,
        "verbose": any(opt.get("verbose", False) for opt in travel_opts),
    }


def _retune_lora_key(patcher, key):
    """Make the retuned strength take effect for one key. Low-VRAM weights re-patch
    live from the patches dict on every forward, so their cached prepared patches
    are invalidated instead (the aimdo prefetch path commits them, which would
    otherwise keep the stale strength). Fully-loaded weights are recomputed from
    the original and written back - the loaded-weight equivalent of
    patch_weight_to_device. A key the model has not loaded yet needs neither: the
    strength is already in its patch tuple and the upcoming load applies it."""
    op = _comfy_utils.get_attr(patcher.model, key.rsplit(".", 1)[0])
    lowvram = []
    for func_list in (getattr(op, "weight_function", None), getattr(op, "bias_function", None)):
        for f in func_list or []:
            if getattr(f, "is_lowvram_patch", False):
                lowvram.append(f)
    if lowvram:
        for f in lowvram:
            f.clear_prepared()
        return
    if key not in patcher.backup:
        return
    weight, set_func, convert_func = comfy.model_patcher.get_key_weight(patcher.model, key)
    device = patcher.load_device
    temp_weight = comfy.model_management.cast_to_device(patcher.backup[key].weight, device, comfy.model_management.lora_compute_dtype(device), copy=True)
    if convert_func is not None:
        temp_weight = convert_func(temp_weight, inplace=True)
    out_weight = comfy.lora.calculate_weight(patcher.patches[key], temp_weight, key)
    if set_func is None:
        out_weight = comfy.float.stochastic_rounding(out_weight, weight.dtype, seed=_comfy_utils.string_to_seed(key))
        if patcher.weight_inplace_update:
            _comfy_utils.copy_to_param(patcher.model, key, out_weight)
        else:
            _comfy_utils.set_attr_param(patcher.model, key, out_weight)
    else:
        set_func(out_weight, inplace_update=patcher.weight_inplace_update, seed=_comfy_utils.string_to_seed(key), return_weight=False)


def _update_travel_step(travel_state, step, new_row=False):
    """Retune every travel lora to its strength at the given (global) step index and
    repaint the affected model weights. Skips loras whose strength did not change.
    new_row starts the log on a fresh line once the sampler progress bar is already
    printing on the current one (the per-step callbacks, not the pre-sampling call)."""
    patcher = travel_state["patcher"]
    active = {}
    for lora in travel_state["loras"]:
        strength = _travel_step_strength(lora, step)
        if strength == lora["current"]:
            continue
        for key, idx in lora["entries"]:
            e = patcher.patches[key][idx]
            patcher.patches[key][idx] = (strength, e[1], e[2], e[3], e[4])
        for key, _idx in lora["entries"]:
            _retune_lora_key(patcher, key)
        lora["current"] = strength
        if strength != 0.0:
            active[lora["name"]] = active.get(lora["name"], 0.0) + strength
    if travel_state["verbose"] and active:
        if new_row:
            print()
        print("Gibby Lora Travel: step {}/{}".format(step + 1, travel_state["steps"]))
        for name, s in active.items():
            print("  {}={:.2f}".format(name, s))


def _sample_with_lora_travels(travel_state, noise, cfg_value, sampler_obj, sigmas_tensor, positive, negative,
                              latent_image, noise_mask, callback, disable_pbar, seed, step_offset=0):
    """Sample with the travel loras retuned on every step: each step is denoised with
    the strength the travels have at its global index (i + step_offset), so the run
    proceeds step by step (0-1, 1-2, ...) with per-step lora strengths."""
    steps = len(sigmas_tensor) - 1
    for lora in travel_state["loras"]:
        lora["current"] = 0.0
    _update_travel_step(travel_state, step_offset)
    if callback is not None:
        base_callback = callback
        def callback(step, x0, x, total_steps):
            if step + 1 < steps:
                _update_travel_step(travel_state, step_offset + step + 1, new_row=True)
            base_callback(step, x0, x, total_steps)
    return comfy.sample.sample_custom(travel_state["patcher"], noise, cfg_value, sampler_obj, sigmas_tensor,
                                      positive, negative, latent_image, noise_mask=noise_mask,
                                      callback=callback, disable_pbar=disable_pbar, seed=seed)


def _find_matching(text, start, open_ch, close_ch):
    """The index of the closing char matching the opening char at start; -1 when unbalanced."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == open_ch:
            depth += 1
        elif text[i] == close_ch:
            depth -= 1
            if depth == 0:
                return i
    return -1


def _split_travel(text):
    """Split a [...] group content on the top-level colons: colons inside () and [] stay in their part."""
    parts, cur, depth = [], [], 0
    for ch in text:
        if ch in "([":
            depth += 1
        elif ch in "])":
            depth -= 1
        if ch == ":" and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return parts


def _parse_travel_group(content):
    """The parsed contents of a [...] group: a (before, after, step) switch when the
    content ends in a step, otherwise the content as plain text (Forge prompt travel:
    [before:after:step], step with a dot=fraction of steps, without=absolute step)."""
    parts = _split_travel(content)
    if len(parts) < 2:
        return content
    try:
        float(parts[-1])
    except ValueError:
        return content
    return (_parse_travel_expr(":".join(parts[:-2])), _parse_travel_expr(parts[-2]), parts[-1])


def _parse_travel_expr(text):
    """Parse a travel prompt into a tree: plain text is a string, a sequence is a
    list, and a switch is a (before, after, step) tuple - before is active until
    the step, after from it on. A group without a trailing step is plain text, an
    unbalanced bracket stays as-is, and () groups are atomic weight syntax."""
    nodes, i, n = [], 0, len(text)
    while i < n:
        ch = text[i]
        if ch in "([":
            j = _find_matching(text, i, ch, ")" if ch == "(" else "]")
            if j == -1:
                nodes.append(ch)
                i += 1
            elif ch == "(":
                nodes.append(text[i:j + 1])
                i = j + 1
            else:
                nodes.append(_parse_travel_group(text[i + 1:j]))
                i = j + 1
        else:
            j = i
            while j < n and text[j] not in "([":
                j += 1
            nodes.append(text[i:j])
            i = j
    out = []
    for node in nodes:
        if isinstance(node, str) and out and isinstance(out[-1], str):
            out[-1] += node
        else:
            out.append(node)
    return out[0] if len(out) == 1 else out


def _resolve_travel_step(node, steps_value):
    """Resolve every switch step of a travel tree against the total step count
    (a step with a dot is a fraction of the steps, without one an absolute step)
    and clamp it to [0, steps_value]."""
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        return [_resolve_travel_step(n, steps_value) for n in node]
    before, after, step = node
    val = int(float(step) * steps_value) if "." in step else int(float(step))
    return (_resolve_travel_step(before, steps_value), _resolve_travel_step(after, steps_value),
            max(0, min(steps_value, val)))


def _render_travel(node, step):
    """The prompt text of a travel tree at a (global) step index."""
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        return "".join(_render_travel(n, step) for n in node)
    before, after, start = node
    return _render_travel(after if step >= start else before, step)


def _plan_prompt_travel(text, steps_value):
    """The per-step prompt strings for a travel prompt text, one per global step index."""
    tree = _resolve_travel_step(_parse_travel_expr(text), steps_value)
    return [_render_travel(tree, i) for i in range(steps_value)]


def _prepare_prompt_travels(ctx, opt, steps_value):
    """Plan the option's per-step prompts and encode every unique prompt once with
    the context clip (the option's prompts override the context's). Constant prompts
    are set on the context and sampled normally; switching prompts come back as a
    state for the per-step sampling loop. Returns the state (None when there is
    nothing to travel)."""
    if opt is None:
        return None
    positive = opt.get("positive") or ""
    negative = opt.get("negative") or ""
    if not positive.strip() and not negative.strip():
        return None
    if steps_value is None or steps_value <= 0:
        return None
    clip = ctx.get("clip")
    if clip is None:
        print("Gibby KSampler (Context): Prompt Travel options skipped: the context has no clip")
        return None

    pos_plan = _plan_prompt_travel(positive, steps_value)
    neg_plan = _plan_prompt_travel(negative, steps_value)

    cache = {}
    def encode(text):
        if text not in cache:
            cache[text], = CLIPTextEncode().encode(clip, text)
        return cache[text]

    pos_conds = [encode(t) for t in pos_plan]
    if ctx.get("cfg") == 1:
        zero_cache = {}
        def zero_out(cond):
            if id(cond) not in zero_cache:
                zero_cache[id(cond)], = ConditioningZeroOut().zero_out(cond)
            return zero_cache[id(cond)]
        neg_conds = [zero_out(c) for c in pos_conds]
    else:
        neg_conds = [encode(t) for t in neg_plan]

    if all(c is pos_conds[0] for c in pos_conds) and all(c is neg_conds[0] for c in neg_conds):
        ctx["positive"] = pos_conds[0]
        ctx["negative"] = neg_conds[0]
        return None
    return {
        "positive": pos_conds,
        "negative": neg_conds,
        "positive_text": pos_plan,
        "negative_text": neg_plan,
        "steps": steps_value,
        "verbose": opt.get("verbose", False),
        "current_pos": pos_conds[0],
        "current_neg": neg_conds[0],
    }


def _log_prompt_step(state, step, new_row=False):
    """The verbose log for a step: the active positive/negative prompts. new_row
    starts the log on a fresh line once the sampler progress bar is already
    printing on the current one (the per-step callbacks)."""
    if not state["verbose"]:
        return
    if new_row:
        print()
    print("Gibby Prompt Travel: step {}/{}".format(step + 1, state["steps"]))
    print("  positive: {}".format(state["positive_text"][step]))
    print("  negative: {}".format(state["negative_text"][step]))


def _set_cond_text(conds_list, encoded, inner, noise, device, prompt_type):
    """Replace the text-derived parts of the processed conds with an encoded prompt
    and rebuild the model conds from them - the model reads the text through
    model_conds, which process_conds built from the prompt at the start of the run."""
    for processed, (cross_attn, cond_dict) in zip(conds_list, encoded):
        processed["cross_attn"] = cross_attn
        for key in ("pooled_output", "prompt"):
            if key in cond_dict:
                processed[key] = cond_dict[key]
    if hasattr(inner, "extra_conds"):
        comfy.samplers.encode_model_conds(inner.extra_conds, conds_list, noise, device, prompt_type)


def _update_prompt_step(state, guider, model, noise, step, new_row=False):
    """Swap the guider's conds to the prompts of the given (global) step and log
    them when verbose. Skips the swap when the step's prompts are unchanged."""
    pos, neg = state["positive"][step], state["negative"][step]
    if pos is state["current_pos"] and neg is state["current_neg"]:
        return
    inner = model.model if hasattr(model, "model") else model
    device = noise.device
    _set_cond_text(guider.conds["positive"], pos, inner, noise, device, "positive")
    _set_cond_text(guider.conds["negative"], neg, inner, noise, device, "negative")
    state["current_pos"], state["current_neg"] = pos, neg
    _log_prompt_step(state, step, new_row)


def _sample_with_travels(model_obj, lora_state, prompt_state, noise, cfg_value, sampler_obj, sigmas_tensor,
                         positive, negative, latent_image, noise_mask, callback, disable_pbar, seed, step_offset=0):
    """Sample with per-step travels: lora strengths retuned and/or prompts swapped
    on every step, indexed by global step (step_offset + local), so the run proceeds
    step by step (0-1, 1-2, ...) with the travels' values at each step. Plain sampling
    when nothing travels; a lora-only run keeps the sample_custom path."""
    if prompt_state is None:
        if lora_state is None:
            return comfy.sample.sample_custom(model_obj, noise, cfg_value, sampler_obj, sigmas_tensor,
                                               positive, negative, latent_image, noise_mask=noise_mask,
                                               callback=callback, disable_pbar=disable_pbar, seed=seed)
        return _sample_with_lora_travels(lora_state, noise, cfg_value, sampler_obj, sigmas_tensor,
                                          positive, negative, latent_image, noise_mask, callback,
                                          disable_pbar, seed, step_offset)
    steps = len(sigmas_tensor) - 1
    model = lora_state["patcher"] if lora_state is not None else model_obj
    if lora_state is not None:
        for lora in lora_state["loras"]:
            lora["current"] = 0.0
        _update_travel_step(lora_state, step_offset)
    guider = comfy.samplers.CFGGuider(model)
    guider.set_conds(prompt_state["positive"][step_offset], prompt_state["negative"][step_offset])
    guider.set_cfg(cfg_value)
    prompt_state["current_pos"] = prompt_state["positive"][step_offset]
    prompt_state["current_neg"] = prompt_state["negative"][step_offset]
    _log_prompt_step(prompt_state, step_offset)

    def travel_callback(step, x0, x, total_steps):
        if step + 1 < steps:
            next_step = step_offset + step + 1
            if lora_state is not None:
                _update_travel_step(lora_state, next_step, new_row=True)
            _update_prompt_step(prompt_state, guider, model, noise, next_step, new_row=True)
        if callback is not None:
            callback(step, x0, x, total_steps)

    samples = guider.sample(noise, latent_image, sampler_obj, sigmas_tensor, noise_mask, travel_callback, disable_pbar, seed)
    return samples.to(device=comfy.model_management.intermediate_device(), dtype=comfy.model_management.intermediate_dtype())


def _apply_travel_clip_loras(ctx, travel_opts, steps_value):
    """Apply the travels' clip loras once, at the strength the loras enter with
    (start_str), so conditioning encoded from clip carries them. No-op when clip
    is absent or a travel has no active steps."""
    clip = ctx.get("clip")
    if clip is None or not travel_opts or not steps_value or steps_value <= 0:
        return
    for opt in travel_opts:
        start, end = _resolve_step_range(steps_value, opt.get("start_step", 0.0), opt.get("end_step", 100.0))
        if end <= start:
            continue
        start_str = opt.get("start_str", 0.6)
        for item in opt.get("lora_stack") or []:
            if not item or len(item) < 3 or item[0] in (None, "None"):
                continue
            name, strength_clip = item[0], item[2]
            if not strength_clip:
                continue
            lora = _load_lora_state(name)
            if lora is None:
                continue
            clip = _load_lora_into_clip(clip, lora, strength_clip * start_str)
    ctx["clip"] = clip


def _travel_signature(travel_opts, steps_value):
    """Hashable signature of the travel settings for the sample cache; None when there is no travel."""
    if not travel_opts or not steps_value or steps_value <= 0:
        return None
    sig = []
    for opt in travel_opts:
        start, end = _resolve_step_range(steps_value, opt.get("start_step", 0.0), opt.get("end_step", 100.0))
        stack = tuple(tuple(item) for item in (opt.get("lora_stack") or []))
        sig.append((start, end, opt.get("start_str", 0.6), opt.get("end_str", 1.0), stack))
    return tuple(sig)


def _prompt_travel_signature(opt, steps_value):
    """Hashable signature of the prompt travel settings for the sample cache; None when there is none."""
    if opt is None or not steps_value or steps_value <= 0:
        return None
    positive = opt.get("positive") or ""
    negative = opt.get("negative") or ""
    if not positive.strip() and not negative.strip():
        return None
    return (positive, negative)


def _steps_display(steps_value, start_step, end_step):
    """Total steps, or 'start-end/total' when a start/end sub-range is active."""
    if steps_value is None or steps_value <= 0:
        return str(steps_value)
    if start_step != 0.0 or end_step < 10000.0:
        s, e = _resolve_step_range(steps_value, start_step, end_step)
        return f"{s}-{e}/{steps_value}"
    return str(steps_value)


def _log_start(verbose, ctx, seed, steps_value, denoise, start_step=0.0, end_step=10000.0):
    """Console log at the start of a run: model, sampling params, resolution, video length."""
    if not verbose:
        return
    model_name = ctx.get("model_name") or "(unknown)"
    w, h, length = _image_dims(ctx.get("image"))
    if w is None:
        w, h, length = _latent_dims(ctx.get("latent"), ctx.get("vae"))
    if w in (None, 0):
        w, h = ctx.get("width", 0), ctx.get("height", 0)
    res = f"{w}x{h}" if w and h else "unknown"
    length_s = f" length={length} frames" if length else ""
    print(f"Gibby KSampler (Context) start: model={model_name} steps={_steps_display(steps_value, start_step, end_step)} "
          f"cfg={ctx.get('cfg', 8.0)} sampler={ctx.get('sampler', 'euler')} scheduler={ctx.get('scheduler', 'normal')} "
          f"denoise={denoise} seed={seed} resolution={res}{length_s}")


def _log_finish(verbose, image, latent, t_start, vae=None):
    """Console log at the end of a run: resulting resolution and total time."""
    if not verbose:
        return
    w, h, length = _image_dims(image)
    if w is None:
        w, h, length = _latent_dims(latent, vae)
    res = f"{w}x{h}" if w and h else "unknown"
    length_s = f" length={length} frames" if length else ""
    print(f"Gibby KSampler (Context) finish: resolution={res}{length_s} total_time={time.time() - t_start:.2f}s")


def _encode_image(vae, image, tiled_decode, tile_size, overlap, temporal_size, temporal_overlap, verbose=False):
    """VAE-encode an image, tiled when the tiled VAE settings are on."""
    if verbose:
        print("Gibby KSampler (Context): VAE encode starting")
    t0 = time.time()
    if tiled_decode:
        latent, = VAEEncodeTiled().encode(vae, image, tile_size, overlap, temporal_size, temporal_overlap)
    else:
        latent, = VAEEncode().encode(vae, image)
    if verbose:
        print(f"Gibby KSampler (Context): VAE encode took {time.time() - t0:.2f}s")
    return latent


def _decode_latent(vae, latent, tiled_decode, tile_size, overlap, temporal_size, temporal_overlap, verbose=False):
    """VAE-decode a latent, tiled when the tiled VAE settings are on."""
    if verbose:
        print("Gibby KSampler (Context): VAE decode starting")
    t0 = time.time()
    if tiled_decode:
        image, = VAEDecodeTiled().decode(vae, latent, tile_size, overlap, temporal_size, temporal_overlap)
    else:
        image, = VAEDecode().decode(vae, latent)
    if verbose:
        print(f"Gibby KSampler (Context): VAE decode took {time.time() - t0:.2f}s")
    return image


def _normalize_mask(mask):
    """Normalize the untrusted mask input to a (B,H,W) float mask. The mask input
    can receive non-mask tensors (e.g. an image bridged into it); anything that is
    not a 2D spatial mask returns None and is treated as no mask."""
    if mask is None or not torch.is_tensor(mask):
        return None
    mask = mask.float()
    if mask.dim() == 4:
        # A channel-first mask always has a small channel dim (shape[1]); a
        # channels-last (B,H,W,C) image bridged into the input has a large one.
        if mask.shape[1] > 4:
            return None
        if mask.shape[1] == 1:
            mask = mask.squeeze(1)   # (B,1,H,W) channel-first
        else:
            mask = mask[:, 0]        # (B,C,H,W) -> first channel
    if mask.dim() != 3:
        return None
    return mask


def _get_mask_bbox(mask):
    """Get bounding box of non-zero mask area. Returns (x, y, w, h) or None."""
    m = mask.squeeze()
    if m.dim() > 2:
        m = m[0, 0]
    rows = (m > 0.001).any(dim=1)
    cols = (m > 0.001).any(dim=0)
    if not rows.any() or not cols.any():
        return None
    top = rows.nonzero().squeeze(-1)[0].item()
    bottom = rows.nonzero().squeeze(-1)[-1].item()
    left = cols.nonzero().squeeze(-1)[0].item()
    right = cols.nonzero().squeeze(-1)[-1].item()
    return (left, top, right - left + 1, bottom - top + 1)


def _resize_to_target(img, mask, megapixels, scale_factor, multiple, method):
    """Resize image and mask to target size. Returns (img, mask, new_w, new_h)."""
    h, w = img.shape[1], img.shape[2]

    if scale_factor != 1.0:
        h = int(h * scale_factor)
        w = int(w * scale_factor)

    if megapixels > 0:
        target_px = megapixels * 1_000_000
        current_px = h * w
        if current_px > 0:
            scale = (target_px / current_px) ** 0.5
            h = int(h * scale)
            w = int(w * scale)

    if multiple > 1:
        h = max(multiple, (h // multiple) * multiple)
        w = max(multiple, (w // multiple) * multiple)

    if h == img.shape[1] and w == img.shape[2]:
        return img, mask, w, h

    import torch.nn.functional as F
    mode = {"bilinear": "bilinear", "area": "area", "nearest": "nearest", "lanczos": "bilinear"}.get(method, "bilinear")
    # F.interpolate expects (N, C, H, W) — permute channels-last to channels-first
    img = F.interpolate(img.permute(0, 3, 1, 2), size=(h, w), mode=mode, align_corners=False).permute(0, 2, 3, 1)
    if mask.dim() == 3:
        mask = F.interpolate(mask.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False).squeeze(1)
    else:
        mask = F.interpolate(mask, size=(h, w), mode="bilinear", align_corners=False)
    return img, mask, w, h


def _scale_mask(mask, scale):
    """Rescale mask: scale<1 shrinks white area, scale>1 expands. scale=1 is no-op."""
    if scale == 1.0:
        return mask
    import torch.nn.functional as F
    # Resize mask to scale size, then back — with threshold to keep hard edges
    b, h, w = mask.shape
    m4 = mask.unsqueeze(1).float()
    small_h = max(2, int(h * scale))
    small_w = max(2, int(w * scale))
    m4 = F.interpolate(m4, size=(small_h, small_w), mode="bilinear", align_corners=False)
    m4 = F.interpolate(m4, size=(h, w), mode="bilinear", align_corners=False)
    return m4.squeeze(1).to(mask.dtype)


def _inpaint_regions(image, mask, opts):
    """Resize the mask to the image and compute crop regions per the crop-inpaint
    options. Returns (mask, regions); regions is empty when there is nothing to inpaint."""
    import torch.nn.functional as F

    # Resize mask to image size if mismatch
    if mask.shape[1] != image.shape[1] or mask.shape[2] != image.shape[2]:
        if mask.dim() == 3:
            mask = F.interpolate(mask.unsqueeze(1), size=(image.shape[1], image.shape[2]), mode="bilinear", align_corners=False).squeeze(1)
        else:
            mask = F.interpolate(mask, size=(image.shape[1], image.shape[2]), mode="bilinear", align_corners=False)

    # Get mask regions (single or split)
    mask_mode = opts.get("mask_mode", "single")
    crop_factor = opts.get("crop_factor", 3.0)
    img_h, img_w = image.shape[1], image.shape[2]

    regions = []
    if mask_mode == "split":
        # Find disconnected regions using contour detection
        import cv2
        import numpy as np
        mask_2d = (mask.squeeze(0).cpu().numpy() * 255).astype(np.uint8) if mask.dim() == 3 else (mask.cpu().numpy() * 255).astype(np.uint8)
        mask_2d_float = mask.squeeze(0) if mask.dim() == 3 else mask
        contours, hierarchy = cv2.findContours(mask_2d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is not None:
            for j, contour in enumerate(contours):
                if hierarchy[0][j][3] != -1:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                if w < 4 or h < 4:
                    continue
                # Expand by crop_factor
                cw = int(w * crop_factor)
                ch = int(h * crop_factor)
                cx = x + w // 2
                cy = y + h // 2
                x0 = max(0, cx - cw // 2)
                y0 = max(0, cy - ch // 2)
                x1 = min(img_w, x0 + cw)
                y1 = min(img_h, y0 + ch)
                # Isolate this segment so other segments inside the crop
                # window are not inpainted together with it
                seg = np.zeros((img_h, img_w), dtype=np.uint8)
                cv2.drawContours(seg, [contour], -1, 1, -1)
                seg_mask = mask_2d_float * torch.from_numpy(seg > 0).to(mask.device)
                regions.append((x0, y0, x1, y1, seg_mask))
        # Filter by mask_indices if provided
        mask_indices_str = opts.get("mask_indices", "")
        if mask_indices_str:
            import re as _re
            indices = [int(s) for s in _re.findall(r'\d+', mask_indices_str)]
            valid = [i for i in indices if i < len(regions)]
            if valid:
                regions = [regions[i] for i in valid]
            else:
                regions = []
    else:
        # Single region: use full mask bbox
        bbox = _get_mask_bbox(mask)
        if bbox is not None:
            mx, my, mw, mh = bbox
            cw = int(mw * crop_factor)
            ch = int(mh * crop_factor)
            cx = mx + mw // 2
            cy = my + mh // 2
            x0 = max(0, cx - cw // 2)
            y0 = max(0, cy - ch // 2)
            x1 = min(img_w, x0 + cw)
            y1 = min(img_h, y0 + ch)
            regions.append((x0, y0, x1, y1, None))

    return mask, regions


def _make_noise_mask(model_obj, crop_latent, crop_mask, inpaint_mode, mask_scale_start, mask_scale_end):
    """Build the noise mask for masked_only inpainting. Returns (noise_mask, model_obj)."""
    if inpaint_mode != "masked_only":
        return None, model_obj

    if abs(mask_scale_start - mask_scale_end) <= 0.001 and mask_scale_start == 1.0:
        masked = SetLatentNoiseMask().set_mask(crop_latent, crop_mask)
        if isinstance(masked, tuple):
            masked = masked[0]
        return masked.get("noise_mask"), model_obj

    b, h, w = crop_mask.shape
    mask_2d = crop_mask.squeeze(0) if crop_mask.dim() == 3 else crop_mask
    ys, xs = torch.where(mask_2d > 0.5)
    if len(ys) > 0:
        cy = ys.float().mean().item()
        cx = xs.float().mean().item()
        max_r = torch.sqrt((ys.float() - cy) ** 2 + (xs.float() - cx) ** 2).max().item()
    else:
        cy, cx, max_r = h / 2, w / 2, min(h, w) / 2
    yy, xx = torch.meshgrid(
        torch.arange(h, device=crop_mask.device, dtype=torch.float),
        torch.arange(w, device=crop_mask.device, dtype=torch.float),
        indexing='ij'
    )
    dist = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    if abs(mask_scale_start - mask_scale_end) > 0.001:
        r_start = max_r * min(mask_scale_start, mask_scale_end)
        r_end = max_r * max(mask_scale_start, mask_scale_end)
        if r_end > r_start:
            apply_mask = ((r_end - dist) / (r_end - r_start)).clamp(0, 1)
        else:
            apply_mask = (dist <= r_start).float()
        if mask_scale_start < 1.0:
            apply_mask = apply_mask * crop_mask
    else:
        apply_mask = (dist <= max_r * mask_scale_start).float()
        if mask_scale_start < 1.0:
            apply_mask = apply_mask * crop_mask

    masked = SetLatentNoiseMask().set_mask(crop_latent, apply_mask)
    if isinstance(masked, tuple):
        masked = masked[0]
    noise_mask = masked.get("noise_mask")
    from comfy_extras.nodes_differential_diffusion import DifferentialDiffusion
    model_obj = DifferentialDiffusion.execute(model_obj, strength=1.0)[0]
    return noise_mask, model_obj


def _color_match(image, reference, opts):
    """Match the image's color back to the reference with Transfer Color per the
    option's color_match settings. Returns the image unchanged when it is off."""
    if not opts.get("color_match", True):
        return image
    method = opts.get("color_match_method", "mkl_lab")
    strength = float(opts.get("color_match_strength", 1.0))
    matched, = ColorTransfer.execute(image, reference, method, {"source_stats": "per_frame"}, strength)
    return matched


def _crop_region(image, mask, seg_mask, x0, y0, x1, y1):
    """Crop the image and its mask to a region; in split mode only this
    segment's mask pixels are kept."""
    crop_img = image[:, y0:y1, x0:x1, :]
    if seg_mask is not None:
        crop_mask = seg_mask[y0:y1, x0:x1].unsqueeze(0)
    else:
        crop_mask = mask[:, y0:y1, x0:x1]
    return crop_img, crop_mask


def _composite_region(full, crop, mask, y0, y1, x0, x1):
    """Blend a processed crop back into the full image's region, feathered by the mask."""
    import torch.nn.functional as F
    if crop.dim() == 3:
        crop = crop.unsqueeze(0)
    m = mask.float()
    if m.dim() == 3:
        m = m.unsqueeze(1)
    h, w = y1 - y0, x1 - x0
    paste = F.interpolate(crop.permute(0, 3, 1, 2), size=(h, w), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
    pm = F.interpolate(m, size=(h, w), mode="bilinear", align_corners=False)
    pm = F.avg_pool2d(pm, kernel_size=5, stride=1, padding=2)
    pm = pm.clamp(0, 1).squeeze(1).unsqueeze(-1)
    full[:, y0:y1, x0:x1, :3] = paste[:, :, :, :3] * pm + full[:, y0:y1, x0:x1, :3] * (1 - pm)


def _encode_sample_decode(image, mask, vae, model_obj, seed, steps_value, cfg_value, sampler_obj,
                          sigmas_tensor, positive, negative, inpaint_opts,
                          tiled_decode, tile_size, overlap, temporal_size, temporal_overlap,
                          decode=True, clear_after_model=False, clear_after_vae=False, clear_verbose=False,
                          verbose=False, travel_state=None, prompt_state=None, step_offset=0):
    """Shared sample step: encode the image, sample it (masked when a mask is
    present, per the crop-inpaint options), and decode the result. Returns
    (image, model_obj); image is None when decode is off."""
    latent = _encode_image(vae, image.float(), tiled_decode, tile_size, overlap, temporal_size, temporal_overlap, verbose)
    latent_samples = comfy.sample.fix_empty_latent_channels(
        model_obj, latent["samples"],
        latent.get("downscale_ratio_spacial", None),
        latent.get("downscale_ratio_temporal", None)
    )
    noise = comfy.sample.prepare_noise(latent_samples, seed)
    noise_mask = None
    if mask is not None:
        noise_mask, model_obj = _make_noise_mask(model_obj, latent, mask,
                                                 (inpaint_opts or {}).get("inpaint_mode", "masked_only"),
                                                 (inpaint_opts or {}).get("mask_scale_start", 1.0),
                                                 (inpaint_opts or {}).get("mask_scale_end", 1.0))
    callback = latent_preview.prepare_callback(model_obj, steps_value)
    samples = _sample_with_travels(model_obj, travel_state, prompt_state, noise, cfg_value, sampler_obj, sigmas_tensor,
                                    positive, negative, latent_samples, noise_mask, callback,
                                    not _comfy_utils.PROGRESS_BAR_ENABLED, seed, step_offset)
    # The model is done after sampling: free it before the decode so large
    # resolutions don't choke the VAE (Clear VRAM options)
    if clear_after_model:
        _clear_vram("after model", clear_verbose)
    if not decode:
        return None, model_obj
    image = _decode_latent(vae, {"samples": samples}, tiled_decode, tile_size, overlap, temporal_size, temporal_overlap, verbose)
    if clear_after_vae:
        _clear_vram("after vae", clear_verbose)
    return image, model_obj


def _crop_inpaint(ctx, opts, model_obj, seed, steps_value,
                  sampler_obj, sigmas_tensor, cfg_value, positive, negative, decode, options_list=None,
                  clear_after_model=False, clear_after_vae=False, clear_verbose=False, verbose=False,
                  travel_state=None, prompt_state=None):
    """Crop image by mask bbox, resize, sample each crop, composite back.
    Returns None when there is nothing to inpaint, so the caller samples the whole image."""
    image = ctx["image"]
    mask = ctx["mask"]
    vae = ctx.get("vae")

    mask, regions = _inpaint_regions(image, mask, opts)
    if not regions:
        return None

    opt_verbose = opts.get("verbose", False)
    if opt_verbose and opts.get("mask_mode", "single") == "split":
        print(f"Gibby Crop-Inpaint: {len(regions)} masks")

    # Process each region
    megapixels = opts.get("megapixels", 0.0)
    scale_factor = opts.get("scale_factor", 1.0)
    multiple = opts.get("multiple", 8)
    method = opts.get("upscale_method", "bilinear")
    tiled_decode, tile_size, overlap, temporal_size, temporal_overlap = _tiled_settings(options_list)

    result_image = image.clone()

    for idx, (x0, y0, x1, y1, seg_mask) in enumerate(regions):
        cw = x1 - x0
        ch = y1 - y0
        if opt_verbose:
            print(f"Gibby Crop-Inpaint: mask {idx + 1}/{len(regions)} crop size={cw}x{ch}")

        crop_img, crop_mask = _crop_region(result_image, mask, seg_mask, x0, y0, x1, y1)
        crop_img, crop_mask, _, _ = _resize_to_target(crop_img, crop_mask, megapixels, scale_factor, multiple, method)
        refined_crop, model_obj = _encode_sample_decode(crop_img, crop_mask, vae, model_obj, seed, steps_value,
                                                        cfg_value, sampler_obj, sigmas_tensor, positive, negative, opts,
                                                        tiled_decode, tile_size, overlap, temporal_size, temporal_overlap,
                                                        decode and vae is not None,
                                                        clear_after_model, clear_after_vae, clear_verbose, verbose,
                                                        travel_state=travel_state, prompt_state=prompt_state)
        if refined_crop is not None:
            _composite_region(result_image, refined_crop, crop_mask, y0, y1, x0, x1)

    # Without a decode the VAE is done after the last region's encode: free it.
    if clear_after_vae and not (decode and vae is not None):
        _clear_vram("after vae", clear_verbose)

    # Match the result's color back to the original image
    result_image = _color_match(result_image, image, opts)

    # Update context
    ctx["image"] = result_image
    ctx.pop("latent", None)
    ctx.pop("mask", None)

    return io.NodeOutput(ctx, None, result_image, None, vae, ctx.get("vae_audio"), options_list)


def _iterative_upscale_step(image, mask, target_w, target_h, method, upscale_model, vae, model_obj,
                            seed, steps_value, cfg_value, sampler_obj, sigmas_tensor, positive, negative,
                            inpaint_opts=None, tiled_decode=False, tile_size=512, overlap=64,
                            temporal_size=64, temporal_overlap=8,
                            clear_after_model=False, clear_after_vae=False, clear_verbose=False, verbose=False,
                            travel_state=None, prompt_state=None, step_offset=0):
    """One iterative upscale step: scale image (and mask) to target size, encode, ksample, decode.
    With a mask, each step samples only the masked area (crop-inpaint options control the mask)."""
    import torch.nn.functional as F

    if upscale_model is not None:
        # Upscale with the model until at/above the target width, then bring the
        # result back to the exact target size
        w = image.shape[2]
        while image.shape[2] < target_w:
            image, = ImageUpscaleWithModel().execute(upscale_model, image)
            if image.shape[2] == w:
                break  # x1 model: no growth
        method = "bilinear"

    if image.shape[2] != target_w or image.shape[1] != target_h:
        mode = {"bilinear": "bilinear", "area": "area", "nearest": "nearest", "lanczos": "bilinear"}.get(method, "bilinear")
        image = F.interpolate(image.permute(0, 3, 1, 2), size=(target_h, target_w), mode=mode, align_corners=False).permute(0, 2, 3, 1)

    if mask is not None and (mask.shape[-2] != target_h or mask.shape[-1] != target_w):
        if mask.dim() == 3:
            mask = F.interpolate(mask.unsqueeze(1), size=(target_h, target_w), mode="bilinear", align_corners=False).squeeze(1)
        else:
            mask = F.interpolate(mask, size=(target_h, target_w), mode="bilinear", align_corners=False)

    image, model_obj = _encode_sample_decode(image, mask, vae, model_obj, seed, steps_value, cfg_value, sampler_obj,
                                              sigmas_tensor, positive, negative, inpaint_opts,
                                              tiled_decode, tile_size, overlap, temporal_size, temporal_overlap,
                                              clear_after_model=clear_after_model, clear_after_vae=clear_after_vae,
                                              clear_verbose=clear_verbose, verbose=verbose,
                                              travel_state=travel_state, prompt_state=prompt_state, step_offset=step_offset)
    return image, mask


def _iterative_upscale(ctx, opts, options_list, model_obj, seed, steps_value,
                       cfg_value, sampler_obj, positive, negative,
                       start_step=0.0, end_step=10000.0, leftover_noise=False,
                       inpaint_opts=None, skip_color_match=False,
                       clear_after_model=False, clear_after_vae=False, clear_verbose=False, verbose=False,
                       travel_state=None, prompt_state=None):
    """Iterative pixel-space upscale along a linear scale path (simple step mode).

    State (next_step, base size) lives in the options dict; on first run it is
    initialized from the current image and the options returned with the
    updated next_step, so the output can be fed back in for further steps.
    A mask in the context is resized with the image and applied on every step
    (inpaint mode and mask scaling come from the crop-inpaint options).
    In total mode, after the final planned step the result's color is matched
    back to the original image (Transfer Color, mkl_lab).
    """
    image = ctx["image"]
    mask = ctx.get("mask")
    vae = ctx["vae"]
    # VAEs round-trip exactly only at multiples of their spatial downscale
    # ratio - align the per-step target sizes to it
    vae_ratio = vae.downscale_ratio
    if isinstance(vae_ratio, (tuple, list)):
        # Video VAE: (temporal, h, w)
        h_ratio, w_ratio = int(vae_ratio[1]), int(vae_ratio[2])
    else:
        h_ratio = w_ratio = int(vae_ratio)
    scheduler_name = ctx.get("scheduler", "normal")
    sampler_name = ctx.get("sampler", "euler")
    if sampler_name not in comfy.samplers.KSampler.SAMPLERS:
        sampler_name = comfy.samplers.KSampler.SAMPLERS[0]

    factor = float(opts.get("upscale_factor", 2.0))
    total_steps = max(int(opts.get("steps", 3)), 1)
    start_denoise = float(opts.get("start_denoise", 1.0))
    target_denoise = float(opts.get("target_denoise", 0.6))
    method = opts.get("upscale_method", "bilinear")
    mode = opts.get("mode", "total")
    upscale_model = opts.get("upscale_model")
    opt_verbose = opts.get("verbose", False)
    tiled_decode, tile_size, overlap, temporal_size, temporal_overlap = _tiled_settings(options_list)

    # The engine's output cache can hand back the same options object on
    # repeated runs - never mutate it, the step state travels on a copy
    opts = dict(opts)
    if "next_step" not in opts:
        # First run: record the initial image size and reset the step counter
        opts["next_step"] = 0
        opts["base_w"] = image.shape[2]
        opts["base_h"] = image.shape[1]

    original_image = image  # color reference for the final step

    base_w = int(opts["base_w"])
    base_h = int(opts["base_h"])
    next_step = int(opts["next_step"])

    if mode == "total" and next_step < total_steps:
        step_indices = range(next_step + 1, total_steps + 1)
    else:
        step_indices = [next_step + 1]

    # The per-step sub-run starts at the resolved start step: travel loras are
    # indexed by global step, so shift their range by it
    step_offset = _resolve_step_range(steps_value, start_step, end_step)[0]

    for i in step_indices:
        scale = 1.0 + (factor - 1.0) * i / total_steps
        target_w = max(1, int(round(base_w * scale / w_ratio)) * w_ratio)
        target_h = max(1, int(round(base_h * scale / h_ratio)) * h_ratio)
        # Denoise ramps from start to target across the planned steps, then holds at target
        if i > total_steps:
            denoise_step = target_denoise
        elif total_steps == 1:
            denoise_step = start_denoise
        else:
            denoise_step = start_denoise + (target_denoise - start_denoise) * (i - 1) / (total_steps - 1)
        sigmas_tensor = _calculate_sigmas(model_obj, scheduler_name, steps_value, sampler_name, denoise_step)
        sigmas_tensor, skip = _apply_start_end_steps(sigmas_tensor, steps_value, start_step, end_step, leftover_noise)
        if skip:
            # Start step beyond the available steps: keep the image, advance the step
            next_step = i
            continue
        if opt_verbose:
            print(f"Gibby Iterative Upscale: step {i}/{total_steps} scale={scale:.2f} size={target_w}x{target_h} denoise={denoise_step:.3f}")
        image, mask = _iterative_upscale_step(image, mask, target_w, target_h, method, upscale_model, vae, model_obj,
                                                seed, steps_value, cfg_value, sampler_obj, sigmas_tensor, positive, negative, inpaint_opts,
                                                tiled_decode, tile_size, overlap, temporal_size, temporal_overlap,
                                                clear_after_model, clear_after_vae, clear_verbose, verbose,
                                                travel_state, prompt_state, step_offset)
        next_step = i

    # After the final planned step, match the result's color back to the original image
    if not skip_color_match and mode == "total" and next_step == total_steps:
        image = _color_match(image, original_image, opts)

    opts["next_step"] = next_step
    # New list, not in-place: the input list may be a cached engine object
    options_list = [opts if o.get("type") == "iterative_upscale" else o for o in options_list]

    # Update context
    ctx["image"] = image
    if mask is None:
        ctx.pop("mask", None)
    else:
        ctx["mask"] = mask
    ctx.pop("latent", None)

    return io.NodeOutput(ctx, None, image, None, vae, ctx.get("vae_audio"), options_list)


def _iterative_inpaint_upscale(ctx, inpaint_opts, iterative_opts, options_list, model_obj, seed, steps_value,
                               cfg_value, sampler_obj, positive, negative,
                               start_step=0.0, end_step=10000.0, leftover_noise=False,
                               clear_after_model=False, clear_after_vae=False, clear_verbose=False, verbose=False,
                               travel_state=None, prompt_state=None):
    """Iterative inpaint upscale: crop the mask regions (crop-inpaint options), iteratively
    upscale each crop with the mask applied on every step, then composite the refined crops
    back into the original-resolution image - the result keeps the original size."""
    image = ctx["image"]
    vae = ctx["vae"]
    mask, regions = _inpaint_regions(image, ctx["mask"], inpaint_opts)
    if not regions:
        return None

    opt_verbose = inpaint_opts.get("verbose", False)
    if opt_verbose and inpaint_opts.get("mask_mode", "single") == "split":
        print(f"Gibby Crop-Inpaint: {len(regions)} masks")

    # Both options may carry color match - the one earlier in the options list
    # wins; when the inpaint one wins, the per-crop match is skipped and the
    # full image is matched back after compositing
    inpaint_wins = options_list.index(inpaint_opts) < options_list.index(iterative_opts)

    # The result stays at the original resolution: each crop is refined by the
    # iterative upscale, then composited back into its original region (the
    # refined crop is downscaled to fit). The original image is not upscaled.
    full = image.clone()

    megapixels = inpaint_opts.get("megapixels", 0.0)
    scale_factor = inpaint_opts.get("scale_factor", 1.0)
    multiple = inpaint_opts.get("multiple", 8)
    crop_method = inpaint_opts.get("upscale_method", "bilinear")

    out_options = options_list
    for idx, (x0, y0, x1, y1, seg_mask) in enumerate(regions):
        if opt_verbose:
            print(f"Gibby Crop-Inpaint: mask {idx + 1}/{len(regions)} crop size={x1 - x0}x{y1 - y0}")
        crop_img, crop_mask = _crop_region(image, mask, seg_mask, x0, y0, x1, y1)
        crop_img, crop_mask, _, _ = _resize_to_target(crop_img, crop_mask, megapixels, scale_factor, multiple, crop_method)
        crop_ctx = {"image": crop_img, "mask": crop_mask, "vae": vae,
                    "scheduler": ctx.get("scheduler", "normal"), "sampler": ctx.get("sampler", "euler")}
        # Pass the crop (image + mask) to the iterative upscale
        out = _iterative_upscale(crop_ctx, iterative_opts, options_list, model_obj, seed,
                                 steps_value, cfg_value, sampler_obj, positive, negative,
                                 start_step, end_step, leftover_noise, inpaint_opts=inpaint_opts,
                                 skip_color_match=inpaint_wins,
                                 clear_after_model=clear_after_model, clear_after_vae=clear_after_vae,
                                 clear_verbose=clear_verbose, verbose=verbose, travel_state=travel_state, prompt_state=prompt_state)
        out_options = out[6]
        # Composite the refined crop back into its original-resolution region
        if x1 > x0 and y1 > y0:
            if opt_verbose:
                print(f"Gibby Crop-Inpaint: mask {idx + 1}/{len(regions)} composited {out[2].shape[2]}x{out[2].shape[1]} -> {x1 - x0}x{y1 - y0}")
            _composite_region(full, out[2], crop_ctx.get("mask"), y0, y1, x0, x1)

    if inpaint_wins:
        full = _color_match(full, image, inpaint_opts)

    if opt_verbose:
        print(f"Gibby Crop-Inpaint: composited {len(regions)} region(s) into {full.shape[2]}x{full.shape[1]} (final image)")

    # Update context
    ctx["image"] = full
    ctx.pop("latent", None)
    ctx.pop("mask", None)

    return io.NodeOutput(ctx, None, full, None, vae, ctx.get("vae_audio"), out_options)


def _calculate_sigmas(model, scheduler_name, steps, sampler_name="euler", denoise=1.0):
    """Calculate sigmas from scheduler name and step count, respecting denoise level."""
    if scheduler_name not in comfy.samplers.KSampler.SCHEDULERS:
        scheduler_name = comfy.samplers.KSampler.SCHEDULERS[0]
    
    device = comfy.model_management.get_torch_device()
    sampler_obj = comfy.samplers.KSampler(model, steps, device, sampler=sampler_name)
    sampler_obj.scheduler = scheduler_name
    
    # Handle denoise like KSampler.set_steps() does
    if denoise is None or denoise > 0.9999:
        sigmas = sampler_obj.calculate_sigmas(steps).to(device)
    else:
        if denoise <= 0.0:
            sigmas = torch.FloatTensor([])
        else:
            new_steps = int(steps/denoise)
            sigmas_full = sampler_obj.calculate_sigmas(new_steps).to(device)
            sigmas = sigmas_full[-(steps + 1):]

    return sigmas


def _apply_start_end_steps(sigmas_tensor, steps_value, start_step, end_step, leftover_noise):
    """Slice sigmas to a start/end step sub-range, like the regular KSampler.

    start/end are multipliers of the step count when abs < 1, offsets from the
    total when negative, otherwise absolute step numbers. Returns (sigmas, skip);
    skip means the start step is at or beyond the available steps and sampling
    should not run.
    """
    start_step_val = round(steps_value * start_step) if abs(start_step) < 1 else int(round(start_step))
    end_step_val = round(steps_value * end_step) if abs(end_step) < 1 else int(round(end_step))
    actual_start_step = steps_value + start_step_val if start_step_val < 0 else start_step_val
    actual_end_step = steps_value + end_step_val if end_step_val < 0 else end_step_val

    if actual_end_step < (len(sigmas_tensor) - 1):
        sigmas_tensor = sigmas_tensor[:actual_end_step + 1]
        if not leftover_noise:
            sigmas_tensor[-1] = 0

    skip = actual_start_step >= (len(sigmas_tensor) - 1)
    if not skip and actual_start_step > 0:
        sigmas_tensor = sigmas_tensor[actual_start_step:]

    return sigmas_tensor, skip


def _ensure_conditioning(ctx):
    """Encode positive/negative conditioning from clip if the context lacks it."""
    if ctx.get("positive") is None and ctx.get("clip") is not None:
        ctx["positive"], = CLIPTextEncode().encode(ctx["clip"], ctx.get("positive_prompt", ""))
    if ctx.get("negative") is None:
        if ctx.get("cfg") == 1 and ctx.get("positive") is not None:
            ctx["negative"], = ConditioningZeroOut().zero_out(ctx["positive"])
        elif ctx.get("clip") is not None:
            ctx["negative"], = CLIPTextEncode().encode(ctx["clip"], ctx.get("negative_prompt", ""))


def _resolve_early_params(ctx, steps):
    """Resolve steps/cfg/sampler for the pre-evaluate sampling paths."""
    base_steps = ctx.get("steps") or 0
    if steps > 0:
        steps_value = steps
    elif ctx.get("step_refiner", 0) > 0:
        steps_value = ctx["step_refiner"]
    else:
        steps_value = base_steps if base_steps > 0 else 20
    cfg_value = ctx.get("cfg", 8.0)
    sampler_name = ctx.get("sampler", "euler")
    if sampler_name not in comfy.samplers.KSampler.SAMPLERS:
        sampler_name = comfy.samplers.KSampler.SAMPLERS[0]
    sampler_obj = comfy.samplers.sampler_object(sampler_name)
    return steps_value, cfg_value, sampler_name, sampler_obj


class GibbyKSamplerContext(io.ComfyNode):
    """Sample using parameters from a CONTEXT object with optional overrides."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Gibby_KSampler_Context",
            display_name="KSampler (Context)",
            category="gibby",
            description=(
                "Samples using parameters from a CONTEXT object. Override model, latent, image, "
                "mask, sampler, or sigmas as needed. Automatically encodes images and updates context."
            ),
            inputs=[
                _CONTEXT_TYPE.Input("context"),
                io.Model.Input("model", optional=True),
                io.Latent.Input("latent", optional=True),
                io.Image.Input("image", optional=True),
                io.Mask.Input("mask", optional=True),
                io.Sampler.Input("sampler", optional=True),
                io.Sigmas.Input("sigmas", optional=True),
                _KSAMPLER_OPTIONS_TYPE.Input("options", optional=True, tooltip="Connect an options node (Crop-Inpaint, Iterative Upscale, Tiled VAE, Clear VRAM) or Merge KSampler Options"),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate="fixed"),
                io.Int.Input("steps", default=0, min=0, max=10000),
                io.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Float.Input("start_step", default=0.0, min=-10000.0, max=10000.0, step=0.01, advanced=True),
                io.Float.Input("end_step", default=10000.0, min=-10000.0, max=10000.0, step=0.01, advanced=True),
                io.Boolean.Input("add_noise", default=True, advanced=True),
                io.Boolean.Input("leftover_noise", default=False, advanced=True),
                io.Boolean.Input("decode", default=True, advanced=True, tooltip="Decode latent to image/audio. When False, stores latent in context instead."),
                io.Boolean.Input("verbose", default=False, advanced=True, tooltip="Log to console: start params/resolution, VAE encode/decode times, and finish resolution/total time."),
            ],
            outputs=[
                _CONTEXT_TYPE.Output(display_name="context"),
                io.Latent.Output(display_name="latent"),
                io.Image.Output(display_name="image"),
                io.Audio.Output(display_name="audio"),
                io.Vae.Output(display_name="vae"),
                io.Vae.Output(display_name="audio_vae"),
                _KSAMPLER_OPTIONS_TYPE.Output(display_name="options"),
            ],
        )

    @classmethod
    def execute(cls, context, model=None, latent=None, image=None, mask=None, options=None,
                sampler=None, sigmas=None, seed=0, steps=0, denoise=1.0,
                start_step=0.0, end_step=10000.0, add_noise=True, leftover_noise=False,
                decode=True, verbose=False) -> io.NodeOutput:
        t_start = time.time()
        options_list = options or []
        travel_opts = [o for o in options_list if o.get("type") == "lora_travel"]
        prompt_opt = _find_option(options_list, "prompt_travel")
        clear_opt = _find_option(options_list, "clear_vram")
        clear_at_start = clear_opt.get("at_start", True) if clear_opt else False
        clear_after_finish = clear_opt.get("after_finish", True) if clear_opt else False
        clear_after_model = clear_opt.get("after_model", True) if clear_opt else False
        clear_after_vae = clear_opt.get("after_vae", True) if clear_opt else False
        clear_verbose = clear_opt.get("verbose", False) if clear_opt else False
        if clear_at_start:
            _clear_vram("at start", clear_verbose)

        # Start with context dict and apply overrides
        ctx = dict(context) if isinstance(context, dict) else {}

        if model is not None:
            ctx["model"] = model

        if latent is not None:
            ctx["latent"] = latent

        if image is not None:
            ctx["image"] = image
            # A connected image takes priority over a latent inherited from
            # the context (a directly-connected latent still wins): clear it
            # so the image is encoded fresh downstream, and drop the cached
            # sample that was built from the old latent
            if latent is None and ctx.get("latent") is not None:
                ctx.pop("latent", None)
                ctx.pop("_kctx_sampled", None)

        # A directly-connected mask overrides the context's before evaluation.
        # The input is untrusted (an image can be bridged into it) - keep only
        # usable masks, treat anything else as absent.
        mask = _normalize_mask(mask)
        if mask is not None:
            ctx["mask"] = mask

        # Tiled VAE settings come from the options list (Tiled VAE options node):
        # presence of the option switches encode/decode to the tiled variants
        tiled_decode, tile_size, overlap, temporal_size, temporal_overlap = _tiled_settings(options)

        # Apply image-based options (crop-inpaint, iterative upscale) before
        # evaluate(), which would otherwise encode the image for nothing
        inpaint_opt = _find_option(options_list, "inpaint")
        iterative_opt = _find_option(options_list, "iterative_upscale")
        has_image = ctx.get("image") is not None
        has_mask = ctx.get("mask") is not None
        has_vae = ctx.get("vae") is not None
        result = None
        if inpaint_opt is not None and iterative_opt is not None and has_image and has_mask and has_vae:
            # Crop first, then pass each crop (image + mask) to the iterative
            # upscale with the mask applied on every step, composite back
            model_obj = ctx.get("model")
            steps_value, cfg_value, sampler_name, sampler_obj = _resolve_early_params(ctx, steps)
            _apply_travel_clip_loras(ctx, travel_opts, steps_value)
            _ensure_conditioning(ctx)
            travel_state = _prepare_lora_travels(model_obj, travel_opts, steps_value)
            prompt_state = _prepare_prompt_travels(ctx, prompt_opt, steps_value)
            _log_start(verbose, ctx, seed, steps_value, denoise, start_step, end_step)
            result = _iterative_inpaint_upscale(ctx, inpaint_opt, iterative_opt, options_list, model_obj, seed,
                                                 steps_value, cfg_value, sampler_obj, ctx.get("positive"), ctx.get("negative"),
                                                 start_step, end_step, leftover_noise,
                                                 clear_after_model, clear_after_vae, clear_verbose, verbose=verbose,
                                                 travel_state=travel_state, prompt_state=prompt_state)
            if result is None:
                # No mask regions: iteratively upscale the whole image
                result = _iterative_upscale(ctx, iterative_opt, options_list, model_obj, seed,
                                             steps_value, cfg_value, sampler_obj, ctx.get("positive"), ctx.get("negative"),
                                             start_step, end_step, leftover_noise,
                                             clear_after_model=clear_after_model, clear_after_vae=clear_after_vae,
                                             clear_verbose=clear_verbose, verbose=verbose,
                                             travel_state=travel_state, prompt_state=prompt_state)
        elif iterative_opt is not None and has_image and has_vae:
            # No crop-inpaint (or no mask for it): iterative on the existing image,
            # masked on every step when a mask is present
            model_obj = ctx.get("model")
            steps_value, cfg_value, sampler_name, sampler_obj = _resolve_early_params(ctx, steps)
            _apply_travel_clip_loras(ctx, travel_opts, steps_value)
            _ensure_conditioning(ctx)
            travel_state = _prepare_lora_travels(model_obj, travel_opts, steps_value)
            prompt_state = _prepare_prompt_travels(ctx, prompt_opt, steps_value)
            _log_start(verbose, ctx, seed, steps_value, denoise, start_step, end_step)
            result = _iterative_upscale(ctx, iterative_opt, options_list, model_obj, seed,
                                         steps_value, cfg_value, sampler_obj, ctx.get("positive"), ctx.get("negative"),
                                         start_step, end_step, leftover_noise,
                                         clear_after_model=clear_after_model, clear_after_vae=clear_after_vae,
                                         clear_verbose=clear_verbose, verbose=verbose,
                                         travel_state=travel_state, prompt_state=prompt_state)
        elif inpaint_opt is not None and has_image and has_mask:
            # Crop-inpaint only: crop, sample each crop, composite back. With an
            # empty mask there are no regions and the normal path below samples
            # the whole image/latent
            model_obj = ctx.get("model")
            steps_value, cfg_value, sampler_name, sampler_obj = _resolve_early_params(ctx, steps)
            _apply_travel_clip_loras(ctx, travel_opts, steps_value)
            _ensure_conditioning(ctx)
            travel_state = _prepare_lora_travels(model_obj, travel_opts, steps_value)
            prompt_state = _prepare_prompt_travels(ctx, prompt_opt, steps_value)
            _log_start(verbose, ctx, seed, steps_value, denoise, start_step, end_step)
            scheduler_name = ctx.get("scheduler", "normal")
            sigmas_tensor = _calculate_sigmas(model_obj, scheduler_name, steps_value, sampler_name, denoise)
            result = _crop_inpaint(ctx, inpaint_opt, model_obj, seed, steps_value,
                                   sampler_obj, sigmas_tensor, cfg_value, ctx.get("positive"), ctx.get("negative"), decode,
                                   options_list, clear_after_model, clear_after_vae, clear_verbose, verbose,
                                   travel_state=travel_state, prompt_state=prompt_state)
        if result is not None:
            if clear_after_finish:
                _clear_vram("after finish", clear_verbose)
            _log_finish(verbose, result[2], result[1], t_start, ctx.get("vae"))
            return result

        # Resolve sampling parameters from context with widget overrides
        model_obj = ctx.get("model")
        if "seed_value" not in ctx:
            ctx["seed_value"] = seed

        # Denoise: use widget value only if image exists, otherwise force 1.0
        has_image = ctx.get("image") is not None
        denoise_value = denoise if has_image else 1.0

        # Steps: widget wins; with an image present, prefer refiner steps from context -
        # unless a sub-range of the base steps was explicitly requested via start/end step.
        base_steps = ctx.get("steps") or 0
        if steps > 0:
            steps_value = steps
        elif has_image and ctx.get("step_refiner", 0) > 0 and not (start_step != 0 or end_step < base_steps):
            steps_value = ctx["step_refiner"]
        else:
            steps_value = ctx.get("steps")

        # Lora Travel options: the traveling clip loras are applied once, at the
        # strength the loras enter with, before the conditioning is encoded from clip
        run_steps = len(sigmas) - 1 if sigmas is not None else steps_value
        _apply_travel_clip_loras(ctx, travel_opts, run_steps)

        # Fill in derived values on demand (latent from image, conditioning, width/height).
        # With tiled_decode on, the image is encoded with VAE Encode (Tiled). The
        # image->latent encode runs inside evaluate(); time that call when it will run.
        encode_will_run = ctx.get("latent") is None and ctx.get("image") is not None and ctx.get("vae") is not None
        if verbose and encode_will_run:
            print("Gibby KSampler (Context): VAE encode starting")
        t0 = time.time()
        GibbyContext.evaluate(ctx, tiled=tiled_decode, tile_size=tile_size, overlap=overlap,
                               temporal_size=temporal_size, temporal_overlap=temporal_overlap)
        if verbose and encode_will_run:
            print(f"Gibby KSampler (Context): VAE encode took {time.time() - t0:.2f}s")

        # Apply mask only if it came from the input (not context) and latent exists.
        if mask is not None and ctx.get("latent") is not None and "noise_mask" not in ctx["latent"]:
            ctx["latent"], = SetLatentNoiseMask().set_mask(ctx["latent"], mask)

        _log_start(verbose, ctx, seed, steps_value, denoise_value, start_step, end_step)

        # Cache check: if this context already has a sampled result with matching
        # parameters, return it instead of re-sampling. This handles the case where
        # the latent output wasn't connected during the first run (result discarded
        # by the engine) but is connected now.
        travel_sig = _travel_signature(travel_opts, run_steps)
        prompt_sig = _prompt_travel_signature(prompt_opt, run_steps)
        cached = ctx.get("_kctx_sampled")
        if cached is not None:
            if (cached.get("seed") == seed and cached.get("steps") == steps_value
                    and cached.get("denoise") == denoise_value
                    and cached.get("add_noise") == add_noise
                    and cached.get("leftover_noise") == leftover_noise
                    and cached.get("decode") == decode
                    and cached.get("tiled_decode") == tiled_decode
                    and cached.get("tile_size") == tile_size
                    and cached.get("overlap") == overlap
                    and cached.get("temporal_size") == temporal_size
                    and cached.get("temporal_overlap") == temporal_overlap
                    and cached.get("start_step") == start_step
                    and cached.get("end_step") == end_step
                    and cached.get("lora_travel") == travel_sig
                    and cached.get("prompt_travel") == prompt_sig):
                out_latent = cached["out_latent"]
                decoded_image = ctx.get("image")
                decoded_audio = ctx.get("audio")
                if clear_after_finish:
                    _clear_vram("after finish", clear_verbose)
                if verbose:
                    print("Gibby KSampler (Context): cache hit, reusing sampled result")
                _log_finish(verbose, decoded_image, out_latent, t_start, ctx.get("vae"))
                return io.NodeOutput(ctx, out_latent, decoded_image, decoded_audio, ctx.get("vae"), ctx.get("vae_audio") or ctx.get("audio_vae"), options)

        cfg_value = ctx.get("cfg")

        # Resolve sampler and sigmas
        if sampler is not None:
            # Use provided SAMPLER object directly
            sampler_obj = sampler
        else:
            # Create sampler from context's sampler string
            sampler_name = ctx.get("sampler", "euler")
            if sampler_name not in comfy.samplers.KSampler.SAMPLERS:
                sampler_name = comfy.samplers.KSampler.SAMPLERS[0]
            sampler_obj = comfy.samplers.sampler_object(sampler_name)
        
        if sigmas is not None:
            # Use provided SIGMAS tensor directly (ignore start/end steps)
            sigmas_tensor = sigmas
            use_start_end_steps = False
        else:
            # Check if model is Flux2 - if so, use Flux2Scheduler
            is_flux2 = isinstance(model_obj.model, comfy.model_base.Flux2) if hasattr(model_obj, 'model') else False
            
            if is_flux2 and get_schedule is not None:
                # Use Flux2Scheduler logic
                # Get width/height from context, latent, or image
                width = ctx.get("width", 0)
                height = ctx.get("height", 0)
                
                if width == 0 or height == 0:
                    if ctx.get("image") is not None:
                        img_h, img_w = ctx["image"].shape[1], ctx["image"].shape[2]
                        width, height = img_w, img_h
                    elif ctx.get("latent") is not None:
                        lat_h, lat_w = ctx["latent"]["samples"].shape[2], ctx["latent"]["samples"].shape[3]
                        width, height = lat_w * 16, lat_h * 16  # Flux2 uses 16x downscale
                    else:
                        width, height = 1024, 1024  # Default
                
                seq_len = (width * height / (16 * 16))
                sigmas_tensor = get_schedule(steps_value, round(seq_len)).to(comfy.model_management.get_torch_device())
                
                # If no start/end step provided and denoise < 1, use low_sigmas
                has_start_end = start_step != 0.0 or end_step < 10000.0
                if not has_start_end and denoise_value < 1.0:
                    steps = max(sigmas_tensor.shape[-1] - 1, 0)
                    total_steps = round(steps * denoise_value)
                    sigmas_tensor = sigmas_tensor[-(total_steps + 1):]
                
                use_start_end_steps = True
            else:
                # Regular scheduler
                scheduler_name = ctx.get("scheduler", "normal")
                sampler_name_for_sigmas = ctx.get("sampler", "euler")
                sigmas_tensor = _calculate_sigmas(model_obj, scheduler_name, steps_value, sampler_name_for_sigmas, denoise_value)
                use_start_end_steps = True

        # Prompt Travel options: plan the per-step prompts and encode the unique
        # ones with the context clip; constant prompts replace the context
        # conditioning, switching ones travel per step
        prompt_state = _prepare_prompt_travels(ctx, prompt_opt, run_steps)

        # Conditioning was filled in by evaluate() above if it was missing.
        positive = ctx.get("positive")
        negative = ctx.get("negative")

        # Prepare noise
        latent_image = ctx["latent"]["samples"]
        
        # Fix latent channels to match model expectations (like SamplerCustom does)
        latent_image = comfy.sample.fix_empty_latent_channels(
            model_obj, 
            latent_image, 
            ctx["latent"].get("downscale_ratio_spacial", None),
            ctx["latent"].get("downscale_ratio_temporal", None)
        )
        
        if not add_noise:
            noise = comfy.sample.prepare_empty_noise(latent_image)
        else:
            batch_inds = ctx["latent"].get("batch_index") if "batch_index" in ctx["latent"] else None
            noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

        # Get noise mask from latent
        noise_mask = None
        if "noise_mask" in ctx["latent"]:
            noise_mask = ctx["latent"]["noise_mask"]

        # Prepare callback for progress display
        callback = latent_preview.prepare_callback(model_obj, steps_value)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        # Perform sampling with start/end step handling (only when using calculated sigmas)
        samples = None
        if use_start_end_steps:
            sigmas_tensor, skip = _apply_start_end_steps(sigmas_tensor, steps_value, start_step, end_step, leftover_noise)
            if skip:
                # Start step beyond available steps, return latent as-is or zeros
                samples = latent_image if ctx["latent"] is not None else torch.zeros_like(noise)

        # Sample with the (possibly sliced) sigmas
        if samples is None:
            travel_state = _prepare_lora_travels(model_obj, travel_opts, run_steps)
            # The sliced sub-run starts at the resolved start step: travels are
            # indexed by global step, so shift their range by it
            step_offset = _resolve_step_range(steps_value, start_step, end_step)[0] if use_start_end_steps else 0
            samples = _sample_with_travels(model_obj, travel_state, prompt_state, noise, cfg_value, sampler_obj, sigmas_tensor,
                                            positive, negative, latent_image, noise_mask, callback,
                                            disable_pbar, seed, step_offset)

        # The model is done after sampling: free it before the decode so the
        # VAE gets the VRAM back (Clear VRAM options).
        if decode and clear_after_model:
            _clear_vram("after model", clear_verbose)

        # Build output latent
        out_latent = ctx["latent"].copy()
        out_latent.pop("downscale_ratio_spacial", None)
        out_latent.pop("downscale_ratio_temporal", None)
        out_latent["samples"] = samples

        # Decode to image for context update and output
        decoded_image = None
        decoded_audio = None
        
        if decode:
            if ctx.get("vae") is not None:
                decoded_image = _decode_latent(ctx["vae"], out_latent, tiled_decode, tile_size, overlap, temporal_size, temporal_overlap, verbose)

            # Decode to audio for context update and output (overrides any existing context.audio)
            if ctx.get("vae_audio") is not None and vae_decode_audio is not None:
                if verbose:
                    print("Gibby KSampler (Context): VAE audio decode starting")
                t0 = time.time()
                decoded_audio = vae_decode_audio(ctx["vae_audio"], out_latent)
                if verbose:
                    print(f"Gibby KSampler (Context): VAE audio decode took {time.time() - t0:.2f}s")

            if clear_after_vae:
                _clear_vram("after vae", clear_verbose)
        elif clear_after_model or clear_after_vae:
            # No decode: the VAE is done with the pre-sample encode, so both
            # are free to go.
            _clear_vram("after model and vae", clear_verbose)

        # Update context after sampling
        ctx.pop("mask", None)    # Remove mask
        
        if decode:
            # When decoding, remove latent and store decoded image/audio
            ctx.pop("latent", None)
            
            if decoded_image is not None:
                ctx["image"] = decoded_image  # Store decoded image in context

            if decoded_audio is not None:
                ctx["audio"] = decoded_audio  # Decoded audio overrides context.audio
        else:
            # When not decoding, clear image/audio and store latent
            ctx.pop("image", None)
            ctx.pop("audio", None)
            ctx["latent"] = out_latent

        # Store sampled result in context for cache reuse
        ctx["_kctx_sampled"] = {
            "out_latent": out_latent,
            "seed": seed,
            "steps": steps_value,
            "denoise": denoise_value,
            "add_noise": add_noise,
            "leftover_noise": leftover_noise,
            "decode": decode,
            "tiled_decode": tiled_decode,
            "tile_size": tile_size,
            "overlap": overlap,
            "temporal_size": temporal_size,
            "temporal_overlap": temporal_overlap,
            "start_step": start_step,
            "end_step": end_step,
            "lora_travel": travel_sig,
            "prompt_travel": prompt_sig,
        }

        if clear_after_finish:
            _clear_vram("after finish", clear_verbose)

        _log_finish(verbose, decoded_image, out_latent, t_start, ctx.get("vae"))
        return io.NodeOutput(ctx, out_latent, decoded_image, decoded_audio, ctx.get("vae"), ctx.get("vae_audio") or ctx.get("audio_vae"), options)

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
- Iterative Options with no image in the context: the 1st step is the basic
  generation (denoise 1.0, base steps, no mask), and the image it generates
  is treated as if it had been provided (color match reference)
- Crop-inpaint with an empty mask has no regions to inpaint: sampling is
  skipped entirely and the image passes through
- Handles start_step/end_step as offsets from total steps when negative
- Flux2 models use the Flux2 schedule for the sampled size (Flux3 keeps its own
  constant shift and the context's scheduler); denoise takes that
  schedule's low-noise tail (like the native SplitSigmasDenoise) and start/end
  steps then slice the result
- Updates context after sampling (removes latent/mask, stores decoded image)
- Options are applied in list order: crop-inpaint and iterative upscale run
  before evaluate, tiled VAE settings apply to encode/decode, clear VRAM
  options free VRAM at the start/end of the execution and after sampling /
  encode-decode (like Clean VRAM Used), lora travel retunes the lora
  strengths per step, and prompt travel is native: when a context prompt
  carries a Forge-style [before:after:step] group the positive/negative prompts
  are swapped per step, a [before:after:step:radius] group crossfades them
  linearly around the step (radius steps each side stays pure), the per-step
  prompts are re-encoded grafted onto the context conditionings so
  reference payloads (flux2 reference_latents, h3 minimax refs/keyframes) survive;
  the options output returns the (updated) options for feeding back in
- DyPE options (soft dependency on the ComfyUI-DyPE pack, see dype_helper.py):
  dype/sega patch the model. A patch is applied per sample at the size actually
  being sampled, so combined with the iterative upscale every step is patched to
  its own new resolution; the patch never leaves the step, so the outbound
  context carries the original unpatched model
"""

import os
import time
import weakref

import folder_paths
import torch
from scipy import ndimage as ndi
import comfy.samplers
import comfy.sample
import comfy.model_management
import comfy.float
import comfy.lora
import comfy.lora_convert
import comfy.model_patcher
import latent_preview
from nodes import ConditioningZeroOut
from comfy_api.latest import io

try:
    from comfy_extras.nodes_audio import vae_decode_audio
except ImportError:
    vae_decode_audio = None

try:
    from comfy_extras.nodes_flux import get_schedule
except ImportError:
    get_schedule = None

import comfy.utils as _comfy_utils
from nodes import SetLatentNoiseMask
from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
from comfy_extras.nodes_differential_diffusion import DifferentialDiffusion
from comfy_extras.nodes_post_processing import ColorTransfer
from ..context import (_CONTEXT_TYPE, GibbyContext, _recondition_text, _latent_downscale,
                       _image_dims, _latent_dims, _is_flux2, clear_vram, ctx_from, ctx_size,
                       ctx_set_sampled_image, ctx_set_image, ctx_set_latent, drop_image_derived, ensure_conditioning, encode_image, decode_latent,
                       parse_index_spec,
                    _mp, _image_count_s, _latent_count_s, _log)
from ..context.prompt_travel import _has_travel_group, _plan_prompt_travel
from ..lora_loader import iter_lora_stack
from ..ksampler_options.crop_inpaint import _KSAMPLER_OPTIONS_TYPE
from ..crop_paste.crop_image_by_mask import _process_mask, _mask_box, _cover_crop
from ..crop_paste.paste_image_by_mask import _paste_back
from ..resolution_latent import _resize_to_mp_scale, _resize_image, _resize_mask, _mask_bbox
from .dype_helper import DYPE_PATCH_METHODS, apply_dype_patch


def _find_option(options_list, opt_type):
    if options_list is None:
        return None
    for o in options_list:
        if o.get("type") == opt_type:
            return o
    return None


def _with_color_match(opt, color_opt):
    """Copy of the option dict carrying the color match option's settings."""
    o = dict(opt)
    o["color_match_method"] = color_opt.get("color_match_method", "mkl_lab")
    o["color_match_strength"] = color_opt.get("color_match_strength", 1.0)
    return o


def _tiled_settings(options_list):
    """Tiled VAE settings from the options list: (above_megapixels, tile_size, overlap, temporal_size, temporal_overlap).
    above_megapixels is None when the option is absent (never tiled) and 0.0 for always tiled."""
    opts = _find_option(options_list, "tiled_vae")
    if opts is None:
        return None, 512, 64, 64, 8
    return opts.get("above_megapixels", 1.0), opts.get("tile_size", 512), opts.get("overlap", 64), opts.get("temporal_size", 64), opts.get("temporal_overlap", 8)


def _tiled_for(w, h, above_megapixels):
    """Whether the tiled VAE is used for a w x h image: never without the option,
    always with a 0 threshold, otherwise for sizes at or above the threshold in MP."""
    if above_megapixels is None:
        return False
    if above_megapixels <= 0:
        return True
    return (w * h) / 1_000_000 >= above_megapixels


def _resolve_step(steps_value, v):
    """A step value resolved against the step count: abs < 1 is a fraction of the
    steps, a negative value an offset from the total, otherwise an absolute step."""
    val = round(steps_value * v) if abs(v) < 1 else int(round(v))
    return steps_value + val if val < 0 else int(val)


def _resolve_step_range(steps_value, start_step, end_step):
    """Actual (start, end) step indices per the start/end step rules, clamped to [0, steps_value]."""
    def clamp(v):
        return max(0, min(steps_value, _resolve_step(steps_value, v)))
    return clamp(start_step), clamp(end_step)


def _load_lora_state(lora_name):
    """The lora's state dict from disk, via the LoraLoader path; None when the file is missing."""
    path = folder_paths.get_full_path("loras", lora_name)
    if not path or not os.path.isfile(path):
        _log("Lora Travel", f"'{lora_name}' not found in the loras folder - skipped")
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
    ramp from start_str (at start) to end_str (at the last active step, end-1) - the
    ramp value itself when the stack strength is 0 (absolute, not a modifier); 0.0
    outside [start, end)."""
    if not (lora["start"] <= step < lora["end"]):
        return 0.0
    last = lora["end"] - 1
    if last == lora["start"]:
        factor = lora["start_str"]
    else:
        factor = lora["start_str"] + (lora["end_str"] - lora["start_str"]) * (step - lora["start"]) / (last - lora["start"])
    return factor if lora["strength"] == 0 else lora["strength"] * factor


def _prepare_lora_travels(model_obj, travel_opts, steps_value, iteration_steps=None):
    """Apply every travel's lora stack to a clone of the model at zero strength and
    record each lora's patch entries, so the per-step callback can retune them.
    With iteration_steps (per_iteration travels with an iterative upscale), the
    strength is retuned per iteration step instead: start/end resolve against the
    iteration count and every sampling step of an iteration shares one strength.
    Returns None when there is nothing to travel."""
    if not travel_opts or not steps_value or steps_value <= 0:
        return None

    per_iteration = iteration_steps is not None and any(opt.get("per_iteration", False) for opt in travel_opts)
    range_steps = iteration_steps if per_iteration else steps_value
    patcher = model_obj.clone()
    loras = []
    for opt in travel_opts:
        start, end = _resolve_step_range(range_steps, opt.get("start_step", 0.0), opt.get("end_step", 100.0))
        if end <= start:
            continue
        start_str = opt.get("start_str", 0.6)
        end_str = opt.get("end_str", 1.0)
        for name, strength, _sc in iter_lora_stack(opt.get("lora_stack")):
            if strength is None:
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
        "steps": range_steps,
        "per_iteration": per_iteration,
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
        _log("Lora Travel", "step {}/{}".format(step + 1, travel_state["steps"]))
        for name, s in active.items():
            print("  {}={:.2f}".format(name, s))


def _sample_with_lora_travels(travel_state, noise, cfg_value, sampler_obj, sigmas_tensor, positive, negative,
                              latent_image, noise_mask, callback, disable_pbar, seed, step_offset=0):
    """Sample with the travel loras retuned on every step: each step is denoised with
    the strength the travels have at its global index (i + step_offset), so the run
    proceeds step by step (0-1, 1-2, ...) with per-step lora strengths. With
    per_iteration on, the strengths were set per iteration step by the iterative
    upscale loop and every sampling step of an iteration shares them."""
    if travel_state.get("per_iteration"):
        return comfy.sample.sample_custom(travel_state["patcher"], noise, cfg_value, sampler_obj, sigmas_tensor,
                                          positive, negative, latent_image, noise_mask=noise_mask,
                                          callback=callback, disable_pbar=disable_pbar, seed=seed)
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


def _prepare_prompt_travels(ctx, pos_text, neg_text, steps_value, verbose):
    """Plan the per-step prompts from the context's raw prompts and encode every
    unique prompt once, grafted onto the context's conditionings so their reference
    payloads (flux2 reference_latents, h3 minimax_refs/keyframes) survive. Only runs
    when a prompt carries a travel group; a prompt without one keeps the context's
    own conditioning - it is that prompt's encoding already - so only the traveling
    side is ever encoded. Constant prompts are set on the context and sampled
    normally, switching ones come back as a state for the per-step loop.
    Returns the state (None when there is nothing to travel)."""
    pos_travels = _has_travel_group(pos_text)
    neg_travels = _has_travel_group(neg_text)
    if not pos_travels and not neg_travels:
        return None
    if not pos_text.strip() and not neg_text.strip():
        return None
    if steps_value is None or steps_value <= 0:
        return None
    clip = ctx.get("clip")
    if clip is None:
        _log("Prompt Travel skipped", "the context has no clip")
        return None

    pos_plan = _plan_prompt_travel(pos_text, steps_value)
    neg_plan = _plan_prompt_travel(neg_text, steps_value)

    base_pos = ctx.get("positive")
    base_neg = ctx.get("negative")
    cache = {}
    def encode(text, base):
        key = (text, id(base))
        if key not in cache:
            cache[key] = _recondition_text(clip, base, text, ctx)
        return cache[key]

    def cond_for(item, base):
        if isinstance(item, tuple):
            return ("mix", [(encode(t, base), w) for t, w in item[1]])
        return encode(item, base)

    def conds_for(plan, base, travels):
        if not travels and base is not None:
            return [base] * len(plan)
        return [cond_for(item, base) for item in plan]

    pos_conds = conds_for(pos_plan, base_pos, pos_travels)
    if ctx.get("cfg") == 1:
        zero_cache = {}
        def zero_out(cond):
            if isinstance(cond, tuple):
                return ("mix", [(zero_out(c), w) for c, w in cond[1]])
            if id(cond) not in zero_cache:
                zero_cache[id(cond)], = ConditioningZeroOut().zero_out(cond)
            return zero_cache[id(cond)]
        neg_conds = [zero_out(c) for c in pos_conds]
    else:
        neg_conds = conds_for(neg_plan, base_neg, neg_travels)

    if all(c is pos_conds[0] for c in pos_conds) and all(c is neg_conds[0] for c in neg_conds):
        ctx["positive"] = pos_conds[0]
        ctx["negative"] = neg_conds[0]
        return None

    def show(item):
        if isinstance(item, tuple):
            return " + ".join("{} {:.2f}".format(t, w) for t, w in item[1])
        return item

    return {
        "positive": pos_conds,
        "negative": neg_conds,
        "positive_text": [show(t) for t in pos_plan],
        "negative_text": [show(t) for t in neg_plan],
        "steps": steps_value,
        "verbose": verbose,
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
    _log("Prompt Travel", "step {}/{}".format(step + 1, state["steps"]))
    print("  positive: {}".format(state["positive_text"][step]))
    print("  negative: {}".format(state["negative_text"][step]))


def _pad_match(x, y):
    """Zero-pad x and y along the dim where they differ so they share a shape."""
    if x.shape == y.shape:
        return x, y
    dim = next(d for d in range(x.dim()) if x.shape[d] != y.shape[d])
    n = max(x.shape[dim], y.shape[dim])
    def pad(t):
        if t.shape[dim] == n:
            return t
        pairs = [(0, 0)] * t.dim()
        pairs[dim] = (0, n - t.shape[dim])
        return torch.nn.functional.pad(t, tuple(v for p in reversed(pairs) for v in p))
    return pad(x), pad(y)


def _mix_encoded(conds):
    """Weighted mix of encoded prompts: conds is a list of (encoded, weight) pairs
    summing to 1. The cross_attn sequences and the float values (pooled output,
    t5xxl weights) are zero-padded to the common length and summed, the integer
    token keys (t5xxl ids, minimax tags) keep the longest one so the length-coupled
    model paths (anima llm adapter, h3 tag runs) stay consistent, and the shared
    payloads (reference latents, h3 refs/keyframes) come from the first endpoint
    untouched."""
    out = []
    for i, (cross, vals) in enumerate(conds[0][0]):
        mixed = None
        for enc, w in conds:
            c = enc[i][0]
            if mixed is None:
                mixed = c * w
            else:
                mixed, c = _pad_match(mixed, c)
                mixed = mixed + c * w
        mixed_vals = {}
        for key, x in vals.items():
            if not torch.is_tensor(x):
                mixed_vals[key] = x
                continue
            if not x.is_floating_point():
                best = x
                for enc, w in conds[1:]:
                    y = enc[i][1].get(key)
                    if torch.is_tensor(y) and y.numel() > best.numel():
                        best = y
                mixed_vals[key] = best
                continue
            acc = None
            for enc, w in conds:
                y = enc[i][1].get(key)
                if not torch.is_tensor(y):
                    continue
                if acc is None:
                    acc = y * w
                else:
                    acc, y = _pad_match(acc, y)
                    acc = acc + y * w
            mixed_vals[key] = acc
        out.append([mixed, mixed_vals])
    return out


def _materialize_cond(item):
    """The encoded conds of a per-step prompt item: itself when a plain prompt, the
    weighted mix when a ("mix", [(encoded, weight), ...]) item."""
    if not isinstance(item, tuple):
        return item
    return _mix_encoded(item[1])


def _set_cond_text(conds_list, encoded, inner, noise, device, prompt_type):
    """Replace the text-derived parts of the processed conds with an encoded prompt
    and rebuild the model conds from them - the model reads the text through
    model_conds, which process_conds built from the prompt at the start of the run.
    minimax_token_tags is h3's per-token modality tags and t5xxl_ids/t5xxl_weights
    are anima's, all length-coupled to the text sequence, so they must swap with the
    text (the ref latents are left untouched)."""
    for processed, (cross_attn, cond_dict) in zip(conds_list, encoded):
        processed["cross_attn"] = cross_attn
        for key in ("pooled_output", "prompt", "minimax_token_tags", "t5xxl_ids", "t5xxl_weights"):
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
    _set_cond_text(guider.conds["positive"], _materialize_cond(pos), inner, noise, device, "positive")
    _set_cond_text(guider.conds["negative"], _materialize_cond(neg), inner, noise, device, "negative")
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
    # per_iteration travels keep the strength the iterative upscale loop set for
    # the whole iteration; only prompt steps travel per sampling step
    lora_per_step = lora_state is not None and not lora_state.get("per_iteration")
    if lora_per_step:
        for lora in lora_state["loras"]:
            lora["current"] = 0.0
        _update_travel_step(lora_state, step_offset)
    guider = comfy.samplers.CFGGuider(model)
    guider.set_conds(_materialize_cond(prompt_state["positive"][step_offset]),
                     _materialize_cond(prompt_state["negative"][step_offset]))
    guider.set_cfg(cfg_value)
    prompt_state["current_pos"] = prompt_state["positive"][step_offset]
    prompt_state["current_neg"] = prompt_state["negative"][step_offset]
    _log_prompt_step(prompt_state, step_offset)

    def travel_callback(step, x0, x, total_steps):
        if step + 1 < steps:
            next_step = step_offset + step + 1
            if lora_per_step:
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
        for name, _sm, strength_clip in iter_lora_stack(opt.get("lora_stack")):
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


def _prompt_travel_signature(pos_text, neg_text, steps_value):
    """Hashable signature of the prompt travel settings for the sample cache; None when there is none."""
    if not steps_value or steps_value <= 0:
        return None
    pos_text = pos_text or ""
    neg_text = neg_text or ""
    if not pos_text.strip() and not neg_text.strip():
        return None
    return (pos_text, neg_text)


# The objects a cached sample was built from. Every node that swaps one of them
# (Lora Loader the model, Reference Latent the conditioning, Context Loader the
# checkpoint) copies the context, so the cached sample rides along and has to
# notice it went stale.
_SAMPLE_INPUTS = ("model", "positive", "negative", "vae")


def _sample_inputs(ctx):
    """Weak references to the context's model and conditioning, taken before
    evaluate() fills in the derived values so a re-run of this node compares the
    same objects. Weak references keep nothing alive - a strong reference to the
    model or the VAE would pin gigabytes the context has already dropped - and a
    dead reference is a miss, never a false hit. A conditioning list takes no weak
    reference, so it travels by reference: the conditioning nodes replace the whole
    list, which is exactly what has to be noticed. The latent and the image are not
    here because they invalidate themselves (ctx_set_latent, ctx_set_image and the
    input overrides below drop the cached sample)."""
    inputs = {}
    for key in _SAMPLE_INPUTS:
        value = ctx.get(key)
        if value is None:
            continue
        try:
            inputs[key] = weakref.ref(value)
        except TypeError:
            inputs[key] = value
    return inputs


def _referent(value):
    return value() if isinstance(value, weakref.ReferenceType) else value


def _same_sample_inputs(cached, inputs):
    """Whether the cached sample was built from these very same objects."""
    refs = cached.get("inputs")
    if refs is None or refs.keys() != inputs.keys():
        return False
    for key, old in refs.items():
        if _referent(old) is not _referent(inputs[key]):
            return False
    return True


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
    res = f"{w}x{h} ({_mp(w, h)} Mp)" if w and h else "unknown"
    length_s = f" length={length} frames" if length else ""
    _log("start:", f"model={model_name} steps={_steps_display(steps_value, start_step, end_step)} "
          f"cfg={ctx.get('cfg')} sampler={ctx.get('sampler')} scheduler={ctx.get('scheduler')} "
          f"denoise={denoise} seed={seed} resolution={res}{length_s}")


def _log_finish(verbose, image, latent, t_start, vae=None):
    """Console log at the end of a run: total time, resulting resolution and frame/image count."""
    if not verbose:
        return
    w, h, _ = _image_dims(image)
    count_s = _image_count_s(image)
    if w is None:
        w, h, _ = _latent_dims(latent, vae)
        count_s = _latent_count_s(latent)
    res = f"{w}x{h} ({_mp(w, h)} Mp)" if w and h else "unknown"
    _log("finish:", f"total_time={time.time() - t_start:.2f}s resolution={res}{count_s}")


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


def _inpaint_regions(image, mask, opts):
    """Resize the mask to the image, process it (grow, invert) and compute crop
    regions per the crop-inpaint options with the crop node's methods (Image Crop
    By Mask (Batch)). Returns (mask, regions); regions is empty when there is
    nothing to inpaint. A region is (x0, y0, x1, y1, cx, cy, w, h, seg_mask,
    item): item is the batch item the region belongs to (None = the whole
    batch) - single mode with a batch mask and mask_indices makes each
    selected item its own region."""
    # Resize mask to image size if mismatch
    if mask.shape[1] != image.shape[1] or mask.shape[2] != image.shape[2]:
        mask = _resize_mask(mask, image.shape[2], image.shape[1])

    # Grow first, then invert (the crop node's method): the crop regions and
    # the inpaint mask both follow the processed mask
    mask = _process_mask(mask, opts.get("mask_grow", 0), opts.get("mask_invert", False))

    # Get mask regions (single or split)
    mask_mode = opts.get("mask_mode", "single")
    crop_factor = opts.get("crop_factor", 3.0)
    img_h, img_w = image.shape[1], image.shape[2]

    regions = []
    if mask_mode == "split":
        # Find disconnected regions: label the mask's connected components (8-
        # connectivity, so diagonally touching areas stay one region) and take
        # each one's bounding box (scipy ships with ComfyUI)
        mask_2d = mask.squeeze(0) if mask.dim() == 3 else mask
        labels, _ = ndi.label((mask_2d > 0.5).cpu().numpy(),
                              structure=ndi.generate_binary_structure(2, 2))
        for k, sl in enumerate(ndi.find_objects(labels), start=1):
            y, x = sl[0], sl[1]
            h, w = y.stop - y.start, x.stop - x.start
            if w < 4 or h < 4:
                continue
            cx, cy, x0, y0, cw, ch = _mask_box((x.start, y.start, w, h), img_w, img_h, crop_factor)
            # Isolate this segment so other segments inside the crop
            # window are not inpainted together with it
            seg_mask = mask_2d * torch.from_numpy(labels == k).to(mask.device)
            regions.append((x0, y0, x0 + cw, y0 + ch, cx, cy, cw, ch, seg_mask, None))
        # Filter by mask_indices if provided: "0-2, 5, 7" -> 0, 1, 2, 5, 7
        selected = parse_index_spec(opts.get("mask_indices", ""), len(regions))
        if selected is not None:
            regions = [regions[i] for i in selected]
    else:
        # Single region: the crop node's box for the full mask
        bbox = _mask_bbox(mask)
        if bbox is not None:
            cx, cy, x0, y0, w, h = _mask_box(bbox, img_w, img_h, crop_factor)
            regions.append((x0, y0, x0 + w, y0 + h, cx, cy, w, h, None, None))
        # A batch mask with mask_indices: each selected item is its own region
        # (its own mask's box) instead of the whole batch
        selected = parse_index_spec(opts.get("mask_indices", ""), mask.shape[0])
        if mask.shape[0] > 1 and selected:
            regions = []
            for k in selected:
                bbox = _mask_bbox(mask[k:k + 1])
                if bbox is None:
                    continue
                cx, cy, x0, y0, w, h = _mask_box(bbox, img_w, img_h, crop_factor)
                regions.append((x0, y0, x0 + w, y0 + h, cx, cy, w, h, None, k))

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
    model_obj = DifferentialDiffusion.execute(model_obj, strength=1.0)[0]
    return noise_mask, model_obj


def _color_match(image, reference, opts):
    """Match the image's color back to the reference with Transfer Color per the
    option's color match settings. Returns the image unchanged when the strength is 0
    (no color match option connected, or its strength is off)."""
    strength = float(opts.get("color_match_strength", 0.0))
    if strength <= 0:
        return image
    method = opts.get("color_match_method", "mkl_lab")
    # Transfer Color's Lab methods only handle RGB: match on the color channels
    # and carry any alpha through
    alpha = image[..., 3:4] if image.shape[-1] > 3 else None
    # ColorTransfer iterates its batch dim, so flatten a video's (B, F, H, W, C)
    # to (B*F, H, W, C) and repeat the still reference so every frame matches it
    if image.dim() == 5:
        b, f, h, w = image.shape[:4]
        target = image[..., :3].reshape(b * f, h, w, 3)
        reference = reference[..., :3].repeat_interleave(f, dim=0)
        matched, = ColorTransfer.execute(target, reference, method, {"source_stats": "per_frame"}, strength)
        matched = matched.reshape(b, f, h, w, 3)
    else:
        matched, = ColorTransfer.execute(image[..., :3], reference[..., :3], method, {"source_stats": "per_frame"}, strength)
    if alpha is not None:
        matched = torch.cat([matched, alpha.to(matched.device, matched.dtype)], dim=-1)
    return matched


def _encode_sample_decode(image, mask, vae, model_obj, seed, steps_value, cfg_value, sampler_obj,
                          sigmas_tensor, positive, negative, inpaint_opts,
                          tiled_above, tile_size, overlap, temporal_size, temporal_overlap,
                          decode=True, clear_after_model=False, clear_after_vae=False, clear_verbose=False,
                          verbose=False, travel_state=None, prompt_state=None, step_offset=0, latent=None,
                          dype_opt=None):
    """Shared sample step: encode the image (or use the given latent), sample it
    (masked when a mask is present, per the crop-inpaint options), and decode the
    result. Returns (image, model_obj); image is None when decode is off.

    The DyPE patch is applied to this step's own latent size and kept local: the
    returned model is the unpatched one, so a multi-step caller (iterative
    upscale, crop-inpaint) re-patches per step at that step's resolution and the
    context keeps the original model."""
    if image is not None:
        tiled = _tiled_for(image.shape[2], image.shape[1], tiled_above)
    else:
        w, h, _ = _latent_dims(latent, vae)
        tiled = _tiled_for(w, h, tiled_above) if w else False
    if latent is None:
        latent = encode_image(vae, image.float(), tiled, tile_size, overlap, temporal_size, temporal_overlap, verbose)
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
    # DyPE is sized to the grid being sampled, so it is patched per step
    dype_w, dype_h, _ = _latent_dims(latent, vae)
    step_model = apply_dype_patch(model_obj, dype_opt, dype_w, dype_h)
    callback = latent_preview.prepare_callback(step_model, steps_value)
    samples = _sample_with_travels(step_model, travel_state, prompt_state, noise, cfg_value, sampler_obj, sigmas_tensor,
                                    positive, negative, latent_samples, noise_mask, callback,
                                    not _comfy_utils.PROGRESS_BAR_ENABLED, seed, step_offset)
    # The model is done after sampling: free it before the decode so large
    # resolutions don't choke the VAE (Clear VRAM options)
    if clear_after_model:
        clear_vram("after model", clear_verbose)
    if not decode:
        return None, model_obj
    image = decode_latent(vae, {"samples": samples}, tiled, tile_size, overlap, temporal_size, temporal_overlap, verbose)
    if clear_after_vae:
        clear_vram("after vae", clear_verbose)
    return image, model_obj


def _crop_inpaint(ctx, opts, model_obj, seed, steps_value,
                  sampler_obj, denoise, cfg_value, positive, negative, decode, options_list=None,
                  clear_after_model=False, clear_after_vae=False, clear_verbose=False, verbose=False,
                  travel_state=None, prompt_state=None, dype_opt=None):
    """Crop image by mask bbox, resize, sample each crop, composite back.
    Each crop is sampled with the DyPE patch sized to that crop."""
    image = ctx["image"]
    mask = ctx["mask"]
    vae = ctx.get("vae")

    mask, regions = _inpaint_regions(image, mask, opts)

    opt_verbose = opts.get("verbose", False)
    if opt_verbose and opts.get("mask_mode", "single") == "split":
        _log("Crop-Inpaint", f"{len(regions)} masks")

    # Process each region
    megapixels = opts.get("megapixels", 0.0)
    scale_factor = opts.get("scale_factor", 1.0)
    multiple = opts.get("multiple", 32)
    method = opts.get("upscale_method", "bilinear")
    tiled_above, tile_size, overlap, temporal_size, temporal_overlap = _tiled_settings(options_list)

    result_image = image.clone()

    for idx, (x0, y0, x1, y1, cx, cy, w, h, seg_mask, item) in enumerate(regions):
        target_w, target_h = _resize_to_mp_scale(w, h, megapixels, scale_factor, multiple)
        src = result_image[item:item + 1] if item is not None else result_image
        region_mask = seg_mask.unsqueeze(0) if seg_mask is not None else (mask[item:item + 1] if item is not None else mask)
        crop_img, crop_mask, (sx0, sy0, src_w, src_h) = _cover_crop(
            src, region_mask, cx, cy, w, h, target_w, target_h, method)
        if opt_verbose:
            _log("Crop-Inpaint", f"mask {idx + 1}/{len(regions)} crop size={x1 - x0}x{y1 - y0} -> {target_w}x{target_h}")
        # The sigmas are sized to this crop: a Flux2 model gets its schedule
        # (get_schedule) from it
        sigmas_tensor = _calculate_sigmas(model_obj, ctx["scheduler"], steps_value, ctx["sampler"],
                                          denoise, target_w, target_h)
        refined_crop, model_obj = _encode_sample_decode(crop_img, crop_mask, vae, model_obj, seed, steps_value,
                                                        cfg_value, sampler_obj, sigmas_tensor, positive, negative, opts,
                                                        tiled_above, tile_size, overlap, temporal_size, temporal_overlap,
                                                        decode and vae is not None,
                                                        clear_after_model, clear_after_vae, clear_verbose, verbose,
                                                        travel_state=travel_state, prompt_state=prompt_state,
                                                        dype_opt=dype_opt)
        if refined_crop is not None:
            # Match the crop's color back to the original crop before pasting,
            # so the match is anchored to this region, not the whole image
            refined_crop = _color_match(refined_crop, crop_img, opts)
            if opt_verbose:
                _log("Crop-Inpaint", f"mask {idx + 1}/{len(regions)} composited {refined_crop.shape[2]}x{refined_crop.shape[1]} -> {src_w}x{src_h}")
            _paste_back(src, refined_crop, crop_mask, sx0, sy0, src_w, src_h)

    # Without a decode the VAE is done after the last region's encode: free it.
    if clear_after_vae and not (decode and vae is not None):
        clear_vram("after vae", clear_verbose)

    # Update context: the image was replaced, so the latent, mask and tiling
    # info built from the old one are stale
    ctx_set_sampled_image(ctx, result_image)

    return io.NodeOutput(ctx, None, result_image, None, vae, ctx.get("vae_audio"), options_list)


def _iterative_upscale_step(image, mask, target_w, target_h, method, upscale_model, vae, model_obj,
                            seed, steps_value, cfg_value, sampler_obj, sigmas_tensor, positive, negative,
                            inpaint_opts=None, tiled_above=None, tile_size=512, overlap=64,
                            temporal_size=64, temporal_overlap=8,
                            clear_after_model=False, clear_after_vae=False, clear_verbose=False, verbose=False,
                            travel_state=None, prompt_state=None, step_offset=0, dype_opt=None):
    """One iterative upscale step: scale image (and mask) to target size, encode, ksample, decode.
    With a mask, each step samples only the masked area (crop-inpaint options control the mask)."""
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
        image = _resize_image(image, target_w, target_h, method)

    if mask is not None and (mask.shape[-2] != target_h or mask.shape[-1] != target_w):
        mask = _resize_mask(mask, target_w, target_h)

    image, model_obj = _encode_sample_decode(image, mask, vae, model_obj, seed, steps_value, cfg_value, sampler_obj,
                                              sigmas_tensor, positive, negative, inpaint_opts,
                                              tiled_above, tile_size, overlap, temporal_size, temporal_overlap,
                                              clear_after_model=clear_after_model, clear_after_vae=clear_after_vae,
                                              clear_verbose=clear_verbose, verbose=verbose,
                                              travel_state=travel_state, prompt_state=prompt_state, step_offset=step_offset,
                                              dype_opt=dype_opt)
    return image, mask


def _iterative_upscale(ctx, opts, options_list, model_obj, seed, steps_value,
                       cfg_value, sampler_obj, positive, negative,
                       start_step=0.0, end_step=10000.0, leftover_noise=False,
                       inpaint_opts=None,
                       clear_after_model=False, clear_after_vae=False, clear_verbose=False, verbose=False,
                       travel_state=None, prompt_state=None, steps=0, dype_opt=None):
    """Iterative pixel-space upscale along a linear scale path (simple step mode).

    Every step samples with the DyPE patch sized to that step's target size
    (dype_opt), so the resolution the patch advertises always matches the grid
    actually being denoised.

    A target size (target_size_image in the options) overrides the upscale
    factor: the final resolution is the provided size, and the per-step scale
    path runs to it.
    State (next_step, base size) lives in the options dict; on first run it is
    initialized from the current image and the options returned with the
    updated next_step, so the output can be fed back in for further steps.
    A mask in the context is resized with the image and applied on every step
    (inpaint mode and mask scaling come from the crop-inpaint options) - it
    only applies once an image exists.
    With no image in the context, the 1st step is the basic generation (the
    context's latent sampled as-is, denoise overridden to 1.0, base steps, no
    mask); the image it generates is then treated as if it had been provided
    (it becomes the color match reference) and the remaining steps ramp
    start_denoise -> target_denoise.
    In total mode, after the final planned step the result's color is matched
    back to the original image when a color match option is connected.
    """
    image = ctx.get("image")
    mask = ctx.get("mask")
    vae = ctx["vae"]
    # VAEs round-trip exactly only at multiples of their spatial downscale
    # ratio - align the per-step target sizes to it
    w_ratio, h_ratio = _latent_downscale(vae, None)
    scheduler_name = ctx["scheduler"]
    sampler_name = ctx["sampler"]

    factor = float(opts.get("upscale_factor", 2.0))
    # The target size image overrides the upscale factor: the final
    # resolution is the provided size (aligned to the VAE downscale ratio),
    # bypassing the base * factor calculation
    final_w = int(opts["target_w"]) if opts.get("target_w") is not None else None
    final_h = int(opts["target_h"]) if opts.get("target_h") is not None else None
    if final_w is not None:
        final_w = max(1, int(round(final_w / w_ratio)) * w_ratio)
        final_h = max(1, int(round(final_h / h_ratio)) * h_ratio)
    total_steps = max(int(opts.get("steps", 3)), 1)
    start_denoise = float(opts.get("start_denoise", 1.0))
    target_denoise = float(opts.get("target_denoise", 0.6))
    method = opts.get("upscale_method", "bilinear")
    mode = opts.get("mode", "total")
    upscale_model = opts.get("upscale_model")
    opt_verbose = opts.get("verbose", False)
    tiled_above, tile_size, overlap, temporal_size, temporal_overlap = _tiled_settings(options_list)

    # No image in the context: the 1st step is the basic generation (denoise
    # overridden to 1.0, base steps, no mask - the mask only applies once an
    # image exists); make sure the context has a latent to sample
    no_image = image is None
    if no_image:
        GibbyContext.evaluate(ctx)
    # Steps of the basic-gen 1st step: the widget's steps, else the base steps
    # (the refiner steps the upscale steps use only apply to re-sampling an
    # existing image)
    gen_steps = steps_value if steps > 0 else (ctx.get("steps") or steps_value)

    # The engine's output cache can hand back the same options object on
    # repeated runs - never mutate it, the step state travels on a copy
    opts = dict(opts)
    if "next_step" not in opts:
        # First run: record the initial size (the image, or the latent when the
        # 1st step is the basic generation) and reset the step counter
        opts["next_step"] = 0
        if image is not None:
            opts["base_w"] = image.shape[2]
            opts["base_h"] = image.shape[1]
        else:
            w, h, _ = _latent_dims(ctx["latent"], vae)
            opts["base_w"] = w
            opts["base_h"] = h
            # The 1st step is the basic generation: the denoise ramp of the
            # upscale steps starts from the 2nd one
            opts["gen_first"] = True

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

    per_iteration = travel_state is not None and travel_state.get("per_iteration")
    # Set when the flow started with no image: the 1st step was the basic
    # generation, so the denoise ramp of the upscale steps covers 2..N
    gen_first = opts.get("gen_first", False)

    for i in step_indices:
        # Every pass at the same size (upscale_factor 1, or a target size
        # equal to the base) would sample identical noise - vary the seed per
        # iteration
        same_size = factor == 1.0 or (final_w is not None and final_w == base_w and final_h == base_h)
        step_seed = seed + i if same_size else seed
        if no_image and i == 1:
            # Basic generation: sample the context's latent as-is (no mask -
            # it only applies once an image exists), full denoise, no upscale
            gen_step = True
            scale = 1.0
            target_w, target_h = base_w, base_h
            denoise_step = 1.0
            run_steps = gen_steps
        else:
            gen_step = False
            if final_w is not None:
                # Target size: the linear scale path runs to the provided size
                scale_w = 1.0 + (final_w / base_w - 1.0) * i / total_steps
                scale_h = 1.0 + (final_h / base_h - 1.0) * i / total_steps
                target_w = max(1, int(round(base_w * scale_w / w_ratio)) * w_ratio)
                target_h = max(1, int(round(base_h * scale_h / h_ratio)) * h_ratio)
                scale = scale_w
            else:
                scale = 1.0 + (factor - 1.0) * i / total_steps
                target_w = max(1, int(round(base_w * scale / w_ratio)) * w_ratio)
                target_h = max(1, int(round(base_h * scale / h_ratio)) * h_ratio)
            # Denoise ramps from start to target across the planned steps, then
            # holds at target; the basic-gen 1st step (no image) is 1.0, so the
            # ramp covers the rest
            if i > total_steps:
                denoise_step = target_denoise
            elif gen_first:
                denoise_step = target_denoise if total_steps == 2 else start_denoise + (target_denoise - start_denoise) * (i - 2) / (total_steps - 2)
            elif total_steps == 1:
                denoise_step = start_denoise
            else:
                denoise_step = start_denoise + (target_denoise - start_denoise) * (i - 1) / (total_steps - 1)
            run_steps = steps_value
        # The sigmas are sized to the step's image: a Flux2 model gets its
        # schedule (get_schedule) from it
        sigmas_tensor = _calculate_sigmas(model_obj, scheduler_name, run_steps, sampler_name, denoise_step, target_w, target_h)
        sigmas_tensor, skip = _apply_start_end_steps(sigmas_tensor, run_steps, start_step, end_step, leftover_noise)
        if skip:
            # Start step beyond the available steps: keep the image, advance the
            # step (a skipped basic gen keeps the latent and the step so the
            # next run retries it)
            if not gen_step:
                next_step = i
            continue
        if opt_verbose:
            _log("Iterative Upscale", f"step {i}/{total_steps} scale={scale:.2f} size={target_w}x{target_h} denoise={denoise_step:.3f}")
        if per_iteration:
            # One strength for the whole iteration: all its sampling steps share it
            _update_travel_step(travel_state, i - 1)
        if gen_step:
            # The context's latent as-is; prompt travel is planned over the
            # upscale steps' steps, so the basic gen samples the context conds
            # as-is
            image, model_obj = _encode_sample_decode(None, None, vae, model_obj, step_seed, run_steps, cfg_value, sampler_obj,
                                                      sigmas_tensor, positive, negative, None,
                                                      tiled_above, tile_size, overlap, temporal_size, temporal_overlap,
                                                      clear_after_model=clear_after_model, clear_after_vae=clear_after_vae,
                                                      clear_verbose=clear_verbose, verbose=verbose,
                                                      travel_state=travel_state, prompt_state=None,
                                                      step_offset=_resolve_step_range(run_steps, start_step, end_step)[0],
                                                      latent=ctx["latent"], dype_opt=dype_opt)
            # Treat the generated image as if it had been provided: it becomes
            # the color match reference for the final step
            if original_image is None:
                original_image = image
        else:
            image, mask = _iterative_upscale_step(image, mask, target_w, target_h, method, upscale_model, vae, model_obj,
                                                    step_seed, steps_value, cfg_value, sampler_obj, sigmas_tensor, positive, negative, inpaint_opts,
                                                    tiled_above, tile_size, overlap, temporal_size, temporal_overlap,
                                                    clear_after_model, clear_after_vae, clear_verbose, verbose,
                                                    travel_state, prompt_state, step_offset,
                                                    dype_opt=dype_opt)
        next_step = i

    # After the final planned step, match the result's color back to the
    # original image (the provided one, or the basic-gen 1st step's image
    # when none was)
    if mode == "total" and next_step == total_steps and original_image is not None:
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
    if image is not None:
        # The image was replaced: the latent, the cached sample, and the tiling
        # info of the old one are all stale now.
        drop_image_derived(ctx)
        ctx.pop("tiling_info", None)

    return io.NodeOutput(ctx, None, image, None, vae, ctx.get("vae_audio"), options_list)


def _iterative_inpaint_upscale(ctx, inpaint_opts, iterative_opts, options_list, model_obj, seed, steps_value,
                               cfg_value, sampler_obj, positive, negative,
                               start_step=0.0, end_step=10000.0, leftover_noise=False,
                               clear_after_model=False, clear_after_vae=False, clear_verbose=False, verbose=False,
                               travel_state=None, prompt_state=None, dype_opt=None):
    """Iterative inpaint upscale: crop the mask regions (crop-inpaint options), iteratively
    upscale each crop with the mask applied on every step, then composite the refined crops
    back into the original-resolution image - the result keeps the original size.
    Each crop's steps sample with the DyPE patch sized to that step."""
    image = ctx["image"]
    vae = ctx["vae"]
    mask, regions = _inpaint_regions(image, ctx["mask"], inpaint_opts)

    opt_verbose = inpaint_opts.get("verbose", False)
    if opt_verbose and inpaint_opts.get("mask_mode", "single") == "split":
        _log("Crop-Inpaint", f"{len(regions)} masks")

    # The result stays at the original resolution: each crop is refined by the
    # iterative upscale, then composited back into its original region (the
    # refined crop is downscaled to fit). The original image is not upscaled.
    full = image.clone()

    megapixels = inpaint_opts.get("megapixels", 0.0)
    scale_factor = inpaint_opts.get("scale_factor", 1.0)
    multiple = inpaint_opts.get("multiple", 32)
    crop_method = inpaint_opts.get("upscale_method", "bilinear")

    out_options = options_list
    for idx, (x0, y0, x1, y1, cx, cy, w, h, seg_mask, item) in enumerate(regions):
        target_w, target_h = _resize_to_mp_scale(w, h, megapixels, scale_factor, multiple)
        src = image[item:item + 1] if item is not None else image
        full_src = full[item:item + 1] if item is not None else full
        region_mask = seg_mask.unsqueeze(0) if seg_mask is not None else (mask[item:item + 1] if item is not None else mask)
        crop_img, crop_mask, (sx0, sy0, src_w, src_h) = _cover_crop(
            src, region_mask, cx, cy, w, h, target_w, target_h, crop_method)
        if opt_verbose:
            _log("Crop-Inpaint", f"mask {idx + 1}/{len(regions)} crop size={x1 - x0}x{y1 - y0} -> {target_w}x{target_h}")
        crop_ctx = {"image": crop_img, "mask": crop_mask, "vae": vae,
                    "scheduler": ctx["scheduler"], "sampler": ctx["sampler"]}
        # Pass the crop (image + mask) to the iterative upscale
        out = _iterative_upscale(crop_ctx, iterative_opts, options_list, model_obj, seed,
                                 steps_value, cfg_value, sampler_obj, positive, negative,
                                 start_step, end_step, leftover_noise, inpaint_opts=inpaint_opts,
                                 clear_after_model=clear_after_model, clear_after_vae=clear_after_vae,
                                 clear_verbose=clear_verbose, verbose=verbose, travel_state=travel_state, prompt_state=prompt_state,
                                 dype_opt=dype_opt)
        out_options = out[6]
        refined = out[2]
        # Composite the refined crop back into its original-resolution region
        if opt_verbose:
            _log("Crop-Inpaint", f"mask {idx + 1}/{len(regions)} composited {refined.shape[2]}x{refined.shape[1]} -> {src_w}x{src_h}")
        _paste_back(full_src, refined, crop_ctx.get("mask"), sx0, sy0, src_w, src_h)

    if opt_verbose:
        _log("Crop-Inpaint", f"composited {len(regions)} region(s) into {full.shape[2]}x{full.shape[1]} (final image)")

    # Update context: the image was replaced, so the latent, mask and tiling
    # info built from the old one are stale
    ctx_set_sampled_image(ctx, full)

    return io.NodeOutput(ctx, None, full, None, vae, ctx.get("vae_audio"), out_options)


def _calculate_sigmas(model, scheduler_name, steps, sampler_name, denoise=1.0, width=None, height=None):
    """Calculate sigmas from scheduler name and step count, respecting denoise level.
    A Flux2 model uses the Flux2 schedule (get_schedule) for the size instead, its
    shift needs the sequence length; a Flux3 model has a constant reference shift
    in its own model sampling, so it stays on the context's scheduler.
    denoise always takes the schedule's low-noise tail, like the native SplitSigmasDenoise."""
    if _is_flux2(model) and get_schedule is not None:
        if not width or not height:
            width, height = 1024, 1024
        seq_len = (width * height) / (16 * 16)
        sigmas = get_schedule(steps, round(seq_len)).to(comfy.model_management.get_torch_device())
        if denoise is not None and denoise <= 0.0:
            sigmas = torch.FloatTensor([])
        elif denoise is not None and denoise < 0.9999:
            # A low denoise on a short schedule still samples one step
            total_steps = max(round(steps * denoise), 1)
            sigmas = sigmas[-(total_steps + 1):]
        return sigmas

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
    actual_start_step = _resolve_step(steps_value, start_step)
    actual_end_step = _resolve_step(steps_value, end_step)

    if actual_end_step < (len(sigmas_tensor) - 1):
        sigmas_tensor = sigmas_tensor[:actual_end_step + 1]
        if not leftover_noise:
            sigmas_tensor[-1] = 0

    skip = actual_start_step >= (len(sigmas_tensor) - 1)
    if not skip and actual_start_step > 0:
        sigmas_tensor = sigmas_tensor[actual_start_step:]

    return sigmas_tensor, skip


def _dype_model(model_obj, ctx, dype_opt, verbose=False):
    """The model with the DyPE patch applied for the single-sample path
    (dype/sega). The patch stays on this local clone - the context and
    the outputs keep the original unpatched model, and a context with no size
    leaves the model untouched."""
    if dype_opt is None or dype_opt.get("method") not in DYPE_PATCH_METHODS:
        return model_obj
    w, h, _ = _latent_dims(ctx.get("latent"), ctx.get("vae"))
    if not w:
        w, h = ctx_size(ctx)
    return apply_dype_patch(model_obj, dype_opt, int(w), int(h))


def _require_sampling_params(steps_value, cfg_value, sampler_name, scheduler_name,
                            need_steps=True, need_sampler=True, need_scheduler=True):
    """Check the sampling parameters resolved from the context. This node has no
    defaults: a context that carries none is an error naming what to connect,
    instead of a silent 20-step euler run.
    need_* is False when a SAMPLER/SIGMAS object input replaces that parameter."""
    missing = []
    if need_steps and not steps_value:
        missing.append("steps")
    if cfg_value is None:
        missing.append("cfg")
    if need_sampler and not sampler_name:
        missing.append("sampler")
    if need_scheduler and not scheduler_name:
        missing.append("scheduler")
    if missing:
        raise ValueError(f"The context has no {' or '.join(missing)}. Sampling parameters come from the "
                         f"context: connect a Context, Context Loader, or Sampling Parameters node "
                         f"(this node's steps widget can supply steps on its own).")
    if need_sampler and sampler_name not in comfy.samplers.KSampler.SAMPLERS:
        raise ValueError(f"The context's sampler '{sampler_name}' is not a sampler ComfyUI offers. "
                         f"Pick one of the Context node's sampler options.")
    if need_scheduler and scheduler_name not in comfy.samplers.KSampler.SCHEDULERS:
        raise ValueError(f"The context's scheduler '{scheduler_name}' is not a scheduler ComfyUI offers. "
                         f"Pick one of the Context node's scheduler options.")
    return steps_value, cfg_value, sampler_name, scheduler_name


def _resolve_early_params(ctx, steps):
    """Resolve steps/cfg/sampler for the pre-evaluate sampling paths."""
    base_steps = ctx.get("steps") or 0
    if steps > 0:
        steps_value = steps
    elif ctx.get("step_refiner", 0) > 0:
        steps_value = ctx["step_refiner"]
    else:
        steps_value = base_steps
    steps_value, cfg_value, sampler_name, _ = _require_sampling_params(
        steps_value, ctx.get("cfg"), ctx.get("sampler"), ctx.get("scheduler"))
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
                "mask, sampler, or sigmas as needed. Automatically encodes images and updates context. "
                "DyPE options patch the model per sample - see DyPE options."
            ),
            inputs=[
                _CONTEXT_TYPE.Input("context"),
                io.Model.Input("model", optional=True),
                io.Latent.Input("latent", optional=True),
                io.Image.Input("image", optional=True),
                io.Mask.Input("mask", optional=True),
                io.Sampler.Input("sampler", optional=True),
                io.Sigmas.Input("sigmas", optional=True),
                _KSAMPLER_OPTIONS_TYPE.Input("options", optional=True, tooltip="Connect an options node (Crop-Inpaint, Iterative Options, Color Match, Tiled VAE, Clear VRAM, DyPE) or Merge KSampler Options"),
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
        clear_opt = _find_option(options_list, "clear_vram")
        clear_at_start = clear_opt.get("at_start", True) if clear_opt else False
        clear_after_finish = clear_opt.get("after_finish", True) if clear_opt else False
        clear_after_model = clear_opt.get("after_model", True) if clear_opt else False
        clear_after_vae = clear_opt.get("after_vae", True) if clear_opt else False
        clear_verbose = clear_opt.get("verbose", False) if clear_opt else False
        if clear_at_start:
            clear_vram("at start", clear_verbose)

        # Start with a copy of the context (never mutate the input) and apply overrides
        ctx = ctx_from(context)
        # The raw prompts drive native prompt travel (Forge [before:after:step] groups)
        pos_prompt_text = ctx.get("positive_prompt") or ""
        neg_prompt_text = ctx.get("negative_prompt") or ""

        if model is not None:
            ctx["model"] = model

        if latent is not None:
            # A connected latent replaces the context's, so the cached sample was
            # built from a different one
            ctx_set_latent(ctx, latent)

        if image is not None:
            # A connected image takes priority over a latent inherited from the
            # context: drop that latent and the sample cached from the old image so
            # the image is encoded and sampled fresh, and put a directly-connected
            # latent input back on top - it still wins over the inherited latent
            ctx_set_image(ctx, image)
            if latent is not None:
                ctx["latent"] = latent

        # A directly-connected mask overrides the context's before evaluation.
        # The input is untrusted (an image can be bridged into it) - keep only
        # usable masks, treat anything else as absent.
        mask = _normalize_mask(mask)
        if mask is not None:
            ctx["mask"] = mask

        # Remember which objects the sample is about to be built from for the
        # result cache; taken here, before evaluate() encodes or conditions
        # anything, so a re-run of this node compares the same values
        sample_inputs = _sample_inputs(ctx)

        # Tiled VAE settings come from the options list (Tiled VAE options node):
        # encode/decode use the tiled variants for sizes at or above above_megapixels
        tiled_above, tile_size, overlap, temporal_size, temporal_overlap = _tiled_settings(options)

        # Apply image-based options (crop-inpaint, iterative upscale) before
        # evaluate(), which would otherwise encode the image for nothing
        inpaint_opt = _find_option(options_list, "inpaint")
        iterative_opt = _find_option(options_list, "iterative_upscale")
        # DyPE (soft dependency): patches the model per sample; absent when the
        # options node is not wired
        dype_opt = _find_option(options_list, "dype")
        # The color match option enables and tunes color matching for
        # crop-inpaint and the iterative upscale; without it no matching runs
        color_opt = _find_option(options_list, "color_match")
        if color_opt is not None:
            if inpaint_opt is not None:
                inpaint_opt = _with_color_match(inpaint_opt, color_opt)
            if iterative_opt is not None:
                iterative_opt = _with_color_match(iterative_opt, color_opt)
        # The iteration count per_iteration travels ramp across
        iter_steps = max(int(iterative_opt.get("steps", 3)), 1) if iterative_opt is not None else None
        has_image = ctx.get("image") is not None
        has_mask = ctx.get("mask") is not None
        has_vae = ctx.get("vae") is not None
        # Crop-inpaint with an empty mask: no regions to inpaint, skip sampling
        # entirely and pass the image through
        if inpaint_opt is not None and has_image and has_mask and not _inpaint_regions(ctx["image"], ctx["mask"], inpaint_opt)[1]:
            ctx.pop("latent", None)
            ctx.pop("mask", None)
            if clear_after_finish:
                clear_vram("after finish", clear_verbose)
            if verbose:
                _log("Crop-Inpaint skipped", "the mask is empty")
            return io.NodeOutput(ctx, None, ctx["image"], None, ctx.get("vae"), ctx.get("vae_audio"), options_list)
        result = None
        if inpaint_opt is not None and iterative_opt is not None and has_image and has_mask and has_vae:
            # Crop first, then pass each crop (image + mask) to the iterative
            # upscale with the mask applied on every step, composite back
            model_obj = ctx.get("model")
            steps_value, cfg_value, sampler_name, sampler_obj = _resolve_early_params(ctx, steps)
            _apply_travel_clip_loras(ctx, travel_opts, steps_value)
            ensure_conditioning(ctx)
            travel_state = _prepare_lora_travels(model_obj, travel_opts, steps_value, iteration_steps=iter_steps)
            prompt_state = _prepare_prompt_travels(ctx, pos_prompt_text, neg_prompt_text, steps_value, verbose)
            _log_start(verbose, ctx, seed, steps_value, denoise, start_step, end_step)
            result = _iterative_inpaint_upscale(ctx, inpaint_opt, iterative_opt, options_list, model_obj, seed,
                                                 steps_value, cfg_value, sampler_obj, ctx.get("positive"), ctx.get("negative"),
                                                 start_step, end_step, leftover_noise,
                                                 clear_after_model, clear_after_vae, clear_verbose, verbose=verbose,
                                                 travel_state=travel_state, prompt_state=prompt_state, dype_opt=dype_opt)
        elif iterative_opt is not None and has_vae:
            # No crop-inpaint (or no mask for it): iterative on the existing image,
            # masked on every step when a mask is present. With no image in the
            # context the 1st step is the basic generation (denoise 1.0), and the
            # image it generates is treated as if it had been provided
            model_obj = ctx.get("model")
            steps_value, cfg_value, sampler_name, sampler_obj = _resolve_early_params(ctx, steps)
            _apply_travel_clip_loras(ctx, travel_opts, steps_value)
            ensure_conditioning(ctx)
            travel_state = _prepare_lora_travels(model_obj, travel_opts, steps_value, iteration_steps=iter_steps)
            prompt_state = _prepare_prompt_travels(ctx, pos_prompt_text, neg_prompt_text, steps_value, verbose)
            _log_start(verbose, ctx, seed, steps_value, denoise, start_step, end_step)
            result = _iterative_upscale(ctx, iterative_opt, options_list, model_obj, seed,
                                         steps_value, cfg_value, sampler_obj, ctx.get("positive"), ctx.get("negative"),
                                         start_step, end_step, leftover_noise,
                                         clear_after_model=clear_after_model, clear_after_vae=clear_after_vae,
                                         clear_verbose=clear_verbose, verbose=verbose,
                                         travel_state=travel_state, prompt_state=prompt_state, steps=steps,
                                         dype_opt=dype_opt)
        elif inpaint_opt is not None and has_image and has_mask:
            # Crop-inpaint only: crop, sample each crop, composite back (an empty
            # mask has no regions and is skipped above)
            model_obj = ctx.get("model")
            steps_value, cfg_value, sampler_name, sampler_obj = _resolve_early_params(ctx, steps)
            _apply_travel_clip_loras(ctx, travel_opts, steps_value)
            ensure_conditioning(ctx)
            travel_state = _prepare_lora_travels(model_obj, travel_opts, steps_value)
            prompt_state = _prepare_prompt_travels(ctx, pos_prompt_text, neg_prompt_text, steps_value, verbose)
            _log_start(verbose, ctx, seed, steps_value, denoise, start_step, end_step)
            result = _crop_inpaint(ctx, inpaint_opt, model_obj, seed, steps_value,
                                   sampler_obj, denoise, cfg_value, ctx.get("positive"), ctx.get("negative"), decode,
                                   options_list, clear_after_model, clear_after_vae, clear_verbose, verbose,
                                   travel_state=travel_state, prompt_state=prompt_state, dype_opt=dype_opt)
        if result is not None:
            if clear_after_finish:
                clear_vram("after finish", clear_verbose)
            _log_finish(verbose, result[2], result[1], t_start, ctx.get("vae"))
            return result

        # Resolve sampling parameters from context with widget overrides
        model_obj = ctx.get("model")
        if ctx.get("seed") is None:
            ctx["seed"] = seed

        # Denoise: use widget value only if image exists, otherwise force 1.0
        has_image = ctx.get("image") is not None
        denoise_value = denoise if has_image else 1.0
        # The provided image is the color match reference for the final decode
        original_image = ctx.get("image") if color_opt is not None else None

        # Steps: widget wins; with an image present, prefer refiner steps from context -
        # unless a sub-range of the base steps was explicitly requested via start/end step.
        base_steps = ctx.get("steps") or 0
        if steps > 0:
            steps_value = steps
        elif has_image and ctx.get("step_refiner", 0) > 0 and not (start_step != 0 or end_step < base_steps):
            steps_value = ctx["step_refiner"]
        else:
            steps_value = ctx.get("steps")

        # A SAMPLER object input replaces the context's sampler, and a SIGMAS object
        # replaces the schedule, so those parameters are not needed then
        steps_value, cfg_value, sampler_name, scheduler_name = _require_sampling_params(
            steps_value, ctx.get("cfg"), ctx.get("sampler"), ctx.get("scheduler"),
            need_steps=sigmas is None, need_sampler=sampler is None, need_scheduler=sigmas is None)

        # Lora Travel options: the traveling clip loras are applied once, at the
        # strength the loras enter with, before the conditioning is encoded from clip
        run_steps = len(sigmas) - 1 if sigmas is not None else steps_value
        _apply_travel_clip_loras(ctx, travel_opts, run_steps)

        # Fill in derived values on demand (latent from image, conditioning, width/height).
        # The image->latent encode runs inside evaluate(), tiled per the threshold;
        # time that call when it will run.
        encode_will_run = ctx.get("latent") is None and ctx.get("image") is not None and ctx.get("vae") is not None
        orig_size = None
        if encode_will_run:
            w, h, _ = _image_dims(ctx["image"])
            # The VAE encode snaps the image to its downscale ratio; remember the
            # original size to stretch the decode back to it
            orig_size = (w, h)
            # The VAE truncates to that snapped size instead of downsampling, so
            # pre-scale first: the encode covers the whole scene and the post-decode
            # stretch back to the original size is geometrically exact
            dw, dh = _latent_downscale(ctx["vae"], None)
            sw, sh = w // dw * dw, h // dh * dh
            if (sw, sh) != (w, h):
                if verbose:
                    _log("pre-scaled", f"{w}x{h} ({_mp(w, h)} Mp) -> {sw}x{sh} ({_mp(sw, sh)} Mp) for VAE encode")
                ctx["image"] = _resize_image(ctx["image"], sw, sh)
                w, h = sw, sh
            encode_tiled = _tiled_for(w, h, tiled_above)
            if verbose:
                _log("VAE encode starting", f"{w}x{h} ({_mp(w, h)} Mp){_image_count_s(ctx['image'])}")
        else:
            encode_tiled = False
        t0 = time.time()
        GibbyContext.evaluate(ctx, tiled=encode_tiled, tile_size=tile_size, overlap=overlap,
                               temporal_size=temporal_size, temporal_overlap=temporal_overlap)
        if verbose and encode_will_run:
            _log("VAE encode took", f"{time.time() - t0:.2f}s")

        # Apply mask only if it came from the input (not context) and latent exists.
        if mask is not None and ctx.get("latent") is not None and "noise_mask" not in ctx["latent"]:
            ctx["latent"], = SetLatentNoiseMask().set_mask(ctx["latent"], mask)

        # DyPE model patch: after evaluate, so it is sized to the latent that is
        # actually about to be sampled
        model_obj = _dype_model(model_obj, ctx, dype_opt, verbose)

        _log_start(verbose, ctx, seed, steps_value, denoise_value, start_step, end_step)

        # The final decode's tiled VAE is decided by the latent's size
        w, h, _ = _latent_dims(ctx.get("latent"), ctx.get("vae"))
        tiled_decode = _tiled_for(w, h, tiled_above) if w else False

        # Cache check: if this context already has a sampled result built from
        # these very same objects and parameters, return it instead of re-sampling.
        # This handles the case where the latent output wasn't connected during the
        # first run (result discarded by the engine) but is connected now.
        travel_sig = _travel_signature(travel_opts, run_steps)
        # DyPE: scalars only, so the comparison never touches a model object
        dype_sig = None if dype_opt is None else {k: v for k, v in dype_opt.items()
                                                  if isinstance(v, (bool, int, float, str))}
        prompt_sig = _prompt_travel_signature(pos_prompt_text, neg_prompt_text, run_steps)
        cached = ctx.get("_kctx_sampled")
        if cached is not None:
            if (_same_sample_inputs(cached, sample_inputs)
                    and cached.get("seed") == seed and cached.get("steps") == steps_value
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
                    and cached.get("prompt_travel") == prompt_sig
                    and cached.get("dype") == dype_sig):
                out_latent = cached["out_latent"]
                decoded_image = ctx.get("image")
                decoded_audio = ctx.get("audio")
                if clear_after_finish:
                    clear_vram("after finish", clear_verbose)
                if verbose:
                    _log("cache hit", "reusing sampled result")
                _log_finish(verbose, decoded_image, out_latent, t_start, ctx.get("vae"))
                return io.NodeOutput(ctx, out_latent, decoded_image, decoded_audio, ctx.get("vae"), ctx.get("vae_audio") or ctx.get("audio_vae"), options)

        # Resolve sampler and sigmas
        if sampler is not None:
            # Use provided SAMPLER object directly
            sampler_obj = sampler
        else:
            # Create sampler from context's sampler string
            sampler_obj = comfy.samplers.sampler_object(sampler_name)

        if sigmas is not None:
            # Use provided SIGMAS tensor directly (ignore start/end steps)
            sigmas_tensor = sigmas
            use_start_end_steps = False
        else:
            # The Flux2 schedule and the regular schedulers share the denoise rule:
            # the schedule is built for the full step count, denoise takes its
            # low-noise tail, and start/end steps slice the result below
            sigmas_width, sigmas_height = ctx_size(ctx)
            sigmas_tensor = _calculate_sigmas(model_obj, scheduler_name, steps_value, sampler_name,
                                              denoise_value, sigmas_width, sigmas_height)
            use_start_end_steps = True

        # Prompt travel (native): when a raw prompt carries a [before:after:step]
        # group, plan the per-step prompts and encode the unique ones grafted onto
        # the context's conditionings (reference payloads preserved); constant
        # prompts replace the context conditioning, switching ones travel per step
        prompt_state = _prepare_prompt_travels(ctx, pos_prompt_text, neg_prompt_text, run_steps, verbose)

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
            clear_vram("after model", clear_verbose)

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
                decoded_image = decode_latent(ctx["vae"], out_latent, tiled_decode, tile_size, overlap, temporal_size, temporal_overlap, verbose)
                # The VAE encode snapped the image to its downscale ratio; stretch
                # the decoded result back to the original image size
                if decoded_image is not None and orig_size is not None:
                    dw, dh, _ = _image_dims(decoded_image)
                    if (dw, dh) != orig_size:
                        if verbose:
                            _log("stretched", f"{dw}x{dh} ({_mp(dw, dh)} Mp) -> {orig_size[0]}x{orig_size[1]} ({_mp(orig_size[0], orig_size[1])} Mp)")
                        decoded_image = _resize_image(decoded_image, *orig_size)

                # Match the decoded result's color back to the original image
                if decoded_image is not None and original_image is not None:
                    decoded_image = _color_match(decoded_image, original_image, color_opt)

            # Decode to audio for context update and output (overrides any existing context.audio)
            if ctx.get("vae_audio") is not None and vae_decode_audio is not None:
                if verbose:
                    _log("VAE audio decode starting")
                t0 = time.time()
                decoded_audio = vae_decode_audio(ctx["vae_audio"], out_latent)
                if verbose:
                    _log("VAE audio decode took", f"{time.time() - t0:.2f}s")

            if clear_after_vae:
                clear_vram("after vae", clear_verbose)
        elif clear_after_model or clear_after_vae:
            # No decode: the VAE is done with the pre-sample encode, so both
            # are free to go.
            clear_vram("after model and vae", clear_verbose)

        # Update context after sampling: the image is replaced, so the tiling
        # info of the old one (if any) is stale
        ctx.pop("mask", None)    # Remove mask
        ctx.pop("tiling_info", None)

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
            "inputs": sample_inputs,
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
            "dype": dype_sig,
        }

        if clear_after_finish:
            clear_vram("after finish", clear_verbose)

        _log_finish(verbose, decoded_image, out_latent, t_start, ctx.get("vae"))
        return io.NodeOutput(ctx, out_latent, decoded_image, decoded_audio, ctx.get("vae"), ctx.get("vae_audio") or ctx.get("audio_vae"), options)

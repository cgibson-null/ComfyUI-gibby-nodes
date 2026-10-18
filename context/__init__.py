"""
Context node
------------
A general-purpose context bundler (inspired by rgthree's "Context Big"). It
exposes a wide set of inputs/outputs so values can be collected into one
CONTEXT object and passed along a workflow, then unpacked again downstream.

Every input is optional: connect what you need, or link a base context in to
inherit its values for any field left unconnected. This node never generates
anything on its own - use Context Loader for model loading and on-demand
generation (conditioning from prompts, latent from image).

The evaluate() staticmethod fills missing derived values (conditioning,
latent) from what a context already holds; it is shared with Context Loader
and KSampler (Context).
"""

import datetime
import gc
import logging
import os
import re
import time
import weakref
from collections import OrderedDict

import torch
import comfy.samplers
import comfy.model_management
import comfy.model_base
from nodes import ConditioningZeroOut, VAEEncode, VAEEncodeTiled, VAEDecode, VAEDecodeTiled, SetLatentNoiseMask
from comfy_extras.nodes_audio import VAEEncodeAudio
from comfy_extras.nodes_lt import LTXVConcatAVLatent
from comfy_api.latest import io

from ..lora_loader import _apply_lora, _lora_stack, iter_lora_stack
from .prompt_travel import _travel_step_text

_CONTEXT_TYPE = io.Custom("CONTEXT")
# The crop nodes' originals/mask/rects payload, carried in the context's crop_info
_CROP_INFO_TYPE = io.Custom("GIBBY_CROP_INFO")


def ctx_from(context):
    """A fresh context dict: a copy when context is a dict, else an empty one.
    Nodes always work on this copy, so the context they received is never
    mutated (a mutated input would leak into earlier nodes on re-runs)."""
    return dict(context) if isinstance(context, dict) else {}


def drop_image_derived(ctx):
    """Drop the latent and the cached sample after the context's image changes:
    both were built from the image it carried before, so they're now stale -
    the sampler would otherwise reuse the old latent or return the cached
    result instead of encoding/sampling the new image."""
    ctx.pop("latent", None)
    ctx.pop("_kctx_sampled", None)


def ctx_set_image(ctx, image):
    """Override the context's image and drop the latent and cached sample built
    from the old one, so the sampler re-encodes the new image instead of reusing
    them."""
    ctx["image"] = image
    drop_image_derived(ctx)


def ctx_set_latent(ctx, latent):
    """Override the context's latent and drop the cached sample built from the old
    one, so the sampler samples the new latent instead of returning that result."""
    ctx["latent"] = latent
    ctx.pop("_kctx_sampled", None)


def ctx_set_sampled_image(ctx, image):
    """Store a sampled image on the context and drop everything the sample
    replaced: the latent and cached sample built from the old image, the mask it
    was sampled with, and the tiling info of the old image."""
    ctx["image"] = image
    drop_image_derived(ctx)
    ctx.pop("mask", None)
    ctx.pop("tiling_info", None)


def set_sampling(ctx, steps, step_refiner, cfg, sampler, scheduler):
    """Override the context's sampling parameters (steps, step_refiner, cfg, sampler, scheduler)."""
    ctx["steps"] = steps
    ctx["step_refiner"] = step_refiner
    ctx["cfg"] = cfg
    ctx["sampler"] = sampler
    ctx["scheduler"] = scheduler


def _latent_downscale(vae, channels):
    """(w, h) spatial downscale ratio of a VAE, guessed from latent channels when no VAE is given."""
    r = getattr(vae, "downscale_ratio", None) if vae is not None else None
    if r is not None:
        if isinstance(r, (tuple, list)):
            return int(r[2]), int(r[1])
        return int(r), int(r)
    if vae is not None and getattr(vae, "vae_ratio", None) is not None:
        return int(vae.vae_ratio), int(vae.vae_ratio)
    d = 16 if channels == 128 else 8
    return d, d


def _image_dims(image):
    """(w, h, length) from a channels-last image tensor; length is None for stills."""
    if image is None:
        return None, None, None
    if image.dim() == 4:
        return image.shape[2], image.shape[1], None
    if image.dim() == 5:
        return image.shape[3], image.shape[2], image.shape[1]
    return None, None, None


def _latent_dims(latent, vae=None):
    """(w, h, length) in pixels from a latent dict's samples tensor; length is None for stills."""
    if latent is None:
        return None, None, None
    s = latent["samples"]
    if s.ndim == 4:
        lw, lh, length = s.shape[3], s.shape[2], None
    elif s.ndim == 5:
        lw, lh, length = s.shape[4], s.shape[3], s.shape[2]
    else:
        return None, None, None
    dw, dh = _latent_downscale(vae, s.shape[1])
    return lw * dw, lh * dh, length


_console_colors_ready = False


def _init_console_colors():
    """One-time best-effort ANSI enabling for Windows consoles: colorama when
    importable, else the os.system('') VT trick; a no-op elsewhere."""
    if os.name != "nt":
        return
    try:
        import colorama
        colorama.just_fix_windows_console()
        return
    except Exception:
        pass
    try:
        os.system("")
    except Exception:
        pass


def _log(event, rest=""):
    """One verbose console line: orange [Gibby], the current time (HH:MM:SS.mmm)
    in plain text, the event name in yellow, then the rest of the message."""
    global _console_colors_ready
    if not _console_colors_ready:
        _init_console_colors()
        _console_colors_ready = True
    now = datetime.datetime.now()
    ts = f"{now.strftime('%H:%M:%S')}.{now.microsecond // 1000:03d}"
    print(f"\033[38;5;208m[Gibby]\033[0m {ts} \033[33m{event}\033[0m" + (f" {rest}" if rest else ""))


def clear_vram(event="", verbose=False):
    """Free VRAM like Easy-Use's Clean VRAM Used: gc, CUDA sync, unload all models, empty the cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()
    if verbose:
        _log("Clear VRAM", event)


_INDEX_RANGE = re.compile(r"(-?\d+)\s*-\s*(-?\d+)")


def parse_index_spec(spec, count, field="indices"):
    """The 0-based indices a user-typed spec selects out of `count` items, in the
    order written and without duplicates; None when the spec is empty.
    Accepts indices ("5"), negative indices ("-2" = second to last), inclusive
    ranges ("0-2"), and python-style slices (":3", "4:6", "-3:", "::2"), mixed
    in a comma-separated list. Out-of-range indices are dropped; a malformed
    part raises, naming `field`."""
    if not (spec or "").strip():
        return None
    indices = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if ":" in part:
                bounds = part.split(":")
                if len(bounds) > 3:
                    raise ValueError
                nums = [int(b) if b.strip() else None for b in bounds] + [None] * (3 - len(bounds))
                picked = range(*slice(*nums).indices(count))
            elif match := _INDEX_RANGE.fullmatch(part):
                a, b = (int(v) for v in match.groups())
                picked = range(min(a, b), max(a, b) + 1)
            else:
                picked = (int(part),)
        except ValueError:
            raise ValueError(f"Invalid {field} entry: {part!r} (indices like '0,2', a range like '0-2', or a slice like '1:3')")
        for i in picked:
            i += count if i < 0 else 0
            if 0 <= i < count and i not in indices:
                indices.append(i)
    return indices


def _mp(w, h):
    """Megapixels of a w x h resolution as a 2-decimal string; '' when either dimension is missing."""
    if not w or not h:
        return ""
    return f"{(w * h) / 1_000_000:.2f}"


def _image_count_s(image):
    """' N frames' for a (B, F, H, W, C) video, ' N images' for a multi-image (B, H, W, C) batch, '' for a single still."""
    if image is None:
        return ""
    if image.dim() == 5:
        return f" {image.shape[1]} frames"
    if image.dim() == 4 and image.shape[0] > 1:
        return f" {image.shape[0]} images"
    return ""


def _latent_count_s(latent):
    """' N frames' for a video latent's samples, ' N images' for a batch of latents, '' for a single still."""
    if latent is None:
        return ""
    # samples can be a NestedTensor (no .dim()), so use ndim/shape like _latent_dims
    s = latent["samples"]
    if s.ndim == 5:
        return f" {s.shape[2]} frames"
    if s.ndim == 4 and s.shape[0] > 1:
        return f" {s.shape[0]} images"
    return ""


def _is_model(model_obj, model_classes):
    """Whether a model wrapper (ModelPatcher) wraps one of model_classes."""
    return model_obj is not None and hasattr(model_obj, "model") and isinstance(model_obj.model, model_classes)


def _is_flux2(model_obj):
    """Whether a model wrapper (ModelPatcher) wraps a Flux2 (or klein) model -
    the family whose recipe wants the sequence-length-aware Flux2 schedule."""
    return _is_model(model_obj, comfy.model_base.Flux2)


def _is_flux2_latent(model_obj):
    """Whether a model wrapper (ModelPatcher) wants Flux2's 128ch /16x latents.
    Flux3 shares the Flux2 latent format but keeps its own constant shift, so
    the latent format and the schedule ask with different predicates."""
    return _is_model(model_obj, (comfy.model_base.Flux2, comfy.model_base.Flux3))


def _clip_takes_images(clip):
    """Whether a clip's text encoder runs a vision tower - the same contract core
    uses (sd1_clip.SDClipModel.process_tokens hands image embeds to the
    transformer's preprocess_embed). True for the Qwen-VL / Qwen3.5 / Gemma3
    encoders, false for the text-only ones (flux2's mistral, klein's qwen3)."""
    te = getattr(clip, "cond_stage_model", None)
    name = getattr(te, "clip", None)
    inner = getattr(te, name, None) if isinstance(name, str) else None
    return hasattr(getattr(inner, "transformer", None), "preprocess_embed")


def empty_latent(width, height, batch_size=1, flux2=None, model_obj=None):
    """An empty latent dict for width x height: the 128ch /16x Flux2 one when
    flux2 (or model_obj is a Flux2/Flux3 model), else the standard 4ch /8x one."""
    if flux2 is None:
        flux2 = _is_flux2_latent(model_obj)
    if flux2:
        samples = torch.zeros([batch_size, 128, height // 16, width // 16],
                              device=comfy.model_management.intermediate_device(), dtype=comfy.model_management.intermediate_dtype())
        return {"samples": samples, "downscale_ratio_spacial": 16}
    samples = torch.zeros([batch_size, 4, height // 8, width // 8],
                          device=comfy.model_management.intermediate_device(), dtype=comfy.model_management.intermediate_dtype())
    return {"samples": samples, "downscale_ratio_spacial": 8}


def ctx_size(ctx):
    """(width, height) from the context: the explicit width/height when both are
    set, else the image's, else the latent's scaled by the VAE's downscale ratio.
    (0, 0) when there is nothing to measure."""
    width, height = ctx.get("width") or 0, ctx.get("height") or 0
    if width > 0 and height > 0:
        return int(width), int(height)
    if ctx.get("image") is not None:
        img_h, img_w = ctx["image"].shape[1], ctx["image"].shape[2]
        return int(img_w), int(img_h)
    if ctx.get("latent") is not None:
        lat_h, lat_w = ctx["latent"]["samples"].shape[2], ctx["latent"]["samples"].shape[3]
        dw, dh = _latent_downscale(ctx.get("vae"), ctx["latent"]["samples"].shape[1])
        return lat_w * dw, lat_h * dh
    return 0, 0


def _stringify(value):
    """Convert an arbitrary value to a string, mirroring KJNodes' 'Something To String'.

    Primitives become their str() form, lists/tuples are joined with ', ', and
    anything else falls back to str(). None becomes an empty string so the
    output is always a valid STRING.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value)


# Encoded prompts per clip object, oldest first; the clip is a weak key, so the
# entries of a clip that is unloaded go with it.
_COND_CACHE = weakref.WeakKeyDictionary()
_COND_CACHE_LIMIT = 8


def _encode_text(clip, text):
    """Encode a prompt with clip, reusing the encoding made earlier with the same
    clip object: the same prompt across runs, and the prompt a context's conditioning
    and its prompt travel plan share. Keyed weakly by the clip, so unloading it drops
    its entries, and a lora applied to a clip clones it, so a different text encoder
    misses instead of reusing stale embeddings. A clip keeps its last
    _COND_CACHE_LIMIT prompts."""
    entries = _COND_CACHE.get(clip)
    if entries is None:
        entries = _COND_CACHE[clip] = OrderedDict()
    if text in entries:
        entries.move_to_end(text)
        return entries[text]
    encoded = clip.encode_from_tokens_scheduled(clip.tokenize(text))
    entries[text] = encoded
    while len(entries) > _COND_CACHE_LIMIT:
        entries.popitem(last=False)
    return encoded


def _graft_conds(base_cond, new_conds):
    """The freshly encoded conds with base_cond's values underneath them, so the
    reference payloads (flux2 reference_latents, h3 minimax_refs/keyframes) survive;
    the new prompt's own values win."""
    if base_cond is None:
        return new_conds
    out = []
    for (new_mc, new_vals), (_, base_vals) in zip(new_conds, base_cond):
        vals = dict(base_vals)
        vals.update(new_vals)
        out.append([new_mc, vals])
    return out


def _images_in_tokens(tokens):
    """Whether tokenize() actually placed the images. A tokenizer with no image
    template leaves the vision block out of the text and the images disappear
    without an error - MOSS-Image's Qwen3.5 tokenizer rewrites the no-template
    path as text-only, so the encoder's vision tower is never asked."""
    return any(isinstance(token[0], dict) and token[0].get("type") == "image"
               for batch in tokens.values() for row in batch for token in row)


def _recondition_text(clip, base_cond, text, ctx, images=None, keep_vision=False):
    """Re-encode `text` against clip, grafted onto base_cond's values so its
    reference payloads survive (flux2 reference_latents, h3 minimax_refs/keyframes).
    For h3 the context carries the pre-processed ref items / keyframe images, and
    the text is tokenized with them so the Qwen vision tokens stay consistent.
    With images the text is tokenized with them: qwen2.1 drops the vision tokens
    and splices reference latents into the slots they open (keep_vision=False),
    every other vision encoder keeps the tokens and conditions on them alone."""
    ref_items = ctx.get("h3_ref_items")
    kf_imgs = ctx.get("h3_kf_imgs")
    if ref_items:
        tokens = clip.tokenize(text, minimax_ref_items=ref_items)
        return _graft_conds(base_cond, clip.encode_from_tokens_scheduled(tokens))
    if images:
        tokens = clip.tokenize(text, images=images, keep_vision=keep_vision, prevent_empty_text=True)
        if not _images_in_tokens(tokens):
            logging.warning(f"{type(clip.tokenizer).__name__} put no images in the tokens: the reference images are dropped and the prompt conditions on text alone.")
        return _graft_conds(base_cond, clip.encode_from_tokens_scheduled(tokens))
    if kf_imgs:
        tokens = clip.tokenize(text, images=kf_imgs)
        return _graft_conds(base_cond, clip.encode_from_tokens_scheduled(tokens))
    # Only the plain-text path is cached: tokenizing with ref items or images depends
    # on tensors that have no cheap identity to key on.
    return _graft_conds(base_cond, _encode_text(clip, text))


def _conditioning_text(ctx, side):
    """The text one side's conditioning is encoded from: the first prompt of its
    travel plan when the prompt travels, otherwise the prompt itself."""
    return _travel_step_text(ctx.get(side + "_prompt") or "", ctx.get("steps"))


def recondition_prompts(ctx, positive_text, negative_text, images=None, keep_vision=False):
    """Re-encode positive/negative prompt texts against the context clip, grafted
    onto the context's existing conditionings so their reference payloads survive.
    A prompt that travels is encoded as the first prompt of its travel plan; the
    sampler swaps in the other steps. Returns (positive, negative); falls back to
    the existing conditionings when there is no clip or nothing to re-encode."""
    clip = ctx.get("clip")
    if clip is None or (not positive_text and not negative_text):
        return ctx.get("positive"), ctx.get("negative")
    steps = ctx.get("steps")
    positive = _recondition_text(clip, ctx.get("positive"), _travel_step_text(positive_text, steps), ctx, images, keep_vision)
    if ctx.get("cfg") == 1:
        negative, = ConditioningZeroOut().zero_out(positive)
    elif negative_text:
        negative = _recondition_text(clip, ctx.get("negative"), _travel_step_text(negative_text, steps), ctx, images, keep_vision)
    else:
        negative = ctx.get("negative")
    return positive, negative


def ensure_conditioning(ctx):
    """Fill in the context's missing positive/negative conditioning from its clip,
    zeroing the negative when cfg=1. A prompt that travels conditions with the first
    prompt of its travel plan; the sampler swaps in the other steps. Mutates ctx."""
    if ctx.get("positive") is None and ctx.get("clip") is not None:
        ctx["positive"] = _encode_text(ctx["clip"], _conditioning_text(ctx, "positive"))
    if ctx.get("negative") is None:
        if ctx.get("cfg") == 1 and ctx.get("positive") is not None:
            ctx["negative"], = ConditioningZeroOut().zero_out(ctx["positive"])
        elif ctx.get("clip") is not None:
            ctx["negative"] = _encode_text(ctx["clip"], _conditioning_text(ctx, "negative"))


def encode_image(vae, image, tiled=False, tile_size=512, overlap=64, temporal_size=64, temporal_overlap=8, verbose=False):
    """VAE-encode an image, tiled when the tiled VAE settings are on."""
    if verbose:
        w, h, _ = _image_dims(image)
        _log("VAE encode starting", f"{w}x{h} ({_mp(w, h)} Mp){_image_count_s(image)}")
    t0 = time.time()
    if tiled:
        latent, = VAEEncodeTiled().encode(vae, image, tile_size, overlap, temporal_size, temporal_overlap)
    else:
        latent, = VAEEncode().encode(vae, image)
    if verbose:
        _log("VAE encode took", f"{time.time() - t0:.2f}s")
    return latent


def decode_latent(vae, latent, tiled=False, tile_size=512, overlap=64, temporal_size=64, temporal_overlap=8, verbose=False):
    """VAE-decode a latent, tiled when the tiled VAE settings are on."""
    if verbose:
        w, h, _ = _latent_dims(latent, vae)
        _log("VAE decode starting", f"{w}x{h} ({_mp(w, h)} Mp){_latent_count_s(latent)}")
    t0 = time.time()
    if tiled:
        image, = VAEDecodeTiled().decode(vae, latent, tile_size, overlap, temporal_size, temporal_overlap)
    else:
        image, = VAEDecode().decode(vae, latent)
    if verbose:
        w, h, _ = _image_dims(image)
        _log("VAE decode took", f"{time.time() - t0:.2f}s {w}x{h} ({_mp(w, h)} Mp){_image_count_s(image)}")
    return image


class GibbyContext(io.ComfyNode):
    """Bundle many values into a single CONTEXT object and pass them along."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Gibby_Context",
            display_name="Context",
            category="gibby/context",
            description=(
                "Collects many values into one CONTEXT object (and passes each "
                "through individually). Every input is optional; connect a base "
                "context in to inherit its values for any field left unconnected."
            ),
            inputs=[
                _CONTEXT_TYPE.Input("context", optional=True),
                io.Model.Input("model", optional=True),
                io.Clip.Input("clip", optional=True),
                io.Vae.Input("vae", optional=True),
                io.Vae.Input("vae_audio", optional=True),
                io.Conditioning.Input("positive", optional=True),
                io.Conditioning.Input("negative", optional=True),
                io.Latent.Input("latent", optional=True),
                io.Image.Input("image", optional=True),
                io.Mask.Input("mask", optional=True),
                io.Audio.Input("audio", optional=True),
                io.Mask.Input("mask_audio", optional=True),
                io.AnyType.Input("model_name", optional=True),
                _lora_stack.Input("lora_stack", optional=True),
                io.Int.Input("seed_value", optional=True, force_input=True),
                io.Int.Input("steps", optional=True, force_input=True),
                io.Int.Input("step_refiner", optional=True, force_input=True),
                io.Float.Input("cfg", optional=True, force_input=True),
                # Combo + forceInput: link-only slots that accept sampler/scheduler links like rgthree's Context Big.
                io.Combo.Input("sampler", options=comfy.samplers.KSampler.SAMPLERS, optional=True, extra_dict={"forceInput": True}),
                io.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS, optional=True, extra_dict={"forceInput": True}),
                io.Int.Input("width", optional=True, force_input=True),
                io.Int.Input("height", optional=True, force_input=True),
                io.String.Input("positive_prompt", optional=True, force_input=True),
                io.String.Input("negative_prompt", optional=True, force_input=True),
                io.AnyType.Input("any", optional=True, extra_dict={"forceInput": True}),
            ],
            outputs=[
                _CONTEXT_TYPE.Output(display_name="context"),
                io.Model.Output(display_name="model"),
                io.Clip.Output(display_name="clip"),
                io.Vae.Output(display_name="vae"),
                io.Vae.Output(display_name="vae_audio"),
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
                io.Image.Output(display_name="image"),
                io.Mask.Output(display_name="mask"),
                io.Audio.Output(display_name="audio"),
                io.Mask.Output(display_name="mask_audio"),
                io.String.Output(display_name="model_name"),
                _lora_stack.Output(display_name="lora_stack"),
                io.Int.Output(display_name="seed"),
                io.Int.Output(display_name="steps"),
                io.Int.Output(display_name="step_refiner"),
                io.Float.Output(display_name="cfg"),
                io.AnyType.Output(display_name="sampler"),
                io.AnyType.Output(display_name="scheduler"),
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
                io.String.Output(display_name="positive_prompt"),
                io.String.Output(display_name="negative_prompt"),
                io.AnyType.Output(display_name="any"),
            ],
        )

    @classmethod
    def execute(cls, context=None, model=None, clip=None, vae=None,
                vae_audio=None, positive=None, negative=None, latent=None, image=None, mask=None,
                audio=None, mask_audio=None, model_name=None, lora_stack=None,
                seed_value=None, steps=None, step_refiner=None, cfg=None, sampler=None, scheduler=None,
                width=None, height=None, positive_prompt=None, negative_prompt=None,
                **kwargs) -> io.NodeOutput:
        any_value = kwargs.get('any')
        # Build the context dict from directly-provided values.
        ctx = {
            "model": model,
            "clip": clip,
            "vae": vae,
            "vae_audio": vae_audio,
            "positive": positive,
            "negative": negative,
            "latent": latent,
            "image": image,
            "mask": mask,
            "audio": audio,
            "mask_audio": mask_audio,
            "model_name": model_name,
            "lora_stack": lora_stack,
            "seed": seed_value,
            "steps": steps,
            "step_refiner": step_refiner,
            "cfg": cfg,
            "sampler": sampler,
            "scheduler": scheduler,
            "width": width,
            "height": height,
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "any": any_value,
        }

        # Fall back to the base context's values for anything not connected,
        # carrying through keys this node doesn't model (crop_info and the like).
        if isinstance(context, dict):
            for key, value in context.items():
                if ctx.get(key) is None:
                    ctx[key] = value

        # Stringify after inheritance so an unconnected model_name can still be
        # picked up from the base context instead of becoming "".
        ctx["model_name"] = _stringify(ctx["model_name"])

        # Apply LoRAs from a directly-connected lora_stack to model/clip. A
        # stack inherited from the base context is already baked into that
        # context's model, so it must not be applied again here.
        for name, sm, sc in iter_lora_stack(lora_stack):
            ctx["model"], ctx["clip"] = _apply_lora(ctx["model"], ctx["clip"], name, sm, sc)

        return io.NodeOutput(
            ctx,  # context
            ctx["model"],  # model
            ctx["clip"],  # clip
            ctx["vae"],  # vae
            ctx["vae_audio"],  # vae_audio
            ctx["positive"],  # positive
            ctx["negative"],  # negative
            ctx["latent"],  # latent
            ctx["image"],  # image
            ctx["mask"],  # mask
            ctx["audio"],  # audio
            ctx["mask_audio"],  # mask_audio
            ctx["model_name"],  # model_name (string)
            ctx["lora_stack"],  # lora_stack
            ctx["seed"],  # seed
            ctx["steps"],  # steps
            ctx["step_refiner"],  # step_refiner
            ctx["cfg"],  # cfg
            ctx["sampler"],  # sampler (string)
            ctx["scheduler"],  # scheduler (string)
            ctx["width"],  # width
            ctx["height"],  # height
            ctx["positive_prompt"],  # positive_prompt
            ctx["negative_prompt"],  # negative_prompt
            ctx["any"],  # any_value
        )

    @staticmethod
    def evaluate(ctx, tiled=False, tile_size=512, overlap=64, temporal_size=64, temporal_overlap=8):
        """Fill missing context values on demand from available inputs.
        Mutates ctx - pass it a copy (ctx_from) of any context you don't own."""
        # Conditioning from the prompts, zeroing the negative when cfg=1.
        ensure_conditioning(ctx)

        # Latent: only fill if missing - use existing latent if present.
        if ctx.get("latent") is None:
            # Image takes priority - encode it fresh with mask if available.
            if ctx.get("image") is not None and ctx.get("vae") is not None:
                ctx["latent"] = encode_image(ctx["vae"], ctx["image"], tiled, tile_size, overlap, temporal_size, temporal_overlap)
                if ctx.get("mask") is not None:
                    ctx["latent"], = SetLatentNoiseMask().set_mask(ctx["latent"], ctx["mask"])

                # If audio also available, combine into AV latent. A model without
                # audio slots keeps the image latent instead of failing the run.
                if ctx.get("vae_audio") is not None and ctx.get("audio") is not None:
                    try:
                        audio_latent, = VAEEncodeAudio().encode(ctx["vae_audio"], ctx["audio"])
                        ctx["latent"], = LTXVConcatAVLatent().execute(ctx["latent"], audio_latent)
                    except Exception as exc:
                        _log("AV latent skipped", f"{type(exc).__name__}: {exc}")

            # If no image but audio + vae_audio available, encode audio-only latent
            elif ctx.get("vae_audio") is not None and ctx.get("audio") is not None:
                ctx["latent"], = VAEEncodeAudio().encode(ctx["vae_audio"], ctx["audio"])

        # Resolve width/height from image or latent if they're missing.
        width, height = ctx_size(ctx)
        if width > 0 and height > 0:
            ctx["width"] = width
            ctx["height"] = height

        # Empty latent: nothing to encode or sample from, but width/height were given.
        if ctx.get("latent") is None and ctx.get("width", 0) > 0 and ctx.get("height", 0) > 0:
            ctx["latent"] = empty_latent(ctx["width"], ctx["height"], model_obj=ctx.get("model"))

        return ctx

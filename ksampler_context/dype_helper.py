"""
DyPE bridge - ComfyUI-DyPE as a soft dependency
-----------------------------------------------
Everything the KSampler (Context) needs to drive the DyPE pack: the high-res
model patches (DyPE, SEGA, SPA) and the cascade hijacks (PixelRush, FreeScale,
HiFlow).

The pack is optional and its load order relative to this pack is not
guaranteed, so:

- availability is a filesystem check (``dype_installed``), never an import or a
  sys.modules probe - it answers correctly before ComfyUI has imported DyPE;
- the pack is imported lazily on first use (``_pack``) and cached; when
  ComfyUI already imported it, that module object is reused instead of loading
  a second copy;
- the DyPE options node is only registered when the pack is present (see the
  pack ``__init__``), and every entry point here degrades to a no-op when the
  pack is missing, so a graph saved with DyPE wired in still runs without it.

The native training resolution the patches need is never asked for: it is a
constant of each architecture (``BASE_RESOLUTIONS``), looked up from the
resolved model type.
"""

import importlib
import importlib.util
import logging
import os
import sys

import folder_paths
import torch
import comfy.utils

_PACK_DIR_NAME = "ComfyUI-DyPE"
_PACK_DIR = os.path.join(folder_paths.base_path, "custom_nodes", _PACK_DIR_NAME)
_PACK_MODULE = "_gibby_dype_pack"

#: Model patches: the KSampler patches the model and samples normally.
DYPE_PATCH_METHODS = ("dype", "sega", "spa")
#: Cascades: the KSampler hands the base latent to the pack and skips sampling.
DYPE_CASCADE_METHODS = ("pixelrush", "freescale", "hiflow")

#: Native training resolution per DyPE architecture. The checkpoint records its
#: trained extent only for Z-Image (axes_lens) and Anima; for every other
#: architecture it is a constant of the training recipe, so it is looked up
#: from the detected type instead of asked for. What matters is the trained
#: TOKEN grid: FLUX.2/klein (16x VAE, patch 1) and FLUX.1 (8x VAE, patch 2) are
#: both 64x64 tokens at 1024px, so one "flux" entry covers them - and the DyPE
#: detector reports klein as "flux" anyway.
BASE_RESOLUTIONS = {
    "flux": 1024,
    "nunchaku": 1024,
    "zimage": 1024,
    "qwen": 1328,
    "qwen21": 1328,
    "krea2": 1328,
    "anima": 1920,
}

_pack_cache = False
_pack_errors = []


def dype_installed():
    """Whether the ComfyUI-DyPE pack is present. Filesystem-only, so it is
    correct regardless of which custom node pack imported first."""
    return os.path.isfile(os.path.join(_PACK_DIR, "__init__.py"))


def _load_pack():
    """The DyPE pack module, imported once and cached. None when the pack is
    not installed or failed to import (logged once)."""
    global _pack_cache
    if _pack_cache is not False:
        return _pack_cache
    _pack_cache = None
    if not dype_installed():
        return None
    # Reuse ComfyUI's own import when it happened (its loader names the module
    # after the folder path, so match on the file location).
    for mod in list(sys.modules.values()):
        path = getattr(mod, "__file__", None)
        if path and os.path.dirname(os.path.abspath(path)) == os.path.abspath(_PACK_DIR):
            _pack_cache = mod
            return mod
    try:
        spec = importlib.util.spec_from_file_location(
            _PACK_MODULE, os.path.join(_PACK_DIR, "__init__.py"),
            submodule_search_locations=[_PACK_DIR])
        module = importlib.util.module_from_spec(spec)
        sys.modules[_PACK_MODULE] = module
        spec.loader.exec_module(module)
        _pack_cache = module
    except Exception as exc:
        sys.modules.pop(_PACK_MODULE, None)
        _pack_errors.append(str(exc))
        logging.warning(f"[Gibby Nodes] DyPE: could not import the ComfyUI-DyPE pack: {exc}")
    return _pack_cache


def _submodule(pack, name):
    """A DyPE submodule (``"src.patch_utils"``) of an already-loaded pack."""
    return importlib.import_module(f"{pack.__name__}.{name}")


def _log(opts, message):
    if opts.get("verbose", False):
        logging.info(f"[Gibby Nodes] DyPE {opts.get('method', '?')}: {message}")


# ---------------------------------------------------------------------------
# Model type
# ---------------------------------------------------------------------------

def resolve_model_type(model, requested="auto"):
    """The DyPE architecture key for a model patcher ("flux", "qwen", ...).

    Uses the pack's own detector so the answer matches what the patch would
    resolve internally; "auto" when the pack or the detector is unavailable.
    """
    pack = _load_pack()
    if pack is None or model is None:
        return "auto"
    try:
        detect = _submodule(pack, "src.model_detect")
        return detect.resolve_model_type(model.get_model_object("diffusion_model"), requested)
    except Exception:
        return "auto"


# ---------------------------------------------------------------------------
# Model patches
# ---------------------------------------------------------------------------

def apply_dype_patch(model, opts, width, height):
    """Patch a model with the option's method, or return it unchanged for the
    cascade methods (they hijack sampling instead), a missing option, and when
    there is no size to size the patch to."""
    method = opts.get("method") if opts else None
    if method not in DYPE_PATCH_METHODS:
        return model
    if not width or not height or width <= 0 or height <= 0:
        logging.warning(f"[Gibby Nodes] DyPE {method}: no size for this step, model left unpatched")
        return model

    pack = _load_pack()
    if pack is None:
        raise ModuleNotFoundError(
            "DyPE options are connected but the ComfyUI-DyPE pack is not installed - "
            "install it or remove the DyPE options node."
        )
    patch_utils = _submodule(pack, "src.patch_utils")
    model_type = opts.get("model_type", "auto")
    arch = resolve_model_type(model, model_type)
    base_resolution = BASE_RESOLUTIONS.get(arch, 1024)
    _log(opts, f"patching {method} at {width}x{height} ({arch}, native {base_resolution})")

    if method == "dype":
        return patch_utils.apply_dype_to_model(
            model, model_type, width, height, opts["dype_method"],
            opts["yarn_alt_scaling"], opts.get("enable_dype", True),
            opts["dype_scale"], opts["dype_exponent"],
            opts["base_shift"], opts["max_shift"], base_resolution, opts["dype_start_sigma"])
    if method == "sega":
        return patch_utils.apply_sega_to_model(
            model, model_type, width, height, opts["sega_method"],
            opts["mscale_alpha"], opts["mscale_beta"], opts["mscale_min"],
            opts["spread_min"], opts["spread_max"], opts["spread_alpha"],
            opts["base_mscale_formula"], opts["base_mscale_coefficient"],
            base_resolution, opts["base_shift"], opts["max_shift"])
    # spa
    spa = _submodule(pack, "src.spa")
    layer_filter = opts.get("spa_layer_filter", "")
    parsed = spa.parse_layer_filter(layer_filter) if layer_filter else None
    bundle = int(opts.get("bundle_size", 0))
    return spa.apply_spa_to_model(
        model, model_type, width, height,
        bundle_size=(bundle if bundle > 0 else None),
        spa_start_sigma=float(opts.get("spa_start_sigma", 1.0)),
        spa_steps=int(opts.get("spa_steps", 3)),
        spa_layer_filter=parsed,
        proportional_attention=bool(opts.get("proportional_attention", False)))


# ---------------------------------------------------------------------------
# Cascade hijacks
# ---------------------------------------------------------------------------

def is_dype_cascade(opts):
    """Whether the option takes over sampling (PixelRush / FreeScale / HiFlow)."""
    return bool(opts) and opts.get("method") in DYPE_CASCADE_METHODS


def warn_cascade_overrides(opts, sigmas=None, sampler=None, start_step=0.0, end_step=10000.0,
                           add_noise=True, denoise=1.0, lora_travel=False, prompt_travel=False):
    """One warning listing the KSampler features the cascade cannot honour: it runs
    the pack's own sampler, schedule and noise, so only the model, the conditioning,
    the latent, the seed and the cfg carry over."""
    ignored = []
    if sigmas is not None:
        ignored.append("the connected sigmas")
    if sampler is not None:
        ignored.append("the connected sampler")
    if start_step != 0.0 or end_step < 10000.0:
        ignored.append("start/end step")
    if not add_noise:
        ignored.append("add_noise off")
    if denoise < 1.0:
        ignored.append("denoise (the method's own noise controls apply)")
    if lora_travel:
        ignored.append("lora travel")
    if prompt_travel:
        ignored.append("prompt travel")
    if ignored:
        logging.warning(f"[Gibby Nodes] DyPE {opts.get('method')} takes sampling over - ignored: {', '.join(ignored)}")


def warn_cascade_not_in_step(opts):
    """The crop-inpaint and iterative upscale steps sample through the regular
    sampler, which a cascade cannot hook - say so instead of sampling unpatched
    in silence."""
    logging.warning(
        f"[Gibby Nodes] DyPE {opts.get('method')} does not run inside a crop-inpaint / "
        "iterative upscale step (those sample normally) - use dype/sega/spa with them")


def _restore_outside_mask(cascaded, base, mask, method):
    """Keep the cascade inside the mask and the base latent outside it: a cascade
    samples the whole latent, so the area the mask excludes comes back from the
    base latent resized to the cascade's output size (the same bicubic the
    cascades use for their own latent resizes)."""
    if base.shape[1] != cascaded.shape[1]:
        logging.warning(
            f"[Gibby Nodes] DyPE {method}: the cascade output has {cascaded.shape[1]} channels "
            f"vs the context latent's {base.shape[1]} - the mask was not applied")
        return cascaded
    if base.shape[2:] != cascaded.shape[2:]:
        base = comfy.utils.common_upscale(base, cascaded.shape[-1], cascaded.shape[-2], "bicubic", "disabled")
    mask = comfy.utils.reshape_mask(mask, cascaded.shape).to(cascaded)
    return cascaded * mask + base.to(cascaded) * (1.0 - mask)


def run_dype_cascade(opts, model, vae, positive, negative, latent, seed, cfg, steps):
    """Run the cascade on the context's base latent; returns a latent dict at
    the cascade's output size. The pack's own node execute() is reused so its
    VAE adapters, channel fixes and 5D bridging stay in one place.

    The 0 the option's guidance/steps widgets document as "the context's value"
    is resolved here. PixelRush and FreeScale draw their noise from the global
    RNG, so it is seeded from the context seed to keep a run reproducible, and a
    masked context has everything outside the mask restored from the base latent
    once the cascade is done."""
    pack = _load_pack()
    if pack is None:
        raise ModuleNotFoundError(
            "DyPE options are connected but the ComfyUI-DyPE pack is not installed - "
            "install it or remove the DyPE options node."
        )
    method = opts["method"]
    nodes = _submodule(pack, f"nodes.{method}")
    node = {"pixelrush": nodes.PixelRushNode, "freescale": nodes.FreeScaleNode,
            "hiflow": nodes.HiFlowNode}[method]
    latent_image = {"samples": latent["samples"], **{k: v for k, v in latent.items() if k != "samples"}}
    cfg = float(opts.get("cfg") or cfg)
    steps = int(opts.get("steps") or steps)
    torch.manual_seed(seed & 0x7FFFFFFFFFFFFFFF)

    if method == "pixelrush":
        out = node.execute(
            model=model, vae=vae, positive=positive, negative=negative, latent_image=latent_image,
            cfg=cfg, num_cascade_stages=int(opts.get("cascade_stages", 1)),
            k_timestep=int(opts.get("k_timestep", 249)), noise_lambda=float(opts.get("noise_lambda", 0.95)),
            noise_injection=opts.get("noise_injection", "slerp"), overlap=float(opts.get("overlap", 0.5)),
            gaussian_sigma=float(opts.get("gaussian_sigma", 24.0)),
            patch_h=int(opts.get("patch_h", 0)), patch_w=int(opts.get("patch_w", 0)),
            refiner_model=opts.get("refiner_model"))
    elif method == "freescale":
        out = node.execute(
            model=model, vae=vae, positive=positive, negative=negative, latent_image=latent_image,
            cfg=cfg, num_inference_steps=steps,
            target_resolution=int(opts.get("target_resolution", 2048)),
            noise_timestep=int(opts.get("noise_timestep", 700)),
            fast_mode=bool(opts.get("fast_mode", True)))
    else:
        out = node.execute(
            model=model, vae=vae, positive=positive, negative=negative, latent_image=latent_image,
            cfg=cfg, steps=steps,
            guidance=float(opts.get("guidance", 4.5)), steps_per_stage=int(opts.get("steps_per_stage", 16)),
            tau=float(opts.get("tau", 0.6)), filter_ratio=float(opts.get("filter_ratio", 0.2)),
            alpha_scale=float(opts.get("alpha_scale", 1.0)), beta_scale=float(opts.get("beta_scale", 0.5)),
            upsampling=opts.get("upsampling", "latent"), scale_factor=float(opts.get("scale_factor", 2.0)),
            sharpen=float(opts.get("sharpen", 1.0)), noise_seed=int(seed),
            denoise=float(opts.get("denoise", 1.0)))

    result = out[0]
    result = result if isinstance(result, dict) else {"samples": result}

    mask = latent.get("noise_mask")
    if mask is not None:
        result = dict(result)
        result["samples"] = _restore_outside_mask(result["samples"], latent["samples"], mask, method)
        result.pop("noise_mask", None)    # baked into the samples now
    return result

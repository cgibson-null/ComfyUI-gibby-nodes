"""
DyPE bridge - ComfyUI-DyPE as a soft dependency
-----------------------------------------------
Everything the KSampler (Context) needs to drive the DyPE pack: the high-res
model patches (DyPE, SEGA).

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

_PACK_DIR_NAME = "ComfyUI-DyPE"
_PACK_DIR = os.path.join(folder_paths.base_path, "custom_nodes", _PACK_DIR_NAME)
_PACK_MODULE = "_gibby_dype_pack"

#: The methods the KSampler patches the model with, then samples normally.
#: SPA is deliberately absent: its attention hook reads .shape straight off the
#: q/k/v it is handed, which ComfyUI wraps in an AttentionTensorContainer, so it
#: raises until the DyPE pack unwraps it.
DYPE_PATCH_METHODS = ("dype", "sega")

#: Native training resolution per DyPE architecture. The checkpoint records its
#: trained extent only for Z-Image (axes_lens) and Anima; for every other
#: architecture it is a constant of the training recipe, so it is looked up
#: from the detected type instead of asked for. What matters is the trained
#: TOKEN grid: FLUX.2/klein and FLUX.3 (16x VAE, patch 1) and FLUX.1 (8x VAE,
#: patch 2) are all 64x64 tokens at 1024px, so one "flux" entry covers them - and
#: the DyPE detector reports klein and flux3 as "flux" anyway (flux3's DiT has a
#: pe_embedder).
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
    """Patch a model with the option's method, or return it unchanged for an
    unknown method, a missing option, and when there is no size to size the
    patch to."""
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


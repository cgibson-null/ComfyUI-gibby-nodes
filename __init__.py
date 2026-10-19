"""
Gibby Nodes
-----------
A small collection of ComfyUI custom nodes built on the modern V3 node
schema, each living in its own self-contained folder:

- lora_loader/: multi-lora loader with growable rows (Lora Loader).
- fast_groups/: frontend-only Muter / Bypasser virtual nodes.
- group_header_toggles/: queue/bypass/mute buttons on group headers, configured via native settings (GibbyNodes > Group Header Toggles).
- any/: the type-agnostic utilities - any_switch.py (Any Switch - outputs one connected input by index, only that branch runs), any_batch_list.py (Any Batch/List - batches or lists every connected input) and any_count.py (Any Count - how many items a value holds, 0 for nothing connected or a null).
- context/: the CONTEXT object family - Context (bundler), Context Loader (models + params into a full context), Context Override, Sampling Parameters (Context).
- resolution_latent/: Resize Image / Empty Latent (Context) - empty latent from width/height or aspect ratio + megapixels, resizes linked image/mask to match.
- seedvr2_upscale.py: SeedVR2 Upscale (Context) - the whole SeedVR2 upscale in one node: resize to the megapixels/scale target (or a target size image), pad, tiled VAE encode, sample with the latent as the SeedVR2 conditioning, split/merge the temporal chunks of a video, decode, then align and color-correct the result back to the resized input.
- image_saver/: Image Saver (Context) - saves images with civitai-compatible metadata; settings and lora names come from the context.
- pipes/: media_pipe.py (the media pipe dict shared between nodes - ref images, keyframes with positions, videos, audios - h3_pipe is it plus the h3 params), h3_pipe.py (H3 Pipe Create / H3 Pipe Apply - reusable MiniMax H3 conditioning pipe, raw refs + keyframes stored, encoded at apply time) and pipe_any.py (Pipe Any - combine multiple Any inputs into a pipe dict, or override an existing pipe).
- reference_latent_context/: Reference Latent (Context) - sets reference latents on conditioning from provided images (resized to the megapixels target, scaled, encoded with context VAE when ref_latents is on); a text encoder with a vision tower also sees the images at vl_megapixels and the prompts are re-encoded with them, so a t2i model like krea conditions on a reference without any edit path; hidream-o1 and sensenova get the resized image itself instead of a latent, which is what their models read; optionally references the context's own image at its own size; references a media pipe's images and passes the pipe through, appending the wired images to it; without a context it just builds the pipe from the wired images.
- detection/: Mask/Segment (Context) - detects/segments objects on the context image with an ultralytics bbox detector, a SAM/SeC model, or a SAM3.1 checkpoint; writes the resulting image and mask back into the context.
- mask_edit/: Mask Edit (Context) - the mask manipulation the detection and crop nodes used to carry (grow, invert, edge feather, isolated-pixel removal, hole filling, background removal) plus the mask tinted preview.
- llm/: LLM Connect - Connectivity, Sampling Options, Unsloth Load Options and Generate for a llama.cpp / llama-swap / Unsloth server (OpenAI-compatible API), with auto-growing image/video references; Generate also takes a media pipe (refs + keyframes, labeled) and outputs one.
- tiling.py: Split Tiles (Context) / Combine Tiles (Context) - splits the context image into a batch of uniform overlapping tiles with crop-inpaint masks; recombines the tile batch (at the upscaled size) with a crossfade across the overlaps.
- ksampler_options/dype.py: DyPE options - the ComfyUI-DyPE pack as a soft dependency (registered only when that pack is installed). dype/sega patch the model for high-resolution generation; KSampler (Context) applies them through ksampler_context/dype_helper.py, re-patching per sample at the size actually being sampled so the outbound context keeps the original model.

INSTALL:
Put this whole folder in ComfyUI/custom_nodes/, then restart ComfyUI.
"""

import asyncio
import logging
import os

from comfy_api.latest import ComfyExtension, io

from .lora_loader import GibbyLoraLoader
from .context import GibbyContext
from .context.context_override import GibbyContextOverride
from .context.sampling_parameters import GibbySamplingParametersContext
from .context.context_loader import GibbyContextLoader
from .ksampler_context import GibbyKSamplerContext
from .any.any_switch import GibbyAnySwitch
from .any.any_batch_list import GibbyAnyBatchList
from .any.any_count import GibbyAnyCount
from .resolution_latent import GibbyEmptyLatentResolution
from .seedvr2_upscale import GibbySeedvr2Upscale
from .image_saver import GibbyImageSaverContext
from .pipes.h3_pipe import H3PipeCreate, H3PipeApply
from .reference_latent_context import GibbyReferenceLatentContext
from .pipes.pipe_any import PipeAny
from .crop_paste.crop_image_by_mask import GibbyCropImageByMaskBatch
from .crop_paste.paste_image_by_mask import GibbyPasteImageByMaskBatch
from .mask_edit import GibbyMaskEdit
from .ksampler_options.clear_vram import GibbyClearVramOptions
from .ksampler_options.color_match import GibbyColorMatchOptions
from .ksampler_options.crop_inpaint import GibbyCropInpaintOptions
from .ksampler_options.iterative_upscale import GibbyIterativeUpscaleOptions
from .ksampler_options.lora_travel import GibbyLoraTravelOptions
from .ksampler_options.tiled_vae import GibbyTiledVaeOptions
from .ksampler_options.dype import GibbyDypeOptions
from .ksampler_context.dype_helper import dype_installed
from .tiling import GibbySplitTiles, GibbyCombineTiles
from .ksampler_options.merge_ksampler_options import GibbyMergeKSamplerOptions
from .context.merge_contexts import GibbyMergeContexts
from .detection import GibbyDetection
from .llm import GibbyConnectivity, GibbySamplingOptions, GibbyLoadOptions, GibbyGenerate
from .llm.llamacpp_client import list_models as _llm_list_models

# WEB_DIRECTORY points at the plugin root so ComfyUI discovers every node's
# JS file (it globs recursively) and serves them under /extensions/<this>.
WEB_DIRECTORY = "."

_NODE_DIR = os.path.dirname(os.path.abspath(__file__))
_FOLDER_NAME = os.path.basename(_NODE_DIR)

# --- Serve each JS file ourselves, with an explicit Content-Type -----------
# Two earlier attempts tried to fix Windows' broken .js MIME-type guessing by
# patching the databases ComfyUI's web server consults - but there's no way to
# be fully sure which database (if any) is actually being read on any given
# setup. Rather than keep guessing at that indirectly, this registers a route
# for each file's URL and serves it ourselves, with the Content-Type set
# directly and explicitly. No guessing involved at all, so it can't be
# derailed by any registry/database problem.
try:
    import glob as _glob
    from aiohttp import web
    from server import PromptServer

    def _serve_js_file(path):
        async def handler(request):
            with open(path, "rb") as f:
                body = f.read()
            return web.Response(
                body=body, content_type="text/javascript", charset="utf-8"
            )
        return handler

    # One route per JS file of the pack (lora_loader's rows, the Fast Groups
    # nodes, the group header buttons, the context suggestions, the crop
    # preview, the tracker/overlap/step rows, the LLM dropdown, the runtime
    # badge). Discovered by globbing, so a new JS file is covered without
    # remembering to list it here.
    for _path in _glob.glob(os.path.join(_NODE_DIR, "**", "*.js"), recursive=True):
        PromptServer.instance.routes.get(
            "/extensions/" + _FOLDER_NAME + "/" + os.path.relpath(_path, _NODE_DIR).replace(os.sep, "/")
        )(_serve_js_file(_path))

    # Model-list route for the LLM Connect subpack's Connectivity dropdown.
    @PromptServer.instance.routes.post("/gibby_llm/get_models")
    async def _llm_get_models(request):
        data = await request.json()
        url = data.get("url", "")
        api_key = data.get("api_key", "") or None
        try:
            loop = asyncio.get_running_loop()
            models = await loop.run_in_executor(None, lambda: _llm_list_models(url, api_key=api_key))
        except Exception as exc:
            logging.warning(f"[Gibby Nodes] Could not list models from {url!r}: {exc}")
            return web.json_response([])
        return web.json_response(models)

    logging.info("[Gibby Nodes] Registered dedicated routes for JS files")
except Exception as e:
    logging.warning(f"[Gibby Nodes] Could not register dedicated JS route: {e}")


class GibbyNodesExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        # DyPE options drive the optional ComfyUI-DyPE pack, so they are only
        # offered when it is installed. dype_installed() is a filesystem check,
        # which answers correctly whichever of the two packs imports first.
        dype_nodes = [GibbyDypeOptions] if dype_installed() else []
        return [GibbyAnyBatchList, GibbyAnyCount, GibbyAnySwitch, GibbyClearVramOptions, GibbyContext, GibbyContextLoader, GibbyContextOverride, GibbyColorMatchOptions, GibbyCombineTiles, GibbyConnectivity, GibbyCropImageByMaskBatch, GibbyCropInpaintOptions, GibbyDetection, *dype_nodes, GibbyGenerate, GibbyIterativeUpscaleOptions, GibbyLoadOptions, GibbyLoraTravelOptions, GibbyMaskEdit, GibbyMergeContexts, GibbyMergeKSamplerOptions, GibbyPasteImageByMaskBatch, H3PipeApply, H3PipeCreate, GibbyImageSaverContext, GibbyKSamplerContext, GibbyLoraLoader, GibbySamplingOptions, PipeAny, GibbyReferenceLatentContext, GibbyEmptyLatentResolution, GibbySamplingParametersContext, GibbySeedvr2Upscale, GibbySplitTiles, GibbyTiledVaeOptions]


async def comfy_entrypoint() -> GibbyNodesExtension:
    return GibbyNodesExtension()

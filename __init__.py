"""
Gibby Nodes: a collection of ComfyUI custom nodes on the V3 node schema, each in
its own self-contained folder. Put this whole folder in ComfyUI/custom_nodes/ and
restart ComfyUI.

- lora_loader/: Lora Loader - multi-lora loader with growable rows.
- group/: fast_groups.js and group_header_toggles.js - frontend-only Muter/Bypasser
  virtual nodes and queue/bypass/mute buttons on group headers.
- any/: Any Switch, Any Batch/List, Any Count, Show Any.
- context/: Context, Context Loader, Context Override, Sampling Parameters (Context).
- resolution_latent/: Resize Image / Empty Latent (Context).
- upscale/: the one-node upscales - SeedVR2, RelayVSR.
- h3/: H3 Latent Upscale plus the H3 nodes around the media pipe - H3 Pipe Create /
  Apply, H3 Timeline (Context) and its Assemble / Pin, H3 Mask Latent, H3 Chunk.
- image_saver/: Image Saver (Context).
- pipes/: the media pipe dict and Pipe Any.
- reference_latent_context.py: Reference Latent (Context).
- detection/: Mask/Segment (Context). mask_edit/: Mask Edit (Context).
- crop_paste/: Image Crop By Mask (Batch) (Context) and its paste counterpart.
- llm/: LLM Connect - Connectivity, Sampling Options, Unsloth Load Options, Generate.
- tiling.py: Split Tiles / Combine Tiles (Context).
- ksampler_context/: KSampler (Context), driven by the option nodes in
  ksampler_options/ (Crop-Inpaint, Iterative, Color Match, Tiled VAE, Clear VRAM,
  Lora Travel, DyPE).
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
from .any.any_show import GibbyShowAny
from .resolution_latent import GibbyEmptyLatentResolution
from .upscale.seedvr2_upscale import GibbySeedvr2Upscale
from .upscale.relayvsr_upscale import GibbyRelayvsrUpscale
from .h3.h3_latent_upscale import GibbyH3LatentUpscale
from .image_saver import GibbyImageSaverContext
from .h3.h3_pipe import H3PipeCreate, H3PipeApply
from .h3.h3_timeline import (GibbyH3Timeline, GibbyH3TimelineAssemble, GibbyH3TimelinePin,
                             GibbyH3MaskLatent, GibbyH3Chunk)
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
# Windows guesses the .js MIME type wrongly and there is no reliable way to know
# which database ComfyUI's web server actually consults, so instead of patching
# one we register a route per file and set the Content-Type directly.
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

    # One route per JS file of the pack, discovered by globbing so a new file is
    # covered without listing it here.
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
        return [GibbyAnyBatchList, GibbyAnyCount, GibbyAnySwitch, GibbyClearVramOptions, GibbyContext, GibbyContextLoader, GibbyContextOverride, GibbyColorMatchOptions, GibbyCombineTiles, GibbyConnectivity, GibbyCropImageByMaskBatch, GibbyCropInpaintOptions, GibbyDetection, *dype_nodes, GibbyGenerate, GibbyIterativeUpscaleOptions, GibbyLoadOptions, GibbyLoraTravelOptions, GibbyMaskEdit, GibbyMergeContexts, GibbyMergeKSamplerOptions, GibbyPasteImageByMaskBatch, H3PipeApply, H3PipeCreate, GibbyImageSaverContext, GibbyKSamplerContext, GibbyLoraLoader, GibbySamplingOptions, PipeAny, GibbyReferenceLatentContext, GibbyEmptyLatentResolution, GibbyH3LatentUpscale, GibbyH3Timeline, GibbyH3TimelineAssemble, GibbyH3TimelinePin, GibbyH3MaskLatent, GibbyH3Chunk, GibbySamplingParametersContext, GibbySeedvr2Upscale, GibbyRelayvsrUpscale, GibbyShowAny, GibbySplitTiles, GibbyTiledVaeOptions]


async def comfy_entrypoint() -> GibbyNodesExtension:
    return GibbyNodesExtension()

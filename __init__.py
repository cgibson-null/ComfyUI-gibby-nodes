"""
Gibby Nodes
-----------
A small collection of ComfyUI custom nodes built on the modern V3 node
schema, each living in its own self-contained folder:

- lora_loader/: multi-lora loader with growable rows (Lora Loader).
- fast_groups/: frontend-only Muter / Bypasser virtual nodes.
- group_header_toggles/: queue/bypass/mute buttons on group headers, configured via native settings (GibbyNodes > Group Header Toggles).
- context/: the CONTEXT object family - Context (bundler), Context Loader (models + params into a full context), Context Override, Sampling Parameters (Context).
- resolution_latent/: Resize Image / Empty Latent (Context) - empty latent from width/height or aspect ratio + megapixels, resizes linked image/mask to match.
- image_saver/: Image Saver (Context) - saves images with civitai-compatible metadata; settings and lora names come from the context.
- H3_pipe/: H3 Pipe Create / H3 Pipe Apply - reusable MiniMax H3 conditioning pipe (raw refs stored, encoded at apply time).
- reference_latent_context/: Reference Latent (Context) - sets reference latents on conditioning from provided images (resized to 1MP, scaled, encoded with context VAE).
- pipe_any/: Pipe Any - combine multiple Any inputs into a pipe dict, or override an existing pipe.

INSTALL:
Put this whole folder in ComfyUI/custom_nodes/, then restart ComfyUI.
"""

import logging
import os

from comfy_api.latest import ComfyExtension, io

from .lora_loader import GibbyLoraLoader
from .context import GibbyContext
from .context.context_override import GibbyContextOverride
from .context.sampling_parameters import GibbySamplingParametersContext
from .context.context_loader import GibbyContextLoader
from .ksampler_context import GibbyKSamplerContext
from .any_switch import GibbyAnySwitch
from .resolution_latent import GibbyEmptyLatentResolution
from .image_saver import GibbyImageSaverContext
from .h3_pipe import H3PipeCreate, H3PipeApply
from .reference_latent_context import GibbyReferenceLatentContext
from .pipe_any import PipeAny

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

    # lora_loader.js for the Lora Loader rows, fast_groups.js for the Fast
    # Groups Muter / Bypasser nodes, group_header_toggles.js for the queue/
    # bypass/mute buttons on group headers, context.js for Context suggestions,
    # and resolution_latent.js for Resize Image / Empty Latent (Context)'s mode switcher.
    for _rel in (
        "lora_loader/lora_loader.js",
        "fast_groups.js",
        "group_header_toggles.js",
        "context/context.js",
        "resolution_latent/resolution_latent.js",
        "ksampler_context/ksampler_context.js",
    ):
        PromptServer.instance.routes.get(
            f"/extensions/{_FOLDER_NAME}/{_rel}"
        )(_serve_js_file(os.path.join(_NODE_DIR, *_rel.split("/"))))

    logging.info("[Gibby Nodes] Registered dedicated routes for JS files")
except Exception as e:
    logging.warning(f"[Gibby Nodes] Could not register dedicated JS route: {e}")


class GibbyNodesExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [GibbyAnySwitch, GibbyContext, GibbyContextLoader, GibbyContextOverride, H3PipeApply, H3PipeCreate, GibbyImageSaverContext, GibbyKSamplerContext, GibbyLoraLoader, PipeAny, GibbyReferenceLatentContext, GibbyEmptyLatentResolution, GibbySamplingParametersContext]


async def comfy_entrypoint() -> GibbyNodesExtension:
    return GibbyNodesExtension()

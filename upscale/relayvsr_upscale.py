"""
RelayVSR Upscale (Context): the whole RelayVSR video upscale in one node - the
keyframe conditioning, the single distilled sampling step, and the Flash decoder.

RelayVSR sets its own size: only every 15th frame goes through the Wan keyframe
model and the decoder restores the rest, so the input is used at its own
resolution, the model's x2/x4 factor is the output size, and nothing is tiled.
"""

import folder_paths
from comfy_api.latest import io
from comfy_extras.nodes_wan import RelayVSRConditioning, VSRDecode
from nodes import KSampler

from ..context import _CONTEXT_TYPE, ctx_from, ctx_set_sampled_image
from ..context.context_loader import _load_unet, _load_vae


def _relayvsr_files(folder):
    """The files of a model folder whose name mentions relayvsr. A relayvsr file
    named differently is still usable through the context with the list on None."""
    return [f for f in folder_paths.get_filename_list(folder) if "relayvsr" in f.lower()]


class GibbyRelayvsrUpscale(io.ComfyNode):
    """RelayVSR upscale of the context image: keyframe conditioning, one step, Flash decode."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        unets = _relayvsr_files("diffusion_models")
        vaes = _relayvsr_files("vae")
        return io.Schema(
            node_id="Gibby_Relayvsr_Upscale",
            display_name="RelayVSR Upscale (Context)",
            category="gibby",
            search_aliases=["relayvsr", "vsr", "upscale", "video upscale", "super resolution", "keyframe"],
            description=(
                "The whole RelayVSR video upscale in one node: keyframe conditioning, the single distilled "
                "sampling step, and the Flash decoder. Model and VAE must share one upscale factor."
            ),
            inputs=[
                _CONTEXT_TYPE.Input("context", optional=True, tooltip="Provides the image, model and VAE when the inputs below are unconnected; the output context carries the upscaled image"),
                io.Image.Input("image", optional=True, tooltip="Video (frames, H, W, C) at its own resolution; the context's image when unconnected. A (B, H, W, C) batch reads as B frames of one video, like the core RelayVSR nodes"),
                io.Combo.Input("unet_name", options=["None"] + unets, default=unets[0] if unets else "None", tooltip="RelayVSR diffusion model (files mentioning relayvsr); None uses the context's model. Its x2/x4 factor is the output size"),
                io.Combo.Input("vae_name", options=["None"] + vaes, default=vaes[0] if vaes else "None", tooltip="RelayVSR Flash decoder VAE, at the same factor as the model above; None uses the context's VAE"),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate="fixed", tooltip="The noise the one step starts from: a new seed re-synthesizes the texture, not the composition"),
                io.Float.Input("detail", default=1.0, min=0.0, max=1.0, step=0.01, tooltip="How much of the detail the decoder adds over the bicubic upscale is kept; 0 is a plain bicubic upscale"),
            ],
            outputs=[
                _CONTEXT_TYPE.Output(display_name="context"),
                io.Image.Output(display_name="image"),
                io.Audio.Output(display_name="audio"),
            ],
        )

    @classmethod
    def execute(cls, context=None, image=None, unet_name="None", vae_name="None",
                seed=0, detail=1.0) -> io.NodeOutput:
        ctx = ctx_from(context)
        if image is None:
            image = ctx.get("image")
        if image is None:
            raise ValueError("RelayVSR Upscale: nothing to upscale - connect an image or give the context one.")

        model = _load_unet(unet_name, "default") if unet_name != "None" else ctx.get("model")
        vae = _load_vae(vae_name) if vae_name != "None" else ctx.get("vae")
        if model is None:
            raise ValueError("RelayVSR Upscale: no RelayVSR model - pick unet_name or connect a context that has one.")
        if vae is None:
            raise ValueError("RelayVSR Upscale: no RelayVSR Flash decoder - pick vae_name or connect a context that has one.")

        # The core nodes read (frames, H, W, C): a (B, F, H, W, C) video flattens
        # to one video and is reshaped back after the decode. Alpha is dropped.
        lq = image[..., :3]
        if lq.dim() == 5:
            lq = lq.reshape(-1, *lq.shape[-3:])

        conditioning = RelayVSRConditioning.execute(model, lq)
        # cfg 1 never runs the negative, and the conditioning is a trained
        # embedding with no prompt to negate.
        sampled = KSampler().sample(model, seed, 1, 1.0, "euler", "simple",
                                    conditioning[0], conditioning[0], conditioning[1])[0]
        decoded = VSRDecode.execute(vae, sampled, lq, detail)[0]
        out = decoded.reshape(*image.shape[:2], *decoded.shape[-3:]) if image.dim() == 5 else decoded

        ctx_set_sampled_image(ctx, out)
        return io.NodeOutput(ctx, out, ctx.get("audio"))

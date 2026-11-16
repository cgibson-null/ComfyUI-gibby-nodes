"""
RelayVSR Upscale (Context)
--------------------------
The whole RelayVSR video upscale in one node: the keyframe conditioning, the
one step of sampling the model is trained for, and the Flash decoder - the
RelayVSR Conditioning node, the KSampler and the VSR Decode node, collapsed.

Model and VAE come from the node's own file lists, filtered to the relayvsr
files; "None" in either list takes the context's model or VAE instead. The two
have to be the same upscale factor. The image is the connected one, or the
context's when nothing is linked, and the result replaces the context's image.

RelayVSR sets its own size: only every 15th frame goes through the Wan
keyframe model and the Flash decoder restores all of them from those keyframes
and the low-quality video, so the input is used at its own resolution, there is
no target size to ask for and nothing to tile - the model's x2/x4 factor is the
output size, and the decoder windows over the frames by itself.

Sampling is what the model is trained for: euler/simple at cfg 1 and denoise
1.0, in the one step it was distilled for. None of it is a knob - at one step
every usable scheduler starts at the same sigma, cfg 1 has no negative to run
because the conditioning is a trained embedding rather than a prompt, and under
1.0 the step would start from a blend the model never saw. The latent is pure
noise, so the seed is the one degree of freedom the sampler keeps, and `detail`
is how much of the detail the decoder adds to the bicubic upscale of each frame
is kept.

The context's audio is handed straight back out: the upscale is spatial only and
keeps every frame, so a soundtrack still lines up with the result and does not
have to be routed around the node.
"""

import folder_paths
from comfy_api.latest import io
from comfy_extras.nodes_wan import RelayVSRConditioning, VSRDecode
from nodes import KSampler

from .context import _CONTEXT_TYPE, ctx_from, ctx_set_sampled_image
from .context.context_loader import _load_unet, _load_vae


def _relayvsr_files(folder):
    """The files of a model folder whose name mentions relayvsr - the only cheap
    way to offer just the RelayVSR models and Flash decoders without opening
    every checkpoint in the folder. A relayvsr file named differently is still
    usable through the context with the list on None."""
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
                "Runs the whole RelayVSR video upscale: the keyframe conditioning, the single step of "
                "sampling the model is trained for, and the Flash decoder that restores every frame from "
                "those keyframes and the low-quality video. "
                "Model and VAE come from the node's relayvsr-filtered lists and must be the same upscale "
                "factor; None takes the context's instead. The image is the connected one, or the context's, "
                "and the result replaces the context's image. "
                "The model decides the size - every frame is upscaled by the model's own x2/x4 factor, the "
                "input is used at its own resolution, and the decoder windows over the frames by itself, so "
                "there is nothing to tile. "
                "Sampling is fixed at euler/simple, 1 step, cfg 1 and denoise 1.0, which is what the model "
                "is distilled for; at one step every usable scheduler starts at the same sigma. The seed is "
                "the degree of freedom the sampler keeps, and detail is how much of the detail the decoder "
                "adds to the bicubic upscale of each frame is kept (0 = bicubic). "
                "The context's audio is output untouched - the upscale is spatial only and keeps every "
                "frame, so the soundtrack still lines up."
            ),
            inputs=[
                _CONTEXT_TYPE.Input("context", optional=True, tooltip="Provides the image, model and VAE when the inputs below are not connected; the output context carries the upscaled image"),
                io.Image.Input("image", optional=True, tooltip="Video (frames, H, W, C) to upscale at its own resolution; the context's image when unconnected. A (B, H, W, C) batch is read as B frames of one video, the way the core RelayVSR nodes read it"),
                io.Combo.Input("unet_name", options=["None"] + unets, default=unets[0] if unets else "None", tooltip="RelayVSR diffusion model (the files mentioning relayvsr); None uses the context's model. Its x2/x4 factor is the output size"),
                io.Combo.Input("vae_name", options=["None"] + vaes, default=vaes[0] if vaes else "None", tooltip="RelayVSR Flash decoder VAE (the files mentioning relayvsr), at the same factor as the model above; None uses the context's VAE"),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate="fixed", tooltip="The noise the one step starts from - the model adds detail on top of the low-quality frames, so a new seed re-synthesizes the texture rather than the composition"),
                io.Float.Input("detail", default=1.0, min=0.0, max=1.0, step=0.01, tooltip="How much of the detail the decoder adds to the bicubic upscale of each frame is kept; 1 is the full model output, 0 is a plain bicubic upscale"),
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

        # The core nodes read (frames, H, W, C), so a (B, F, H, W, C) video
        # flattens to one video and goes back to its own layout after the decode.
        # The Flash decoder reads the frames themselves, so an alpha is dropped.
        lq = image[..., :3]
        if lq.dim() == 5:
            lq = lq.reshape(-1, *lq.shape[-3:])

        conditioning = RelayVSRConditioning.execute(model, lq)
        # cfg 1 never runs the negative, and the conditioning is a trained
        # embedding with no prompt behind it, so there is nothing to negate.
        sampled = KSampler().sample(model, seed, 1, 1.0, "euler", "simple",
                                    conditioning[0], conditioning[0], conditioning[1])[0]
        decoded = VSRDecode.execute(vae, sampled, lq, detail)[0]
        out = decoded.reshape(*image.shape[:2], *decoded.shape[-3:]) if image.dim() == 5 else decoded

        ctx_set_sampled_image(ctx, out)
        return io.NodeOutput(ctx, out, ctx.get("audio"))

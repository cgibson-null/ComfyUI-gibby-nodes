"""
H3 Latent Upscale (Context): a MiniMax H3 clip's video upscaled inside the latent
space - the core H3 latent upscaler, its loader, and the megapixels/scale size
math of Resize Image / Empty Latent (Context) in one node.

retain_audio puts a denoise mask on the output latent (1 over video, 0 over
audio) so the sampling pass that follows regenerates the picture and returns the
soundtrack as it was handed over.
"""

import folder_paths
import comfy.nested_tensor
from comfy_api.latest import io
from comfy_extras.nodes_hunyuan import LatentUpscaleModelLoader
from comfy_extras.nodes_minimax_h3 import CANVAS_MULTIPLE, MiniMaxH3LatentUpscaleWithModel

from ..context import WeakCache, _CONTEXT_TYPE, ctx_from, ctx_set_latent
from ..resolution_latent import _resize_to_mp_scale

# No loader node in the graph for ComfyUI to cache, so without this the file is
# re-read on every execution.
_last_upscaler = WeakCache()  # model_name -> (model,)


def _h3_upscaler_files():
    """The latent_upscale_models files whose name mentions minimax_h3."""
    return [f for f in folder_paths.get_filename_list("latent_upscale_models")
            if "minimax_h3" in f.lower()] or ["(no minimax_h3 latent upscale model found)"]


class GibbyH3LatentUpscale(io.ComfyNode):
    """Upscale the video of a MiniMax H3 AV latent with the H3 latent upscaler."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        models = _h3_upscaler_files()
        return io.Schema(
            node_id="Gibby_H3_Latent_Upscale",
            display_name="H3 Latent Upscale (Context)",
            category="gibby",
            search_aliases=["minimax h3", "latent upscale", "video upscale", "av latent", "audio mask"],
            description=(
                "Upscales the video of a MiniMax H3 AV latent with the H3 latent upscaler, "
                "without leaving latent space. The audio half passes through unchanged."
            ),
            inputs=[
                _CONTEXT_TYPE.Input("context", optional=True, tooltip=(
                    "The output context carries the upscaled latent and its size; nothing is read from it.")),
                io.Latent.Input("av_latent", tooltip=(
                    "The clip's H3 AV latent - a sampler output, the nested video/audio pair, not the empty "
                    "latent from H3 Pipe Apply.")),
                io.Combo.Input("latent_upscale_model", options=models, default=models[0], tooltip=(
                    "The H3 latent upscaler (latent_upscale_models files mentioning minimax_h3)")),
                io.Float.Input("megapixels", default=0.0, min=0.0, max=64.0, step=0.01, tooltip=(
                    "Target total pixels at the latent's aspect ratio (0 = its own size); "
                    "scale_factor applies on top")),
                io.Float.Input("scale_factor", default=2.0, min=0.05, max=16.0, step=0.01, tooltip=(
                    "Applied after megapixels")),
                io.Boolean.Input("retain_audio", display_name="Retain audio", default=True, tooltip=(
                    "Mask the audio stream off (0 over audio, 1 over video) so the sampling pass "
                    "that follows returns it untouched; off regenerates both streams")),
            ],
            outputs=[
                _CONTEXT_TYPE.Output(display_name="context"),
                io.Latent.Output(display_name="av_latent"),
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
            ],
        )

    @classmethod
    def execute(cls, context=None, av_latent=None, latent_upscale_model="", megapixels=0.0,
                scale_factor=2.0, retain_audio=True) -> io.NodeOutput:
        ctx = ctx_from(context)
        # A copy: the core node returns its input untouched when the size does not
        # change, and the mask must not land on the caller's latent.
        latent = dict(av_latent) if isinstance(av_latent, dict) else {"samples": av_latent}
        samples = latent["samples"]
        if not getattr(samples, "is_nested", False) or len(samples.tensors) < 2:
            raise ValueError(
                "H3 Latent Upscale: av_latent is not a MiniMax H3 AV latent (a nested video/audio "
                "pair). Wire the sampler output of an H3 graph.")

        # The latent is /16 spatially; the size math works in pixels like every
        # other node's and lands on the H3 canvas grid.
        video = samples.tensors[0]
        w, h = video.shape[-1] * 16, video.shape[-2] * 16
        tw, th = _resize_to_mp_scale(w, h, megapixels, scale_factor, CANVAS_MULTIPLE)

        model, = _last_upscaler.get_or_load(
            latent_upscale_model, lambda: LatentUpscaleModelLoader.execute(latent_upscale_model))
        out, = MiniMaxH3LatentUpscaleWithModel.execute(model, latent, tw, th)

        if retain_audio:
            # The sampler composites out * mask + latent_image * (1 - mask), so an
            # all-zero audio mask hands the soundtrack back exactly as it came in.
            out_v, out_a = out["samples"].tensors
            out["noise_mask"] = comfy.nested_tensor.NestedTensor(
                (out_v.new_ones((1, 1) + tuple(out_v.shape[2:])),
                 out_a.new_zeros((1, 1) + tuple(out_a.shape[2:]))))

        # Only the latent goes on the context: the width/height sockets carry the
        # new size, and the context's image is still the pre-upscale one, so
        # overriding them would make ctx_size lie about it.
        ctx_set_latent(ctx, out)
        return io.NodeOutput(ctx, out, tw, th)

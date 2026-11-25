"""
SeedVR2 Upscale (Context): the whole SeedVR2 upscale in one node - resize, pad,
encode, sample with that latent as the conditioning, decode, color-correct, and
split a long video into overlapping temporal chunks around the sampling.

Target size is megapixels (0 = the input's own pixel count) times scale_factor,
floored to the VAE's spatial downscale, exactly like Resize Image / Empty
Latent (Context); a target_size_image overrides both.
"""

import time

import folder_paths
from comfy_api.latest import io
from comfy.ldm.seedvr.constants import BYTEDANCE_VAE_SPATIAL_DOWNSAMPLE
from comfy_extras.nodes_seedvr import (
    SeedVR2Conditioning,
    SeedVR2PostProcessing,
    SeedVR2Preprocess,
    SeedVR2TemporalChunk,
    SeedVR2TemporalMerge,
)
from nodes import KSampler

from ..context import (
    _CONTEXT_TYPE, _image_count_s, _image_dims, _log, _mp, clear_vram,
    ctx_from, ctx_set_sampled_image, decode_latent, encode_image,
)
from ..context.context_loader import _load_unet, _load_vae
from ..resolution_latent import _resize_image, _resize_to_mp_scale


def _seedvr2_files(folder):
    """The files of a model folder whose name mentions seedvr. A seedvr file named
    differently is still usable through the context with the list on None."""
    return [f for f in folder_paths.get_filename_list(folder) if "seedvr" in f.lower()]


def _chunking_inputs(with_frames):
    """The widgets the chunking_mode options carry: temporal_overlap belongs to
    both of them, frames_per_chunk only to manual."""
    inputs = [io.Int.Input("temporal_overlap", default=0, min=0, max=16384,
                           tooltip="Latent frames shared between adjacent chunks and crossfaded at the merge; 0 = no overlap")]
    if with_frames:
        inputs.append(io.Int.Input("frames_per_chunk", default=21, min=1, max=16384, step=4,
                                   tooltip="Pixel frames per temporal chunk (4n+1: 1, 5, 9, 13, ...)"))
    return inputs


class GibbySeedvr2Upscale(io.ComfyNode):
    """SeedVR2 upscale of the context image: resize, encode, sample, decode, color-match."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        unets = _seedvr2_files("diffusion_models")
        vaes = _seedvr2_files("vae")
        return io.Schema(
            node_id="Gibby_Seedvr2_Upscale",
            display_name="SeedVR2 Upscale (Context)",
            category="gibby",
            search_aliases=["seedvr2", "upscale", "video upscale", "color correction", "chunk", "temporal"],
            description=(
                "The whole SeedVR2 upscale in one node: resize, encode, sample with that latent as the "
                "conditioning, decode, and color-correct back to the input."
            ),
            inputs=[
                _CONTEXT_TYPE.Input("context", optional=True, tooltip="Provides the image, model and VAE when the inputs below are unconnected; the output context carries the upscaled image"),
                io.Image.Input("image", optional=True, tooltip="Image or video (frames, H, W, C) to upscale; the context's image when unconnected"),
                io.Combo.Input("unet_name", options=["None"] + unets, default=unets[0] if unets else "None", tooltip="SeedVR2 diffusion model (files mentioning seedvr); None uses the context's model"),
                io.Combo.Input("vae_name", options=["None"] + vaes, default=vaes[0] if vaes else "None", tooltip="SeedVR2 VAE (files mentioning seedvr); None uses the context's VAE"),
                # Target size - the Resize Image / Empty Latent (Context) math.
                io.Float.Input("megapixels", default=0.0, min=0.0, max=64.0, step=0.01, tooltip="Total pixels at the input's aspect ratio (0 = its own size); scale_factor applies on top"),
                io.Float.Input("scale_factor", default=2.0, min=0.05, max=16.0, step=0.01, tooltip="Resolution multiplier applied after megapixels; overridden by target_size_image"),
                io.Combo.Input("upscale_method", options=["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], default="lanczos"),
                io.Image.Input("target_size_image", optional=True, tooltip="Image defining the final size: the input is resized to its width x height instead of megapixels and scale_factor"),
                io.Combo.Input("color_correction_method", options=["lab", "wavelet", "adain", "none"], default="lab", tooltip="Match the result's colors to the resized input: lab is the most faithful, wavelet keeps the upscaled detail, adain is fastest but a global tint"),
                io.DynamicCombo.Input("chunking_mode", tooltip="none samples the whole clip at once; auto sizes the chunk to the free VRAM; manual uses frames_per_chunk.", options=[
                    io.DynamicCombo.Option("none", []),
                    io.DynamicCombo.Option("auto", _chunking_inputs(False)),
                    io.DynamicCombo.Option("manual", _chunking_inputs(True)),
                ]),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate="fixed", tooltip="The noise the sampled latent starts from; every chunk of one run shares it"),
                io.Int.Input("steps", default=1, min=1, max=100, advanced=True, tooltip="SeedVR2 is a one-step model; more steps cost a full extra pass and rarely help"),
                io.Boolean.Input("tiled", default=True, tooltip="Encode and decode with the tiled VAE - the safe path at upscale sizes; off uses the plain VAE nodes"),
                io.Int.Input("vae_tile_size", default=512, min=64, max=4096, step=64, advanced=True),
                io.Int.Input("vae_tile_overlap", default=64, min=0, max=4096, step=32, advanced=True),
                io.Int.Input("vae_temporal_size", default=64, min=8, max=4096, step=4, advanced=True, tooltip="Tiled VAE: frames to encode/decode at a time (video VAEs only)"),
                io.Int.Input("vae_temporal_overlap", default=8, min=4, max=4096, step=4, advanced=True),
                io.Boolean.Input("clean_vram", default=False, tooltip="Unload all models and empty the cache before the SeedVR2 model loads"),
                io.Boolean.Input("verbose", default=False, advanced=True),
            ],
            outputs=[
                _CONTEXT_TYPE.Output(display_name="context"),
                io.Image.Output(display_name="image"),
            ],
        )

    @classmethod
    def execute(cls, context=None, image=None, unet_name="None", vae_name="None",
                megapixels=0.0, scale_factor=2.0, upscale_method="lanczos", target_size_image=None,
                color_correction_method="lab", chunking_mode=None, seed=0, steps=1, tiled=True,
                vae_tile_size=512, vae_tile_overlap=64, vae_temporal_size=64, vae_temporal_overlap=8,
                clean_vram=False, verbose=False) -> io.NodeOutput:
        t0 = time.time()
        if clean_vram:
            clear_vram("at start", verbose)
        ctx = ctx_from(context)
        if image is None:
            image = ctx.get("image")
        if image is None:
            raise ValueError("SeedVR2 Upscale: nothing to upscale - connect an image or give the context one.")

        model = _load_unet(unet_name, "default") if unet_name != "None" else ctx.get("model")
        vae = _load_vae(vae_name) if vae_name != "None" else ctx.get("vae")
        if model is None:
            raise ValueError("SeedVR2 Upscale: no SeedVR2 model - pick unet_name or connect a context that has one.")
        if vae is None:
            raise ValueError("SeedVR2 Upscale: no SeedVR2 VAE - pick vae_name or connect a context that has one.")

        # The resized input is both what gets encoded and the reference the result
        # is aligned and color-matched to.
        w, h, _ = _image_dims(image)
        if target_size_image is not None:
            tw, th, _ = _image_dims(target_size_image)
        else:
            tw, th = _resize_to_mp_scale(w, h, megapixels, scale_factor, BYTEDANCE_VAE_SPATIAL_DOWNSAMPLE)
        resized = image if (tw, th) == (w, h) else _resize_image(image, tw, th, upscale_method)

        padded = SeedVR2Preprocess.execute(resized)[0]
        latent = encode_image(vae, padded, tiled, vae_tile_size, vae_tile_overlap,
                              vae_temporal_size, vae_temporal_overlap, verbose)

        chunking_mode = chunking_mode or {"chunking_mode": "none"}
        if chunking_mode["chunking_mode"] == "none":
            chunks, overlap = [latent], 0
        else:
            chunks, overlap = SeedVR2TemporalChunk.execute(latent, chunking_mode.get("temporal_overlap", 0), chunking_mode)
        if verbose:
            _log("SeedVR2 start", f"{w}x{h} -> {tw}x{th} ({_mp(tw, th)} Mp){_image_count_s(resized)} "
                                  f"chunks={len(chunks)} overlap={overlap} steps={steps} tiled={tiled} "
                                  f"color={color_correction_method} seed={seed}")

        sampled = []
        for chunk in chunks:
            cond = SeedVR2Conditioning.execute(model, chunk)
            sampled.append(KSampler().sample(model, seed, steps, 1.0, "euler", "simple",
                                             cond[0], cond[1], chunk)[0])
        latent = sampled[0] if len(sampled) == 1 else SeedVR2TemporalMerge.execute(sampled, [overlap])[0]

        image = decode_latent(vae, latent, tiled, vae_tile_size, vae_tile_overlap,
                              vae_temporal_size, vae_temporal_overlap, verbose)
        image = SeedVR2PostProcessing.execute(image, resized, color_correction_method)[0]

        ctx_set_sampled_image(ctx, image)
        if verbose:
            ow, oh, _ = _image_dims(image)
            _log("SeedVR2 finish", f"total_time={time.time() - t0:.2f}s resolution={ow}x{oh} "
                                   f"({_mp(ow, oh)} Mp){_image_count_s(image)}")
        return io.NodeOutput(ctx, image)

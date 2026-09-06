"""
Reference Latent (Context) node
-------------------------------
Sets reference latents on conditioning from provided images.

Each image is resized to 1 megapixel, then scaled by the provided scale factor,
encoded using the context's VAE, and set as reference_latents on the positive
conditioning (and negative if cfg != 1).
"""

import math

import torch
import comfy.utils
import node_helpers
from comfy_api.latest import io

from .context import _CONTEXT_TYPE


def _resize_to_megapixels(image, target_mp=1.0):
    """Resize image to target megapixels while preserving aspect ratio."""
    h, w = image.shape[1], image.shape[2]
    pixels = h * w
    target_pixels = target_mp * 1024 * 1024
    
    if pixels <= target_pixels:
        return image
    
    scale = math.sqrt(target_pixels / pixels)
    new_h = max(1, int(h * scale))
    new_w = max(1, int(w * scale))
    
    # Move to [C, H, W] for comfy.utils.common_upscale
    samples = image.movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, new_w, new_h, "bilinear", "disabled")
    # Move back to [B, H, W, C]
    return samples.movedim(1, -1)


class GibbyReferenceLatentContext(io.ComfyNode):
    """Set reference latents on conditioning from provided images."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Gibby_ReferenceLatent_Context",
            display_name="Reference Latent (Context)",
            category="gibby",
            description=(
                "Sets reference latents on conditioning from provided images. "
                "Each image is resized to 1MP, scaled by the scale factor, "
                "encoded with the context VAE, and referenced on positive/negative conditioning."
            ),
            inputs=[
                _CONTEXT_TYPE.Input("context"),
                io.Autogrow.Input(
                    "images",
                    template=io.Autogrow.TemplateNames(
                        io.Image.Input("image"),
                        names=[f"image_{i}" for i in range(1, 26)],  # Up to 25 images
                        min=0,
                    ),
                    tooltip="Reference images to encode as reference latents",
                ),
                io.Float.Input(
                    "scale", 
                    default=1.0, 
                    min=0.0, 
                    max=100.0, 
                    step=0.01,
                    tooltip="Scale factor applied after resizing to 1MP"
                ),
            ],
            outputs=[
                _CONTEXT_TYPE.Output(display_name="context"),
            ],
        )

    @classmethod
    def execute(cls, context, images: io.Autogrow.Type = None, scale=1.0) -> io.NodeOutput:
        # Filter out None images
        images = images or {}
        images = [images[name] for name in sorted(images, key=lambda n: int(n.rsplit("_", 1)[-1])) if images[name] is not None]

        if not images:
            return io.NodeOutput(context)

        vae = context.get("vae")
        if vae is None:
            return io.NodeOutput(context)

        positive = context.get("positive")
        negative = context.get("negative")
        cfg = context.get("cfg", 1.0)

        ref_latents = []

        # Process each image: resize to 1MP, apply scale, encode
        for image in images:
            # Resize to 1MP
            resized = _resize_to_megapixels(image, target_mp=1.0)

            # Apply scale factor to dimensions
            h, w = resized.shape[1], resized.shape[2]
            new_h = max(1, int(h * scale))
            new_w = max(1, int(w * scale))

            # Resize to final dimensions
            samples = resized.movedim(-1, 1)
            samples = comfy.utils.common_upscale(samples, new_w, new_h, "bilinear", "disabled")

            # Encode using VAE
            latent = vae.encode(samples.movedim(1, -1)[:, :, :, :3])
            ref_latents.append(latent)

        # Update positive conditioning
        if positive is not None and ref_latents:
            context["positive"] = node_helpers.conditioning_set_values(positive, {"reference_latents": ref_latents})

        # Update negative conditioning if cfg != 1
        if negative is not None and cfg != 1.0 and ref_latents:
            context["negative"] = node_helpers.conditioning_set_values(negative, {"reference_latents": ref_latents})

        return io.NodeOutput(context)

"""
Reference Latent (Context) node
-------------------------------
Sets reference latents on conditioning from provided images.

Each image is resized to the megapixels target, then scaled by the provided
scale factor, encoded using the context's VAE, and set as reference_latents
on the positive conditioning (and negative if cfg != 1).

For Qwen Image 2.1 the size is floored to a multiple of 32 so every vision
slot covers 2x2 latents, and the context prompts are re-encoded with the same
resized images so the text encoder sees them and the model splices the
latents at the vision slots; other models (flux2/klein) take the latents as-is.
"""

import comfy.utils
import comfy.model_base
import node_helpers
from comfy_api.latest import io

from .context import _CONTEXT_TYPE, recondition_prompts
from .resolution_latent import _resize_to_mp_scale


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
                "Each image is resized to the megapixels target (0 = its own size), "
                "scaled by the scale factor, encoded with the context VAE, and "
                "referenced on positive/negative conditioning. "
                "With Qwen Image 2.1 the prompts are also re-encoded with the images."
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
                    "megapixels",
                    default=1.0,
                    min=0.0,
                    max=20.0,
                    step=0.01,
                    tooltip="Target size in megapixels before the scale factor; 0 keeps each image's own size"
                ),
                io.Float.Input(
                    "scale", 
                    default=1.0, 
                    min=0.0, 
                    max=100.0, 
                    step=0.01,
                    tooltip="Scale factor applied after resizing to the megapixels target"
                ),
            ],
            outputs=[
                _CONTEXT_TYPE.Output(display_name="context"),
            ],
        )

    @classmethod
    def execute(cls, context, images: io.Autogrow.Type = None, megapixels=1.0, scale=1.0) -> io.NodeOutput:
        # Filter out None images
        images = images or {}
        images = [images[name] for name in sorted(images, key=lambda n: int(n.rsplit("_", 1)[-1])) if images[name] is not None]

        if not images:
            return io.NodeOutput(context)

        vae = context.get("vae")
        if vae is None:
            return io.NodeOutput(context)

        model = context.get("model")
        is_qwen21 = model is not None and hasattr(model, "model") and isinstance(model.model, comfy.model_base.QwenImage21)

        positive = context.get("positive")
        negative = context.get("negative")
        cfg = context.get("cfg", 1.0)

        ref_latents = []
        images_vl = []

        # Process each image: resize to the megapixels target, apply scale, encode
        for image in images:
            # channels-last [B,H,W,C]: w is shape[2], h is shape[1]. qwen2.1 floors to
            # a multiple of 32 so every vision slot covers 2x2 latents (Text Encode Qwen Image 2.1)
            tw, th = _resize_to_mp_scale(image.shape[2], image.shape[1], megapixels, scale, 32 if is_qwen21 else 1)
            resized = image
            if (tw, th) != (image.shape[2], image.shape[1]):
                resized = comfy.utils.common_upscale(image.movedim(-1, 1), tw, th, "bilinear", "disabled").movedim(1, -1)

            if is_qwen21:
                # the vision tower sees alpha over white, the vae keeps all four
                rgb = resized[:, :, :, :3]
                if resized.shape[-1] > 3:
                    rgb = rgb * resized[:, :, :, 3:] + (1.0 - resized[:, :, :, 3:])
                images_vl.append(rgb)
                latent = vae.encode(resized)
            else:
                latent = vae.encode(resized[:, :, :, :3])
            ref_latents.append(latent)

        # qwen2.1 splices the latents at the vision slots, so the text must be encoded with the images
        if is_qwen21:
            positive, negative = recondition_prompts(context, context.get("positive_prompt", ""), context.get("negative_prompt", ""), images=images_vl)

        # Update positive conditioning
        if positive is not None and ref_latents:
            context["positive"] = node_helpers.conditioning_set_values(positive, {"reference_latents": ref_latents})

        # Update negative conditioning if cfg != 1
        if negative is not None and cfg != 1.0 and ref_latents:
            context["negative"] = node_helpers.conditioning_set_values(negative, {"reference_latents": ref_latents})

        return io.NodeOutput(context)

"""
Reference Latent (Context) node
-------------------------------
Sets reference latents on conditioning from provided images.

Each image is resized to the megapixels target, then scaled by the provided
scale factor, encoded using the context's VAE, and set as reference_latents
on the positive conditioning (and negative if cfg != 1).

For Qwen Image 2.1 the size is floored to 2x the VAE's downscale ratio so
every vision slot covers 2x2 latents, and the context prompts are re-encoded
with the same resized images so the text encoder sees them and the model
splices the latents at the vision slots; other models (flux2/klein) floor to
the VAE ratio and take the latents as-is.

With ref_ctx_img the context's own image is referenced too: always
when no image input is connected (at its own size, no rescale), and with image
inputs connected only when the toggle is on - then it leads the reference list
at its own size, and when the context has no image the first image input becomes
the context image and is stored back on the output context.
"""

import comfy.model_base
import node_helpers
from comfy_api.latest import io

from .context import _CONTEXT_TYPE, recondition_prompts, _latent_downscale, ctx_from
from .resolution_latent import _resize_image_to_mp


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
                "With Qwen Image 2.1 the prompts are also re-encoded with the images. "
                "ref_ctx_img also references the context's own image: always when "
                "no image is connected, otherwise only when enabled - it leads at its own size, "
                "and with no context image the first image becomes the context image and is stored back."
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
                io.Boolean.Input(
                    "ref_ctx_img",
                    display_name="ref_ctx_img",
                    default=False,
                    tooltip="Also reference the context's own image at its own size (no rescale): always when no image is connected, otherwise only when enabled. With no context image the first image becomes the context image and is stored back on the output context."
                ),
            ],
            outputs=[
                _CONTEXT_TYPE.Output(display_name="context"),
            ],
        )

    @classmethod
    def execute(cls, context, images: io.Autogrow.Type = None, megapixels=1.0, scale=1.0, ref_ctx_img=False) -> io.NodeOutput:
        # The wired image_N, in order (None slots dropped).
        images = images or {}
        explicit = [images[name] for name in sorted(images, key=lambda n: int(n.rsplit("_", 1)[-1])) if images[name] is not None]

        # The context image takes the main slot: always when no image_N is wired
        # (toggle irrelevant), otherwise only with ref_ctx_img on - then it leads
        # the reference list at its own size, or, when the context has no image,
        # image_1 becomes the context image and is stored back on the output.
        ctx_image = context.get("image")
        if not explicit:
            main, extras = ctx_image, []
        elif ref_ctx_img and ctx_image is not None:
            main, extras = ctx_image, explicit
        elif ref_ctx_img:
            main, extras = explicit[0], explicit[1:]
        else:
            main, extras = None, explicit

        vae = context.get("vae")
        if (main is None and not extras) or vae is None:
            return io.NodeOutput(ctx_from(context))

        model = context.get("model")
        is_qwen21 = model is not None and hasattr(model, "model") and isinstance(model.model, comfy.model_base.QwenImage21)
        # Floor refs to the VAE's downscale ratio so the encode doesn't truncate them;
        # qwen2.1 uses 2x so every vision slot covers 2x2 latents (Text Encode Qwen Image 2.1)
        dw, _ = _latent_downscale(vae, None)
        multiple = 2 * dw if is_qwen21 else dw

        positive = context.get("positive")
        negative = context.get("negative")
        cfg = context.get("cfg", 1.0)

        ref_latents = []
        images_vl = []

        def encode_ref(resized):
            # the vision tower sees alpha over white, the vae keeps all four
            if is_qwen21:
                rgb = resized[:, :, :, :3]
                if resized.shape[-1] > 3:
                    rgb = rgb * resized[:, :, :, 3:] + (1.0 - resized[:, :, :, 3:])
                images_vl.append(rgb)
                return vae.encode(resized)
            return vae.encode(resized[:, :, :, :3])

        # The context image is referenced at its own size; the wired image_N to the target.
        if main is not None:
            main_resized = _resize_image_to_mp(main, 0.0, 1.0, multiple)
            ref_latents.append(encode_ref(main_resized))
        for image in extras:
            ref_latents.append(encode_ref(_resize_image_to_mp(image, megapixels, scale, multiple)))

        # qwen2.1 splices the latents at the vision slots, so the text must be encoded with the images
        if is_qwen21:
            positive, negative = recondition_prompts(context, context.get("positive_prompt", ""), context.get("negative_prompt", ""), images=images_vl)

        # Copy before mutating: the engine's output cache can hand the same
        # context object to other consumers and to repeated runs.
        ctx = ctx_from(context)
        if main is not None:
            ctx["image"] = main_resized

        # Update positive conditioning
        if positive is not None and ref_latents:
            ctx["positive"] = node_helpers.conditioning_set_values(positive, {"reference_latents": ref_latents})

        # Update negative conditioning if cfg != 1
        if negative is not None and cfg != 1.0 and ref_latents:
            ctx["negative"] = node_helpers.conditioning_set_values(negative, {"reference_latents": ref_latents})

        return io.NodeOutput(ctx)

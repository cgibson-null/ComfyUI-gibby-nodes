"""
Reference Latent (Context) node
-------------------------------
Sets reference latents on conditioning from provided images.

Each image is resized to the megapixels target, then scaled by the provided
scale factor, encoded using the context's VAE, and set as reference_latents
on the positive conditioning (and negative if cfg != 1).

Two independent paths, picked by the model and by its text encoder:

- reference latents (ref_latents, on by default): the image is VAE-encoded and
  set on the conditioning, which is what the edit models were trained to take
  (flux2/klein, flux3, qwen image edit, qwen 2.1). Off for a model that only has
  the latent-concat path by accident (krea) - it was never trained on it.
- the text encoder seeing the images: any model whose encoder runs a vision tower
  (the Qwen-VL family: qwen image, qwen 2.1, krea2, mage, boogu, ideogram4,
  joyimage, kandinsky5) re-encodes the context prompts with the images at
  vl_megapixels, so a t2i model conditions on them with no edit path at all.
  Flux3 is left out - its Qwen3-VL encoder is text-only by training and its DiT
  splices nothing. A tokenizer with no image template drops the images and warns
  about it (MOSS-Image, and the passthrough encoders below).

HiDream-O1 and SenseNova take the reference under reference_latents too, but as
raw images rather than latents: their encoders are passthroughs that emit token
ids and the DiT does the encoding. They get the resized image unchanged, which is
what core's HiDream-O1 Reference Images attaches, and need no VAE. (That node also
marks the negative prompt_type=negative - an image-only pass - which this one
leaves to whoever owns the negative.)

Qwen Image 2.1 splices the latents at the vision slots, so it floors the size to
2x the VAE's downscale ratio - every slot covers 2x2 latents - and shares one size
between the encoder and the VAE. Everything else floors to the VAE ratio and
resizes the encoder's copy separately (flux2/klein and flux3 take the latents
as-is; flux3's text encoder is text-only and its DiT patchifies nothing, so the
ratio alone is the whole constraint).

With ref_ctx_img the context's own image is referenced too: always
when no image input is connected (at its own size, no rescale), and with image
inputs connected only when the toggle is on - then it leads the reference list
at its own size, and when the context has no image the first image input becomes
the context image and is stored back on the output context.

A media pipe's reference images are referenced too - the wired images append
to them in the output pipe; keyframes/videos/audios pass through untouched.

Without a context the node is just a media pipe creator: the wired images
append to the pipe and the context passes through empty.
"""

import comfy.model_base
import node_helpers
from comfy_api.latest import io

from .context import _CONTEXT_TYPE, _clip_takes_images, _is_model, recondition_prompts, _latent_downscale, ctx_from, ctx_set_image
from .pipes.media_pipe import iter_numbered_slots, pipe_add_images
from .resolution_latent import _resize_image_to_mp


def _over_white(image):
    """The image with its alpha composited over white; the vision tower has no
    alpha channel, while the VAE keeps all four."""
    rgb = image[:, :, :, :3]
    if image.shape[-1] > 3:
        rgb = rgb * image[:, :, :, 3:] + (1.0 - image[:, :, :, 3:])
    return rgb


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
                "A text encoder with a vision tower also sees the images, at "
                "vl_megapixels, so a t2i model (krea) conditions on them with "
                "ref_latents off. HiDream-O1 and SenseNova get the image itself "
                "rather than a latent, which is what their models read. "
                "ref_ctx_img also references the context's own image: always when "
                "no image is connected, otherwise only when enabled - it leads at its own size, "
                "and with no context image the first image becomes the context image and is stored back. "
                "A media pipe's reference images are referenced too - the wired images "
                "append to them in the output pipe. Without a context the node is just "
                "a media pipe creator: the wired images append to the pipe."
            ),
            inputs=[
                # All inputs optional, so the frontend renders them in declaration
                # order and media_pipe lands right under context
                _CONTEXT_TYPE.Input("context", optional=True, tooltip=(
                    "Base context; without it the node just builds a media pipe "
                    "from the connected images")),
                io.Dict.Input("media_pipe", optional=True, tooltip=(
                    "Media pipe - its reference images are referenced too (the wired images "
                    "append to them in the output pipe); keyframes/videos/audios pass through untouched.")),
                io.Autogrow.Input(
                    "images",
                    template=io.Autogrow.TemplateNames(
                        io.Image.Input("image", optional=True),
                        names=[f"image_{i}" for i in range(1, 26)],  # Up to 25 images
                        min=0,
                    ),
                    optional=True,
                    tooltip="Reference images to encode as reference latents",
                ),
                io.Float.Input(
                    "megapixels",
                    default=1.0,
                    min=0.0,
                    max=20.0,
                    step=0.01,
                    optional=True,
                    tooltip="Target size in megapixels before the scale factor; 0 keeps each image's own size"
                ),
                io.Float.Input(
                    "scale",
                    default=1.0,
                    min=0.0,
                    max=100.0,
                    step=0.01,
                    optional=True,
                    tooltip="Scale factor applied after resizing to the megapixels target"
                ),
                io.Boolean.Input(
                    "ref_latents",
                    default=True,
                    optional=True,
                    tooltip="VAE-encode the images and reference them on the conditioning: on for the edit models (flux2/klein, flux3, qwen image edit, qwen 2.1). Off for a model that was never trained to take them (krea) - its text encoder still sees the images."
                ),
                io.Float.Input(
                    "vl_megapixels",
                    default=0.15,
                    min=0.0,
                    max=20.0,
                    step=0.01,
                    optional=True,
                    tooltip="Size the text encoder sees the images at; 0 reuses the VAE-sized image. Ignored by qwen 2.1, whose encoder and VAE must share one size so every vision slot covers 2x2 latents."
                ),
                io.Boolean.Input(
                    "ref_ctx_img",
                    display_name="ref_ctx_img",
                    default=False,
                    optional=True,
                    tooltip="Also reference the context's own image at its own size (no rescale): always when no image is connected, otherwise only when enabled. With no context image the first image becomes the context image and is stored back on the output context."
                ),
            ],
            outputs=[
                _CONTEXT_TYPE.Output(display_name="context"),
                io.Dict.Output(display_name="media_pipe"),
            ],
        )

    @classmethod
    def execute(cls, context=None, media_pipe=None, images: io.Autogrow.Type = None, megapixels=1.0, scale=1.0, ref_latents=True, vl_megapixels=0.15, ref_ctx_img=False) -> io.NodeOutput:
        # Without a context there is no VAE to encode with: the node just builds
        # the media pipe from the connected images
        context = context or {}
        # The wired image_N, in order (None slots dropped).
        images = images or {}
        explicit = [value for _index, _name, value in iter_numbered_slots(images, "image")]

        # Pipe refs lead the reference list, the wired images append to them.
        pipe_refs = [value for _index, _name, value in iter_numbered_slots((media_pipe or {}).get("ref_images") or {}, "ref_image")]

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
        model = context.get("model")
        # hidream-o1 and sensenova read raw images off reference_latents and do their
        # own patch-grid conditioning, so they get the pixels and no VAE at all
        raw_refs = _is_model(model, (comfy.model_base.HiDreamO1, comfy.model_base.SenseNovaU15))
        if (main is None and not extras and not pipe_refs) or (vae is None and ref_latents and not raw_refs):
            return io.NodeOutput(ctx_from(context), pipe_add_images(media_pipe, explicit))

        is_qwen21 = _is_model(model, comfy.model_base.QwenImage21)
        # flux3's Qwen3-VL encoder can see images but was trained text-only, and its
        # DiT splices nothing into the sequence, so it takes the latents alone
        vl_refs = _clip_takes_images(context.get("clip")) and not _is_model(model, comfy.model_base.Flux3)
        # Floor refs to the VAE's downscale ratio so the encode doesn't truncate them;
        # qwen2.1 uses 2x so every vision slot covers 2x2 latents (Text Encode Qwen Image 2.1)
        dw, _ = _latent_downscale(vae, None)
        multiple = 2 * dw if is_qwen21 else dw

        positive = context.get("positive")
        negative = context.get("negative")
        cfg = context.get("cfg", 1.0)

        refs = []
        images_vl = []

        def add_ref(resized):
            if vl_refs:
                # qwen2.1 must see the exact image the VAE encodes; the others take
                # the much smaller copy core's qwen image edit nodes feed the tower
                vl = resized if is_qwen21 or vl_megapixels <= 0 else _resize_image_to_mp(resized, vl_megapixels, 1.0, dw)
                images_vl.append(_over_white(vl))
            if ref_latents:
                if raw_refs:
                    refs.append(resized)
                else:
                    refs.append(vae.encode(resized if is_qwen21 else resized[:, :, :, :3]))

        # The context image is referenced at its own size; the pipe refs and
        # the wired image_N to the target.
        if main is not None:
            main_resized = _resize_image_to_mp(main, 0.0, 1.0, multiple)
            add_ref(main_resized)
        for image in pipe_refs:
            add_ref(_resize_image_to_mp(image, megapixels, scale, multiple))
        for image in extras:
            add_ref(_resize_image_to_mp(image, megapixels, scale, multiple))

        # qwen2.1 splices the latents at the vision slots, so the text must be encoded
        # with the images and the tokens dropped; the other encoders keep the tokens and
        # condition on them, which is all a model without the edit path can do.
        if images_vl:
            positive, negative = recondition_prompts(context, context.get("positive_prompt", ""), context.get("negative_prompt", ""), images=images_vl, keep_vision=not (is_qwen21 and ref_latents))

        # Copy before mutating: the engine's output cache can hand the same
        # context object to other consumers and to repeated runs.
        ctx = ctx_from(context)
        if main is not None:
            # The context's image becomes the resized one, so the latent and the
            # sampler's cached sample built from the unsized image are stale
            ctx_set_image(ctx, main_resized)

        # Update positive conditioning
        if positive is not None and refs:
            ctx["positive"] = node_helpers.conditioning_set_values(positive, {"reference_latents": refs})

        # Update negative conditioning if cfg != 1
        if negative is not None and cfg != 1.0 and refs:
            ctx["negative"] = node_helpers.conditioning_set_values(negative, {"reference_latents": refs})

        return io.NodeOutput(ctx, pipe_add_images(media_pipe, explicit))

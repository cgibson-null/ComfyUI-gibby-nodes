"""
Context node
------------
A general-purpose context bundler (inspired by rgthree's "Context Big"). It
exposes a wide set of inputs/outputs so values can be collected into one
CONTEXT object and passed along a workflow, then unpacked again downstream.

Every input is optional: connect what you need, or link a base context in to
inherit its values for any field left unconnected. This node never generates
anything on its own - use Context Loader for model loading and on-demand
generation (conditioning from prompts, latent from image).

The evaluate() staticmethod fills missing derived values (conditioning,
latent) from what a context already holds; it is shared with Context Loader
and KSampler (Context).
"""

import torch
import comfy.samplers
import comfy.model_management
import comfy.model_base
from nodes import CLIPTextEncode, ConditioningZeroOut, VAEEncode, SetLatentNoiseMask
from comfy_extras.nodes_audio import VAEEncodeAudio
from comfy_extras.nodes_lt import LTXVConcatAVLatent
from comfy_api.latest import io

from ..lora_loader import _apply_lora, _format_lora_tag

_CONTEXT_TYPE = io.Custom("CONTEXT")
_lora_stack = io.Custom("LORA_STACK")


def _stringify(value):
    """Convert an arbitrary value to a string, mirroring KJNodes' 'Something To String'.

    Primitives become their str() form, lists/tuples are joined with ', ', and
    anything else falls back to str(). None becomes an empty string so the
    output is always a valid STRING.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value)


class GibbyContext(io.ComfyNode):
    """Bundle many values into a single CONTEXT object and pass them along."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Gibby_Context",
            display_name="Context",
            category="gibby/context",
            description=(
                "Collects many values into one CONTEXT object (and passes each "
                "through individually). Every input is optional; connect a base "
                "context in to inherit its values for any field left unconnected."
            ),
            inputs=[
                _CONTEXT_TYPE.Input("context", optional=True),
                io.Model.Input("model", optional=True),
                io.Clip.Input("clip", optional=True),
                io.Vae.Input("vae", optional=True),
                io.Vae.Input("vae_audio", optional=True),
                io.Conditioning.Input("positive", optional=True),
                io.Conditioning.Input("negative", optional=True),
                io.Latent.Input("latent", optional=True),
                io.Image.Input("image", optional=True),
                io.Mask.Input("mask", optional=True),
                io.Audio.Input("audio", optional=True),
                io.Mask.Input("mask_audio", optional=True),
                io.AnyType.Input("model_name", optional=True),
                _lora_stack.Input("lora_stack", optional=True),
                io.Int.Input("seed_value", optional=True, force_input=True),
                io.Int.Input("steps", optional=True, force_input=True),
                io.Int.Input("step_refiner", optional=True, force_input=True),
                io.Float.Input("cfg", optional=True, force_input=True),
                # Combo + forceInput: link-only slots that accept sampler/scheduler links like rgthree's Context Big.
                io.Combo.Input("sampler", options=comfy.samplers.KSampler.SAMPLERS, optional=True, extra_dict={"forceInput": True}),
                io.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS, optional=True, extra_dict={"forceInput": True}),
                io.Int.Input("width", optional=True, force_input=True),
                io.Int.Input("height", optional=True, force_input=True),
                io.String.Input("positive_prompt", optional=True, force_input=True),
                io.String.Input("negative_prompt", optional=True, force_input=True),
                io.AnyType.Input("any", optional=True, extra_dict={"forceInput": True}),
            ],
            outputs=[
                _CONTEXT_TYPE.Output(display_name="context"),
                io.Model.Output(display_name="model"),
                io.Clip.Output(display_name="clip"),
                io.Vae.Output(display_name="vae"),
                io.Vae.Output(display_name="vae_audio"),
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
                io.Image.Output(display_name="image"),
                io.Mask.Output(display_name="mask"),
                io.Audio.Output(display_name="audio"),
                io.Mask.Output(display_name="mask_audio"),
                io.String.Output(display_name="model_name"),
                _lora_stack.Output(display_name="lora_stack"),
                io.Int.Output(display_name="seed"),
                io.Int.Output(display_name="steps"),
                io.Int.Output(display_name="step_refiner"),
                io.Float.Output(display_name="cfg"),
                io.AnyType.Output(display_name="sampler"),
                io.AnyType.Output(display_name="scheduler"),
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
                io.String.Output(display_name="positive_prompt"),
                io.String.Output(display_name="negative_prompt"),
                io.AnyType.Output(display_name="any"),
                io.String.Output(display_name="lora_names"),
            ],
        )

    @classmethod
    def execute(cls, context=None, model=None, clip=None, vae=None,
                vae_audio=None, positive=None, negative=None, latent=None, image=None, mask=None,
                audio=None, mask_audio=None, model_name=None, lora_stack=None,
                seed_value=None, steps=None, step_refiner=None, cfg=None, sampler=None, scheduler=None,
                width=None, height=None, positive_prompt=None, negative_prompt=None,
                **kwargs) -> io.NodeOutput:
        any_value = kwargs.get('any')
        # Build the context dict from directly-provided values.
        ctx = {
            "model": model,
            "clip": clip,
            "vae": vae,
            "vae_audio": vae_audio,
            "positive": positive,
            "negative": negative,
            "latent": latent,
            "image": image,
            "mask": mask,
            "audio": audio,
            "mask_audio": mask_audio,
            "model_name": model_name,
            "lora_stack": lora_stack,
            "seed": seed_value,
            "steps": steps,
            "step_refiner": step_refiner,
            "cfg": cfg,
            "sampler": sampler,
            "scheduler": scheduler,
            "width": width,
            "height": height,
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "any": any_value,
        }

        # Fall back to the base context's values for anything not connected.
        if isinstance(context, dict):
            for key in ctx:
                if ctx[key] is None and key in context:
                    ctx[key] = context[key]

        # Stringify after inheritance so an unconnected model_name can still be
        # picked up from the base context instead of becoming "".
        ctx["model_name"] = _stringify(ctx["model_name"])

        # Apply LoRAs from a directly-connected lora_stack to model/clip. A
        # stack inherited from the base context is already baked into that
        # context's model, so it must not be applied again here.
        if isinstance(lora_stack, list):
            for item in lora_stack:
                if not item or len(item) < 3 or item[0] == "None":
                    continue
                name, sm, sc = item[0], item[1], item[2]
                ctx["model"], ctx["clip"] = _apply_lora(ctx["model"], ctx["clip"], name, sm, sc)

        # LoRA names for downstream nodes, like Lora Loader's lora_names output.
        lora_tags = []
        if isinstance(ctx["lora_stack"], list):
            for item in ctx["lora_stack"]:
                if not item or len(item) < 3 or item[0] == "None":
                    continue
                lora_tags.append(_format_lora_tag(item[0], item[1]))
        ctx["lora_names"] = ", ".join(lora_tags)

        return io.NodeOutput(
            ctx,  # context
            ctx["model"],  # model
            ctx["clip"],  # clip
            ctx["vae"],  # vae
            ctx["vae_audio"],  # vae_audio
            ctx["positive"],  # positive
            ctx["negative"],  # negative
            ctx["latent"],  # latent
            ctx["image"],  # image
            ctx["mask"],  # mask
            ctx["audio"],  # audio
            ctx["mask_audio"],  # mask_audio
            ctx["model_name"],  # model_name (string)
            ctx["lora_stack"],  # lora_stack
            ctx["seed"],  # seed
            ctx["steps"],  # steps
            ctx["step_refiner"],  # step_refiner
            ctx["cfg"],  # cfg
            ctx["sampler"],  # sampler (string)
            ctx["scheduler"],  # scheduler (string)
            ctx["width"],  # width
            ctx["height"],  # height
            ctx["positive_prompt"],  # positive_prompt
            ctx["negative_prompt"],  # negative_prompt
            ctx["any"],  # any_value
            ctx["lora_names"],  # lora_names
        )

    @staticmethod
    def evaluate(ctx):
        """Fill missing context values on demand from available inputs."""
        # Positive conditioning: encode positive_prompt if clip exists (even empty string).
        if ctx.get("positive") is None and ctx.get("clip") is not None:
            ctx["positive"], = CLIPTextEncode().encode(ctx["clip"], ctx.get("positive_prompt", ""))

        # Negative conditioning: zero out if cfg=1 and positive exists, else encode negative_prompt (even if empty).
        if ctx.get("negative") is None:
            if ctx.get("cfg") == 1 and ctx.get("positive") is not None:
                ctx["negative"], = ConditioningZeroOut().zero_out(ctx["positive"])
            elif ctx.get("clip") is not None:
                ctx["negative"], = CLIPTextEncode().encode(ctx["clip"], ctx.get("negative_prompt", ""))

        # Latent: only fill if missing - use existing latent if present.
        if ctx.get("latent") is None:
            # Image takes priority - encode it fresh with mask if available.
            if ctx.get("image") is not None and ctx.get("vae") is not None:
                try:
                    ctx["latent"], = VAEEncode().encode(ctx["vae"], ctx["image"])
                    if ctx.get("mask") is not None:
                        ctx["latent"], = SetLatentNoiseMask().set_mask(ctx["latent"], ctx["mask"])
                    
                    # If audio also available, combine into AV latent
                    if ctx.get("vae_audio") is not None and ctx.get("audio") is not None:
                        audio_latent, = VAEEncodeAudio().encode(ctx["vae_audio"], ctx["audio"])
                        ctx["latent"], = LTXVConcatAVLatent().execute(ctx["latent"], audio_latent)
                except Exception:
                    ctx["latent"] = None

            # If no image but audio + vae_audio available, encode audio-only latent
            elif ctx.get("vae_audio") is not None and ctx.get("audio") is not None:
                ctx["latent"], = VAEEncodeAudio().encode(ctx["vae_audio"], ctx["audio"])

        # Resolve width/height from image or latent if they're 0.
        if ctx.get("width", 0) == 0 or ctx.get("height", 0) == 0:
            if ctx.get("image") is not None:
                img_h, img_w = ctx["image"].shape[1], ctx["image"].shape[2]
                ctx["width"] = img_w
                ctx["height"] = img_h
            elif ctx.get("latent") is not None:
                lat_h, lat_w = ctx["latent"]["samples"].shape[2], ctx["latent"]["samples"].shape[3]
                # Determine downscale ratio based on latent channels
                channels = ctx["latent"]["samples"].shape[1]
                downscale = 16 if channels == 128 else 8  # Flux2 uses 128 channels/16x, others use 4/8x
                ctx["width"] = lat_w * downscale
                ctx["height"] = lat_h * downscale

        # Empty latent: nothing to encode or sample from, but width/height were given.
        if ctx.get("latent") is None and ctx.get("width", 0) > 0 and ctx.get("height", 0) > 0:
            # Check if model is Flux2 - use different latent format
            model_obj = ctx.get("model")
            is_flux2 = isinstance(model_obj.model, comfy.model_base.Flux2) if model_obj is not None and hasattr(model_obj, 'model') else False
            
            if is_flux2:
                # Flux2: 128 channels, 16x downscale
                samples = torch.zeros([1, 128, ctx["height"] // 16, ctx["width"] // 16], device=comfy.model_management.intermediate_device(), dtype=comfy.model_management.intermediate_dtype())
                ctx["latent"] = {"samples": samples}
            else:
                # Regular: 4 channels, 8x downscale
                samples = torch.zeros([1, 4, ctx["height"] // 8, ctx["width"] // 8], device=comfy.model_management.intermediate_device(), dtype=comfy.model_management.intermediate_dtype())
                ctx["latent"] = {"samples": samples, "downscale_ratio_spacial": 8}

        return ctx

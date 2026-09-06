"""
Context Override node
---------------------
Takes a CONTEXT object and selectively overrides parts of it: connected
inputs replace their context values directly, while two toggles gate groups
of widgets that override sampling parameters or prompts (regenerating the
conditionings from them).
"""

import comfy.samplers
from nodes import CLIPTextEncode, ConditioningZeroOut
from comfy_api.latest import io

from . import _CONTEXT_TYPE


class GibbyContextOverride(io.ComfyNode):
    """Override selected values of a CONTEXT object via inputs and widgets."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Gibby_Context_Override",
            display_name="Context Override",
            category="gibby/context",
            description=(
                "Overrides parts of a CONTEXT object. Connected inputs replace their "
                "values directly; the toggles gate widget overrides for sampling "
                "parameters and prompts (which also regenerates the conditionings)."
            ),
            inputs=[
                _CONTEXT_TYPE.Input("context"),
                io.Model.Input("model", optional=True),
                io.Conditioning.Input("positive", optional=True),
                io.Conditioning.Input("negative", optional=True),
                io.Image.Input("image", optional=True),
                io.Mask.Input("mask", optional=True),
                io.Audio.Input("audio", optional=True),
                io.Mask.Input("mask_audio", optional=True),
                io.Boolean.Input("override_sampling", default=False, tooltip="Override the context's sampling values with the widgets below."),
                io.Int.Input("steps", default=20),
                io.Int.Input("step_refiner", default=0),
                io.Float.Input("cfg", default=1.0),
                io.Combo.Input("sampler", options=comfy.samplers.KSampler.SAMPLERS, default="euler"),
                io.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS, default="normal"),
                io.Boolean.Input("override_prompts", default=False, tooltip="Override the context's prompts with the widgets below and regenerate its conditionings from them."),
                io.String.Input("positive_prompt", multiline=True, default=""),
                io.String.Input("negative_prompt", multiline=True, default=""),
            ],
            outputs=[
                _CONTEXT_TYPE.Output(display_name="context"),
                io.Model.Output(display_name="model"),
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Image.Output(display_name="image"),
                io.Mask.Output(display_name="mask"),
                io.Audio.Output(display_name="audio"),
                io.Mask.Output(display_name="mask_audio"),
            ],
        )

    @classmethod
    def execute(cls, context, model=None, positive=None, negative=None, image=None, mask=None,
                audio=None, mask_audio=None, override_sampling=False, steps=20, step_refiner=0,
                cfg=1.0, sampler="euler", scheduler="normal", override_prompts=False,
                positive_prompt="", negative_prompt="") -> io.NodeOutput:
        ctx = dict(context) if isinstance(context, dict) else {}

        # Connected inputs replace their context values directly.
        if model is not None:
            ctx["model"] = model
        if positive is not None:
            ctx["positive"] = positive
        if negative is not None:
            ctx["negative"] = negative
        if image is not None:
            ctx["image"] = image
        if mask is not None:
            ctx["mask"] = mask
        if audio is not None:
            ctx["audio"] = audio
        if mask_audio is not None:
            ctx["mask_audio"] = mask_audio

        # Sampling parameters, only when the toggle is on.
        if override_sampling:
            ctx["steps"] = steps
            ctx["step_refiner"] = step_refiner
            ctx["cfg"] = cfg
            ctx["sampler"] = sampler
            ctx["scheduler"] = scheduler

        # Prompts plus regenerated conditionings, only when the toggle is on.
        if override_prompts:
            ctx["positive_prompt"] = positive_prompt
            ctx["negative_prompt"] = negative_prompt
            if ctx.get("clip") is not None:
                ctx["positive"], = CLIPTextEncode().encode(ctx["clip"], positive_prompt)
                if ctx.get("cfg") == 1:
                    ctx["negative"], = ConditioningZeroOut().zero_out(ctx["positive"])
                elif negative_prompt:
                    ctx["negative"], = CLIPTextEncode().encode(ctx["clip"], negative_prompt)

        return io.NodeOutput(
            ctx,  # context
            ctx.get("model"),  # model
            ctx.get("positive"),  # positive
            ctx.get("negative"),  # negative
            ctx.get("image"),  # image
            ctx.get("mask"),  # mask
            ctx.get("audio"),  # audio
            ctx.get("mask_audio"),  # mask_audio
        )

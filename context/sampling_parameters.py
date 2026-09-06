"""
Sampling Parameters (Context) node
----------------------------------
Based on rgthree's KSampler Config: steps/refiner/cfg/sampler/scheduler widgets
that pass through as individual outputs, plus an optional CONTEXT input/output.
The widget values override the matching fields of the passed-through context
(steps, step_refiner, cfg, sampler, scheduler).
"""

import comfy.samplers
from nodes import MAX_RESOLUTION
from comfy_api.latest import io

from . import _CONTEXT_TYPE


class GibbySamplingParametersContext(io.ComfyNode):
    """Set sampling parameters as widgets and/or on a CONTEXT object."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Gibby_Sampling_Parameters_Context",
            display_name="Sampling Parameters (Context)",
            category="gibby/context",
            description=(
                "Holds sampling parameters as widgets and passes them through "
                "individually. With a context connected, the widget values "
                "override its steps, step_refiner, cfg, sampler and scheduler."
            ),
            # All inputs optional (like rgthree's Context Big) so the context input,
            # declared first, renders at the top instead of after required widgets.
            inputs=[
                _CONTEXT_TYPE.Input("context", optional=True),
                io.Int.Input("steps_total", default=30, min=1, max=MAX_RESOLUTION, step=1, optional=True),
                io.Int.Input("refiner_step", default=24, min=1, max=MAX_RESOLUTION, step=1, optional=True),
                io.Float.Input("cfg", default=8.0, min=0.0, max=100.0, step=0.5, optional=True),
                io.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS, optional=True),
                io.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS, optional=True),
            ],
            outputs=[
                _CONTEXT_TYPE.Output(display_name="context"),
                io.Int.Output(display_name="steps"),
                io.Int.Output(display_name="refiner_step"),
                io.Float.Output(display_name="cfg"),
                io.Combo.Output(display_name="sampler", options=comfy.samplers.KSampler.SAMPLERS),
                io.Combo.Output(display_name="scheduler", options=comfy.samplers.KSampler.SCHEDULERS),
            ],
        )

    @classmethod
    def execute(cls, context=None, steps_total=30, refiner_step=24, cfg=8.0, sampler_name="euler", scheduler="normal") -> io.NodeOutput:
        ctx = dict(context) if isinstance(context, dict) else {}
        # Widget values override the matching context fields.
        ctx["steps"] = steps_total
        ctx["step_refiner"] = refiner_step
        ctx["cfg"] = cfg
        ctx["sampler"] = sampler_name
        ctx["scheduler"] = scheduler

        return io.NodeOutput(ctx, steps_total, refiner_step, cfg, sampler_name, scheduler)

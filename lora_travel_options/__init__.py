from comfy_api.latest import ComfyExtension, io
from ..crop_inpaint_options import _KSAMPLER_OPTIONS_TYPE
from ..lora_loader import _lora_stack


class GibbyLoraTravelOptions(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GibbyLoraTravelOptions",
            display_name="Lora Travel options",
            category="gibby/flow",
            search_aliases=["lora", "travel", "ramp", "strength", "options", "config"],
            description="Outputs lora travel options. Feed into KSampler (Context) options input; the lora stack is applied only on the active steps, with its strength ramping linearly from start_str to end_str across them.",
            inputs=[
                _lora_stack.Input("lora_stack", tooltip="Lora stack (Lora Loader output) whose strength travels across the sampling steps"),
                io.Float.Input("start_step", default=0.0, min=-10000.0, max=10000.0, step=0.01, tooltip="First step the loras are applied at. >=1=absolute step, 0-1=percentage of steps, <0=steps from the end"),
                io.Float.Input("end_step", default=100.0, min=-10000.0, max=10000.0, step=0.01, tooltip="Step the loras stop at (exclusive). >=1=absolute step, 0-1=percentage of steps, <0=steps from the end"),
                io.Float.Input("start_str", default=0.6, min=-10.0, max=10.0, step=0.01, tooltip="Strength multiplier at start_step"),
                io.Float.Input("end_str", default=1.0, min=-10.0, max=10.0, step=0.01, tooltip="Strength multiplier at end_step; ramps linearly from start_str"),
                io.Boolean.Input("verbose", default=False, tooltip="Print the active lora names and strengths on each active step"),
            ],
            outputs=[
                _KSAMPLER_OPTIONS_TYPE.Output("options"),
            ],
        )

    @classmethod
    def execute(cls, lora_stack, start_step=0.0, end_step=100.0, start_str=0.6, end_str=1.0, verbose=False):
        option = {
            "type": "lora_travel",
            "lora_stack": lora_stack,
            "start_step": start_step,
            "end_step": end_step,
            "start_str": start_str,
            "end_str": end_str,
            "verbose": verbose,
        }
        return io.NodeOutput([option])


class GibbyLoraTravelOptionsExtension(ComfyExtension):
    async def get_node_list(self):
        return [GibbyLoraTravelOptions]


async def comfy_entrypoint() -> GibbyLoraTravelOptionsExtension:
    return GibbyLoraTravelOptionsExtension()

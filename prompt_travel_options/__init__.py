from comfy_api.latest import ComfyExtension, io
from ..crop_inpaint_options import _KSAMPLER_OPTIONS_TYPE


class GibbyPromptTravelOptions(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GibbyPromptTravelOptions",
            display_name="Prompt Travel options",
            category="gibby/flow",
            search_aliases=["prompt", "travel", "switch", "steps", "options", "config"],
            description="Outputs prompt travel options. Feed into KSampler (Context) options input; the prompts travel across the sampling steps - a [before:after:step] group uses before until the step and after from it on, and the groups nest: [[a:b:3]:c:6].",
            inputs=[
                io.String.Input("positive", multiline=True, default="", tooltip="Positive prompt with travel groups: [a:b:step] switches from a to b at step, the groups nest: [[a:b:3]:c:6]. step: with a dot=fraction of steps (.6=60%), without=absolute step (3). () weight groups are kept as-is"),
                io.String.Input("negative", multiline=True, default="", tooltip="Negative prompt, same travel syntax as positive"),
                io.Boolean.Input("verbose", default=False, tooltip="Print the active positive/negative prompts on each step"),
            ],
            outputs=[
                _KSAMPLER_OPTIONS_TYPE.Output("options"),
            ],
        )

    @classmethod
    def execute(cls, positive="", negative="", verbose=False):
        option = {
            "type": "prompt_travel",
            "positive": positive,
            "negative": negative,
            "verbose": verbose,
        }
        return io.NodeOutput([option])


class GibbyPromptTravelOptionsExtension(ComfyExtension):
    async def get_node_list(self):
        return [GibbyPromptTravelOptions]


async def comfy_entrypoint() -> GibbyPromptTravelOptionsExtension:
    return GibbyPromptTravelOptionsExtension()

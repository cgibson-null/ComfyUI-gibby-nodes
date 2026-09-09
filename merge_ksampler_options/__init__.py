from comfy_api.latest import ComfyExtension, io
from ..crop_inpaint_options import _KSAMPLER_OPTIONS_TYPE


class GibbyMergeKSamplerOptions(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        template = io.Autogrow.TemplateNames(
            input=_KSAMPLER_OPTIONS_TYPE.Input("option", optional=True),
            names=[f"option{i}" for i in range(1, 11)],
            min=0,
        )
        return io.Schema(
            node_id="GibbyMergeKSamplerOptions",
            display_name="Merge KSampler Options",
            category="gibby/flow",
            search_aliases=["merge", "combine", "options"],
            description="Merges options lists into one - KSampler (Context) applies each option in order.",
            inputs=[
                io.Autogrow.Input("options", template=template),
            ],
            outputs=[
                _KSAMPLER_OPTIONS_TYPE.Output("options"),
            ],
        )

    @classmethod
    def execute(cls, options: io.Autogrow.Type) -> io.NodeOutput:
        merged = []
        for opt_list in options.values():
            if opt_list:
                merged.extend(opt_list)
        return io.NodeOutput(merged)


class GibbyMergeKSamplerOptionsExtension(ComfyExtension):
    async def get_node_list(self):
        return [GibbyMergeKSamplerOptions]


async def comfy_entrypoint() -> GibbyMergeKSamplerOptionsExtension:
    return GibbyMergeKSamplerOptionsExtension()

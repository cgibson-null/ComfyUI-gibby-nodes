from comfy_api.latest import ComfyExtension, io


class GibbyPauseExecution(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GibbyPauseExecution",
            display_name="Pause Execution",
            category="gibby/flow",
            search_aliases=["pause", "stop", "break", "wait", "continue", "resume"],
            description="Blocks execution until user clicks Continue. Passes input through unchanged.",
            inputs=[
                io.AnyType.Input("input"),
                io.Boolean.Input("stop", default=True, tooltip="When true, blocks execution at this node"),
            ],
            outputs=[
                io.AnyType.Output("output"),
            ],
        )

    @classmethod
    def execute(cls, input, stop=True):
        if stop:
            return io.NodeOutput(input, block_execution="paused")
        return io.NodeOutput(input)


class GibbyPauseExecutionExtension(ComfyExtension):
    async def get_node_list(self):
        return [GibbyPauseExecution]


async def comfy_entrypoint() -> GibbyPauseExecutionExtension:
    return GibbyPauseExecutionExtension()

from comfy_api.latest import io


class GibbyAnySwitch(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        any_template = io.MatchType.Template("any")  # allowed_types defaults to AnyType - matches everything
        autogrow_template = io.Autogrow.TemplatePrefix(
                io.MatchType.Input("input", any_template),
                prefix="any", min=0, max=50)
        return io.Schema(
            node_id="Gibby_AnySwitch",
            display_name="Any Switch",
            category="gibby",
            description="Outputs a connected input by index; works with any type.",
            inputs=[
                io.Int.Input(
                    "index",
                    default=-1,
                    min=-1,
                    tooltip="Which connected input to output; -1 outputs the first connected input"
                ),
                io.Autogrow.Input("inputs", template=autogrow_template),
            ],
            outputs=[io.MatchType.Output(any_template, display_name="*")]
        )

    @classmethod
    def execute(cls, index: int, inputs: io.Autogrow.Type) -> io.NodeOutput:
        connected = [value for value in inputs.values() if value is not None]
        if index < 0:
            return io.NodeOutput(connected[0]) if connected else io.NodeOutput(None)
        if index >= len(connected):
            raise ValueError(f"index {index} is out of range: only {len(connected)} input(s) connected")
        return io.NodeOutput(connected[index])

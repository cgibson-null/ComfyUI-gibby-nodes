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
            description="Outputs the first connected input; works with any type.",
            inputs=[io.Autogrow.Input("inputs", template=autogrow_template)],
            outputs=[io.MatchType.Output(any_template, display_name="*")]
        )

    @classmethod
    def execute(cls, inputs: io.Autogrow.Type) -> io.NodeOutput:
        for value in inputs.values():
            if value is not None:
                return io.NodeOutput(value)
        return io.NodeOutput(None)

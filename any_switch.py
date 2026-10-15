from comfy_api.latest import io


class GibbyAnySwitch(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        any_template = io.MatchType.Template("any")  # allowed_types defaults to AnyType - matches everything
        autogrow_template = io.Autogrow.TemplatePrefix(
                io.MatchType.Input("input", any_template, lazy=True),
                prefix="any", min=0, max=50)
        return io.Schema(
            node_id="Gibby_AnySwitch",
            display_name="Any Switch",
            category="gibby",
            description=("Outputs a connected input by index; works with any type. "
                         "Only the selected input is executed, the other connected "
                         "branches stay unevaluated."),
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
    def _selected(cls, index: int, inputs: dict) -> str | None:
        """The name of the input to output: the index-th connected one, or the
        first connected one when index is -1. The connected slots are the keys,
        whether or not they have been evaluated yet, and none is None."""
        names = list(inputs)
        if not names:
            return None
        if index >= len(names):
            raise ValueError(f"index {index} is out of range: only {len(names)} input(s) connected")
        return names[0] if index < 0 else names[index]

    @classmethod
    def check_lazy_status(cls, index: int, inputs: dict) -> list[str]:
        """Ask ComfyUI to evaluate the selected input only, which is what keeps
        the other branches from running. The lazy call hands each dynamic slot
        as a (value, prompt key) tuple, value None while unevaluated, and the
        prompt key is the name to ask for."""
        name = cls._selected(index, inputs)
        if name is None:
            return []
        value, key = inputs[name]
        return [] if value is not None else [key]

    @classmethod
    def execute(cls, index: int, inputs: io.Autogrow.Type) -> io.NodeOutput:
        name = cls._selected(index, inputs)
        return io.NodeOutput(None if name is None else inputs[name])

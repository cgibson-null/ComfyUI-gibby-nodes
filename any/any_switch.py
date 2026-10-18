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

    @staticmethod
    def _slots(inputs) -> dict:
        """The connected autogrow slots as a name -> value dict.

        ComfyUI hands the group as that dict, with one exception: an empty group
        in the lazy call arrives wrapped in the same (value, prompt key) tuple
        the non-empty slots are wrapped in - ({}, "inputs") - so unwrap whatever
        shape comes in instead of assuming the dict."""
        if isinstance(inputs, tuple):
            inputs = next((x for x in inputs if isinstance(x, dict)), None)
        return inputs or {}

    @classmethod
    def _selected(cls, index: int, slots: dict) -> str | None:
        """The name of the input to output: the index-th connected one, or the
        first connected one when index is -1. The connected slots are the keys,
        whether or not they have been evaluated yet, and none is None."""
        names = list(slots)
        if not names:
            return None
        if index >= len(names):
            raise ValueError(f"index {index} is out of range: only {len(names)} input(s) connected")
        return names[0] if index < 0 else names[index]

    @classmethod
    def check_lazy_status(cls, index: int, inputs) -> list[str]:
        """Ask ComfyUI to evaluate the selected input only, which is what keeps
        the other branches from running. The lazy call hands each dynamic slot
        as a (value, prompt key) tuple, value None while unevaluated, and the
        prompt key is the name to ask for. Nothing connected asks for nothing,
        which lets execute run and output None."""
        slots = cls._slots(inputs)
        name = cls._selected(index, slots)
        if name is None:
            return []
        value, key = slots[name]
        return [] if value is not None else [key]

    @classmethod
    def execute(cls, index: int, inputs: io.Autogrow.Type) -> io.NodeOutput:
        slots = cls._slots(inputs)
        name = cls._selected(index, slots)
        return io.NodeOutput(None if name is None else slots[name])

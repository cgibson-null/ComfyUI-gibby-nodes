import torch
from comfy_api.latest import io


class GibbyAnyBatchList(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        any_template = io.MatchType.Template("any")  # allowed_types defaults to AnyType - matches everything
        autogrow_template = io.Autogrow.TemplatePrefix(
                io.MatchType.Input("input", any_template),
                prefix="any", min=1, max=50)
        return io.Schema(
            node_id="Gibby_AnyBatchList",
            display_name="Any Batch/List",
            category="gibby",
            search_aliases=["batch any", "make list", "list any", "combine any", "stack any"],
            description=("Batches or lists every connected input, whatever its type. "
                         "Batch concatenates tensors along dim 0 the way Batch Images "
                         "does and joins lists such as conditioning; List - and batch "
                         "of anything neither, like a model - outputs a Python list."),
            inputs=[
                io.Boolean.Input(
                    "mode",
                    default=False,
                    label_off="Batch",
                    label_on="List",
                    tooltip="Batch: one batched value. List: a Python list of the connected inputs"
                ),
                io.Autogrow.Input("inputs", template=autogrow_template),
            ],
            outputs=[io.MatchType.Output(any_template, display_name="*")]
        )

    @classmethod
    def execute(cls, mode: bool, inputs: io.Autogrow.Type) -> io.NodeOutput:
        """inputs holds only the connected slots, in slot order."""
        values = list(inputs.values())
        if mode:
            return io.NodeOutput(values)
        if not values:
            return io.NodeOutput(None)
        if all(isinstance(v, torch.Tensor) for v in values):
            # one tensor IS the batch: cat would allocate and copy it for nothing
            return io.NodeOutput(values[0] if len(values) == 1 else torch.cat(values, dim=0))
        if all(isinstance(v, list) for v in values):
            # conditioning and the other list-valued types batch by concatenating
            return io.NodeOutput([item for v in values for item in v])
        return io.NodeOutput(values)

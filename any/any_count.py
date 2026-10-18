from comfy_api.latest import io


class GibbyAnyCount(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Gibby_AnyCount",
            display_name="Any Count",
            category="gibby",
            search_aliases=["batch count", "count", "length", "size", "how many", "empty", "null"],
            description=("How many items a value holds: the batch size of a tensor "
                         "or latent, the length of a conditioning or a list, 1 for a "
                         "single object. An unconnected input and a null both count "
                         "0, so an optional branch can be measured instead of having "
                         "to be wired."),
            inputs=[
                io.AnyType.Input(
                    "value",
                    optional=True,
                    tooltip="Anything: a tensor counts its first dimension, a latent its batch, a conditioning or list its length, a single object 1"
                ),
            ],
            outputs=[
                io.Int.Output(display_name="count"),
            ],
        )

    @classmethod
    def count(cls, value) -> int:
        """How many items value holds. None counts 0 - nothing connected and a
        null from a switch both land here. A tensor (or numpy array, or the
        NestedTensor a latent can carry) counts its first dimension, a latent
        dict its samples tensor, a str or bytes one value rather than its
        character count, and anything with a length - conditioning, a list, a
        dict - its length; a model, a vae and the other singletons are one."""
        if value is None:
            return 0
        shape = getattr(value, "shape", None)
        if isinstance(shape, tuple) and shape:
            return int(shape[0])
        if isinstance(shape, tuple):  # a 0-dim tensor is still one value
            return 1
        if isinstance(value, dict) and value.get("samples") is not None:
            return cls.count(value["samples"])
        if isinstance(value, (str, bytes)):
            return 1
        try:
            return len(value)
        except TypeError:
            return 1

    @classmethod
    def execute(cls, value=None) -> io.NodeOutput:
        return io.NodeOutput(cls.count(value))

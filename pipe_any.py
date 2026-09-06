"""
Pipe Any node
-------------
Creates a pipe dict from multiple Any inputs, or overrides an existing pipe.

Similar to ImageBatchMulti but for Any type:
- count input controls the number of any_1, any_2, etc. inputs
- Top input/output: pipe (dict) for existing pipes to override
- If pipe is provided and any inputs are connected, those override pipe elements
- Outputs the resulting pipe dict plus all individual Any outputs
"""

from comfy_api.latest import io


class PipeAny(io.ComfyNode):
    """Combine multiple Any inputs into a pipe dict, or override an existing pipe."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="PipeAny",
            display_name="Pipe Any",
            category="gibby",
            description=(
                "Creates a pipe dict from multiple Any inputs. "
                "If an existing pipe is provided, connected inputs override it at their positions. "
                "Set count and click update to change the number of inputs/outputs."
            ),
            inputs=[
                # Top: existing pipe to override
                io.Dict.Input("pipe", optional=True,
                              tooltip="Optional existing pipe to override with connected inputs"),
                # Any inputs (all optional)
                io.AnyType.Input("any_1", optional=True),
                io.AnyType.Input("any_2", optional=True),
                io.AnyType.Input("any_3", optional=True),
                io.AnyType.Input("any_4", optional=True),
                io.AnyType.Input("any_5", optional=True),
                io.AnyType.Input("any_6", optional=True),
                io.AnyType.Input("any_7", optional=True),
                io.AnyType.Input("any_8", optional=True),
                io.AnyType.Input("any_9", optional=True),
                io.AnyType.Input("any_10", optional=True),
            ],
            outputs=[
                # Top: resulting pipe dict
                io.Dict.Output(display_name="pipe"),
                # Any outputs (controlled by count via JS)
                io.AnyType.Output(display_name="any_1"),
                io.AnyType.Output(display_name="any_2"),
                io.AnyType.Output(display_name="any_3"),
                io.AnyType.Output(display_name="any_4"),
                io.AnyType.Output(display_name="any_5"),
                io.AnyType.Output(display_name="any_6"),
                io.AnyType.Output(display_name="any_7"),
                io.AnyType.Output(display_name="any_8"),
                io.AnyType.Output(display_name="any_9"),
                io.AnyType.Output(display_name="any_10"),
            ],
        )

    @classmethod
    def execute(cls, pipe=None, **kwargs):
        # Start with existing pipe or empty
        if pipe is None:
            result = {}
        else:
            result = dict(pipe)

        # Override pipe elements with connected any inputs
        for i in range(1, 11):
            key = f"any_{i}"
            value = kwargs.get(key)
            if value is not None:
                result[key] = value

        # Return outputs: pipe dict first, then any_1 through any_10 from the result
        return (result,) + tuple(result.get(f"any_{i}") for i in range(1, 11))

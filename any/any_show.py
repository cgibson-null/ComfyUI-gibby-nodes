from comfy_api.latest import io


class GibbyShowAny(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Gibby_ShowAny",
            display_name="Show Any",
            category="gibby",
            search_aliases=["show any", "preview any", "display any", "inspect", "debug",
                            "print value", "watch", "show list"],
            is_input_list=True,
            is_output_node=True,
            inputs=[io.AnyType.Input("any", optional=True)],
            outputs=[io.AnyType.Output("output")],
            hidden=[io.Hidden.unique_id, io.Hidden.extra_pnginfo],
        )

    @classmethod
    def describe(cls, value) -> str:
        """One value as one row: a tensor by its shape, dtype and device rather than
        its numbers, a dict by its keys, an object with no useful __str__ by its
        class. A container inside a row keeps its own lines."""
        if value is None:
            return "None"
        if isinstance(value, str):
            return value
        if isinstance(value, (bool, int, float)):
            return str(value)
        shape = getattr(value, "shape", None)
        if isinstance(shape, tuple):  # torch tensor, numpy array, NestedTensor
            return " ".join(str(x) for x in (type(value).__name__, tuple(shape),
                                             getattr(value, "dtype", ""),
                                             getattr(value, "device", "")) if x)
        if isinstance(value, dict):
            return "\n".join(f"{k}: {cls.describe(v)}" for k, v in value.items()) or "{}"
        if isinstance(value, (list, tuple)):
            return "\n".join(cls.describe(v) for v in value) or "[]"
        text = str(value)
        return type(value).__name__ if text == object.__repr__(value) else text

    @classmethod
    def execute(cls, any=None) -> io.NodeOutput:
        # a connected list is shown as one row per item, the way EasyUse's does
        values = []
        for value in any or []:
            if value is None:
                continue
            if isinstance(value, list):
                values.extend(cls.describe(item) for item in value)
            else:
                values.append(cls.describe(value))

        # the workflow embedded in the output is the queue-time snapshot the client
        # sent, so the rows are written into it for the shown text to travel with it.
        # V3 hidden inputs arrive on cls.hidden, never as execute kwargs
        workflow = (cls.hidden.extra_pnginfo or {}).get("workflow")
        if workflow:
            uid = str(cls.hidden.unique_id)
            node = next((n for n in workflow.get("nodes", []) if str(n["id"]) == uid), None)
            if node is not None:
                node["widgets_values"] = values

        return io.NodeOutput(values[0] if len(values) == 1 else values, ui={"text": values})

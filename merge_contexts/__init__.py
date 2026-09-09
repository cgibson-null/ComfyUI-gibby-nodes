from comfy_api.latest import ComfyExtension, io
from ..context import _CONTEXT_TYPE


def _is_unset(value):
    # Values that keep the base context's field instead of overriding it.
    if value is None:
        return True
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value == 0:
        return True
    return isinstance(value, str) and value == ""


class GibbyMergeContexts(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        template = io.Autogrow.TemplateNames(
            input=_CONTEXT_TYPE.Input("context", optional=True),
            names=[f"context{i}" for i in range(1, 11)],
            min=0,
        )
        return io.Schema(
            node_id="GibbyMergeContexts",
            display_name="Merge contexts",
            category="gibby/context",
            search_aliases=["merge", "combine", "context"],
            description=(
                "Merges contexts into one: the 1st provided context is the base, then each "
                "following context overrides its fields in order. A value overrides only when "
                "it is set - None, zero and empty string keep the base's value."
            ),
            inputs=[
                io.Autogrow.Input("contexts", template=template),
            ],
            outputs=[
                _CONTEXT_TYPE.Output("context"),
            ],
        )

    @classmethod
    def execute(cls, contexts: io.Autogrow.Type) -> io.NodeOutput:
        base = {}
        for value in contexts.values():
            if not isinstance(value, dict):
                continue
            if not base:
                base = value.copy()
                continue
            for key, item in value.items():
                if not _is_unset(item):
                    base[key] = item
        return io.NodeOutput(base)


class GibbyMergeContextsExtension(ComfyExtension):
    async def get_node_list(self):
        return [GibbyMergeContexts]


async def comfy_entrypoint() -> GibbyMergeContextsExtension:
    return GibbyMergeContextsExtension()

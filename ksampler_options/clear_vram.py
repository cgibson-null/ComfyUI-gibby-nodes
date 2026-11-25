from comfy_api.latest import io
from .crop_inpaint import _KSAMPLER_OPTIONS_TYPE


class GibbyClearVramOptions(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GibbyClearVramOptions",
            display_name="Clear VRAM Options",
            category="gibby/flow",
            search_aliases=["vram", "clear", "gpu", "memory", "unload", "options", "config"],
            description="Clear VRAM options for KSampler (Context): frees VRAM like Clean VRAM Used at the ticked points.",
            inputs=[
                io.Boolean.Input("at_start", default=True, tooltip="Before the model or the VAE is loaded"),
                io.Boolean.Input("after_finish", default=True),
                io.Boolean.Input("after_model", default=True, tooltip="After sampling, unloading the model before the VAE decode"),
                io.Boolean.Input("after_vae", default=True),
                io.Boolean.Input("verbose", default=False),
            ],
            outputs=[
                _KSAMPLER_OPTIONS_TYPE.Output("options"),
            ],
        )

    @classmethod
    def execute(cls, at_start=True, after_finish=True, after_model=True, after_vae=True, verbose=False):
        option = {
            "type": "clear_vram",
            "at_start": at_start,
            "after_finish": after_finish,
            "after_model": after_model,
            "after_vae": after_vae,
            "verbose": verbose,
        }
        return io.NodeOutput([option])

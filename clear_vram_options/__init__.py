from comfy_api.latest import ComfyExtension, io
from ..crop_inpaint_options import _KSAMPLER_OPTIONS_TYPE


class GibbyClearVramOptions(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GibbyClearVramOptions",
            display_name="Clear VRAM Options",
            category="gibby/flow",
            search_aliases=["vram", "clear", "gpu", "memory", "unload", "options", "config"],
            description="Outputs clear VRAM options. Feed into KSampler (Context) options input - frees VRAM like Clean VRAM Used at the start and end of the execution and once the sampler is done with the model and/or the VAE.",
            inputs=[
                io.Boolean.Input("at_start", default=True, tooltip="Clear VRAM at the very start of the execution, before the model or the VAE is loaded"),
                io.Boolean.Input("after_finish", default=True, tooltip="Clear VRAM at the very end of the execution, after sampling and decode"),
                io.Boolean.Input("after_model", default=True, tooltip="Clear VRAM after sampling (unloads the model before the VAE decode)"),
                io.Boolean.Input("after_vae", default=True, tooltip="Clear VRAM after VAE encode/decode (unloads the VAE)"),
                io.Boolean.Input("verbose", default=False, tooltip="Log each VRAM clear to the console with its event (at start / after model / after vae / after finish)"),
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


class GibbyClearVramOptionsExtension(ComfyExtension):
    async def get_node_list(self):
        return [GibbyClearVramOptions]


async def comfy_entrypoint() -> GibbyClearVramOptionsExtension:
    return GibbyClearVramOptionsExtension()

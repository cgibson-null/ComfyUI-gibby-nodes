from comfy_api.latest import ComfyExtension, io
from ..crop_inpaint_options import _KSAMPLER_OPTIONS_TYPE


class GibbyTiledVaeOptions(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GibbyTiledVaeOptions",
            display_name="Tiled VAE options",
            category="gibby/flow",
            search_aliases=["tiled", "vae", "tile", "options", "config"],
            description="Outputs tiled VAE options. Feed into KSampler (Context) options input - connecting it switches the sampler to tiled VAE encode/decode.",
            inputs=[
                io.Int.Input("tile_size", default=512, min=64, max=4096, step=32),
                io.Int.Input("overlap", default=64, min=0, max=4096, step=32),
                io.Int.Input("temporal_size", default=64, min=8, max=4096, step=4, tooltip="Only used for video VAEs: Amount of frames to decode at a time."),
                io.Int.Input("temporal_overlap", default=8, min=4, max=4096, step=4, tooltip="Only used for video VAEs: Amount of frames to overlap."),
            ],
            outputs=[
                _KSAMPLER_OPTIONS_TYPE.Output("options"),
            ],
        )

    @classmethod
    def execute(cls, tile_size=512, overlap=64, temporal_size=64, temporal_overlap=8):
        option = {
            "type": "tiled_vae",
            "tile_size": tile_size,
            "overlap": overlap,
            "temporal_size": temporal_size,
            "temporal_overlap": temporal_overlap,
        }
        return io.NodeOutput([option])


class GibbyTiledVaeOptionsExtension(ComfyExtension):
    async def get_node_list(self):
        return [GibbyTiledVaeOptions]


async def comfy_entrypoint() -> GibbyTiledVaeOptionsExtension:
    return GibbyTiledVaeOptionsExtension()

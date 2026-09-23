from comfy_api.latest import io
from ..crop_inpaint_options import _KSAMPLER_OPTIONS_TYPE


class GibbyColorMatchOptions(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GibbyColorMatchOptions",
            display_name="Color Match options",
            category="gibby/flow",
            search_aliases=["color", "match", "transfer", "options", "config"],
            description="Outputs color match options. Feed into KSampler (Context) options input: after sampling (a run from a provided image, crop-inpainting, or the final iterative step) the result's color is matched back to the original with Transfer Color (0 strength = no match).",
            inputs=[
                io.Combo.Input("color_match_method", default="mkl_lab", options=["reinhard_lab", "mkl_lab", "histogram"], tooltip="Transfer Color method"),
                io.Float.Input("color_match_strength", default=1.0, min=0.0, max=10.0, step=0.01, tooltip="Transfer Color strength (0 = no match)"),
            ],
            outputs=[
                _KSAMPLER_OPTIONS_TYPE.Output("options"),
            ],
        )

    @classmethod
    def execute(cls, color_match_method="mkl_lab", color_match_strength=1.0):
        option = {
            "type": "color_match",
            "color_match_method": color_match_method,
            "color_match_strength": color_match_strength,
        }
        return io.NodeOutput([option])

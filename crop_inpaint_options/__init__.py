from comfy_api.latest import ComfyExtension, io

# Shared by all options nodes feeding KSampler (Context); carries a list of
# option dicts (may hold non-serializable values like an upscale model).
_KSAMPLER_OPTIONS_TYPE = io.Custom("GIBBY_KSAMPLER_OPTIONS")


class GibbyCropInpaintOptions(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GibbyCropInpaintOptions",
            display_name="Crop-Inpaint options",
            category="gibby/flow",
            search_aliases=["inpaint", "crop", "detailer", "options", "config"],
            description="Outputs crop-inpaint options. Feed into KSampler (Context) options input.",
            inputs=[
                io.Float.Input("crop_factor", default=3.0, min=1.0, max=10.0, step=0.1, tooltip="Crop region size multiplier on mask bbox"),
                io.Float.Input("megapixels", default=0.0, min=0.0, max=100.0, step=0.1, tooltip="Target megapixels for crop (0=off). Applied after crop_factor."),
                io.Float.Input("scale_factor", default=1.0, min=0.1, max=10.0, step=0.1, tooltip="Resolution multiplier for crop (1.0=off). Applied after crop_factor."),
                io.Int.Input("multiple", default=8, min=1, max=256, step=1, tooltip="Round crop size down to this multiple"),
                io.Combo.Input("upscale_method", default="lanczos", options=["bilinear", "area", "nearest", "lanczos"]),
                io.Boolean.Input("inpaint_mode", default=True, label_on="masked_only", label_off="whole", tooltip="masked_only: denoise only inside mask. whole: denoise entire crop."),
                io.Boolean.Input("mask_mode", default=True, label_on="single", label_off="split", tooltip="single: treat mask as one region. split: split disconnected mask areas into separate crops."),
                io.Float.Input("mask_scale_start", default=1.0, min=0.0, max=3.0, step=0.05, tooltip="Mask size multiplier on first step. <1=smaller, >1=larger than actual mask."),
                io.Float.Input("mask_scale_end", default=1.0, min=0.0, max=3.0, step=0.05, tooltip="Mask size multiplier on last step."),
                io.String.Input("mask_indices", default="", tooltip="When mask_mode=split: which mask indices to process. Extracts integers from string, ignores indices >= mask count. Empty=all."),
                io.Boolean.Input("color_match", default=True, tooltip="After inpainting, match the result's color back to the original image with Transfer Color"),
                io.Combo.Input("color_match_method", default="mkl_lab", options=["reinhard_lab", "mkl_lab", "histogram"], tooltip="Transfer Color method for color_match"),
                io.Float.Input("color_match_strength", default=1.0, min=0.0, max=10.0, step=0.01, tooltip="Transfer Color strength for color_match (0=off)"),
            ],
            outputs=[
                _KSAMPLER_OPTIONS_TYPE.Output("options"),
            ],
        )

    @classmethod
    def execute(cls, crop_factor=3.0, megapixels=0.0, scale_factor=1.0, multiple=8,
                upscale_method="lanczos", inpaint_mode=True, mask_mode=True,
                mask_scale_start=1.0, mask_scale_end=1.0, mask_indices="",
                color_match=True, color_match_method="mkl_lab", color_match_strength=1.0):
        option = {
            "type": "inpaint",
            "crop_factor": crop_factor,
            "megapixels": megapixels,
            "scale_factor": scale_factor,
            "multiple": multiple,
            "upscale_method": upscale_method,
            "inpaint_mode": "masked_only" if inpaint_mode else "whole",
            "mask_mode": "single" if mask_mode else "split",
            "mask_scale_start": mask_scale_start,
            "mask_scale_end": mask_scale_end,
            "mask_indices": mask_indices,
            "color_match": color_match,
            "color_match_method": color_match_method,
            "color_match_strength": color_match_strength,
        }
        return io.NodeOutput([option])


class GibbyCropInpaintOptionsExtension(ComfyExtension):
    async def get_node_list(self):
        return [GibbyCropInpaintOptions]


async def comfy_entrypoint() -> GibbyCropInpaintOptionsExtension:
    return GibbyCropInpaintOptionsExtension()

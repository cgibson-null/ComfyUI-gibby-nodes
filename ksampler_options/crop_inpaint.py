import nodes
from comfy_api.latest import io

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
            description="Crop-inpaint options for KSampler (Context): each sampling step runs on a crop of the mask region instead of the whole image.",
            inputs=[
                io.Float.Input("crop_factor", default=3.0, min=1.0, max=10.0, step=0.1, tooltip="Crop region size multiplier on mask bbox"),
                io.Float.Input("megapixels", default=0.0, min=0.0, max=100.0, step=0.1, tooltip="0 = off; applied after crop_factor"),
                io.Float.Input("scale_factor", default=1.0, min=0.1, max=10.0, step=0.1, tooltip="1.0 = off; applied after crop_factor"),
                io.Int.Input("multiple", default=32, min=1, max=256, step=1),
                io.Int.Input("mask_grow", default=0, min=-nodes.MAX_RESOLUTION, max=nodes.MAX_RESOLUTION, step=1,
                             tooltip="Negative shrinks; applied before the crop"),
                io.Boolean.Input("mask_invert", default=False,
                                 tooltip="After mask_grow, before the crop"),
                io.Combo.Input("upscale_method", default="lanczos", options=["bilinear", "area", "nearest", "lanczos"]),
                io.Boolean.Input("inpaint_mode", default=True, label_on="masked_only", label_off="whole", tooltip="masked_only: denoise only inside mask. whole: denoise entire crop."),
                io.Boolean.Input("mask_mode", default=True, label_on="single", label_off="split", tooltip="single: treat mask as one region. split: split disconnected mask areas into separate crops."),
                io.Float.Input("mask_scale_start", default=1.0, min=0.0, max=3.0, step=0.05, tooltip="Mask size multiplier on first step. <1=smaller, >1=larger than actual mask."),
                io.Float.Input("mask_scale_end", default=1.0, min=0.0, max=3.0, step=0.05, tooltip="Mask size multiplier on last step."),
                io.String.Input("mask_indices", default="", tooltip="Indices to process: '0,2', '0-2' (both ends) or a slice (':3', '-3:', '::2'); empty is all. split: the mask regions; single with a batch: the batch items, each its own crop."),
                io.Boolean.Input("verbose", default=False),
            ],
            outputs=[
                _KSAMPLER_OPTIONS_TYPE.Output("options"),
            ],
        )

    @classmethod
    def execute(cls, crop_factor=3.0, megapixels=0.0, scale_factor=1.0, multiple=32, mask_grow=0, mask_invert=False,
                upscale_method="lanczos", inpaint_mode=True, mask_mode=True,
                mask_scale_start=1.0, mask_scale_end=1.0, mask_indices="",
                verbose=False):
        option = {
            "type": "inpaint",
            "crop_factor": crop_factor,
            "megapixels": megapixels,
            "scale_factor": scale_factor,
            "multiple": multiple,
            "mask_grow": mask_grow,
            "mask_invert": mask_invert,
            "upscale_method": upscale_method,
            "inpaint_mode": "masked_only" if inpaint_mode else "whole",
            "mask_mode": "single" if mask_mode else "split",
            "mask_scale_start": mask_scale_start,
            "mask_scale_end": mask_scale_end,
            "mask_indices": mask_indices,
            "verbose": verbose,
        }
        return io.NodeOutput([option])

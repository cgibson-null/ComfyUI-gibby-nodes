import json
import torch
from comfy_api.latest import io

from .context import _CONTEXT_TYPE, ctx_from
from .crop_image_by_mask import _CROP_INFO_TYPE, _store_crop_info
from .detection import _bbox_xyxy


class GibbyCropImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GibbyCropImage",
            display_name="Crop Image (Context)",
            category="gibby/image",
            search_aliases=["crop", "cut", "trim", "mask", "bboxes"],
            description=(
                "Crops the image to the region and outputs the crop, a mask of the "
                "crop area, the region's bboxes and center/corner coords. Optional "
                "context in/out: its values pass through; the output context carries "
                "the cropped image, the mask and the crop info (which holds the "
                "originals for pasting back with Image Paste By Mask (Batch) (Context))."
            ),
            inputs=[
                _CONTEXT_TYPE.Input("context", optional=True,
                                    tooltip="Base context; its values pass through and it carries the cropped image, mask and crop info"),
                io.Image.Input("image"),
                io.BoundingBox.Input("crop_region", component="ImageCrop"),
            ],
            outputs=[
                _CONTEXT_TYPE.Output("context"),
                io.Image.Output("image"),
                io.Mask.Output("mask"),
                io.BoundingBox.Output("bboxes"),
                io.String.Output("positive_coords"),
                io.String.Output("negative_coords"),
                _CROP_INFO_TYPE.Output("crop_info",
                                       tooltip="For Image Paste By Mask (Batch) (Context): the original image, the mask and the crop rect as the paste rect"),
            ],
        )

    @classmethod
    def execute(cls, image, crop_region, context=None) -> io.NodeOutput:
        B, H, W, C = image.shape
        # The crop region's box (x1, y1, x2, y2), clamped to the image; the same
        # bbox formats the Mask/Segment (Context) node accepts
        x1, y1, x2, y2 = _bbox_xyxy(crop_region)
        x1, y1 = int(min(x1, W - 1)), int(min(y1, H - 1))
        x2, y2 = int(min(x2, W)), int(min(y2, H))

        # Cropped image
        img = image[:, y1:y2, x1:x2, :]

        # Mask: 1.0 in crop area (foreground), 0.0 outside (background)
        mask = torch.zeros(B, H, W, dtype=torch.float32)
        mask[:, y1:y2, x1:x2] = 1.0

        # Bboxes: single box in original image coordinates
        bboxes = {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1}

        # Positive coords: center of the crop region
        cx = x1 + (x2 - x1) // 2
        cy = y1 + (y2 - y1) // 2
        positive_coords = json.dumps([{"x": cx, "y": cy}])

        # Negative coords: 4 corners of the full image
        negative_coords = json.dumps([
            {"x": 0, "y": 0},
            {"x": W - 1, "y": 0},
            {"x": 0, "y": H - 1},
            {"x": W - 1, "y": H - 1},
        ])

        # The context carries the node's cropped outputs; the originals live in
        # crop_info, the crop rect (the same for every frame) as the paste rect.
        rect = (x1, y1, x2 - x1, y2 - y1)
        ctx = ctx_from(context)
        info = _store_crop_info(ctx, image, mask, [rect] * B, img)

        return io.NodeOutput(ctx, img, mask, bboxes, positive_coords, negative_coords, info)

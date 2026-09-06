import json
import torch
from comfy_api.latest import ComfyExtension, io


class GibbyCropImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GibbyCropImage",
            display_name="Crop Image (Context)",
            category="gibby/image",
            search_aliases=["crop", "cut", "trim", "mask", "bboxes"],
            inputs=[
                io.Image.Input("image"),
                io.BoundingBox.Input("crop_region", component="ImageCrop"),
            ],
            outputs=[
                io.Image.Output("image"),
                io.Mask.Output("mask"),
                io.BoundingBox.Output("bboxes"),
                io.String.Output("positive_coords"),
                io.String.Output("negative_coords"),
            ],
        )

    @classmethod
    def execute(cls, image, crop_region) -> io.NodeOutput:
        x = int(crop_region.get("x", 0))
        y = int(crop_region.get("y", 0))
        width = int(crop_region.get("width", 512))
        height = int(crop_region.get("height", 512))

        B, H, W, C = image.shape
        x = min(x, W - 1)
        y = min(y, H - 1)
        to_x = min(width + x, W)
        to_y = min(height + y, H)

        # Cropped image
        img = image[:, y:to_y, x:to_x, :]

        # Mask: 1.0 in crop area (foreground), 0.0 outside (background)
        mask = torch.zeros(B, H, W, dtype=torch.float32)
        mask[:, y:to_y, x:to_x] = 1.0

        # Bboxes: single box in original image coordinates
        bboxes = {"x": x, "y": y, "width": to_x - x, "height": to_y - y}

        # Positive coords: center of the crop region
        cx = x + (to_x - x) // 2
        cy = y + (to_y - y) // 2
        positive_coords = json.dumps([{"x": cx, "y": cy}])

        # Negative coords: 4 corners of the full image
        negative_coords = json.dumps([
            {"x": 0, "y": 0},
            {"x": W - 1, "y": 0},
            {"x": 0, "y": H - 1},
            {"x": W - 1, "y": H - 1},
        ])

        return io.NodeOutput(img, mask, bboxes, positive_coords, negative_coords)


class GibbyCropImageExtension(ComfyExtension):
    async def get_node_list(self):
        return [GibbyCropImage]


async def comfy_entrypoint() -> GibbyCropImageExtension:
    return GibbyCropImageExtension()

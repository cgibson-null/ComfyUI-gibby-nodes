import comfy.utils
import torch
import torch.nn.functional as F
from comfy_api.latest import io

from .crop_image_by_mask import _CROP_INFO_TYPE


class GibbyPasteImageByMaskBatch(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Gibby_PasteImageByMask_Batch",
            display_name="Image Paste By Mask (Batch)",
            category="gibby/image",
            search_aliases=["paste", "mask", "batch"],
            description="Pastes images back onto the originals by mask: from an Image Crop By Mask (Batch) crop info (the exact inverse of the crop), or standalone with images_original and masks, where each mask's bbox is the paste area.",
            inputs=[
                _CROP_INFO_TYPE.Input("crop_info", optional=True,
                                      tooltip="Output of Image Crop By Mask (Batch): the originals, the mask and the exact paste rectangles"),
                io.Image.Input("images_to_paste"),
                io.Image.Input("images_original", optional=True,
                               tooltip="Overrides the originals from the crop info"),
                io.Mask.Input("masks", optional=True,
                              tooltip="Overrides the mask from the crop info, e.g. the original mask expanded with blur"),
            ],
            outputs=[
                io.Image.Output("image"),
            ],
        )

    @classmethod
    def execute(cls, images_to_paste, crop_info=None, images_original=None, masks=None):
        if crop_info is None:
            if images_original is None or masks is None:
                raise ValueError("connect a crop_info output, or provide both images_original and masks")
            original, mask = images_original, masks
        else:
            original = images_original if images_original is not None else crop_info["image"]
            mask = masks if masks is not None else crop_info["mask"]

        # Soft masks (e.g. the crop mask expanded with blur) blend as-is
        B, H, W, C = original.shape
        if mask.shape[1:] != (H, W):
            mask = F.interpolate(mask.unsqueeze(1), size=(H, W), mode="nearest-exact").squeeze(1)
        BM = mask.shape[0]

        if images_to_paste.shape[0] != B:
            raise ValueError(f"images_to_paste has {images_to_paste.shape[0]} images, the originals have {B}")

        # Paste rectangle per frame: the crop's source rect, or the mask's bbox
        if crop_info is not None:
            rects = crop_info["rects"]
        else:
            rects = []
            for i in range(B):
                m = mask[min(i, BM - 1)]
                rows = torch.any(m > 0, dim=1)
                if rows.any():
                    ys = torch.where(rows)[0]
                    xs = torch.where(torch.any(m > 0, dim=0))[0]
                    rects.append((int(xs[0]), int(ys[0]), int(xs[-1] - xs[0] + 1), int(ys[-1] - ys[0] + 1)))
                else:
                    rects.append((0, 0, W, H))

        out = original.clone()
        for i in range(B):
            sx0, sy0, sw, sh = rects[i]
            m = mask[min(i, BM - 1)][sy0:sy0 + sh, sx0:sx0 + sw]
            m = m.clamp(0, 1).unsqueeze(-1).to(original.dtype)
            pasted = images_to_paste[i:i + 1]
            if pasted.shape[2] != sh or pasted.shape[3] != sw:
                pasted = comfy.utils.common_upscale(pasted.movedim(-1, 1), sw, sh, "lanczos", "disabled").movedim(1, -1)
            region = out[i:i + 1, sy0:sy0 + sh, sx0:sx0 + sw, :]
            out[i, sy0:sy0 + sh, sx0:sx0 + sw] = (pasted * m + region * (1 - m)).squeeze(0)

        return io.NodeOutput(out)

import torch
from comfy_api.latest import io

from .context import _CONTEXT_TYPE, ctx_from
from .crop_image_by_mask import _CROP_INFO_TYPE
from .resolution_latent import _resize_image, _resize_mask, _mask_bbox


class GibbyPasteImageByMaskBatch(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Gibby_PasteImageByMask_Batch",
            display_name="Image Paste By Mask (Batch) (Context)",
            category="gibby/image",
            search_aliases=["paste", "mask", "batch"],
            description="Pastes images back onto the originals by mask: from an Image Crop By Mask (Batch) (Context) crop info (the exact inverse of the crop), or standalone with images_original and masks, where each mask's bbox is the paste area. Optional context in/out: its image is pasted back by default and its crop_info is used when none is connected; the output context carries the pasted-back images.",
            inputs=[
                _CONTEXT_TYPE.Input("context", optional=True,
                                     tooltip="Base context; its image is pasted back by default and its crop_info is used when none is connected"),
                _CROP_INFO_TYPE.Input("crop_info", optional=True,
                                      tooltip="Overrides the context's crop info: the originals, the mask and the exact paste rectangles"),
                io.Image.Input("images_to_paste", optional=True,
                               tooltip="Overrides the context's image: what to paste back"),
                io.Image.Input("images_original", optional=True,
                               tooltip="Overrides the originals from the crop info"),
                io.Mask.Input("masks", optional=True,
                              tooltip="Overrides the mask from the crop info, e.g. the original mask expanded with blur"),
            ],
            outputs=[
                _CONTEXT_TYPE.Output("context"),
                io.Image.Output("image"),
            ],
        )

    @classmethod
    def execute(cls, context=None, images_to_paste=None, crop_info=None, images_original=None, masks=None):
        ctx = ctx_from(context)
        # Connected inputs win over the context's values
        crop_info = crop_info if crop_info is not None else ctx.get("crop_info")
        images_to_paste = images_to_paste if images_to_paste is not None else ctx.get("image")
        if images_to_paste is None:
            raise ValueError("nothing to paste: connect images_to_paste or a context carrying an image")

        if crop_info is None:
            if images_original is None or masks is None:
                raise ValueError("connect a crop_info (or a context carrying one), or provide both images_original and masks")
            original, mask = images_original, masks
        else:
            original = images_original if images_original is not None else crop_info["image"]
            mask = masks if masks is not None else crop_info["mask"]

        # Soft masks (e.g. the crop mask expanded with blur) blend as-is
        B, H, W, C = original.shape
        if mask.shape[1:] != (H, W):
            mask = _resize_mask(mask, W, H, "nearest-exact")
        BM = mask.shape[0]

        if images_to_paste.shape[0] != B:
            raise ValueError(f"images_to_paste has {images_to_paste.shape[0]} images, the originals have {B}")

        # Paste rectangle per frame: the crop's source rect, or the mask's bbox
        if crop_info is not None:
            rects = crop_info["rects"]
        else:
            rects = []
            for i in range(B):
                bbox = _mask_bbox(mask[min(i, BM - 1)])
                rects.append(bbox if bbox is not None else (0, 0, W, H))

        out = original.clone()
        for i in range(B):
            sx0, sy0, sw, sh = rects[i]
            m = mask[min(i, BM - 1)][sy0:sy0 + sh, sx0:sx0 + sw]
            m = m.clamp(0, 1).unsqueeze(-1).to(original.dtype)
            pasted = images_to_paste[i:i + 1]
            if pasted.shape[2] != sh or pasted.shape[3] != sw:
                pasted = _resize_image(pasted, sw, sh, "lanczos")
            region = out[i:i + 1, sy0:sy0 + sh, sx0:sx0 + sw, :]
            out[i, sy0:sy0 + sh, sx0:sx0 + sw] = (pasted * m + region * (1 - m)).squeeze(0)

        # The context carries the pasted-back images; the mask and crop info are discarded
        ctx["image"] = out
        ctx.pop("mask", None)
        ctx.pop("crop_info", None)
        return io.NodeOutput(ctx, out)

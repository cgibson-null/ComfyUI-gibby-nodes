import torch
from comfy_api.latest import io

from .context import _CONTEXT_TYPE, ctx_from
from .crop_image_by_mask import _CROP_INFO_TYPE
from .resolution_latent import _resize_image, _resize_mask, _mask_bbox


def _paste_back(full, pasted, mask, x0, y0, w, h):
    """Blend pasted (B, H, W, C) back into full's (x0, y0, w, h) rect, weighted by
    the mask (B, H, W) - both resized to the rect when they differ. full is
    modified in place."""
    if pasted.shape[1] != h or pasted.shape[2] != w:
        pasted = _resize_image(pasted, w, h, "lanczos")
    if mask.shape[-2:] != (h, w):
        mask = _resize_mask(mask, w, h, "bilinear")
    m = mask.clamp(0, 1).unsqueeze(-1).to(full.dtype)
    full[:, y0:y0 + h, x0:x0 + w, :] = pasted * m + full[:, y0:y0 + h, x0:x0 + w, :] * (1 - m)


class GibbyPasteImageByMaskBatch(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Gibby_PasteImageByMask_Batch",
            display_name="Image Paste By Mask (Batch) (Context)",
            category="gibby/image",
            search_aliases=["paste", "mask", "batch"],
            description="Pastes images back onto the originals by mask: from an Image Crop By Mask (Batch) (Context) crop info (the exact inverse of the crop), or standalone with images_original and masks, where each mask's bbox is the paste area; when the batches differ, the smaller size wins. With composite on it instead pastes like Image Composite Masked: the pasted image is resized to the original's size when it differs and a source-sized mask is applied to it (no mask pastes opaquely). Optional context in/out: its image is pasted back by default and its crop_info is used when none is connected; the output context carries the pasted-back images.",
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
                              tooltip="Overrides the mask from the crop info, e.g. the original mask expanded with blur; in composite mode it is source-sized and applied to the pasted image (no mask pastes opaquely)"),
                io.Boolean.Input("composite", default=False,
                                 tooltip="Paste like Image Composite Masked: the pasted image is resized to the original's size when it differs and the mask is source-sized, applied to it (no mask pastes opaquely)"),
            ],
            outputs=[
                _CONTEXT_TYPE.Output("context"),
                io.Image.Output("image"),
            ],
        )

    @classmethod
    def execute(cls, context=None, images_to_paste=None, crop_info=None, images_original=None, masks=None,
                composite=False):
        ctx = ctx_from(context)
        # Connected inputs win over the context's values
        crop_info = crop_info if crop_info is not None else ctx.get("crop_info")
        images_to_paste = images_to_paste if images_to_paste is not None else ctx.get("image")
        if images_to_paste is None:
            raise ValueError("nothing to paste: connect images_to_paste or a context carrying an image")

        if composite:
            # Paste like Image Composite Masked: the pasted image is resized to the original's
            # size when it differs and the mask is source-sized, applied to it (absent = opaque)
            if images_original is None and crop_info is None:
                raise ValueError("composite mode needs images_original or a crop_info (or a context carrying one)")
            original = images_original if images_original is not None else crop_info["image"]
            mask = masks if masks is not None else (crop_info["mask"] if crop_info is not None else None)
        elif crop_info is None:
            if images_original is None or masks is None:
                raise ValueError("connect a crop_info (or a context carrying one), or provide both images_original and masks")
            original, mask = images_original, masks
        else:
            original = images_original if images_original is not None else crop_info["image"]
            mask = masks if masks is not None else crop_info["mask"]

        # Soft masks (e.g. the crop mask expanded with blur) blend as-is
        B, H, W, C = original.shape
        if mask is not None:
            if not composite and mask.shape[1:] != (H, W):
                mask = _resize_mask(mask, W, H, "nearest-exact")
            BM = mask.shape[0]

        # Batches need not match: composite pastes onto every original (a shorter source
        # batch reuses its last frame, like Image Composite Masked), otherwise the
        # smaller size wins
        n = B if composite else min(B, images_to_paste.shape[0])

        # Match the originals' channel count (e.g., an RGBA paste over RGB drops the alpha)
        if images_to_paste.shape[-1] != C:
            c = images_to_paste.shape[-1]
            images_to_paste = images_to_paste[..., :C] if c > C else torch.cat(
                [images_to_paste, images_to_paste.new_zeros(images_to_paste.shape[:-1] + (C - c,))], -1)

        out = original[:n].clone()
        if composite:
            # Like Image Composite Masked: the pasted image is resized to the original's size
            # when it differs and blended over the whole frame with a source-sized mask; no
            # mask pastes opaquely
            S = images_to_paste.shape[0]
            for i in range(n):
                src = images_to_paste[min(i, S - 1):min(i, S - 1) + 1]
                if src.shape[1:3] != (H, W):
                    src = _resize_image(src, W, H, "bilinear")
                if mask is None:
                    m = src.new_ones(1, 1, H, W)
                else:
                    m = mask[min(i, BM - 1)].unsqueeze(0)
                _paste_back(out[i:i + 1], src, m, 0, 0, W, H)
        else:
            # Paste rectangle per frame: the crop's source rect, or the mask's bbox
            if crop_info is not None:
                rects = crop_info["rects"]
            else:
                rects = []
                for i in range(n):
                    bbox = _mask_bbox(mask[min(i, BM - 1)])
                    rects.append(bbox if bbox is not None else (0, 0, W, H))
            for i in range(n):
                sx0, sy0, sw, sh = rects[i]
                m = mask[min(i, BM - 1)][sy0:sy0 + sh, sx0:sx0 + sw].unsqueeze(0)
                _paste_back(out[i:i + 1], images_to_paste[i:i + 1], m, sx0, sy0, sw, sh)

        # The context carries the pasted-back images; the mask and crop info are discarded
        ctx["image"] = out
        ctx.pop("mask", None)
        ctx.pop("crop_info", None)
        return io.NodeOutput(ctx, out)

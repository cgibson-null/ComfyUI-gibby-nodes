import torch
import nodes
from comfy_api.latest import io
from comfy_extras.nodes_mask import GrowMask, InvertMask

from .context import _CONTEXT_TYPE, ctx_from
from .resolution_latent import _resize_to_mp_scale, _resize_image, _resize_mask, _mask_bbox, _pad_color_tensor, _color_alpha


# Carries the originals, the mask and the paste rectangles to Image Paste By Mask (Batch) (Context)
_CROP_INFO_TYPE = io.Custom("GIBBY_CROP_INFO")

# Two consecutive crop regions count as a shot change when they barely overlap
_CUT_IOU = 0.2
# or when their area jumps — the mask degenerating into a body region (or back to a face)
_SIZE_JUMP = 2.0
# A real mask fills most of its bbox; sparse masks are noise
_MIN_FILL = 0.25
# A mask smaller than this fraction of a frame dimension is noise
_MIN_SIZE = 0.05


def _process_mask(mask, mask_grow, mask_invert):
    """Grow first, then invert: growing before inverting leaves a margin of
    unmasked pixels around the region (inverting first would grow the
    complement into it)."""
    if mask_grow:
        mask, = GrowMask.execute(mask, mask_grow, True)
    if mask_invert:
        mask, = InvertMask.execute(mask)
    return mask


def _bbox_iou(a, b):
    x0 = max(a[4], b[4])
    y0 = max(a[5], b[5])
    x1 = min(a[4] + a[6], b[4] + b[6])
    y1 = min(a[5] + a[7], b[5] + b[7])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = a[6] * a[7] + b[6] * b[7] - inter
    return inter / union if union else 0.0


def _stabilize_regions(regions, W, H, center_smoothing, size_source):
    # Segment the batch into shots: a cut is where consecutive crops stop overlapping
    # or their area jumps
    starts = [0]
    for i in range(1, len(regions)):
        a, b = regions[i - 1], regions[i]
        area_a, area_b = a[6] * a[7], b[6] * b[7]
        jump = area_b / area_a if area_a else float("inf")
        if _bbox_iou(a, b) < _CUT_IOU or jump > _SIZE_JUMP or jump < 1 / _SIZE_JUMP:
            starts.append(i)

    # Lock the crop size per shot (or per clip) to its largest crop; the first
    # frame of a shot is cropped raw, at its own size
    n = len(regions)
    if size_source == "clip_max":
        sizes = [(max(r[6] for r in regions), max(r[7] for r in regions))] * n
    else:  # shot_max
        sizes = []
        for s, start in enumerate(starts):
            end = starts[s + 1] if s + 1 < len(starts) else n
            w = max(r[6] for r in regions[start:end])
            h = max(r[7] for r in regions[start:end])
            sizes.append((regions[start][6], regions[start][7]))
            sizes.extend([(w, h)] * (end - start - 1))

    # Smooth the crop center per shot; the first frame of a shot jumps to its raw center
    shot_starts = set(starts)
    out = []
    prev = None
    for i, r in enumerate(regions):
        if i == 0 or i in shot_starts:
            cx, cy = r[2], r[3]
        else:
            cx = r[2] * (1 - center_smoothing) + prev[0] * center_smoothing
            cy = r[3] * (1 - center_smoothing) + prev[1] * center_smoothing
        prev = (cx, cy)
        w, h = sizes[i]
        x0 = max(0, min(int(cx - w / 2), W - 1))
        y0 = max(0, min(int(cy - h / 2), H - 1))
        x1 = max(x0 + 1, min(int(cx + w / 2), W))
        y1 = max(y0 + 1, min(int(cy + h / 2), H))
        out.append((r[0], r[1], cx, cy, x0, y0, x1 - x0, y1 - y0))
    return out


def _mask_box(bbox, W, H, crop_factor):
    """(cx, cy, x0, y0, w, h): the crop box from a mask's bbox (x, y, w, h) scaled by
    crop_factor and clamped to the image; the full frame when the bbox is None.
    The center is the pixel range's center, so crop_factor=1.0 reproduces the bbox."""
    x_min, y_min, bw, bh = bbox if bbox is not None else (0, 0, W, H)
    cx = (x_min + x_min + bw) / 2
    cy = (y_min + y_min + bh) / 2
    w = bw * crop_factor
    h = bh * crop_factor
    x0 = max(0, min(int(cx - w / 2), W - 1))
    y0 = max(0, min(int(cy - h / 2), H - 1))
    x1 = max(x0 + 1, min(int(cx + w / 2), W))
    y1 = max(y0 + 1, min(int(cy + h / 2), H))
    return (cx, cy, x0, y0, x1 - x0, y1 - y0)


def _cover_crop(image, mask, cx, cy, w, h, target_w, target_h, method="lanczos"):
    """Fit the crop box (centered on cx, cy, size w x h) to the target size: the crop
    stays centered and is scaled up until it fills the target (cover-fit), so a shape
    mismatch trims the crop's edges. Returns (crop, crop_mask, rect); rect (sx0, sy0,
    src_w, src_h) is where the crop came from. image is (B, H, W, C), mask (B, H, W)."""
    s = max(target_w / w, target_h / h)
    src_w = max(1, int(target_w / s))
    src_h = max(1, int(target_h / s))
    sx0 = max(0, min(int(cx - src_w / 2), image.shape[2] - src_w))
    sy0 = max(0, min(int(cy - src_h / 2), image.shape[1] - src_h))
    crop = image[:, sy0:sy0 + src_h, sx0:sx0 + src_w, :]
    crop_mask = mask[:, sy0:sy0 + src_h, sx0:sx0 + src_w]
    if src_w != target_w or src_h != target_h:
        crop = _resize_image(crop, target_w, target_h, method)
        # Nearest keeps the rounded mask's hard edges
        crop_mask = _resize_mask(crop_mask, target_w, target_h, "nearest")
    return crop, crop_mask, (sx0, sy0, src_w, src_h)


class GibbyCropImageByMaskBatch(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Gibby_CropImageByMask_Batch",
            display_name="Image Crop By Mask (Batch) (Context)",
            category="gibby/image",
            search_aliases=["crop", "mask", "batch"],
            description="Crops each image by its own mask (image i by mask i); every crop is scaled up until it fills the total size, so subjects keep a consistent scale across the batch - when a crop's shape differs from the canvas its edges are trimmed. Optional temporal stabilization for video batches: constant crop size per shot and a smoothed crop center, with cuts detected from mask jumps. Also outputs the crop info for pasting results back with Image Paste By Mask (Batch) (Context). Optional context in/out: its image and mask are used when none are connected; the output context carries the cropped image and mask plus the crop info (which holds the originals).",
            inputs=[
                _CONTEXT_TYPE.Input("context", optional=True,
                                     tooltip="Base context; its image and mask are used when none are connected"),
                io.Image.Input("image", optional=True, tooltip="Overrides the context image"),
                io.Mask.Input("mask", optional=True, tooltip="Overrides the context mask"),
                io.Float.Input("crop_factor", default=1.5, min=1.0, max=10.0, step=0.1,
                               tooltip="Scales the cropped area outside of mask bounds"),
                io.Float.Input("megapixels", default=1.0, min=0.0, max=100.0, step=0.1,
                               tooltip="Target megapixels for the cropped area (0=off, just rescale). Applied after crop_factor."),
                io.Float.Input("scale_factor", default=1.0, min=0.1, max=10.0, step=0.1,
                               tooltip="Resolution multiplier for the cropped area (1.0=off). Applied after crop_factor."),
                io.Int.Input("multiple", default=32, min=1, max=256, step=1,
                             tooltip="Round the cropped size down to this multiple so sampling won't re-adjust it"),
                io.Int.Input("mask_grow", default=0, min=-nodes.MAX_RESOLUTION, max=nodes.MAX_RESOLUTION, step=1,
                             tooltip="Grows the mask by this many pixels before cropping (negative shrinks it), like the native Grow Mask with tapered corners"),
                io.Boolean.Input("mask_invert", default=False,
                                 tooltip="Inverts the mask after growing and before cropping, like the native Invert Mask"),
                io.Boolean.Input("enable_smoothing", default=False,
                                 tooltip="Stabilize the crop across the batch: constant size per shot, smoothed center, cuts from mask jumps; empty and noise masks hold the previous crop"),
                io.Float.Input("center_smoothing", default=0.8, min=0.0, max=1.0, step=0.05,
                               tooltip="Weight of the previous frame's crop center (0=off); resets at detected cuts"),
                io.Combo.Input("size_source", default="shot_max", options=["shot_max", "clip_max"],
                               tooltip="shot_max: constant size per shot. clip_max: one size for the whole clip."),
                io.Boolean.Input("remove_bg", default=False,
                                 tooltip="Replaces the unmasked area with the color"),
                io.Color.Input("color", optional=True, socketless=False, default="#000000",
                               tooltip="Background color for remove_bg; connect the hex output of a Color Picker. Alpha 0 makes the background transparent instead (RGBA output)"),
            ],
            outputs=[
                _CONTEXT_TYPE.Output("context"),
                io.Image.Output("image"),
                io.Mask.Output("mask"),
                _CROP_INFO_TYPE.Output("crop_info"),
            ],
        )

    @classmethod
    def execute(cls, context=None, image=None, mask=None, crop_factor=1.5, megapixels=1.0, scale_factor=1.0, multiple=32,
                mask_grow=0, mask_invert=False, enable_smoothing=False,
                center_smoothing=0.8, size_source="shot_max",
                remove_bg=False, color="#000000"):
        ctx = ctx_from(context)
        # Connected image/mask override the context's
        image = image if image is not None else ctx.get("image")
        mask = mask if mask is not None else ctx.get("mask")
        if image is None:
            raise ValueError("No image to crop: connect an image or a context carrying one")
        if mask is None:
            raise ValueError("No mask to crop by: connect a mask or a context carrying one")
        B, H, W, C = image.shape
        if mask.shape[1:] != (H, W):
            mask = _resize_mask(mask, W, H, "nearest-exact")
        mask = mask.round()
        # Grow first, then invert: the crop region and the output mask follow the processed mask
        mask = _process_mask(mask, mask_grow, mask_invert)
        BM = mask.shape[0]

        bg = _pad_color_tensor(color, image.dtype, image.device)
        # A fully transparent color makes the unmasked area transparent instead of colored
        transparent = remove_bg and _color_alpha(color) == 0.0

        # Bounding box per mask; None when the mask is empty. A mask that fills less than
        # _MIN_FILL of its bbox, or is smaller than _MIN_SIZE of a frame dimension, is noise.
        boxes = []
        failed = []
        for m in mask:
            bbox = _mask_bbox(m)
            if bbox is not None:
                x_min, y_min, w, h = bbox
                fill = float((m > 0).sum()) / (w * h)
                boxes.append(bbox)
                failed.append(fill < _MIN_FILL or w < _MIN_SIZE * W or h < _MIN_SIZE * H)
            else:
                boxes.append(None)
                failed.append(False)

        # Crop region per image: its mask's bbox scaled by crop_factor, clamped to the image.
        # An empty or noise mask borrows the closest good mask before it, so the crop holds.
        regions = []  # (image_idx, mask_idx, cx, cy, x0, y0, w, h)
        for i in range(B):
            idx = min(i, BM - 1)
            if boxes[idx] is None or failed[idx]:
                for j in range(idx - 1, -1, -1):
                    if boxes[j] is not None and not failed[j]:
                        idx = j
                        break
            cx, cy, x0, y0, w, h = _mask_box(boxes[idx], W, H, crop_factor)
            regions.append((i, idx, cx, cy, x0, y0, w, h))

        if enable_smoothing:
            regions = _stabilize_regions(regions, W, H, center_smoothing, size_source)

        max_w = max(w for *_, w, h in regions)
        max_h = max(h for *_, w, h in regions)
        # Target pixel count for the cropped area, keeping its aspect (Resize Image principle)
        max_w, max_h = _resize_to_mp_scale(max_w, max_h, megapixels, scale_factor, multiple)

        # Fit every crop to the total size: the crop stays centered and is scaled up
        # uniformly until it fills the canvas (cover-fit). When the crop's shape differs
        # from the canvas, its edges are trimmed; widening the source area instead would
        # shrink subjects by a per-frame aspect mismatch, so a far shot would end up with
        # a smaller subject than the close-ups it shares the canvas with.
        out = []
        out_mask = []
        rects = []  # (sx0, sy0, src_w, src_h) per frame: where the crop came from
        for i, idx, cx, cy, x0, y0, w, h in regions:
            region, m, rect = _cover_crop(image[i:i + 1], mask[idx:idx + 1], cx, cy, w, h, max_w, max_h)
            rects.append(rect)
            if remove_bg:
                mb = m.clamp(0, 1).unsqueeze(-1).to(image.dtype)
                if transparent:
                    # Join Image with Alpha's method: keep the RGB, the mask becomes the alpha
                    region = torch.cat((region[..., :3], mb), dim=-1)
                else:
                    region = region * mb + bg * (1 - mb)
            out.append(region.squeeze(0))
            out_mask.append(m.squeeze(0))

        out_image = torch.stack(out)
        out_mask = torch.stack(out_mask)
        info = {"image": image, "mask": mask, "rects": rects}
        # The context carries the node's cropped outputs; the originals live in crop_info
        ctx["image"] = out_image
        ctx["mask"] = out_mask
        ctx["crop_info"] = info
        return io.NodeOutput(ctx, out_image, out_mask, info)

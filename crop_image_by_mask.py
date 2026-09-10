import math

import comfy.utils
import torch
import torch.nn.functional as F
from comfy_extras.color_util import hex_to_rgb
from comfy_api.latest import io


# Carries the originals, the mask and the paste rectangles to Image Paste By Mask (Batch)
_CROP_INFO_TYPE = io.Custom("GIBBY_CROP_INFO")

# Two consecutive crop regions count as a shot change when they barely overlap
_CUT_IOU = 0.2
# or when their area jumps — the mask degenerating into a body region (or back to a face)
_SIZE_JUMP = 2.0
# A real mask fills most of its bbox; sparse masks are noise
_MIN_FILL = 0.25
# A mask smaller than this fraction of a frame dimension is noise
_MIN_SIZE = 0.05


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


class GibbyCropImageByMaskBatch(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Gibby_CropImageByMask_Batch",
            display_name="Image Crop By Mask (Batch)",
            category="gibby/image",
            search_aliases=["crop", "mask", "batch"],
            description="Crops each image by its own mask (image i by mask i); smaller crops are scaled up to the total size, taking the surrounding pixels from the original image. Optional temporal stabilization for video batches: constant crop size per shot and a smoothed crop center, with cuts detected from mask jumps. Also outputs the crop info for pasting results back with Image Paste By Mask (Batch).",
            inputs=[
                io.Image.Input("image"),
                io.Mask.Input("mask"),
                io.Float.Input("crop_factor", default=1.5, min=1.0, max=10.0, step=0.1,
                               tooltip="Scales the cropped area outside of mask bounds"),
                io.Float.Input("megapixels", default=1.0, min=0.0, max=100.0, step=0.1,
                               tooltip="Target megapixels for the cropped area (0=off, just rescale). Applied after crop_factor."),
                io.Float.Input("scale_factor", default=1.0, min=0.1, max=10.0, step=0.1,
                               tooltip="Resolution multiplier for the cropped area (1.0=off). Applied after crop_factor."),
                io.Int.Input("multiple", default=8, min=1, max=256, step=1,
                             tooltip="Round the cropped size down to this multiple so sampling won't re-adjust it"),
                io.Boolean.Input("enable_smoothing", default=False,
                                 tooltip="Stabilize the crop across the batch: constant size per shot, smoothed center, cuts from mask jumps; empty and noise masks hold the previous crop"),
                io.Float.Input("center_smoothing", default=0.8, min=0.0, max=1.0, step=0.05,
                               tooltip="Weight of the previous frame's crop center (0=off); resets at detected cuts"),
                io.Combo.Input("size_source", default="shot_max", options=["shot_max", "clip_max"],
                               tooltip="shot_max: constant size per shot. clip_max: one size for the whole clip."),
                io.Boolean.Input("remove_bg", default=False,
                                 tooltip="Replaces the unmasked area with the color"),
                io.Color.Input("color", optional=True, socketless=False, default="#000000",
                               tooltip="Background color for remove_bg; connect the hex output of a Color Picker"),
            ],
            outputs=[
                io.Image.Output("image"),
                io.Mask.Output("mask"),
                _CROP_INFO_TYPE.Output("crop_info"),
            ],
        )

    @classmethod
    def execute(cls, image, mask, crop_factor=1.5, megapixels=1.0, scale_factor=1.0, multiple=8, enable_smoothing=False,
                center_smoothing=0.8, size_source="shot_max",
                remove_bg=False, color="#000000"):
        B, H, W, C = image.shape
        if mask.shape[1:] != (H, W):
            mask = F.interpolate(mask.unsqueeze(1), size=(H, W), mode="nearest-exact").squeeze(1)
        mask = mask.round()
        BM = mask.shape[0]

        r, g, b = hex_to_rgb(color)
        bg = torch.tensor([r, g, b], dtype=image.dtype, device=image.device) / 255.0

        # Bounding box per mask; None when the mask is empty. A mask that fills less than
        # _MIN_FILL of its bbox, or is smaller than _MIN_SIZE of a frame dimension, is noise.
        boxes = []
        failed = []
        for m in mask:
            rows = torch.any(m > 0, dim=1)
            if rows.any():
                ys = torch.where(rows)[0]
                xs = torch.where(torch.any(m > 0, dim=0))[0]
                x_min, x_max, y_min, y_max = int(xs[0]), int(xs[-1]), int(ys[0]), int(ys[-1])
                w = x_max - x_min + 1
                h = y_max - y_min + 1
                fill = float((m > 0).sum()) / (w * h)
                boxes.append((x_min, x_max, y_min, y_max))
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
            if boxes[idx] is None:
                x_min, x_max, y_min, y_max = 0, W - 1, 0, H - 1
            else:
                x_min, x_max, y_min, y_max = boxes[idx]

            # Center of the pixel range, so crop_factor=1.0 reproduces the bbox exactly
            cx = (x_min + x_max + 1) / 2
            cy = (y_min + y_max + 1) / 2
            w = (x_max - x_min + 1) * crop_factor
            h = (y_max - y_min + 1) * crop_factor
            x0 = max(0, min(int(cx - w / 2), W - 1))
            y0 = max(0, min(int(cy - h / 2), H - 1))
            x1 = max(x0 + 1, min(int(cx + w / 2), W))
            y1 = max(y0 + 1, min(int(cy + h / 2), H))
            regions.append((i, idx, cx, cy, x0, y0, x1 - x0, y1 - y0))

        if enable_smoothing:
            regions = _stabilize_regions(regions, W, H, center_smoothing, size_source)

        max_w = max(w for *_, w, h in regions)
        max_h = max(h for *_, w, h in regions)
        if megapixels > 0:
            # Target pixel count for the cropped area, keeping its aspect (Resize Image principle)
            scale_by = math.sqrt(megapixels * 1024 * 1024 / (max_w * max_h))
            max_w *= scale_by
            max_h *= scale_by
        max_w *= scale_factor
        max_h *= scale_factor
        # Round down to a multiple so sampling doesn't re-adjust the size
        max_w = max(multiple, int(max_w) // multiple * multiple)
        max_h = max(multiple, int(max_h) // multiple * multiple)

        # Fit every crop to the total size: the crop stays centered and is scaled up
        # uniformly, the rest of the canvas is taken from the original image around it
        out = []
        rects = []  # (sx0, sy0, src_w, src_h) per frame: where the crop came from
        for i, idx, cx, cy, x0, y0, w, h in regions:
            s = min(max_w / w, max_h / h)
            src_w = min(int(max_w / s), W)
            src_h = min(int(max_h / s), H)
            sx0 = max(0, min(int(cx - src_w / 2), W - src_w))
            sy0 = max(0, min(int(cy - src_h / 2), H - src_h))
            rects.append((sx0, sy0, src_w, src_h))

            region = image[i:i + 1, sy0:sy0 + src_h, sx0:sx0 + src_w, :]
            if remove_bg:
                m = mask[idx][sy0:sy0 + src_h, sx0:sx0 + src_w]
                m = m.clamp(0, 1).unsqueeze(-1).to(image.dtype)
                region = region * m + bg * (1 - m)
            if src_w != max_w or src_h != max_h:
                region = comfy.utils.common_upscale(region.movedim(-1, 1), max_w, max_h, "lanczos", "disabled").movedim(1, -1)
            out.append(region.squeeze(0))

        info = {"image": image, "mask": mask, "rects": rects}
        return io.NodeOutput(torch.stack(out), mask, info)

"""
Resize Image / Empty Latent (Context)
-------------------------------------
Creates an empty latent from a resolution spec instead of raw width/height.

Four modes (switched by the horizontal mode toggle in resolution_latent.js):
- keep_ar: resizes the linked media to the megapixels target at its own aspect
  ratio (native Scale Image to Total Pixels math);
- custom: explicit width x height;
- aspect_ratio: core Resolution Selector presets + megapixels;
- custom_aspect_ratio: manual x : y floats + megapixels.

Megapixels 0 in the ratio modes: no target pixel count - the media's own size
is used (rescaled by scale_factor) instead, or 1 MP when no media is linked.

All widgets are always visible; the mode only selects which math drives the
target size.

Shared across all modes: swap_dimensions, scale_factor (resolution multiplier),
multiple (round down to this multiple, advanced) and batch_size. The latent type
toggle picks between the standard 4ch /8x image latent and Flux2's 128ch /16x one.
Outputs LATENT plus the computed width/height for downstream sizing.

With an image or mask connected: the mode's target size becomes a box that the
media is fitted into using keep_proportion (Resize Image v2 semantics - stretch,
resize, pad with pad_color/crop_position, crop, total_pixels), and the resized
media plus a matching latent are output.

Optional upscale_model (Load Upscale Model): when connected and an image is
linked, the image is first upscaled with it (Upscale Image (using Model))
before the resize. The size spec follows the original image's size, so the
upscaled result is fitted back into the target box (scale factor, not the
model's factor, sets the final size).
"""

import math

import torch
import comfy.model_management
import comfy.model_base
import comfy.utils
from comfy_api.latest import io
from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
from comfy_extras.color_util import hex_to_rgb

from ..context import _CONTEXT_TYPE


# Same presets as core's Resolution Selector.
ASPECT_RATIOS = {
    "1:1 (Square)": (1, 1),
    "2:3 (Portrait Photo)": (2, 3),
    "3:2 (Photo)": (3, 2),
    "3:4 (Portrait Standard)": (3, 4),
    "4:3 (Standard)": (4, 3),
    "9:16 (Portrait Widescreen)": (9, 16),
    "16:9 (Widescreen)": (16, 9),
    "21:9 (Ultrawide)": (21, 9),
}

KEEP_PROPORTIONS = ["stretch", "resize", "pad", "crop", "total_pixels"]
CROP_POSITIONS = ["center", "top", "bottom", "left", "right"]


def _fit_size(sw, sh, box_w, box_h, keep_proportion, step):
    # Content size that fits the source (sw x sh) into the target box per Resize
    # Image v2's keep_proportion modes; always snapped to a valid latent grid.
    if keep_proportion == "total_pixels":
        total, ar = box_w * box_h, sw / sh
        return max(step, int(math.sqrt(total * ar)) // step * step), \
               max(step, int(math.sqrt(total / ar)) // step * step)
    if keep_proportion in ("resize", "pad"):
        ratio = min(box_w / sw, box_h / sh)
        return max(step, int(round(sw * ratio)) // step * step), \
               max(step, int(round(sh * ratio)) // step * step)
    # stretch and crop fill the whole box.
    return box_w, box_h


def _pad_amounts(dw, dh, position):
    if position == "top":
        pl = dw // 2; pr = dw - pl; pt, pb = 0, dh
    elif position == "bottom":
        pl = dw // 2; pr = dw - pl; pt, pb = dh, 0
    elif position == "left":
        pl, pr = 0, dw; pt = dh // 2; pb = dh - pt
    elif position == "right":
        pl, pr = dw, 0; pt = dh // 2; pb = dh - pt
    else:  # center
        pl = dw // 2; pr = dw - pl; pt = dh // 2; pb = dh - pt
    return pl, pr, pt, pb


def _crop_offsets(ow, oh, cw, ch, position):
    if position == "top":
        x = (ow - cw) // 2; y = 0
    elif position == "bottom":
        x = (ow - cw) // 2; y = oh - ch
    elif position == "left":
        x, y = 0, (oh - ch) // 2
    elif position == "right":
        x, y = ow - cw, (oh - ch) // 2
    else:  # center
        x = (ow - cw) // 2; y = (oh - ch) // 2
    return x, y


def _pad_color_tensor(pad_color, dtype, device):
    # Same format as the core Color Picker: #RRGGBB or #RRGGBBAA.
    if len(pad_color) not in (7, 9) or pad_color[0] != "#":
        raise ValueError("Color must be in format #RRGGBB or #RRGGBBAA")
    r, g, b = hex_to_rgb(pad_color[:7])
    return torch.tensor([v / 255.0 for v in (r, g, b)], dtype=dtype, device=device)


def _color_pad(img, left, right, top, bottom, bg):
    # img [B,H,W,C]; canvas filled with the background color, content at (top, left).
    B, H, W, C = img.shape
    out = torch.empty((B, H + top + bottom, W + left + right, C), dtype=img.dtype, device=img.device)
    for c in range(C):
        out[:, :, :, c] = bg[c % 3]
    out[:, top:top + H, left:left + W] = img
    return out


class GibbyEmptyLatentResolution(io.ComfyNode):
    """Create an empty latent from a resolution spec instead of raw width/height."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Gibby_EmptyLatent_Resolution",
            display_name="Resize Image / Empty Latent (Context)",
            category="gibby",
            description=(
                "Creates an empty latent from a resolution spec instead of raw width/height. "
                "Modes: keep AR (megapixels at the media's aspect ratio, like Scale Image to Total Pixels), "
                "custom (width x height), aspect ratio (preset + megapixels) or custom aspect ratio "
                "(manual w:h + megapixels). Megapixels 0 rescales the media's own size instead, or "
                "defaults to 1 MP with no media linked. "
                "Shared: swap dimensions, scale factor, multiple and batch size; latent type toggle for "
                "standard 8x vs Flux2 16x latents. With an image or mask connected it is fitted into the "
                "target box per keep_proportion (stretch/resize/pad/crop/total_pixels) and output resized."
            ),
            inputs=[
                # Optional context: width/height overridden with finalized size; latent or image+latent updated.
                _CONTEXT_TYPE.Input("context", optional=True),
                # Optional: when linked, it is resized per keep_proportion and the size follows.
                io.Image.Input("image", optional=True),
                io.Mask.Input("mask", optional=True),
                # Mode switcher (rendered as a horizontal toggle by resolution_latent.js).
                io.Combo.Input("mode", options=["keep_ar", "custom", "aspect_ratio", "custom_aspect_ratio"], default="aspect_ratio"),
                # custom mode: explicit width x height.
                io.Int.Input("width", default=1024, min=8, max=16384),
                io.Int.Input("height", default=1024, min=8, max=16384),
                # aspect ratio modes.
                io.Combo.Input("aspect_ratio", options=list(ASPECT_RATIOS), default="3:4 (Portrait Standard)"),
                # custom aspect ratio mode: manual w : h floats.
                io.Float.Input("x", default=3.0, min=0.1, max=99.0, step=0.01),
                io.Float.Input("y", default=4.0, min=0.1, max=99.0, step=0.01),
                io.Float.Input("megapixels", default=1.0, min=0.0, max=20.0, step=0.01),
                # Resolution multiplier applied to any mode's target size.
                io.Float.Input("scale_factor", default=1.0, min=0.05, max=16.0, step=0.01),
                # media resize (only used when an image or mask is connected).
                io.Combo.Input("upscale_method", options=["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], default="lanczos"),
                io.Combo.Input("keep_proportion", options=KEEP_PROPORTIONS, default="stretch", advanced=True),
                io.Color.Input("pad_color", default="#000000", advanced=True),
                io.Combo.Input("crop_position", options=CROP_POSITIONS, default="center", advanced=True),
                # shared settings (plain booleans render as pill toggles).
                io.Boolean.Input("swap_dimensions", display_name="Swap dimensions", default=False),
                io.Int.Input("multiple", default=8, min=1, max=256, advanced=True),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                # latent format: on = Flux 2's 128ch /16x latent, off = standard 4ch /8x.
                io.Boolean.Input("flux2_latent", display_name="Flux 2 latent (16x)", default=False),
                # Optional Load Upscale Model: when linked and an image is present, upscale it
                # (Upscale Image (using Model)) before the resize.
                io.UpscaleModel.Input("upscale_model", optional=True),
            ],
            outputs=[
                # Updated context with finalized dimensions/latent/image (new context when none in).
                _CONTEXT_TYPE.Output(display_name="context"),
                io.Latent.Output(display_name="latent"),
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
                # Resized input media (None when not connected).
                io.Image.Output(display_name="image"),
                io.Mask.Output(display_name="mask"),
            ],
        )

    @classmethod
    def execute(cls, context=None, image=None, mask=None, upscale_model=None, mode="aspect_ratio", width=1024, height=1024, aspect_ratio="3:4 (Portrait Standard)", x=3.0, y=4.0,
                megapixels=1.0, swap_dimensions=False, scale_factor=1.0,
                multiple=8, batch_size=1, flux2_latent=False, upscale_method="lanczos",
                keep_proportion="stretch", pad_color="#000000", crop_position="center") -> io.NodeOutput:
        # If no input image but context has one, use context's image
        if image is None and isinstance(context, dict):
            image = context.get("image")

        # Original media W/H before any model upscale; the size spec follows
        # this, and the upscaled image is fitted into the target box.
        OW = OH = 0
        if image is not None:
            OW, OH = image.shape[2], image.shape[1]
        elif mask is not None:
            OW, OH = mask.shape[2], mask.shape[1]

        # Optional model upscale (Load Upscale Model + Upscale Image (using Model)),
        # applied before the resize so the upscaled image is what gets fitted.
        if upscale_model is not None and image is not None:
            image, = ImageUpscaleWithModel.execute(upscale_model, image)

        has_media = image is not None or mask is not None
        # Source W/H; same shape indices for [B,H,W,C] images and [B,H,W] masks.
        SW = SH = 0
        if has_media:
            src = image if image is not None else mask
            SW, SH = src.shape[2], src.shape[1]

        # Use model type from context if provided, else fall back to toggle
        is_flux2 = flux2_latent
        if isinstance(context, dict):
            model_obj = context.get("model")
            if model_obj is not None and hasattr(model_obj, 'model'):
                is_flux2 = isinstance(model_obj.model, comfy.model_base.Flux2)

        # Latent grids must be divisible by 8 (image) or 16 (flux2), so round down to the lcm of multiple and that base.
        base = 16 if is_flux2 else 8
        step = multiple * base // math.gcd(multiple, base)

        if mode == "custom":
            w, h = float(width), float(height)
        else:
            # Resolution Selector math at the selected ratio; keep_ar uses media's ratio or falls back to aspect_ratio.
            if mode == "keep_ar" and has_media:
                rw, rh = float(SW), float(SH)
            elif mode == "aspect_ratio" or (mode == "keep_ar" and not has_media):
                rw, rh = ASPECT_RATIOS[aspect_ratio]
            else:
                rw, rh = float(x), float(y)
            if megapixels == 0 and has_media:
                # 0 megapixels: no target pixel count, rescale the original media's size
                # (before upscale_model), so the upscaled image is fitted back to it.
                w, h = float(OW), float(OH)
            else:
                # 0 megapixels without media defaults to 1 MP.
                scale_by = math.sqrt((megapixels or 1.0) * 1024 * 1024 / (rw * rh))
                w = round(rw * scale_by / step) * step
                h = round(rh * scale_by / step) * step

        w *= scale_factor
        h *= scale_factor
        if swap_dimensions:
            w, h = h, w

        w = max(step, int(w) // step * step)
        h = max(step, int(h) // step * step)

        # Media mode: fit the source into the (w x h) box per keep_proportion.
        if has_media:
            out_w, out_h = _fit_size(SW, SH, w, h, keep_proportion, step)
            pad_left, pad_right, pad_top, pad_bottom = 0, 0, 0, 0
            if keep_proportion == "pad":
                pad_left, pad_right, pad_top, pad_bottom = _pad_amounts(w - out_w, h - out_h, crop_position)

            # Crop the source to the target aspect ratio before resizing.
            x = y = 0
            cw, ch = SW, SH
            if keep_proportion == "crop":
                new_aspect = w / h
                if SW / SH > new_aspect:
                    cw, ch = round(SH * new_aspect), SH
                else:
                    cw, ch = SW, round(SW / new_aspect)
                x, y = _crop_offsets(SW, SH, cw, ch, crop_position)

            # Resize the connected media to match the final dimensions.
            if image is not None:
                img = comfy.utils.common_upscale(image[:, y:y + ch, x:x + cw].movedim(-1, 1), out_w, out_h, upscale_method, "disabled").movedim(1, -1)
                if keep_proportion == "pad":
                    bg = _pad_color_tensor(pad_color, image.dtype, image.device)
                    img = _color_pad(img, pad_left, pad_right, pad_top, pad_bottom, bg)

            # Masks are [B,H,W]; bilinear keeps them smooth like core resize paths.
            if mask is not None:
                msk = comfy.utils.common_upscale(mask[:, y:y + ch, x:x + cw].unsqueeze(1), out_w, out_h, "bilinear", "disabled").squeeze(1)
                if keep_proportion == "pad":
                    # Replicate edge values into the padding like kijai's pad node.
                    msk = torch.nn.functional.pad(msk, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")

            fw = out_w + pad_left + pad_right
            fh = out_h + pad_top + pad_bottom
        else:
            fw, fh = w, h

        if is_flux2:
            samples = torch.zeros([batch_size, 128, fh // 16, fw // 16], device=comfy.model_management.intermediate_device())
            latent = {"samples": samples}
        else:
            samples = torch.zeros([batch_size, 4, fh // 8, fw // 8], device=comfy.model_management.intermediate_device(), dtype=comfy.model_management.intermediate_dtype())
            latent = {"samples": samples, "downscale_ratio_spacial": 8}

        # Update context with finalized dimensions and media. This node never
        # encodes: with an image present its stale latent is cleared, otherwise
        # the empty latent of the resulting size is stored. Without a context
        # input a new one is created, so the output is always a context with
        # width, height and the empty latent.
        ctx = context.copy() if isinstance(context, dict) else {}
        ctx["width"] = int(fw)
        ctx["height"] = int(fh)

        if image is not None:
            ctx["image"] = img
            ctx["latent"] = None
        else:
            ctx["latent"] = latent
        if mask is not None:
            ctx["mask"] = msk

        return io.NodeOutput(
            ctx,  # context (new context when none connected)
            latent, int(fw), int(fh), img if image is not None else None, msk if mask is not None else None
        )

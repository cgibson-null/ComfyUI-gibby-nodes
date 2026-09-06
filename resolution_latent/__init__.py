"""
Resize Image / Empty Latent (Context)
-------------------------------------
Creates an empty latent from a resolution spec instead of raw width/height.

Four modes (switched by the horizontal mode toggle in resolution_latent.js):
- keep_ar: resizes the linked media to the megapixels target at its own aspect
  ratio (native Scale Image to Total Pixels math); only offered when an image or
  mask is linked; all size widgets stay hidden except megapixels;
- custom: explicit width x height;
- aspect_ratio: core Resolution Selector presets + megapixels;
- custom_aspect_ratio: manual x : y floats + megapixels.

Shared across all modes: swap_dimensions, scale_factor (resolution multiplier),
multiple (round down to this multiple, advanced) and batch_size. The latent type
toggle picks between the standard 4ch /8x image latent and Flux2's 128ch /16x one.
Outputs LATENT plus the computed width/height for downstream sizing.

With an image or mask connected: the mode's target size becomes a box that the
media is fitted into using keep_proportion (Resize Image v2 semantics - stretch,
resize, pad with pad_color/crop_position, crop, total_pixels), and the resized
media plus a matching latent are output.
"""

import math

import torch
import comfy.model_management
import comfy.model_base
import comfy.utils
from nodes import VAEEncode, SetLatentNoiseMask
from comfy_api.latest import io

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
    vals = [float(v) for v in pad_color.split(",")]
    if len(vals) == 1:
        vals = vals * 3
    scale = 1 / 255.0 if max(vals[:3]) > 1 else 1.0
    return torch.tensor([v * scale for v in vals[:3]], dtype=dtype, device=device)


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
                "(manual w:h + megapixels). "
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
                io.Combo.Input("mode", options=["keep_ar", "custom", "aspect_ratio", "custom_aspect_ratio"], default="custom"),
                # custom mode: explicit width x height.
                io.Int.Input("width", default=1024, min=8, max=16384),
                io.Int.Input("height", default=1024, min=8, max=16384),
                # aspect ratio modes.
                io.Combo.Input("aspect_ratio", options=list(ASPECT_RATIOS), default="3:4 (Portrait Standard)"),
                # custom aspect ratio mode: manual w : h floats.
                io.Float.Input("x", default=3.0, min=0.1, max=99.0, step=0.01),
                io.Float.Input("y", default=4.0, min=0.1, max=99.0, step=0.01),
                io.Float.Input("megapixels", default=1.0, min=0.05, max=20.0, step=0.01),
                # Resolution multiplier applied to any mode's target size.
                io.Float.Input("scale_factor", default=1.0, min=0.05, max=16.0, step=0.01),
                # media resize (only used when an image or mask is connected).
                io.Combo.Input("upscale_method", options=["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], default="lanczos"),
                io.Combo.Input("keep_proportion", options=KEEP_PROPORTIONS, default="stretch", advanced=True),
                io.String.Input("pad_color", default="0, 0, 0", advanced=True),
                io.Combo.Input("crop_position", options=CROP_POSITIONS, default="center", advanced=True),
                # shared settings (plain booleans render as pill toggles).
                io.Boolean.Input("swap_dimensions", display_name="Swap dimensions", default=False),
                io.Int.Input("multiple", default=8, min=1, max=256, advanced=True),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                # latent format: on = Flux 2's 128ch /16x latent, off = standard 4ch /8x.
                io.Boolean.Input("flux2_latent", display_name="Flux 2 latent (16x)", default=False),
                # Optional VAE override: used to encode a connected image when no context is provided,
                # or to override the context's VAE.
                io.Vae.Input("vae", optional=True),
            ],
            outputs=[
                # Updated context with finalized dimensions/latent/image (None if no context in).
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
    def execute(cls, context=None, image=None, mask=None, mode="custom", width=1024, height=1024, aspect_ratio="3:4 (Portrait Standard)", x=3.0, y=4.0,
                megapixels=1.0, swap_dimensions=False, scale_factor=1.0,
                multiple=8, batch_size=1, flux2_latent=False, upscale_method="lanczos",
                keep_proportion="stretch", pad_color="0, 0, 0", crop_position="center", vae=None) -> io.NodeOutput:
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

        if mode == "custom" or (mode == "keep_ar" and not has_media):
            # keep_ar without media falls back to the (hidden) width/height fields.
            w, h = float(width), float(height)
        else:
            # Resolution Selector math at the selected ratio; keep_ar keeps the media's exact ratio.
            if mode == "keep_ar":
                rw, rh = float(SW), float(SH)
            elif mode == "aspect_ratio":
                rw, rh = ASPECT_RATIOS[aspect_ratio]
            else:
                rw, rh = float(x), float(y)
            scale_by = math.sqrt(megapixels * 1024 * 1024 / (rw * rh))
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

        # Resolve VAE: directly-connected overrides context
        effective_vae = vae
        if effective_vae is None and isinstance(context, dict):
            effective_vae = context.get("vae")

        # Update context with finalized dimensions and media.
        if isinstance(context, dict):
            ctx = context.copy()
            ctx["width"] = int(fw)
            ctx["height"] = int(fh)

            if image is not None:
                ctx["image"] = img
                if effective_vae is not None:
                    ctx["latent"], = VAEEncode().encode(effective_vae, img)
                    if mask is not None:
                        ctx["mask"] = msk
                        ctx["latent"], = SetLatentNoiseMask().set_mask(ctx["latent"], msk)
            else:
                ctx["latent"] = latent
        elif image is not None and effective_vae is not None:
            # No context, but image + VAE: encode to latent
            latent, = VAEEncode().encode(effective_vae, img)
            if mask is not None:
                latent, = SetLatentNoiseMask().set_mask(latent, msk)

        return io.NodeOutput(
            ctx if isinstance(context, dict) else None,  # context (None when not connected)
            latent, int(fw), int(fh), img if image is not None else None, msk if mask is not None else None
        )

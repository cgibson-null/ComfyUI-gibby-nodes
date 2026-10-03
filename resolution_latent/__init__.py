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
Outputs the context, empty_latent, encoded_latent and the computed width/height
for downstream sizing.

With an image or mask connected: the mode's target size becomes a box that the
media is fitted into using keep_proportion (Resize Image v2 semantics - stretch,
resize, pad with pad_color/crop_position, pad_around (keeps the original size
centered and pads the box around it, like Pad Image for Outpainting), crop,
total_pixels), and the resized media plus a matching latent are output.

Optional upscale_model (Load Upscale Model): when connected and an image is
linked, the image is first upscaled with it (Upscale Image (using Model))
before the resize. The size spec follows the original image's size, so the
upscaled result is fitted back into the target box (scale factor, not the
model's factor, sets the final size).

Load image (Load Image node functionality): loads a file from the input folder
by path (upload + preview + mask drawing on the preview, like the core Load
Image - the drawn mask is the file's alpha). The loaded image replaces the
connected one, and the file's alpha replaces the connected or context mask.

With an image or a context's image connected, the node shows the output
image's temp preview in the node on execution (like the core Preview
Image). The enable preview toggle (off by default) controls it: off, no
temp file is written at all; it is always written while Load image or
Crop image is on (the toggle is hidden then), because downstream crop
views read the context's image from it. A mask drawn on that preview is
used the same way: the core mask editor rewrites the file combo to its
clipspace-painted-masked-*.png upload, and the file's alpha becomes the
mask - replacing the connected one, resized to the output size. Crop
image's box mask still wins over it.
Crop image is independent of it: it crops the image the resize uses (the
loaded file with Load image on, else the connected or context image) to the
drawn box (the crop preview's AR follows the size mode when set to selected),
the box mask becomes the mask, and the crop info (the original image, the box
mask and the box as the paste rect) is output and stored for pasting back
with Image Paste By Mask (Batch) (Context).
"""

import math
import os

import numpy as np
import torch
import folder_paths
import comfy.model_management
import comfy.utils
from PIL import Image, ImageOps, ImageSequence
from comfy_api.latest import io, ui
from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
from comfy_extras.color_util import hex_to_rgb

from ..context import _CONTEXT_TYPE, _CROP_INFO_TYPE, _is_flux2, ctx_from, ctx_set_image, empty_latent, encode_image


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
# Picks the preset closest to the connected media's ratio; 3:4 without media.
CLOSEST_RATIO = "closest to image"

KEEP_PROPORTIONS = ["stretch", "resize", "pad", "pad_around", "crop", "total_pixels"]
CROP_POSITIONS = ["center", "top", "bottom", "left", "right"]


def _resize_to_mp_scale(w, h, megapixels, scale_factor, multiple):
    """Target size from a source size: exact megapixels at the source's aspect
    ratio (0 = own size), then scale_factor, then floored to a multiple."""
    if megapixels > 0:
        scale_by = math.sqrt(megapixels * 1024 * 1024 / (w * h))
        w *= scale_by
        h *= scale_by
    w *= scale_factor
    h *= scale_factor
    return max(multiple, int(w) // multiple * multiple), max(multiple, int(h) // multiple * multiple)


def _resize_image(image, w, h, method="bilinear"):
    """Resize a channels-last image (4D still or 5D frame batch) to an exact (w, h)."""
    return comfy.utils.common_upscale(image.movedim(-1, 1), w, h, method, "disabled").movedim(1, -1)


def _resize_image_to_mp(image, megapixels, scale_factor, multiple, method="bilinear"):
    """Resize a channels-last image to the megapixels/scale target floored to a multiple; unchanged when already at it."""
    w, h = image.shape[2], image.shape[1]
    tw, th = _resize_to_mp_scale(w, h, megapixels, scale_factor, multiple)
    if (tw, th) != (w, h):
        image = _resize_image(image, tw, th, method)
    return image


def _resize_mask(mask, w, h, mode="bilinear"):
    """Resize a (B,H,W) or (B,C,H,W) mask to an exact (w, h); nearest modes keep hard edges."""
    import torch.nn.functional as F
    m = mask.unsqueeze(1) if mask.dim() == 3 else mask
    if mode in ("nearest", "nearest-exact"):
        m = F.interpolate(m, size=(h, w), mode=mode)
    else:
        m = F.interpolate(m, size=(h, w), mode=mode, align_corners=False)
    return m.squeeze(1) if mask.dim() == 3 else m


def _mask_bbox(mask):
    """(x, y, w, h) of a single mask's non-zero area, or None when it is empty."""
    m = mask.squeeze()
    if m.dim() > 2:
        m = m[0, 0]
    rows = (m > 0.001).any(dim=1)
    cols = (m > 0.001).any(dim=0)
    if not rows.any() or not cols.any():
        return None
    top = rows.nonzero().squeeze(-1)[0].item()
    bottom = rows.nonzero().squeeze(-1)[-1].item()
    left = cols.nonzero().squeeze(-1)[0].item()
    right = cols.nonzero().squeeze(-1)[-1].item()
    return (left, top, right - left + 1, bottom - top + 1)


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
    if keep_proportion == "pad_around":
        # Keep the original size (never upscale; downscale only when the box
        # is smaller), unsnapped - the content keeps its exact pixels and the
        # snapped box sets the canvas it is padded into.
        ratio = min(1.0, box_w / sw, box_h / sh)
        return max(1, int(round(sw * ratio))), max(1, int(round(sh * ratio)))
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
    # The core Color Picker's #RRGGBB / #RRGGBBAA, or "R, G, B[, A]" 0-255 as
    # the frontend's color widget stores it.
    if pad_color.startswith("#"):
        if len(pad_color) not in (7, 9):
            raise ValueError("Color must be in format #RRGGBB or R, G, B")
        rgb = hex_to_rgb(pad_color[:7])
    else:
        try:
            rgb = [float(v) for v in pad_color.split(",")]
        except ValueError:
            raise ValueError("Color must be in format #RRGGBB or R, G, B")
        if not 3 <= len(rgb) <= 4:
            raise ValueError("Color must be in format #RRGGBB or R, G, B")
        rgb = rgb[:3]
    return torch.tensor([v / 255.0 for v in rgb], dtype=dtype, device=device)


def _color_alpha(pad_color):
    # The color's alpha channel (1.0 when absent): #RRGGBBAA or "R, G, B[, A]" 0-255
    if pad_color.startswith("#"):
        return int(pad_color[7:9], 16) / 255.0 if len(pad_color) == 9 else 1.0
    parts = pad_color.split(",")
    return float(parts[3]) / 255.0 if len(parts) >= 4 else 1.0


def _color_pad(img, left, right, top, bottom, bg):
    # img [B,H,W,C]; canvas filled with the background color, content at (top, left).
    B, H, W, C = img.shape
    out = torch.empty((B, H + top + bottom, W + left + right, C), dtype=img.dtype, device=img.device)
    for c in range(C):
        out[:, :, :, c] = bg[c % 3]
    out[:, top:top + H, left:left + W] = img
    return out


def _load_image_file(image_path):
    """(image, mask) from an input folder file like the core Load Image:
    channels-last float 0-1 frames, and 1.0 over the file's transparent areas
    (None when it has no alpha or the alpha is fully opaque)."""
    img = Image.open(image_path)
    frames, masks = [], []
    for i in ImageSequence.Iterator(img):
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        if frames and image.size != (frames[-1].shape[2], frames[-1].shape[1]):
            continue
        frames.append(torch.from_numpy(np.array(image).astype(np.float32) / 255.0)[None])
        if "A" in i.getbands():
            masks.append(1.0 - torch.from_numpy(np.array(i.getchannel("A")).astype(np.float32) / 255.0)[None])
        else:
            masks.append(None)
    if not frames:
        raise ValueError(f"Invalid image file: {image_path}")
    image = torch.cat(frames)
    mask = None
    if any(m is not None for m in masks):
        mask = torch.cat([m if m is not None else torch.zeros(1, image.shape[1], image.shape[2]) for m in masks])
        if not bool(mask.any()):
            mask = None
    device = comfy.model_management.intermediate_device()
    dtype = comfy.model_management.intermediate_dtype()
    return image.to(device=device, dtype=dtype), mask.to(device=device, dtype=dtype) if mask is not None else None


def _load_media(image):
    """(media, created mask) for the Load image toggle: the file is loaded
    like the core Load Image and its alpha is the created mask."""
    image_path = folder_paths.get_annotated_filepath(image)
    if not os.path.isfile(image_path):
        raise ValueError(f"Invalid image file: {image}")
    return _load_image_file(image_path)


def _is_mask_editor_file(image):
    """The core mask editor's upload: the file it rewrites the node's file
    combo to when a mask is drawn on the node's preview."""
    if not isinstance(image, str) or not image:
        return False
    name, _ = folder_paths.annotated_filepath(image)
    return os.path.basename(name).startswith("clipspace-painted-masked-")


def _drawn_mask(image):
    """The mask drawn on the node's preview: the core mask editor rewrites the
    file combo to its clipspace file, whose alpha is the drawn mask. None when
    the combo holds no such file, the file is missing, or it has no usable
    alpha (an untouched preview)."""
    if not _is_mask_editor_file(image):
        return None
    image_path = folder_paths.get_annotated_filepath(image)
    if not os.path.isfile(image_path):
        return None
    return _load_image_file(image_path)[1]


def _crop_media(img, crop_region):
    """(crop, box mask, crop info) for the Crop image toggle: the image cut to
    the drawn box, the box area in the full image (1.0 in the box, 0.0
    outside, like the Crop Image node's mask) and the originals for pasting
    back with Image Paste By Mask (Batch) (Context)."""
    B, H, W, _ = img.shape
    region = crop_region or {}
    x1 = int(min(max(region.get("x", 0), 0), W - 1))
    y1 = int(min(max(region.get("y", 0), 0), H - 1))
    x2 = int(min(max(region.get("x", 0) + region.get("width", 512), 1), W))
    y2 = int(min(max(region.get("y", 0) + region.get("height", 512), 1), H))
    # Mask: 1.0 in the crop area (foreground), 0.0 outside (background)
    box_mask = torch.zeros(B, H, W, dtype=img.dtype, device=img.device)
    box_mask[:, y1:y2, x1:x2] = 1.0
    rect = (x1, y1, x2 - x1, y2 - y1)
    return img[:, y1:y2, x1:x2, :], box_mask, {"image": img, "mask": box_mask, "rects": [rect] * B}


class GibbyEmptyLatentResolution(io.ComfyNode):
    """Create an empty latent from a resolution spec instead of raw width/height."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        # The input folder's image files, like the core Load Image's combo.
        input_dir = folder_paths.get_input_directory()
        files = sorted(folder_paths.filter_files_content_types(
            [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))], ["image"]))
        return io.Schema(
            node_id="Gibby_EmptyLatent_Resolution",
            display_name="Resize Image / Empty Latent (Context)",
            category="gibby",
            # output node: runs (and refreshes the preview) even with nothing connected
            is_output_node=True,
            description=(
                "Creates an empty latent from a resolution spec instead of raw width/height. "
                "Modes: keep AR (megapixels at the media's aspect ratio, like Scale Image to Total Pixels), "
                "custom (width x height), aspect ratio (preset, or closest to the image, + megapixels) "
                "or custom aspect ratio (manual w:h + megapixels). Megapixels 0 rescales the media's own size instead, or "
                "defaults to 1 MP with no media linked. "
                "Shared: swap dimensions, scale factor, multiple and batch size; latent type toggle for "
                "standard 8x vs Flux2 16x latents. With an image or mask connected (the context's when "
                "nothing is) it is fitted into the target box per keep_proportion (stretch/resize/pad/pad_around/"
                "crop/total_pixels; pad_around keeps the original size centered and pads the box around it like "
                "Pad Image for Outpainting) and output resized. "
                "Load image loads a file from the input folder (upload, preview and mask drawing like the core "
                "Load Image) instead of the connected image; its alpha replaces the connected or context mask when non-empty. "
                "Crop image is independent of it: it crops the image the resize uses (the loaded file with Load image on, "
                "else the connected or context image) to the drawn box, the box mask becomes the mask, and the crop info "
                "is output and stored for pasting back with Image Paste By Mask (Batch) (Context). "
                "Enable preview (off by default, hidden while Load image or Crop image is on) writes and shows the "
                "output image's temp preview in the node on execution (like the core Preview Image); off writes no "
                "preview file at all, and while Load image or Crop image is on the preview is always written - "
                "downstream crop views read the context's image from it. "
                "An optional vae overrides the context's vae; encoded_latent outputs the resized image "
                "encoded with it (the context's vae when unconnected), or empty_latent without an image or vae. "
                "mask_padded marks the pad border (1 = padding, 0 elsewhere). The image output is black and the "
                "mask outputs are empty (all 0) when there is no media or mask, so downstream previews never fail."
            ),
            inputs=[
                # Optional context: width/height overridden with finalized size; latent or image+latent updated.
                _CONTEXT_TYPE.Input("context", optional=True),
                # Optional: when linked, it is resized per keep_proportion and the size follows.
                io.Image.Input("connected_image", optional=True),
                io.Mask.Input("mask", optional=True),
                # Mode switcher (rendered as a horizontal toggle by resolution_latent.js).
                io.Combo.Input("mode", options=["keep_ar", "custom", "aspect_ratio", "custom_aspect_ratio"], default="aspect_ratio"),
                # custom mode: explicit width x height.
                io.Int.Input("width", default=1024, min=8, max=16384),
                io.Int.Input("height", default=1024, min=8, max=16384),
                # aspect ratio modes.
                io.Combo.Input("aspect_ratio", options=list(ASPECT_RATIOS) + [CLOSEST_RATIO], default="3:4 (Portrait Standard)"),
                # custom aspect ratio mode: manual w : h floats.
                io.Float.Input("x", default=3.0, min=0.1, max=99.0, step=0.01),
                io.Float.Input("y", default=4.0, min=0.1, max=99.0, step=0.01),
                io.Float.Input("megapixels", default=1.0, min=0.0, max=20.0, step=0.01),
                # Resolution multiplier applied to any mode's target size.
                io.Float.Input("scale_factor", default=1.0, min=0.05, max=16.0, step=0.01),
                # media resize (only used when an image or mask is connected).
                io.Combo.Input("upscale_method", options=["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], default="lanczos"),
                io.Combo.Input("keep_proportion", options=KEEP_PROPORTIONS, default="stretch", advanced=True),
                io.Color.Input("pad_color", default="#000000", advanced=True,
                               tooltip="Pad border color for the pad fit mode; alpha 0 makes the border transparent instead (RGBA output)"),
                io.Combo.Input("crop_position", options=CROP_POSITIONS, default="center", advanced=True),
                # shared settings (plain booleans render as pill toggles).
                io.Boolean.Input("swap_dimensions", display_name="Swap dimensions", default=False),
                io.Int.Input("multiple", default=8, min=1, max=256, advanced=True),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                # latent format: on = Flux 2's 128ch /16x latent, off = standard 4ch /8x.
                io.Boolean.Input("flux2_latent", display_name="Flux 2 latent (16x)", default=False),
                # The executed temp preview in the node (like the core Preview Image):
                # off by default - off writes no temp file at all. Hidden by the
                # frontend while load or crop is on, where the preview is always
                # written (downstream crop views read the context's image from it).
                io.Boolean.Input("enable_preview", display_name="Enable preview", default=False, optional=True,
                                  tooltip="Write and show the output image's temp preview in the node on execution (like the core Preview Image); off writes no preview file at all. Hidden while Load image or Crop image is on, where the preview is always written - downstream crop views read the context's image from it"),
                # Load image: the core Load Image's path combo (upload, preview, mask
                # drawing) replaces the connected image with the file's. All five are
                # optional: a prompt without them (pre-feature workflows, API users)
                # runs exactly like before the feature.
                io.Boolean.Input("load_image", display_name="Load image", default=False, optional=True,
                                 tooltip="Load an image from the input folder instead of the connected one; its alpha (the mask drawn on the preview) replaces the connected or context mask when non-empty"),
                io.Combo.Input("image", options=files, default=files[0] if files else "", upload=io.UploadType.image, optional=True,
                               tooltip="The file in the input folder to load (upload a file, or draw a mask on the preview - it is saved as the file's alpha). With Load image off it is where a mask drawn on the node's executed preview lands (the core mask editor), and its alpha is used as the mask"),
                io.Boolean.Input("crop_image", display_name="Crop image", default=False, optional=True,
                                 tooltip="Crop the image the resize uses (the loaded file with Load image on, else the connected or context image) to the drawn box before the resize; the box mask becomes the mask and the crop info is output and stored in the context"),
                io.Combo.Input("crop_ar", options=["custom", "selected"], default="custom", optional=True,
                               tooltip="The crop box's aspect ratio: custom is a free box, selected follows the size mode (the image's AR for keep AR, width x height for custom, the preset or manual ratio for the AR modes)"),
                io.BoundingBox.Input("crop_region", default={"x": 0, "y": 0, "width": 512, "height": 512}, optional=True,
                                     tooltip="The crop box in the image's pixels (drawn on the preview)"),
                # Optional Load Upscale Model: when linked and an image is present, upscale it
                # (Upscale Image (using Model)) before the resize.
                io.UpscaleModel.Input("upscale_model", optional=True),
                # Optional: overrides the context's vae; with an image present it encodes it
                # for the encoded_latent output (the context's vae is used when unconnected).
                io.Vae.Input("vae", optional=True),
            ],
            outputs=[
                # Updated context with finalized dimensions/latent/image (new context when none in).
                _CONTEXT_TYPE.Output(display_name="context"),
                io.Latent.Output(display_name="empty_latent"),
                # The resized image encoded with the vae when both are present, else empty_latent.
                io.Latent.Output(display_name="encoded_latent"),
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
                # Resized input media (black when not connected).
                io.Image.Output(display_name="image"),
                # Empty (all 0) when there is no mask.
                io.Mask.Output(display_name="mask"),
                # 1 where the pad border was filled in, 0 elsewhere.
                io.Mask.Output(display_name="mask_padded"),
                # Crop image on only: for Image Paste By Mask (Batch) (Context) - the loaded
                # image, the box mask and the box as the paste rect; also stored in the context.
                _CROP_INFO_TYPE.Output(display_name="crop_info"),
            ],
        )

    @classmethod
    def validate_inputs(cls, load_image=False, image=None):
        # The file is only read while the load image toggle is on, so a
        # prompt without it (or with a non-file value) is no failure then.
        # Like the core Load Image: the file list is a registration-time
        # snapshot, so check the disk instead (clip-space mask files and
        # fresh uploads are not in the list).
        if not load_image:
            return True
        # Like _load_media: an empty value resolves to the input dir itself,
        # so exists_annotated_filepath alone would pass it.
        if not isinstance(image, str) or not image or not os.path.isfile(folder_paths.get_annotated_filepath(image)):
            return "Invalid image file: {}".format(image)

        return True

    @classmethod
    def execute(cls, context=None, connected_image=None, mask=None, enable_preview=False,
                load_image=False, image=None,
                crop_image=False, crop_ar="custom", crop_region=None, upscale_model=None, mode="aspect_ratio", width=1024, height=1024, aspect_ratio="3:4 (Portrait Standard)", x=3.0, y=4.0,
                megapixels=1.0, swap_dimensions=False, scale_factor=1.0,
                multiple=8, batch_size=1, flux2_latent=False, upscale_method="lanczos",
                keep_proportion="stretch", pad_color="#000000", crop_position="center", vae=None) -> io.NodeOutput:
        # Load image: the file's media replaces the connected one, and the
        # file's alpha wins over the connected mask. Without it, a mask drawn
        # on the node's preview wins the same way (the core mask editor
        # rewrites the file combo to its clipspace file).
        crop_mask = None
        crop_info = None
        drawn_mask = None
        if load_image:
            connected_image, mask = _load_media(image)
        else:
            drawn_mask = _drawn_mask(image)
            # The context's mask, like its image: the fallback when none is connected
            if mask is None and isinstance(context, dict):
                mask = context.get("mask")

        # If no input image but context has one, use context's image
        if connected_image is None and isinstance(context, dict):
            connected_image = context.get("image")

        # Crop image: the box in the image the resize uses (loaded > connected
        # > context); the box mask replaces the mask and the crop info holds
        # the originals for pasting back.
        if crop_image and connected_image is not None:
            connected_image, crop_mask, crop_info = _crop_media(connected_image, crop_region)
            mask = None

        # The connected vae overrides the context's.
        if vae is None and isinstance(context, dict):
            vae = context.get("vae")

        # Original media W/H before any model upscale; the size spec follows
        # this, and the upscaled image is fitted into the target box.
        OW = OH = 0
        if connected_image is not None:
            OW, OH = connected_image.shape[2], connected_image.shape[1]
        elif mask is not None:
            OW, OH = mask.shape[2], mask.shape[1]

        # Optional model upscale (Load Upscale Model + Upscale Image (using Model)),
        # applied before the resize so the upscaled image is what gets fitted.
        if upscale_model is not None and connected_image is not None:
            connected_image, = ImageUpscaleWithModel.execute(upscale_model, connected_image)

        has_media = connected_image is not None or mask is not None
        # Source W/H; same shape indices for [B,H,W,C] images and [B,H,W] masks.
        SW = SH = 0
        if has_media:
            src = connected_image if connected_image is not None else mask
            SW, SH = src.shape[2], src.shape[1]

        # The context's model decides the latent format when present, else the toggle
        model_obj = context.get("model") if isinstance(context, dict) else None
        is_flux2 = _is_flux2(model_obj) if model_obj is not None else flux2_latent

        # Latent grids must be divisible by 8 (image) or 16 (flux2), so round down to the lcm of multiple and that base.
        base = 16 if is_flux2 else 8
        step = multiple * base // math.gcd(multiple, base)

        if mode == "custom":
            w, h, mp = float(width), float(height), 0.0
        elif megapixels == 0 and has_media:
            # 0 megapixels: no target pixel count, rescale the original media's size
            # (before upscale_model), so the upscaled image is fitted back to it.
            w, h, mp = float(OW), float(OH), 0.0
        else:
            # Resolution Selector math at the selected ratio; keep_ar uses media's ratio or falls back to aspect_ratio.
            if mode == "keep_ar" and has_media:
                rw, rh = float(SW), float(SH)
            elif mode == "aspect_ratio" or (mode == "keep_ar" and not has_media):
                if aspect_ratio == CLOSEST_RATIO:
                    # Closest preset to the media's own ratio; 3:4 without media.
                    if has_media:
                        rw, rh = min(ASPECT_RATIOS.values(), key=lambda r: abs(r[0] / r[1] - SW / SH))
                    else:
                        rw, rh = ASPECT_RATIOS["3:4 (Portrait Standard)"]
                else:
                    rw, rh = ASPECT_RATIOS[aspect_ratio]
            else:
                rw, rh = float(x), float(y)
            # 0 megapixels without media defaults to 1 MP.
            w, h, mp = rw, rh, megapixels or 1.0

        w, h = _resize_to_mp_scale(w, h, mp, scale_factor, step)
        if swap_dimensions:
            w, h = h, w

        # Media mode: fit the source into the (w x h) box per keep_proportion.
        if has_media:
            out_w, out_h = _fit_size(SW, SH, w, h, keep_proportion, step)
            # pad and pad_around fill the box around the content with pad_color.
            is_pad = keep_proportion in ("pad", "pad_around")
            pad_left, pad_right, pad_top, pad_bottom = 0, 0, 0, 0
            if is_pad:
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
            if connected_image is not None:
                img = _resize_image(connected_image[:, y:y + ch, x:x + cw], out_w, out_h, upscale_method)
                if is_pad:
                    bg = _pad_color_tensor(pad_color, connected_image.dtype, connected_image.device)
                    if _color_alpha(pad_color) == 0.0:
                        # Transparent pad: the content is opaque, the border is alpha 0
                        img = _color_pad(img[..., :3], pad_left, pad_right, pad_top, pad_bottom, bg)
                        alpha = torch.ones(img.shape[0], out_h, out_w, dtype=img.dtype, device=img.device)
                        alpha = torch.nn.functional.pad(alpha, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0)
                        img = torch.cat((img, alpha.unsqueeze(-1)), dim=-1)
                    else:
                        img = _color_pad(img, pad_left, pad_right, pad_top, pad_bottom, bg)

            # Masks are [B,H,W]; bilinear keeps them smooth like core resize paths.
            if mask is not None:
                msk = comfy.utils.common_upscale(mask[:, y:y + ch, x:x + cw].unsqueeze(1), out_w, out_h, "bilinear", "disabled").squeeze(1)
                if is_pad:
                    # Replicate edge values into the padding like kijai's pad node.
                    msk = torch.nn.functional.pad(msk, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")

            fw = out_w + pad_left + pad_right
            fh = out_h + pad_top + pad_bottom
            # 1 where the pad border was filled in, 0 over the content; empty for the other fit modes
            ref = img if connected_image is not None else msk
            mask_padded = torch.zeros(ref.shape[0], fh, fw, dtype=torch.float32, device=ref.device)
            if is_pad:
                mask_padded[:, :pad_top, :] = 1
                mask_padded[:, pad_top + out_h:, :] = 1
                mask_padded[:, :, :pad_left] = 1
                mask_padded[:, :, pad_left + out_w:] = 1
        else:
            fw, fh = w, h
            mask_padded = torch.zeros(batch_size, fh, fw, dtype=torch.float32, device=comfy.model_management.intermediate_device())

        # The image/mask outputs are always tensors: a None output would crash
        # the previews downstream of them and abort the run. Without a
        # connected image the image is black; without a mask the mask is empty
        # (0 = no masked area).
        if connected_image is None:
            img = torch.zeros(batch_size, fh, fw, 3, dtype=comfy.model_management.intermediate_dtype(), device=comfy.model_management.intermediate_device())

        # The drawn mask lives in the preview's space (the last run's output
        # size), so resize it straight to the final size - not through the
        # source-space crop the connected mask gets.
        if drawn_mask is not None:
            drawn_mask = _resize_mask(drawn_mask, int(fw), int(fh))

        latent = empty_latent(fw, fh, batch_size=batch_size, flux2=is_flux2)

        # encoded_latent: the resized image encoded with the vae when both are
        # present, otherwise the empty latent.
        encoded_latent = latent
        if connected_image is not None and vae is not None:
            encoded_latent = encode_image(vae, img)

        # Update context with finalized dimensions and media. The context keeps
        # its own latent policy: with an image present its stale latent is
        # cleared (the sampler encodes it, optionally tiled), otherwise the
        # empty latent of the resulting size is stored. Without a context
        # input a new one is created, so the output is always a context with
        # width, height and the empty latent.
        ctx = ctx_from(context)
        ctx["width"] = int(fw)
        ctx["height"] = int(fh)
        if vae is not None:
            ctx["vae"] = vae

        if connected_image is not None:
            ctx_set_image(ctx, img)
        else:
            ctx["latent"] = latent
        if crop_mask is not None:
            # The box mask is in the original image's space, but the context's
            # image is the resized crop, so the two don't correspond - the
            # context carries no mask (crop_info keeps it for pasting back).
            ctx.pop("mask", None)
        elif drawn_mask is not None:
            ctx["mask"] = drawn_mask
        elif mask is not None:
            ctx["mask"] = msk
        if crop_info is not None:
            ctx["crop_info"] = crop_info

        out_mask = crop_mask if crop_mask is not None else (
            drawn_mask if drawn_mask is not None else (msk if mask is not None else None))
        if out_mask is None:
            out_mask = torch.zeros(batch_size, fh, fw, dtype=torch.float32, device=comfy.model_management.intermediate_device())

        # The temp preview of the output image is what downstream crop views
        # show for the context's image - a runtime tensor with no path of
        # its own, so without it they would fall back to the loaded file.
        # Only the first image of a batch is previewed. It is written while
        # load or crop is on (the toggle is hidden then) and otherwise only
        # with the toggle on - off writes no temp file at all.
        preview = ui.PreviewImage(img[:1], cls=cls) if (enable_preview or load_image or crop_image) else None
        return io.NodeOutput(
            ctx,  # context (new context when none connected)
            latent, encoded_latent, int(fw), int(fh), img, out_mask,
            mask_padded, crop_info,
            ui=preview
        )

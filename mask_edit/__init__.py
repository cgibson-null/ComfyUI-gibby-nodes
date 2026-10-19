"""
Mask Edit node
--------------
The mask manipulation the detection and crop nodes used to carry, in one node:
grow and invert (the native Grow Mask / Invert Mask), an edge feather (the
gaussian the native Blur Image applies, on the mask), isolated-pixel removal and
hole filling (the Essentials Mask Fix opening and the hole closing KJ's blur
grow uses), and background removal, which replaces the unmasked area with
rembg_color (alpha 0 makes it transparent instead).

It also owns the tinted preview: the mask color blended over the image (KJNodes
ImageAndMaskPreview style); a batch of frames previews as a video at
preview_fps, blended and encoded one frame at a time the way the native
SAM3 Track Preview does it (a display-only artifact, so it skips the Save Video
encoder settings for a fast x264 preset).

Chain it after Mask/Segment (Context) to clean a detection up before cropping by
it, or after Image Crop By Mask (Batch) (Context) to knock the background out of
the crop. The edited image and mask replace the context's. A crop info that
describes the image being edited is refreshed to the edited mask, so a paste back
honors the edit; a crop's own - whose mask is the full frame the crop came from -
does not match and passes through untouched, so Image Paste By Mask (Batch)
(Context) still pastes onto the frame the crop was taken from.
"""

import os
import random
from fractions import Fraction

import av
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi
import folder_paths
import nodes
from comfy_api.latest import io, ui
from comfy_extras.color_util import hex_to_rgb
from comfy_extras.nodes_mask import GrowMask, InvertMask
from comfy_extras.nodes_post_processing import gaussian_kernel

from ..context import _CONTEXT_TYPE, ctx_from, ctx_set_image
from ..crop_paste.crop_image_by_mask import _process_mask
from ..resolution_latent import _mask_bbox, _resize_mask, _pad_color_tensor, _color_alpha


def _fix_mask(mask, remove_isolated_pixels, fill_holes):
    # remove_isolated_pixels: morphological opening (essentials Mask Fix)
    # fill_holes: close the holes the mask encloses (KJ Grow Mask With Blur)
    if not remove_isolated_pixels and not fill_holes:
        return mask
    out = []
    for m in mask:
        arr = m.cpu().numpy().astype(np.float32)
        if remove_isolated_pixels:
            arr = ndi.grey_opening(arr, size=(remove_isolated_pixels, remove_isolated_pixels))
        if fill_holes:
            arr = ndi.binary_fill_holes(arr > 0.5).astype(np.float32)
        out.append(torch.from_numpy(arr))
    return torch.stack(out).to(mask.device)


def _feather_mask(mask, mask_blur):
    # The mask edge ramped over mask_blur pixels: the gaussian kernel the native
    # Blur Image uses, convolved on the mask's own device so the mask stays where
    # it was. sigma is radius / 3 rather than the node's fixed 1.0, so a wider
    # radius is a wider ramp instead of the same blur.
    if not mask_blur:
        return mask
    k = mask_blur * 2 + 1
    kernel = gaussian_kernel(k, mask_blur / 3.0, device=mask.device, dtype=mask.dtype).reshape(1, 1, k, k)
    return F.conv2d(F.pad(mask.unsqueeze(1), (mask_blur,) * 4, "reflect"), kernel).squeeze(1)


def _mask_color(color):
    # KJNodes ImageAndMaskPreview style: the mask color's hex alpha is the opacity
    if len(color) not in (7, 9) or color[0] != "#":
        raise ValueError("mask_color must be in format #RRGGBB or #RRGGBBAA")
    r, g, b = (c / 255 for c in hex_to_rgb(color[:7]))
    alpha = 1.0 if len(color) == 7 else int(color[7:9], 16) / 255.0
    return (r, g, b), alpha


def _overlay(image, mask, color):
    # The mask color blended over the image where the mask is set
    (r, g, b), alpha = _mask_color(color)
    tint = torch.tensor([r, g, b], dtype=image.dtype, device=image.device)
    out = image.clone()
    out.lerp_(tint, (mask * alpha).unsqueeze(-1).to(image.device))
    return out


def _preview_video(images, mask, fps, color):
    # The SAM3 Track Preview path: blend and encode one frame at a time into a
    # temp mp4 and hand the file to the node preview, instead of building the
    # tinted batch and running it through Create Video -> Save Video. The
    # preview is display-only, so it encodes with the fast x264 preset instead
    # of Save Video's crf 18 / medium.
    (r, g, b), alpha = _mask_color(color)
    tint = torch.tensor([r, g, b], dtype=images.dtype, device=images.device)
    mask = mask.to(images.device)
    H, W = images.shape[1:3]
    frame = torch.empty(H, W, 3, dtype=torch.uint8)
    frame_np = frame.numpy()

    prefix = "ComfyUI_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for _ in range(5))
    full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
        prefix, folder_paths.get_temp_directory(), images.shape[3], images.shape[2])
    file = f"{filename}_{counter:05}_.mp4"
    with av.open(os.path.join(full_output_folder, file), mode="w", format="mp4") as output:
        stream = output.add_stream("h264", rate=Fraction(round(fps * 1000), 1000))
        stream.width, stream.height, stream.pix_fmt = W, H, "yuv420p"
        stream.options = {"crf": "23", "preset": "veryfast"}
        for image, m in zip(images, mask):
            image = image.clone()
            image.lerp_(tint, (m * alpha).unsqueeze(-1))
            frame.copy_(image.clamp_(0, 1).mul_(255).byte())
            vframe = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
            output.mux(stream.encode(vframe.reformat(format="yuv420p")))
        output.mux(stream.encode(None))
    return ui.PreviewVideo([ui.SavedResult(file, subfolder, io.FolderType.temp)])


class GibbyMaskEdit(io.ComfyNode):
    """Edit the mask (clean up, feather, grow, invert), knock the background out
    of the image with it, and preview the mask tinted over the image."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Gibby_Mask_Edit",
            display_name="Mask Edit (Context)",
            category="gibby/image",
            # output node: runs (and refreshes the preview) even with nothing connected
            is_output_node=True,
            search_aliases=["mask", "grow", "invert", "blur", "feather", "background", "preview"],
            description=(
                "Edits the mask (connected mask wins over the context one) and stores the "
                "resulting image and mask back in the context; the context's model and clip "
                "pass through untouched. A crop info that describes the image being edited is "
                "refreshed to the edited mask, so a paste back honors the edit, while a crop's "
                "own passes through untouched. The mask is cleaned first "
                "(remove_isolated_pixels drops isolated specks, fill_holes closes the holes the "
                "mask encloses), then its edge is feathered by mask_blur pixels and it is grown "
                "and inverted, so growing enlarges what is left instead of the specks. remove_bg "
                "replaces the unmasked area with rembg_color "
                "(an alpha 0 color makes it transparent instead). With preview on, the node "
                "displays the image with the mask tinted in the mask color. Use it after "
                "Mask/Segment (Context) to clean a detection up before cropping by it, or after "
                "Image Crop By Mask (Batch) (Context) to knock the background out of the crop."
            ),
            inputs=[
                _CONTEXT_TYPE.Input("context", optional=True,
                                    tooltip="Base context; its image and mask are used when none are connected"),
                io.Image.Input("image", optional=True, tooltip="Overrides the context image"),
                io.Mask.Input("mask", optional=True, tooltip="Overrides the context mask"),

                # Every input is optional: the frontend renders the required block
                # before the optional one, so a single optional block is the only way
                # the widgets below keep this declaration order (execute's defaults
                # match each widget's default for the case an input is omitted).
                io.Int.Input("remove_isolated_pixels", optional=True, default=0, min=0, step=1,
                             tooltip="Opening kernel that drops isolated mask specks (0 = off); runs first"),
                io.Boolean.Input("fill_holes", optional=True, default=False, tooltip="Close the holes the mask encloses"),
                io.Int.Input("mask_blur", optional=True, default=0, min=0, max=63, step=1,
                             tooltip="Feathers the mask edge over this many pixels (0 = off), the gaussian kernel of the native Blur Image at sigma = radius / 3; runs after the cleanup so the ramp follows the cleaned shape"),
                io.Int.Input("mask_grow", optional=True, default=0, min=-nodes.MAX_RESOLUTION, max=nodes.MAX_RESOLUTION, step=1,
                             tooltip="Grows the mask by this many pixels (negative shrinks it), like the native Grow Mask with tapered corners"),
                io.Boolean.Input("mask_invert", optional=True, default=False,
                                 tooltip="Inverts the mask after growing, like the native Invert Mask"),

                io.Boolean.Input("remove_bg", optional=True, default=False,
                                 tooltip="Replaces the unmasked area with rembg_color"),
                io.Color.Input("rembg_color", optional=True, socketless=False, default="#000000",
                               tooltip="Background color; connect the hex output of a Color Picker. Alpha 0 makes the background transparent instead (RGBA output)"),

                io.Boolean.Input("preview", optional=True, default=True, tooltip="Show the mask tinted in the mask color over the image (video if the input is a batch of frames)"),
                io.Float.Input("preview_fps", optional=True, default=30.0, min=1.0, max=120.0, step=1.0,
                               tooltip="Frame rate of the preview when the input is a batch of frames (video)"),
                io.Color.Input("mask_color", optional=True, default="#FF00FF80", tooltip="Mask tint color, #RRGGBB or #RRGGBBAA (the alpha is the opacity)"),
            ],
            outputs=[
                _CONTEXT_TYPE.Output("context"),
                io.Image.Output("image"),
                io.Mask.Output("mask"),
            ],
        )

    @classmethod
    def execute(cls, context=None, image=None, mask=None, remove_isolated_pixels=0, fill_holes=False,
                mask_blur=0, mask_grow=0, mask_invert=False, remove_bg=False, rembg_color="#000000",
                preview=True, preview_fps=30.0, mask_color="#FF00FF80"):
        ctx = ctx_from(context)
        # Connected image/mask override the context's
        image = image if image is not None else ctx.get("image")
        mask = mask if mask is not None else ctx.get("mask")
        if image is None or mask is None:
            # The frontend restores a node's links by input index, so a workflow
            # saved against a different input order can hand the context to a
            # widget input; from here that looks exactly like a missing context.
            wired = "/".join(k for k, v in (("preview", preview), ("preview_fps", preview_fps),
                                            ("rembg_color", rembg_color), ("mask_color", mask_color))
                             if isinstance(v, dict))
            hint = f" - the context is wired to {wired}, re-link it to the context input" if wired else ""
            raise ValueError(f"No {'image' if image is None else 'mask'} to edit: connect it or use a context carrying one{hint}")
        _, H, W, _ = image.shape
        if mask.shape[1:] != (H, W):
            mask = _resize_mask(mask, W, H, "nearest-exact")

        # Clean up before feathering and growing: growing would enlarge the
        # specks the opening is there to drop
        mask = _fix_mask(mask, remove_isolated_pixels, fill_holes)
        mask = _feather_mask(mask, mask_blur)
        mask = _process_mask(mask, mask_grow, mask_invert)

        if remove_bg:
            mb = mask.clamp(0, 1).unsqueeze(-1).to(device=image.device, dtype=image.dtype)
            if _color_alpha(rembg_color) == 0.0:
                # A fully transparent color makes the unmasked area transparent
                # instead of colored: Join Image with Alpha's method, the mask
                # becomes the alpha
                image = torch.cat((image[..., :3], mb), dim=-1)
            else:
                image = image * mb + _pad_color_tensor(rembg_color, image.dtype, image.device) * (1 - mb)

        # The context keeps its model and clip; the edited image replaces its
        # own, so the latent (and cached sample) built from the old image are
        # stale.
        ctx_set_image(ctx, image)
        ctx["mask"] = mask

        # A crop info holding this same image and mask describes the space this
        # node edits, so the paste back follows the edit. After a crop the mask
        # is crop-sized and no longer matches the full-frame mask the crop info
        # stores, which is what leaves that geometry alone.
        info = ctx.get("crop_info")
        if info is not None and info["image"].shape[1:3] == image.shape[1:3] and info["mask"].shape == mask.shape:
            ctx["crop_info"] = {"image": image, "mask": mask,
                                "rects": [_mask_bbox(m) or (0, 0, W, H) for m in mask]}

        # The preview is display-only; the image output stays plain
        if not preview:
            return io.NodeOutput(ctx, image, mask)
        if image.shape[0] > 1:
            return io.NodeOutput(ctx, image, mask, ui=_preview_video(image, mask, preview_fps, mask_color))
        return io.NodeOutput(ctx, image, mask, ui=ui.PreviewImage(_overlay(image, mask, mask_color)))

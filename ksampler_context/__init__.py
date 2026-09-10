"""
KSampler (Context) node
-----------------------
A sampler that pulls most parameters from a CONTEXT object, with optional
overrides for model, latent, image, mask, sampler, and sigmas. Designed to work
seamlessly with the Context node's on-demand generation features.

Key behaviors:
- Uses context values unless overridden by inputs or widgets
- Encodes images to latents automatically (with mask support)
- With an image present, samples with refiner steps instead of base steps
- Handles start_step/end_step as offsets from total steps when negative
- Automatically uses Flux2Scheduler for Flux2 models
- Updates context after sampling (removes latent/mask, stores decoded image)
- Options are applied in list order: crop-inpaint and iterative upscale run
  before evaluate, tiled VAE settings apply to encode/decode; the options
  output returns the (updated) options for feeding back in
"""

import torch
import comfy.samplers
import comfy.sample
import comfy.model_management
import latent_preview
import comfy.model_base
from nodes import VAEDecode, CLIPTextEncode, ConditioningZeroOut
from comfy_api.latest import io

try:
    from comfy_extras.nodes_audio import vae_decode_audio
except ImportError:
    vae_decode_audio = None

try:
    from comfy_extras.nodes_flux import Flux2Scheduler, get_schedule
except ImportError:
    Flux2Scheduler = None
    get_schedule = None

import comfy.utils as _comfy_utils
from nodes import VAEEncode, VAEDecode, VAEDecodeTiled, VAEEncodeTiled, SetLatentNoiseMask
from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
from comfy_extras.nodes_post_processing import ColorTransfer
from ..context import _CONTEXT_TYPE, GibbyContext
from ..crop_inpaint_options import _KSAMPLER_OPTIONS_TYPE


def _find_option(options_list, opt_type):
    if options_list is None:
        return None
    for o in options_list:
        if o.get("type") == opt_type:
            return o
    return None


def _tiled_settings(options_list):
    """Tiled VAE settings from the options list: (tiled, tile_size, overlap, temporal_size, temporal_overlap)."""
    opts = _find_option(options_list, "tiled_vae")
    if opts is None:
        return False, 512, 64, 64, 8
    return True, opts.get("tile_size", 512), opts.get("overlap", 64), opts.get("temporal_size", 64), opts.get("temporal_overlap", 8)


def _encode_image(vae, image, tiled_decode, tile_size, overlap, temporal_size, temporal_overlap):
    """VAE-encode an image, tiled when the tiled VAE settings are on."""
    if tiled_decode:
        latent, = VAEEncodeTiled().encode(vae, image, tile_size, overlap, temporal_size, temporal_overlap)
    else:
        latent, = VAEEncode().encode(vae, image)
    return latent


def _decode_latent(vae, latent, tiled_decode, tile_size, overlap, temporal_size, temporal_overlap):
    """VAE-decode a latent, tiled when the tiled VAE settings are on."""
    if tiled_decode:
        image, = VAEDecodeTiled().decode(vae, latent, tile_size, overlap, temporal_size, temporal_overlap)
    else:
        image, = VAEDecode().decode(vae, latent)
    return image


def _normalize_mask(mask):
    """Normalize the untrusted mask input to a (B,H,W) float mask. The mask input
    can receive non-mask tensors (e.g. an image bridged into it); anything that is
    not a 2D spatial mask returns None and is treated as no mask."""
    if mask is None or not torch.is_tensor(mask):
        return None
    mask = mask.float()
    if mask.dim() == 4:
        # A channel-first mask always has a small channel dim (shape[1]); a
        # channels-last (B,H,W,C) image bridged into the input has a large one.
        if mask.shape[1] > 4:
            return None
        if mask.shape[1] == 1:
            mask = mask.squeeze(1)   # (B,1,H,W) channel-first
        else:
            mask = mask[:, 0]        # (B,C,H,W) -> first channel
    if mask.dim() != 3:
        return None
    return mask


def _get_mask_bbox(mask):
    """Get bounding box of non-zero mask area. Returns (x, y, w, h) or None."""
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


def _resize_to_target(img, mask, megapixels, scale_factor, multiple, method):
    """Resize image and mask to target size. Returns (img, mask, new_w, new_h)."""
    h, w = img.shape[1], img.shape[2]

    if scale_factor != 1.0:
        h = int(h * scale_factor)
        w = int(w * scale_factor)

    if megapixels > 0:
        target_px = megapixels * 1_000_000
        current_px = h * w
        if current_px > 0:
            scale = (target_px / current_px) ** 0.5
            h = int(h * scale)
            w = int(w * scale)

    if multiple > 1:
        h = max(multiple, (h // multiple) * multiple)
        w = max(multiple, (w // multiple) * multiple)

    if h == img.shape[1] and w == img.shape[2]:
        return img, mask, w, h

    import torch.nn.functional as F
    mode = {"bilinear": "bilinear", "area": "area", "nearest": "nearest", "lanczos": "bilinear"}.get(method, "bilinear")
    # F.interpolate expects (N, C, H, W) — permute channels-last to channels-first
    img = F.interpolate(img.permute(0, 3, 1, 2), size=(h, w), mode=mode, align_corners=False).permute(0, 2, 3, 1)
    if mask.dim() == 3:
        mask = F.interpolate(mask.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False).squeeze(1)
    else:
        mask = F.interpolate(mask, size=(h, w), mode="bilinear", align_corners=False)
    return img, mask, w, h


def _scale_mask(mask, scale):
    """Rescale mask: scale<1 shrinks white area, scale>1 expands. scale=1 is no-op."""
    if scale == 1.0:
        return mask
    import torch.nn.functional as F
    # Resize mask to scale size, then back — with threshold to keep hard edges
    b, h, w = mask.shape
    m4 = mask.unsqueeze(1).float()
    small_h = max(2, int(h * scale))
    small_w = max(2, int(w * scale))
    m4 = F.interpolate(m4, size=(small_h, small_w), mode="bilinear", align_corners=False)
    m4 = F.interpolate(m4, size=(h, w), mode="bilinear", align_corners=False)
    return m4.squeeze(1).to(mask.dtype)


def _inpaint_regions(image, mask, opts):
    """Resize the mask to the image and compute crop regions per the crop-inpaint
    options. Returns (mask, regions); regions is empty when there is nothing to inpaint."""
    import torch.nn.functional as F

    # Resize mask to image size if mismatch
    if mask.shape[1] != image.shape[1] or mask.shape[2] != image.shape[2]:
        if mask.dim() == 3:
            mask = F.interpolate(mask.unsqueeze(1), size=(image.shape[1], image.shape[2]), mode="bilinear", align_corners=False).squeeze(1)
        else:
            mask = F.interpolate(mask, size=(image.shape[1], image.shape[2]), mode="bilinear", align_corners=False)

    # Get mask regions (single or split)
    mask_mode = opts.get("mask_mode", "single")
    crop_factor = opts.get("crop_factor", 3.0)
    img_h, img_w = image.shape[1], image.shape[2]

    regions = []
    if mask_mode == "split":
        # Find disconnected regions using contour detection
        import cv2
        import numpy as np
        mask_2d = (mask.squeeze(0).cpu().numpy() * 255).astype(np.uint8) if mask.dim() == 3 else (mask.cpu().numpy() * 255).astype(np.uint8)
        mask_2d_float = mask.squeeze(0) if mask.dim() == 3 else mask
        contours, hierarchy = cv2.findContours(mask_2d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is not None:
            for j, contour in enumerate(contours):
                if hierarchy[0][j][3] != -1:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                if w < 4 or h < 4:
                    continue
                # Expand by crop_factor
                cw = int(w * crop_factor)
                ch = int(h * crop_factor)
                cx = x + w // 2
                cy = y + h // 2
                x0 = max(0, cx - cw // 2)
                y0 = max(0, cy - ch // 2)
                x1 = min(img_w, x0 + cw)
                y1 = min(img_h, y0 + ch)
                # Isolate this segment so other segments inside the crop
                # window are not inpainted together with it
                seg = np.zeros((img_h, img_w), dtype=np.uint8)
                cv2.drawContours(seg, [contour], -1, 1, -1)
                seg_mask = mask_2d_float * torch.from_numpy(seg > 0).to(mask.device)
                regions.append((x0, y0, x1, y1, seg_mask))
        # Filter by mask_indices if provided
        mask_indices_str = opts.get("mask_indices", "")
        if mask_indices_str:
            import re as _re
            indices = [int(s) for s in _re.findall(r'\d+', mask_indices_str)]
            valid = [i for i in indices if i < len(regions)]
            if valid:
                regions = [regions[i] for i in valid]
            else:
                regions = []
    else:
        # Single region: use full mask bbox
        bbox = _get_mask_bbox(mask)
        if bbox is not None:
            mx, my, mw, mh = bbox
            cw = int(mw * crop_factor)
            ch = int(mh * crop_factor)
            cx = mx + mw // 2
            cy = my + mh // 2
            x0 = max(0, cx - cw // 2)
            y0 = max(0, cy - ch // 2)
            x1 = min(img_w, x0 + cw)
            y1 = min(img_h, y0 + ch)
            regions.append((x0, y0, x1, y1, None))

    return mask, regions


def _make_noise_mask(model_obj, crop_latent, crop_mask, inpaint_mode, mask_scale_start, mask_scale_end):
    """Build the noise mask for masked_only inpainting. Returns (noise_mask, model_obj)."""
    if inpaint_mode != "masked_only":
        return None, model_obj

    if abs(mask_scale_start - mask_scale_end) <= 0.001 and mask_scale_start == 1.0:
        masked = SetLatentNoiseMask().set_mask(crop_latent, crop_mask)
        if isinstance(masked, tuple):
            masked = masked[0]
        return masked.get("noise_mask"), model_obj

    b, h, w = crop_mask.shape
    mask_2d = crop_mask.squeeze(0) if crop_mask.dim() == 3 else crop_mask
    ys, xs = torch.where(mask_2d > 0.5)
    if len(ys) > 0:
        cy = ys.float().mean().item()
        cx = xs.float().mean().item()
        max_r = torch.sqrt((ys.float() - cy) ** 2 + (xs.float() - cx) ** 2).max().item()
    else:
        cy, cx, max_r = h / 2, w / 2, min(h, w) / 2
    yy, xx = torch.meshgrid(
        torch.arange(h, device=crop_mask.device, dtype=torch.float),
        torch.arange(w, device=crop_mask.device, dtype=torch.float),
        indexing='ij'
    )
    dist = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    if abs(mask_scale_start - mask_scale_end) > 0.001:
        r_start = max_r * min(mask_scale_start, mask_scale_end)
        r_end = max_r * max(mask_scale_start, mask_scale_end)
        if r_end > r_start:
            apply_mask = ((r_end - dist) / (r_end - r_start)).clamp(0, 1)
        else:
            apply_mask = (dist <= r_start).float()
        if mask_scale_start < 1.0:
            apply_mask = apply_mask * crop_mask
    else:
        apply_mask = (dist <= max_r * mask_scale_start).float()
        if mask_scale_start < 1.0:
            apply_mask = apply_mask * crop_mask

    masked = SetLatentNoiseMask().set_mask(crop_latent, apply_mask)
    if isinstance(masked, tuple):
        masked = masked[0]
    noise_mask = masked.get("noise_mask")
    from comfy_extras.nodes_differential_diffusion import DifferentialDiffusion
    model_obj = DifferentialDiffusion.execute(model_obj, strength=1.0)[0]
    return noise_mask, model_obj


def _color_match(image, reference, opts):
    """Match the image's color back to the reference with Transfer Color per the
    option's color_match settings. Returns the image unchanged when it is off."""
    if not opts.get("color_match", True):
        return image
    method = opts.get("color_match_method", "mkl_lab")
    strength = float(opts.get("color_match_strength", 1.0))
    matched, = ColorTransfer.execute(image, reference, method, {"source_stats": "per_frame"}, strength)
    return matched


def _crop_inpaint(ctx, opts, model_obj, seed, steps_value, denoise_value,
                  sampler_obj, sigmas_tensor, cfg_value, positive, negative, decode, options_list=None):
    """Crop image by mask bbox, resize, encode, sample, composite back."""
    image = ctx["image"]
    mask = ctx["mask"]
    vae = ctx.get("vae")

    mask, regions = _inpaint_regions(image, mask, opts)
    if not regions:
        return io.NodeOutput(ctx, ctx.get("latent"), image, None, vae, ctx.get("vae_audio"), options_list)

    # Process each region
    megapixels = opts.get("megapixels", 0.0)
    scale_factor = opts.get("scale_factor", 1.0)
    multiple = opts.get("multiple", 8)
    method = opts.get("upscale_method", "bilinear")
    mask_scale_start = opts.get("mask_scale_start", 1.0)
    mask_scale_end = opts.get("mask_scale_end", 1.0)
    inpaint_mode = opts.get("inpaint_mode", "masked_only")
    tiled_decode, tile_size, overlap, temporal_size, temporal_overlap = _tiled_settings(options_list)

    result_image = image.clone()

    for idx, (x0, y0, x1, y1, seg_mask) in enumerate(regions):
        cw = x1 - x0
        ch = y1 - y0

        # Crop image and mask (mask is 3D: B,H,W); in split mode only this
        # segment's pixels are kept
        crop_img = result_image[:, y0:y1, x0:x1, :]
        if seg_mask is not None:
            crop_mask = seg_mask[y0:y1, x0:x1].unsqueeze(0)
        else:
            crop_mask = mask[:, y0:y1, x0:x1]

        # Resize
        crop_img, crop_mask, new_w, new_h = _resize_to_target(crop_img, crop_mask, megapixels, scale_factor, multiple, method)

        # Encode (ensure float32 for VAE)
        crop_img = crop_img.float()
        crop_latent = _encode_image(vae, crop_img, tiled_decode, tile_size, overlap, temporal_size, temporal_overlap)

        # Prepare noise
        latent_image = crop_latent["samples"]
        latent_image = comfy.sample.fix_empty_latent_channels(
            model_obj, latent_image,
            crop_latent.get("downscale_ratio_spacial", None),
            crop_latent.get("downscale_ratio_temporal", None)
        )
        noise = comfy.sample.prepare_noise(latent_image, seed)

        # Set noise mask
        noise_mask, model_obj = _make_noise_mask(model_obj, crop_latent, crop_mask, inpaint_mode, mask_scale_start, mask_scale_end)

        # Sample
        callback = latent_preview.prepare_callback(model_obj, steps_value)
        samples = comfy.sample.sample_custom(
            model_obj, noise, cfg_value, sampler_obj, sigmas_tensor,
            positive, negative, latent_image,
            noise_mask=noise_mask, callback=callback,
            disable_pbar=not _comfy_utils.PROGRESS_BAR_ENABLED, seed=seed
        )

        # Decode
        if decode and vae is not None:
            refined_crop = _decode_latent(vae, {"samples": samples}, tiled_decode, tile_size, overlap, temporal_size, temporal_overlap)
        else:
            refined_crop = None

        # Composite back
        import torch.nn.functional as F
        if refined_crop is not None:
            if refined_crop.dim() == 3:
                refined_crop = refined_crop.unsqueeze(0)
            paste_img = F.interpolate(
                refined_crop.permute(0, 3, 1, 2), size=(ch, cw), mode="bilinear", align_corners=False
            ).permute(0, 2, 3, 1)
            if crop_mask.dim() == 3:
                paste_mask = F.interpolate(crop_mask.unsqueeze(1), size=(ch, cw), mode="bilinear", align_corners=False).squeeze(1)
            else:
                paste_mask = F.interpolate(crop_mask, size=(ch, cw), mode="bilinear", align_corners=False)
            pm = paste_mask.float().unsqueeze(1)
            pm = F.avg_pool2d(pm, kernel_size=5, stride=1, padding=2)
            pm = pm.clamp(0, 1).squeeze(1).unsqueeze(-1)
            result_image[:, y0:y1, x0:x1, :3] = (
                paste_img[:, :, :, :3] * pm +
                result_image[:, y0:y1, x0:x1, :3] * (1 - pm)
            )

    # Match the result's color back to the original image
    result_image = _color_match(result_image, image, opts)

    # Update context
    ctx["image"] = result_image
    ctx.pop("latent", None)
    ctx.pop("mask", None)

    return io.NodeOutput(ctx, None, result_image, None, vae, ctx.get("vae_audio"), options_list)


def _iterative_upscale_step(image, mask, target_w, target_h, method, upscale_model, vae, model_obj,
                            seed, steps_value, cfg_value, sampler_obj, sigmas_tensor, positive, negative,
                            inpaint_opts=None, tiled_decode=False, tile_size=512, overlap=64,
                            temporal_size=64, temporal_overlap=8):
    """One iterative upscale step: scale image (and mask) to target size, encode, ksample, decode.
    With a mask, each step samples only the masked area (crop-inpaint options control the mask)."""
    import torch.nn.functional as F

    if upscale_model is not None:
        # Upscale with the model until at/above the target width, then bring the
        # result back to the exact target size
        w = image.shape[2]
        while image.shape[2] < target_w:
            image, = ImageUpscaleWithModel().execute(upscale_model, image)
            if image.shape[2] == w:
                break  # x1 model: no growth
        method = "bilinear"

    if image.shape[2] != target_w or image.shape[1] != target_h:
        mode = {"bilinear": "bilinear", "area": "area", "nearest": "nearest", "lanczos": "bilinear"}.get(method, "bilinear")
        image = F.interpolate(image.permute(0, 3, 1, 2), size=(target_h, target_w), mode=mode, align_corners=False).permute(0, 2, 3, 1)

    if mask is not None and (mask.shape[-2] != target_h or mask.shape[-1] != target_w):
        if mask.dim() == 3:
            mask = F.interpolate(mask.unsqueeze(1), size=(target_h, target_w), mode="bilinear", align_corners=False).squeeze(1)
        else:
            mask = F.interpolate(mask, size=(target_h, target_w), mode="bilinear", align_corners=False)

    latent = _encode_image(vae, image.float(), tiled_decode, tile_size, overlap, temporal_size, temporal_overlap)
    latent_samples = comfy.sample.fix_empty_latent_channels(
        model_obj, latent["samples"],
        latent.get("downscale_ratio_spacial", None),
        latent.get("downscale_ratio_temporal", None)
    )
    noise = comfy.sample.prepare_noise(latent_samples, seed)
    noise_mask = None
    if mask is not None:
        inpaint_mode = (inpaint_opts or {}).get("inpaint_mode", "masked_only")
        mask_scale_start = (inpaint_opts or {}).get("mask_scale_start", 1.0)
        mask_scale_end = (inpaint_opts or {}).get("mask_scale_end", 1.0)
        noise_mask, model_obj = _make_noise_mask(model_obj, latent, mask, inpaint_mode, mask_scale_start, mask_scale_end)
    callback = latent_preview.prepare_callback(model_obj, steps_value)
    samples = comfy.sample.sample_custom(
        model_obj, noise, cfg_value, sampler_obj, sigmas_tensor,
        positive, negative, latent_samples,
        noise_mask=noise_mask,
        callback=callback,
        disable_pbar=not _comfy_utils.PROGRESS_BAR_ENABLED, seed=seed
    )
    image = _decode_latent(vae, {"samples": samples}, tiled_decode, tile_size, overlap, temporal_size, temporal_overlap)
    return image, mask


def _iterative_upscale(ctx, opts, options_list, model_obj, seed, steps_value,
                       cfg_value, sampler_obj, positive, negative,
                       start_step=0.0, end_step=10000.0, leftover_noise=False,
                       inpaint_opts=None, skip_color_match=False):
    """Iterative pixel-space upscale along a linear scale path (simple step mode).

    State (next_step, base size) lives in the options dict; on first run it is
    initialized from the current image and the options returned with the
    updated next_step, so the output can be fed back in for further steps.
    A mask in the context is resized with the image and applied on every step
    (inpaint mode and mask scaling come from the crop-inpaint options).
    In total mode, after the final planned step the result's color is matched
    back to the original image (Transfer Color, mkl_lab).
    """
    image = ctx["image"]
    mask = ctx.get("mask")
    vae = ctx["vae"]
    # VAEs round-trip exactly only at multiples of their spatial downscale
    # ratio - align the per-step target sizes to it
    vae_ratio = vae.downscale_ratio
    if isinstance(vae_ratio, (tuple, list)):
        # Video VAE: (temporal, h, w)
        h_ratio, w_ratio = int(vae_ratio[1]), int(vae_ratio[2])
    else:
        h_ratio = w_ratio = int(vae_ratio)
    scheduler_name = ctx.get("scheduler", "normal")
    sampler_name = ctx.get("sampler", "euler")
    if sampler_name not in comfy.samplers.KSampler.SAMPLERS:
        sampler_name = comfy.samplers.KSampler.SAMPLERS[0]

    factor = float(opts.get("upscale_factor", 2.0))
    total_steps = max(int(opts.get("steps", 3)), 1)
    start_denoise = float(opts.get("start_denoise", 1.0))
    target_denoise = float(opts.get("target_denoise", 0.6))
    method = opts.get("upscale_method", "bilinear")
    mode = opts.get("mode", "total")
    upscale_model = opts.get("upscale_model")
    verbose = opts.get("verbose", False)
    tiled_decode, tile_size, overlap, temporal_size, temporal_overlap = _tiled_settings(options_list)

    # The engine's output cache can hand back the same options object on
    # repeated runs - never mutate it, the step state travels on a copy
    opts = dict(opts)
    if "next_step" not in opts:
        # First run: record the initial image size and reset the step counter
        opts["next_step"] = 0
        opts["base_w"] = image.shape[2]
        opts["base_h"] = image.shape[1]

    original_image = image  # color reference for the final step

    base_w = int(opts["base_w"])
    base_h = int(opts["base_h"])
    next_step = int(opts["next_step"])

    if mode == "total" and next_step < total_steps:
        step_indices = range(next_step + 1, total_steps + 1)
    else:
        step_indices = [next_step + 1]

    for i in step_indices:
        scale = 1.0 + (factor - 1.0) * i / total_steps
        target_w = max(1, int(round(base_w * scale / w_ratio)) * w_ratio)
        target_h = max(1, int(round(base_h * scale / h_ratio)) * h_ratio)
        # Denoise ramps from start to target across the planned steps, then holds at target
        if i > total_steps:
            denoise_step = target_denoise
        elif total_steps == 1:
            denoise_step = start_denoise
        else:
            denoise_step = start_denoise + (target_denoise - start_denoise) * (i - 1) / (total_steps - 1)
        sigmas_tensor = _calculate_sigmas(model_obj, scheduler_name, steps_value, sampler_name, denoise_step)
        sigmas_tensor, skip = _apply_start_end_steps(sigmas_tensor, steps_value, start_step, end_step, leftover_noise)
        if skip:
            # Start step beyond the available steps: keep the image, advance the step
            next_step = i
            continue
        if verbose:
            print(f"Gibby Iterative Upscale: step {i}/{total_steps} scale={scale:.2f} size={target_w}x{target_h} denoise={denoise_step:.3f}")
        image, mask = _iterative_upscale_step(image, mask, target_w, target_h, method, upscale_model, vae, model_obj,
                                                seed, steps_value, cfg_value, sampler_obj, sigmas_tensor, positive, negative, inpaint_opts,
                                                tiled_decode, tile_size, overlap, temporal_size, temporal_overlap)
        next_step = i

    # After the final planned step, match the result's color back to the original image
    if not skip_color_match and mode == "total" and next_step == total_steps:
        image = _color_match(image, original_image, opts)

    opts["next_step"] = next_step
    # New list, not in-place: the input list may be a cached engine object
    options_list = [opts if o.get("type") == "iterative_upscale" else o for o in options_list]

    # Update context
    ctx["image"] = image
    if mask is None:
        ctx.pop("mask", None)
    else:
        ctx["mask"] = mask
    ctx.pop("latent", None)

    return io.NodeOutput(ctx, None, image, None, vae, ctx.get("vae_audio"), options_list)


def _iterative_inpaint_upscale(ctx, inpaint_opts, iterative_opts, options_list, model_obj, seed, steps_value,
                               cfg_value, sampler_obj, positive, negative,
                               start_step=0.0, end_step=10000.0, leftover_noise=False):
    """Iterative inpaint upscale: crop the mask regions (crop-inpaint options), iteratively
    upscale each crop with the mask applied on every step, then composite the results back
    into the full image upscaled by the iterative total factor."""
    import torch.nn.functional as F
    image = ctx["image"]
    vae = ctx["vae"]
    mask, regions = _inpaint_regions(image, ctx["mask"], inpaint_opts)
    if not regions:
        return None

    # Both options may carry color match - the one earlier in the options list
    # wins; when the inpaint one wins, the per-crop match is skipped and the
    # full image is matched back after compositing
    inpaint_wins = options_list.index(inpaint_opts) < options_list.index(iterative_opts)

    img_h, img_w = image.shape[1], image.shape[2]
    factor = float(iterative_opts.get("upscale_factor", 2.0))
    method = iterative_opts.get("upscale_method", "bilinear")
    mode = {"bilinear": "bilinear", "area": "area", "nearest": "nearest", "lanczos": "bilinear"}.get(method, "bilinear")
    vae_ratio = vae.downscale_ratio
    if isinstance(vae_ratio, (tuple, list)):
        h_ratio, w_ratio = int(vae_ratio[1]), int(vae_ratio[2])
    else:
        h_ratio = w_ratio = int(vae_ratio)
    final_w = max(1, int(round(img_w * factor / w_ratio)) * w_ratio)
    final_h = max(1, int(round(img_h * factor / h_ratio)) * h_ratio)
    full = F.interpolate(image.permute(0, 3, 1, 2), size=(final_h, final_w), mode=mode, align_corners=False).permute(0, 2, 3, 1).clone()

    megapixels = inpaint_opts.get("megapixels", 0.0)
    scale_factor = inpaint_opts.get("scale_factor", 1.0)
    multiple = inpaint_opts.get("multiple", 8)
    crop_method = inpaint_opts.get("upscale_method", "bilinear")

    out_options = options_list
    for (x0, y0, x1, y1, seg_mask) in regions:
        crop_img = image[:, y0:y1, x0:x1, :]
        crop_mask = seg_mask[y0:y1, x0:x1].unsqueeze(0) if seg_mask is not None else mask[:, y0:y1, x0:x1]
        crop_img, crop_mask, _, _ = _resize_to_target(crop_img, crop_mask, megapixels, scale_factor, multiple, crop_method)
        crop_ctx = {"image": crop_img, "mask": crop_mask, "vae": vae,
                    "scheduler": ctx.get("scheduler", "normal"), "sampler": ctx.get("sampler", "euler")}
        out = _iterative_upscale(crop_ctx, iterative_opts, options_list, model_obj, seed,
                                 steps_value, cfg_value, sampler_obj, positive, negative,
                                 start_step, end_step, leftover_noise, inpaint_opts=inpaint_opts,
                                 skip_color_match=inpaint_wins)
        out_options = out[6]
        final_crop = out[2]
        final_mask = crop_ctx.get("mask")
        # Composite the refined crop into the upscaled full image
        fx0, fx1 = int(round(x0 * final_w / img_w)), int(round(x1 * final_w / img_w))
        fy0, fy1 = int(round(y0 * final_h / img_h)), int(round(y1 * final_h / img_h))
        rw, rh = fx1 - fx0, fy1 - fy0
        if rw < 1 or rh < 1:
            continue
        paste_img = F.interpolate(final_crop.permute(0, 3, 1, 2), size=(rh, rw), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
        pm = F.interpolate(final_mask.float().unsqueeze(1) if final_mask.dim() == 3 else final_mask.float(),
                           size=(rh, rw), mode="bilinear", align_corners=False)
        pm = F.avg_pool2d(pm, kernel_size=5, stride=1, padding=2)
        pm = pm.clamp(0, 1).squeeze(1).unsqueeze(-1)
        full[:, fy0:fy1, fx0:fx1, :3] = (
            paste_img[:, :, :, :3] * pm +
            full[:, fy0:fy1, fx0:fx1, :3] * (1 - pm)
        )

    if inpaint_wins:
        full = _color_match(full, image, inpaint_opts)

    # Update context
    ctx["image"] = full
    ctx.pop("latent", None)
    ctx.pop("mask", None)

    return io.NodeOutput(ctx, None, full, None, vae, ctx.get("vae_audio"), out_options)


def _calculate_sigmas(model, scheduler_name, steps, sampler_name="euler", denoise=1.0):
    """Calculate sigmas from scheduler name and step count, respecting denoise level."""
    if scheduler_name not in comfy.samplers.KSampler.SCHEDULERS:
        scheduler_name = comfy.samplers.KSampler.SCHEDULERS[0]
    
    device = comfy.model_management.get_torch_device()
    sampler_obj = comfy.samplers.KSampler(model, steps, device, sampler=sampler_name)
    sampler_obj.scheduler = scheduler_name
    
    # Handle denoise like KSampler.set_steps() does
    if denoise is None or denoise > 0.9999:
        sigmas = sampler_obj.calculate_sigmas(steps).to(device)
    else:
        if denoise <= 0.0:
            sigmas = torch.FloatTensor([])
        else:
            new_steps = int(steps/denoise)
            sigmas_full = sampler_obj.calculate_sigmas(new_steps).to(device)
            sigmas = sigmas_full[-(steps + 1):]

    return sigmas


def _apply_start_end_steps(sigmas_tensor, steps_value, start_step, end_step, leftover_noise):
    """Slice sigmas to a start/end step sub-range, like the regular KSampler.

    start/end are multipliers of the step count when abs < 1, offsets from the
    total when negative, otherwise absolute step numbers. Returns (sigmas, skip);
    skip means the start step is at or beyond the available steps and sampling
    should not run.
    """
    start_step_val = round(steps_value * start_step) if abs(start_step) < 1 else int(round(start_step))
    end_step_val = round(steps_value * end_step) if abs(end_step) < 1 else int(round(end_step))
    actual_start_step = steps_value + start_step_val if start_step_val < 0 else start_step_val
    actual_end_step = steps_value + end_step_val if end_step_val < 0 else end_step_val

    if actual_end_step < (len(sigmas_tensor) - 1):
        sigmas_tensor = sigmas_tensor[:actual_end_step + 1]
        if not leftover_noise:
            sigmas_tensor[-1] = 0

    skip = actual_start_step >= (len(sigmas_tensor) - 1)
    if not skip and actual_start_step > 0:
        sigmas_tensor = sigmas_tensor[actual_start_step:]

    return sigmas_tensor, skip


def _ensure_conditioning(ctx):
    """Encode positive/negative conditioning from clip if the context lacks it."""
    if ctx.get("positive") is None and ctx.get("clip") is not None:
        ctx["positive"], = CLIPTextEncode().encode(ctx["clip"], ctx.get("positive_prompt", ""))
    if ctx.get("negative") is None:
        if ctx.get("cfg") == 1 and ctx.get("positive") is not None:
            ctx["negative"], = ConditioningZeroOut().zero_out(ctx["positive"])
        elif ctx.get("clip") is not None:
            ctx["negative"], = CLIPTextEncode().encode(ctx["clip"], ctx.get("negative_prompt", ""))


def _resolve_early_params(ctx, steps):
    """Resolve steps/cfg/sampler for the pre-evaluate sampling paths."""
    base_steps = ctx.get("steps") or 0
    if steps > 0:
        steps_value = steps
    elif ctx.get("step_refiner", 0) > 0:
        steps_value = ctx["step_refiner"]
    else:
        steps_value = base_steps if base_steps > 0 else 20
    cfg_value = ctx.get("cfg", 8.0)
    sampler_name = ctx.get("sampler", "euler")
    if sampler_name not in comfy.samplers.KSampler.SAMPLERS:
        sampler_name = comfy.samplers.KSampler.SAMPLERS[0]
    sampler_obj = comfy.samplers.sampler_object(sampler_name)
    return steps_value, cfg_value, sampler_name, sampler_obj


class GibbyKSamplerContext(io.ComfyNode):
    """Sample using parameters from a CONTEXT object with optional overrides."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Gibby_KSampler_Context",
            display_name="KSampler (Context)",
            category="gibby",
            description=(
                "Samples using parameters from a CONTEXT object. Override model, latent, image, "
                "mask, sampler, or sigmas as needed. Automatically encodes images and updates context."
            ),
            inputs=[
                _CONTEXT_TYPE.Input("context"),
                io.Model.Input("model", optional=True),
                io.Latent.Input("latent", optional=True),
                io.Image.Input("image", optional=True),
                io.Mask.Input("mask", optional=True),
                io.Sampler.Input("sampler", optional=True),
                io.Sigmas.Input("sigmas", optional=True),
                _KSAMPLER_OPTIONS_TYPE.Input("options", optional=True, tooltip="Connect an options node (Crop-Inpaint, Iterative Upscale, Tiled VAE) or Merge KSampler Options"),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate="fixed"),
                io.Int.Input("steps", default=0, min=0, max=10000),
                io.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Float.Input("start_step", default=0.0, min=-10000.0, max=10000.0, step=0.01, advanced=True),
                io.Float.Input("end_step", default=10000.0, min=-10000.0, max=10000.0, step=0.01, advanced=True),
                io.Boolean.Input("add_noise", default=True, advanced=True),
                io.Boolean.Input("leftover_noise", default=False, advanced=True),
                io.Boolean.Input("decode", default=True, advanced=True, tooltip="Decode latent to image/audio. When False, stores latent in context instead."),
            ],
            outputs=[
                _CONTEXT_TYPE.Output(display_name="context"),
                io.Latent.Output(display_name="latent"),
                io.Image.Output(display_name="image"),
                io.Audio.Output(display_name="audio"),
                io.Vae.Output(display_name="vae"),
                io.Vae.Output(display_name="audio_vae"),
                _KSAMPLER_OPTIONS_TYPE.Output(display_name="options"),
            ],
        )

    @classmethod
    def execute(cls, context, model=None, latent=None, image=None, mask=None, options=None,
                sampler=None, sigmas=None, seed=0, steps=0, denoise=1.0,
                start_step=0.0, end_step=10000.0, add_noise=True, leftover_noise=False,
                decode=True) -> io.NodeOutput:
        # Start with context dict and apply overrides
        ctx = dict(context) if isinstance(context, dict) else {}

        if model is not None:
            ctx["model"] = model

        if latent is not None:
            ctx["latent"] = latent

        if image is not None:
            ctx["image"] = image
            # A connected image takes priority over a latent inherited from
            # the context (a directly-connected latent still wins): clear it
            # so the image is encoded fresh downstream, and drop the cached
            # sample that was built from the old latent
            if latent is None and ctx.get("latent") is not None:
                ctx.pop("latent", None)
                ctx.pop("_kctx_sampled", None)

        # A directly-connected mask overrides the context's before evaluation.
        # The input is untrusted (an image can be bridged into it) - keep only
        # usable masks, treat anything else as absent.
        mask = _normalize_mask(mask)
        if mask is not None:
            ctx["mask"] = mask

        # Tiled VAE settings come from the options list (Tiled VAE options node):
        # presence of the option switches encode/decode to the tiled variants
        tiled_decode, tile_size, overlap, temporal_size, temporal_overlap = _tiled_settings(options)

        # Apply image-based options (crop-inpaint, iterative upscale) in list
        # order before evaluate(), which would otherwise encode the image for nothing
        options_list = options or []
        result = None
        # Both options together run the iterative inpaint upscale: crop, upscale with
        # the mask applied on every step, composite back
        inpaint_opt = _find_option(options_list, "inpaint")
        iterative_opt = _find_option(options_list, "iterative_upscale")
        if (inpaint_opt is not None and iterative_opt is not None
                and ctx.get("image") is not None and ctx.get("mask") is not None and ctx.get("vae") is not None):
            _ensure_conditioning(ctx)
            model_obj = ctx.get("model")
            steps_value, cfg_value, sampler_name, sampler_obj = _resolve_early_params(ctx, steps)
            result = _iterative_inpaint_upscale(ctx, inpaint_opt, iterative_opt, options_list, model_obj, seed,
                                                 steps_value, cfg_value, sampler_obj, ctx.get("positive"), ctx.get("negative"),
                                                 start_step, end_step, leftover_noise)
            if result is None:
                # No mask regions: just iteratively upscale the full image
                result = _iterative_upscale(ctx, iterative_opt, options_list, model_obj, seed,
                                             steps_value, cfg_value, sampler_obj, ctx.get("positive"), ctx.get("negative"),
                                             start_step, end_step, leftover_noise)
        for opt in options_list:
            if result is not None:
                break
            if opt.get("type") == "inpaint" and ctx.get("image") is not None and ctx.get("mask") is not None:
                # Only encode conditioning, skip image→latent
                _ensure_conditioning(ctx)
                model_obj = ctx.get("model")
                denoise_value = denoise  # image is present here by definition
                steps_value, cfg_value, sampler_name, sampler_obj = _resolve_early_params(ctx, steps)
                scheduler_name = ctx.get("scheduler", "normal")
                sigmas_tensor = _calculate_sigmas(model_obj, scheduler_name, steps_value, sampler_name, denoise_value)
                result = _crop_inpaint(ctx, opt, model_obj, seed, steps_value, denoise_value,
                                       sampler_obj, sigmas_tensor, cfg_value, ctx.get("positive"), ctx.get("negative"), decode,
                                       options_list)
            elif opt.get("type") == "iterative_upscale" and ctx.get("image") is not None and ctx.get("vae") is not None:
                _ensure_conditioning(ctx)
                model_obj = ctx.get("model")
                steps_value, cfg_value, sampler_name, sampler_obj = _resolve_early_params(ctx, steps)
                result = _iterative_upscale(ctx, opt, options_list, model_obj, seed,
                                             steps_value, cfg_value, sampler_obj, ctx.get("positive"), ctx.get("negative"),
                                             start_step, end_step, leftover_noise)
        if result is not None:
            return result

        # Fill in derived values on demand (latent from image, conditioning, width/height).
        # With tiled_decode on, the image is encoded with VAE Encode (Tiled).
        GibbyContext.evaluate(ctx, tiled=tiled_decode, tile_size=tile_size, overlap=overlap,
                               temporal_size=temporal_size, temporal_overlap=temporal_overlap)

        # Apply mask only if it came from the input (not context) and latent exists.
        if mask is not None and ctx.get("latent") is not None and "noise_mask" not in ctx["latent"]:
            ctx["latent"], = SetLatentNoiseMask().set_mask(ctx["latent"], mask)

        # Resolve sampling parameters from context with widget overrides
        model_obj = ctx.get("model")
        if "seed_value" not in ctx:
            ctx["seed_value"] = seed
        
        # Denoise: use widget value only if image exists, otherwise force 1.0
        has_image = ctx.get("image") is not None
        denoise_value = denoise if has_image else 1.0

        # Steps: widget wins; with an image present, prefer refiner steps from context -
        # unless a sub-range of the base steps was explicitly requested via start/end step.
        base_steps = ctx.get("steps") or 0
        if steps > 0:
            steps_value = steps
        elif has_image and ctx.get("step_refiner", 0) > 0 and not (start_step != 0 or end_step < base_steps):
            steps_value = ctx["step_refiner"]
        else:
            steps_value = ctx.get("steps")

        # Cache check: if this context already has a sampled result with matching
        # parameters, return it instead of re-sampling. This handles the case where
        # the latent output wasn't connected during the first run (result discarded
        # by the engine) but is connected now.
        cached = ctx.get("_kctx_sampled")
        if cached is not None:
            if (cached.get("seed") == seed and cached.get("steps") == steps_value
                    and cached.get("denoise") == denoise_value
                    and cached.get("add_noise") == add_noise
                    and cached.get("leftover_noise") == leftover_noise
                    and cached.get("decode") == decode
                    and cached.get("tiled_decode") == tiled_decode
                    and cached.get("tile_size") == tile_size
                    and cached.get("overlap") == overlap
                    and cached.get("temporal_size") == temporal_size
                    and cached.get("temporal_overlap") == temporal_overlap
                    and cached.get("start_step") == start_step
                    and cached.get("end_step") == end_step):
                out_latent = cached["out_latent"]
                decoded_image = ctx.get("image")
                decoded_audio = ctx.get("audio")
                return io.NodeOutput(ctx, out_latent, decoded_image, decoded_audio, ctx.get("vae"), ctx.get("vae_audio") or ctx.get("audio_vae"), options)

        cfg_value = ctx.get("cfg")

        # Resolve sampler and sigmas
        if sampler is not None:
            # Use provided SAMPLER object directly
            sampler_obj = sampler
        else:
            # Create sampler from context's sampler string
            sampler_name = ctx.get("sampler", "euler")
            if sampler_name not in comfy.samplers.KSampler.SAMPLERS:
                sampler_name = comfy.samplers.KSampler.SAMPLERS[0]
            sampler_obj = comfy.samplers.sampler_object(sampler_name)
        
        if sigmas is not None:
            # Use provided SIGMAS tensor directly (ignore start/end steps)
            sigmas_tensor = sigmas
            use_start_end_steps = False
        else:
            # Check if model is Flux2 - if so, use Flux2Scheduler
            is_flux2 = isinstance(model_obj.model, comfy.model_base.Flux2) if hasattr(model_obj, 'model') else False
            
            if is_flux2 and get_schedule is not None:
                # Use Flux2Scheduler logic
                # Get width/height from context, latent, or image
                width = ctx.get("width", 0)
                height = ctx.get("height", 0)
                
                if width == 0 or height == 0:
                    if ctx.get("image") is not None:
                        img_h, img_w = ctx["image"].shape[1], ctx["image"].shape[2]
                        width, height = img_w, img_h
                    elif ctx.get("latent") is not None:
                        lat_h, lat_w = ctx["latent"]["samples"].shape[2], ctx["latent"]["samples"].shape[3]
                        width, height = lat_w * 16, lat_h * 16  # Flux2 uses 16x downscale
                    else:
                        width, height = 1024, 1024  # Default
                
                seq_len = (width * height / (16 * 16))
                sigmas_tensor = get_schedule(steps_value, round(seq_len)).to(comfy.model_management.get_torch_device())
                
                # If no start/end step provided and denoise < 1, use low_sigmas
                has_start_end = start_step != 0.0 or end_step < 10000.0
                if not has_start_end and denoise_value < 1.0:
                    steps = max(sigmas_tensor.shape[-1] - 1, 0)
                    total_steps = round(steps * denoise_value)
                    sigmas_tensor = sigmas_tensor[-(total_steps + 1):]
                
                use_start_end_steps = True
            else:
                # Regular scheduler
                scheduler_name = ctx.get("scheduler", "normal")
                sampler_name_for_sigmas = ctx.get("sampler", "euler")
                sigmas_tensor = _calculate_sigmas(model_obj, scheduler_name, steps_value, sampler_name_for_sigmas, denoise_value)
                use_start_end_steps = True

        # Conditioning was filled in by evaluate() above if it was missing.
        positive = ctx.get("positive")
        negative = ctx.get("negative")

        # Prepare noise
        latent_image = ctx["latent"]["samples"]
        
        # Fix latent channels to match model expectations (like SamplerCustom does)
        latent_image = comfy.sample.fix_empty_latent_channels(
            model_obj, 
            latent_image, 
            ctx["latent"].get("downscale_ratio_spacial", None),
            ctx["latent"].get("downscale_ratio_temporal", None)
        )
        
        if not add_noise:
            noise = comfy.sample.prepare_empty_noise(latent_image)
        else:
            batch_inds = ctx["latent"].get("batch_index") if "batch_index" in ctx["latent"] else None
            noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

        # Get noise mask from latent
        noise_mask = None
        if "noise_mask" in ctx["latent"]:
            noise_mask = ctx["latent"]["noise_mask"]

        # Prepare callback for progress display
        callback = latent_preview.prepare_callback(model_obj, steps_value)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        # Perform sampling with start/end step handling (only when using calculated sigmas)
        samples = None
        if use_start_end_steps:
            sigmas_tensor, skip = _apply_start_end_steps(sigmas_tensor, steps_value, start_step, end_step, leftover_noise)
            if skip:
                # Start step beyond available steps, return latent as-is or zeros
                samples = latent_image if ctx["latent"] is not None else torch.zeros_like(noise)

        # Sample with the (possibly sliced) sigmas
        if samples is None:
            samples = comfy.sample.sample_custom(
                model_obj, noise, cfg_value, sampler_obj, sigmas_tensor,
                positive, negative, latent_image,
                noise_mask=noise_mask, callback=callback,
                disable_pbar=disable_pbar, seed=seed
            )

        # Build output latent
        out_latent = ctx["latent"].copy()
        out_latent.pop("downscale_ratio_spacial", None)
        out_latent.pop("downscale_ratio_temporal", None)
        out_latent["samples"] = samples

        # Decode to image for context update and output
        decoded_image = None
        decoded_audio = None
        
        if decode:
            if ctx.get("vae") is not None:
                decoded_image = _decode_latent(ctx["vae"], out_latent, tiled_decode, tile_size, overlap, temporal_size, temporal_overlap)

            # Decode to audio for context update and output (overrides any existing context.audio)
            if ctx.get("vae_audio") is not None and vae_decode_audio is not None:
                decoded_audio = vae_decode_audio(ctx["vae_audio"], out_latent)

        # Update context after sampling
        ctx.pop("mask", None)    # Remove mask
        
        if decode:
            # When decoding, remove latent and store decoded image/audio
            ctx.pop("latent", None)
            
            if decoded_image is not None:
                ctx["image"] = decoded_image  # Store decoded image in context

            if decoded_audio is not None:
                ctx["audio"] = decoded_audio  # Decoded audio overrides context.audio
        else:
            # When not decoding, clear image/audio and store latent
            ctx.pop("image", None)
            ctx.pop("audio", None)
            ctx["latent"] = out_latent

        # Store sampled result in context for cache reuse
        ctx["_kctx_sampled"] = {
            "out_latent": out_latent,
            "seed": seed,
            "steps": steps_value,
            "denoise": denoise_value,
            "add_noise": add_noise,
            "leftover_noise": leftover_noise,
            "decode": decode,
            "tiled_decode": tiled_decode,
            "tile_size": tile_size,
            "overlap": overlap,
            "temporal_size": temporal_size,
            "temporal_overlap": temporal_overlap,
            "start_step": start_step,
            "end_step": end_step,
        }

        return io.NodeOutput(ctx, out_latent, decoded_image, decoded_audio, ctx.get("vae"), ctx.get("vae_audio") or ctx.get("audio_vae"), options)

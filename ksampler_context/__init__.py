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
"""

import torch
import comfy.samplers
import comfy.sample
import comfy.model_management
import latent_preview
import comfy.model_base
from nodes import VAEDecode
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

from ..context import _CONTEXT_TYPE, GibbyContext


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
            ],
        )

    @classmethod
    def execute(cls, context, model=None, latent=None, image=None, mask=None, sampler=None, sigmas=None,
                seed=0, steps=0, denoise=1.0, start_step=0.0, end_step=10000.0,
                add_noise=True, leftover_noise=False, decode=True) -> io.NodeOutput:
        # Start with context dict and apply overrides
        ctx = dict(context) if isinstance(context, dict) else {}

        if model is not None:
            ctx["model"] = model

        if latent is not None:
            ctx["latent"] = latent

        if image is not None:
            ctx["image"] = image

        # A directly-connected mask overrides the context's before evaluation.
        if mask is not None:
            ctx["mask"] = mask

        # Fill in derived values on demand (latent from image, conditioning, width/height).
        GibbyContext.evaluate(ctx)

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

        cfg_value = ctx.get("cfg")
        
        # Handle start_step/end_step:
        # - If abs(value) < 1: treat as multiplier first (e.g., 0.2 * steps = offset)
        # - If negative: offset from total steps (e.g., -3 means 3 steps before end)
        # - Otherwise: absolute step number
        start_step_val = round(steps_value * start_step) if abs(start_step) < 1 else int(round(start_step))
        end_step_val = round(steps_value * end_step) if abs(end_step) < 1 else int(round(end_step))
        actual_start_step = steps_value + start_step_val if start_step_val < 0 else start_step_val
        actual_end_step = steps_value + end_step_val if end_step_val < 0 else end_step_val
        
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
        force_full_denoise = not leftover_noise
        
        samples = None
        if use_start_end_steps:
            # Apply end_step slicing first
            if actual_end_step < (len(sigmas_tensor) - 1):
                sigmas_tensor = sigmas_tensor[:actual_end_step + 1]
                if force_full_denoise:
                    sigmas_tensor[-1] = 0

            # Then apply start_step handling
            if actual_start_step >= (len(sigmas_tensor) - 1):
                # Start step beyond available steps, return latent as-is or zeros
                samples = latent_image if ctx["latent"] is not None else torch.zeros_like(noise)
            elif actual_start_step > 0:
                sigmas_tensor = sigmas_tensor[actual_start_step:]

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
                decoded_image, = VAEDecode().decode(ctx["vae"], out_latent)

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

        return io.NodeOutput(ctx, out_latent, decoded_image, decoded_audio, ctx.get("vae"), ctx.get("vae_audio") or ctx.get("audio_vae"))

"""
Context Loader node
-------------------
A self-contained context source with a mode selector for the loading category:

- diffusion model: separate unet/clip/vae selectors, each left on "None" to skip.
- checkpoint: loads a full checkpoint like Load Checkpoint.

With a context connected, its values carry through for anything the node
does not load or receive connected; the loaded models, connected values and
widget values override its matching fields.

Carries the sampling parameters as widgets, and always fills in derived values on demand:

- positive/negative conditioning encoded from the prompt widgets via clip
- latent from image via VAE encode if present (with optional noise mask)
- empty latent when width/height are given with nothing to sample from

Use Context to merge or override individual values of this context downstream.
"""

import logging
import weakref

import torch

import folder_paths
import comfy.samplers
import comfy.sd
from nodes import UNETLoader, VAELoader, CLIPLoader, CheckpointLoaderSimple
from comfy_api.latest import io

try:
    from comfy_extras.nodes_model_advanced import ModelAttentionBackend
except ImportError:
    ModelAttentionBackend = None
    logging.warning("[Gibby Context Loader] ModelAttentionBackend is missing - ck_attn is disabled (update ComfyUI)")

from ..lora_loader import _apply_lora, iter_lora_stack
from . import _CONTEXT_TYPE, _lora_stack, ctx_from, ctx_set_image, ctx_set_latent, GibbyContext

# execute() runs once per list item when a list input is mapped over, so the
# model loads are cached: each part is built once per key, not once per item.
# A single entry per part keeps the cache from holding extra models in memory.
class _WeakCache:
    """One load, held weakly by key.

    A list-mapped execute() has to build each part once for the whole list, but
    the module must not be what keeps the result alive afterwards: a strong
    reference held the last checkpoint, clip and VAE in host RAM for the rest of
    the server session, past unload_all_models from the Clear VRAM options. A dead
    reference is a miss, so the model goes when ComfyUI drops it."""

    def __init__(self):
        self.key = None
        self.refs = None

    def get_or_load(self, key, load):
        """The objects load() returns (a tuple), for this key only."""
        if self.key == key:
            values = tuple(ref() for ref in self.refs)
            if all(values):
                return values
        values = tuple(load())
        self.key = key
        self.refs = tuple(weakref.ref(value) for value in values)
        return values


_last_checkpoint = _WeakCache()  # ckpt_name -> (model, clip, vae)
_last_unet = _WeakCache()  # (unet_name, weight_dtype) -> (model,)
_last_clip = _WeakCache()  # (clip_path, type, device) -> (clip,)
_last_vae = _WeakCache()  # vae_name -> (vae,)
_last_vae_audio = _WeakCache()  # vae_audio_name -> (vae_audio,)


def _load_checkpoint(ckpt_name):
    return _last_checkpoint.get_or_load(ckpt_name, lambda: CheckpointLoaderSimple().load_checkpoint(ckpt_name))


def _load_unet(unet_name, weight_dtype):
    model, = _last_unet.get_or_load((unet_name, weight_dtype), lambda: UNETLoader().load_unet(unet_name, weight_dtype))
    return model


def _load_clip(clip_path, type_, device):
    def load():
        clip_type = getattr(comfy.sd.CLIPType, type_.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        model_options = {}
        if device == "cpu":
            model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")
        return comfy.sd.load_clip(ckpt_paths=[clip_path], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=clip_type, model_options=model_options),  # the CLIP itself, unlike the loaders above

    model, = _last_clip.get_or_load((clip_path, type_, device), load)
    return model


def _load_vae(vae_name):
    vae, = _last_vae.get_or_load(vae_name, lambda: VAELoader().load_vae(vae_name))
    return vae


def _load_vae_audio(vae_audio_name):
    vae_audio, = _last_vae_audio.get_or_load(vae_audio_name, lambda: VAELoader().load_vae(vae_audio_name))
    return vae_audio


class GibbyContextLoader(io.ComfyNode):
    """Load models from files and build a complete CONTEXT object."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Gibby_ContextLoader",
            display_name="Context Loader",
            category="gibby/context",
            description=(
                "Loads models via a mode selector (diffusion model parts or checkpoint), "
                "carries sampling parameters, and fills in derived values. With a context "
                "connected, its values carry through for anything the node does not load "
                "or receive connected; the widget values override its matching fields."
            ),
            inputs=[
                _CONTEXT_TYPE.Input("context", optional=True, tooltip=(
                    "Base context; its values carry through for anything this node "
                    "does not load or receive connected"
                )),
                io.DynamicCombo.Input(
                    "mode",
                    options=[
                        io.DynamicCombo.Option("diffusion_model", [
                            io.Combo.Input("unet_name", options=["None"] + folder_paths.get_filename_list("diffusion_models"), default="None"),
                            io.Combo.Input("weight_dtype", options=["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], default="default", advanced=True),
                            io.Combo.Input("clip_name", options=["None"] + folder_paths.get_filename_list("text_encoders"), default="None"),
                            io.Combo.Input("type", options=CLIPLoader.INPUT_TYPES()["required"]["type"][0], default="krea2"),
                            io.Combo.Input("device", options=["default", "cpu"], default="default", advanced=True),
                            io.Combo.Input("vae_name", options=["None"] + VAELoader.vae_list(VAELoader), default="None"),
                            io.Combo.Input("vae_audio_name", options=["None"] + VAELoader.vae_list(VAELoader), optional=True, default="None"),
                        ]),
                        io.DynamicCombo.Option("checkpoint", [
                            io.Combo.Input("ckpt_name", options=["None"] + folder_paths.get_filename_list("checkpoints"), default="None"),
                        ]),
                    ],
                ),
                # Shared by both modes.
                io.Latent.Input("latent", optional=True),
                io.Image.Input("image", optional=True),
                io.Mask.Input("mask", optional=True),
                io.Audio.Input("audio", optional=True),
                io.Mask.Input("mask_audio", optional=True),
                _lora_stack.Input("lora_stack", optional=True),
                io.Int.Input("steps", default=20, advanced=True),
                io.Int.Input("step_refiner", default=0, advanced=True),
                io.Float.Input("cfg", default=1.0, advanced=True),
                io.Combo.Input("sampler", options=comfy.samplers.KSampler.SAMPLERS, default="euler", advanced=True),
                io.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS, default="normal", advanced=True),
                io.Int.Input("width", default=0, advanced=True),
                io.Int.Input("height", default=0, advanced=True),
                io.String.Input("positive_prompt", multiline=True, default="", advanced=True),
                io.String.Input("negative_prompt", multiline=True, default="", advanced=True),
                io.Boolean.Input("ck_attn", default=True, tooltip="Comfy Kitchen int8 attention; falls back to pytorch attention if unavailable."),
            ],
            outputs=[
                _CONTEXT_TYPE.Output(display_name="context"),
                io.Model.Output(display_name="model"),
                io.Clip.Output(display_name="clip"),
                io.Vae.Output(display_name="vae"),
                io.Vae.Output(display_name="vae_audio"),
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, context=None, mode=None, latent=None, image=None, mask=None,
                audio=None, mask_audio=None, lora_stack=None, steps=20, step_refiner=0,
                cfg=1.0, sampler="euler", scheduler="normal", width=0, height=0, positive_prompt="", negative_prompt="", ck_attn=True) -> io.NodeOutput:
        if mode is None:
            raise ValueError("Context Loader requires a 'mode' input.")

        # Start from the base context so anything this node does not load or
        # receive carries through (image, prompts, crop_info and the like).
        ctx = ctx_from(context)
        for key in ("model", "clip", "vae", "vae_audio", "positive", "negative", "latent",
                    "image", "mask", "audio", "mask_audio", "model_name", "lora_stack"):
            ctx.setdefault(key, None)

        model = clip = vae = vae_audio = None
        model_name = None

        selected = mode.get("mode")
        if selected == "checkpoint":
            ckpt_name = mode.get("ckpt_name")
            if ckpt_name and ckpt_name != "None":
                model, clip, vae = _load_checkpoint(ckpt_name)
                model_name = ckpt_name
        elif selected == "diffusion_model":
            unet_name = mode.get("unet_name")
            weight_dtype = mode.get("weight_dtype", "default")
            type_ = mode.get("type", "stable_diffusion")
            device = mode.get("device", "default")
            vae_name = mode.get("vae_name")
            vae_audio_name = mode.get("vae_audio_name")

            if unet_name and unet_name != "None":
                model_name = unet_name
                model = _load_unet(unet_name, weight_dtype)

            # The clip: native ComfyUI loading, one text encoder like Clip Loader.
            clip_name = mode.get("clip_name")
            if clip_name and clip_name != "None":
                clip = _load_clip(folder_paths.get_full_path_or_raise("text_encoders", clip_name), type_, device)

            if vae_name and vae_name != "None":
                vae = _load_vae(vae_name)
            if vae_audio_name and vae_audio_name != "None":
                vae_audio = _load_vae_audio(vae_audio_name)
        else:
            raise ValueError(f"Unknown mode '{selected}'.")

        # Loaded models and connected values override the context's.
        if model is not None:
            ctx["model"] = model
        if clip is not None:
            ctx["clip"] = clip
        if vae is not None:
            ctx["vae"] = vae
        if vae_audio is not None:
            ctx["vae_audio"] = vae_audio
        if model_name is not None:
            ctx["model_name"] = model_name
        if latent is not None:
            ctx_set_latent(ctx, latent)
        if image is not None:
            # A new image invalidates the latent and cached sample built from
            # the context's old one.
            ctx_set_image(ctx, image)
        if mask is not None:
            ctx["mask"] = mask
        if audio is not None:
            ctx["audio"] = audio
        if mask_audio is not None:
            ctx["mask_audio"] = mask_audio
        if lora_stack is not None:
            ctx["lora_stack"] = lora_stack

        # Widget values override the matching context fields.
        ctx["steps"] = steps
        ctx["step_refiner"] = step_refiner
        ctx["cfg"] = cfg
        ctx["sampler"] = sampler
        ctx["scheduler"] = scheduler
        ctx["width"] = width
        ctx["height"] = height
        ctx["positive_prompt"] = positive_prompt
        ctx["negative_prompt"] = negative_prompt

        # Apply LoRAs from a directly-connected lora_stack to the context's model/clip.
        for name, sm, sc in iter_lora_stack(lora_stack):
            ctx["model"], ctx["clip"] = _apply_lora(ctx["model"], ctx["clip"], name, sm, sc)

        if ctx["model"] is not None and ck_attn and ModelAttentionBackend is not None:
            ctx["model"], = ModelAttentionBackend.execute(ctx["model"], "comfy kitchen attention")

        # Always fill in derived values (conditioning from prompts, latent from image/width-height).
        GibbyContext.evaluate(ctx)

        return io.NodeOutput(
            ctx,  # context
            ctx["model"],  # model
            ctx["clip"],  # clip
            ctx["vae"],  # vae
            ctx["vae_audio"],  # vae_audio
            ctx["positive"],  # positive
            ctx["negative"],  # negative
            ctx["latent"],  # latent
        )

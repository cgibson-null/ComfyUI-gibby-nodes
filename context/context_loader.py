"""
Context Loader node
-------------------
A self-contained context source with a mode selector for the loading category:

- diffusion model: separate unet/clip/vae selectors, each left on "None" to skip.
  The clips row count (1-4) works like Clip Loader - Dual/Triple/Quad.
- checkpoint: loads a full checkpoint like Load Checkpoint.

Carries the sampling parameters as widgets, and always fills in derived values on demand:

- positive/negative conditioning encoded from the prompt widgets via clip
- latent from image via VAE encode if present (with optional noise mask)
- empty latent when width/height are given with nothing to sample from

Use Context to merge or override individual values of this context downstream.
"""

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
    print("Gibby Context Loader: ModelAttentionBackend is missing - ck_attn is disabled (update ComfyUI)")

from ..lora_loader import _apply_lora
from . import _CONTEXT_TYPE, _lora_stack, GibbyContext

def _clip_name_inputs(count):
    clip_options = ["None"] + folder_paths.get_filename_list("text_encoders")
    return [io.Combo.Input(f"clip_name{i}", options=clip_options, default="None") for i in range(1, count + 1)]


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
                "carries sampling parameters, and fills in derived values."
            ),
            inputs=[
                io.DynamicCombo.Input(
                    "mode",
                    options=[
                        io.DynamicCombo.Option("diffusion_model", [
                            io.Combo.Input("unet_name", options=["None"] + folder_paths.get_filename_list("diffusion_models"), default="None"),
                            io.Combo.Input("weight_dtype", options=["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], default="default", advanced=True),
                            io.DynamicCombo.Input(
                                "clip_count",
                                display_name="clips",
                                options=[
                                    io.DynamicCombo.Option("1", _clip_name_inputs(1)),
                                    io.DynamicCombo.Option("2", _clip_name_inputs(2)),
                                    io.DynamicCombo.Option("3", _clip_name_inputs(3)),
                                    io.DynamicCombo.Option("4", _clip_name_inputs(4)),
                                ],
                            ),
                            io.Combo.Input("type", options=CLIPLoader.INPUT_TYPES()["required"]["type"][0], default="stable_diffusion"),
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
    def execute(cls, mode=None, latent=None, image=None, mask=None,
                audio=None, mask_audio=None, lora_stack=None, steps=20, step_refiner=0,
                cfg=1.0, sampler="euler", scheduler="normal", width=0, height=0, positive_prompt="", negative_prompt="", ck_attn=True) -> io.NodeOutput:
        if mode is None:
            raise ValueError("Context Loader requires a 'mode' input.")

        model = clip = vae = vae_audio = None
        model_name = None

        selected = mode.get("mode")
        if selected == "checkpoint":
            ckpt_name = mode.get("ckpt_name")
            if ckpt_name and ckpt_name != "None":
                model, clip, vae = CheckpointLoaderSimple().load_checkpoint(ckpt_name)
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
                model, = UNETLoader().load_unet(unet_name, weight_dtype)

            # Clip(s): native ComfyUI loading; multiple paths like Clip Loader - Dual/Triple/Quad.
            clips = mode.get("clip_count", {}) or {}
            count = int(clips.get("clip_count", 1))
            clip_paths = []
            for i in range(1, count + 1):
                name = clips.get(f"clip_name{i}")
                if name and name != "None":
                    clip_paths.append(folder_paths.get_full_path_or_raise("text_encoders", name))
            if clip_paths:
                clip_type = getattr(comfy.sd.CLIPType, type_.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
                model_options = {}
                if device == "cpu":
                    model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")
                clip = comfy.sd.load_clip(ckpt_paths=clip_paths, embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=clip_type, model_options=model_options)

            if vae_name and vae_name != "None":
                vae, = VAELoader().load_vae(vae_name)
            if vae_audio_name and vae_audio_name != "None":
                vae_audio, = VAELoader().load_vae(vae_audio_name)
        else:
            raise ValueError(f"Unknown mode '{selected}'.")

        ctx = {
            "model": model,
            "clip": clip,
            "vae": vae,
            "vae_audio": vae_audio,
            # Conditioning is generated below from the prompt widgets.
            "positive": None,
            "negative": None,
            "latent": latent,
            "image": image,
            "mask": mask,
            "audio": audio,
            "mask_audio": mask_audio,
            "model_name": model_name,
            "lora_stack": lora_stack,
            "steps": steps,
            "step_refiner": step_refiner,
            "cfg": cfg,
            "sampler": sampler,
            "scheduler": scheduler,
            "width": width,
            "height": height,
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt,
        }

        # Apply LoRAs from a directly-connected lora_stack to the freshly-loaded model/clip.
        if isinstance(lora_stack, list):
            for item in lora_stack:
                if not item or len(item) < 3 or item[0] == "None":
                    continue
                name, sm, sc = item[0], item[1], item[2]
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

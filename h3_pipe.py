"""H3 Pipe nodes.

- H3PipeCreate: stores raw reference media + config in a reusable h3_pipe dict.
- H3PipeApply: clip + vaes + h3_pipe → conditioning + latent (all VAE encoding happens here).
"""
import math
import torchaudio

import nodes
import node_helpers
from comfy_api.latest import io
from comfy_extras.nodes_minimax_h3 import (
    _empty_av_latent, _resize, adapt_canvas,
    CANVAS_MULTIPLE, REF_IMAGE_SHORT_EDGE, FPS,
)
from .context import _CONTEXT_TYPE


def _encode_ref_audio(audio_vae, audio):
    waveform = audio["waveform"]
    sr = audio["sample_rate"]
    vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
    if sr != vae_sr:
        waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
    z = audio_vae.encode(waveform[:1].movedim(1, -1))
    return z, z.shape[-1]


def _build_ref_blocks_and_items(vae, audio_vae, frame_count, width, height, ref_image_size,
                                first_frame, last_frame, also_ref_first_frame,
                                ref_images, ref_videos, ref_video_audios, ref_audios):
    """Shared helper: encode all refs into blocks + tokenization hints."""
    keyframe_images = []
    if first_frame is not None:
        img = _resize(first_frame[:1], width, height, "center")
        keyframe_images.append(img)
    if last_frame is not None:
        img = _resize(last_frame[:1], width, height, "center")
        keyframe_images.append(img)

    ref_items, ref_blocks = [], []

    for img in (ref_images or {}).values():
        if img is None:
            continue
        h, w = img.shape[1], img.shape[2]
        scale = (min(1.0, math.sqrt((width * height) / (w * h)))
                 if ref_image_size == "match"
                 else min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h)))
        tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        resized = _resize(img[:1], tw, th, "disabled")
        ref_items.append({"type": "image", "data": resized})
        ref_blocks.append({"kind": "image", "latent_h": th // 16,
                           "latent_w": tw // 16, "latent": vae.encode(resized)})

    if also_ref_first_frame and first_frame is not None:
        ref_items.append({"type": "image", "data": keyframe_images[0]})
        ref_blocks.append({"kind": "image", "latent_h": height // 16,
                           "latent_w": width // 16, "latent": vae.encode(keyframe_images[0])})

    ref_video_audios = ref_video_audios or {}
    for name, video_frames in (ref_videos or {}).items():
        if video_frames is None:
            continue
        soundtrack = ref_video_audios.get("ref_video_audio_" + name.rsplit("_", 1)[-1])
        vh, vw = video_frames.shape[1], video_frames.shape[2]
        cw, ch = adapt_canvas(vw, vh)
        if vw * vh < cw * ch:
            cw = max(CANVAS_MULTIPLE, round(vw / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            ch = max(CANVAS_MULTIPLE, round(vh / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        frames = _resize(video_frames, cw, ch, "disabled")
        if frames.shape[0] > frame_count:
            frames = frames[:frame_count]
        n = frames.shape[0]
        if n < 5:
            raise ValueError("MiniMax H3 reference videos need at least 5 frames")
        while n % 17 != 5:
            n -= 1
        frames = frames[:n]
        audio_latent, ref_audio_t = None, 0
        if soundtrack is not None:
            audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, soundtrack)
            ref_items.append({"type": "audio"})
        sample_idx = list(range(0, frames.shape[0], FPS // 2))
        ref_items.append({"type": "video", "data": frames[sample_idx],
                          "timestamps": [i / 2.0 for i in range(len(sample_idx))]})
        video_latent = vae.encode(frames)
        ref_blocks.append({"kind": "video_audio" if ref_audio_t else "video",
                           "latent_t": video_latent.shape[2], "latent_h": ch // 16,
                           "latent_w": cw // 16, "ref_audio_t": ref_audio_t,
                           "latent": video_latent, "audio_latent": audio_latent})

    for audio in (ref_audios or {}).values():
        if audio is None:
            continue
        audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, audio)
        ref_items.append({"type": "audio"})
        ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t,
                           "audio_latent": audio_latent})

    return keyframe_images, ref_items, ref_blocks


def _apply_pipe_to_conditioning(clip, vae, audio_vae, pipe, target_width=None, target_height=None):
    """Shared helper: take clip + vaes + h3_pipe → conditioning + latent.

    Encodes all refs from raw media stored in pipe at apply time.
    If target_width/target_height provided, keyframes are resized to match.
    """
    prompt = pipe["prompt"]
    width = pipe["width"]
    height = pipe["height"]
    length = pipe["length"]

    # Use target resolution for keyframes if provided, otherwise use pipe resolution
    kf_width = target_width if target_width is not None else width
    kf_height = target_height if target_height is not None else height

    # Compute frame count
    n = length
    while n % 17 != 5:
        n -= 1
    if n < 5:
        raise ValueError("MiniMax H3 requires at least 5 frames")
    frame_count = n

    first_frame = pipe.get("first_frame")
    last_frame = pipe.get("last_frame")
    ref_image_size = pipe.get("ref_image_size", "match")
    also_ref_first_frame = pipe.get("also_ref_first_frame", False)

    # Build refs from raw media stored in pipe (encoding happens here)
    keyframe_images, ref_items, ref_blocks = _build_ref_blocks_and_items(
        vae, audio_vae, frame_count, width, height, ref_image_size,
        first_frame, last_frame, also_ref_first_frame,
        pipe.get("ref_images", {}), pipe.get("ref_videos", {}),
        pipe.get("ref_video_audios", {}), pipe.get("ref_audios", {}))

    # Tokenize prompt with refs or keyframe images
    if ref_items:
        tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
    else:
        kf_imgs = []
        if first_frame is not None:
            kf_imgs.append(_resize(first_frame[:1], kf_width, kf_height, "center"))
        if last_frame is not None:
            kf_imgs.append(_resize(last_frame[:1], kf_width, kf_height, "center"))
        tokens = clip.tokenize(prompt, images=kf_imgs)

    cond = clip.encode_from_tokens_scheduled(tokens)

    # Build keyframes with encoded latents (using target resolution if provided)
    keyframes = []
    if first_frame is not None:
        img = _resize(first_frame[:1], kf_width, kf_height, "center")
        keyframes.append({"resolved_frame_index": 0, "image": img})
    if last_frame is not None:
        img = _resize(last_frame[:1], kf_width, kf_height, "center")
        keyframes.append({"resolved_frame_index": frame_count - 1, "image": img})

    # Attach metadata to conditioning
    values = {}
    if keyframes:
        for kf in keyframes:
            kf["latent"] = vae.encode(kf.pop("image"))
        values["minimax_keyframes"] = keyframes
        values["minimax_frame_count"] = frame_count
    if ref_blocks:
        values["minimax_refs"] = ref_blocks
    if values:
        cond = node_helpers.conditioning_set_values(cond, values)

    latent, _ = _empty_av_latent(width, height, length)
    return cond, latent


class H3PipeCreate(io.ComfyNode):
    """Create or override an h3_pipe carrying raw reference media and configuration.

    Stores unencoded references (images/videos/audios) plus config in a reusable dict.
    All VAE encoding happens later in H3 Pipe Apply.

    If an existing h3_pipe is provided, its values are overridden by any connected inputs.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3PipeCreate",
            display_name="H3 Pipe Create",
            category="gibby",
            description=("Creates an h3_pipe carrying raw references and config. "
                         "Connect optional existing pipe to override its values."),
            inputs=[
                io.Dict.Input("h3_pipe", optional=True,
                              tooltip="Optional existing pipe to override with new values"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True, optional=True),
                io.Int.Input("width", default=1344, min=32, max=nodes.MAX_RESOLUTION, step=32, optional=True),
                io.Int.Input("height", default=768, min=32, max=nodes.MAX_RESOLUTION, step=32, optional=True),
                io.Int.Input("length", default=124, min=5, max=3600, step=17,
                             tooltip="Frames at 24 fps; snaps to 17k+5.", optional=True),
                io.Combo.Input("ref_image_size", options=["match", "max"], default="match", optional=True),
                io.Image.Input("first_frame", optional=True),
                io.Image.Input("last_frame", optional=True),
                io.Autogrow.Input("ref_images", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_image"), prefix="ref_image_", min=0, max=9)),
                io.Autogrow.Input("ref_videos", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_video"), prefix="ref_video_", min=0, max=3)),
                io.Autogrow.Input("ref_video_audios", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_video_audio"), prefix="ref_video_audio_", min=0, max=3)),
                io.Autogrow.Input("ref_audios", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_audio"), prefix="ref_audio_", min=0, max=3)),
                io.Boolean.Input("also_ref_first_frame", default=False,
                                 tooltip="Also expose first_frame as the next <Picture N> reference.", optional=True),
            ],
            outputs=[io.Dict.Output(display_name="h3_pipe")],
        )

    @classmethod
    def execute(cls, h3_pipe=None, prompt=None, width=None, height=None, length=None,
                ref_image_size=None, first_frame=None, last_frame=None,
                ref_images=None, ref_videos=None, ref_video_audios=None,
                ref_audios=None, also_ref_first_frame=None):
        if h3_pipe is None:
            pipe = {}
        else:
            pipe = dict(h3_pipe)

        # Override config fields only when explicitly provided (non-default/non-empty)
        if not "prompt" in pipe or prompt is not None and str(prompt).strip():
            pipe["prompt"] = prompt
        if not "width" in pipe or width is not None and width != 1344:
            pipe["width"] = width
        if not "height" in pipe or height is not None and height != 768:
            pipe["height"] = height
        if not "length" in pipe is None or length is not None and length != 124:
            pipe["length"] = length
        if ref_image_size is not None and ref_image_size != "match":
            pipe["ref_image_size"] = ref_image_size
        if also_ref_first_frame is not None and also_ref_first_frame != False:
            pipe["also_ref_first_frame"] = also_ref_first_frame

        # Override keyframe images (only if connected; keep existing otherwise)
        if first_frame is not None:
            pipe["first_frame"] = first_frame

        if last_frame is not None:
            pipe["last_frame"] = last_frame

        # Override reference media (only if connected; store raw tensors, no encoding)
        if ref_images is not None and any(v is not None for v in ref_images.values()):
            pipe["ref_images"] = {k: v for k, v in ref_images.items() if v is not None}

        if ref_videos is not None and any(v is not None for v in ref_videos.values()):
            pipe["ref_videos"] = {k: v for k, v in ref_videos.items() if v is not None}

        if ref_video_audios is not None and any(v is not None for v in ref_video_audios.values()):
            pipe["ref_video_audios"] = {k: v for k, v in ref_video_audios.items() if v is not None}

        if ref_audios is not None and any(v is not None for v in ref_audios.values()):
            pipe["ref_audios"] = {k: v for k, v in ref_audios.items() if v is not None}

        return io.NodeOutput(pipe)


class H3PipeApply(io.ComfyNode):
    """Apply an h3_pipe with clip and vaes to produce conditioning + latent.

    All VAE encoding of refs happens here from raw media stored in the pipe.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3PipeApply",
            display_name="H3 Pipe Apply",
            category="gibby",
            description=("Applies an h3_pipe with clip and vaes to create conditioning + latent. "
                         "Same output as MiniMax H3 Hybrid Cond."),
            inputs=[
                _CONTEXT_TYPE.Input("context", optional=True),
                io.Dict.Input("h3_pipe"),
                io.Clip.Input("clip", optional=True),
                io.Vae.Input("vae", optional=True),
                io.Vae.Input("audio_vae", optional=True),
                io.Int.Input("target_width", default=0, min=0, max=nodes.MAX_RESOLUTION, step=32,
                             optional=True,
                             tooltip="Optional target width in pixels for keyframes (for dual-pass upscale workflows)"),
                io.Int.Input("target_height", default=0, min=0, max=nodes.MAX_RESOLUTION, step=32,
                             optional=True,
                             tooltip="Optional target height in pixels for keyframes (for dual-pass upscale workflows)"),
            ],
            outputs=[
                _CONTEXT_TYPE.Output(display_name="context"),
                io.Conditioning.Output(display_name="positive"),
                io.Latent.Output(),
            ],
        )

    @classmethod
    def execute(cls, context=None, clip=None, vae=None, audio_vae=None, h3_pipe=None, target_width=0, target_height=0):
        ctx = dict(context) if isinstance(context, dict) else {}

        # Directly-connected clip/vae override context values
        clip = clip if clip is not None else ctx.get("clip")
        vae = vae if vae is not None else ctx.get("vae")
        audio_vae = audio_vae if audio_vae is not None else ctx.get("audio_vae")
        if clip is not None:
            ctx["clip"] = clip
        if vae is not None:
            ctx["vae"] = vae
        if audio_vae is not None:
            ctx["audio_vae"] = audio_vae

        tw = target_width if target_width > 0 else None
        th = target_height if target_height > 0 else None
        cond, latent = _apply_pipe_to_conditioning(clip, vae, audio_vae, h3_pipe, tw, th)

        # Store results back in context
        ctx["positive"] = cond
        ctx["latent"] = latent

        return io.NodeOutput(ctx, cond, latent)

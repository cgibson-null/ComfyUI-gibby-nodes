"""H3 Pipe nodes.

- H3PipeCreate: stores raw reference media + config in a reusable h3_pipe dict.
- H3PipeApply: clip + vaes + h3_pipe → conditioning + latent (all VAE encoding happens here).

The h3_pipe is a media pipe - ref_images, keyframes with their position spec,
ref_videos (+ soundtracks), ref_audios - plus the h3 params (prompt, width,
height, length, ref_image_size). The same dict also feeds Reference Latent
(Context) and Generate; the shape lives in media_pipe.py.
"""
import logging
import math
import torchaudio

import nodes
import node_helpers
from comfy_api.latest import io
from comfy_extras.nodes_minimax_h3 import (
    _empty_av_latent, _resize, adapt_canvas,
    CANVAS_MULTIPLE, REF_IMAGE_SHORT_EDGE, FPS,
)
from .context import _CONTEXT_TYPE, ctx_from
from .media_pipe import slot_order


def _snap_frames(n):
    """Frame count snapped down to MiniMax H3's 17k+5 requirement."""
    n = int(n)
    while n % 17 != 5:
        n -= 1
    if n < 5:
        raise ValueError("MiniMax H3 requires at least 5 frames")
    return n


def _encode_ref_audio(audio_vae, audio):
    waveform = audio["waveform"]
    sr = audio["sample_rate"]
    vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
    if sr != vae_sr:
        waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
    z = audio_vae.encode(waveform[:1].movedim(1, -1))
    return z, z.shape[-1]


def _parse_positions(indices):
    """Comma-separated position spec: an integer is a frame index (negative
    counts from the end), a decimal is a percentage of the video."""
    positions = []
    for token in str(indices).split(","):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        positions.append(value if "." in token else int(value))
    return positions


def _is_first_pos(p):
    return (isinstance(p, int) and p == 0) or (isinstance(p, float) and p == 0.0)


def _is_last_pos(p):
    return (isinstance(p, int) and p == -1) or (isinstance(p, float) and p == 1.0)


def _assemble_keyframes(first_frame, last_frame, keyframes, indices, even_distribution, loop):
    """first/last frame + keyframe batch → (images, position spec).

    even_distribution spreads the full batch (first/last frame included)
    evenly from 0 to the last frame - with loop, to just before it, so the
    last frame stays free for the loop anchor; otherwise the parsed indices
    position the keyframe batch between the first/last frame. loop appends a
    duplicate of the first image at the last frame.
    """
    batch = [keyframes[i:i + 1] for i in range(keyframes.shape[0])] if keyframes is not None else []

    parsed = []
    if not even_distribution:
        parsed = _parse_positions(indices)
        mid = min(len(parsed), len(batch))
        if len(parsed) != len(batch):
            logging.warning("WARNING: H3 Pipe Create: {} positions for {} keyframes - pairing the first {} in order".format(
                len(parsed), len(batch), mid))
        parsed, batch = parsed[:mid], batch[:mid]
        if first_frame is not None and any(_is_first_pos(p) for p in parsed):
            logging.warning("WARNING: H3 Pipe Create: first frame and a keyframe both anchor frame 0 - keeping both")
        if last_frame is not None and any(_is_last_pos(p) for p in parsed):
            logging.warning("WARNING: H3 Pipe Create: last frame and a keyframe both anchor the last frame - keeping both")

    images = []
    if first_frame is not None:
        images.append(first_frame[:1])
    images.extend(batch)
    if last_frame is not None:
        images.append(last_frame[:1])
    if not images:
        return None, None

    if even_distribution:
        n = len(images)
        if loop:
            positions = [i / n for i in range(n)]
        else:
            positions = [0] + [i / (n - 1) for i in range(1, n - 1)] + ([-1] if n > 1 else [])
    else:
        positions = ([0] if first_frame is not None else []) + parsed + ([-1] if last_frame is not None else [])

    if loop:
        if not even_distribution and any(_is_last_pos(p) for p in parsed):
            logging.warning("WARNING: H3 Pipe Create: loop anchors the first keyframe at the last frame, which already has a keyframe - keeping both")
        images.append(images[0])
        positions.append(-1)

    return images, positions


def _resolve_keyframe_index(pos, frame_count):
    """Position spec → frame index: an integer as-is (negative from the
    end), a float as a percentage of the video."""
    if isinstance(pos, float):
        pos = min(1.0, max(0.0, pos))
        return int(round(pos * (frame_count - 1)))
    idx = pos if pos >= 0 else frame_count + pos
    if idx < 0 or idx >= frame_count:
        raise ValueError("keyframe index {} is outside the video's {} frames".format(pos, frame_count))
    return idx


def _build_ref_blocks_and_items(vae, audio_vae, frame_count, width, height, ref_image_size,
                                ref_images, ref_videos, ref_video_audios, ref_audios):
    """Shared helper: encode all refs into blocks + tokenization hints"""
    ref_items, ref_blocks = [], []

    for name in sorted((ref_images or {}), key=slot_order):
        img = ref_images[name]
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

    ref_video_audios = ref_video_audios or {}
    for name in sorted((ref_videos or {}), key=slot_order):
        video_frames = ref_videos[name]
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
        if frames.shape[0] < 5:
            raise ValueError("MiniMax H3 reference videos need at least 5 frames")
        frames = frames[:_snap_frames(frames.shape[0])]
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

    for name in sorted((ref_audios or {}), key=slot_order):
        audio = ref_audios[name]
        if audio is None:
            continue
        audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, audio)
        ref_items.append({"type": "audio"})
        ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t,
                           "audio_latent": audio_latent})

    return ref_items, ref_blocks


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
    frame_count = _snap_frames(length)
    ref_image_size = pipe.get("ref_image_size", "match")

    # Build refs from raw media stored in pipe (encoding happens here)
    ref_items, ref_blocks = _build_ref_blocks_and_items(
        vae, audio_vae, frame_count, width, height, ref_image_size,
        pipe.get("ref_images", {}), pipe.get("ref_videos", {}),
        pipe.get("ref_video_audios", {}), pipe.get("ref_audios", {}))

    # Resolve keyframe positions and resize to the target canvas
    keyframes = []
    for img, pos in zip(pipe.get("keyframes") or [], pipe.get("keyframe_positions") or []):
        idx = _resolve_keyframe_index(pos, frame_count)
        keyframes.append((idx, _resize(img, kf_width, kf_height, "center")))
    keyframes.sort(key=lambda pair: pair[0])

    # Tokenize prompt with refs, or with the start/end keyframes as reference images
    kf_imgs = []
    if ref_items:
        tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
    else:
        for idx, img in keyframes:
            if idx in (0, frame_count - 1):
                kf_imgs.append(img)
        tokens = clip.tokenize(prompt, images=kf_imgs)

    cond = clip.encode_from_tokens_scheduled(tokens)

    # Attach keyframe latents and refs to conditioning
    values = {}
    if keyframes:
        values["minimax_keyframes"] = [{"resolved_frame_index": idx, "latent": vae.encode(img)}
                                       for idx, img in keyframes]
        values["minimax_frame_count"] = frame_count
    if ref_blocks:
        values["minimax_refs"] = ref_blocks
    if values:
        cond = node_helpers.conditioning_set_values(cond, values)

    latent, _ = _empty_av_latent(width, height, length)
    return cond, latent, ref_items, kf_imgs


class H3PipeCreate(io.ComfyNode):
    """Create or override an h3_pipe carrying raw reference media and configuration.

    Stores unencoded references (images/keyframes/videos/audios) plus config in a reusable dict.
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
                io.Image.Input("keyframes", optional=True,
                               tooltip="Keyframe batch - each frame anchors the video at a position from 'indices' (or evenly distributed). first/last frame above join the batch at its start/end."),
                io.String.Input("indices", default="", optional=True,
                                tooltip="Comma-separated keyframe positions: an integer is a frame index (negative counts from the end, -1 = last), a decimal is a percentage of the video (0.5 = middle). Ignored with even_distribution."),
                io.Boolean.Input("even_distribution", default=False, optional=True,
                                 tooltip="Spread the full keyframe batch (first/last frame included) evenly from 0 to the last frame."),
                io.Boolean.Input("loop", default=False, optional=True,
                                 tooltip="Also anchor a duplicate of the first keyframe at the last frame (for looping videos)."),
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
            ],
            outputs=[io.Dict.Output(display_name="h3_pipe")],
        )

    @classmethod
    def execute(cls, h3_pipe=None, prompt=None, width=None, height=None, length=None,
                ref_image_size=None, first_frame=None, last_frame=None,
                keyframes=None, indices=None, even_distribution=None, loop=None,
                ref_images=None, ref_videos=None, ref_video_audios=None,
                ref_audios=None):
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
        if not "length" in pipe or length is not None and length != 124:
            pipe["length"] = length
        if ref_image_size is not None and ref_image_size != "match":
            pipe["ref_image_size"] = ref_image_size

        # Rebuild the keyframes section when any keyframe input is provided
        if (first_frame is not None or last_frame is not None or keyframes is not None
                or (indices is not None and str(indices).strip())
                or (even_distribution is not None and even_distribution)
                or (loop is not None and loop)):
            kf_imgs, positions = _assemble_keyframes(first_frame, last_frame, keyframes,
                                                      indices or "", even_distribution, loop)
            if kf_imgs:
                pipe["keyframes"] = kf_imgs
                pipe["keyframe_positions"] = positions
            else:
                pipe.pop("keyframes", None)
                pipe.pop("keyframe_positions", None)

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
    The drop_* toggles prune the pipe first; without clip/vae (wired or from
    the context) the node just outputs the pruned h3_pipe.
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
                io.Boolean.Input("drop_ref_images", default=False, optional=True,
                                 tooltip="Drop the pipe's reference images before applying"),
                io.Boolean.Input("drop_keyframes", default=False, optional=True,
                                 tooltip="Drop the pipe's keyframes before applying"),
                io.Boolean.Input("drop_ref_videos", default=False, optional=True,
                                 tooltip="Drop the pipe's reference videos before applying"),
                io.Boolean.Input("drop_ref_video_audios", default=False, optional=True,
                                 tooltip="Drop the pipe's reference video soundtracks before applying"),
                io.Boolean.Input("drop_ref_audios", default=False, optional=True,
                                 tooltip="Drop the pipe's reference audios before applying"),
            ],
            outputs=[
                _CONTEXT_TYPE.Output(display_name="context"),
                io.Conditioning.Output(display_name="positive"),
                io.Latent.Output(),
                io.Dict.Output(display_name="h3_pipe"),
            ],
        )

    @classmethod
    def execute(cls, context=None, clip=None, vae=None, audio_vae=None, h3_pipe=None,
                target_width=0, target_height=0, drop_ref_images=False, drop_keyframes=False,
                drop_ref_videos=False, drop_ref_video_audios=False, drop_ref_audios=False):
        ctx = ctx_from(context)

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

        # Prune the pipe per the drop toggles
        pipe = dict(h3_pipe)
        if drop_ref_images:
            pipe.pop("ref_images", None)
        if drop_keyframes:
            pipe.pop("keyframes", None)
            pipe.pop("keyframe_positions", None)
        if drop_ref_videos:
            pipe.pop("ref_videos", None)
        if drop_ref_video_audios:
            pipe.pop("ref_video_audios", None)
        if drop_ref_audios:
            pipe.pop("ref_audios", None)

        # Without clip/vae there is nothing to encode - just hand out the pruned pipe
        if clip is None or vae is None:
            return io.NodeOutput(ctx, None, None, pipe)

        tw = target_width if target_width > 0 else None
        th = target_height if target_height > 0 else None
        cond, latent, ref_items, kf_imgs = _apply_pipe_to_conditioning(clip, vae, audio_vae, pipe, tw, th)

        # Store results back in context
        ctx["positive"] = cond
        # The raw prompt the positive was built from, so KSampler (Context) can
        # detect [before:after:step] travel groups in it and re-encode per step.
        ctx["positive_prompt"] = pipe.get("prompt") or ""
        ctx["latent"] = latent
        # Bare-minimum h3 re-condition data: the pre-processed ref items / keyframe
        # images, so a prompt re-encode reuses them instead of re-deriving the media.
        ctx["h3_ref_items"] = ref_items
        ctx["h3_kf_imgs"] = kf_imgs

        return io.NodeOutput(ctx, cond, latent, pipe)

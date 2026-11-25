"""H3 Timeline (Context): extend, prepend, loop or insert into an existing
MiniMax H3 clip by denoising only the new span.

The source clip rides on inside the target latent with its denoise mask at 0.
The sampler composites `out * mask + latent_image * (1 - mask)`
(comfy/samplers.py, KSamplerX0Inpaint), so the kept frames come back bit for
bit and the model reads them as its own context at their real time coordinates
- not as keyframes to reproduce or a reference to imitate.

Everything is quantised to the video VAE's grid: it encodes 17-frame chunks
into 5 latent steps, and the 1,4,4,4,4 frame pattern restarts every 5 steps,
so a block may only be placed where its own cycle phase matches the position
it is given, and a clip's latent must stay 5g+2 steps long. That fixes the
gap length per mode:

    extend / prepend / insert   gap = 17x frames    (5x steps)
    loop                        gap = 17x+9 frames  (5x+3 steps)

because the loop's two kept blocks each take 5h+2 steps of the total.

The audio stream follows the picture at the same boundaries using core's own
mapping, and is kept wherever the picture is kept.

Wiring: build the prompt for the NEW span with the usual conditioning (H3 Pipe
Create / Apply), sample the latent this node puts on the context, then decode.
trim_head / trim_tail say how many frames of the result are kept source.
"""
import logging

import torch

import comfy.audio
import comfy.nested_tensor
from comfy_api.latest import io
from comfy.ldm.minimax.model import FRAME_PER_TOKEN
from comfy_extras.nodes_minimax_h3 import AUDIO_LATENT_FPS, FPS
from comfy_extras.nodes_post_processing import ColorTransfer

from ..context import _CONTEXT_TYPE, ctx_from, ctx_set_image, ctx_set_latent, encode_image
from ..ksampler_options.crop_inpaint import _KSAMPLER_OPTIONS_TYPE

# Its own socket rather than a plain DICT: the core builds the drag-out list from
# every node that takes the dragged type as a required input, so a DICT timeline
# would offer H3 Pipe Apply, and a DICT pipe would offer Timeline Assemble.
_H3_TIMELINE_TYPE = io.Custom("GIBBY_H3_TIMELINE")

RUN_MULTIPLE = 17   # pixel frames the video VAE encodes per chunk
STEP_MULTIPLE = 5   # latent steps one chunk produces


def _pixel_starts(steps):
    """Pixel frame each latent step starts at, plus the end of the last one."""
    starts = [0]
    for k in range(steps):
        starts.append(starts[-1] + FRAME_PER_TOKEN[k % 5])
    return starts


def _frames_for(steps, phase=0):
    """Pixel frames a run of `steps` latent steps covers, starting at `phase`."""
    return sum(FRAME_PER_TOKEN[(phase + k) % 5] for k in range(steps))


def _steps_for_frames(n):
    """Latent steps covering exactly n frames from cycle position 0, else None."""
    k, covered = 0, 0
    while covered < n:
        covered += FRAME_PER_TOKEN[k % 5]
        k += 1
    return k if covered == n else None


def _snap_run(frames):
    """Frames snapped down to 17k+5: the only lengths that are a whole number of
    latent steps starting at cycle position 0. 0 when there is no such length."""
    frames = int(frames)
    return frames - (frames - 5) % RUN_MULTIPLE if frames >= 5 else 0


def _snap_gap(frames, phase):
    """Gap length in latent steps, snapped down to the grid that keeps the whole
    clip 5g+2 steps long with every block on its own cycle phase."""
    steps = max(1, int(round(int(frames) * STEP_MULTIPLE / RUN_MULTIPLE)))
    steps -= (steps - phase) % STEP_MULTIPLE
    return max(steps, phase or STEP_MULTIPLE)


def _snap_boundary(starts, frame):
    """Nearest latent boundary (a value of `starts`) to a requested frame."""
    return starts[min(range(len(starts)), key=lambda i: abs(starts[i] - frame))]


def _av_streams(latent):
    """(video [1,C,T,H,W], audio [1,C,2,A]) out of a MiniMax H3 AV latent."""
    samples = latent.get("samples") if isinstance(latent, dict) else latent
    if hasattr(samples, "unbind"):
        parts = list(samples.unbind())
    elif isinstance(samples, (list, tuple)):
        parts = list(samples)
    else:
        parts = []
    if len(parts) < 2:
        raise ValueError(
            "H3 Timeline: the latent is not a MiniMax H3 AV latent (a nested "
            "video/audio pair). Wire the sampler output of an H3 graph, or feed "
            "the clip's frames instead.")
    video, audio = parts[0], parts[1]
    if video.ndim != 5 or video.shape[1] != 24:
        raise ValueError("H3 Timeline: expected an H3 video latent [B,24,T,H,W], got {}".format(
            tuple(video.shape)))
    if audio.ndim != 4:
        raise ValueError("H3 Timeline: expected an H3 audio latent [B,32,2,A], got {}".format(
            tuple(audio.shape)))
    return video[:1], audio[:1]


def _encode_audio(audio_vae, audio, frame_start, frame_end):
    """The H3 audio VAE latent for frames [frame_start, frame_end) of a clip."""
    if audio is None:
        return None
    if audio_vae is None:
        raise ValueError(
            "H3 Timeline: the clip has audio but no audio VAE - connect one to "
            "audio_vae (or put it on the context) to keep its soundtrack.")
    waveform = audio["waveform"]
    sr = int(audio["sample_rate"])
    vae_sr = int(getattr(audio_vae, "audio_sample_rate", 32000))
    if sr != vae_sr:
        waveform = comfy.audio.resample(waveform, sr, vae_sr)
    lo = max(0, int(round(frame_start / FPS * vae_sr)))
    hi = min(int(waveform.shape[-1]), int(round(frame_end / FPS * vae_sr)))
    if hi <= lo:
        logging.warning("H3 Timeline: the clip's audio does not reach frames %d-%d; "
                        "that span gets new sound instead.", frame_start, frame_end)
        return None
    return audio_vae.encode(waveform[:1, :, lo:hi].movedim(1, -1))


def _sound_steps(block):
    """Audio latent steps a kept (video, sound) block carries, 0 when it has none."""
    return 0 if not block or block[1] is None else int(block[1].shape[-1])


def _per_frame_mask(mask, steps):
    """A MASK as one plane per pixel frame of a clip whose latent is `steps` long.
    Accepts one mask per frame, one per latent step, or one for the whole clip."""
    starts = _pixel_starts(steps)
    frames = starts[-1]
    n = int(mask.shape[0])
    if n == frames:
        return mask
    if n == 1:
        return mask.expand(frames, -1, -1)
    if n == steps:
        return torch.repeat_interleave(
            mask, torch.tensor([starts[k + 1] - starts[k] for k in range(steps)]), dim=0)
    raise ValueError(
        "the mask has %d frames, but the latent is %d pixel frames (%d latent steps) - "
        "give one mask per frame, one per latent step, or one for the whole clip."
        % (n, frames, steps))


def _picture_mask(video, mask):
    """The video denoise mask for a per-frame MASK covering the frames `video`
    holds. Max-pooled onto the latent grid, which is how the model pools the mask
    over each 2x2 DiT patch anyway, so a thin mask survives the downsample."""
    steps = int(video.shape[2])
    starts = _pixel_starts(steps)
    per_frame = _per_frame_mask(mask, steps)
    pooled = torch.nn.functional.adaptive_max_pool2d(
        per_frame.to(device=video.device, dtype=video.dtype).unsqueeze(1),
        (int(video.shape[-2]), int(video.shape[-1])))
    return torch.stack([pooled[starts[k]:starts[k + 1]].amax(dim=0)
                        for k in range(steps)]).permute(1, 0, 2, 3).unsqueeze(0)


class GibbyH3Timeline(io.ComfyNode):
    """Extend, prepend, loop or insert into an existing MiniMax H3 clip."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Gibby_H3_Timeline",
            display_name="H3 Timeline (Context)",
            category="gibby",
            description=(
                "Builds the target latent for a new span of a MiniMax H3 clip: extend, "
                "prepend, loop, or insert a gap. Kept frames carry a denoise mask of 0, so "
                "the sampler returns them untouched."
            ),
            inputs=[
                # The widgets are required, so the frontend renders them first and
                # the optional source/model sockets follow - which is the order
                # this node is wired in: say what to do, then say what to do it to.
                _CONTEXT_TYPE.Input("context", optional=True, tooltip=(
                    "Supplies the video VAE, the audio VAE and the source frames when "
                    "they are not wired directly.")),
                io.Latent.Input("av_latent", optional=True, tooltip=(
                    "The existing clip's H3 AV latent - a sampler output, not the empty "
                    "latent from H3 Pipe Apply. Preferred over frames: the kept parts stay "
                    "exactly what the model made.")),
                io.Image.Input("image", optional=True, tooltip=(
                    "The existing clip's frames; overrides the context's image. Costs a "
                    "VAE round trip on the kept parts - wire av_latent to avoid it.")),
                io.Audio.Input("audio", optional=True, tooltip=(
                    "The existing clip's soundtrack; only used with frames, since a latent "
                    "carries its own. Overrides the context's audio.")),
                io.Vae.Input("vae", optional=True, tooltip="Video VAE; overrides the context's."),
                io.Vae.Input("audio_vae", optional=True, tooltip="H3 audio VAE; overrides the context's."),
                io.Combo.Input("mode", options=["extend", "prepend", "loop", "insert"],
                               default="extend", tooltip=(
                                   "extend: after the clip; prepend: before it; insert: a gap "
                                   "inside it. loop: a bridge from the clip's end back to its "
                                   "start, so clip + result plays as a seamless loop.")),
                io.Int.Input("segment_frames", default=119, min=17, max=3600, step=17, tooltip=(
                    "Frames to generate, on multiples of 17 (119 = ~5s). loop can only land "
                    "on 17x+9, so it snaps down and says so.")),
                io.Int.Input("overlap_frames", default=39, min=5, max=3600, step=17, tooltip=(
                    "How much of the clip to keep as the model's starting point: its tail for "
                    "extend and loop, its head for prepend. Only 17x+5 lengths exist "
                    "(5, 22, 39, 56...); more keeps more motion but costs sequence length.")),
                io.Int.Input("loop_end_frames", default=22, min=5, max=3600, step=17, tooltip=(
                    "loop only: how much of the clip's HEAD to keep at the end of the bridge "
                    "as the target it has to arrive at. Independent of overlap_frames.")),
                io.Int.Input("insert_at_frame", default=120, min=-9999, max=9999, step=1, tooltip=(
                    "insert only: the frame the gap opens at, snapped to the nearest latent "
                    "boundary. Negative counts back from the end of the clip as the video VAE "
                    "sees it, which can be shorter than the frames fed in.")),
            ],
            outputs=[
                _CONTEXT_TYPE.Output(display_name="context"),
                io.Latent.Output(display_name="av_latent"),
                _H3_TIMELINE_TYPE.Output(display_name="h3_timeline"),
            ],
        )

    @classmethod
    def execute(cls, context=None, image=None, av_latent=None, audio=None, vae=None,
                audio_vae=None, mode="extend", segment_frames=119, overlap_frames=39,
                loop_end_frames=22, insert_at_frame=120) -> io.NodeOutput:
        ctx = ctx_from(context)
        vae = vae if vae is not None else ctx.get("vae")
        audio_vae = audio_vae if audio_vae is not None else ctx.get("vae_audio")
        image = image if image is not None else ctx.get("image")
        audio = audio if audio is not None else ctx.get("audio")
        if vae is None:
            raise ValueError("H3 Timeline: no video VAE - wire one in or put it on the context.")
        if av_latent is None and image is None:
            raise ValueError(
                "H3 Timeline: nothing to work on - wire the existing clip's frames or its "
                "H3 AV latent, or put the image on the context.")

        # ---- the source ---------------------------------------------------
        # A latent is sliced; frames are encoded. Only insert needs the whole
        # clip at once, so the other modes encode just the block they keep.
        src_v = src_a = whole = None
        if av_latent is not None:
            src_v, src_a = _av_streams(av_latent)
            starts = _pixel_starts(int(src_v.shape[2]))
            src_frames = starts[-1]
            src_audio = int(src_a.shape[-1])
        elif mode == "insert":
            run = _snap_run(image.shape[0])
            if run < 5:
                raise ValueError("H3 Timeline: insert needs at least 5 frames of source.")
            if run != image.shape[0]:
                logging.warning("H3 Timeline: %d frames is off the video VAE's 17-frame grid; "
                                "working with the first %d.", image.shape[0], run)
            whole = encode_image(vae, image[:run])["samples"]
            starts = _pixel_starts(int(whole.shape[2]))
            src_frames = starts[-1]
        else:
            src_frames = int(image.shape[0])

        def keep_run(requested, what):
            run = _snap_run(requested)
            if run > src_frames:
                run = _snap_run(src_frames)
            if run < 5:
                raise ValueError(
                    "H3 Timeline: %s must be at least 5 frames and the clip has to be "
                    "long enough to keep that many." % what)
            if run != int(requested):
                logging.warning("H3 Timeline: %s snapped from %d to %d frames (the video "
                                "VAE only encodes 17x+5).", what, requested, run)
            return run

        # ---- the timeline, as (kind, frame start, frames, latent steps) ---
        # The gap's length is quantised to the grid that keeps the whole clip
        # 5g+2 steps long with every block on its own cycle phase.
        gap_steps = _snap_gap(segment_frames, 3 if mode == "loop" else 0)
        cut = None

        if mode == "insert":
            cut = int(insert_at_frame)
            if cut < 0:
                cut = src_frames + cut
            cut = _snap_boundary(starts, min(max(cut, 0), src_frames))
            if cut in (0, src_frames):
                raise ValueError(
                    "H3 Timeline: insert_at_frame %d puts the gap at the very edge of the "
                    "clip - use extend or prepend for that." % insert_at_frame)
            segments = [("keep", 0, cut, starts.index(cut)),
                        ("gap", 0, 0, gap_steps),
                        ("keep", cut, src_frames - cut, len(starts) - 1 - starts.index(cut))]
        elif mode == "loop":
            head = keep_run(overlap_frames, "overlap_frames")
            tail = keep_run(loop_end_frames, "loop_end_frames")
            if head + tail > src_frames:
                raise ValueError(
                    "H3 Timeline: keeping %d frames of the tail and %d of the head is more "
                    "than the clip's %d - the bridge would keep the clip twice."
                    % (head, tail, src_frames))
            segments = [("keep", src_frames - head, head, _steps_for_frames(head)),
                        ("gap", 0, 0, gap_steps),
                        ("keep", 0, tail, _steps_for_frames(tail))]
        else:
            run = keep_run(overlap_frames, "overlap_frames")
            # what is kept is what the new span has to join onto: the clip's tail
            # when generating after it, its head when generating before it
            start = src_frames - run if mode == "extend" else 0
            segments = [("keep", start, run, _steps_for_frames(run)), ("gap", 0, 0, gap_steps)]
            if mode == "prepend":
                segments.reverse()

        # How many pixel frames the gap actually covers depends on which cycle
        # phase it opens on, so it can only be said once the blocks around it
        # are fixed. The clip's real length is whatever its step count covers,
        # which is what the audio stream has to be cut to.
        gap_at = next(i for i, seg in enumerate(segments) if seg[0] == "gap")
        gap_frames = _frames_for(gap_steps, sum(seg[3] for seg in segments[:gap_at]) % STEP_MULTIPLE)
        segments[gap_at] = ("gap", 0, gap_frames, gap_steps)
        segment_start = sum(seg[2] for seg in segments[:gap_at])
        if gap_frames != int(segment_frames):
            logging.warning("H3 Timeline: %d frames is off this mode's grid; generating %d "
                            "(%d latent steps).", segment_frames, gap_frames, gap_steps)
        new_frames = _pixel_starts(sum(seg[3] for seg in segments))[-1]

        # ---- materialise it -----------------------------------------------
        def video_block(frame_start, frames, steps):
            if whole is not None or src_v is not None:
                source = whole if whole is not None else src_v
                s0 = starts.index(frame_start)
                return source[:, :, s0:s0 + steps]
            encoded = encode_image(vae, image[frame_start:frame_start + frames])["samples"]
            if int(encoded.shape[2]) != steps:
                raise RuntimeError(
                    "H3 Timeline: %d frames encoded to %d latent steps, expected %d; the "
                    "video VAE's grid no longer matches this node."
                    % (frames, int(encoded.shape[2]), steps))
            return encoded

        # The kept blocks first: the gap tensors are made on their device and
        # dtype, and prepend and insert put the gap somewhere other than the end.
        kept = {}
        missing_sound = False
        for index, (kind, frame_start, frames, steps) in enumerate(segments):
            if kind != "keep" or steps == 0:
                continue
            sound = (src_a[..., int(round(frame_start / src_frames * src_audio)):
                           int(round((frame_start + frames) / src_frames * src_audio))]
                     if src_a is not None else
                     _encode_audio(audio_vae, audio, frame_start, frame_start + frames))
            if sound is None or int(sound.shape[-1]) == 0:
                missing_sound = True
                sound = None
            kept[index] = (video_block(frame_start, frames, steps), sound)

        new_audio = int(round(new_frames / FPS * AUDIO_LATENT_FPS))
        kept_audio = sum(int(s.shape[-1]) for _v, s in kept.values() if s is not None)
        gap_audio = new_audio - kept_audio
        if missing_sound:
            # A hole in the kept audio would desync the two streams, so the
            # model writes the whole soundtrack instead of continuing part of it.
            for index, (block, _sound) in kept.items():
                kept[index] = (block, None)
            kept_audio, gap_audio = 0, new_audio
        if gap_audio < 1:
            raise ValueError(
                "H3 Timeline: the kept audio came to %d steps but the new clip is only "
                "%d - the gap has no room for sound." % (kept_audio, new_audio))

        video_parts, audio_parts, v_masks, a_masks = [], [], [], []
        for index, (kind, _frame_start, _frames, steps) in enumerate(segments):
            if kind == "gap":
                shape = kept[next(iter(kept))][0]
                gap = torch.zeros((1, int(shape.shape[1]), steps) + tuple(shape.shape[-2:]),
                                  device=shape.device, dtype=shape.dtype)
                video_parts.append(gap)
                v_masks.append(gap.new_ones((1, 1, steps) + tuple(shape.shape[-2:])))
                sound = torch.zeros((1, 32, 2, gap_audio), device=shape.device, dtype=shape.dtype)
                audio_parts.append(sound)
                a_masks.append(sound.new_ones((1, 1, 2, gap_audio)))
                continue
            block, sound = kept[index]
            video_parts.append(block)
            v_masks.append(block.new_zeros((1, 1, steps) + tuple(block.shape[-2:])))
            if sound is not None:
                audio_parts.append(sound)
                a_masks.append(sound.new_zeros((1, 1, 2, int(sound.shape[-1]))))

        video = torch.cat(video_parts, dim=2)
        audio_out = torch.cat(audio_parts, dim=-1)
        out = {"samples": comfy.nested_tensor.NestedTensor((video, audio_out)),
               "noise_mask": comfy.nested_tensor.NestedTensor(
                   (torch.cat(v_masks, dim=2), torch.cat(a_masks, dim=-1)))}
        ctx_set_latent(ctx, out)

        # Every kept block as where it sits in the new clip and where it came from
        # in the source, in both clocks. Mode-agnostic, so anything that has to put
        # the source's own picture or sound back into this latent later - a pin
        # before an upscale pass - never re-derives the mode's layout or the grid.
        kept_blocks, at_frame, at_sound = [], 0, 0
        for index, (kind, frame_start, frames, _steps) in enumerate(segments):
            if kind == "keep":
                sound_steps = 0 if missing_sound else _sound_steps(kept.get(index))
                kept_blocks.append({"latent_frame": at_frame, "frames": frames,
                                    "source_frame": frame_start,
                                    "latent_audio": at_sound, "audio_steps": sound_steps})
                at_sound += sound_steps
            else:
                at_sound += gap_audio
            at_frame += frames

        # What the node that puts the finished clip back together needs: where the
        # model's work sits in the sampled output, and how much source has to go on
        # each side of them. The kept frames are also the reference a colour match
        # has to pull the new span towards, and segment_start/segment_frames name
        # that span exactly. The audio trims are in 40 Hz steps rather than frames
        # because the two clocks round differently, and a join that is one step off
        # is a click.
        head = segments[0][2] if mode in ("extend", "loop") else 0
        tail = segments[-1][2] if mode in ("prepend", "loop") else 0
        head_audio = _sound_steps(kept.get(0)) if segments[0][0] == "keep" else 0
        last = len(segments) - 1
        tail_audio = _sound_steps(kept.get(last)) if segments[last][0] == "keep" else 0
        timeline = {"mode": mode, "fps": FPS, "source_frames": src_frames,
                    "segment_start": segment_start, "segment_frames": gap_frames,
                    "trim_head": head, "trim_tail": tail,
                    "audio_trim_head": head_audio, "audio_trim_tail": tail_audio,
                    "insert_at_frame": cut, "kept": kept_blocks,
                    # Carried so the assemble does not have to be wired back to the
                    # source a second time. References, not copies. A source that
                    # came in as a latent has no decoded picture or sound to hand
                    # on, so those stay unwired for the assemble to fill in.
                    "source_image": image, "source_audio": audio}
        ctx["h3_timeline"] = timeline
        logging.info(
            "H3 Timeline: %s -> %d frames (%d video steps, %d audio steps), %d generated, "
            "%d audio steps kept, trim %d/%d",
            mode, new_frames, int(video.shape[2]), int(audio_out.shape[-1]), gap_frames,
            kept_audio, head, tail)
        return io.NodeOutput(ctx, out, timeline)


def _match_colors(span, anchor, opt):
    """Match every frame of the generated span to the one kept frame it continues
    from. Per-frame, not uniform: a span that drifts away from the clip as it goes
    has to be pulled back along its whole length, and one transform built from the
    span's pooled statistics applies its average drift to the very first frame too,
    which opens a hard step at the seam where the two sides already agreed."""
    alpha = span[..., 3:4] if span.shape[-1] > 3 else None
    matched, = ColorTransfer.execute(span[..., :3], anchor[..., :3],
                                     opt.get("color_match_method", "mkl_lab"),
                                     {"source_stats": "per_frame"},
                                     float(opt.get("color_match_strength", 1.0)))
    if alpha is not None:
        matched = torch.cat([matched, alpha.to(matched.device, matched.dtype)], dim=-1)
    return matched


def _join_audio(new, source, steps_head, steps_tail, source_first):
    """Drop the kept audio off the sampled output and put the source's own back on
    the side the mode says. The trims are 40 Hz latent steps, so they convert to
    samples exactly where the sampler kept them rather than where frames say."""
    waveform = new["waveform"]
    sr = int(new["sample_rate"])
    if source is None:
        logging.warning("H3 Timeline Assemble: no source audio wired, so the finished clip's "
                        "sound starts where the sampled output's does. Wire source_audio to "
                        "get the original's full track.")
        return {"waveform": waveform, "sample_rate": sr}
    if int(source["sample_rate"]) != sr:
        raise ValueError("H3 Timeline Assemble: source audio is %d Hz and the sampled audio is "
                         "%d Hz - resample one before joining them."
                         % (int(source["sample_rate"]), sr))
    cut_head = int(round(steps_head / AUDIO_LATENT_FPS * sr))
    cut_tail = int(round(steps_tail / AUDIO_LATENT_FPS * sr))
    length = int(waveform.shape[-1])
    if cut_head + cut_tail >= length:
        raise ValueError(
            "H3 Timeline Assemble: trimming %d + %d samples from %d of audio would leave "
            "nothing - the sampled audio does not match this h3_timeline."
            % (cut_head, cut_tail, length))
    span = waveform[..., cut_head:length - cut_tail] if cut_tail else waveform[..., cut_head:]
    channels = max(int(source["waveform"].shape[1]), waveform.shape[1])
    source_wave = _to_channels(source["waveform"], channels)
    span = _to_channels(span, channels)
    parts = [source_wave, span] if source_first else [span, source_wave]
    _check_join(parts[0], parts[1], sr)
    return {"waveform": torch.cat(parts, dim=-1), "sample_rate": sr}


def _to_channels(waveform, channels):
    """A mono recording under a stereo generation, or the reverse, is worth joining
    rather than failing over; anything else has no sensible mapping."""
    if waveform.shape[1] == channels:
        return waveform
    if waveform.shape[1] != 1:
        raise ValueError("H3 Timeline Assemble: cannot join %d channels of audio to %d."
                         % (waveform.shape[1], channels))
    return waveform.repeat(1, channels, 1)


def _check_join(first, second, sr):
    """Name a dropout at the join, so one can be told apart from a bad splice
    without hunting for it by ear. Returns the warning, None for a clean join."""
    window = max(int(round(0.1 * sr)), 1)
    before = first[..., -window:].float().pow(2).mean().sqrt().item()
    after = second[..., :window].float().pow(2).mean().sqrt().item()
    if before < 0.05 or after > 0.02 or after >= before * 0.25:
        return None
    warning = ("the 100 ms after the join is at %d%% of the level of the 100 ms before it - "
               "the new clip's sound starts near-silent, so the dropout is the generation and "
               "not the splice." % int(round(100 * after / before)))
    logging.warning("H3 Timeline Assemble: %s", warning)
    return warning


class GibbyH3TimelineAssemble(io.ComfyNode):
    """Put a sampled H3 Timeline result back together with the clip it continues."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Gibby_H3_Timeline_Assemble",
            display_name="H3 Timeline Assemble (Context)",
            category="gibby",
            description=(
                "Joins the span the model generated back onto the original clip, on the "
                "side H3 Timeline's mode says; insert needs no join. Picture and sound use "
                "the timeline's own cut points, so the join cannot drift."
            ),
            inputs=[
                _H3_TIMELINE_TYPE.Input("h3_timeline", tooltip=(
                    "The h3_timeline output of H3 Timeline (Context) - it says which frames "
                    "of the sampled output are new and where the source goes.")),
                _CONTEXT_TYPE.Input("context", optional=True, tooltip=(
                    "Supplies the sampled image and audio when they are not wired directly.")),
                io.Image.Input("image", optional=True, tooltip=(
                    "The decoded sampled output - the whole latent decoded, kept frames and "
                    "all. Overrides the context's image.")),
                io.Audio.Input("audio", optional=True, tooltip=(
                    "The sampled output's decoded audio. Overrides the context's audio.")),
                io.Image.Input("source_image", optional=True, tooltip=(
                    "The original clip's frames, for extend / prepend / loop. Defaults to the "
                    "ones the timeline carried; only a source that came in as a latent has to "
                    "be wired here.")),
                io.Audio.Input("source_audio", optional=True, tooltip=(
                    "The original clip's audio; defaults to the one the timeline carried. "
                    "Without it the finished clip's sound starts where the sampled output's "
                    "does, not at the beginning.")),
                _KSAMPLER_OPTIONS_TYPE.Input("color_match_options", optional=True, tooltip=(
                    "Color Match options: match the generated span's colour, frame by frame, "
                    "to the kept frame it continues from. The source is never modified.")),
            ],
            outputs=[
                _CONTEXT_TYPE.Output(display_name="context"),
                io.Image.Output(display_name="images"),
                io.Audio.Output(display_name="audio"),
            ],
        )

    @classmethod
    def execute(cls, h3_timeline=None, context=None, image=None, audio=None,
                source_image=None, source_audio=None, color_match_options=None) -> io.NodeOutput:
        tl = h3_timeline or {}
        mode = tl.get("mode")
        if mode not in ("extend", "prepend", "loop", "insert"):
            raise ValueError(
                "H3 Timeline Assemble: h3_timeline is not the output of H3 Timeline "
                "(Context) - wire that node's h3_timeline output in.")
        ctx = ctx_from(context)
        image = image if image is not None else ctx.get("image")
        audio = audio if audio is not None else ctx.get("audio")
        if image is None:
            raise ValueError(
                "H3 Timeline Assemble: nothing to assemble - wire the decoded sampled output "
                "in, or put it on the context.")

        frames = int(image.shape[0])
        start = int(tl.get("segment_start", 0))
        span = int(tl.get("segment_frames", 0))
        head, tail = int(tl.get("trim_head", 0)), int(tl.get("trim_tail", 0))
        if span < 1 or start + span > frames:
            raise ValueError(
                "H3 Timeline Assemble: this timeline wants frames %d-%d of the sampled "
                "output, which is %d frames - decode the latent this h3_timeline came from."
                % (start, start + span, frames))

        # Only the model's work is matched, and only to the frame it has to sit
        # beside: the kept frame the span continues from, or the one it leads into
        # when the span comes first.
        opt = next((o for o in (color_match_options or []) if o.get("type") == "color_match"), None)
        if opt is not None and float(opt.get("color_match_strength", 0.0)) > 0:
            anchor = image[start - 1:start] if start > 0 else image[start + span:start + span + 1]
            if anchor.shape[0] > 0:
                image = torch.cat([image[:start], _match_colors(image[start:start + span], anchor, opt),
                                   image[start + span:]], dim=0)

        if mode == "insert":
            out_images, out_audio = image, audio
        else:
            source_image = source_image if source_image is not None else tl.get("source_image")
            source_audio = source_audio if source_audio is not None else tl.get("source_audio")
            if source_image is None:
                raise ValueError(
                    "H3 Timeline Assemble: %s needs the original clip's frames - wire "
                    "source_image, or give H3 Timeline the frames instead of only its latent."
                    % mode)
            new = image[start:start + span]
            source_first = mode in ("extend", "loop")
            out_images = torch.cat([source_image, new] if source_first else [new, source_image], dim=0)
            out_audio = _join_audio(audio, source_audio, int(tl.get("audio_trim_head", 0)),
                                    int(tl.get("audio_trim_tail", 0)), source_first) \
                if audio is not None else None

        ctx_set_image(ctx, out_images)
        if out_audio is not None:
            ctx["audio"] = out_audio
        logging.info("H3 Timeline Assemble: %s -> %d frames, %s", mode, out_images.shape[0],
                     "%.3fs of audio" % (out_audio["waveform"].shape[-1] / out_audio["sample_rate"])
                     if out_audio is not None else "no audio")
        return io.NodeOutput(ctx, out_images, out_audio)


class GibbyH3TimelinePin(io.ComfyNode):
    """Put the clip's own frames back into an upscaled H3 latent and mask them off."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Gibby_H3_Timeline_Pin",
            display_name="H3 Timeline Pin (Context)",
            category="gibby",
            description=(
                "Prepares an H3 Timeline latent for the sampling pass that upscales it: a "
                "latent upscale re-samples everything, so this pastes the source's real "
                "latent and soundtrack back over the kept frames and masks them off."
            ),
            inputs=[
                _H3_TIMELINE_TYPE.Input("h3_timeline", tooltip=(
                    "The h3_timeline output of H3 Timeline (Context) - it names the frames "
                    "and the sound that are the source, and where in the clip they came from.")),
                _CONTEXT_TYPE.Input("context", optional=True, tooltip=(
                    "Supplies the latent to pin into and the VAEs to encode the source frames "
                    "with, when they are not wired directly.")),
                io.Latent.Input("av_latent", optional=True, tooltip=(
                    "The source clip as an H3 AV latent at the resolution the upscale "
                    "produced; must be the same clip that went into H3 Timeline. Encoded "
                    "from image when unwired.")),
                io.Image.Input("image", optional=True, tooltip=(
                    "The source clip's frames at the upscaled resolution, encoded with the "
                    "context's VAEs. Used when av_latent is unwired.")),
                io.Audio.Input("audio", optional=True, tooltip=(
                    "The source clip's soundtrack, for when the source comes in as frames. "
                    "Defaults to the one the timeline carried.")),
                io.Latent.Input("upscaled_av_latent", optional=True, tooltip=(
                    "The latent to fix: the first pass's output, upscaled to the new "
                    "resolution. Overrides the context's latent.")),
            ],
            outputs=[
                _CONTEXT_TYPE.Output(display_name="context"),
                io.Latent.Output(display_name="av_latent"),
            ],
        )

    @classmethod
    def execute(cls, h3_timeline=None, context=None, av_latent=None, image=None, audio=None,
                upscaled_av_latent=None) -> io.NodeOutput:
        tl = h3_timeline or {}
        blocks = tl.get("kept")
        if not blocks:
            raise ValueError(
                "H3 Timeline Pin: this h3_timeline has no kept source frames to pin - "
                "it is not the output of an H3 Timeline (Context) run that kept some.")
        ctx = ctx_from(context)
        base = upscaled_av_latent if upscaled_av_latent is not None else ctx.get("latent")
        if base is None:
            raise ValueError(
                "H3 Timeline Pin: nothing to pin into - wire upscaled_av_latent, or put "
                "the upscaled latent on the context.")
        base_v, base_a = _av_streams(base)

        # The source arrives either as a latent or as the frames to encode, the
        # same two ways H3 Timeline takes it. Only the kept blocks are read, and
        # each on its own: that is how H3 Timeline encoded them on the way in, so
        # the two agree instead of differing by where the video VAE's 17-frame
        # chunks happen to fall.
        if av_latent is not None:
            src_v, src_a = _av_streams(av_latent)
            src_starts = _pixel_starts(int(src_v.shape[2]))
            src_frames, src_steps = src_starts[-1], int(src_a.shape[-1])
        elif image is not None:
            if ctx.get("vae") is None:
                raise ValueError(
                    "H3 Timeline Pin: frames supplied with no video VAE on the context - "
                    "connect one, or give the source as an AV latent.")
            audio = audio if audio is not None else tl.get("source_audio")
        else:
            raise ValueError(
                "H3 Timeline Pin: nothing to pin - give the source clip as an AV latent "
                "or as frames.")

        base_starts = _pixel_starts(int(base_v.shape[2]))
        video = base_v.clone()
        sound = base_a.clone()
        mask = base_v.new_ones((1, 1) + tuple(base_v.shape[2:]))
        pinned = 0
        for block in blocks:
            frames, at, from_ = int(block["frames"]), int(block["latent_frame"]), int(block["source_frame"])
            try:
                s0, s1 = base_starts.index(at), base_starts.index(at + frames)
            except ValueError:
                raise ValueError(
                    "H3 Timeline Pin: this latent does not hold the clip the timeline "
                    "describes - pin the first pass's own latent, upscaled but not "
                    "otherwise changed.")
            if av_latent is not None:
                try:
                    r0 = src_starts.index(from_)
                except ValueError:
                    raise ValueError(
                        "H3 Timeline Pin: the source does not cover frames %d-%d of the "
                        "clip - pin the whole source clip, whose length has to sit on the "
                        "video VAE's 17-frame grid." % (from_, from_ + frames))
                block_v = src_v[:, :, r0:r0 + (s1 - s0)]
                lo = int(round(from_ / src_frames * src_steps))
                hi = int(round((from_ + frames) / src_frames * src_steps))
                block_a = src_a[..., lo:hi]
            else:
                block_v = encode_image(ctx["vae"], image[from_:from_ + frames])["samples"]
                block_a = _encode_audio(ctx.get("vae_audio"), audio, from_, from_ + frames)

            if tuple(block_v.shape[-2:]) != tuple(base_v.shape[-2:]):
                raise ValueError(
                    "H3 Timeline Pin: the source is %dx%d and the latent to fix is "
                    "%dx%d - give the source at the resolution the upscale produced."
                    % (block_v.shape[3] * 16, block_v.shape[2] * 16,
                       base_v.shape[3] * 16, base_v.shape[2] * 16))
            if int(block_v.shape[2]) != s1 - s0:
                raise ValueError(
                    "H3 Timeline Pin: frames %d-%d encoded to %d steps where this timeline "
                    "needs %d - give the whole source clip, whose length has to sit on the "
                    "video VAE's 17-frame grid." % (from_, from_ + frames,
                                                     int(block_v.shape[2]), s1 - s0))
            video[:, :, s0:s1] = block_v
            mask[:, :, s0:s1] = 0.0
            pinned += s1 - s0

            steps = int(block.get("audio_steps", 0))
            if steps < 1:
                continue
            if block_a is None or int(block_a.shape[-1]) != steps:
                logging.warning(
                    "H3 Timeline Pin: the source's sound for frames %d-%d came to %s where "
                    "this timeline expects %d steps, so that block keeps the audio the first "
                    "pass delivered.", from_, from_ + frames,
                    "nothing" if block_a is None else str(int(block_a.shape[-1])), steps)
                continue
            sound[..., int(block["latent_audio"]):int(block["latent_audio"]) + steps] = block_a

        # The whole audio stream is masked off, so the upscale pass leaves it
        # exactly as it is handed to it.
        audio_mask = base_a.new_zeros((1, 1) + tuple(base_a.shape[2:]))
        out = {"samples": comfy.nested_tensor.NestedTensor((video, sound)),
               "noise_mask": comfy.nested_tensor.NestedTensor((mask, audio_mask))}
        ctx_set_latent(ctx, out)
        logging.info("H3 Timeline Pin: %d of %d video steps pinned to the source and masked "
                     "off; %d left for the upscale pass",
                     pinned, int(base_v.shape[2]), int(base_v.shape[2]) - pinned)
        return io.NodeOutput(ctx, out)


class GibbyH3MaskLatent(io.ComfyNode):
    """Put a per-frame denoise mask on a MiniMax H3 AV latent, both streams."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Gibby_H3_Mask_Latent",
            display_name="H3 Mask Latent (Context)",
            category="gibby",
            description=(
                "Puts a per-frame denoise mask on a MiniMax H3 AV latent, both streams - Set "
                "Latent Noise Mask covers only the video and regenerates the soundtrack. "
                "Wire after H3 Timeline, never before."
            ),
            inputs=[
                _CONTEXT_TYPE.Input("context", optional=True, tooltip=(
                    "Supplies the latent to mask when it is not wired directly.")),
                io.Latent.Input("av_latent", optional=True, tooltip=(
                    "The H3 AV latent to mask. Overrides the context's latent.")),
                io.Mask.Input("mask", tooltip=(
                    "One mask per pixel frame of the clip, one per latent step, or one for "
                    "the whole clip. White is regenerated; black comes back untouched.")),
            ],
            outputs=[
                _CONTEXT_TYPE.Output(display_name="context"),
                io.Latent.Output(display_name="av_latent"),
            ],
        )

    @classmethod
    def execute(cls, context=None, av_latent=None, mask=None) -> io.NodeOutput:
        ctx = ctx_from(context)
        latent = av_latent if av_latent is not None else ctx.get("latent")
        if latent is None:
            raise ValueError("H3 Mask Latent: nothing to mask - wire the H3 AV latent.")
        if mask is None or int(mask.shape[0]) < 1:
            raise ValueError("H3 Mask Latent: no mask.")

        video, audio = _av_streams(latent)
        steps = int(video.shape[2])
        a_steps = int(audio.shape[-1])
        frames = _pixel_starts(steps)[-1]

        per_frame = _per_frame_mask(mask, steps)
        v_mask = _picture_mask(video, per_frame)

        # The soundtrack follows the picture: a step is re-denoised when any frame
        # it covers is masked anywhere, since that frame needs new sound. The
        # audio grid is finer than the frame grid (40 Hz against 24 fps), so a
        # frame can answer for several steps and every step has to answer for at
        # least one frame - rounding both bounds would collapse some to nothing.
        frame_masked = per_frame.amax(dim=(1, 2)) > 0.5
        a_vals = []
        for a in range(a_steps):
            lo = min(int(a * frames / a_steps), frames - 1)
            hi = min(int((a + 1) * frames / a_steps), frames)
            a_vals.append(frame_masked[lo:max(hi, lo + 1)].any())
        a_mask = torch.stack(a_vals).to(video.dtype)
        a_mask = a_mask.view(1, 1, 1, a_steps).expand(1, 1, 2, a_steps).contiguous()

        # Compose with a mask already on the latent rather than replace it, so a
        # picture mask can be confined to the span H3 Timeline generated.
        # A union, never an intersection: a mask may only add areas to regenerate.
        # Switching an area off would leave a span H3 Timeline left empty as what it
        # holds there - black picture and silence.
        held = latent.get("noise_mask")
        if held is not None:
            if getattr(held, "is_nested", False):
                # The soundtrack keeps the policy that mask already states. H3
                # Timeline opens only the gap for new sound, and a picture mask
                # that runs the length of the clip would otherwise rewrite every
                # step of it - which, relayed a few times, drifts the whole track.
                held_v, held_a = held.unbind()
                v_mask = torch.maximum(v_mask, held_v[:, :1])
                a_mask = held_a[:, :1].expand_as(a_mask).contiguous()
            else:
                v_mask = torch.maximum(v_mask, held[:, :1])

        out = {"samples": comfy.nested_tensor.NestedTensor((video, audio)),
               "noise_mask": comfy.nested_tensor.NestedTensor((v_mask, a_mask))}
        ctx_set_latent(ctx, out)
        logging.info("H3 Mask Latent: %d of %d video steps and %d of %d audio steps masked",
                     int((v_mask[0, 0].amax(dim=(1, 2)) > 0).sum()), steps,
                     int((a_mask[0, 0, 0] > 0).sum()), a_steps)
        return io.NodeOutput(ctx, out)


class GibbyH3Chunk(io.ComfyNode):
    """Cut one temporal window out of a long clip, sliced from its latent or encoded
    from its frames, so a sampling pass only ever sees that window."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Gibby_H3_Chunk",
            display_name="H3 Chunk (Context)",
            category="gibby",
            description=(
                "Cuts one temporal window out of a long MiniMax H3 clip so a sampling pass "
                "only ever sees that window: the way to v2v or inpaint a long clip without "
                "holding all of it at once."
            ),
            inputs=[
                _CONTEXT_TYPE.Input("context", optional=True, tooltip=(
                    "Supplies the latent, image, audio, mask and VAEs when they are not wired "
                    "directly.")),
                io.Latent.Input("av_latent", optional=True, tooltip=(
                    "The whole clip as an H3 AV latent - sliced, nothing is encoded. Wins "
                    "over image and overrides the context's latent. Feed the clip with each "
                    "chunk's result pasted back over for the next window.")),
                io.Latent.Input("previous_av_latent", optional=True, tooltip=(
                    "The chunk before this one, as it came from the sampler. Its tail supplies "
                    "the context band, so the model continues from already-replaced frames "
                    "instead of the untouched source. Needs frame_offset set to the loop's "
                    "accumulated_count.")),
                io.Image.Input("image", optional=True, tooltip=(
                    "The whole clip's frames, one per pixel frame - not a window; the next "
                    "window is cut from these. Encoded for this window when no latent is "
                    "wired, otherwise cropped for Image Paste By Mask. Overrides the "
                    "context's image.")),
                io.Audio.Input("audio", optional=True, tooltip=(
                    "The whole clip's soundtrack, for when the clip comes in as frames. "
                    "Overrides the context's audio.")),
                io.Mask.Input("mask", optional=True, tooltip=(
                    "The whole clip's per-frame masks, one per pixel frame - not a window. "
                    "Pooled onto this window as the denoise mask, and cropped through to it. "
                    "Overrides the context's mask.")),
                io.Vae.Input("vae", optional=True, tooltip="Video VAE; overrides the context's."),
                io.Vae.Input("audio_vae", optional=True,
                             tooltip="H3 audio VAE; overrides the context's."),
                io.Int.Input("chunk_frames", default=119, min=17, max=3600, step=17, tooltip=(
                    "Frames of the clip this window delivers. Snapped to a multiple of 17, "
                    "which is what keeps every later window's start on its cycle boundary.")),
                io.Int.Input("overlap_frames", default=39, min=5, max=3600, step=17, tooltip=(
                    "Frames carried in ahead of the window as motion context, kept untouched "
                    "and trimmed off the output. Only 17k+5 lengths exist (5, 22, 39, 56...); "
                    "the overlap_frames output is the number actually used. Ignored on the "
                    "first window.")),
                io.Int.Input("frame_offset", default=0, min=0, max=99999, step=1, tooltip=(
                    "Frames of the clip already produced before this window - the loop's "
                    "accumulated_count. Snapped back onto the grid.")),
            ],
            outputs=[
                _CONTEXT_TYPE.Output(display_name="context"),
                io.Latent.Output(display_name="av_latent", tooltip=(
                    "The window, both streams: its picture carries the denoise mask, and its "
                    "soundtrack is the clip's own, cropped and masked off.")),
                io.Image.Output(display_name="image", tooltip=(
                    "This window's crop of the input frames, context band included, for the "
                    "decode to be pasted back onto. None when no image was given.")),
                io.Audio.Output(display_name="audio", tooltip=(
                    "This window's crop of the input soundtrack at the same frames as the "
                    "image crop, context band included, so the two join with the same "
                    "overlap. None when no audio was given.")),
                io.Mask.Output(display_name="mask", tooltip=(
                    "This window's crop of the input masks. None when no mask was given.")),
                io.Int.Output(display_name="overlap_frames", tooltip=(
                    "Frames this window shares with the one before it - wire it into Image "
                    "Batch Extend With Overlap's overlap or Close Tensor Loop's "
                    "overlap_frames. The value actually used, so not necessarily what you "
                    "typed. 0 on the first window.")),
            ],
        )

    @classmethod
    def execute(cls, context=None, av_latent=None, previous_av_latent=None, image=None,
                audio=None, mask=None, vae=None, audio_vae=None, chunk_frames=119,
                overlap_frames=39, frame_offset=0) -> io.NodeOutput:
        ctx = ctx_from(context)
        vae = vae if vae is not None else ctx.get("vae")
        audio_vae = audio_vae if audio_vae is not None else ctx.get("vae_audio")
        image = image if image is not None else ctx.get("image")
        audio = audio if audio is not None else ctx.get("audio")
        mask = mask if mask is not None else ctx.get("mask")
        latent = av_latent if av_latent is not None else ctx.get("latent")
        if latent is None and image is None:
            raise ValueError(
                "H3 Chunk: nothing to chunk - wire the long clip's frames or its H3 AV "
                "latent, or put the image on the context.")

        # The window has to start where the video VAE's 17-frame cycle starts, or the
        # model spaces its steps with the wrong 1,4,4,4,4 pattern, and it has to run
        # 17k+5 frames, or the VAE cannot encode the tail of it. Those two pin all
        # three numbers: the output span to a multiple of 17, the overlap to 17k+5,
        # and the offset to wherever both hold at once.
        def snap(frames, what, first):
            """Snap a widget onto the grid and say so, the way H3 Timeline does."""
            snapped = first + RUN_MULTIPLE * max(0, int(round((frames - first) / float(RUN_MULTIPLE))))
            if snapped != int(frames):
                logging.warning("H3 Chunk: %s snapped from %d to %d frames (the video VAE "
                                "only lays out whole 17-frame cycles).", what, frames, snapped)
            return snapped

        chunk_frames = snap(chunk_frames, "chunk_frames", 17)
        overlap_frames = snap(overlap_frames, "overlap_frames", 5) if frame_offset else 0
        clip_frames = (int(image.shape[0]) if latent is None
                       else _pixel_starts(int(_av_streams(latent)[0].shape[2]))[-1])
        start = max(0, (int(frame_offset) - overlap_frames) // 17 * 17) + overlap_frames
        if start != int(frame_offset):
            logging.warning("H3 Chunk: frame_offset %d snapped to %d, so the window starts on "
                            "the video VAE's grid.", frame_offset, start)
        if start >= clip_frames:
            raise ValueError(
                "H3 Chunk: frame_offset %d is past the end of a %d-frame clip - the last "
                "window starts at frame %d."
                % (frame_offset, clip_frames, max(0, (clip_frames - chunk_frames) // 17 * 17)))
        want = min(start + chunk_frames, clip_frames)
        f0 = start - overlap_frames
        f1 = min(f0 + 5 + 17 * ((want - f0 + 11) // 17), clip_frames)
        if start == 0 and f1 - start != chunk_frames:
            logging.warning(
                "H3 Chunk: the first window delivers %d frames, not the %d asked for - it has "
                "no context in front of it, and a window must run 17k+5 frames for the VAE to "
                "encode it. Every window after this one delivers %d.",
                f1 - start, chunk_frames, chunk_frames)

        # Slice the latent, or slice the frames and encode only the window.
        if latent is not None:
            video, sound = _av_streams(latent)
            starts = _pixel_starts(int(video.shape[2]))
            try:
                s0, s1 = starts.index(f0), starts.index(f1)
            except ValueError:
                raise ValueError(
                    "H3 Chunk: this latent does not cover frames %d-%d of the clip - give "
                    "the whole clip's latent." % (f0, f1))
            video = video[:, :, s0:s1]
            sound = sound[..., int(round(f0 / FPS * AUDIO_LATENT_FPS)):
                              int(round(f1 / FPS * AUDIO_LATENT_FPS))]
            held = latent.get("noise_mask")
            if held is not None:
                stream = held.unbind()[0] if getattr(held, "is_nested", False) else held[:, :1]
                v_mask = stream[:, :, s0:s1].clone()
            else:
                v_mask = None
        else:
            if vae is None:
                raise ValueError(
                    "H3 Chunk: encoding the window needs a video VAE - wire one in or put it "
                    "on the context.")
            video = encode_image(vae, image[f0:f1])["samples"]
            sound = _encode_audio(audio_vae, audio, f0, f1)
            if sound is None:
                raise ValueError(
                    "H3 Chunk: encoding the window needs its soundtrack too - wire audio, or "
                    "give the clip as an H3 AV latent.")
            v_mask = None

        covered = _pixel_starts(int(video.shape[2]))[-1]
        if covered != f1 - f0:
            logging.warning("H3 Chunk: frames %d-%d encode to %d, so the window's last %d "
                            "fall off the video VAE's grid and are not sampled.",
                            f0, f1, covered, f1 - f0 - covered)
        ctx_steps = _steps_for_frames(overlap_frames) if overlap_frames else 0
        if previous_av_latent is not None and ctx_steps:
            prev_v, _prev_a = _av_streams(previous_av_latent)
            if tuple(prev_v.shape[-2:]) != tuple(video.shape[-2:]):
                raise ValueError(
                    "H3 Chunk: the previous chunk is %dx%d and this window is %dx%d - the two "
                    "have to be at the same size to carry its tail into this one."
                    % (prev_v.shape[3] * 16, prev_v.shape[2] * 16,
                       video.shape[3] * 16, video.shape[2] * 16))
            if int(prev_v.shape[2]) < ctx_steps:
                raise ValueError(
                    "H3 Chunk: the previous chunk's latent is %d steps and this window needs "
                    "its last %d as context - give the chunk straight before this one."
                    % (int(prev_v.shape[2]), ctx_steps))
            video = video.clone()
            video[:, :, :ctx_steps] = prev_v[:, :, -ctx_steps:]
        if mask is not None:
            added = _picture_mask(video, mask[f0:f0 + covered])
            # A union, never an intersection: a mask may only add areas to regenerate.
            v_mask = added if v_mask is None else torch.maximum(v_mask, added)
        if v_mask is None:
            v_mask = video.new_zeros((1, 1) + tuple(video.shape[2:]))
        v_mask[:, :, :ctx_steps] = 0.0

        for what, src in (("image", image), ("mask", mask)):
            if src is not None and int(src.shape[0]) < f1:
                raise ValueError(
                    "H3 Chunk: the %s has %d frames but this window needs %d-%d of the clip - "
                    "give the whole clip's %s, not a window of them."
                    % (what, int(src.shape[0]), f0, f1, what))
        out_image = out_mask = None
        if image is not None:
            out_image = image[f0:f1]
            ctx_set_image(ctx, out_image)
        if mask is not None:
            out_mask = mask[f0:f1]
            ctx["mask"] = out_mask
        out_audio = None
        if audio is not None:
            sr = int(audio["sample_rate"])
            lo = int(round(f0 / FPS * sr))
            hi = min(int(audio["waveform"].shape[-1]), int(round(f1 / FPS * sr)))
            out_audio = {"waveform": audio["waveform"][..., lo:hi], "sample_rate": sr}

        # The whole audio window is masked off: the clip's own soundtrack carries through
        # every chunk, so no window can rewrite a stretch the last one already delivered.
        a_mask = sound.new_zeros((1, 1) + tuple(sound.shape[2:]))
        out = {"samples": comfy.nested_tensor.NestedTensor((video, sound)),
               "noise_mask": comfy.nested_tensor.NestedTensor((v_mask, a_mask))}
        ctx_set_latent(ctx, out)
        logging.info("H3 Chunk: window frames %d-%d of %d, delivering %d-%d - set the loop's "
                     "overlap_frames to %d, or the output starts inside the context band and "
                     "carries %d kept frames before the new ones. %d latent steps, %d masked "
                     "for denoising, %s.",
                     f0, f1, clip_frames, start, f1, overlap_frames, overlap_frames,
                     int(video.shape[2]), int((v_mask[0, 0].amax(dim=(1, 2)) > 0).sum()),
                     "sliced from its latent" if latent is not None else "encoded from frames")
        return io.NodeOutput(ctx, out, out_image, out_audio, out_mask, overlap_frames)

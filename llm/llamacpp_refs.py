"""Reference-material collection helpers for the Gibby LLM Connect nodes.

Why this module exists
----------------------
llama.cpp's OpenAI-compatible ``/v1/chat/completions`` endpoint has no video
field either -- vision input is a list of ``image_url`` content parts on a
message, each one a still image. So every "video" reference still has to be
reduced to a handful of representative frames before it can reach the model,
exactly like it did for Ollama.

This module funnels arbitrary reference material into a flat list of base64
PNGs. What the caller does with that list (Ollama's ``images`` field vs.
llama.cpp's ``image_url`` content parts) is none of this module's business --
it is ported unchanged from ``ollama_refs.py`` in the Ollama-Enhanced pack,
because none of this logic was ever Ollama-specific to begin with.

* ``images_1``, ``images_2``, ... -- IMAGE batches, passed through verbatim.
* ``video_1``, ``video_2``, ...   -- VIDEO objects or IMAGE batches treated as
  footage, sub-sampled to N evenly-spaced frames.

Ordering is deterministic: all image slots in ascending slot order, then all
video slots in ascending slot order.

Nothing here imports torch at runtime. Frame tensors are duck-typed; the only
thing we need from them is ``.shape``, indexing and ``.cpu().numpy()``.
"""

from __future__ import annotations

import base64
import logging
import re
from io import BytesIO
from typing import Any, Iterator

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Socket name prefixes. The JS extension (web/js/GibbyLLMNode.js) grows
# sockets using these exact prefixes.
IMAGE_SLOT_PREFIX = "images"
VIDEO_SLOT_PREFIX = "video"

# Upper bound on how many sockets the frontend is allowed to grow to. This is a
# sanity rail, not a hard API limit; vision models generally fall apart well
# before this many references.
MAX_REF_SLOTS = 32

# Soft warning threshold for total images sent in a single request.
WARN_TOTAL_IMAGES = 16

# ``images``/``video`` (bare, slot 0) or ``images_3``/``video_12``.
_SLOT_RE = re.compile(r"^(?P<prefix>[A-Za-z_]+?)(?:_(?P<index>\d+))?$")


# --------------------------------------------------------------------------
# Dynamic socket collection
# --------------------------------------------------------------------------

def iter_dynamic_slots(values: dict[str, Any], prefix: str) -> Iterator[tuple[int, Any]]:
    """Yield ``(slot_index, value)`` for every populated ``<prefix>_<n>`` key.

    Slots are yielded in ascending numeric order. A bare ``<prefix>`` key (no
    numeric suffix) is treated as slot 0 so upstream-style workflows keep
    working. Empty slots are skipped.
    """
    found: list[tuple[int, Any]] = []
    for key, value in values.items():
        if value is None:
            continue
        match = _SLOT_RE.match(key)
        if match is None or match.group("prefix") != prefix:
            continue
        raw_index = match.group("index")
        found.append((0 if raw_index is None else int(raw_index), value))
    found.sort(key=lambda pair: pair[0])
    return iter(found)


# --------------------------------------------------------------------------
# Frame sampling
# --------------------------------------------------------------------------

def evenly_spaced_indices(total: int, count: int) -> list[int]:
    """Pick ``count`` evenly-spaced indices out of ``range(total)``.

    Always includes the first and last frame when ``count >= 2``. A request for
    a single frame returns the middle one, which is far more representative of
    a clip than frame 0 (which is often black or a fade-in).
    """
    if total <= 0:
        return []
    count = max(1, min(int(count), total))
    if count == 1:
        return [total // 2]
    step = (total - 1) / (count - 1)
    return sorted({int(round(i * step)) for i in range(count)})


def evenly_spaced_timestamps(start: float, duration: float, count: int) -> list[float]:
    """Same idea as :func:`evenly_spaced_indices` but in the time domain.

    Used when we sample a container with PyAV and only know the duration, not
    the exact frame count. The last sample is nudged slightly inside the clip
    so a seek to exactly ``end`` doesn't land past the final packet.
    """
    count = max(1, int(count))
    if duration <= 0:
        return [start]
    if count == 1:
        return [start + duration / 2.0]
    end = start + duration * 0.999
    step = (end - start) / (count - 1)
    return [start + i * step for i in range(count)]


# --------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------

def _frame_tensor_to_pil(frame: Any) -> Image.Image:
    """Convert a single ComfyUI IMAGE frame (``[H, W, C]``, float 0-1) to PIL."""
    array = frame.cpu().numpy() if hasattr(frame, "cpu") else np.asarray(frame)
    array = 255.0 * array
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def _downscale(image: Image.Image, max_size: int) -> Image.Image:
    """Shrink so the long edge is at most ``max_size``. ``0`` disables this."""
    if max_size <= 0:
        return image
    longest = max(image.width, image.height)
    if longest <= max_size:
        return image
    scale = max_size / float(longest)
    new_size = (max(1, int(round(image.width * scale))),
                max(1, int(round(image.height * scale))))
    return image.resize(new_size, Image.LANCZOS)


def pil_to_b64(image: Image.Image, max_size: int = 0) -> str:
    """Encode a PIL image as a base64 PNG string, optionally downscaled first."""
    image = _downscale(image, max_size)
    if image.mode not in ("RGB", "RGBA", "L"):
        image = image.convert("RGB")
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


# --------------------------------------------------------------------------
# Source adapters
# --------------------------------------------------------------------------

def _is_frame_batch(value: Any) -> bool:
    """True for a ComfyUI IMAGE batch (a ``[N, H, W, C]`` tensor)."""
    shape = getattr(value, "shape", None)
    return shape is not None and len(shape) == 4


def encode_image_batch(value: Any, max_size: int = 0) -> list[str]:
    """Encode every frame of an IMAGE batch. No sub-sampling -- matches upstream."""
    if _is_frame_batch(value):
        return [pil_to_b64(_frame_tensor_to_pil(frame), max_size) for frame in value]
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            out.extend(encode_image_batch(item, max_size))
        return out
    if isinstance(value, Image.Image):
        return [pil_to_b64(value, max_size)]
    # A lone [H, W, C] tensor.
    if getattr(value, "shape", None) is not None and len(value.shape) == 3:
        return [pil_to_b64(_frame_tensor_to_pil(value), max_size)]
    raise TypeError(f"Unsupported image input of type {type(value).__name__}")


def _sample_video_with_pyav(video: Any, frames: int, max_size: int) -> list[str] | None:
    """Sample a VIDEO by seeking, so we never materialise the whole clip.

    ``VideoInput.get_components()`` decodes every frame into RAM -- a 30 second
    1080p clip is roughly 22 GB as float32. Seeking to N timestamps keeps
    memory flat regardless of clip length.

    Returns ``None`` if this path isn't viable, so the caller can fall back.
    """
    try:
        import av  # noqa: PLC0415 - optional, ships with ComfyUI's video support
    except ImportError:
        return None

    get_stream_source = getattr(video, "get_stream_source", None)
    if get_stream_source is None:
        return None

    try:
        source = get_stream_source()
    except Exception as exc:  # pragma: no cover - depends on the concrete impl
        logger.debug("Gibby-LLM: no stream source for video (%s)", exc)
        return None

    try:
        with av.open(source) as container:
            if not container.streams.video:
                return None
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"

            # Honour any trim the upstream loader applied.
            start, window = 0.0, 0.0
            get_window = getattr(video, "get_active_trim_window", None)
            if get_window is not None:
                try:
                    start, window = get_window()
                except Exception:
                    start, window = 0.0, 0.0

            total = 0.0
            if stream.duration is not None and stream.time_base:
                total = float(stream.duration * stream.time_base)
            elif container.duration is not None:
                total = float(container.duration / av.time_base)

            duration = window if window > 0 else max(0.0, total - start)
            targets = evenly_spaced_timestamps(start, duration, frames)

            encoded: list[str] = []
            for target in targets:
                image = _decode_frame_at(container, stream, target)
                if image is not None:
                    encoded.append(pil_to_b64(image, max_size))
            return encoded or None
    except Exception as exc:
        logger.debug("Gibby-LLM: PyAV sampling failed (%s), falling back", exc)
        return None


def _decode_frame_at(container: Any, stream: Any, seconds: float) -> Image.Image | None:
    """Seek to ``seconds`` and return the first decoded frame at or after it."""
    try:
        if stream.time_base:
            offset = int(seconds / float(stream.time_base))
            container.seek(offset, stream=stream, backward=True, any_frame=False)
        else:
            container.seek(0)
    except Exception:
        try:
            container.seek(0)
        except Exception:
            return None

    best: Image.Image | None = None
    for frame in container.decode(stream):
        best = frame.to_image()
        frame_time = float(frame.pts * stream.time_base) if frame.pts is not None else None
        if frame_time is None or frame_time >= seconds:
            break
    return best


def encode_video(value: Any, frames: int, max_size: int = 0) -> list[str]:
    """Reduce one video reference to ``frames`` evenly-spaced base64 PNGs.

    Accepts either a ComfyUI ``VIDEO`` object or a raw IMAGE batch (what
    VideoHelperSuite's Load Video hands back as frames).
    """
    frames = max(1, int(frames))

    # Raw frame batch (VHS and friends) -- index-sample directly.
    if _is_frame_batch(value):
        indices = evenly_spaced_indices(int(value.shape[0]), frames)
        return [pil_to_b64(_frame_tensor_to_pil(value[i]), max_size) for i in indices]

    # Native VIDEO object -- try the memory-safe seek path first.
    if hasattr(value, "get_components"):
        sampled = _sample_video_with_pyav(value, frames, max_size)
        if sampled is not None:
            return sampled
        images = value.get_components().images
        indices = evenly_spaced_indices(int(images.shape[0]), frames)
        return [pil_to_b64(_frame_tensor_to_pil(images[i]), max_size) for i in indices]

    # Anything else: let the image path have a go (lists, PIL, single frame).
    return encode_image_batch(value, max_size)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def collect_reference_images(
    slot_values: dict[str, Any],
    frames_per_video: int = 4,
    max_image_size: int = 0,
    debug: bool = False,
) -> tuple[list[str] | None, list[str]]:
    """Turn every populated image/video slot into one flat list of base64 PNGs.

    Returns ``(images_b64, report)``. ``images_b64`` is ``None`` when nothing
    was connected. The caller decides how to attach that list to a request --
    see ``LLMConnect.py`` for the ``image_url`` content-part wrapping.
    ``report`` is a per-slot human-readable summary for debug output.
    """
    encoded: list[str] = []
    report: list[str] = []

    for index, value in iter_dynamic_slots(slot_values, IMAGE_SLOT_PREFIX):
        frames = encode_image_batch(value, max_image_size)
        encoded.extend(frames)
        report.append(f"{IMAGE_SLOT_PREFIX}_{index}: {len(frames)} image(s)")

    for index, value in iter_dynamic_slots(slot_values, VIDEO_SLOT_PREFIX):
        frames = encode_video(value, frames_per_video, max_image_size)
        encoded.extend(frames)
        kind = "frame batch" if _is_frame_batch(value) else type(value).__name__
        report.append(f"{VIDEO_SLOT_PREFIX}_{index}: {len(frames)} frame(s) from {kind}")

    if len(encoded) > WARN_TOTAL_IMAGES:
        logger.warning(
            "Gibby-LLM: sending %d images in one request. Most vision models "
            "degrade badly past ~%d -- consider lowering frames_per_video.",
            len(encoded), WARN_TOTAL_IMAGES,
        )

    if debug and report:
        print("--- llamacpp enhanced references:")
        for line in report:
            print(f"  {line}")
        print(f"  total: {len(encoded)} image(s) collected")
        print("---------------------------------------------------------")

    return (encoded or None), report

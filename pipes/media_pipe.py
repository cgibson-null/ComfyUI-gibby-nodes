"""Media pipe helpers.

The media pipe is a plain dict carrying raw reference media between nodes
(H3 Pipe Create/Apply, Reference Latent (Context), Generate):

    ref_images, keyframes + keyframe_positions, ref_videos,
    ref_video_audios, ref_audios

The h3_pipe is the same dict plus the h3 params (prompt, width, height,
length, ref_image_size). Keyframe positions are stored unresolved:
an integer is a frame index (negative counts from the end, -1 = last),
a float is a percentage of the video.
"""


def slot_order(name):
    """Sort key for slot names: numbered slots by number, the rest after."""
    tail = name.rsplit("_", 1)[-1]
    return (0, int(tail), name) if tail.isdigit() else (1, 0, name)


def pipe_add_images(pipe, images):
    """A copy of the media pipe with the images appended to its ref_images."""
    out = dict(pipe) if pipe else {}
    refs = dict(out.get("ref_images") or {})
    n = max([int(k.rsplit("_", 1)[-1]) for k in refs if k.rsplit("_", 1)[-1].isdigit()] or [0])
    for image in images:
        n += 1
        refs["ref_image_{}".format(n)] = image
    if refs:
        out["ref_images"] = refs
    return out

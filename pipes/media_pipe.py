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


def iter_numbered_slots(values, prefix):
    """(index, name, value) for every populated <prefix>_<n> key of values, in
    ascending slot order; a bare <prefix> key is slot 0. None values and other
    keys are skipped."""
    found = []
    for name, value in values.items():
        if value is None:
            continue
        if name == prefix:
            index = 0
        elif name.startswith(prefix + "_") and name.rsplit("_", 1)[-1].isdigit():
            index = int(name.rsplit("_", 1)[-1])
        else:
            continue
        found.append((index, name, value))
    found.sort(key=lambda slot: slot[0])
    return iter(found)


def pipe_add_images(pipe, images):
    """A copy of the media pipe with the images appended to its ref_images."""
    out = dict(pipe) if pipe else {}
    refs = dict(out.get("ref_images") or {})
    n = max([index for index, _name, _value in iter_numbered_slots(refs, "ref_image")] or [0])
    for image in images:
        n += 1
        refs["ref_image_{}".format(n)] = image
    if refs:
        out["ref_images"] = refs
    return out

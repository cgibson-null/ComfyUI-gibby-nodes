"""
Split Tiles (Context) / Combine Tiles (Context)
-----------------------------------------------
Tiling for tiled sampling/upscale. Split Tiles cuts the context image into a
batch of uniform, overlapping tiles (rows x cols, overlap as a fraction of
the tile size) and outputs the tiles, a matching batch of masks for the
crop-inpaint options, and the tiling info. The tiling info also travels in
the output context, so the tiles can be recombined with Combine Tiles after
they've been processed (e.g. upscaled and sampled per tile).

Combine Tiles recombines a tile batch (at whatever size it came back at)
into the image: the layout is scaled to the tiles' size (the upscaled case)
and the tiles are crossfaded across the overlap bands.

The tile mask (feathered) is 1 on the tile core and fades to 0 across each
overlap band: with masked_only the overlap keeps its content, and the
recombine crossfades the tiles across the overlap (the weights sum to 1
there), so the seams are blended without a separate pass.
"""

import math

import torch
from comfy_api.latest import io

from .context import _CONTEXT_TYPE, ctx_from, ctx_set_image
from .resolution_latent import _resize_image

# Carries how the image was tiled: the per-tile rects and the grid, so the
# tiles can be recombined (Combine Tiles, KSampler (Context) tiled mode)
_TILING_INFO_TYPE = io.Custom("GIBBY_TILING_INFO")


def _tile_axis(n, count, overlap):
    """(starts, tile_size, overlap_px) for tiling an axis of length n into count
    uniform, overlapping tiles: the tile size comes from the overlap fraction
    (span = count*tile - (count-1)*overlap*tile >= n), the first tile starts at
    0 and the last is pinned to the axis end - the axis is fully covered and
    every neighbor pair overlaps by ~overlap*tile (up to a rounding pixel)."""
    count = min(count, n)
    if count <= 1:
        return [0], n, 0
    u = int(math.ceil(n / (count - (count - 1) * overlap)))
    count = min(count, n - u + 1)
    if count <= 1:
        return [0], n, 0
    # The even stride that spans exactly n: the last start is n - u, so the
    # actual overlap is (count*u - n)/(count-1) ~ overlap*u
    s = (n - u) / (count - 1)
    starts = [min(int(round(i * s)), n - u) for i in range(count - 1)] + [n - u]
    return starts, u, int(round(u - s))


def _tile_grid(w, h, rows, cols, overlap):
    """(rects, rows, cols, tile_w, tile_h): the uniform tile grid of a w x h
    image; rects are (x0, y0, w, h) in row-major tile order."""
    ys, th, _ = _tile_axis(h, rows, overlap)
    xs, tw, _ = _tile_axis(w, cols, overlap)
    rects = [(xs[j], ys[i], tw, th) for i in range(len(ys)) for j in range(len(xs))]
    return rects, len(ys), len(xs), tw, th


def _tile_bands(info, i, j):
    """(band_l, band_r, band_t, band_b) of tile (i, j) in original pixels: how
    far each of its overlap bands extends from the tile's edges (0 on the image
    border, where there is no neighbor)."""
    rects = info["rects"]
    rows, cols = info["rows"], info["cols"]
    tw, th = info["tile_w"], info["tile_h"]
    k = i * cols + j
    x0, y0 = rects[k][0], rects[k][1]
    band_l = rects[k - 1][0] + tw - x0 if j > 0 else 0
    band_r = x0 + tw - rects[k + 1][0] if j < cols - 1 else 0
    band_t = rects[k - cols][1] + th - y0 if i > 0 else 0
    band_b = y0 + th - rects[k + cols][1] if i < rows - 1 else 0
    return max(0, band_l), max(0, band_r), max(0, band_t), max(0, band_b)


def _tile_weight(h, w, band_l, band_r, band_t, band_b, device=None):
    """(h, w) float32 weight: 1 on the tile core, a linear ramp to 0 across each
    overlap band (no band on the image border) - adjacent tiles' weights sum to
    1 across the overlap, so a weighted recombine is a plain crossfade."""
    wx = torch.ones(w, dtype=torch.float32, device=device)
    if band_l > 0:
        wx[:band_l] = torch.linspace(0.0, 1.0, band_l, device=device)
    if band_r > 0:
        wx[-band_r:] = torch.linspace(1.0, 0.0, band_r, device=device)
    wy = torch.ones(h, dtype=torch.float32, device=device)
    if band_t > 0:
        wy[:band_t] = torch.linspace(0.0, 1.0, band_t, device=device)
    if band_b > 0:
        wy[-band_b:] = torch.linspace(1.0, 0.0, band_b, device=device)
    return wy.unsqueeze(1) * wx.unsqueeze(0)


def _tile_masks(info, w, h, device=None):
    """(rows*cols, h, w) batch of per-tile masks: 'feathered' is the recombine
    weight (1 on the core, linear to 0 across each overlap band), 'full' is all
    1s. The bands are scaled to (w, h), which may differ from the tile size
    (the tiles are upscaled before sampling)."""
    rows, cols = info["rows"], info["cols"]
    if info.get("mask_style", "feathered") != "feathered":
        return torch.ones(rows * cols, h, w, dtype=torch.float32, device=device)
    sw, sh = w / info["tile_w"], h / info["tile_h"]
    masks = []
    for i in range(rows):
        for j in range(cols):
            bl, br, bt, bb = _tile_bands(info, i, j)
            masks.append(_tile_weight(h, w,
                                       max(0, round(bl * sw)), max(0, round(br * sw)),
                                       max(0, round(bt * sh)), max(0, round(bb * sh)),
                                       device=device))
    return torch.stack(masks)


def _combine_tiles(tiles, info, new_tw, new_th):
    """Recombine a (B*rows*cols, new_th, new_tw, C) batch of tiles (b-major) into
    the image at the new size: the tile layout is scaled to the new tile size
    (each tile keeps its own span, so the overlaps carry over) and the tiles
    are crossfaded across the overlap bands. Returns (B, out_h, out_w, C)."""
    rects = info["rects"]
    rows, cols = info["rows"], info["cols"]
    n = len(rects)
    B = tiles.shape[0] // n
    C = tiles.shape[-1]
    rw, rh = new_tw / info["tile_w"], new_th / info["tile_h"]
    places = [(round(x0 * rw), round(y0 * rh), max(1, round(tw * rw)), max(1, round(th * rh)))
              for (x0, y0, tw, th) in rects]
    out_w = max(px + pw for px, _, pw, _ in places)
    out_h = max(py + ph for _, py, _, ph in places)
    acc = torch.zeros(B, out_h, out_w, C, dtype=tiles.dtype, device=tiles.device)
    wsum = torch.zeros(B, out_h, out_w, dtype=torch.float32, device=tiles.device)
    for i in range(rows):
        for j in range(cols):
            k = i * cols + j
            px, py, pw, ph = places[k]
            t = tiles[k::n]
            if t.shape[1] != ph or t.shape[2] != pw:
                t = _resize_image(t, pw, ph, "lanczos")
            bl = places[k - 1][0] + places[k - 1][2] - px if j > 0 else 0
            br = px + pw - places[k + 1][0] if j < cols - 1 else 0
            bt = places[k - cols][1] + places[k - cols][3] - py if i > 0 else 0
            bb = py + ph - places[k + cols][1] if i < rows - 1 else 0
            w = _tile_weight(ph, pw, max(0, bl), max(0, br), max(0, bt), max(0, bb),
                             device=tiles.device)
            acc[:, py:py + ph, px:px + pw, :] += t * w.to(acc.dtype).view(1, ph, pw, 1)
            wsum[:, py:py + ph, px:px + pw] += w
    return acc / wsum.unsqueeze(-1).clamp(min=1e-3)


class GibbySplitTiles(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Gibby_Split_Tiles",
            display_name="Split Tiles (Context)",
            category="gibby/image",
            search_aliases=["tile", "split", "tiles", "tiled", "upscale", "mask"],
            description=(
                "Splits the image into a batch of uniform, overlapping tiles (rows x cols, "
                "overlap as a fraction of the tile size) and outputs the tiles, a matching batch "
                "of masks for the crop-inpaint options and the tiling info for Combine Tiles "
                "(Context). The output context carries the tiling info as well. Stills only."
            ),
            inputs=[
                _CONTEXT_TYPE.Input("context", optional=True,
                                     tooltip="Base context; its image is tiled when no image is connected"),
                io.Image.Input("image", optional=True,
                               tooltip="Overrides the context image (the latent and cached sample built from the old image are dropped)"),
                io.Int.Input("rows", default=2, min=1, max=256, step=1, tooltip="Tile rows"),
                io.Int.Input("cols", default=2, min=1, max=256, step=1, tooltip="Tile columns"),
                io.Float.Input("overlap", default=0.25, min=0.0, max=0.5, step=0.01,
                               tooltip="Tile overlap as a fraction of the tile size"),
                io.Combo.Input("mask", default="feathered", options=["feathered", "full"],
                               tooltip="feathered: 1 on the tile core, fading to 0 across each overlap band (masked_only keeps the overlap content, the recombine crossfades it). full: 1 across the whole tile (whole-tile denoise)"),
            ],
            outputs=[
                _CONTEXT_TYPE.Output("context"),
                io.Image.Output("tiles", tooltip="Batch of tiles (b-major)"),
                io.Mask.Output("masks", tooltip="Per-tile masks for the crop-inpaint options"),
                _TILING_INFO_TYPE.Output("tiling_info",
                                          tooltip="How the image was split: for Combine Tiles (Context)"),
            ],
        )

    @classmethod
    def execute(cls, context=None, image=None, rows=2, cols=2, overlap=0.25, mask="feathered") -> io.NodeOutput:
        ctx = ctx_from(context)
        if image is not None:
            ctx_set_image(ctx, image)
        image = ctx.get("image")
        if image is None:
            raise ValueError("no image to tile: connect an image or a context carrying one")
        if image.dim() != 4:
            raise ValueError("Split Tiles (Context) supports stills only")
        B, H, W, C = image.shape
        rects, trows, tcols, tw, th = _tile_grid(W, H, rows, cols, overlap)
        info = {"rects": rects, "rows": trows, "cols": tcols,
                "tile_w": tw, "tile_h": th, "width": W, "height": H, "mask_style": mask}
        tiles = torch.stack([image[b, y0:y0 + th, x0:x0 + tw, :]
                             for b in range(B) for (x0, y0, _, _) in rects])
        masks = _tile_masks(info, tw, th, device=image.device).repeat(B, 1, 1)
        ctx["tiling_info"] = info
        return io.NodeOutput(ctx, tiles, masks, info)


class GibbyCombineTiles(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Gibby_Combine_Tiles",
            display_name="Combine Tiles (Context)",
            category="gibby/image",
            search_aliases=["tile", "combine", "untile", "collect", "recombine", "tiled"],
            description=(
                "Recombines a batch of tiles from a tiling info into the image: the layout is "
                "scaled to the tiles' size (which may be larger than the split - the upscaled "
                "case) and the tiles are crossfaded across the overlap bands."
            ),
            inputs=[
                _CONTEXT_TYPE.Input("context", optional=True,
                                     tooltip="Base context; its tiling info and image are used when none are connected"),
                _TILING_INFO_TYPE.Input("tiling_info", optional=True,
                                         tooltip="From Split Tiles (Context)"),
                io.Image.Input("new_tiles", optional=True,
                               tooltip="The tile batch (b-major), at whatever size it came back at"),
            ],
            outputs=[
                _CONTEXT_TYPE.Output("context"),
                io.Image.Output("image"),
            ],
        )

    @classmethod
    def execute(cls, context=None, tiling_info=None, new_tiles=None) -> io.NodeOutput:
        ctx = ctx_from(context)
        info = tiling_info if tiling_info is not None else ctx.get("tiling_info")
        tiles = new_tiles if new_tiles is not None else ctx.get("image")
        if info is None:
            raise ValueError("no tiling info: connect Split Tiles (Context) tiling_info or a context carrying it")
        if tiles is None:
            raise ValueError("no tiles to combine: connect new_tiles or a context carrying an image")
        n = len(info["rects"])
        if tiles.shape[0] % n != 0:
            raise ValueError(f"tile batch of {tiles.shape[0]} does not match the tiling info's {n} tiles")
        out = _combine_tiles(tiles, info, tiles.shape[2], tiles.shape[1])
        ctx_set_image(ctx, out)
        ctx.pop("mask", None)
        ctx.pop("tiling_info", None)
        return io.NodeOutput(ctx, out)

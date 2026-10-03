// ---------------------------------------------------------------------------
// Gibby Nodes - shared crop preview
//
// The crop preview used by the crop mode of Resize Image / Empty Latent
// (Context): the image letterboxed in a stretchable box (it grows with the
// node), a draggable/resizable box that writes image-pixel coordinates into
// the node's crop_region widget, and a W x H label at the bottom.
// ---------------------------------------------------------------------------

import { app } from "../../scripts/app.js";

// The row's minimum, for the DOM widget's computeLayoutSize: the preview's
// floor. The widget has no maxHeight, so it absorbs the node's extra height
// and the preview stretches with the node.
export const CROP_PREVIEW_MIN_H = 120;

// Resolve the live instance in the graph currently shown: a tab switch
// rebuilds the graph in place (new instance, same id), so a global registry
// would go stale and writes would hit a dead instance.
export function liveNode(id) {
    const graph =
        (app.canvas && app.canvas.getCurrentGraph && app.canvas.getCurrentGraph()) ||
        app.graph;
    return graph ? graph.getNodeById(id) : null;
}

function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
}

function getWidgetValue(node, name) {
    return node.widgets?.find((w) => w.name === name)?.value;
}

// The URL without the store's &rand cache-buster (buildImageUrls appends a
// fresh one on every call), so an unrelated store refresh never reloads
// the image.
function urlKey(url) {
    if (!url) return "";
    try {
        const u = new URL(url, location.origin);
        u.searchParams.delete("rand");
        return u.toString();
    } catch (e) { return url; }
}

export function buildCropPreview(node, { getUrl, getRatio = null, onImageLoad = null }) {
    // The DOM widget's element: the stretchable preview.
    const el = document.createElement("div");
    el.style.cssText = "display:flex; flex-direction:column; gap:6px; height:100%; box-sizing:border-box; min-height:0;";

    const wrap = document.createElement("div");
    wrap.style.cssText =
        "position:relative; margin:0; background:#111; border:1px solid #333; " +
        "border-radius:3px; overflow:hidden; line-height:0; box-sizing:border-box; " +
        "flex:1 1 auto; min-height:0; display:flex; " +
        "align-items:center; justify-content:center;";
    const img = document.createElement("img");
    img.style.cssText = "display:block; user-select:none; -webkit-user-drag:none;";
    const empty = document.createElement("div");
    empty.style.cssText =
        "position:absolute; inset:0; display:flex; align-items:center; justify-content:center; " +
        "color:#888; font-size:12px;";
    empty.textContent = "No image to crop";
    const size = document.createElement("div");
    size.style.cssText =
        "position:absolute; left:4px; bottom:4px; padding:1px 5px; line-height:1.4; " +
        "background:rgba(0,0,0,0.6); color:#ccc; font-size:11px; border-radius:2px; " +
        "display:none;";
    const box = document.createElement("div");
    box.style.cssText =
        "position:absolute; box-sizing:border-box; border:1px solid #4a9eff; " +
        "background:rgba(74,158,255,0.12); cursor:move;";
    wrap.appendChild(img);
    wrap.appendChild(empty);
    wrap.appendChild(box);
    wrap.appendChild(size);

    const HANDLES = [
        ["nw", 0, 0, "nwse-resize"], ["n", 0.5, 0, "ns-resize"],
        ["ne", 1, 0, "nesw-resize"], ["e", 1, 0.5, "ew-resize"],
        ["se", 1, 1, "nwse-resize"], ["s", 0.5, 1, "ns-resize"],
        ["sw", 0, 1, "nesw-resize"], ["w", 0, 0.5, "ew-resize"],
    ];
    const handles = HANDLES.map(([dir, fx, fy, cursor]) => {
        const h = document.createElement("div");
        h.style.cssText =
            "position:absolute; width:8px; height:8px; margin:-4px 0 0 -4px; " +
            "background:#4a9eff; border:1px solid #fff; box-sizing:content-box; " +
            "cursor:" + cursor + ";";
        h._gibbyDir = dir;
        h._gibbyFx = fx;
        h._gibbyFy = fy;
        box.appendChild(h);
        return h;
    });

    el.appendChild(wrap);

    // Display the image at its natural size capped by the container's
    // height (the width follows the AR) and shrunk to fit the container's
    // width: resizing the node only changes the black padding.
    function fitImg() {
        const nat = el._gibbyNatural;
        if (!nat || !wrap.clientWidth || !wrap.clientHeight) return;
        const ar = nat.w / nat.h;
        let ch = Math.min(nat.h, wrap.clientHeight), cw = ch * ar;
        if (cw > wrap.clientWidth) { cw = wrap.clientWidth; ch = cw / ar; }
        img.style.width = Math.round(cw) + "px";
        img.style.height = Math.round(ch) + "px";
    }

    // The rendered image rect inside the wrap, in CSS px (the box's styles
    // are CSS px; the overlay follows the canvas zoom, so bounding rects
    // would mix screen px in). The image is centered in the container (the
    // flex layout), the box is offset the same way.
    function imgBox() {
        const ww = wrap.clientWidth, wh = wrap.clientHeight;
        const cw = img.clientWidth, ch = img.clientHeight;
        const nat = el._gibbyNatural || { w: 1, h: 1 };
        const scale = cw / nat.w; // CSS px per image px
        return {
            left: (ww - cw) / 2,
            top: (wh - ch) / 2,
            scale,
            // Screen px per image px: the canvas zoom, for pointer deltas.
            screenScale: scale * (img.getBoundingClientRect().width / cw || 1),
            W: nat.w,
            H: nat.h,
        };
    }
    function rect() {
        const n = liveNode(node.id);
        const v = (n ? getWidgetValue(n, "crop_region") : null) || {};
        return { x: v.x || 0, y: v.y || 0, w: v.width || 512, h: v.height || 512 };
    }
    // The ratio the box is locked to: the node's own (the resize node's
    // crop_ar).
    function lockedRatio() {
        const n = liveNode(node.id);
        return (n && getRatio(n)) || null;
    }
    function paint() {
        fitImg();
        // No image loaded yet, or the row is hidden (zero size): nothing to
        // anchor the box to.
        const has = !!(el._gibbyNatural && img.clientWidth);
        empty.style.display = has ? "none" : "";
        if (!has) { box.style.display = "none"; return; }
        box.style.display = "";
        const b = imgBox();
        const r = rect();
        box.style.left = (b.left + r.x * b.scale) + "px";
        box.style.top = (b.top + r.y * b.scale) + "px";
        box.style.width = Math.max(1, r.w * b.scale) + "px";
        box.style.height = Math.max(1, r.h * b.scale) + "px";
        for (const h of handles) {
            h.style.left = (h._gibbyFx * r.w * b.scale) + "px";
            h.style.top = (h._gibbyFy * r.h * b.scale) + "px";
        }
    }
    el._gibbyNatural = null;
    el._gibbyPaint = paint;
    function setRect(r) {
        const n = liveNode(node.id);
        const w = n && n.widgets?.find((w) => w.name === "crop_region");
        if (!w) return;
        w.value = {
            x: Math.round(r.x), y: Math.round(r.y),
            width: Math.round(r.w), height: Math.round(r.h),
        };
        paint();
    }

    let drag = null;
    function startDrag(e, dir) {
        const b = imgBox();
        if (!b.scale) return;
        drag = { dir, sx: e.clientX, sy: e.clientY, r: rect(), b };
        e.target.setPointerCapture(e.pointerId);
        e.preventDefault();
    }
    box.addEventListener("pointerdown", (e) => {
        if (e.target !== box) return;
        // Keep the drag inside the box: the new Vue node layer would
        // otherwise start dragging the node with the same pointer.
        e.stopPropagation();
        startDrag(e, "move");
    });
    for (const h of handles) {
        h.addEventListener("pointerdown", (e) => {
            e.stopPropagation();
            // With a locked ratio only the corners resize.
            if (lockedRatio() && h._gibbyDir.length !== 2) return;
            startDrag(e, h._gibbyDir);
        });
    }
    function onPointerMove(e) {
        if (!drag) return;
        const b = drag.b;
        // Pointer deltas are screen px; the overlay follows the canvas zoom.
        const dx = (e.clientX - drag.sx) / b.screenScale;
        const dy = (e.clientY - drag.sy) / b.screenScale;
        const s = drag.r;
        const ratio = lockedRatio();
        const r = { x: s.x, y: s.y, w: s.w, h: s.h };
        if (drag.dir === "move") {
            r.x = clamp(s.x + dx, 0, Math.max(0, b.W - r.w));
            r.y = clamp(s.y + dy, 0, Math.max(0, b.H - r.h));
        } else if (drag.dir === "se") {
            r.w = clamp(s.w + dx, 1, Math.max(1, b.W - s.x));
            if (ratio) { r.h = clamp(r.w / ratio, 1, Math.max(1, b.H - s.y)); r.w = r.h * ratio; }
            else r.h = clamp(s.h + dy, 1, Math.max(1, b.H - s.y));
        } else if (drag.dir === "e") {
            r.w = clamp(s.w + dx, 1, Math.max(1, b.W - s.x));
        } else if (drag.dir === "s") {
            r.h = clamp(s.h + dy, 1, Math.max(1, b.H - s.y));
        } else if (drag.dir === "ne") {
            const bottom = s.y + s.h;
            r.w = clamp(s.w + dx, 1, Math.max(1, b.W - s.x));
            if (ratio) { r.h = clamp(r.w / ratio, 1, Math.max(1, bottom)); r.w = r.h * ratio; }
            else r.h = clamp(s.h - dy, 1, Math.max(1, bottom));
            r.y = bottom - r.h;
        } else if (drag.dir === "nw") {
            const right = s.x + s.w;
            const bottom = s.y + s.h;
            r.w = clamp(s.w - dx, 1, Math.max(1, right));
            if (ratio) { r.h = clamp(r.w / ratio, 1, Math.max(1, bottom)); r.w = r.h * ratio; }
            else r.h = clamp(s.h - dy, 1, Math.max(1, bottom));
            r.x = right - r.w;
            r.y = bottom - r.h;
        } else if (drag.dir === "sw") {
            const right = s.x + s.w;
            r.w = clamp(s.w - dx, 1, Math.max(1, right));
            if (ratio) { r.h = clamp(r.w / ratio, 1, Math.max(1, b.H - s.y)); r.w = r.h * ratio; }
            else r.h = clamp(s.h + dy, 1, Math.max(1, b.H - s.y));
            r.x = right - r.w;
        } else if (drag.dir === "w") {
            const right = s.x + s.w;
            r.w = clamp(s.w - dx, 1, Math.max(1, right));
            r.x = right - r.w;
        } else if (drag.dir === "n") {
            const bottom = s.y + s.h;
            r.h = clamp(s.h - dy, 1, Math.max(1, bottom));
            r.y = bottom - r.h;
        }
        setRect(r);
    }
    function onPointerUp() {
        drag = null;
    }
    for (const t of [box, ...handles]) {
        t.addEventListener("pointermove", onPointerMove);
        t.addEventListener("pointerup", onPointerUp);
        t.addEventListener("pointercancel", onPointerUp);
    }

    img.onload = () => {
        el._gibbyNatural = { w: img.naturalWidth, h: img.naturalHeight };
        size.textContent = img.naturalWidth + " \u00d7 " + img.naturalHeight;
        size.style.display = "";
        paint();
        onImageLoad?.(node);
    };
    img.onerror = () => {
        el._gibbyNatural = null;
        size.style.display = "none";
        paint();
    };

    // Show the current image (called on setup and whenever the source
    // changes).
    el._gibbyLoad = () => {
        const url = getUrl(liveNode(node.id) || node);
        if (url && urlKey(url) !== urlKey(img.src)) img.src = url;
        if (!url) {
            img.removeAttribute("src");
            el._gibbyNatural = null;
            size.style.display = "none";
        }
        paint();
    };
    // Resizing the node resizes the preview (the flex layout); repaint the
    // box to follow.
    new ResizeObserver(() => {
        paint();
    }).observe(wrap);

    return el;
}

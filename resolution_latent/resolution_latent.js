// ---------------------------------------------------------------------------
// Gibby Nodes - Resize Image / Empty Latent (Context) frontend
//
// The node's standard widgets are declared in the backend schema so their
// values serialize into prompts normally. This file upgrades one of them to
// a custom DOM UI using ComfyUI's own node.addDOMWidget() (NOT hand-drawn
// canvas widgets):
//   - "mode" combo          -> horizontal toggle switch (custom / aspect ratio / custom AR, plus keep AR when media is linked)
//   - "crop_ar" combo        -> native labeled toggle (Custom / Selected) above the crop preview
// Rows are shown/hidden per mode via widget.options.hidden (the same flag the
// new frontend's advanced-widget toggle uses). The load image group
// (file combo + upload, crop toggle, crop AR toggle, crop preview, crop
// region) is instead added to and removed from node.widgets physically,
// with the widget store kept in sync the same way the core's dynamic
// widgets do (the Vue rows follow the store, not node.widgets): removed
// rows are not serialized into the prompt, so the backend falls back to
// their schema defaults.
// ---------------------------------------------------------------------------

import { app } from "../../../scripts/app.js";

const NODE_TYPE = "Gibby_EmptyLatent_Resolution";

// Resolve the live instance that owns a switcher. The buttons only exist
// while their node is in the graph on screen, so look the node up in the
// graph currently shown. A tab switch rebuilds the graph in place (new
// instance, same id), and a global instance registry went stale after that -
// first-match lookup then wrote to a dead instance and the buttons stopped
// reacting until a page refresh.
function liveNode(id) {
    const graph =
        (app.canvas && app.canvas.getCurrentGraph && app.canvas.getCurrentGraph()) ||
        app.graph;
    return graph ? graph.getNodeById(id) : null;
}

const MODES = [
    ["keep_ar", "Keep AR"],
    ["custom", "Custom"],
    ["aspect_ratio", "Aspect Ratio"],
    ["custom_aspect_ratio", "Custom AR"],
];

// Standard widget names in schema order (excluding the DOM switcher, which is
// not serialized). configure() maps legacy positional widgets_values onto
// these; with widgets_values_named present the order is irrelevant.
const CANONICAL_WIDGETS = [
    "mode", "width", "height", "aspect_ratio", "x", "y", "megapixels",
    "scale_factor", "upscale_method", "keep_proportion", "pad_color",
    "crop_position", "swap_dimensions", "multiple", "batch_size",
    "flux2_latent", "load_image", "image", "crop_image", "crop_ar",
    "crop_region"
];

// Same presets as the backend's aspect ratio mode; used to resolve the
// "selected" crop box AR.
const ASPECT_RATIOS = {
    "1:1 (Square)": [1, 1],
    "2:3 (Portrait Photo)": [2, 3],
    "3:2 (Photo)": [3, 2],
    "3:4 (Portrait Standard)": [3, 4],
    "4:3 (Standard)": [4, 3],
    "9:16 (Portrait Widescreen)": [9, 16],
    "16:9 (Widescreen)": [16, 9],
    "21:9 (Ultrawide)": [21, 9],
};
const CLOSEST_RATIO = "closest to image";

function findWidget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

// Write through to the standard widget object so prompt serialization and Vue
// reactivity both see the change.
function setWidgetValue(node, name, value) {
    const w = findWidget(node, name);
    if (w) w.value = value;
}

function getWidgetValue(node, name) {
    return findWidget(node, name)?.value;
}

function setWidgetHidden(widget, hidden) {
    if (!widget) return;
    widget.options = widget.options || {};
    widget.options.hidden = !!hidden;
    widget.hidden = !!hidden;
    
    // Try to find and hide the widget's DOM element
    if (widget._gibbyDomEl || widget.el || widget.domElement) {
        const el = widget._gibbyDomEl || widget.el || widget.domElement;
        el.style.display = hidden ? 'none' : '';
    }
}

// The frontend's Pinia stores, reached through the mounted Vue app. The
// core's dynamic widgets (the context loader's mode rows) add/remove rows
// through the widget store, so rows managed the same way behave exactly
// like the core's.
function frontendStore(id) {
    try {
        const el = document.getElementById("vue-app") || document.querySelector("[data-v-app]");
        const pinia = el?.__vue_app__?.config?.globalProperties?.$pinia;
        return pinia?._s?.get?.(id) || null;
    } catch (e) { return null; }
}

// Re-sync the store's row order with node.widgets (the core's
// syncNodeWidgetOrder). The Vue widget list follows this order.
function syncNodeWidgetOrder(node) {
    const store = frontendStore("widgetValue");
    const graph = node.graph?.rootGraph || node.graph;
    if (!store?.setNodeWidgetOrder || !graph?.id || !node.widgets) return;
    try {
        store.setNodeWidgetOrder(graph.id, node.id, node.widgets.map((w) => w.widgetId).filter(Boolean));
    } catch (e) { /* ignore */ }
}

// Remove a row: splice it out of node.widgets and drop it from the widget
// store, which is what actually makes the Vue row disappear.
function removeWidgetRow(node, w) {
    if (!w) return;
    const i = (node.widgets || []).indexOf(w);
    if (i >= 0) node.widgets.splice(i, 1);
    try { w.onRemove?.(); } catch (e) { /* ignore */ }
    const store = frontendStore("widgetValue");
    if (w.widgetId && store?.deleteWidget) {
        try { store.deleteWidget(w.widgetId); } catch (e) { /* ignore */ }
    }
    syncNodeWidgetOrder(node);
}

// Add a row back: splice it into node.widgets and (re-)register it in the
// store so the Vue row reappears at the spliced position.
function addWidgetRow(node, w, after) {
    if (!w || !node.widgets || node.widgets.includes(w)) return;
    let i = after ? node.widgets.indexOf(after) + 1 : node.widgets.length;
    if (i < 1) i = node.widgets.length;
    node.widgets.splice(i, 0, w);
    try { w.setNodeId?.(node.id); } catch (e) { /* ignore */ }
    // DOM widgets keep a parallel visibility store.
    const domStore = frontendStore("domWidget");
    if (w.onRemove && domStore?.registerWidget) {
        try { domStore.registerWidget(w); } catch (e) { /* ignore */ }
    }
    syncNodeWidgetOrder(node);
}

// Clear the node's preview entry (the store-driven preview overlay).
function clearNodePreview(node) {
    node.imgs = undefined;
    const store = frontendStore("nodeOutput");
    if (store?.removeNodeOutputsForNode) {
        try { store.removeNodeOutputsForNode(node); } catch (e) { /* ignore */ }
    }
}

// Set the node's preview entry for the current file combo value - the
// core's own combo callback does exactly this; re-run it when the
// load/crop toggles bring the native preview back (the core's callback
// only fires on a file change, and its upload widget may not have
// injected yet when the rows are set up).
function setFilePreview(node) {
    const value = getWidgetValue(node, "image");
    node.imgs = undefined;
    const store = frontendStore("nodeOutput");
    if (value && store?.setNodeOutputs) {
        try { store.setNodeOutputs(node, String(value)); } catch (e) { /* ignore */ }
    }
    try { node.graph?.setDirtyCanvas(true, true); } catch (e) { /* ignore */ }
}

// Build a horizontal button switcher (low_vram style). Returns the element.
function buildModeSwitch(node) {
    const el = document.createElement("div");
    el.className = "gibby-mode-switch";
    el.style.cssText = "display:flex; gap:4px; padding:0;";

    for (const [value, label] of MODES) {
        const b = document.createElement("button");
        b.textContent = label;
        b._gibbyValue = value;
        b.style.cssText =
            "flex:1; padding:1px 4px; font-size:11px; cursor:pointer; " +
            "background:#2a2a2a; color:#ccc; border:1px solid #444; " +
            "border-radius:3px; font-family:inherit; line-height:1.1;";
        const isSel = () => {
            const n = liveNode(node.id);
            return n ? getWidgetValue(n, "mode") === value : false;
        };
        b.addEventListener("mouseenter", () => { if (!isSel() && !b.classList.contains("gibby-mode-active")) b.style.background = "#3a3a3a"; });
        b.addEventListener("mouseleave", () => { if (!isSel() && !b.classList.contains("gibby-mode-active")) b.style.background = "#2a2a2a"; });
        b.addEventListener("mousedown", (e) => e.stopPropagation());
        b.addEventListener("click", (e) => {
            e.stopPropagation();
            // Resolve the live instance in the graph on screen (survives
            // tab switches and undo/redo, which rebuild the node).
            const n = liveNode(node.id);
            if (!n) return;
            setWidgetValue(n, "mode", value);
            refreshModeVisibility(n);
        });
        el.appendChild(b);
    }

    // Keep button highlight in sync with the widget value.
    const labelOf = (v) => MODES.find(([x]) => x === v)?.[1] || "";
    el._gibbyPaint = () => {
        const n = liveNode(node.id);
        if (!n) return;
        const modeVal = getWidgetValue(n, "mode");
        const curLabel = labelOf(modeVal);
        
        // Paint this node's own buttons directly - el is this node's switcher.
        // (Re-querying the DOM by node-type returns the first matching node,
        // which mis-highlights sibling nodes.)
        for (const b of el.children) {
            const on = b.textContent === curLabel;
            b.classList.toggle("gibby-mode-active", on);
            // Set colors directly to ensure proper display
            if (on) {
                b.style.background = "#4a9eff";
                b.style.color = "#fff";
            } else {
                b.style.background = "#2a2a2a";
                b.style.color = "#ccc";
            }
        }
    };
    return el;
}

// Show/hide rows per mode: keep AR shows only megapixels; custom shows the
// width x height fields; the AR modes show their ratio + megapixels. Resize
// widgets (upscale_method/keep_proportion/pad_color/crop_position) and
// batch_size/flux2_latent stay visible regardless of a linked media input.
function refreshModeVisibility(node) {
    const mode = getWidgetValue(node, "mode") || "aspect_ratio";

    // Standard Vue-rendered rows (hidden via options.hidden).
    setWidgetHidden(findWidget(node, "width"), mode !== "custom");
    setWidgetHidden(findWidget(node, "height"), mode !== "custom");
    setWidgetHidden(findWidget(node, "aspect_ratio"), mode !== "aspect_ratio");
    setWidgetHidden(findWidget(node, "x"), mode !== "custom_aspect_ratio");
    setWidgetHidden(findWidget(node, "y"), mode !== "custom_aspect_ratio");
    setWidgetHidden(findWidget(node, "megapixels"), !["keep_ar", "aspect_ratio", "custom_aspect_ratio"].includes(mode));

    const els = node._gibbyResElements;
    if (!els) return;

    // Repaint the switcher highlight.
    if (els.modeSwitch._gibbyPaint) els.modeSwitch._gibbyPaint();

    // The mode switch writes the value directly (no callback), so re-fit the
    // locked crop box here.
    if (node._gibbyLoadGroup && getWidgetValue(node, "load_image") && getWidgetValue(node, "crop_image")) {
        refitCropBox(node);
    }

    // Force a redraw so the layout system re-measures after rows show/hide.
    try { node.graph?.setDirtyCanvas(true, true); } catch (e) { /* ignore */ }
    // A mode change shows/hides the size rows; refit the node height.
    refitNode(node);

    // The Vue widget list only reads options.hidden when it re-renders; a
    // same-tick showAdvanced toggle-and-restore forces that pass without
    // changing the node's actual advanced state.
    const prev = node.showAdvanced;
    node.showAdvanced = !prev;
    node.showAdvanced = prev;
}

function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
}

// The /view URL of the loaded file. The combo value may be "name",
// "sub/name" or "name [type]".
function imagePreviewUrl(node) {
    const value = getWidgetValue(node, "image");
    if (!value) return null;
    let v = String(value).replace(/\s*\[\w+\]$/, "");
    let subfolder = "";
    const slash = v.indexOf("/");
    if (slash >= 0) {
        subfolder = v.slice(0, slash);
        v = v.slice(slash + 1);
    }
    if (!v) return null;
    const params = new URLSearchParams({ filename: v, subfolder, type: "input" });
    params.set("_", Date.now().toString());
    return "/view?" + params.toString();
}

// The natural aspect ratio of the loaded file (from the crop preview's
// <img> once it has loaded); null before that.
function cropImageAR(node) {
    const g = node._gibbyLoadGroup;
    const nat = g && g.cropPreviewEl && g.cropPreviewEl._gibbyNatural;
    return nat ? nat.w / nat.h : null;
}

// The AR the crop box is locked to with "selected": the node's size mode -
// keep AR the image's, custom the width/height, aspect ratio the preset
// ("closest to image" the image's), custom AR the manual x/y. Swap
// dimensions inverts it (the backend does the same). null for the
// free-form "custom" box.
function lockedRatio(node) {
    if (getWidgetValue(node, "crop_ar") !== "selected") return null;
    const mode = getWidgetValue(node, "mode") || "aspect_ratio";
    const imgAR = cropImageAR(node) || 1;
    let ratio;
    if (mode === "keep_ar") ratio = imgAR;
    else if (mode === "custom") {
        const w = getWidgetValue(node, "width");
        const h = getWidgetValue(node, "height");
        ratio = h ? w / h : imgAR;
    } else if (mode === "aspect_ratio") {
        const ar = getWidgetValue(node, "aspect_ratio");
        if (ar === CLOSEST_RATIO) ratio = imgAR;
        else {
            const p = ASPECT_RATIOS[ar];
            ratio = p ? p[0] / p[1] : imgAR;
        }
    } else {
        const x = getWidgetValue(node, "x");
        const y = getWidgetValue(node, "y");
        ratio = y ? x / y : imgAR;
    }
    return getWidgetValue(node, "swap_dimensions") ? 1 / ratio : ratio;
}

// Keep the box's center, fit it to the locked AR within the image.
function refitCropBox(node) {
    const g = node._gibbyLoadGroup;
    const ratio = lockedRatio(node);
    const nat = g && g.cropPreviewEl && g.cropPreviewEl._gibbyNatural;
    if (!g || !g.region || !ratio || !nat) return;
    const v = g.region.value || {};
    const x = v.x || 0, y = v.y || 0, bw = v.width || 512, bh = v.height || 512;
    const cx = x + bw / 2, cy = y + bh / 2;
    let nw = bw, nh = nw / ratio;
    if (nh > nat.h) { nh = nat.h; nw = nh * ratio; }
    if (nw > nat.w) { nw = nat.w; nh = nw / ratio; }
    g.region.value = {
        x: Math.round(clamp(cx - nw / 2, 0, nat.w - nw)),
        y: Math.round(clamp(cy - nh / 2, 0, nat.h - nh)),
        width: Math.round(nw),
        height: Math.round(nh),
    };
    g.cropPreviewEl._gibbyPaint?.();
}

// The crop preview: the loaded file with a draggable, resizable box (like
// the core Crop Image (Context)). The box writes image-pixel coordinates
// into the crop_region widget.
// The preview row's fixed height: the image letterboxes inside (like the
// core crop editor's fixed container), so resizing the node never changes
// the row height and a manual node height is never fought.
const CROP_PREVIEW_H = 222;

function buildCropPreview(node) {
    const wrap = document.createElement("div");
    wrap.style.cssText =
        "position:relative; margin:0; background:#111; border:1px solid #333; " +
        "border-radius:3px; overflow:hidden; line-height:0; box-sizing:border-box; " +
        "height:" + CROP_PREVIEW_H + "px; display:flex; " +
        "align-items:center; justify-content:center;";
    const img = document.createElement("img");
    img.style.cssText =
        "display:block; user-select:none; -webkit-user-drag:none;";
    const box = document.createElement("div");
    box.style.cssText =
        "position:absolute; box-sizing:border-box; border:1px solid #4a9eff; " +
        "background:rgba(74,158,255,0.12); cursor:move;";
    wrap.appendChild(img);
    wrap.appendChild(box);

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

    // Display the image at its natural size capped by the container's
    // height (the width follows the AR) and shrunk to fit the container's
    // width: resizing the node's width only changes the black padding,
    // like resizing its height does.
    function fitImg() {
        const nat = wrap._gibbyNatural;
        if (!nat || !wrap.clientWidth || !wrap.clientHeight) return;
        const ar = nat.w / nat.h;
        let ch = Math.min(nat.h, wrap.clientHeight), cw = ch * ar;
        if (cw > wrap.clientWidth) { cw = wrap.clientWidth; ch = cw / ar; }
        img.style.width = Math.round(cw) + "px";
        img.style.height = Math.round(ch) + "px";
    }

    // The rendered image rect inside the wrap, in CSS px (the box's styles
    // are CSS px; the overlay follows the canvas zoom, so bounding rects
    // would mix screen px in). The image is centered in the fixed
    // container (the flex layout), the box is offset the same way.
    function imgBox() {
        const ww = wrap.clientWidth, wh = wrap.clientHeight;
        const cw = img.clientWidth, ch = img.clientHeight;
        const nat = wrap._gibbyNatural || { w: 1, h: 1 };
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
    function paint() {
        fitImg();
        // No image loaded yet, or the row is hidden (zero size): nothing to
        // anchor the box to.
        if (!wrap._gibbyNatural || !img.clientWidth) { box.style.display = "none"; return; }
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
    wrap._gibbyPaint = paint;
    function setRect(r) {
        const n = liveNode(node.id);
        const g = n && n._gibbyLoadGroup;
        if (g && g.region) {
            g.region.value = {
                x: Math.round(r.x), y: Math.round(r.y),
                width: Math.round(r.w), height: Math.round(r.h),
            };
        }
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
        if (e.target === box) startDrag(e, "move");
    });
    for (const h of handles) {
        h.addEventListener("pointerdown", (e) => {
            e.stopPropagation();
            const n = liveNode(node.id);
            if (!n) return;
            // With a locked AR only the corners resize.
            if (lockedRatio(n) && h._gibbyDir.length !== 2) return;
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
        const ratio = lockedRatio(liveNode(node.id));
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
    for (const el of [box, ...handles]) {
        el.addEventListener("pointermove", onPointerMove);
        el.addEventListener("pointerup", onPointerUp);
        el.addEventListener("pointercancel", onPointerUp);
    }

    img.onload = () => {
        wrap._gibbyNatural = { w: img.naturalWidth, h: img.naturalHeight };
        const n = liveNode(node.id);
        if (n) {
            refitCropBox(n);
            paint();
        }
    };

    // Show the current file (called when the combo changes and when the row
    // is re-added).
    wrap._gibbyLoad = () => {
        const url = imagePreviewUrl(liveNode(node.id) || node);
        if (url) img.src = url;
        paint();
    };
    // Resizing the node resizes the image (the fixed container letterboxes
    // it); repaint the box to follow (like the core crop editor's resize
    // observer on its container). The row height is fixed, so no refit - a
    // manual node height is never fought.
    new ResizeObserver(() => {
        paint();
    }).observe(wrap);

    return wrap;
}

// Add and remove the load image group's rows (like the context loader's
// mode widgets): splice node.widgets and keep the widget store in sync so
// the Vue rows follow. Removed rows are not serialized into the prompt, so
// the backend falls back to their schema defaults. The image, crop image,
// crop preview and crop region rows never leave node.widgets and are hidden
// instead: the first two and the region are required serialized inputs
// (removing them drops their values from the prompt and the queue fails),
// and the preview is a DOM widget whose floating element must stay laid out.
function syncLoadGroup(node) {
    if (!node || node._removed || !node._gibbyLoadGroup) return;
    const g = node._gibbyLoadGroup;
    const load = !!getWidgetValue(node, "load_image");
    const crop = load && !!getWidgetValue(node, "crop_image");
    const widgets = node.widgets || [];

    // The upload button is injected by the core; resolve it lazily in case
    // it was not there yet when the node was set up.
    if (!g.upload) {
        g.upload = findWidget(node, "upload");
        if (g.upload) g.addable.push(g.upload);
    }

    // The group rows in display order, after the load_image toggle; the
    // whole group hides with it. The image/crop image/region rows are always
    // present (see above) and hidden with the toggle; the upload and AR
    // toggle rows are removed with it.
    let group = [g.image];
    if (load) group.push(g.upload);
    group.push(g.cropImage);
    if (crop) group.push(g.arToggle, g.cropPreview);
    group.push(g.region);
    group = group.filter(Boolean);

    for (const w of [...widgets]) {
        if (g.addable.includes(w) && !group.includes(w)) {
            removeWidgetRow(node, w);
        }
    }
    // Keep the group's rows in place after the load_image toggle: missing
    // rows get added, out-of-place rows get moved (turning crop on must not
    // jump above the file combo and the crop toggle).
    let anchor = findWidget(node, "load_image");
    for (const w of group) {
        const i = widgets.indexOf(w);
        const want = anchor ? widgets.indexOf(anchor) + 1 : -1;
        if (i !== want) {
            if (i >= 0) widgets.splice(i, 1);
            addWidgetRow(node, w, anchor);
        }
        anchor = w;
    }
    // The image, crop image, preview and region stay in node.widgets;
    // hiding them (instead of removing them) keeps them in the layout pass
    // and in the prompt (see above).
    if (g.image) setWidgetHidden(g.image, !load);
    if (g.cropImage) setWidgetHidden(g.cropImage, !load);
    if (g.cropPreview) setWidgetHidden(g.cropPreview, !crop);
    if (g.region) setWidgetHidden(g.region, !crop);

    // Swap the native file preview for the crop preview. The overlay is
    // store-driven, so the store entry is cleared/re-set to make it react;
    // hideOutputImages (the core crop nodes' own flag) stays in sync too.
    const nativePreview = load && !crop;
    node.hideOutputImages = !nativePreview;
    if (nativePreview) {
        setFilePreview(node);
    } else {
        clearNodePreview(node);
        if (crop) g.cropPreviewEl?._gibbyLoad?.();
    }

    if (crop) refitCropBox(node);
    // Keep the native toggle in sync with the (serialized) combo value.
    const arToggle = findWidget(node, "gibby_crop_ar");
    if (arToggle) arToggle.value = getWidgetValue(node, "crop_ar") === "selected";

    try { node.graph?.setDirtyCanvas(true, true); } catch (e) { /* ignore */ }
    // Force a Vue widget-list re-render (same trick as refreshModeVisibility).
    const prev = node.showAdvanced;
    node.showAdvanced = !prev;
    node.showAdvanced = prev;
    refitNode(node);
}

// Refit the node through the core layout pass: it assigns the DOM widgets'
// positions (widget.y) that their floating elements track, and grows or
// shrinks the node with the visible rows (the core dynamic widgets' refit).
function refitNode(node) {
    if (!node || node._removed || !node.size) return;
    try {
        node.size = [node.size[0], node.computeSize([...node.size])[1]];
    } catch (e) { /* ignore */ }
}

function setupResolutionNode(node) {
    if (!node || node._removed || node._gibbyResReady) return;

    // All standard widgets must exist before we can transform them.
    const modeW = findWidget(node, "mode");
    const widthW = findWidget(node, "width");
    const heightW = findWidget(node, "height");
    if (!modeW || !widthW || !heightW) {
        // Widgets not ready yet - retry shortly (fast_groups pattern).
        setTimeout(() => setupResolutionNode(node), 50);
        return;
    }

    node._gibbyResReady = true;
    node._gibbyResElements = {};

    // 1. Mode switcher: hide the combo row, add a DOM toggle at the top.
    setWidgetHidden(modeW, true);
    const modeEl = buildModeSwitch(node);
    const modeDom = node.addDOMWidget("gibby_mode", "GIBBY_MODE_SWITCH", modeEl, {
        getValue: () => getWidgetValue(node, "mode"),
        setValue: (v) => setWidgetValue(node, "mode", v),
    });
    if (modeDom) {
        // Keep the switcher out of the serialized widget list: it mirrors
        // "mode", and its value as the first positional widgets_values entry
        // shifts every other value by one slot on graph reload.
        modeDom.serialize = false;
        modeDom.computeLayoutSize = () => ({ minHeight: modeEl.style.display === "none" ? 0 : 26, minWidth: 1 });
    }

    // Keep width/height and x/y as standard ComfyUI inputs
    // (shown/hidden per mode via options.hidden).

    node._gibbyResElements.modeSwitch = modeEl;

    // 2. Load image group: the file combo (native upload button, preview and
    //    mask editor), the crop toggle, the crop AR toggle and the crop
    //    preview with its drawable box.
    // The crop AR toggle is a native toggle widget (label + Custom/Selected
    // segmented control); the hidden crop_ar combo stays the serialized
    // source of the value.
    const arToggleW = node.addWidget("toggle", "gibby_crop_ar", false, function (v) {
        const n = liveNode(node.id);
        if (!n) return;
        const combo = findWidget(n, "crop_ar");
        if (combo) combo.value = v ? "selected" : "custom";
        if (getWidgetValue(n, "load_image") && getWidgetValue(n, "crop_image")) {
            refitCropBox(n);
        }
    }, { on: "Selected", off: "Custom" });
    if (arToggleW) {
        arToggleW.serialize = false;
        arToggleW.label = "Crop AR";
    }
    const cropEl = buildCropPreview(node);
    const cropDom = node.addDOMWidget("gibby_crop_preview", "GIBBY_CROP_PREVIEW", cropEl, {
        getValue: () => getWidgetValue(node, "crop_region"),
        setValue: (v) => setWidgetValue(node, "crop_region", v),
    });
    if (cropDom) {
        cropDom.serialize = false;
        cropDom.computeLayoutSize = () => ({ minHeight: CROP_PREVIEW_H, minWidth: 1 });
    }

    const loadW = findWidget(node, "load_image");
    const imageW = findWidget(node, "image");
    const cropImageW = findWidget(node, "crop_image");
    const cropArW = findWidget(node, "crop_ar");
    const regionW = findWidget(node, "crop_region");
    const uploadW = findWidget(node, "upload");

    // The crop AR combo row is replaced by the labeled toggle (like the mode
    // combo by its switcher): keep the combo for serialization, hide the row.
    if (cropArW) setWidgetHidden(cropArW, true);

    node._gibbyLoadGroup = {
        image: imageW,
        upload: uploadW,
        cropImage: cropImageW,
        cropAr: cropArW,
        arToggle: arToggleW,
        cropPreview: cropDom,
        cropPreviewEl: cropEl,
        region: regionW,
        byName: {
            load_image: loadW,
            image: imageW,
            crop_image: cropImageW,
            crop_ar: cropArW,
            crop_region: regionW,
        },
    };
    // The image, crop image, crop preview and region rows are not addable:
    // they stay in node.widgets and are hidden instead (see syncLoadGroup).
    node._gibbyLoadGroup.addable = [
        uploadW, arToggleW,
    ].filter(Boolean);

    // File combo: a file change updates the store-driven preview only
    // while the file combo is in use (load on, crop off); with crop on it
    // refreshes the crop preview instead, and with load off it clears it.
    if (imageW && !imageW._gibbyWrapped) {
        imageW._gibbyWrapped = true;
        imageW.callback = function (...args) {
            const n = liveNode(node.id);
            const grp = n && n._gibbyLoadGroup;
            if (!n || !grp) return undefined;
            const load = !!getWidgetValue(n, "load_image");
            const crop = load && !!getWidgetValue(n, "crop_image");
            if (!load) {
                clearNodePreview(n);
            } else if (crop) {
                clearNodePreview(n);
                grp.cropPreviewEl?._gibbyLoad?.();
            } else {
                setFilePreview(n);
            }
            return undefined;
        };
    }
    // The load/crop toggles add and remove their rows.
    for (const w of [loadW, cropImageW]) {
        if (w && !w._gibbyWrapped) {
            w._gibbyWrapped = true;
            const prev = w.callback;
            w.callback = function (...args) {
                if (prev) {
                    try { prev.apply(this, args); } catch (e) { /* ignore */ }
                }
                const n = liveNode(node.id);
                if (n) syncLoadGroup(n);
            };
        }
    }
    // The size-mode widgets feed the "selected" crop AR.
    for (const name of ["mode", "width", "height", "aspect_ratio", "x", "y", "swap_dimensions"]) {
        const w = findWidget(node, name);
        if (w && !w._gibbyArHooked) {
            w._gibbyArHooked = true;
            const prev = w.callback;
            w.callback = function (...args) {
                if (prev) {
                    try { prev.apply(this, args); } catch (e) { /* ignore */ }
                }
                const n = liveNode(node.id);
                if (n && n._gibbyLoadGroup && getWidgetValue(n, "load_image") && getWidgetValue(n, "crop_image")) {
                    refitCropBox(n);
                }
            };
        }
    }

    // Editing the x/y/width/height row updates the region value; repaint
    // the crop box to follow it.
    if (regionW && !regionW._gibbyWrapped) {
        regionW._gibbyWrapped = true;
        const prev = regionW.callback;
        regionW.callback = function (...args) {
            if (prev) {
                try { prev.apply(this, args); } catch (e) { /* ignore */ }
            }
            const n = liveNode(node.id);
            if (n && n._gibbyLoadGroup && getWidgetValue(n, "load_image") && getWidgetValue(n, "crop_image")) {
                n._gibbyLoadGroup.cropPreviewEl?._gibbyPaint?.();
            }
        };
    }

    // Row order: the switcher on top, the size rows, then the load image
    // group at the bottom - the crop rows under the crop toggle, the
    // x/y/width/height row under the crop preview.
    const loadGroup = [loadW, imageW, uploadW, cropImageW, arToggleW, cropDom, cropArW, regionW].filter(Boolean);
    const ordered = [modeDom];
    for (const w of node.widgets) {
        if (w !== modeDom && !loadGroup.includes(w) && !ordered.includes(w)) {
            ordered.push(w);
        }
    }
    for (const w of loadGroup) {
        if (!ordered.includes(w)) ordered.push(w);
    }
    node.widgets.splice(0, node.widgets.length, ...ordered);
    syncNodeWidgetOrder(node);

    refreshModeVisibility(node);
    syncLoadGroup(node);
    // Delay paint to ensure DOM is ready
    setTimeout(() => {
        refreshModeVisibility(node);
        syncLoadGroup(node);
    }, 100);
}

app.registerExtension({
    name: "GibbyNodes.resolutionLatent",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;

        // Add CSS for mode button highlighting
        if (!document.getElementById("gibby-mode-css")) {
            const style = document.createElement("style");
            style.id = "gibby-mode-css";
            style.textContent = `
                .gibby-mode-switch button {
                    transition: none !important;
                }
                .gibby-mode-active {
                    background-color: #4a9eff !important;
                    color: #fff !important;
                }
            `;
            document.head.appendChild(style);
        }

        const origOnAdded = nodeType.prototype.onAdded;
        nodeType.prototype.onAdded = function () {
            setupResolutionNode(this);
            if (origOnAdded) {
                try { origOnAdded.apply(this, arguments); } catch (e) { /* ignore */ }
            }
        };

        // Also hook onConfigure so existing workflows get the DOM UI.
        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            if (origOnConfigure) {
                try { origOnConfigure.apply(this, arguments); } catch (e) { /* ignore */ }
            }
            setupResolutionNode(this);
        };

        // Re-evaluate row visibility when a link is made or broken.
        const origOnConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            if (origOnConnectionsChange) {
                try { origOnConnectionsChange.apply(this, arguments); } catch (e) { /* ignore */ }
            }
            refreshModeVisibility(this);
        };

        // Widget values restore positionally by default, and lists saved
        // while the DOM switcher sat in node.widgets carry its value as the
        // first positional entry, shifting every other value by one slot.
        // Restore by name when the data carries widgets_values_named (which
        // also repairs those shifted lists); fall back to the canonical
        // positional order for legacy data without the named map.
        const origConfigure = nodeType.prototype.configure;
        nodeType.prototype.configure = function (data) {
            origConfigure.apply(this, arguments);

            if (data && data.widgets_values_named) {
                for (const name of CANONICAL_WIDGETS) {
                    let w = findWidget(this, name);
                    // Rows currently removed from node.widgets live on the
                    // load group; restore there so they come back with the
                    // saved values.
                    if (!w && this._gibbyLoadGroup) w = this._gibbyLoadGroup.byName[name];
                    const v = data.widgets_values_named[name];
                    if (w && v !== null && v !== undefined) {
                        w.value = v;
                    }
                }
            } else if (data && Array.isArray(data.widgets_values)) {
                const vals = data.widgets_values;
                for (let i = 0; i < Math.min(vals.length, CANONICAL_WIDGETS.length); i++) {
                    const w = findWidget(this, CANONICAL_WIDGETS[i]);
                    if (w && vals[i] !== null && vals[i] !== undefined) {
                        w.value = vals[i];
                    }
                }
            }

            refreshModeVisibility(this);
            // The saved toggles may differ from the schema defaults the group
            // started with; re-apply the rows.
            syncLoadGroup(this);
        };
    },
});

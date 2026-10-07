// ---------------------------------------------------------------------------
// Gibby Nodes - Resize Image / Empty Latent (Context) frontend
//
// The node's standard widgets are declared in the backend schema so their
// values serialize into prompts normally. This file upgrades one of them to
// a custom DOM UI using ComfyUI's own node.addDOMWidget() (NOT hand-drawn
// canvas widgets):
//   - "mode" combo          -> horizontal toggle switch (custom / aspect ratio / custom AR, plus keep AR when media is linked)
//   - "crop_ar" combo        -> native labeled toggle (Custom / Selected) above the crop preview
// The enable preview toggle (above the load image toggle) gates the
// executed temp preview: off, the backend writes no temp file at all and
// the node hides the last run's preview; it is hidden while load image
// and/or crop image is on, where the preview is always written (the file
// preview and the crop preview are the image UI then, and downstream crop
// views read the context's image from it).
// Rows are shown/hidden per mode via widget.options.hidden (the same flag the
// new frontend's advanced-widget toggle uses). The load image group
// (file combo + upload) and the crop group (crop toggle, crop AR toggle,
// crop preview, crop region) are instead added to and removed from
// node.widgets physically, with the widget store kept in sync the same way
// the core's dynamic widgets do (the Vue rows follow the store, not
// node.widgets): removed rows are not serialized into the prompt, so the
// backend falls back to their schema defaults. The crop group is
// independent of the load group: the crop preview shows the image the resize
// uses (the loaded file, else the connected image's preview, else the
// connected context node's last-run image).
//
// With load and crop off the node's executed temp preview (the backend's
// ui.PreviewImage) is the image UI - like the core Preview Image, a mask can
// be drawn on it: the core mask editor rewrites the file combo to its
// clipspace upload, and the backend uses the file's alpha as the mask. An
// erased mask (the editor rewrites the combo to a fresh fully-opaque file on
// every save, erases included) leaves the executed preview in place and
// clears the file combo - the file is a stale copy of the image it was
// painted on, and the editor prioritizes the combo over the node's current
// image, so a fresh mask session would otherwise paint onto that copy.
// ---------------------------------------------------------------------------

import { app } from "../../../scripts/app.js";
import {
    buildCropPreview,
    liveNode,
    CROP_PREVIEW_MIN_H,
} from "../crop_preview.js";

const NODE_TYPE = "Gibby_EmptyLatent_Resolution";

const MODES = [
    ["keep_ar", "Keep AR"],
    ["custom", "Custom"],
    ["aspect_ratio", "Aspect Ratio"],
    ["custom_aspect_ratio", "Custom AR"],
];

// Standard widget names in schema order (excluding the DOM switcher, which is
// not serialized). configure() maps legacy positional widgets_values onto
// these; with widgets_values_named present the order is irrelevant.
// enable_preview is deliberately not in this list: it postdates the
// positional order, so legacy lists without widgets_values_named must not
// shift onto it (configure's named branch restores it separately).
const CANONICAL_WIDGETS = [
    "mode", "width", "height", "aspect_ratio", "x", "y", "megapixels",
    "scale_factor", "upscale_method", "keep_proportion", "pad_color",
    "crop_position", "swap_dimensions", "multiple", "batch_size",
    "flux2_latent", "load_image", "image", "crop_image", "crop_ar",
    "crop_region"
];

// Same presets as the backend's aspect ratio mode (portrait only, named by
// the ratio); used to resolve the "selected" crop box AR.
const ASPECT_RATIOS = {
    "1:1": [1, 1],
    "2:3": [2, 3],
    "3:4": [3, 4],
    "9:16": [9, 16],
    "9:21": [9, 21],
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
    // A detached clone (copy/duplicate) shares the original's id - touching
    // the store would drop the original node's preview.
    if (!node.graph) return;
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
    // A detached clone (copy/duplicate) shares the original's id - touching
    // the store would overwrite the original node's preview.
    if (!node.graph) return;
    const store = frontendStore("nodeOutput");
    if (value && store?.setNodeOutputs) {
        try { store.setNodeOutputs(node, String(value)); } catch (e) { /* ignore */ }
    }
    try { node.graph?.setDirtyCanvas(true, true); } catch (e) { /* ignore */ }
}

// The node connected to the named input (through a subgraph proxy when
// needed) - the core crop editor's own upstream lookup.
function upstreamNode(node, name) {
    const idx = (node.inputs || []).findIndex((i) => i.name === name);
    if (idx < 0) return null;
    // A detached clone (copy/duplicate) has no graph - getInputNode throws
    // NullGraphError there, and the clone has no links anyway.
    if (!node.graph) return null;
    let n = node.getInputNode?.(idx);
    if (!n) return null;
    if (n.isSubgraphNode?.()) {
        const link = node.getInputLink?.(idx);
        if (!link) return null;
        n = n.resolveSubgraphOutputLink?.(link.origin_slot)?.outputNode ?? null;
    }
    return n;
}

// PreviewImage's temp files (ComfyUI_temp_*) mark an executed output, as
// opposed to the file-combo preview of a loaded file.
function isExecutedPreview(url) {
    try {
        return new URL(url, location.origin).searchParams.get("filename")?.startsWith("ComfyUI_temp_") === true;
    } catch (e) { return false; }
}

// The URL of the image the crop preview shows: the same image the resize
// uses - the loaded file (load on), else the connected image input's
// preview (a Load Image's file, or the last run's output - nothing to show
// while it has none), and only without one the connected context node's
// preview. The context's image is a runtime tensor, so only its executed
// preview (a temp file) is shown - it changes on a run, not on a file
// change. The last run's preview is kept until the next run: a file change
// in the context node replaces the store entry with the file's preview,
// which is not the crop's image.
function cropPreviewUrl(node) {
    if (getWidgetValue(node, "load_image")) return imagePreviewUrl(node);
    const store = frontendStore("nodeOutput");
    if (!store?.getNodeImageUrls) return null;
    const imgNode = upstreamNode(node, "connected_image");
    if (imgNode) {
        // The last run's output (kept across store refreshes) is the crop's
        // actual input when the upstream node executes; a core Load Image
        // never has one, so the file preview stands.
        const urls = store.getNodeImageUrls(imgNode) || [];
        for (const u of urls) {
            if (isExecutedPreview(u)) {
                node._gibbyLastTemp = { url: u, ctx: imgNode.id };
                return u;
            }
        }
        const last = node._gibbyLastTemp;
        if (last && last.ctx === imgNode.id) return last.url;
        return urls.length ? urls[0] : null;
    }
    const ctxNode = upstreamNode(node, "context");
    if (!ctxNode) {
        node._gibbyLastTemp = null;
        return null;
    }
    for (const u of store.getNodeImageUrls(ctxNode) || []) {
        if (isExecutedPreview(u)) {
            node._gibbyLastTemp = { url: u, ctx: ctxNode.id };
            return u;
        }
    }
    const last = node._gibbyLastTemp;
    return last && last.ctx === ctxNode.id ? last.url : null;
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
    if (node._gibbyLoadGroup && getWidgetValue(node, "crop_image")) {
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

// The file the core mask editor uploads and rewrites the combo to when a
// mask is drawn on the preview (clipspace-painted-masked-<ts>.png).
function isMaskEditorFile(value) {
    if (!value) return false;
    const v = String(value).replace(/\s*\[\w+\]$/, "");
    return v.slice(v.lastIndexOf("/") + 1).startsWith("clipspace-painted-masked-");
}

// The last file the combo pointed at that is not a mask editor upload -
// the original a drawn mask goes back to with crop on (the box is the
// mask there, so the drawn one is discarded).
function trackOriginalFile(node, value) {
    const v = value !== undefined ? value : getWidgetValue(node, "image");
    if (v && !isMaskEditorFile(v)) node._gibbyOriginalFile = v;
}

// Whether the clipspace file the combo points at holds a drawn mask: any
// non-opaque alpha pixel (the backend's _drawn_mask does the same). The
// combo can't tell it - the core editor rewrites it to a fresh file on
// every save, erases included, and that rewrite skips the widget callback
// - so the file's alpha decides. Checked once per file, cached on the
// node; null = not checked yet (the file preview stands until it lands).
async function checkDrawnMask(node) {
    const v = getWidgetValue(node, "image");
    const name = v && isMaskEditorFile(v) ? String(v).replace(/\s*\[\w+\]$/, "") : null;
    if (!name) {
        node._gibbyMaskFile = null;
        node._gibbyMaskEmpty = null;
        return;
    }
    if (node._gibbyMaskFile === name && node._gibbyMaskEmpty !== null) return;
    node._gibbyMaskFile = name;
    node._gibbyMaskEmpty = null;
    try {
        // The editor's upload sends no subfolder, so the file lands in the
        // input root (the "clipspace" in the ref is nominal)
        const params = new URLSearchParams({ filename: name, type: "input" });
        const resp = await fetch("/view?" + params.toString());
        if (!resp.ok) {
            // The backend treats a missing file as no mask
            if (node._gibbyMaskFile === name) node._gibbyMaskEmpty = true;
            return;
        }
        const bmp = await createImageBitmap(await resp.blob());
        const canvas = document.createElement("canvas");
        canvas.width = bmp.width;
        canvas.height = bmp.height;
        const ctx = canvas.getContext("2d", { willReadFrequently: true });
        ctx.drawImage(bmp, 0, 0);
        const data = ctx.getImageData(0, 0, bmp.width, bmp.height).data;
        let empty = true;
        for (let i = 3; i < data.length; i += 4) {
            if (data[i] < 255) { empty = false; break; }
        }
        if (node._gibbyMaskFile === name) node._gibbyMaskEmpty = empty;
    } catch (e) { /* keep null: the file preview stands until a check lands */ }
}

// The verdict checkDrawnMask cached for the combo's current file
// (true = erased, false = drawn, null = not decided yet): the verdict is
// per file - the core editor makes a fresh timestamped file on every
// save, so one cached for another file says nothing about this one.
function maskEmptyOf(node) {
    const value = getWidgetValue(node, "image");
    if (!isMaskEditorFile(value)) return null;
    const name = String(value).replace(/\s*\[\w+\]$/, "");
    return node._gibbyMaskFile === name ? node._gibbyMaskEmpty : null;
}

// Whether the file preview (the loaded file, or a mask drawn on the
// executed preview) stands for the executed output: load on, or a mask
// editor file whose alpha still holds a drawn mask (maskEmptyOf; an
// erased mask leaves the executed output as the preview, like a plain
// run).
function filePreviewActive(node) {
    return !!getWidgetValue(node, "load_image") ||
        (isMaskEditorFile(getWidgetValue(node, "image")) &&
         maskEmptyOf(node) !== true);
}

// The mask has been erased: the clipspace file is a stale copy of the
// image it was painted on. Drop it from the node's output entry (the core
// editor's save put it there over the run's output, erases included), and
// clear the file combo - the mask editor reads the combo (the widget value
// store first) and prioritizes it over the node's current image, so a
// fresh mask session would otherwise paint onto the stale copy instead of
// the image the preview shows now.
function resetErasedMask(node) {
    const store = frontendStore("nodeOutput");
    if (store?.getNodeOutputs && store.removeNodeOutputsForNode) {
        const name = String(getWidgetValue(node, "image") || "")
            .replace(/\s*\[\w+\]$/, "");
        let img;
        try {
            img = store.getNodeOutputs(node)?.images?.[0];
        } catch (e) { /* ignore */ }
        if (img && img.filename === name && img.type === "input") {
            node.imgs = undefined;
            try {
                store.removeNodeOutputsForNode(node);
            } catch (e) { /* ignore */ }
        }
    }
    const w = findWidget(node, "image");
    if (w) w.value = "";
    // The store is what the editor's getNodeWidgetValue reads: a direct
    // widget write alone would leave the stale file in it (the core's
    // setNodeWidgetValue writes both).
    const widgets = frontendStore("widgetValue");
    const graphId = node.graph?.rootGraph?.id;
    if (widgets?.setValue && graphId) {
        try {
            widgets.setValue(`${graphId}:${encodeURIComponent(String(node.id))}:image`, "");
        } catch (e) { /* ignore */ }
    }
}

// Restore the file preview in place of the executed output (the core's
// combo callback does this on a file change): the loaded file in load
// mode, or a mask drawn on the executed preview in default mode. A mask
// editor file's alpha decides the latter (checkDrawnMask, async - the
// executed preview stands until the check lands): the core editor
// rewrites the combo to a fresh fully-opaque file on every save, erases
// included, so an erased mask is a stale copy of the image it was
// painted on and must not take the preview over from the executed output
// (a tab switch re-runs this, so it must not restore that copy either).
function restoreFilePreview(node) {
    if (getWidgetValue(node, "load_image") ||
        !isMaskEditorFile(getWidgetValue(node, "image"))) {
        setFilePreview(node);
        return;
    }
    if (maskEmptyOf(node) === true) {
        resetErasedMask(node);
        return;
    }
    setTimeout(async () => {
        const m = liveNode(node.id) || node;
        if (!m || m._removed) return;
        if (getWidgetValue(m, "crop_image") || !filePreviewActive(m)) return;
        await checkDrawnMask(m);
        if (maskEmptyOf(m) === true) {
            resetErasedMask(m);
        } else if (filePreviewActive(m)) {
            setFilePreview(m);
        }
    }, 0);
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


// Add and remove the load image and crop group rows (like the context
// loader's mode widgets): splice node.widgets and keep the widget store in
// sync so the Vue rows follow. Removed rows are not serialized into the
// prompt, so the backend falls back to their schema defaults. The image,
// crop image, crop preview and crop region rows never leave node.widgets
// and are hidden instead: the first two and the region are required
// serialized inputs (removing them drops their values from the prompt and
// the queue fails), and the preview is a DOM widget whose floating element
// must stay laid out. The crop group is independent of the load group.
function syncLoadGroup(node) {
    if (!node || node._removed || !node._gibbyLoadGroup) return;
    const g = node._gibbyLoadGroup;
    const load = !!getWidgetValue(node, "load_image");
    const crop = !!getWidgetValue(node, "crop_image");
    // A mask drawn on the preview rewrites the combo to the core editor's
    // clipspace file; with crop on the box is the mask, so go back to the
    // original file (the drawn mask is discarded).
    if (load && crop && isMaskEditorFile(getWidgetValue(node, "image")) && node._gibbyOriginalFile) {
        g.image.value = node._gibbyOriginalFile;
    }
    // The combo holds a file only while load is on (the backend loads it):
    // restore it when it was cleared while load was off, and clear it when
    // load goes off - a stale file would hijack the core mask editor, which
    // loads the combo file over the node's preview as its base image (a mask
    // editor file is the drawn mask and is kept either way).
    if (load) {
        const v = getWidgetValue(node, "image");
        const nv = v ? String(v).replace(/\s*\[\w+\]$/, "") : "";
        const opts = g.image.options?.values || [];
        const hasFile = nv && (isMaskEditorFile(v) || opts.includes(nv) || opts.includes(v));
        if (!hasFile) g.image.value = node._gibbyOriginalFile || opts[0] || "";
    } else {
        const v = getWidgetValue(node, "image");
        if (v && !isMaskEditorFile(v)) {
            trackOriginalFile(node, v);
            setWidgetValue(node, "image", "");
            clearNodePreview(node);
        }
    }
    const widgets = node.widgets || [];

    // The upload button is injected by the core; resolve it lazily in case
    // it was not there yet when the node was set up.
    if (!g.upload) {
        g.upload = findWidget(node, "upload");
        if (g.upload) g.addable.push(g.upload);
    }

    // The group rows in display order, after the load_image toggle. The
    // image/crop image/region rows are always present (see above) and hidden
    // with their toggle; the upload and AR toggle/preview rows are removed
    // with it.
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
    // and in the prompt (see above). The crop toggle stays visible: the
    // crop group is independent of the load one.
    if (g.image) setWidgetHidden(g.image, !load);
    if (g.cropPreview) setWidgetHidden(g.cropPreview, !crop);
    if (g.region) setWidgetHidden(g.region, !crop);
    // The preview toggle gates only the executed temp preview - the file
    // preview (load) and the crop preview (crop) are the image UI then.
    setWidgetHidden(findWidget(node, "enable_preview"), load || crop);

    // The executed temp preview (the node's own Preview Image) is the image
    // UI in the default mode - like the core Preview Image, a mask can be
    // drawn on it; the file preview (the loaded file with its drawn mask)
    // takes over in load mode, and the crop preview is the image UI in crop
    // mode (hideOutputImages hides the executed outputs there; in the
    // default mode the enable preview toggle off does, keeping the last
    // run's preview in the store for the downstream crop views). An erased
    // mask (a stale copy of the image it was painted on) leaves the
    // executed output as the preview - restoreFilePreview.
    node.hideOutputImages = crop || (!load && !getWidgetValue(node, "enable_preview"));
    if (!crop) {
        restoreFilePreview(node);
    } else {
        clearNodePreview(node);
        g.cropPreviewEl?._gibbyLoad?.();
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
        // options.serialize is what the V3 prompt builder reads (serialize
        // the legacy one).
        modeDom.serialize = false;
        modeDom.options = { ...(modeDom.options || {}), serialize: false };
        // Fixed row: with a maxHeight the layout pass never gives it a
        // share of the node's free height (the crop preview takes it all).
        const modeH = () => modeEl.style.display === "none" ? 0 : 26;
        modeDom.computeLayoutSize = () => ({ minHeight: modeH(), maxHeight: modeH(), minWidth: 1 });
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
        if (getWidgetValue(n, "crop_image")) {
            refitCropBox(n);
        }
    }, { on: "Selected", off: "Custom" });
    if (arToggleW) {
        arToggleW.serialize = false;
        arToggleW.options = { ...(arToggleW.options || {}), serialize: false };
        arToggleW.label = "Crop AR";
    }
    // The shared crop preview: the image the resize uses (cropPreviewUrl)
    // and the box locked to the node's crop_ar when "selected"
    // (lockedRatio). No control rows - the node keeps its own
    // width/height/x/y widgets.
    const cropEl = buildCropPreview(node, {
        getUrl: cropPreviewUrl,
        getRatio: lockedRatio,
        // Refit the box to the locked AR once the image is loaded (the live
        // node - a tab switch rebuilds the graph in place).
        onImageLoad: (n) => refitCropBox(liveNode(n.id) || n),
    });
    const cropDom = node.addDOMWidget("gibby_crop_preview", "GIBBY_CROP_PREVIEW", cropEl, {
        getValue: () => getWidgetValue(node, "crop_region"),
        setValue: (v) => setWidgetValue(node, "crop_region", v),
    });
    if (cropDom) {
        // Out of the serialized widget list: the value mirrors
        // crop_region. options.serialize is what the V3 prompt builder
        // reads (serialize the legacy one).
        cropDom.serialize = false;
        cropDom.options = { ...(cropDom.options || {}), serialize: false };
        // The preview stretches with the node (no maxHeight).
        cropDom.computeLayoutSize = () => ({ minHeight: CROP_PREVIEW_MIN_H, minWidth: 1 });
    }

    const enableW = findWidget(node, "enable_preview");
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

    // Re-resolve the crop preview's image (the nodeOutput store changed: an
    // upstream execution replaced the preview it shows).
    node._gibbyRefreshCropPreview = () => {
        if (getWidgetValue(node, "crop_image")) {
            node._gibbyLoadGroup.cropPreviewEl?._gibbyLoad?.();
        }
    };

    // File combo: a file change updates the store-driven preview while the
    // file combo is in use (load on, crop off, or a mask drawn on the
    // executed preview - the core editor rewrites the combo to its clipspace
    // file); with crop on it refreshes the crop preview instead, and with
    // both off a plain file clears it.
    if (imageW && !imageW._gibbyWrapped) {
        imageW._gibbyWrapped = true;
        imageW.callback = function (...args) {
            const n = liveNode(node.id);
            const grp = n && n._gibbyLoadGroup;
            if (!n || !grp) return undefined;
            trackOriginalFile(n, args[0]);
            const load = !!getWidgetValue(n, "load_image");
            const crop = !!getWidgetValue(n, "crop_image");
            if (crop) {
                clearNodePreview(n);
                grp.cropPreviewEl?._gibbyLoad?.();
            } else if (load || isMaskEditorFile(args[0])) {
                restoreFilePreview(n);
            } else {
                clearNodePreview(n);
            }
            return undefined;
        };
    }
    // Enable preview: re-apply the preview state (off hides the last run's
    // preview on the node - it is kept in the store, which downstream crop
    // views read the context's image from).
    if (enableW && !enableW._gibbyWrapped) {
        enableW._gibbyWrapped = true;
        const prev = enableW.callback;
        enableW.callback = function (...args) {
            if (prev) {
                try { prev.apply(this, args); } catch (e) { /* ignore */ }
            }
            const n = liveNode(node.id);
            if (n) syncLoadGroup(n);
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
                if (n && n._gibbyLoadGroup && getWidgetValue(n, "crop_image")) {
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
            if (n && n._gibbyLoadGroup && getWidgetValue(n, "crop_image")) {
                n._gibbyLoadGroup.cropPreviewEl?._gibbyPaint?.();
            }
        };
    }

    // Row order: the switcher on top, the size rows, then the load image
    // group at the bottom - the preview toggle above the load image toggle,
    // the crop rows under the crop toggle, the x/y/width/height row under
    // the crop preview.
    const loadGroup = [enableW, loadW, imageW, uploadW, cropImageW, arToggleW, cropDom, cropArW, regionW].filter(Boolean);
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

    trackOriginalFile(node);
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

        // The executed temp preview replaces the file preview in the store on
        // every run; restore the file preview while the combo holds one
        // (restoreFilePreview) - the loaded file in load mode, or a mask
        // drawn on the preview in default mode, so the mask stays on top of
        // the image. An erased mask leaves the executed output as the
        // preview instead, like a plain run.
        // Deferred to a macrotask: the store subscription is batched, so a
        // same-tick restore would hide the temp preview from it and the
        // downstream crop preview would never see the run's output.
        const origOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (output) {
            if (origOnExecuted) {
                try { origOnExecuted.apply(this, arguments); } catch (e) { /* ignore */ }
            }
            const n = liveNode(this.id) || this;
            if (!n || n._removed) return;
            if (getWidgetValue(n, "crop_image") || !filePreviewActive(n)) return;
            setTimeout(() => restoreFilePreview(liveNode(n.id) || n), 0);
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
                // enable_preview postdates the positional order (see
                // CANONICAL_WIDGETS); restore it by name here.
                for (const name of [...CANONICAL_WIDGETS, "enable_preview"]) {
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

            trackOriginalFile(this);
            refreshModeVisibility(this);
            // The saved toggles may differ from the schema defaults the group
            // started with; re-apply the rows.
            syncLoadGroup(this);
        };
    },
    // The crop preview's image lives in the nodeOutput store (the loaded
    // file, an upstream node's last-run image, or a context node's executed
    // image) - refresh every resize node's crop preview whenever the store
    // changes.
    setup() {
        const refreshAll = () => {
            const graph =
                (app.canvas && app.canvas.getCurrentGraph && app.canvas.getCurrentGraph()) ||
                app.graph;
            if (!graph) return;
            for (const n of graph.nodes || []) {
                if (n?.type === NODE_TYPE) n._gibbyRefreshCropPreview?.();
            }
        };
        const subscribe = () => {
            const store = frontendStore("nodeOutput");
            if (!store?.$subscribe) {
                // The Vue app is not mounted yet - retry shortly.
                setTimeout(subscribe, 100);
                return;
            }
            store.$subscribe(refreshAll);
        };
        subscribe();
    },
});

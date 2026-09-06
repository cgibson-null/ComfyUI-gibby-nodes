// ---------------------------------------------------------------------------
// Gibby Nodes - Resize Image / Empty Latent (Context) frontend
//
// The node's standard widgets are declared in the backend schema so their
// values serialize into prompts normally. This file upgrades one of them to
// a custom DOM UI using ComfyUI's own node.addDOMWidget() (NOT hand-drawn
// canvas widgets):
//   - "mode" combo          -> horizontal toggle switch (custom / aspect ratio / custom AR, plus keep AR when media is linked)
// Rows are shown/hidden per mode via widget.options.hidden (the same flag the
// new frontend's advanced-widget toggle uses).
// ---------------------------------------------------------------------------

import { app } from "../../../scripts/app.js";

const NODE_TYPE = "Gibby_EmptyLatent_Resolution";

// Global registry to track all instances of this node
const gibbyResolutionNodes = new Set();

// Register a node instance for tracking
function trackNode(node) {
    if (node && !node._removed) {
        gibbyResolutionNodes.add(node);
    }
}

// Unregister a node instance
function untrackNode(node) {
    if (node) {
        gibbyResolutionNodes.delete(node);
    }
}


const MODES = [
    ["keep_ar", "Keep AR"],
    ["custom", "Custom"],
    ["aspect_ratio", "Aspect Ratio"],
    ["custom_aspect_ratio", "Custom AR"],
];

// Canonical order of standard widgets (schema order, excluding our DOM rows).
// Used by serialize/configure overrides to ensure stable value mapping during
// duplicate/undo regardless of which widgets are visually hidden via options.hidden.
const CANONICAL_WIDGETS = [
    "mode", "width", "height", "aspect_ratio", "x", "y", "megapixels",
    "scale_factor", "swap_dimensions",
    "upscale_method", "keep_proportion", "pad_color", "crop_position",
    "multiple", "batch_size", "flux2_latent"
];

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

// Build a horizontal button switcher (low_vram style). Returns the element.
function buildModeSwitch(node) {
    const el = document.createElement("div");
    el.className = "gibby-mode-switch";
    el.style.cssText = "display:flex; gap:4px; padding:0;";

    // Use a getter to always reference the current node (survives undo/redo)
    const getNode = () => node._removed ? null : node;

    for (const [value, label] of MODES) {
        const b = document.createElement("button");
        b.textContent = label;
        b._gibbyValue = value;
        b.style.cssText =
            "flex:1; padding:1px 4px; font-size:11px; cursor:pointer; " +
            "background:#2a2a2a; color:#ccc; border:1px solid #444; " +
            "border-radius:3px; font-family:inherit; line-height:1.1;";
        const isSel = () => {
            const n = [...gibbyResolutionNodes].find(n => !n._removed && n.id === node.id);
            return n ? getWidgetValue(n, "mode") === value : false;
        };
        b.addEventListener("mouseenter", () => { if (!isSel() && !b.classList.contains("gibby-mode-active")) b.style.background = "#3a3a3a"; });
        b.addEventListener("mouseleave", () => { if (!isSel() && !b.classList.contains("gibby-mode-active")) b.style.background = "#2a2a2a"; });
        b.addEventListener("mousedown", (e) => e.stopPropagation());
        b.addEventListener("click", (e) => {
            e.stopPropagation();
            // Find the current node from the registry (survives undo/redo)
            const n = [...gibbyResolutionNodes].find(n => !n._removed && n.id === node.id);
            if (!n) return;
            setWidgetValue(n, "mode", value);
            refreshModeVisibility(n);
        });
        el.appendChild(b);
    }

    // Keep button highlight in sync with the widget value.
    const labelOf = (v) => MODES.find(([x]) => x === v)?.[1] || "";
    el._gibbyPaint = () => {
        // Find current node from registry (survives undo/redo)
        const n = [...gibbyResolutionNodes].find(n => !n._removed && n.id === node.id);
        if (!n) return;
        const modeVal = getWidgetValue(n, "mode");
        const curLabel = labelOf(modeVal);
        
        // Find the mode switch element in the DOM
        let currentEl = null;
        const nodeEl = document.querySelector(`[data-id="${n.id}"]`);
        if (nodeEl) {
            currentEl = nodeEl.querySelector(".gibby-mode-switch");
        } else {
            // Try finding by node-type attribute instead
            const nodeTypeEl = document.querySelector(`[node-type="Gibby_EmptyLatent_Resolution"]`);
            if (nodeTypeEl) {
                currentEl = nodeTypeEl.querySelector(".gibby-mode-switch");
            }
        }
        
        if (!currentEl) return;
        
        for (const b of currentEl.children) {
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

// Re-sync DOM inputs from widget values.
function syncDomInputs(node) {
    // No-op: width/height and x/y are now standard widgets.
}

// Show/hide rows per mode: keep AR hides all size widgets but megapixels;
// custom shows the width x height fields (also when media is linked - they
// become the resize box); the AR modes show their ratio + megapixels.
function refreshModeVisibility(node) {
    const hasMedia = !!node.inputs?.find((i) => (i.name === "image" || i.name === "mask") && i.link);
    const mode = getWidgetValue(node, "mode") || "custom";

    // Standard Vue-rendered rows (hidden via options.hidden).
    setWidgetHidden(findWidget(node, "width"), mode !== "custom");
    setWidgetHidden(findWidget(node, "height"), mode !== "custom");
    setWidgetHidden(findWidget(node, "aspect_ratio"), mode !== "aspect_ratio");
    setWidgetHidden(findWidget(node, "x"), mode !== "custom_aspect_ratio");
    setWidgetHidden(findWidget(node, "y"), mode !== "custom_aspect_ratio");
    setWidgetHidden(findWidget(node, "megapixels"), !["keep_ar", "aspect_ratio", "custom_aspect_ratio"].includes(mode));
    for (const name of ["upscale_method", "keep_proportion", "pad_color", "crop_position"]) {
        setWidgetHidden(findWidget(node, name), !hasMedia);
    }
    for (const name of ["batch_size", "flux2_latent"]) {
        setWidgetHidden(findWidget(node, name), hasMedia);
    }

    const els = node._gibbyResElements;
    if (!els) return;

    // Repaint the switcher highlight.
    if (els.modeSwitch._gibbyPaint) els.modeSwitch._gibbyPaint();

    // Force a redraw so the layout system re-measures after rows show/hide.
    try { node.graph?.setDirtyCanvas(true, true); } catch (e) { /* ignore */ }

    // The Vue widget list only reads options.hidden when it re-renders; a
    // same-tick showAdvanced toggle-and-restore forces that pass without
    // changing the node's actual advanced state.
    const prev = node.showAdvanced;
    node.showAdvanced = !prev;
    node.showAdvanced = prev;
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
    trackNode(node);

    // 1. Mode switcher: hide the combo row, add a DOM toggle at the top.
    setWidgetHidden(modeW, true);
    const modeEl = buildModeSwitch(node);
    const modeDom = node.addDOMWidget("gibby_mode", "GIBBY_MODE_SWITCH", modeEl, {
        getValue: () => getWidgetValue(node, "mode"),
        setValue: (v) => setWidgetValue(node, "mode", v),
    });
    if (modeDom) modeDom.computeLayoutSize = () => ({ minHeight: modeEl.style.display === "none" ? 0 : 26, minWidth: 1 });

    // Keep width/height and x/y as standard ComfyUI inputs
    // (shown/hidden per mode via options.hidden).

    node._gibbyResElements.modeSwitch = modeEl;

    // Row order: switcher on top, then standard rows in schema order.
    const others = node.widgets.filter((w) => w !== modeDom);
    node.widgets.splice(0, node.widgets.length, modeDom, ...others);

    refreshModeVisibility(node);
    // Delay paint to ensure DOM is ready
    setTimeout(() => refreshModeVisibility(node), 100);
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

        const origOnRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            untrackNode(this);
            if (origOnRemoved) {
                try { origOnRemoved.apply(this, arguments); } catch (e) { /* ignore */ }
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

        // Workflow load snaps every node up to max(saved size, computeSize()).
        // Our DOM rows make that larger than the user's chosen height, so it
        // grows back on every tab switch. Never demand more space than we have.
        const origComputeSize = nodeType.prototype.computeSize;
        nodeType.prototype.computeSize = function (currentSize) {
            if (!this.size || !this.size[0]) return origComputeSize.call(this, currentSize);
            const natural = origComputeSize.call(this, currentSize);
            return [Math.min(natural[0], this.size[0]), Math.min(natural[1], this.size[1])];
        };

        // ComfyUI's clone/undo serialize widget values by position and skip
        // options.hidden widgets, so duplicating a node in aspect_ratio mode
        // shifts width=512 into megapixels etc. Force a canonical value list
        // (all standard widgets, schema order) on both ends instead.
        const origSerialize = nodeType.prototype.serialize;
        nodeType.prototype.serialize = function () {
            const data = origSerialize.apply(this);
            if (!data || !Array.isArray(data.widgets_values)) return data;

            const values = [];
            for (const name of CANONICAL_WIDGETS) {
                const w = findWidget(this, name);
                values.push(w ? w.value : null);
            }
            data.widgets_values = values;
            return data;
        };

        const origConfigure = nodeType.prototype.configure;
        nodeType.prototype.configure = function (data) {
            origConfigure.apply(this, arguments);

            if (data && Array.isArray(data.widgets_values)) {
                const vals = data.widgets_values;
                for (let i = 0; i < Math.min(vals.length, CANONICAL_WIDGETS.length); i++) {
                    const w = findWidget(this, CANONICAL_WIDGETS[i]);
                    if (w && vals[i] !== null && vals[i] !== undefined) {
                        w.value = vals[i];
                    }
                }
            }

            refreshModeVisibility(this);
        };
    },
});

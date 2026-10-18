// ---------------------------------------------------------------------------
// Gibby Nodes - Mask/Segment (Context) frontend
//
// The sam3 mode's tracker rows are the Run SAM3 Video Track widgets; they are
// part of the sam3 DynamicCombo, so the core adds them whenever sam3 is
// selected. This hides them again while the tracking toggle is off - the same
// flag the core's advanced-widget toggle uses (widget.options.hidden) - so the
// node exposes a plain toggle instead of a second mode selector.
// A mode change rebuilds the combo's rows, so the flag is re-applied after the
// mode widget's own handler; the values stay serialized either way, and the
// backend only reads them when tracking is on.
// ---------------------------------------------------------------------------

import { app } from "../../../scripts/app.js";
import { liveNode, findWidget, setWidgetHidden, refitNode, rerenderWidgetList, markGraphDirty } from "../gibby_widgets.js";

const NODE_TYPE = "Gibby_Detection";
const TOGGLE = "tracking";
const TRACK_ROWS = ["detection_threshold", "max_objects", "detect_interval", "object_indices"];

// The tracker rows are DynamicCombo rows: the core names them "<combo>.<key>".
const comboRow = (node, name) => findWidget(node, name, { dotted: true });

function syncTrackingRows(node) {
    if (!node || node._removed) return;
    // The toggle row itself only exists with sam3 selected; hook it as soon as
    // the combo has created it (a mode change recreates the row).
    hookWidget(node, TOGGLE, syncTrackingRows);
    const on = !!comboRow(node, TOGGLE)?.value;
    let changed = false;
    for (const name of TRACK_ROWS) {
        const w = comboRow(node, name);
        if (!w) continue;
        changed = changed || !!w.options?.hidden === on;
        setWidgetHidden(w, !on);
    }
    if (!changed) return;
    // The layout pass re-measures the node with the rows shown/hidden; the Vue
    // widget list only re-reads options.hidden when it re-renders.
    refitNode(node);
    rerenderWidgetList(node);
    markGraphDirty(node);
}

// Run fn after the widget's own handler, one tick late so the rows the mode
// combo adds or removes are already in place.
function hookWidget(node, name, fn) {
    const w = comboRow(node, name);
    if (!w || w._gibbyTrackingHooked) return;
    w._gibbyTrackingHooked = true;
    const prev = w.callback;
    w.callback = function (...args) {
        const r = prev?.apply(this, args);
        setTimeout(() => fn(liveNode(node.id) || node), 0);
        return r;
    };
}

function setup(node) {
    if (!node || node._removed || !findWidget(node, "mode")) return false;
    hookWidget(node, "mode", syncTrackingRows);
    syncTrackingRows(node);
    return true;
}

app.registerExtension({
    name: "GibbyNodes.detectionTracking",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (origOnNodeCreated) {
                try { origOnNodeCreated.apply(this, arguments); } catch (e) { /* ignore */ }
            }
            // The combo's rows can still be missing here; retry shortly.
            if (!setup(this)) setTimeout(() => setup(liveNode(this.id) || this), 50);
        };

        // Saved values (and the saved mode) land in configure, which recreates
        // the rows, so the visibility has to be re-applied afterwards.
        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            if (origOnConfigure) {
                try { origOnConfigure.apply(this, arguments); } catch (e) { /* ignore */ }
            }
            if (!setup(this)) setTimeout(() => setup(liveNode(this.id) || this), 50);
        };
    },
});

// ---------------------------------------------------------------------------
// Gibby Nodes - Mask/Segment (Context) frontend
//
// The sam3 mode's tracker rows are the Run SAM3 Video Track widgets; they are
// part of the sam3 DynamicCombo, so the core adds them whenever sam3 is
// selected. This hides them again while the tracking toggle is off, so the node
// exposes a plain toggle instead of a second mode selector. The values stay
// serialized either way, and the backend only reads them when tracking is on.
// ---------------------------------------------------------------------------

import { app } from "../../../scripts/app.js";
import { findWidget, applyRowVisibility, rowVisibility } from "../gibby_widgets.js";

const NODE_TYPE = "Gibby_Detection";
const TOGGLE = "tracking";
const TRACK_ROWS = ["detection_threshold", "max_objects", "detect_interval", "object_indices"];

function syncTrackingRows(node, hook) {
    // The toggle row itself only exists with sam3 selected; hook it as soon as
    // the combo has created it (a mode change recreates the row). The combo
    // names its rows "<combo>.<key>", hence the dotted lookups.
    hook(node, TOGGLE);
    const on = !!findWidget(node, TOGGLE, { dotted: true })?.value;
    applyRowVisibility(node, Object.fromEntries(TRACK_ROWS.map((name) => [name, on])));
}

app.registerExtension({
    name: "GibbyNodes.detectionTracking",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;
        rowVisibility(nodeType, { widgets: ["mode"], sync: syncTrackingRows });
    },
});

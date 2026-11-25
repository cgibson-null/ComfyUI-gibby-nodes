// ---------------------------------------------------------------------------
// Gibby Nodes - H3 Timeline (Context) frontend
//
// Each mode only reads some of the frame widgets: insert keeps the whole clip,
// so it has no overlap rows at all, and only a loop has an arrival side. The
// rows are hidden per mode; the values stay serialized, so switching mode back
// restores them, and the backend only reads the rows the mode owns.
// ---------------------------------------------------------------------------

import { app } from "../../../scripts/app.js";
import { findWidget, applyRowVisibility, rowVisibility } from "../gibby_widgets.js";

const NODE_TYPE = "Gibby_H3_Timeline";
// widget -> the modes that read it
const ROW_MODES = {
    overlap_frames: ["extend", "prepend", "loop"],
    loop_end_frames: ["loop"],
    insert_at_frame: ["insert"],
};

function syncModeRows(node) {
    const mode = findWidget(node, "mode")?.value;
    const wanted = {};
    for (const [name, modes] of Object.entries(ROW_MODES)) wanted[name] = modes.includes(mode);
    applyRowVisibility(node, wanted);
}

app.registerExtension({
    name: "GibbyNodes.h3TimelineModes",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;
        rowVisibility(nodeType, { widgets: ["mode"], sync: syncModeRows });
    },
});

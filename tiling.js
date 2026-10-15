// ---------------------------------------------------------------------------
// Gibby Nodes - Split Tiles (Context) frontend
//
// The core takes a float widget's decimal precision - and the quantum it snaps
// every change to - from the input's step, so the 0.01 overlap row keeps two
// digits and rounds whatever is typed to them. The tile grid wants the exact
// fraction (1/3, 5/16), so the row keeps its 0.01 drag step but takes ten
// digits: options.precision is the digit count, options.round the snap
// quantum (1e-10 = no snapping, the min/max clamp stays).
// ---------------------------------------------------------------------------

import { app } from "../../../scripts/app.js";

const NODE_TYPE = "Gibby_Split_Tiles";
const ROW = "overlap";
const DIGITS = 10;

function setup(node) {
    if (!node || node._removed) return;
    const w = node.widgets?.find((w) => w.name === ROW);
    // The row can still be missing while the node is being built; retry once
    // the core has finished adding the schema's widgets.
    if (!w) {
        setTimeout(() => setup(node), 50);
        return;
    }
    w.options.precision = DIGITS;
    w.options.round = 10 ** -DIGITS;
}

app.registerExtension({
    name: "GibbyNodes.tilingFineOverlap",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (origOnNodeCreated) {
                try { origOnNodeCreated.apply(this, arguments); } catch (e) { /* ignore */ }
            }
            setup(this);
        };
    },
});

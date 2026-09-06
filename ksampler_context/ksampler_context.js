// ---------------------------------------------------------------------------
// Gibby Nodes - KSampler (Context) frontend
//
// Handles step field rounding: when abs(value) > 1, round to whole numbers.
// ---------------------------------------------------------------------------

import { app } from "../../../scripts/app.js";

const NODE_TYPE = "Gibby_KSampler_Context";

function roundStepValue(node, widgetName) {
    const widget = node.widgets?.find((w) => w.name === widgetName);
    if (!widget) return;
    
    const val = widget.value;
    if (Math.abs(val) > 1) {
        // Round to whole number
        widget.value = Math.round(val);
    }
}

app.registerExtension({
    name: "GibbyNodes.KSamplerContext",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;

        const origOnWidgetChanged = nodeType.prototype.onWidgetChanged;
        nodeType.prototype.onWidgetChanged = function (param) {
            // Call original first
            if (origOnWidgetChanged) {
                try { origOnWidgetChanged.apply(this, arguments); } catch (e) { /* ignore */ }
            }

            // Round start_step and end_step if abs > 1
            if (param === "start_step") {
                roundStepValue(this, "start_step");
            } else if (param === "end_step") {
                roundStepValue(this, "end_step");
            }
        };

        // Also handle onConfigure to ensure saved values are rounded
        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            if (origOnConfigure) {
                try { origOnConfigure.apply(this, arguments); } catch (e) { /* ignore */ }
            }
            roundStepValue(this, "start_step");
            roundStepValue(this, "end_step");
        };
    },
});

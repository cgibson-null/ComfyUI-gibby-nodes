// ---------------------------------------------------------------------------
// Gibby Nodes - Show Any frontend
//
// The backend sends the shown values in the executed output's ui.text, one entry
// per row. The core builds its text rows only for PreviewAny, so the node builds
// its own: one read-only multiline row per value, the way EasyUse's Show Any
// displays a list as separate elements. The rows are serialized, and the backend
// also writes them into the workflow embedded in the output, so what was shown
// comes back with the workflow - saved or extracted from an image - without a
// re-run.
// ---------------------------------------------------------------------------

import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";
import { markGraphDirty } from "../gibby_widgets.js";

const NODE_TYPE = "Gibby_ShowAny";
const ROW = "preview_text";

function populate(node, texts) {
    for (const w of [...(node.widgets ?? [])]) {
        if (w.name.startsWith(ROW)) node.removeWidget(w);
    }
    for (const [i, text] of (texts ?? []).entries()) {
        const { widget } = ComfyWidgets["STRING"](node, `${ROW}_${i}`, ["STRING", { multiline: true }], app);
        // widget.serialize keeps the row in the saved workflow, options.serialize keeps
        // a display-only row out of the prompt the node is queued with
        widget.options.serialize = false;
        const el = widget.element || widget.inputEl;
        if (el) {
            el.readOnly = true;
            el.style.opacity = "0.6";
        }
        widget.value = `${text ?? ""}`;
    }
    // Grow only to make room for rows the node is too short for. Refitting down
    // through the core's computeSize would snap the node back to the minimum of
    // its rows on every execution and throw away a size set by hand.
    if (node.size) {
        try {
            const fit = node.computeSize([...node.size])[1];
            if (fit > node.size[1]) node.size = [node.size[0], fit];
        } catch (e) { /* ignore */ }
    }
    markGraphDirty(node);
}

const asTexts = (value) => (Array.isArray(value) ? value.filter((t) => t != null) : value == null ? [] : [value]);

app.registerExtension({
    name: "GibbyNodes.showAnyPreview",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;

        // The rows are built here rather than by the core's restore pass, which
        // only fills widgets that already exist.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            onConfigure?.apply(this, arguments);
            populate(this, asTexts(arguments[0]?.widgets_values ?? this.widgets_values));
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            populate(this, asTexts(message?.text));
        };
    },
});

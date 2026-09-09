// ---------------------------------------------------------------------------
// Gibby Nodes - KSampler (Context) frontend
//
// Handles step field rounding: when abs(value) > 1, round to whole numbers.
//
// Right-click Extensions:
// - "Add second pass upscale": 2x second pass to the right of the node -
//   Resize Image / Empty Latent (Context) (Keep AR, 0 megapixels, scale factor 2)
//   -> KSampler (Context) (denoise 0.6) -> Compare Images (original vs second pass).
// - "Add iterative upscale": Iterative Upscale options -> KSampler (Context),
//   context straight from the original node, Compare Images at the end.
// - "Add iterative step": KSampler (Context) fed by the original node's options
//   output (carries next_step), Compare Images at the end.
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

function setWidget(node, name, value) {
    const w = node.widgets?.find((w) => w.name === name);
    if (w) w.value = value;
}

// Create a row of nodes to the right of the source node and lay them out.
function createChain(node, types) {
    const gap = 60;
    const x = node.pos[0] + node.size[0] + gap;
    const y = node.pos[1];

    let created = [];
    if (typeof node.graph.createNode === "function") {
        // Old litegraph frontend: graph.createNode() adds the node itself.
        for (const type of types) created.push(node.graph.createNode(type));
    } else {
        // New Vue frontend: create the nodes through the canvas' paste path.
        created = app.canvas._deserializeItems({
            nodes: types.map((type, i) => ({id: i + 1, type, pos: [x, y]})),
        }, {position: [x, y]})?.created ?? [];
    }

    // Lay the chain out in a row to the right of the source node.
    let cx = x;
    for (const n of created) {
        if (!n) continue;
        n.setPos(cx, y);
        cx += n.size[0] + gap;
    }

    // Resolve by type so the result doesn't depend on the creation order.
    const byType = {};
    for (const n of created) if (n) byType[n.type] = n;
    return types.map(t => byType[t]);
}

function addSecondPassUpscale(node) {
    const [resize, sampler, compare] = createChain(node, ["Gibby_EmptyLatent_Resolution", NODE_TYPE, "ImageCompare"]);
    if (!resize || !sampler || !compare) return;

    // Keep AR at 0 megapixels rescales the image to its own size x scale factor 2.
    setWidget(resize, "mode", "keep_ar");
    setWidget(resize, "megapixels", 0);
    setWidget(resize, "scale_factor", 2);
    // Second pass refines the upscaled image at 0.6 denoise.
    setWidget(sampler, "denoise", 0.6);

    node.connect(0, resize, 0);  // context -> context
    resize.connect(0, sampler, 0);  // context -> context
    node.connect(2, compare, 0);  // original image -> image_a
    sampler.connect(2, compare, 1);  // second pass image -> image_b
}

function addIterativeUpscale(node) {
    const [options, sampler, compare] = createChain(node, ["GibbyIterativeUpscaleOptions", NODE_TYPE, "ImageCompare"]);
    if (!options || !sampler || !compare) return;

    node.connect(0, sampler, 0);  // context -> context
    options.connect(0, sampler, 7);  // options -> options
    node.connect(2, compare, 0);  // original image -> image_a
    sampler.connect(2, compare, 1);  // upscaled image -> image_b
}

function addIterativeStep(node) {
    const [sampler, compare] = createChain(node, [NODE_TYPE, "ImageCompare"]);
    if (!sampler || !compare) return;

    node.connect(0, sampler, 0);  // context -> context
    node.connect(6, sampler, 7);  // options (carries next_step) -> options
    node.connect(2, compare, 0);  // original image -> image_a
    sampler.connect(2, compare, 1);  // refined image -> image_b
}

app.registerExtension({
    name: "GibbyNodes.KSamplerContext",
    // Right-click menu items (new frontend's replacement for addContextItem):
    // build second-pass / iterative upscale chains.
    getNodeMenuItems(node) {
        if (!node || node.type !== NODE_TYPE) return [];
        return [
            {
                content: "Add second pass upscale",
                callback: () => addSecondPassUpscale(node),
            },
            {
                content: "Add iterative upscale",
                callback: () => addIterativeUpscale(node),
            },
            {
                content: "Add iterative step",
                callback: () => addIterativeStep(node),
            },
        ];
    },
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

        // Also handle onConfigure to ensure saved values are rounded.
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

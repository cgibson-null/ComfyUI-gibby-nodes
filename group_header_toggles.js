import { app } from "../../../scripts/app.js";

// Gibby Nodes - Group Header Fast Toggles
// --------------------------------------
// Three small buttons in the top-right corner of each group header, ported
// from rgthree-comfy's "Show fast toggles in Group Headers" feature:
//   Queue  -> queue only this group's output nodes (native partial execution)
//   Bypass -> flip all contained nodes between ALWAYS and BYPASS as a whole
//   Mute   -> flip all contained nodes between ALWAYS and muted as a whole
// Configured under GibbyNodes > Group Header Toggles in the settings dialog.

const SETTING_PREFIX = "GibbyNodes.GroupHeaderToggles";

// Node execution modes (LiteGraph's enum values).
const MODE_ALWAYS = 0;   // always runs
const MODE_MUTE = 2;     // muted - not run at all
const MODE_BYPASS = 4;   // inputs pass through as outputs

// Button geometry - values from rgthree-comfy.
const BTN_SIZE = 20;
const BTN_MARGIN = [6, 6];   // [x, y] margin around the buttons
const BTN_SPACING = 8;       // draw spacing only (the hit-test uses its own)
const BTN_GRID = BTN_SIZE / 8;

let installed = false;    // guard against installing the patches twice
let lastMouse = null;     // latest pointer event on the canvas, for hover display

// ---------------------------------------------------------------------------
// Settings - ComfyUI native settings (GibbyNodes > Group Header Toggles)
// ---------------------------------------------------------------------------

// [button name, setting suffix] in visual order left-to-right.
const TOGGLES = [
    ["queue", "ToggleQueue"],
    ["bypass", "ToggleBypass"],
    ["mute", "ToggleMute"],
];

function getToggles() {
    const settings = app.ui.settings;
    if (settings.getSettingValue(SETTING_PREFIX + ".ShowMode") === "never") return [];
    // rgthree draws index 0 at the right edge, so reverse.
    return TOGGLES.filter(([, suffix]) => settings.getSettingValue(SETTING_PREFIX + "." + suffix))
        .map(([name]) => name)
        .reverse();
}

function showAlways() {
    return app.ui.settings.getSettingValue(SETTING_PREFIX + ".ShowMode") === "always";
}

// ---------------------------------------------------------------------------
// Group / node helpers - ported from rgthree-comfy's utils.js.
// ---------------------------------------------------------------------------

function getGroupNodes(group) {
    return Array.from(group._children).filter((c) => c instanceof window.LGraphNode);
}

function getOutputNodes(nodes) {
    return nodes.filter((n) => n.mode != MODE_MUTE && n.constructor.nodeData?.output_node);
}

// Depth-first over a node collection, descending into subgraph container nodes.
// Their inner nodes live in their own LGraph and are not on the canvas graph.
function reduceNodesDepthFirst(nodes, fn) {
    const stack = [...nodes];
    while (stack.length > 0) {
        const n = stack.pop();
        fn(n);
        if (n.isSubgraphNode() && n.subgraph?.nodes) {
            for (let i = n.subgraph.nodes.length - 1; i >= 0; i--) stack.push(n.subgraph.nodes[i]);
        }
    }
}

function changeModeOfNodes(nodes, mode) {
    reduceNodesDepthFirst(nodes, (n) => { n.mode = mode; });
}

// ---------------------------------------------------------------------------
// Actions - ported from rgthree-comfy's feature file.
// ---------------------------------------------------------------------------

function showWarning(message) {
    // Minimal stand-in for rgthree.showMessage: a floating toast that auto-dismisses.
    const el = document.createElement("div");
    Object.assign(el.style, {
        position: "fixed", bottom: "2rem", left: "50%", transform: "translateX(-50%)",
        background: "rgba(17, 24, 39, 0.9)", color: "#fbbf24", padding: "8px 16px",
        borderRadius: "6px", fontSize: "13px", zIndex: "9999", pointerEvents: "none",
    });
    el.textContent = message;
    document.body.appendChild(el);
    setTimeout(() => el.remove(), 4000);
}

function clickedOnToggleButton(e, group) {
    const toggles = getToggles();
    const pos = group.pos;   // note: rgthree reads the public getter here (not _pos)
    const size = group.size;
    for (let i = 0; i < toggles.length; i++) {
        if (window.LiteGraph.isInsideRectangle(
            e.canvasX, e.canvasY,
            pos[0] + size[0] - (BTN_SIZE + BTN_MARGIN[0]) * (i + 1),   // hit rect: margin-based spacing
            pos[1] + BTN_MARGIN[1],
            BTN_SIZE, BTN_SIZE,
        )) return toggles[i];
    }
    return null;
}

function onToggleAction(toggleName, group) {
    const action = toggleName.toUpperCase();
    if (action === "QUEUE") {
        // Native partial execution: queue these outputs plus their upstream deps only.
        const outputNodes = getOutputNodes(getGroupNodes(group));
        if (!outputNodes.length) {
            showWarning("No output nodes for group!");
            return;
        }
        app.queuePrompt(0, 1, outputNodes.map((n) => String(n.id)));
    } else if (action === "MUTE" || action === "BYPASS") {
        const toggleMode = action === "MUTE" ? MODE_MUTE : MODE_BYPASS;
        group.recomputeInsideNodes();   // refresh membership before evaluating state
        const nodes = getGroupNodes(group);
        const hasAnyActive = nodes.some((n) => n.mode === MODE_ALWAYS);
        const isAllMuted = !hasAnyActive && nodes.every((n) => n.mode === MODE_MUTE);
        const isAllBypassed = !hasAnyActive && !isAllMuted && nodes.every((n) => n.mode === MODE_BYPASS);
        let newMode;
        if (toggleMode === MODE_MUTE) {
            // all muted -> wake everything up; otherwise mute it.
            newMode = isAllMuted ? MODE_ALWAYS : MODE_MUTE;
        } else {
            // all bypassed -> restore; otherwise bypass.
            newMode = isAllBypassed ? MODE_ALWAYS : MODE_BYPASS;
        }
        changeModeOfNodes(nodes, newMode);
        group.graph.setDirtyCanvas(true, true);   // redraw the button icon states now
    }
}

function handleToggleClick(canvas, e) {
    if (e.button !== 0 || !canvas.graph) return;
    canvas.adjustMouseEvent(e);
    const group = canvas.graph.getGroupOnPos(e.canvasX, e.canvasY);
    if (!group) return;
    const toggle = clickedOnToggleButton(e, group);
    if (!toggle) return;
    onToggleAction(toggle, group);
    // Same as rgthree-comfy: swallow the group selection/drag this click started.
    canvas.selected_group = null;
    canvas.dragging_canvas = false;
}

// ---------------------------------------------------------------------------
// Button drawing - ported from rgthree-comfy's feature file.
// ---------------------------------------------------------------------------

function eyeFrame(midX, midY, yFlip = 1) {
    return `
      M ${midX - BTN_SIZE / 2} ${midY}
      c ${BTN_GRID * 1.5} ${yFlip * BTN_GRID * 2.5}, ${BTN_GRID * (8 - 1.5)} ${yFlip * BTN_GRID * 2.5}, ${BTN_GRID * 8} 0
  `;
}

function eyeLashes(midX, midY, yFlip = 1) {
    return `
    M ${midX - BTN_GRID * 3.46} ${midY + yFlip * BTN_GRID * 0.9} l -1.15  ${1.25 * yFlip}
    M ${midX - BTN_GRID * 2.38} ${midY + yFlip * BTN_GRID * 1.6} l -0.90  ${1.5 * yFlip}
    M ${midX - BTN_GRID * 1.15} ${midY + yFlip * BTN_GRID * 1.95} l -0.50  ${1.75 * yFlip}
    M ${midX + BTN_GRID * 0.0} ${midY + yFlip * BTN_GRID * 2.0} l  0.00  ${2.0 * yFlip}
    M ${midX + BTN_GRID * 1.15} ${midY + yFlip * BTN_GRID * 1.95} l  0.50  ${1.75 * yFlip}
    M ${midX + BTN_GRID * 2.38} ${midY + yFlip * BTN_GRID * 1.6} l  0.90  ${1.5 * yFlip}
    M ${midX + BTN_GRID * 3.46} ${midY + yFlip * BTN_GRID * 0.9} l  1.15  ${1.25 * yFlip}
`;
}

function circlePath(cx, cy, radius) {
    return `
      M ${cx} ${cy}
      m ${radius}, 0
      a ${radius},${radius} 0 1, 1 -${radius * 2},0
      a ${radius},${radius} 0 1, 1  ${radius * 2},0
  `;
}

function drawToggles(canvas, ctx) {
    const toggles = getToggles();
    if (toggles.length === 0 || !canvas.graph) return;   // off or no graph yet
    const graph = canvas.graph;
    let groups;
    if (showAlways()) {
        groups = graph._groups || [];
    } else {
        // "on hover": only the group under the last mouse position.
        if (!lastMouse) return;
        const hoverGroup = graph.getGroupOnPos(lastMouse.canvasX, lastMouse.canvasY);
        groups = hoverGroup ? [hoverGroup] : [];
    }
    if (!groups.length) return;

    ctx.save();
    for (const group of groups) {
        // Whole-group state drives the mute/bypass icon styles.
        const nodes = getGroupNodes(group);
        let anyActive = false;
        let allMuted = !!nodes.length;
        let allBypassed = allMuted;
        for (const node of nodes) {
            anyActive = anyActive || node.mode === MODE_ALWAYS;
            allMuted = allMuted && node.mode === MODE_MUTE;
            allBypassed = allBypassed && node.mode === MODE_BYPASS;
            if (anyActive || (!allMuted && !allBypassed)) break;
        }

        for (let i = 0; i < toggles.length; i++) {
            const toggle = toggles[i];
            const pos = group._pos;   // note: rgthree reads the internal _pos/_size here
            const size = group._size;
            ctx.fillStyle = ctx.strokeStyle = group.color || "#335";
            const x = pos[0] + size[0] - BTN_MARGIN[0] - BTN_SIZE - (BTN_SPACING + BTN_SIZE) * i;
            const y = pos[1] + BTN_MARGIN[1];
            const midX = x + BTN_SIZE / 2;
            const midY = y + BTN_SIZE / 2;

            if (toggle === "queue") {
                // Filled when the group has queueable output nodes, outline only otherwise.
                const outputNodes = getOutputNodes(nodes);
                const oldGlobalAlpha = ctx.globalAlpha;
                if (!outputNodes.length) ctx.globalAlpha = 0.5;
                ctx.lineJoin = "round";
                ctx.lineCap = "round";
                const arrowSizeX = BTN_SIZE * 0.6;
                const arrowSizeY = BTN_SIZE * 0.7;
                const arrow = new Path2D(`M ${x + arrowSizeX / 2} ${midY} l 0 -${arrowSizeY / 2} l ${arrowSizeX} ${arrowSizeY / 2} l -${arrowSizeX} ${arrowSizeY / 2} z`);
                ctx.stroke(arrow);
                if (outputNodes.length) ctx.fill(arrow);
                ctx.globalAlpha = oldGlobalAlpha;
            } else {
                const on = toggle === "bypass" ? allBypassed : allMuted;
                ctx.beginPath();
                ctx.lineJoin = "round";
                ctx.rect(x, y, BTN_SIZE, BTN_SIZE);
                ctx.lineWidth = 2;
                if (toggle === "mute") {
                    // Closed eye when muted, open eye otherwise.
                    const oldGlobalAlpha = ctx.globalAlpha;
                    ctx.lineCap = "round";
                    if (on) {
                        ctx.stroke(new Path2D(`
                            ${eyeFrame(midX, midY)}
                            ${eyeLashes(midX, midY)}
                        `));
                    } else {
                        const radius = BTN_GRID * 1.5;
                        ctx.fill(new Path2D(`
                            ${eyeFrame(midX, midY)}
                            ${eyeFrame(midX, midY, -1)}
                            ${circlePath(midX, midY, radius)}
                            ${circlePath(midX + BTN_GRID / 2, midY - BTN_GRID / 2, BTN_GRID * 0.375)}
                        `), "evenodd");
                        ctx.stroke(new Path2D(`${eyeFrame(midX, midY)} ${eyeFrame(midX, midY, -1)}`));
                        ctx.globalAlpha = canvas.editor_alpha * 0.5;   // faded lashes
                        ctx.stroke(new Path2D(`${eyeLashes(midX, midY)} ${eyeLashes(midX, midY, -1)}`));
                    }
                    ctx.globalAlpha = oldGlobalAlpha;
                } else {
                    // Arc when bypassed, straight line otherwise.
                    const lineChanges = on
                        ? `a ${BTN_GRID * 3}, ${BTN_GRID * 3} 0 1, 1 ${BTN_GRID * 3 * 2},0 l ${BTN_GRID * 2.0} 0`
                        : `l ${BTN_GRID * 8} 0`;
                    ctx.stroke(new Path2D(`
                          M ${x} ${midY}
                          ${lineChanges}
                          M ${x + BTN_SIZE} ${midY} l -2  2
                          M ${x + BTN_SIZE} ${midY} l -2 -2
                      `));
                    ctx.fill(new Path2D(`${circlePath(x + BTN_GRID * 3, midY, BTN_GRID * 1.8)}`));
                }
            }
        }
    }
    ctx.restore();
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

function installPatches() {
    installed = true;
    const proto = window.LGraphCanvas.prototype;

    // Track the latest pointer event for hover-mode button display.
    const originalAdjustMouseEvent = proto.adjustMouseEvent;
    proto.adjustMouseEvent = function (e) {
        const result = originalAdjustMouseEvent.call(this, e);
        lastMouse = e;
        return result;
    };

    // Button clicks: the native mousedown listener is bound to its method when the
    // canvas is constructed, so patching the prototype is bypassed for the live
    // canvas. Listen on the canvas element directly; handleToggleClick is
    // self-contained, so it works whether it runs before or after the native one.
    app.canvas.canvas.addEventListener("pointerdown", (e) => {
        handleToggleClick(app.canvas, e);
    }, true);

    // Button drawing: run after the native drawGroups.
    const originalDrawGroups = proto.drawGroups;
    proto.drawGroups = function (canvasEl, ctx) {
        const result = originalDrawGroups.call(this, canvasEl, ctx);
        drawToggles(this, ctx);
        return result;
    };

    // In hover mode the buttons follow the cursor, so the canvas needs a redraw as
    // it moves - nudge it periodically so they appear on hover.
    setInterval(() => {
        if (app.canvas) {
            app.canvas.setDirty(true, true);
        }
    }, 250);
}

app.registerExtension({
    name: "GibbyNodes.GroupHeaderFastToggles",
    settings: [
        {
            id: SETTING_PREFIX + ".ToggleQueue",
            name: "Queue button",
            type: "boolean",
            defaultValue: true,
            tooltip: "Queue only the clicked group's output nodes (partial execution)",
        },
        {
            id: SETTING_PREFIX + ".ToggleBypass",
            name: "Bypass button",
            type: "boolean",
            defaultValue: true,
            tooltip: "Flip all of a group's nodes between ALWAYS and BYPASS",
        },
        {
            id: SETTING_PREFIX + ".ToggleMute",
            name: "Mute button",
            type: "boolean",
            defaultValue: true,
            tooltip: "Flip all of a group's nodes between ALWAYS and muted",
        },
        {
            id: SETTING_PREFIX + ".ShowMode",
            name: "Show buttons",
            type: "combo",
            options: ["on hover", "always", "never"],
            defaultValue: "on hover",
            tooltip: "'on hover' shows the buttons only over the group under the mouse; 'always' on every group; 'never' hides all buttons",
        },
    ],
    registerCustomNodes() {
        const doRegister = () => {
            if (installed) return true;
            if (typeof window === "undefined" || !window.LiteGraph || !window.LGraphNode || !window.LGraphCanvas || !app.canvas) return false;
            try {
                installPatches();
                return true;
            } catch (e) {
                console.error("[Gibby GroupHeaderToggles] patching LGraphCanvas failed:", e);
                return false;
            }
        };

        if (doRegister()) return;

        // The canvas globals only exist once the graph view has mounted - which
        // can be a beat after this extension loads, so poll for them.
        let attempts = 0;
        const timer = setInterval(() => {
            attempts += 1;
            const ok = doRegister();
            if (ok || attempts >= 80) clearInterval(timer);   // ~20s: give up quietly
        }, 250);
    },
});

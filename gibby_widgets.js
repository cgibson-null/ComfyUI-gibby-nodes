// ---------------------------------------------------------------------------
// Gibby Nodes - shared widget helpers
//
// The pieces every node frontend here needs: the live node instance (a tab
// switch rebuilds the graph in place, so a captured instance goes stale),
// widget lookup and writes, the options.hidden row hide, and the two layout
// passes a show/hide needs - the core's computeSize refit and the Vue widget
// list re-render. The crop preview lives in crop_preview.js.
// ---------------------------------------------------------------------------

import { app } from "../../scripts/app.js";

// Resolve the live instance in the graph currently shown: a tab switch
// rebuilds the graph in place (new instance, same id), so a global registry
// would go stale and writes would hit a dead instance.
export function liveNode(id) {
    const graph =
        (app.canvas && app.canvas.getCurrentGraph && app.canvas.getCurrentGraph()) ||
        app.graph;
    return graph ? graph.getNodeById(id) : null;
}

// The row's widget. The dotted option also matches a DynamicCombo row by its
// key, which the core names "<combo>.<key>".
export function findWidget(node, name, { dotted = false } = {}) {
    return node.widgets?.find((w) => w.name === name || (dotted && w.name.endsWith("." + name)));
}

export function getWidgetValue(node, name) {
    return findWidget(node, name)?.value;
}

// Write through to the standard widget object so prompt serialization and Vue
// reactivity both see the change.
export function setWidgetValue(node, name, value) {
    const w = findWidget(node, name);
    if (w) w.value = value;
}

export function setWidgetHidden(widget, hidden) {
    if (!widget) return;
    widget.options = widget.options || {};
    widget.options.hidden = !!hidden;
    widget.hidden = !!hidden;

    // A DOM widget's floating element is outside the Vue widget list, so it
    // needs the display flag as well.
    const el = widget._gibbyDomEl || widget.el || widget.domElement;
    if (el) el.style.display = hidden ? "none" : "";
}

// Refit the node through the core layout pass: it assigns the DOM widgets'
// positions (widget.y) that their floating elements track, and grows or
// shrinks the node with the visible rows.
export function refitNode(node) {
    if (!node || node._removed || !node.size) return;
    try {
        node.size = [node.size[0], node.computeSize([...node.size])[1]];
    } catch (e) { /* ignore */ }
}

// The Vue widget list only reads options.hidden when it re-renders; a
// same-tick showAdvanced toggle-and-restore forces that pass without changing
// the node's actual advanced state.
export function rerenderWidgetList(node) {
    const prev = node.showAdvanced;
    node.showAdvanced = !prev;
    node.showAdvanced = prev;
}

export function markGraphDirty(node) {
    try { node.graph?.setDirtyCanvas(true, true); } catch (e) { /* ignore */ }
}

// Show/hide the named rows and run the two passes a change needs: the core's
// computeSize refit (it re-measures the node with the rows shown/hidden) and
// the Vue widget list re-render (it only reads options.hidden when it
// re-renders). wanted is {widgetName: show}; a missing widget is skipped.
export function applyRowVisibility(node, wanted) {
    let changed = false;
    for (const [name, show] of Object.entries(wanted)) {
        const w = findWidget(node, name, { dotted: true });
        if (!w) continue;
        changed = changed || !!w.options?.hidden === show;
        setWidgetHidden(w, !show);
    }
    if (!changed) return;
    refitNode(node);
    rerenderWidgetList(node);
    markGraphDirty(node);
}

// The plumbing a "hide some rows based on another widget" extension needs:
// apply sync(node, hook) when the node is created, after configure (the saved
// values and the saved mode land there) and one tick after the named widgets'
// own handlers - a mode combo rebuilds its rows then. Everything is looked up
// by name on every pass: a DynamicCombo recreates its rows, so a captured
// widget or row goes stale. widgets[0] is the widget whose presence says the
// node is ready; until then setup retries once, shortly.
export function rowVisibility(nodeType, { widgets, sync }) {
    const hook = (node, name) => {
        const w = findWidget(node, name, { dotted: true });
        if (!w || w._gibbyRowHook) return;
        w._gibbyRowHook = true;
        const prev = w.callback;
        w.callback = function (...args) {
            const r = prev?.apply(this, args);
            setTimeout(() => sync(liveNode(node.id) || node, hook), 0);
            return r;
        };
    };
    const setup = (node) => {
        if (!node || node._removed) return false;
        for (const name of widgets) hook(node, name);
        sync(node, hook);
        return !!findWidget(node, widgets[0], { dotted: true });
    };
    for (const name of ["onNodeCreated", "onConfigure"]) {
        const orig = nodeType.prototype[name];
        nodeType.prototype[name] = function () {
            if (orig) {
                try { orig.apply(this, arguments); } catch (e) { /* ignore */ }
            }
            if (!setup(this)) setTimeout(() => setup(liveNode(this.id) || this), 50);
        };
    }
}

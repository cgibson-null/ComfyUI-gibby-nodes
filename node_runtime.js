// Gibby Nodes - Node runtime display
// Shows execution time above each node after it runs, including subgraphs (sum of children).

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const SETTING_KEY = "GibbyNodes.RuntimeDisplay";
const ORDER_KEY = "GibbyNodes.RuntimeDisplay.Order";
const TIME_PROP = "gibbyExecTime";
const ORDER_PROP = "gibbyExecOrder";

if (!app.ui.settings.settingsLookup[SETTING_KEY]) {
    app.ui.settings.addSetting({
        id: SETTING_KEY,
        name: "Show execution time",
        type: "boolean",
        defaultValue: false,
        tooltip: "Show execution time above each node",
    });
}
if (!app.ui.settings.settingsLookup[ORDER_KEY]) {
    app.ui.settings.addSetting({
        id: ORDER_KEY,
        name: "Show execution order",
        type: "boolean",
        defaultValue: false,
        tooltip: "Show execution order number before the time",
    });
}

let lastId = null;
let lastStart = null;
let execOrder = 0;

function findNodeById(id, graph) {
    if (!graph) return null;
    if (id.includes(":")) {
        const [parentId, childId] = id.split(":");
        const parent = findNodeById(parentId, graph);
        if (parent?.subgraph?.nodes) {
            return parent.subgraph.nodes.find(n => String(n.id) === childId) || null;
        }
        return null;
    }
    return graph.getNodeById(id) || graph._nodes?.find(n => String(n.id) === String(id)) || null;
}

function sumSubgraphTime(node) {
    let total = 0;
    if (node[TIME_PROP]) total += node[TIME_PROP];
    if (node.isSubgraphNode && node.isSubgraphNode() && node.subgraph?.nodes) {
        for (const child of node.subgraph.nodes) {
            total += sumSubgraphTime(child);
        }
    }
    return total;
}

function minSubgraphOrder(node) {
    let min = null;
    function walk(n) {
        if (n[ORDER_PROP] != null) {
            if (min == null || n[ORDER_PROP] < min) min = n[ORDER_PROP];
        }
        if (n.isSubgraphNode && n.isSubgraphNode() && n.subgraph?.nodes) {
            for (const child of n.subgraph.nodes) walk(child);
        }
    }
    if (node[ORDER_PROP] != null) min = node[ORDER_PROP];
    if (node.subgraph?.nodes) {
        for (const child of node.subgraph.nodes) walk(child);
    }
    return min;
}

function clearAllOrders(graph) {
    if (!graph) return;
    function walk(nodes) {
        for (const node of nodes) {
            if (ORDER_PROP in node) delete node[ORDER_PROP];
            if (node.isSubgraphNode && node.isSubgraphNode() && node.subgraph?.nodes) {
                walk(node.subgraph.nodes);
            }
        }
    }
    walk(graph._nodes || []);
}

app.registerExtension({
    name: "GibbyNodes.RuntimeDisplay",

    setup() {
        const origDraw = LGraphCanvas.prototype.draw;
        LGraphCanvas.prototype.draw = function (...args) {
            const result = origDraw.apply(this, args);
            const showTime = app.ui.settings.getSettingValue(SETTING_KEY);
            const showOrder = app.ui.settings.getSettingValue(ORDER_KEY);
            if (!showTime && !showOrder) return result;
            if (!this.graph) return result;

            const nodes = this.graph._nodes || [];
            let hasAnything = false;
            for (const n of nodes) {
                if (TIME_PROP in n || ORDER_PROP in n || (n.isSubgraphNode && n.isSubgraphNode())) { hasAnything = true; break; }
            }
            if (!hasAnything) return result;

            try {
                const ctx = this.canvas?.getContext("2d");
                if (!ctx) return result;

                ctx.save();
                const ds = this.ds;
                ctx.setTransform(ds.scale, 0, 0, ds.scale, ds.offset[0] * ds.scale, ds.offset[1] * ds.scale);

                for (const node of nodes) {
                    let parts = [];

                    if (showOrder) {
                        let order = node[ORDER_PROP] != null ? node[ORDER_PROP] : (node.isSubgraphNode && node.isSubgraphNode() ? minSubgraphOrder(node) : null);
                        if (order != null) parts.push(String(order + 1));
                    }

                    if (showTime) {
                        let duration = null;
                        if (node.isSubgraphNode && node.isSubgraphNode()) {
                            duration = sumSubgraphTime(node);
                        } else if (TIME_PROP in node) {
                            duration = node[TIME_PROP];
                        }
                        if (duration != null) parts.push(parseFloat(duration).toFixed(3) + "s");
                    }

                    if (parts.length === 0) continue;
                    const text = parts.join(" ");
                    ctx.font = "12px monospace";
                    const w = ctx.measureText(text).width + 10;
                    const h = 18;
                    const x = node.pos[0];
                    const y = node.pos[1] - LiteGraph.NODE_TITLE_HEIGHT - 20;

                    ctx.fillStyle = "#1a1a2e";
                    ctx.fillRect(x, y, w, h);
                    ctx.fillStyle = "#ccc";
                    ctx.fillText(text, x + 5, y + 13);
                }
                ctx.restore();
            } catch (e) {}
            return result;
        };

        api.addEventListener("executing", (data) => {
            const enabled = app.ui.settings.getSettingValue(SETTING_KEY) || app.ui.settings.getSettingValue(ORDER_KEY);
            if (!enabled) return;

            const id = data?.detail || null;

            // null = run boundary: finalize the last pending node, then reset
            if (!id) {
                if (lastId && lastStart) {
                    const delta = (Date.now() - lastStart) / 1000;
                    const node = findNodeById(lastId, app.graph);
                    if (node) {
                        node[TIME_PROP] = delta;
                        node[ORDER_PROP] = execOrder;
                    }
                }
                execOrder = 0;
                lastId = null;
                lastStart = null;
                return;
            }

            // Finalize previous node's time and order
            if (lastId && lastStart) {
                const delta = (Date.now() - lastStart) / 1000;
                const node = findNodeById(lastId, app.graph);
                if (node) {
                    node[TIME_PROP] = delta;
                    node[ORDER_PROP] = execOrder;
                }
                execOrder++;
            } else if (!lastId) {
                // First node of a new run: clear stale orders, keep times
                clearAllOrders(app.graph);
                execOrder = 0;
            }

            lastId = id;
            lastStart = Date.now();
        });
    },
});

import { app } from "../../../scripts/app.js";

// Gibby Nodes - Fast Groups Muter / Bypasser
// --------------------------------------
// A self-contained reimplementation of rgthree's "Fast Groups Muter" and
// "Fast Groups Bypasser", with the exact same filtering options, but built the
// way this package does everything else: real HTML elements added through
// ComfyUI's own node.addDOMWidget() - NOT hand-drawn canvas widgets.
//
// HOW THEY WORK (same as rgthree's):
//   * Frontend-only "virtual" nodes (isVirtualNode=true) that are never sent
//     to the backend - they only flip the modes of the nodes inside groups.
//   * They list every group in the workflow (one toggle row each). Toggling a
//     row flips ALL nodes inside that group:
//         - Muter    -> Active (ALWAYS) <-> Muted (NEVER)
//         - Bypasser -> Active (ALWAYS) <-> Bypassed (BYPASS)
//   * Filtering / ordering via node "Properties" (right-click -> Properties):
//         matchColors, matchTitle (regex), showNav, showAllGraphs,
//         sort (position | alphanumeric | custom alphabet),
//         customSortAlphabet, toggleRestriction (default | max one | always one)
//   * Right-click the node for bulk actions: Mute/Bypass all, Enable all,
//     Toggle all.

// LiteGraph node "mode" values (mirrors rgthree's muter.js constants).
const MODE_ALWAYS = 0;  // running
const MODE_MUTE = 2;    // muted (Fast Groups Muter "off")
const MODE_BYPASS = 4;  // bypassed (Fast Groups Bypasser "off")

const MUTER_TYPE = "Gibby_FastGroupsMuter";
const BYPASSER_TYPE = "Gibby_FastGroupsBypasser";
const CATEGORY = "gibby";

const ROW_HEIGHT = 26;          // height of a single toggle row
const MIN_WIDTH = 300;          // node is always at least this wide
const REFRESH_INTERVAL_MS = 400;

// ---------------------------------------------------------------------------
// Shared refresh scheduler: all Fast Groups nodes refresh together on one
// timer instead of each spinning up its own. The timer only runs while at
// least one of these nodes is on the canvas, and self-cleans removed nodes.
// ---------------------------------------------------------------------------
const _refreshers = new Set();
let _refreshTimer = null;

function ensureRefreshTimer() {
    if (_refreshers.size > 0 && _refreshTimer === null) {
        _refreshTimer = setInterval(tickRefresh, REFRESH_INTERVAL_MS);
    } else if (_refreshers.size === 0 && _refreshTimer !== null) {
        clearInterval(_refreshTimer);
        _refreshTimer = null;
    }
}

function tickRefresh() {
    for (const node of Array.from(_refreshers)) {
        if (node._removed) {
            _refreshers.delete(node);
            continue;
        }
        try {
            node._ensureInitialized();
        } catch (e) {
            console.error("[Gibby FastGroups] refresh failed:", e);
        }
    }
    if (_refreshers.size === 0 && _refreshTimer !== null) {
        clearInterval(_refreshTimer);
        _refreshTimer = null;
    }
}

function registerFastGroupsNode(node) {
    if (!node || node._removed) return;
    _refreshers.add(node);
    ensureRefreshTimer();
}

function unregisterFastGroupsNode(node) {
    _refreshers.delete(node);
    if (_refreshers.size === 0 && _refreshTimer !== null) {
        clearInterval(_refreshTimer);
        _refreshTimer = null;
    }
}

// ---------------------------------------------------------------------------
// Node context-menu hook for the NEW frontend (nodes 2.0).
// ---------------------------------------------------------------------------
// The new Vue menu builds its items via `canvas.getNodeMenuOptions(node)`,
// which (in the LiteGraph compat layer):
//   * calls node.getExtraMenuOptions(canvas, menu)  -> we use this for the
//     bulk actions (Mute/Bypass all, Enable all, Toggle all);
//   * ends with node.graph?.onGetNodeMenuOptions?.(menu, node) -> we use this
//     to inject the node's Properties (the built-in "Properties" item is on
//     the converter's blocklist and is never re-added by the Vue menu).
// We chain any pre-existing hook and only inject for our own node types. The
// guard flag makes this idempotent across multiple nodes / refreshes.
// ---------------------------------------------------------------------------
function hookGraphMenu(graph) {
    if (!graph || graph._gibbyFastGroupsMenuHooked) return;
    const original = graph.onGetNodeMenuOptions;
    graph.onGetNodeMenuOptions = (menu, node) => {
        if (typeof original === "function") {
            try { original.call(graph, menu, node); } catch (e) { /* ignore */ }
        }
        // No additional menu items - settings are accessed via the gear icon
    };
    graph._gibbyFastGroupsMenuHooked = true;
}

// Build the settings panel content for the node
function buildSettingsContent(node) {
    const p = node.properties || {};
    const refresh = () => { try { node._refreshFastGroups(); } catch (e) { /* ignore */ } };
    
    // Get comfy's node colors
    const getComfyColors = () => {
        const colors = {};
        if (typeof LGraphCanvas !== 'undefined' && LGraphCanvas.node_colors) {
            for (const [name, colorData] of Object.entries(LGraphCanvas.node_colors)) {
                colors[name] = colorData.groupcolor || colorData.color || '#808080';
            }
        }
        return colors;
    };
    
    const colors = getComfyColors();
    const colorOptions = Object.keys(colors).sort();
    
    // Match Colors dropdown - allows selecting multiple colors
    const matchColorsDiv = document.createElement("div");
    matchColorsDiv.style.marginBottom = "8px";
    const matchColorsLabel = document.createElement("label");
    matchColorsLabel.textContent = "Match Colors:";
    matchColorsLabel.style.cssText = "display: block; margin-bottom: 4px; font-size: 11px; color: #aaa;";
    matchColorsDiv.appendChild(matchColorsLabel);
    
    const matchColorsSelect = document.createElement("select");
    matchColorsSelect.multiple = true;
    matchColorsSelect.size = 5;
    matchColorsSelect.style.cssText = "width: 100%; font-size: 11px; padding: 2px; background: #2a2a2a; color: #ddd; border: 1px solid #444;";
    
    // Add "None" option
    const noneOption = document.createElement("option");
    noneOption.value = "";
    noneOption.textContent = "(None)";
    matchColorsSelect.appendChild(noneOption);
    
    // Add color options
    for (const colorName of colorOptions) {
        const option = document.createElement("option");
        option.value = colorName;
        option.textContent = colorName;
        // Set background color to show the actual color
        option.style.backgroundColor = colors[colorName];
        if (p.matchColors && p.matchColors.toLowerCase().includes(colorName.toLowerCase())) {
            option.selected = true;
        }
        matchColorsSelect.appendChild(option);
    }
    
    matchColorsSelect.addEventListener("change", () => {
        const selected = Array.from(matchColorsSelect.selectedOptions).map(o => o.value).filter(v => v !== "").join(",");
        p.matchColors = selected;
        refresh();
    });
    matchColorsDiv.appendChild(matchColorsSelect);
    
    // Match Title input
    const matchTitleDiv = document.createElement("div");
    matchTitleDiv.style.marginBottom = "8px";
    const matchTitleLabel = document.createElement("label");
    matchTitleLabel.textContent = "Match Title (regex):";
    matchTitleLabel.style.cssText = "display: block; margin-bottom: 4px; font-size: 11px; color: #aaa;";
    matchTitleDiv.appendChild(matchTitleLabel);
    
    const matchTitleInput = document.createElement("input");
    matchTitleInput.type = "text";
    matchTitleInput.value = p.matchTitle || "";
    matchTitleInput.placeholder = "Regex pattern";
    matchTitleInput.style.cssText = "width: 100%; font-size: 11px; padding: 4px; background: #2a2a2a; color: #ddd; border: 1px solid #444; box-sizing: border-box;";
    matchTitleInput.addEventListener("change", () => {
        p.matchTitle = matchTitleInput.value;
        refresh();
    });
    matchTitleDiv.appendChild(matchTitleInput);
    
    // Show Nav toggle
    const showNavDiv = document.createElement("div");
    showNavDiv.style.marginBottom = "8px";
    const showNavLabel = document.createElement("label");
    showNavLabel.style.cssText = "display: flex; align-items: center; gap: 6px; cursor: pointer; font-size: 11px; color: #ddd;";
    const showNavCheckbox = document.createElement("input");
    showNavCheckbox.type = "checkbox";
    showNavCheckbox.checked = p.showNav !== false;
    showNavCheckbox.addEventListener("change", () => {
        p.showNav = showNavCheckbox.checked;
        refresh();
    });
    showNavLabel.appendChild(showNavCheckbox);
    showNavLabel.appendChild(document.createTextNode("Show Nav"));
    showNavDiv.appendChild(showNavLabel);
    
    // Show All Graphs toggle
    const showAllGraphsDiv = document.createElement("div");
    showAllGraphsDiv.style.marginBottom = "8px";
    const showAllGraphsLabel = document.createElement("label");
    showAllGraphsLabel.style.cssText = "display: flex; align-items: center; gap: 6px; cursor: pointer; font-size: 11px; color: #ddd;";
    const showAllGraphsCheckbox = document.createElement("input");
    showAllGraphsCheckbox.type = "checkbox";
    showAllGraphsCheckbox.checked = p.showAllGraphs !== false;
    showAllGraphsCheckbox.addEventListener("change", () => {
        p.showAllGraphs = showAllGraphsCheckbox.checked;
        refresh();
    });
    showAllGraphsLabel.appendChild(showAllGraphsCheckbox);
    showAllGraphsLabel.appendChild(document.createTextNode("Show All Graphs"));
    showAllGraphsDiv.appendChild(showAllGraphsLabel);
    
    // Sort dropdown
    const sortDiv = document.createElement("div");
    sortDiv.style.marginBottom = "8px";
    const sortLabel = document.createElement("label");
    sortLabel.textContent = "Sort:";
    sortLabel.style.cssText = "display: block; margin-bottom: 4px; font-size: 11px; color: #aaa;";
    sortDiv.appendChild(sortLabel);
    
    const sortSelect = document.createElement("select");
    sortSelect.style.cssText = "width: 100%; font-size: 11px; padding: 4px; background: #2a2a2a; color: #ddd; border: 1px solid #444;";
    const sorts = ["position", "alphanumeric", "custom alphabet"];
    for (const s of sorts) {
        const option = document.createElement("option");
        option.value = s;
        option.textContent = s;
        if ((p.sort || "position") === s) option.selected = true;
        sortSelect.appendChild(option);
    }
    sortSelect.addEventListener("change", () => {
        p.sort = sortSelect.value;
        refresh();
    });
    sortDiv.appendChild(sortSelect);
    
    // Custom Alphabet input
    const customAlphabetDiv = document.createElement("div");
    customAlphabetDiv.style.marginBottom = "8px";
    const customAlphabetLabel = document.createElement("label");
    customAlphabetLabel.textContent = "Custom Alphabet:";
    customAlphabetLabel.style.cssText = "display: block; margin-bottom: 4px; font-size: 11px; color: #aaa;";
    customAlphabetDiv.appendChild(customAlphabetLabel);
    
    const customAlphabetInput = document.createElement("input");
    customAlphabetInput.type = "text";
    customAlphabetInput.value = p.customSortAlphabet || "";
    customAlphabetInput.placeholder = "Characters (no commas)";
    customAlphabetInput.style.cssText = "width: 100%; font-size: 11px; padding: 4px; background: #2a2a2a; color: #ddd; border: 1px solid #444; box-sizing: border-box;";
    customAlphabetInput.addEventListener("change", () => {
        p.customSortAlphabet = customAlphabetInput.value;
        refresh();
    });
    customAlphabetDiv.appendChild(customAlphabetInput);
    
    // Toggle Restriction dropdown
    const toggleRestrictionDiv = document.createElement("div");
    toggleRestrictionDiv.style.marginBottom = "4px";
    const toggleRestrictionLabel = document.createElement("label");
    toggleRestrictionLabel.textContent = "Toggle Restriction:";
    toggleRestrictionLabel.style.cssText = "display: block; margin-bottom: 4px; font-size: 11px; color: #aaa;";
    toggleRestrictionDiv.appendChild(toggleRestrictionLabel);
    
    const toggleRestrictionSelect = document.createElement("select");
    toggleRestrictionSelect.style.cssText = "width: 100%; font-size: 11px; padding: 4px; background: #2a2a2a; color: #ddd; border: 1px solid #444;";
    const restrictions = ["default", "max one", "always one"];
    for (const r of restrictions) {
        const option = document.createElement("option");
        option.value = r;
        option.textContent = r;
        if ((p.toggleRestriction || "default") === r) option.selected = true;
        toggleRestrictionSelect.appendChild(option);
    }
    toggleRestrictionSelect.addEventListener("change", () => {
        p.toggleRestriction = toggleRestrictionSelect.value;
        refresh();
    });
    toggleRestrictionDiv.appendChild(toggleRestrictionSelect);
    
    // Build the container
    const container = document.createElement("div");
    container.style.cssText = "padding: 8px; background: #1e1e1e; border-radius: 4px;";
    container.appendChild(matchColorsDiv);
    container.appendChild(matchTitleDiv);
    container.appendChild(showNavDiv);
    container.appendChild(showAllGraphsDiv);
    container.appendChild(sortDiv);
    container.appendChild(customAlphabetDiv);
    container.appendChild(toggleRestrictionDiv);
    
    return container;
}

// ---------------------------------------------------------------------------
// Group / node helpers (self-contained, no dependency on rgthree).
// ---------------------------------------------------------------------------

// Return the real LGraphNode instances inside a group. The group's own
// `_children` set can be stale, so prefer computing from the bounding box:
// a node "belongs" to the group if its center is inside the group.
// Subgraph nodes ARE included here (unlike nested group rectangles) - they're
// real nodes that can be muted/bypassed, and setNodeMode() recurses into them.
function getNodesInGroup(group) {
    const graph = group.graph;
    if (!graph) return [];
    const bounds = group._bounding; // [x, y, w, h]
    let nodes = [];
    if (bounds && graph._nodes) {
        for (const n of graph._nodes) {
            if (!n || n === group) continue;
            if (n.type === "GroupNode") continue;
            let b = null;
            try { b = n.getBounding ? n.getBounding() : null; } catch (e) { b = null; }
            if (!b && n.pos && n.size) b = [n.pos[0], n.pos[1], n.size[0], n.size[1]];
            if (!b) continue;
            const cx = b[0] + b[2] * 0.5;
            const cy = b[1] + b[3] * 0.5;
            if (cx >= bounds[0] && cx < bounds[0] + bounds[2] &&
                cy >= bounds[1] && cy < bounds[1] + bounds[3]) {
                nodes.push(n);
            }
        }
    }
    if (nodes.length) return nodes;
    // Fallback: trust the group's own child set (filter to real nodes).
    if (group._children) {
        nodes = Array.from(group._children).filter((c) => c && typeof c.mode === "number");
    }
    return nodes;
}

// A group counts as "enabled" when at least one of its nodes is not
// muted/bypassed (i.e. still in ALWAYS mode).
function isGroupEnabled(group) {
    const nodes = getNodesInGroup(group);
    if (!nodes.length) return false;
    return nodes.some((n) => n.mode === MODE_ALWAYS);
}

function setNodeMode(n, mode) {
    // Depth-first: also set the mode of every node inside any subgraph. ComfyUI
    // does NOT propagate a subgraph node's mode to its contents on its own, so a
    // plain `n.mode = mode` would leave the subgraph's inner nodes untouched.
    const stack = [n];
    while (stack.length) {
        const node = stack.pop();
        if (!node) continue;
        if (typeof node.setMode === "function") node.setMode(mode);
        else node.mode = mode;
        if (node.isSubgraphNode && node.isSubgraphNode() && node.subgraph && node.subgraph.nodes) {
            for (const child of node.subgraph.nodes) stack.push(child);
        }
    }
    if (n.graph) n.graph.setDirtyCanvas(true, true);
}

// Normalise a color (name or hex) to a lowercase "#rrggbb" string, or null.
function normalizeColor(raw) {
    if (!raw) return null;
    let color = String(raw).trim().toLowerCase();
    try {
        if (typeof LGraphCanvas !== "undefined" && LGraphCanvas.node_colors && LGraphCanvas.node_colors[color]) {
            color = LGraphCanvas.node_colors[color].groupcolor || LGraphCanvas.node_colors[color].color || color;
        }
    } catch (e) { /* ignore */ }
    color = String(color).replace("#", "").trim();
    if (color.length === 3) color = color.replace(/(.)(.)(.)/, "$1$1$2$2$3$3");
    if (!/^[0-9a-f]{6}$/.test(color)) return null;
    return "#" + color;
}

// Collect every group in the root graph and all sub-graphs.
// graph.subgraphs always resolves to the root graph's flat _subgraphs map
// (every subgraph at every level), so iterate it exactly once - recursing into
// a subgraph's .subgraphs returns that same map and loops forever.
function collectAllGroups() {
    const out = [];
    let root = null;
    try {
        root = app.rootGraph || (app.graph && app.graph.rootGraph) || app.graph;
    } catch (e) { root = app.graph; }
    if (!root) return out;
    const pushGroups = (g) => {
        if (!g || !g._groups) return;
        g._groups.forEach((gr) => { if (gr) out.push(gr); });
    };
    pushGroups(root);
    if (root.subgraphs) {
        for (const sub of root.subgraphs.values()) pushGroups(sub);
    }
    return out;
}

function sortGroups(groups, sortMode, customAlphabet) {
    const arr = [...groups];
    if (sortMode === "alphanumeric") {
        arr.sort((a, b) => (a.title || "").localeCompare(b.title || ""));
        return arr;
    }
    if (sortMode === "custom alphabet" && customAlphabet && customAlphabet.length) {
        arr.sort((a, b) => {
            const aTitle = (a.title || "").toLowerCase();
            const bTitle = (b.title || "").toLowerCase();
            let aIndex = -1, bIndex = -1;
            for (const [index, alpha] of customAlphabet.entries()) {
                if (aIndex < 0 && aTitle.startsWith(alpha)) aIndex = index;
                if (bIndex < 0 && bTitle.startsWith(alpha)) bIndex = index;
                if (aIndex > -1 && bIndex > -1) break;
            }
            if (aIndex > -1 && bIndex > -1) {
                const ret = aIndex - bIndex;
                return ret !== 0 ? ret : aTitle.localeCompare(bTitle);
            }
            if (aIndex > -1) return -1;
            if (bIndex > -1) return 1;
            return aTitle.localeCompare(bTitle);
        });
        return arr;
    }
    // default: canvas position (top-to-bottom, then left-to-right)
    arr.sort((a, b) => {
        const aY = Math.floor((a._pos ? a._pos[1] : 0) / 30);
        const bY = Math.floor((b._pos ? b._pos[1] : 0) / 30);
        if (aY === bY) {
            return Math.floor((a._pos ? a._pos[0] : 0) / 30) - Math.floor((b._pos ? b._pos[0] : 0) / 30);
        }
        return aY - bY;
    });
    return arr;
}

// Apply the color / title / current-graph filters.
function filterGroups(groups, props) {
    let currentGraph = null;
    try {
        currentGraph = (app.canvas && app.canvas.getCurrentGraph && app.canvas.getCurrentGraph()) || app.graph;
    } catch (e) { currentGraph = app.graph; }

    const matchColors = String(props.matchColors || "")
        .split(",")
        .map((c) => normalizeColor(c))
        .filter(Boolean);
    const matchTitle = String(props.matchTitle || "").trim();

    const out = [];
    for (const group of groups) {
        if (props.showAllGraphs === false && currentGraph && group.graph !== currentGraph) continue;

        if (matchColors.length) {
            const gcolor = normalizeColor(group.color);
            if (!gcolor || !matchColors.includes(gcolor)) continue;
        }

        if (matchTitle) {
            let ok = false;
            try {
                ok = !!new RegExp(matchTitle, "i").exec(group.title || "");
            } catch (e) {
                // Fall back to a plain substring match if the pattern is invalid.
                ok = (group.title || "").toLowerCase().includes(matchTitle.toLowerCase());
            }
            if (!ok) continue;
        }

        out.push(group);
    }
    return out;
}

// Recenter the canvas on a group at the CURRENT zoom. Prefer the canvas's own
// centerOnNode() - it applies the transform the way LiteGraph actually does
// (screen = (graph + offset) * scale * dpr) and handles devicePixelRatio. The
// old code computed offset as "cw/2 - cx*scale", the inverse of the real
// transform, so the view landed far off the group (worse at high zoom/DPI).
function navigateToGroup(group) {
    const canvas = app.canvas;
    if (!canvas || !canvas.ds || !group._bounding) return;
    const b = group._bounding;
    
    // Ensure maximum zoom level of 100%
    if (canvas.ds.scale > 1) {
        canvas.ds.scale = 1;
    }
    
    if (typeof canvas.centerOnNode === "function") {
        canvas.centerOnNode({ pos: [b[0], b[1]], size: [b[2], b[3]] });
        return;
    }
    
    const cx = b[0] + b[2] / 2;
    const cy = b[1] + b[3] / 2;
    const dpr = (typeof window !== "undefined" && window.devicePixelRatio) || 1;
    const scale = canvas.ds.scale || 1;
    canvas.ds.offset[0] = -cx + canvas.canvas.width * 0.5 / (scale * dpr);
    canvas.ds.offset[1] = -cy + canvas.canvas.height * 0.5 / (scale * dpr);
    canvas.setDirty(true, true);
}

// ---------------------------------------------------------------------------
// Non-reactive storage for DOM references.
// ---------------------------------------------------------------------------
// CRITICAL for the new frontend: the node instance is a Vue 3 reactive
// proxy. If we stored DOM elements on `this` (e.g. `this._container = div`),
// Vue would wrap them in reactive proxies, and DOM APIs like
// `appendChild()` would reject them ("parameter 1 is not of type 'Node'"),
// because a Vue proxy of an element is NOT a valid Node.
//
// So every DOM reference (the container div, each row div, the placeholder)
// lives in this module-level WeakMap instead, keyed by the node. The Map is
// plain JS (not reactive), so the real DOM objects stay intact and can be
// handed straight to `appendChild`, `remove()`, etc. WeakMap also means the
// entries are garbage-collected automatically when the node is removed.
// ---------------------------------------------------------------------------
const _domState = new WeakMap(); // node -> { container, placeholder, rows: Map }

function getDomState(node) {
    let state = _domState.get(node);
    if (!state) {
        state = { container: null, placeholder: null, rows: new Map() };
        _domState.set(node, state);
    }
    return state;
}
function clearDomState(node) {
    const state = _domState.get(node);
    if (state) {
        state.rows.forEach((r) => { try { r.element.remove(); } catch (e) { /* ignore */ } });
        state.rows.clear();
        if (state.placeholder) { try { state.placeholder.remove(); } catch (e) { /* ignore */ } }
        state.container = null;
        state.placeholder = null;
    }
    _domState.delete(node);
}

// ---------------------------------------------------------------------------
// Base class shared by the Muter and the Bypasser.
// ---------------------------------------------------------------------------
function buildFastGroupsClasses() {
    // Resolve the node base class at CALL TIME, not at module import time.
    // The new ComfyUI frontend only exposes `LGraphNode` as a global once
    // the canvas has mounted, which can be AFTER the extension module is
    // imported - referencing it at the top level used to kill the entire
    // import with "ReferenceError: LGraphNode is not defined", which is why
    // the Muter/Bypasser nodes never showed up while lora_loader.js (which
    // never touches LGraphNode) kept working.
    const Base =
        (typeof window !== "undefined" && window.LGraphNode) ||
        (typeof globalThis !== "undefined" && globalThis.LGraphNode) ||
        null;
    if (!Base) return null;

    class GibbyBaseFastGroups extends Base {
    constructor(title) {
        super(title);
        this.isVirtualNode = true;          // never sent to the backend
        this.serialize_widgets = false;     // state is derived from node modes
        this.collapsable = true;
        this.modeOn = MODE_ALWAYS;
        this.modeOff = MODE_MUTE;           // overridden by the Bypasser
        this.helpActions = "mute and unmute";
        // DOM references (container / rows / placeholder) are deliberately NOT
        // stored on `this` - see the _domState WeakMap at the top of this file.
        // Storing real DOM nodes on the (Vue-reactive) node instance turns them
        // into proxies that DOM APIs like appendChild() will reject.
        this._visibleGroups = [];
        this._lastSignature = null;
        this._lastConfigKey = null;
        this._removed = false;

        // Default filtering options (all editable via right-click -> Properties)
        this.properties = {
            matchColors: "",
            matchTitle: "",
            showNav: true,
            showAllGraphs: true,
            sort: "position",
            customSortAlphabet: "",
            toggleRestriction: "default",
        };
    }

    // ---- lifecycle ------------------------------------------------------
    onNodeCreated() {
        registerFastGroupsNode(this);
        this._ensureInitialized();
    }
    onAdded() {
        this._removed = false;
        // Wait for the node to be fully attached to the graph
        setTimeout(() => this._ensureInitialized(), 50);
    }
    onRemoved() {
        this._removed = true;
        unregisterFastGroupsNode(this);
        clearDomState(this);
        this._visibleGroups = [];
        this._lastSignature = null;
        this._lastConfigKey = null;
        // Clean up outside click handler
        if (this._outsideClickHandler) {
            document.removeEventListener("mousedown", this._outsideClickHandler);
            this._outsideClickHandler = null;
        }
        // Clean up gear icon
        if (this._gearIconEl) {
            try { this._gearIconEl.remove(); } catch (e) { /* ignore */ }
            this._gearIconEl = null;
        }
        // Clean up settings panel
        if (this._settingsPanel) {
            try { this._settingsPanel.remove(); } catch (e) { /* ignore */ }
            this._settingsPanel = null;
        }
    }

    // ---- initialisation -------------------------------------------------
    _ensureInitialized() {
        if (this._removed) return;
        if (!this.graph) return; // Wait until node has a graph
        hookGraphMenu(this.graph); // idempotent; wires the nodes-2.0 context menu
        if (!getDomState(this).container) {
            try {
                this._buildContainer();
            } catch (e) {
                // Not attached to a graph yet - the shared timer retries.
                console.warn("[Gibby FastGroups] container deferred:", e);
            }
        }
        registerFastGroupsNode(this); // idempotent
        if (getDomState(this).container) this._refreshFastGroups();
    }

    _buildContainer() {
        if (getDomState(this).container) return;
        const container = document.createElement("div");
        container.style.width = "100%";
        container.style.boxSizing = "border-box";
        container.style.padding = "4px 6px 6px 6px";
        container.style.display = "flex";
        container.style.flexDirection = "column";
        container.style.gap = "3px";
        container.style.minHeight = `${ROW_HEIGHT}px`;
        container.style.position = "relative";
        // Quick-action toolbar pinned to the top of the node.
        container.appendChild(this._buildToolbar());
        this.addDOMWidget("fast_groups", "GIBBY_FAST_GROUPS", container, {
            getHeight: () => this._containerHeight(),
        });
        // Only record it (in the non-reactive WeakMap) once the widget accepted it.
        getDomState(this).container = container;
        
        // Add gear icon to node header (top right)
        this._buildHeaderGearIcon();
    }

    // Quick-action buttons (Mute/Bypass all, Enable all, Toggle all) shown at
    // the top of the node. Plain DOM - kept out of the reactive node instance.
    _buildToolbar() {
        const toolbar = document.createElement("div");
        toolbar.style.display = "flex";
        toolbar.style.gap = "4px";
        toolbar.style.padding = "2px 0 5px 0";
        toolbar.style.borderBottom = "1px solid #3a3a3a";
        toolbar.style.marginBottom = "2px";

        const allLabel = this.modeOff === MODE_BYPASS ? "Bypass all" : "Mute all";
        const buttons = [
            { label: "Toggle all", action: () => this._actionToggleAll() },
            { label: "Enable all", action: () => this._actionAll(true) },
            { label: allLabel, action: () => this._actionAll(false) },
        ];

        for (const btn of buttons) {
            const b = document.createElement("button");
            b.textContent = btn.label;
            b.style.cssText =
                "flex:1; padding:3px 4px; font-size:11px; cursor:pointer; " +
                "background:#2a2a2a; color:#ccc; border:1px solid #444; " +
                "border-radius:3px; font-family:inherit; line-height:1.2;";
            b.addEventListener("mouseenter", () => { b.style.background = "#3a3a3a"; });
            b.addEventListener("mouseleave", () => { b.style.background = "#2a2a2a"; });
            // Prevent the canvas from treating the click as a node drag/select.
            b.addEventListener("mousedown", (e) => e.stopPropagation());
            b.addEventListener("click", (e) => {
                e.stopPropagation();
                try { btn.action(); } catch (err) {
                    console.error("[Gibby FastGroups] button action failed:", err);
                }
            });
            toolbar.appendChild(b);
        }
        return toolbar;
    }

    // ---- sizing ---------------------------------------------------------
    _containerHeight() {
        // toolbar + rows + the gaps between them + vertical padding; be
        // generous so nothing gets clipped regardless of whether computeSize()
        // also accounts for the DOM widget's own height.
        const count = Math.max(1, this._visibleGroups.length);
        return count * (ROW_HEIGHT + 3) + 12 + 34; // 34 = toolbar block
    }

    computeSize() {
        const size = super.computeSize();
        size[0] = Math.max(MIN_WIDTH, size[0] || MIN_WIDTH);
        size[1] = Math.max(this._containerHeight() + 8, size[1] || 60);
        return size;
    }

    // ---- refresh --------------------------------------------------------
    _resolveSort() {
        const props = this.properties;
        let sortMode = props.sort || "position";
        let customAlphabet = null;
        if (sortMode === "custom alphabet") {
            const s = String(props.customSortAlphabet || "").replace(/\n/g, "");
            if (s && s.trim()) {
                customAlphabet = s.includes(",")
                    ? s.toLocaleLowerCase().split(",").map((x) => x.trim()).filter(Boolean)
                    : s.toLocaleLowerCase().trim().split("");
            }
            if (!customAlphabet || !customAlphabet.length) {
                sortMode = "alphanumeric";
                customAlphabet = null;
            }
        }
        return { sortMode, customAlphabet };
    }

    _refreshFastGroups() {
        const state = getDomState(this);
        if (!state.container) return; // container is built by _ensureInitialized
        
        const props = this.properties;

        // If a layout-affecting option (currently showNav) changed, rebuild all
        // rows from scratch rather than patching them in place.
        const configKey = `nav:${props.showNav !== false}`;
        if (configKey !== this._lastConfigKey) {
            for (const row of state.rows.values()) row.element.remove();
            state.rows.clear();
            if (state.placeholder) {
                state.placeholder.remove();
                state.placeholder = null;
            }
            this._lastConfigKey = configKey;
            this._lastSignature = null;
        }

        const { sortMode, customAlphabet } = this._resolveSort();

        let groups = collectAllGroups();
        groups = filterGroups(groups, props);
        groups = sortGroups(groups, sortMode, customAlphabet);

        // Skip all DOM work unless something actually changed.
        const signature = groups
            .map((g) => `${g.id}|${g.title || ""}|${isGroupEnabled(g) ? 1 : 0}`)
            .join(";");
        
        // Also check if existing rows are still in DOM (they might have been removed on undo)
        const rowsNeedRebuild = Array.from(state.rows.values()).some(
            (row) => !row.element.isConnected
        );
        
        if (signature === this._lastSignature && !rowsNeedRebuild) return;
        this._lastSignature = signature;

        // Drop rows whose group is no longer visible.
        for (const [group, row] of Array.from(state.rows.entries())) {
            if (!groups.includes(group)) {
                row.element.remove();
                state.rows.delete(group);
            }
        }

        // Create any new rows, then update label/state and enforce order.
        for (const group of groups) {
            let row = state.rows.get(group);
            if (!row) {
                row = this._createRow(group);
                state.rows.set(group, row);
            }
            const enabled = isGroupEnabled(group);
            row.labelEl.textContent = group.title || "";
            row.enabledState = enabled;
            this._updateRowVisual(row, enabled);
            state.container.appendChild(row.element); // reorders in place
        }

        this._updatePlaceholder(groups.length);
        this._visibleGroups = groups;

        this.setSize(this.computeSize());
        if (this.graph) this.graph.setDirtyCanvas(true, false);
    }

    _updatePlaceholder(count) {
        const state = getDomState(this);
        if (count === 0) {
            if (!state.placeholder) {
                const el = document.createElement("div");
                el.textContent = this._placeholderText();
                el.style.color = "#777";
                el.style.fontSize = "11px";
                el.style.textAlign = "center";
                el.style.height = `${ROW_HEIGHT}px`;
                el.style.lineHeight = `${ROW_HEIGHT}px`;
                el.style.cursor = "default";
                state.placeholder = el;
            }
            state.container.appendChild(state.placeholder);
        } else if (state.placeholder) {
            state.placeholder.remove();
            state.placeholder = null;
        }
    }

    _placeholderText() {
        return "No matching groups - adjust the node's Properties filter";
    }

    // ---- row rendering --------------------------------------------------
    _createRow(group) {
        const row = document.createElement("div");
        row.style.display = "flex";
        row.style.alignItems = "center";
        row.style.gap = "6px";
        row.style.width = "100%";
        row.style.boxSizing = "border-box";
        row.style.height = `${ROW_HEIGHT}px`;

        // Optional "jump to group" arrow.
        if (this.properties.showNav !== false) {
            const nav = document.createElement("div");
            nav.textContent = "\u25b6"; // ▶
            nav.title = "Jump to this group";
            nav.style.cursor = "pointer";
            nav.style.color = "#89A";
            nav.style.fontSize = "11px";
            nav.style.flex = "0 0 auto";
            nav.style.padding = "0 3px";
            nav.style.userSelect = "none";
            nav.addEventListener("click", (e) => {
                e.stopPropagation();
                e.preventDefault();
                navigateToGroup(group);
            });
            row.appendChild(nav);
        }

        // The ON/OFF toggle.
        const toggle = document.createElement("div");
        toggle.title = "Toggle this group";
        toggle.style.cursor = "pointer";
        toggle.style.flex = "0 0 auto";
        toggle.style.width = "50px";
        toggle.style.height = "18px";
        toggle.style.borderRadius = "9px";
        toggle.style.boxSizing = "border-box";
        toggle.style.userSelect = "none";
        toggle.addEventListener("click", (e) => {
            e.stopPropagation();
            e.preventDefault();
            this._doModeChange(group, undefined);
        });
        row.appendChild(toggle);

        // The group title.
        const labelEl = document.createElement("div");
        labelEl.style.flex = "1 1 auto";
        labelEl.style.minWidth = "0";
        labelEl.style.fontSize = "11px";
        labelEl.style.color = "#ddd";
        labelEl.style.whiteSpace = "nowrap";
        labelEl.style.overflow = "hidden";
        labelEl.style.textOverflow = "ellipsis";
        row.appendChild(labelEl);

        const rowObj = { element: row, toggleEl: toggle, labelEl, enabledState: false };
        this._updateRowVisual(rowObj, false);
        getDomState(this).container.appendChild(row); // `row` is the <div>; rowObj.element is the same node
        return rowObj;
    }

    _updateRowVisual(row, enabled) {
        const t = row.toggleEl;
        t.style.display = "flex";
        t.style.alignItems = "center";
        t.style.justifyContent = "center";
        t.style.fontSize = "10px";
        t.style.fontWeight = "bold";
        t.style.letterSpacing = "0.5px";
        if (enabled) {
            t.style.background = "#3a7d44";
            t.style.color = "#cfe8d2";
            t.textContent = "ON";
        } else {
            t.style.background = "#3a3a3a";
            t.style.color = "#999";
            t.textContent = "OFF";
        }
    }

    // ---- mode changes ---------------------------------------------------
    _doModeChange(group, force) {
        const restriction = this.properties.toggleRestriction || "default";
        const enabledNow = isGroupEnabled(group);
        let newValue = (force !== undefined) ? force : !enabledNow;

        if (newValue && restriction && restriction.includes("one")) {
            // Enabling this one disables every other group first.
            for (const g of this._visibleGroups) {
                if (g !== group && isGroupEnabled(g)) this._applyMode(g, false);
            }
        } else if (!newValue && restriction === "always one") {
            // Keep at least one group enabled.
            const enabledCount = this._visibleGroups.filter((g) => isGroupEnabled(g)).length;
            if (enabledCount <= 1) newValue = true;
        }

        this._applyMode(group, newValue);
        this._lastSignature = null; // force a visual refresh
        this._refreshFastGroups();
    }

    _applyMode(group, enabled) {
        const mode = enabled ? this.modeOn : this.modeOff;
        for (const n of getNodesInGroup(group)) {
            setNodeMode(n, mode);
        }
        if (this.graph) this.graph.setDirtyCanvas(true, true);
    }

    _countEnabled() {
        return this._visibleGroups.filter((g) => isGroupEnabled(g)).length;
    }

    // ---- bulk actions (right-click menu) --------------------------------
    _actionAll(enabled) {
        const restriction = this.properties.toggleRestriction || "default";
        const onlyOne = restriction && restriction.includes("one");
        const alwaysOne = restriction === "always one";
        const groups = this._visibleGroups;
        for (let i = 0; i < groups.length; i++) {
            let value = enabled;
            if (enabled && onlyOne) value = i === 0;          // only first stays on
            if (!enabled && alwaysOne) value = i === 0;        // keep first on
            this._applyMode(groups[i], value);
        }
        this._lastSignature = null;
        this._refreshFastGroups();
    }

    _actionToggleAll() {
        const restriction = this.properties.toggleRestriction || "default";
        const onlyOne = restriction && restriction.includes("one");
        let foundOne = false;
        const groups = this._visibleGroups;
        for (let i = 0; i < groups.length; i++) {
            let newValue = onlyOne && foundOne ? false : !isGroupEnabled(groups[i]);
            foundOne = foundOne || newValue;
            this._applyMode(groups[i], newValue);
        }
        if (!foundOne && restriction === "always one" && groups.length) {
            this._applyMode(groups[groups.length - 1], true);
        }
        this._lastSignature = null;
        this._refreshFastGroups();
    }

    getExtraMenuOptions() {
        // No additional menu options - settings are accessed via the gear icon
        return [];
    }

    // ---- settings panel -------------------------------------------------
    _buildHeaderGearIcon() {
        const container = getDomState(this).container;
        if (!container) return;

        const gearIcon = document.createElement("div");
        // Position snugly in the top right corner of the header
        gearIcon.style.cssText = "position: absolute; top: -36px; right: 2px; width: 18px; height: 18px; " +
            "display: flex; align-items: center; justify-content: center; cursor: pointer; " +
            "font-size: 13px; color: #888; user-select: none; z-index: 10; " +
            "background: rgba(0,0,0,0.3); border-radius: 3px;";
        gearIcon.textContent = "⚙";
        gearIcon.title = "Settings";

        gearIcon.addEventListener("mousedown", (e) => e.stopPropagation());
        gearIcon.addEventListener("click", (e) => {
            e.stopPropagation();
            e.preventDefault();
            this._toggleSettingsPanel();
        });

        container.appendChild(gearIcon);
        // Store reference for cleanup
        this._gearIconEl = gearIcon;
    }

    _buildCloseIcon() {
        const closeIcon = document.createElement("div");
        closeIcon.style.cssText = "position: absolute; top: 4px; right: 4px; width: 18px; height: 18px; " +
            "display: flex; align-items: center; justify-content: center; cursor: pointer; " +
            "font-size: 14px; color: #888; user-select: none; z-index: 10;";
        closeIcon.textContent = "✕";
        closeIcon.title = "Close settings";

        closeIcon.addEventListener("mousedown", (e) => e.stopPropagation());
        closeIcon.addEventListener("click", (e) => {
            e.stopPropagation();
            e.preventDefault();
            this._closeSettingsPanel();
        });

        return closeIcon;
    }

    _buildSettingsPanel() {
        if (this._settingsPanel) return;

        const container = getDomState(this).container;
        if (!container) return;

        this._settingsPanel = document.createElement("div");
        // Position to the right of the node, aligned with the header
        this._settingsPanel.style.cssText = "position: absolute; top: -24px; left: 100%; width: 240px; " +
            "background: #1e1e1e; border: 1px solid #444; border-radius: 6px; padding: 8px; " +
            "box-shadow: 0 4px 12px rgba(0,0,0,0.5); z-index: 100; display: none; margin-left: 8px;";

        // Add close icon
        this._settingsPanel.appendChild(this._buildCloseIcon());

        // Add settings content
        const content = buildSettingsContent(this);
        this._settingsPanel.appendChild(content);

        // Append to container
        container.appendChild(this._settingsPanel);
    }

    _toggleSettingsPanel() {
        if (!this._settingsPanel) {
            this._buildSettingsPanel();
            this._setupOutsideClickHandler();
        }
        const panel = this._settingsPanel;
        const showing = panel.style.display === "block";
        panel.style.display = showing ? "none" : "block";
        
        // Close any other open settings panels
        for (const node of _refreshers) {
            if (node !== this && node._settingsPanel && node._settingsPanel.style.display === "block") {
                node._settingsPanel.style.display = "none";
            }
        }
    }

    _closeSettingsPanel() {
        if (this._settingsPanel) {
            this._settingsPanel.style.display = "none";
        }
    }

    _setupOutsideClickHandler() {
        if (this._outsideClickHandler) return;
        
        this._outsideClickHandler = (e) => {
            // If the click is outside the settings panel and not on the gear icon, close it
            if (this._settingsPanel && this._settingsPanel.style.display === "block") {
                const isInsidePanel = this._settingsPanel.contains(e.target);
                const isOnGear = e.target.closest && e.target.closest('[title="Settings"]');
                if (!isInsidePanel && !isOnGear) {
                    this._closeSettingsPanel();
                }
            }
        };
        
        document.addEventListener("mousedown", this._outsideClickHandler);
    }
}
// ---------------------------------------------------------------------------
// Concrete nodes.
// ---------------------------------------------------------------------------
class GibbyFastGroupsMuter extends GibbyBaseFastGroups {
    constructor() {
        super("Fast Groups Muter");
        this.modeOn = MODE_ALWAYS;
        this.modeOff = MODE_MUTE;      // off => muted
        this.helpActions = "mute and unmute";
    }
    _placeholderText() {
        return "No matching groups to mute - adjust the node's Properties filter";
    }
}

class GibbyFastGroupsBypasser extends GibbyBaseFastGroups {
    constructor() {
        super("Fast Groups Bypasser");
        this.modeOn = MODE_ALWAYS;
        this.modeOff = MODE_BYPASS;    // off => bypassed
        this.helpActions = "bypass and enable";
    }
    _placeholderText() {
        return "No matching groups to bypass - adjust the node's Properties filter";
    }
}

    return {
        Base,
        GibbyFastGroupsMuter,
        GibbyFastGroupsBypasser,
    };
}

// ---------------------------------------------------------------------------
// Register both as frontend-only virtual nodes (no backend needed).
// Registration is deferred until LGraphNode is actually available (see
// buildFastGroupsClasses) and retried for a while, because on the new
// frontend the global only exists once the canvas has mounted - which can
// be a beat after this extension loads.
// ---------------------------------------------------------------------------
app.registerExtension({
    name: "GibbyNodes.fastGroups",
    registerCustomNodes() {
        const doRegister = () => {
            try {
                const built = buildFastGroupsClasses();
                if (!built) return false;
                const {
                    Base,
                    GibbyFastGroupsMuter: Muter,
                    GibbyFastGroupsBypasser: Bypasser,
                } = built;

                // title_mode is cosmetic; the constant lives on the LiteGraph
                // namespace in the new frontend builds.
                const NORMAL_TITLE =
                    (typeof window !== "undefined" && window.LiteGraph && window.LiteGraph.NORMAL_TITLE) ||
                    (Base.NORMAL_TITLE !== undefined ? Base.NORMAL_TITLE : undefined);
                const titleMode =
                    NORMAL_TITLE !== undefined ? { title_mode: NORMAL_TITLE } : {};

                // registerNodeType() lives on the LiteGraph namespace object
                // (window.LiteGraph) in the new frontend builds, but is a
                // static of the LGraphNode class in older ones. Try both.
                // Find the LiteGraph object and its registerNodeType method
                const liteGraphObj =
                    (typeof window !== "undefined" && window.LiteGraph) ||
                    (typeof globalThis !== "undefined" && globalThis.LiteGraph) ||
                    null;
                
                if (!liteGraphObj || typeof liteGraphObj.registerNodeType !== "function") {
                    throw new Error("registerNodeType not found on LiteGraph");
                }

                // Call with proper context to ensure 'this' is set correctly
                const registerNodeType = liteGraphObj.registerNodeType.bind(liteGraphObj);

                registerNodeType(
                    MUTER_TYPE,
                    Object.assign(Muter, {
                        title: "Fast Groups Muter",
                        ...titleMode,
                        collapsable: true,
                    })
                );
                Muter.category = CATEGORY;

                registerNodeType(
                    BYPASSER_TYPE,
                    Object.assign(Bypasser, {
                        title: "Fast Groups Bypasser",
                        ...titleMode,
                        collapsable: true,
                    })
                );
                Bypasser.category = CATEGORY;
                return true;
            } catch (e) {
                console.error("[Gibby FastGroups] registration failed:", e);
                return false;
            }
        };

        if (doRegister()) return;

        // Not ready yet - poll until the canvas (and therefore window.LGraphNode)
        // exists, then refresh the node definitions so the Add Node menu picks
        // up the late registration.
        let attempts = 0;
        const timer = setInterval(() => {
            attempts += 1;
            const ok = doRegister();
            if (ok || attempts >= 120) {
                clearInterval(timer);
                if (!ok) {
                    console.error(
                        "[Gibby FastGroups] LGraphNode / registerNodeType never became available; " +
                        "Fast Groups Muter / Bypasser were not registered"
                    );
                } else if (attempts > 1) {
                    try {
                        if (typeof app.refreshComboInNodes === "function") {
                            app.refreshComboInNodes();
                        }
                    } catch (e) { /* ignore */ }
                }
            }
        }, 250);
    },
});
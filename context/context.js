import { app } from "../../../scripts/app.js";

// Gibby Nodes - Slot suggestions
// ---------------------------------
// The right-click/drag "Add Node" menu on a custom-type slot lists nodes from
// LiteGraph.slot_types_default_out/in (core Comfy.SlotDefaults), which only
// considers REQUIRED inputs and slices each list to the
// Comfy.NodeSuggestions.number setting. Nodes whose slot is optional by design
// never make it in - append them manually after registration (like rgthree
// does): dragging from an output offers nodes that accept the type, dragging
// into an input offers nodes that produce it.
//
// The core rebuilds both LiteGraph globals from its own lists on every node
// registration (including subgraph types registered when a workflow loads) and
// on suggestion-count changes, which wipes a one-time patch - so the append is
// re-applied after each rebuild by hooking the core extension's setDefaults.
//
// Entries are kept as object entries {node: className, title: label}: the title
// reaches the node created from the entry (createDefaultNodeForSlot). The
// drag-out menu itself does not show it - LGraphCanvas.showConnectionMenu
// flattens every entry to entry.node before building the ContextMenu, so the
// suggestion rows would read as ids (GibbyLoraLoader). ContextMenu.addItem is
// therefore wrapped to relabel a bare class-name entry with the node's display
// name; the clicked value stays the class name, so the entry still resolves.

const SUGGESTED_TYPES = ["CONTEXT", "GIBBY_KSAMPLER_OPTIONS", "GIBBY_CROP_INFO", "GIBBY_TILING_INFO", "LORA_STACK", "LLAMACPP_CONNECTIVITY", "LLAMACPP_OPTIONS"];

const classToDisplay = new Map(); // comfyClass -> display_name (when different)

function entryName(entry) {
    return typeof entry === "string" ? entry : entry.node;
}

function displayEntry(entry) {
    if (typeof entry === "string") {
        const display = classToDisplay.get(entry);
        if (display) return {node: entry, title: display};
    }
    return entry;
}

// The display name a menu row should show, for an entry that carries no label of
// its own: a bare class name, or an entry the core reduced to {node: className}.
function displayLabel(entry) {
    if (typeof entry === "string") return classToDisplay.get(entry);
    if (entry && typeof entry.node === "string" && entry.title === undefined && entry.content === undefined) {
        return classToDisplay.get(entry.node);
    }
    return undefined;
}

// Relabel the id rows of every litegraph menu (the drag-out connection menu
// among them) as they are built. Rows that already carry a title/content - the
// node palette's category and node entries - are left alone.
function hookMenuLabels() {
    const proto = LiteGraph.ContextMenu?.prototype;
    if (!proto || proto._gibbyLabelled) return;
    proto._gibbyLabelled = true;
    const addItem = proto.addItem;
    proto.addItem = function (label, entry, options) {
        const element = addItem.call(this, label, entry, options);
        const display = element && displayLabel(entry);
        if (display) element.textContent = display;
        return element;
    };
}

// Autogrow inputs (COMFY_AUTOGROW_V3) hide their real type inside
// spec.template.input - e.g. Merge KSampler Options' option1...option10.
function autogrowMatches(spec, type) {
    const templateInput = spec?.template?.input;
    if (!templateInput || typeof templateInput !== "object") return false;
    return Object.values(templateInput).some(
        (group) => typeof group === "object" && group &&
            Object.values(group).some((s) => specMatches(s, type))
    );
}

function specMatches(spec, type) {
    // spec can be ["TYPE", {}] or {type: "TYPE"} or "TYPE"
    if (Array.isArray(spec)) {
        if (spec[0] === type) return true;
        if (spec[0] === "COMFY_AUTOGROW_V3") return autogrowMatches(spec[1], type);
        return false;
    }
    if (typeof spec === "string") return spec === type;
    if (typeof spec === "object" && spec) {
        const t = spec.type;
        if (Array.isArray(t)) return t.includes(type);
        if (t === type) return true;
        return autogrowMatches(spec, type);
    }
    return false;
}

function hasTypeInput(nodeData, type) {
    const required = nodeData?.input?.required || {};
    const optional = nodeData?.input?.optional || {};
    return [...Object.values(required), ...Object.values(optional)].some(
        (spec) => specMatches(spec, type)
    );
}

function hasTypeOutput(nodeData, type) {
    return (nodeData?.output || []).some((spec) => specMatches(spec, type));
}

const inputTypes = new Map(); // type -> Set of node classes with an input of that type
const outputTypes = new Map(); // type -> Set of node classes with an output of that type

function track(nodeType, nodeData) {
    for (const type of SUGGESTED_TYPES) {
        if (hasTypeInput(nodeData, type)) {
            if (!inputTypes.has(type)) inputTypes.set(type, new Set());
            inputTypes.get(type).add(nodeType);
        }
        if (hasTypeOutput(nodeData, type)) {
            if (!outputTypes.has(type)) outputTypes.set(type, new Set());
            outputTypes.get(type).add(nodeType);
        }
    }
}

// Append every tracked node to the drag lists for its types, in place, sorted
// by name. Idempotent - safe to run after each core rebuild.
function applySuggestions() {
    const sortSuggestions = (suggestions) => {
        const items = suggestions.slice().sort((a, b) => {
            const nameA = (a.title || entryName(a)).toLowerCase();
            const nameB = (b.title || entryName(b)).toLowerCase();
            return nameA.localeCompare(nameB);
        });
        for (let i = 0; i < suggestions.length; i++) {
            suggestions[i] = items[i];
        }
    };

    for (const type of SUGGESTED_TYPES) {
        // Dragging from an output of this type: nodes that accept it
        const outSuggestions =
            LiteGraph.slot_types_default_out[type] ||
            (LiteGraph.slot_types_default_out[type] = ["Reroute"]);
        for (const t of inputTypes.get(type) || []) {
            if (!outSuggestions.some((e) => entryName(e) === t)) outSuggestions.push(t);
        }
        for (let i = 0; i < outSuggestions.length; i++) outSuggestions[i] = displayEntry(outSuggestions[i]);
        sortSuggestions(outSuggestions);

        // Dragging into an input of this type: nodes that produce it
        const inSuggestions =
            LiteGraph.slot_types_default_in[type] ||
            (LiteGraph.slot_types_default_in[type] = ["Reroute"]);
        for (const t of outputTypes.get(type) || []) {
            if (!inSuggestions.some((e) => entryName(e) === t)) inSuggestions.push(t);
        }
        for (let i = 0; i < inSuggestions.length; i++) inSuggestions[i] = displayEntry(inSuggestions[i]);
        sortSuggestions(inSuggestions);
    }

    // Core-owned lists (standard types) get the same display-name treatment.
    for (const lists of [LiteGraph.slot_types_default_out, LiteGraph.slot_types_default_in]) {
        for (const type in lists) {
            const entries = lists[type];
            for (let i = 0; i < entries.length; i++) entries[i] = displayEntry(entries[i]);
        }
    }
}

// Wrap the core SlotDefaults extension's setDefaults so our nodes survive
// every rebuild of the LiteGraph globals it performs.
function hookSlotDefaults() {
    const slotDefaults = app.extensionManager?.enabledExtensions?.find((e) => e.name === "Comfy.SlotDefaults");
    if (!slotDefaults || slotDefaults._gibbyHooked) return;
    slotDefaults._gibbyHooked = true;
    const rebuild = slotDefaults.setDefaults;
    slotDefaults.setDefaults = (value) => {
        rebuild.call(slotDefaults, value);
        scheduleSuggestions();
    };
    applySuggestions();
}

// The core rebuilds the drag lists on every node registration, and this pack
// re-appends its entries after each rebuild, so applying them per registration
// did the whole pass once per node - hundreds of times over a normal startup.
// Registrations arrive in synchronous batches, so one pass per batch is enough:
// the lists are only read when a menu opens.
let suggestionsPending = false;
function scheduleSuggestions() {
    if (suggestionsPending) return;
    suggestionsPending = true;
    queueMicrotask(() => {
        suggestionsPending = false;
        applySuggestions();
    });
}

app.registerExtension({
    name: "Gibby.SlotSuggestions",
    setup() {
        hookMenuLabels();
        hookSlotDefaults();
    },
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.display_name && nodeData.display_name !== nodeType.comfyClass) {
            classToDisplay.set(nodeType.comfyClass, nodeData.display_name);
        }
        track(nodeType.comfyClass, nodeData);
        hookSlotDefaults();
        scheduleSuggestions();
    },
});

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

const SUGGESTED_TYPES = ["CONTEXT", "GIBBY_KSAMPLER_OPTIONS", "GIBBY_CROP_INFO", "LORA_STACK"];

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
            const nameA = (a.content || String(a)).toLowerCase();
            const nameB = (b.content || String(b)).toLowerCase();
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
            if (!outSuggestions.includes(t)) outSuggestions.push(t);
        }
        sortSuggestions(outSuggestions);

        // Dragging into an input of this type: nodes that produce it
        const inSuggestions =
            LiteGraph.slot_types_default_in[type] ||
            (LiteGraph.slot_types_default_in[type] = ["Reroute"]);
        for (const t of outputTypes.get(type) || []) {
            if (!inSuggestions.includes(t)) inSuggestions.push(t);
        }
        sortSuggestions(inSuggestions);
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
        applySuggestions();
    };
    applySuggestions();
}

app.registerExtension({
    name: "Gibby.SlotSuggestions",
    setup() {
        hookSlotDefaults();
    },
    beforeRegisterNodeDef(nodeType, nodeData) {
        track(nodeType.comfyClass, nodeData);
        hookSlotDefaults();
        applySuggestions();
    },
});

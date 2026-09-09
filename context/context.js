import { app } from "../../../scripts/app.js";

// Gibby Nodes - Slot suggestions
// ---------------------------------
// The right-click/drag "Add Node" menu on a custom-type slot lists nodes from
// LiteGraph.slot_types_default_out/in (core Comfy.SlotDefaults), which only
// considers REQUIRED inputs and slices each list to the
// Comfy.NodeSuggestions.number setting. Nodes whose slot is optional by design
// never make it in - append them manually after registration (like rgthree
// does) so every node that accepts or produces the type ends up in the list
// in both directions.

const SUGGESTED_TYPES = ["CONTEXT", "GIBBY_KSAMPLER_OPTIONS"];

function specMatches(spec, type) {
    // spec can be ["TYPE", {}] or {type: "TYPE"} or "TYPE"
    if (Array.isArray(spec) && spec[0] === type) return true;
    if (typeof spec === "string" && spec === type) return true;
    if (typeof spec === "object" && spec && !Array.isArray(spec)) {
        const t = spec.type;
        if (Array.isArray(t)) return t.includes(type);
        return t === type;
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
let scheduled = false;

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

app.registerExtension({
    name: "Gibby.SlotSuggestions",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        track(nodeType.comfyClass, nodeData);

        if (scheduled) return;
        scheduled = true;
        // Wait for all node registrations to finish so the core SlotDefaults
        // extension has rebuilt its lists, then append ourselves and sort.
        setTimeout(() => {
            // Refresh the sets in case new nodes were registered
            for (const [nodeType, nodeData] of Object.entries(LiteGraph.registered_node_types)) {
                track(nodeType, nodeData);
            }
            // Sort suggestion items by display name, in place
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
                const both = new Set([
                    ...(inputTypes.get(type) || []),
                    ...(outputTypes.get(type) || []),
                ]);

                // Dragging from an output of this type
                const outSuggestions =
                    LiteGraph.slot_types_default_out[type] ||
                    (LiteGraph.slot_types_default_out[type] = ["Reroute"]);
                for (const t of both) {
                    if (!outSuggestions.includes(t)) outSuggestions.push(t);
                }
                sortSuggestions(outSuggestions);

                // Dragging into an input of this type
                if (!LiteGraph.slot_types_default_in) LiteGraph.slot_types_default_in = {};
                const inSuggestions =
                    LiteGraph.slot_types_default_in[type] ||
                    (LiteGraph.slot_types_default_in[type] = ["Reroute"]);
                for (const t of both) {
                    if (!inSuggestions.includes(t)) inSuggestions.push(t);
                }
                sortSuggestions(inSuggestions);
            }
        }, 1000);
    },
});

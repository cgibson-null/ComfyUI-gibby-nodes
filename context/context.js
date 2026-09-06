import { app } from "../../../scripts/app.js";

// Gibby Nodes - Context suggestions
// ---------------------------------
// The right-click/drag "Add Node" menu on a CONTEXT output slot lists nodes that
// have a REQUIRED context input (see Comfy.SlotDefaults in the core frontend),
// showing raw class types like "Gibby_KSampler_Context". Two fixes:
// 1. Nodes whose context input is optional by design never show up there -
//    append them manually after registration (like rgthree does). We track every
//    node that accepts a CONTEXT so all of them end up in the list.
// 2. Show display names instead of class types in the menu. Core renders an
//    item's `content` property as its label when present, and resolves node
//    types through object key lookup (which coerces with toString) - so a
//    String wrapper carrying both keeps creation working while renaming it.

function hasContextInput(nodeData) {
    const required = nodeData?.input?.required || {};
    const optional = nodeData?.input?.optional || {};
    const inputs = [...Object.entries(required), ...Object.entries(optional)];
    return inputs.some(([, spec]) => {
        if (!spec) return false;
        // spec can be ["CONTEXT", {}] or {type: "CONTEXT"} or "CONTEXT"
        if (Array.isArray(spec) && spec[0] === "CONTEXT") return true;
        if (typeof spec === "string" && spec === "CONTEXT") return true;
        if (typeof spec === "object" && !Array.isArray(spec)) {
            const type = spec.type;
            if (Array.isArray(type)) return type.includes("CONTEXT");
            return type === "CONTEXT";
        }
        return false;
    });
}

function hasContextOutput(nodeData) {
    const outputs = nodeData?.output || [];
    return outputs.some((type) => {
        if (Array.isArray(type) && type[0] === "CONTEXT") return true;
        if (typeof type === "string" && type === "CONTEXT") return true;
        if (typeof type === "object" && !Array.isArray(type)) {
            const t = type.type;
            if (Array.isArray(t)) return t.includes("CONTEXT");
            return t === "CONTEXT";
        }
        return false;
    });
}

function contextSuggestionItem(nodeType) {
    // Use plain string to avoid issues with createDefaultNodeForSlot
    // Display names can be handled differently
    return nodeType;
}

const contextInputTypes = new Set();
const contextOutputTypes = new Set();
let scheduled = false;

app.registerExtension({
    name: "Gibby.ContextSuggestions",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (hasContextInput(nodeData)) {
            contextInputTypes.add(nodeType.comfyClass);
        }
        if (hasContextOutput(nodeData)) {
            contextOutputTypes.add(nodeType.comfyClass);
        }

        if (scheduled) return;
        scheduled = true;
        // Wait for all node registrations to finish so the core SlotDefaults
        // extension has rebuilt its lists, then append ourselves and rename.
        setTimeout(() => {
            // Refresh the sets in case new nodes were registered
            for (const [nodeType, nodeData] of Object.entries(LiteGraph.registered_node_types)) {
                if (hasContextInput(nodeData)) {
                    contextInputTypes.add(nodeType);
                }
                if (hasContextOutput(nodeData)) {
                    contextOutputTypes.add(nodeType);
                }
            }
            // Helper to sort suggestion items by display name
            const sortSuggestions = (suggestions) => {
                const items = [];
                for (let i = 0; i < suggestions.length; i++) {
                    const item = suggestions[i];
                    if (typeof item === "string") {
                        items.push(contextSuggestionItem(item));
                    } else {
                        items.push(item);
                    }
                }
                // Sort by content/display name
                items.sort((a, b) => {
                    const nameA = (a.content || String(a)).toLowerCase();
                    const nameB = (b.content || String(b)).toLowerCase();
                    return nameA.localeCompare(nameB);
                });
                // Update the suggestions array in place
                for (let i = 0; i < suggestions.length; i++) {
                    suggestions[i] = items[i];
                }
            };

            // For CONTEXT output -> show nodes with CONTEXT inputs
            const outSuggestions = LiteGraph.slot_types_default_out["CONTEXT"] || [];
            for (const type of contextInputTypes) {
                if (!outSuggestions.includes(type)) {
                    outSuggestions.push(type);
                }
            }
            sortSuggestions(outSuggestions);

            // For CONTEXT input -> show nodes with CONTEXT outputs
            // slot_types_default_in might not exist, so we create it
            if (!LiteGraph.slot_types_default_in) {
                LiteGraph.slot_types_default_in = {};
            }
            if (!LiteGraph.slot_types_default_in["CONTEXT"]) {
                LiteGraph.slot_types_default_in["CONTEXT"] = [];
            }
            const inSuggestions = LiteGraph.slot_types_default_in["CONTEXT"];
            for (const type of contextOutputTypes) {
                if (!inSuggestions.includes(type)) {
                    inSuggestions.push(type);
                }
            }
            sortSuggestions(inSuggestions);
        }, 1000);
    },
});

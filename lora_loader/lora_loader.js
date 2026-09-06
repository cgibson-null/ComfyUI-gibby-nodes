import { app } from "../../../scripts/app.js";

// Gibby Nodes - compact dynamic row widgets
// -------------------------------------------------------------
// All lora rows live inside ONE real HTML container element, added via
// ComfyUI's own node.addDOMWidget() function exactly once - not drawn by
// hand on a canvas, and not one widget per row. Individual rows are added
// and removed as plain child <div>s that this code owns outright, rather
// than as separate ComfyUI-managed widgets - this sidesteps any ambiguity
// about how each renderer wraps/positions per-widget DOM elements (which
// caused stale "ghost" rows to survive removal under the legacy renderer
// after being created under Nodes 2.0). Only ONE element's lifecycle is
// ever handed to ComfyUI, and it's never destroyed/recreated - it just
// grows and shrinks its own children - so there's nothing for either
// renderer to leave behind.

const NODE_NAME = "Gibby_LoraLoader";
const ROWS_WIDGET_NAME = "lora_rows";
const ROWS_WIDGET_TYPE = "GIBBY_LORA_ROWS";
const MAX_SLOTS = 50;
const ROW_HEIGHT = 24;

// Fetch the list of available loras once (from the built-in LoraLoader
// node's own schema, so it's always current) and reuse it everywhere.
let _loraOptionsPromise = null;

function fetchLoraOptions() {
    return fetch("/object_info/LoraLoader")
        .then((r) => r.json())
        .then((data) => {
            const names = (data?.LoraLoader?.input?.required?.lora_name?.[0] || []).slice().sort();
            return ["None", ...names];
        })
        .catch(() => ["None"]);
}

function getLoraOptions() {
    if (!_loraOptionsPromise) {
        _loraOptionsPromise = fetchLoraOptions();
    }
    return _loraOptionsPromise;
}

// Invalidate the cached lora list and force a fresh fetch. Called when
// ComfyUI's model refresh button is pressed (via refreshComboInNode).
function invalidateLoraCache() {
    _loraOptionsPromise = null;
}

function styleField(el) {
    el.style.background = "#1b1b1b";
    el.style.color = "#ddd";
    el.style.border = "1px solid #444";
    el.style.borderRadius = "4px";
    el.style.fontSize = "11px";
    el.style.padding = "2px 4px";
    el.style.boxSizing = "border-box";
    el.style.height = "20px";
}

function styleModalField(el) {
    el.style.background = "#1b1b1b";
    el.style.color = "#ddd";
    el.style.border = "1px solid #444";
    el.style.borderRadius = "4px";
    el.style.fontSize = "12px";
    el.style.padding = "4px 6px";
    el.style.boxSizing = "border-box";
    el.style.width = "100%";
    el.style.fontFamily = "inherit";
}

function makeSmallButton(label, color) {
    const btn = document.createElement("button");
    btn.textContent = label;
    btn.style.flex = "0 0 18px";
    btn.style.width = "18px";
    btn.style.height = "18px";
    btn.style.lineHeight = "16px";
    btn.style.padding = "0";
    btn.style.fontSize = "12px";
    btn.style.cursor = "pointer";
    btn.style.background = "#1b1b1b";
    btn.style.color = color;
    btn.style.border = "1px solid #444";
    btn.style.borderRadius = "3px";
    return btn;
}

// A pill-style ON/OFF toggle, matching the Fast Groups Muter's row toggles
// (a 50x18 rounded pill showing ON/OFF) instead of a plain checkbox.
function makeTogglePill(initialOn, onFlip) {
    const el = document.createElement("div");
    el.title = "Enable/disable this lora";
    el.style.cursor = "pointer";
    el.style.flex = "0 0 auto";
    el.style.width = "50px";
    el.style.height = "18px";
    el.style.borderRadius = "9px";
    el.style.boxSizing = "border-box";
    el.style.userSelect = "none";
    el.style.display = "flex";
    el.style.alignItems = "center";
    el.style.justifyContent = "center";
    el.style.fontSize = "10px";
    el.style.fontWeight = "bold";
    el.style.letterSpacing = "0.5px";
    let on = initialOn !== false;
    function paint() {
        el.style.background = on ? "#3a7d44" : "#3a3a3a";
        el.style.color = on ? "#cfe8d2" : "#999";
        el.textContent = on ? "ON" : "OFF";
    }
    paint();
    el.addEventListener("click", (e) => {
        e.stopPropagation();
        e.preventDefault();
        on = !on;
        paint();
        onFlip?.();
    });
    return {
        element: el,
        getOn: () => on,
        setOn: (v) => { on = v !== false; paint(); },
    };
}

// Lets a number input be adjusted by scrolling, but only while it's
// actually focused (clicked into) - otherwise scrolling the mouse wheel
// while just passing over the node would change values unintentionally
// instead of panning/zooming the canvas like normal.
//
// This is registered on window, in the CAPTURE phase, rather than on each
// input directly. Nodes 2.0's canvas has its own wheel-event routing for
// zoom/pan that sits between the browser and any DOM widget - if it claims
// the event first, a listener on the input itself never gets a turn at
// all. A window-level capture listener runs before the event reaches that
// routing layer, so it gets first refusal regardless of which renderer is
// active, and explicitly stops the event from going further once it's
// been used to adjust a focused value.
const _wheelAdjustRegistry = new Map();

function addWheelAdjust(el, config) {
    _wheelAdjustRegistry.set(el, config);
}

function removeWheelAdjust(el) {
    _wheelAdjustRegistry.delete(el);
}

window.addEventListener(
    "wheel",
    (e) => {
        const config = _wheelAdjustRegistry.get(document.activeElement);
        if (!config) return;

        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation();

        const el = document.activeElement;
        const { step, min, max, onChange } = config;
        const direction = e.deltaY < 0 ? 1 : -1;
        let value = (parseFloat(el.value) || 0) + direction * step;
        if (typeof min === "number") value = Math.max(min, value);
        if (typeof max === "number") value = Math.min(max, value);
        const decimals = (String(step).split(".")[1] || "").length;
        value = parseFloat(value.toFixed(decimals));
        el.value = value;
        el.select(); // Keep value selected after wheel adjustment
        onChange?.(value);
    },
    { capture: true, passive: false }
);

function isVideoMedia(img) {
    if (img.type === "video") return true;
    if (img.type === "image") return false;
    return /\.(mp4|webm|mov)(\?|$)/i.test(img.url || "");
}

// A searchable, filter-as-you-type lora picker. Native <select> elements
// only support "jump to the first option starting with what I typed",
// which is useless for oddly-named loras - this instead shows a live
// filtered list, like a normal search box.
// Parse filter text into AND/OR groups:
// - Space-separated = AND (all must match)
// - | separated within a group = OR (any can match)
// - ! prefix on a term = exclude (must NOT contain it)
// Example: "anime 3d|render" → must contain "anime" AND ("3d" or "render")
function parseFilter(text) {
    if (!text || !text.trim()) return null;
    const normalized = text.replace(/\\/g, "/").toLowerCase();
    // Split by spaces to get AND groups
    return normalized.split(/\s+/).filter(Boolean).map((group) => {
        // Each group can have OR alternatives separated by |
        return group.split("|").map((t) => t.trim()).filter(Boolean);
    });
}

function matchesFilter(text, filterGroups) {
    if (!filterGroups || filterGroups.length === 0) return true;
    const lower = text.toLowerCase().replace(/\\/g, "/");
    // Each group is an AND condition: all groups must match. Terms starting
    // with ! are exclusions - the lora must NOT contain them.
    for (const orGroup of filterGroups) {
        const positives = [];
        for (const term of orGroup) {
            if (term.startsWith("!")) {
                if (lower.includes(term.slice(1))) return false;
            } else {
                positives.push(term);
            }
        }
        // A group with only exclusions passes once those hold above.
        if (positives.length && !positives.some((t) => lower.includes(t))) return false;
    }
    return true;
}

function createSearchableLoraSelect(loraOptions, initialLora, onCommit, getGlobalFilter, onChange, isMissingLora = false) {
    const wrapper = document.createElement("div");
    wrapper.style.position = "relative";
    wrapper.style.flex = "1 1 auto";
    wrapper.style.minWidth = "0";

    const input = document.createElement("input");
    input.type = "text";
    input.autocomplete = "off";
    input.spellcheck = false;
    styleField(input);
    input.style.width = "100%";
    input.style.cursor = "text";

    // Allow missing loras (not in options) to be preserved from workflows
    let currentValue = (isMissingLora || loraOptions.includes(initialLora)) ? initialLora : "None";
    input.value = currentValue;
    // Mark if this is a missing lora loaded from a workflow
    let hasMissingLora = isMissingLora && currentValue !== "None";

    const dropdown = document.createElement("div");
    dropdown.style.position = "fixed";
    dropdown.style.zIndex = "10000";
    dropdown.style.maxHeight = "440px";
    dropdown.style.overflowY = "auto";
    dropdown.style.background = "#1b1b1b";
    dropdown.style.border = "1px solid #555";
    dropdown.style.borderRadius = "4px";
    dropdown.style.display = "none";
    dropdown.style.boxShadow = "0 4px 10px rgba(0,0,0,0.5)";
    document.body.appendChild(dropdown);

    let selectedIndex = -1; // for keyboard navigation

    function positionDropdown() {
        const rect = input.getBoundingClientRect();
        const width = Math.max(rect.width, 380);
        let left = rect.left;
        if (left + width > window.innerWidth - 8) {
            left = Math.max(8, window.innerWidth - width - 8);
        }
        dropdown.style.left = `${left}px`;
        dropdown.style.top = `${rect.bottom + 2}px`;
        dropdown.style.width = `${width}px`;
    }

    function renderOptions(filterText) {
        dropdown.innerHTML = "";
        selectedIndex = -1; // reset selection when options change
        const globalFilterRaw = (getGlobalFilter ? getGlobalFilter() : "") || "";
        const localGroups = parseFilter(filterText);
        const globalGroups = parseFilter(globalFilterRaw);

        const filtered = loraOptions.filter((o) => {
            if (!localGroups && !globalGroups) return true;
            const matchesLocal = matchesFilter(o, localGroups);
            const matchesGlobal = matchesFilter(o, globalGroups);
            return matchesLocal && matchesGlobal;
        });

        if (filtered.length === 0) {
            const none = document.createElement("div");
            none.textContent = "No matches";
            none.style.padding = "6px 10px";
            none.style.color = "#888";
            none.style.fontSize = "12px";
            dropdown.appendChild(none);
            return;
        }

        // Try to find and highlight the current value if it's in the list
        const currentIndex = filtered.indexOf(currentValue);
        if (currentIndex !== -1) selectedIndex = currentIndex;

        for (let i = 0; i < filtered.length; i++) {
            const opt = filtered[i];
            const optEl = document.createElement("div");
            optEl.textContent = opt;
            optEl.style.padding = "5px 10px";
            optEl.style.fontSize = "12px";
            optEl.style.cursor = "pointer";
            optEl.style.whiteSpace = "nowrap";
            optEl.style.overflow = "hidden";
            optEl.style.textOverflow = "ellipsis";

            // Highlight the keyboard-selected option
            if (i === selectedIndex) {
                optEl.style.background = "#4a9eff";
                optEl.style.color = "#fff";
            }

            optEl.addEventListener("mouseenter", () => {
                selectedIndex = i;
                updateHighlight();
            });
            optEl.addEventListener("mouseleave", () => {
                if (i === selectedIndex) {
                    optEl.style.background = "#4a9eff";
                    optEl.style.color = "#fff";
                } else {
                    optEl.style.background = "";
                    optEl.style.color = "";
                }
            });
            optEl.addEventListener("mousedown", (e) => {
                e.preventDefault();
                selectOption(opt);
            });
            dropdown.appendChild(optEl);
        }

        // Scroll selected item into view
        if (selectedIndex >= 0 && selectedIndex < filtered.length) {
            const el = dropdown.children[selectedIndex];
            if (el) el.scrollIntoView({ block: "nearest" });
        }
    }

    function updateHighlight() {
        for (let i = 0; i < dropdown.children.length; i++) {
            const el = dropdown.children[i];
            if (i === selectedIndex && el.textContent !== "No matches") {
                el.style.background = "#4a9eff";
                el.style.color = "#fff";
            } else {
                el.style.background = "";
                el.style.color = "";
            }
        }
    }

    function selectOption(opt) {
        input.value = opt;
        const changed = opt !== currentValue;
        currentValue = opt;
        hideDropdown();
        if (changed) {
            onCommit?.(currentValue);
            onChange?.(currentValue);
        }
    }

    function showDropdown() {
        positionDropdown();
        renderOptions("");
        dropdown.style.display = "block";
    }

    function hideDropdown() {
        dropdown.style.display = "none";
    }

    input.addEventListener("focus", () => {
        input.select();
        showDropdown();
    });
    input.addEventListener("input", () => {
        positionDropdown();
        renderOptions(input.value);
        dropdown.style.display = "block";
    });
    input.addEventListener("blur", () => {
        setTimeout(() => {
            // Allow missing loras to persist (they were loaded from a workflow)
            if (loraOptions.includes(input.value) || (hasMissingLora && input.value !== "None")) {
                const changed = input.value !== currentValue;
                currentValue = input.value;
                if (changed) {
                    onCommit?.(currentValue);
                    // Clear the missing flag if user selected a valid lora
                    if (loraOptions.includes(currentValue)) {
                        hasMissingLora = false;
                    }
                }
            } else {
                input.value = currentValue;
            }
            hideDropdown();
        }, 150);
    });
    input.addEventListener("keydown", (e) => {
        const visibleOptions = dropdown.children.length > 0 && dropdown.children[0].textContent !== "No matches" ? dropdown.children.length : 0;

        if (e.key === "ArrowDown") {
            e.preventDefault();
            if (dropdown.style.display === "none") {
                showDropdown();
            } else if (visibleOptions > 0) {
                selectedIndex = Math.min(selectedIndex + 1, visibleOptions - 1);
                updateHighlight();
                // Update input to show the selected option's text for easier typing
                const el = dropdown.children[selectedIndex];
                if (el && el.textContent !== "No matches") {
                    input.value = el.textContent;
                }
            }
        } else if (e.key === "ArrowUp") {
            e.preventDefault();
            if (visibleOptions > 0) {
                selectedIndex = Math.max(selectedIndex - 1, 0);
                updateHighlight();
                const el = dropdown.children[selectedIndex];
                if (el && el.textContent !== "No matches") {
                    input.value = el.textContent;
                }
            }
        } else if (e.key === "Enter") {
            e.preventDefault();
            // Select the highlighted option, or accept current typed value
            if (selectedIndex >= 0 && selectedIndex < visibleOptions) {
                const el = dropdown.children[selectedIndex];
                if (el && el.textContent !== "No matches") {
                    selectOption(el.textContent);
                } else {
                    input.blur();
                }
            } else {
                input.blur();
            }
        } else if (e.key === "Escape") {
            input.value = currentValue;
            hideDropdown();
            input.blur();
        }
    });

    const onScrollOrResize = () => {
        if (dropdown.style.display !== "none") positionDropdown();
    };
    window.addEventListener("scroll", onScrollOrResize, true);
    window.addEventListener("resize", onScrollOrResize);

    wrapper.appendChild(input);

    return {
        wrapper,
        getValue: () => currentValue,
        setValue: (value) => {
            // Always allow setting the value, even if it's not in the options list
            // (this allows showing missing loras with a red highlight)
            input.value = value;
            currentValue = value;
            // Set the missing flag if the value is not in the options list
            if (value !== "None" && !loraOptions.includes(value)) {
                hasMissingLora = true;
            }
        },
        // Update the options list (called when ComfyUI refreshes model lists).
        // Always preserves current selection, even if not in the new list.
        updateOptions: (newOptions) => {
            loraOptions = newOptions;
            // Mark as missing if not in the list (will highlight red)
            if (currentValue !== "None" && !loraOptions.includes(currentValue)) {
                hasMissingLora = true;
            }
        },
        destroy: () => {
            dropdown.remove();
            window.removeEventListener("scroll", onScrollOrResize, true);
            window.removeEventListener("resize", onScrollOrResize);
        },
    };
}

// Opens a focused view of one gallery image/video - lets a video actually
// play (with sound, since it's now the only thing on screen) and gives the
// prompt text an explicit "Copy Prompt" button instead of just a hover
// tooltip.
function openMediaPreview(img) {
    const backdrop = document.createElement("div");
    backdrop.style.position = "fixed";
    backdrop.style.inset = "0";
    backdrop.style.background = "rgba(0,0,0,0.6)";
    backdrop.style.zIndex = "20100";

    const panel = document.createElement("div");
    panel.style.position = "fixed";
    panel.style.top = "50%";
    panel.style.left = "50%";
    panel.style.transform = "translate(-50%, -50%)";
    panel.style.width = "min(560px, 92vw)";
    panel.style.maxHeight = "88vh";
    panel.style.overflowY = "auto";
    panel.style.background = "#232323";
    panel.style.border = "1px solid #555";
    panel.style.borderRadius = "8px";
    panel.style.padding = "14px";
    panel.style.zIndex = "20101";
    panel.style.boxShadow = "0 8px 30px rgba(0,0,0,0.6)";

    function close() {
        backdrop.remove();
        panel.remove();
        document.removeEventListener("keydown", onKeydown);
    }
    function onKeydown(e) {
        if (e.key === "Escape") close();
    }
    document.addEventListener("keydown", onKeydown);
    backdrop.addEventListener("mousedown", close);
    panel.addEventListener("mousedown", (e) => e.stopPropagation());

    const closeBtn = document.createElement("button");
    closeBtn.textContent = "\u00d7";
    closeBtn.style.position = "absolute";
    closeBtn.style.top = "6px";
    closeBtn.style.right = "10px";
    closeBtn.style.background = "transparent";
    closeBtn.style.border = "none";
    closeBtn.style.color = "#ccc";
    closeBtn.style.fontSize = "22px";
    closeBtn.style.cursor = "pointer";
    closeBtn.addEventListener("click", close);
    panel.appendChild(closeBtn);

    let mediaEl;
    if (isVideoMedia(img)) {
        mediaEl = document.createElement("video");
        mediaEl.src = img.url;
        mediaEl.controls = true;
        mediaEl.autoplay = true;
        mediaEl.loop = true;
        // Not muted here - this is a deliberate, single-item, user-triggered
        // view, unlike the grid thumbnails.
    } else {
        mediaEl = document.createElement("img");
        mediaEl.src = img.url;
    }
    mediaEl.style.display = "block";
    mediaEl.style.maxWidth = "100%";
    mediaEl.style.maxHeight = "55vh";
    mediaEl.style.margin = "4px auto 12px";
    mediaEl.style.borderRadius = "6px";
    panel.appendChild(mediaEl);

    const promptLabel = document.createElement("div");
    promptLabel.textContent = "Prompt";
    promptLabel.style.color = "#999";
    promptLabel.style.fontSize = "12px";
    promptLabel.style.marginBottom = "4px";
    panel.appendChild(promptLabel);

    const promptBox = document.createElement("textarea");
    promptBox.readOnly = true;
    promptBox.value = img.prompt || "(no prompt data for this item)";
    promptBox.rows = 4;
    styleModalField(promptBox);
    promptBox.style.resize = "vertical";
    panel.appendChild(promptBox);

    const copyBtn = document.createElement("button");
    copyBtn.textContent = "Copy Prompt";
    styleModalField(copyBtn);
    copyBtn.style.width = "auto";
    copyBtn.style.marginTop = "8px";
    copyBtn.style.cursor = "pointer";
    copyBtn.disabled = !img.prompt;
    copyBtn.style.opacity = img.prompt ? "1" : "0.5";
    copyBtn.addEventListener("click", () => {
        navigator.clipboard?.writeText(img.prompt || "").catch(() => {});
        const original = copyBtn.textContent;
        copyBtn.textContent = "Copied!";
        setTimeout(() => (copyBtn.textContent = original), 800);
    });
    panel.appendChild(copyBtn);

    document.body.appendChild(backdrop);
    document.body.appendChild(panel);
}

// The "i" button opens this - a small self-contained info/Civitai panel. Built
// fresh each time it's opened and torn down on close, same reasoning as
// the search dropdown: plain HTML/CSS, nothing hand-drawn.
function openLoraInfoModal(loraFilename) {
    if (!loraFilename || loraFilename === "None") {
        return;
    }

    const backdrop = document.createElement("div");
    backdrop.style.position = "fixed";
    backdrop.style.inset = "0";
    backdrop.style.background = "rgba(0,0,0,0.5)";
    backdrop.style.zIndex = "20000";

    const modal = document.createElement("div");
    modal.style.position = "fixed";
    modal.style.top = "50%";
    modal.style.left = "50%";
    modal.style.transform = "translate(-50%, -50%)";
    modal.style.width = "min(640px, 92vw)";
    modal.style.maxHeight = "84vh";
    modal.style.overflowY = "auto";
    modal.style.background = "#232323";
    modal.style.border = "1px solid #555";
    modal.style.borderRadius = "8px";
    modal.style.padding = "16px";
    modal.style.zIndex = "20001";
    modal.style.color = "#ddd";
    modal.style.fontSize = "13px";
    modal.style.boxShadow = "0 8px 30px rgba(0,0,0,0.6)";

    function close() {
        backdrop.remove();
        modal.remove();
        document.removeEventListener("keydown", onKeydown);
    }
    function onKeydown(e) {
        if (e.key === "Escape") close();
    }
    document.addEventListener("keydown", onKeydown);
    backdrop.addEventListener("mousedown", close);
    modal.addEventListener("mousedown", (e) => e.stopPropagation());

    const closeBtn = document.createElement("button");
    closeBtn.textContent = "\u00d7";
    closeBtn.style.position = "absolute";
    closeBtn.style.top = "8px";
    closeBtn.style.right = "12px";
    closeBtn.style.background = "transparent";
    closeBtn.style.border = "none";
    closeBtn.style.color = "#ccc";
    closeBtn.style.fontSize = "22px";
    closeBtn.style.cursor = "pointer";
    closeBtn.addEventListener("click", close);
    modal.appendChild(closeBtn);

    const title = document.createElement("div");
    title.textContent = loraFilename;
    title.style.fontWeight = "bold";
    title.style.fontSize = "14px";
    title.style.marginBottom = "10px";
    title.style.paddingRight = "24px";
    title.style.wordBreak = "break-all";
    modal.appendChild(title);

    const table = document.createElement("div");
    table.style.display = "grid";
    table.style.gridTemplateColumns = "120px 1fr";
    table.style.rowGap = "8px";
    table.style.columnGap = "10px";
    table.style.alignItems = "center";
    modal.appendChild(table);

    function addFieldRow(label) {
        const labelEl = document.createElement("div");
        labelEl.textContent = label;
        labelEl.style.color = "#999";
        const valueEl = document.createElement("div");
        table.appendChild(labelEl);
        table.appendChild(valueEl);
        return valueEl;
    }

    const fileValue = addFieldRow("File");
    fileValue.textContent = loraFilename;
    fileValue.style.wordBreak = "break-all";

    const hashValue = addFieldRow("Hash (sha256)");
    hashValue.textContent = "...";
    hashValue.style.wordBreak = "break-all";
    hashValue.style.fontSize = "11px";

    const civitaiValue = addFieldRow("Civitai");
    const civitaiLinkContainer = document.createElement("div");
    civitaiLinkContainer.style.marginBottom = "4px";
    const fetchBtn = document.createElement("button");
    fetchBtn.textContent = "Fetch info from Civitai";
    styleModalField(fetchBtn);
    fetchBtn.style.cursor = "pointer";
    fetchBtn.style.width = "auto";
    civitaiValue.appendChild(civitaiLinkContainer);
    civitaiValue.appendChild(fetchBtn);
    const civitaiError = document.createElement("div");
    civitaiError.style.color = "#e88";
    civitaiError.style.fontSize = "11px";
    civitaiError.style.marginTop = "4px";
    civitaiValue.appendChild(civitaiError);

    const nameValue = addFieldRow("Name");
    const nameInput = document.createElement("input");
    nameInput.type = "text";
    styleModalField(nameInput);
    nameValue.appendChild(nameInput);

    const minValue = addFieldRow("Strength Min");
    const minInput = document.createElement("input");
    minInput.type = "text";
    styleModalField(minInput);
    minValue.appendChild(minInput);

    const maxValue = addFieldRow("Strength Max");
    const maxInput = document.createElement("input");
    maxInput.type = "text";
    styleModalField(maxInput);
    maxValue.appendChild(maxInput);

    const notesValue = addFieldRow("Additional Notes");
    const notesInput = document.createElement("textarea");
    notesInput.rows = 2;
    styleModalField(notesInput);
    notesInput.style.resize = "vertical";
    notesValue.appendChild(notesInput);

    // --- Multi-select word pills + a single Copy Selected button --------
    // Click a word to select/deselect it (works across both sections at
    // once), then Copy Selected grabs everything selected in one go.
    const selectedWords = new Set();

    const copySelectedBtn = document.createElement("button");
    copySelectedBtn.textContent = "Copy Selected (0)";
    styleModalField(copySelectedBtn);
    copySelectedBtn.style.width = "auto";
    copySelectedBtn.style.marginTop = "14px";
    copySelectedBtn.style.cursor = "pointer";
    copySelectedBtn.disabled = true;
    copySelectedBtn.style.opacity = "0.5";
    copySelectedBtn.addEventListener("click", () => {
        if (selectedWords.size === 0) return;
        navigator.clipboard?.writeText(Array.from(selectedWords).join(", ")).catch(() => {});
        const original = copySelectedBtn.textContent;
        copySelectedBtn.textContent = "Copied!";
        setTimeout(() => (copySelectedBtn.textContent = original), 800);
    });
    modal.appendChild(copySelectedBtn);

    function updateCopyButton() {
        copySelectedBtn.textContent = `Copy Selected (${selectedWords.size})`;
        copySelectedBtn.disabled = selectedWords.size === 0;
        copySelectedBtn.style.opacity = selectedWords.size === 0 ? "0.5" : "1";
    }

    function stylePill(pill, selected) {
        pill.style.background = selected ? "#3a6f3f" : "#2a3f5f";
        pill.style.color = selected ? "#dfe" : "#bcd";
        pill.style.borderRadius = "10px";
        pill.style.padding = "2px 8px";
        pill.style.fontSize = "11px";
        pill.style.cursor = "pointer";
        pill.style.border = selected ? "1px solid #6c9" : "1px solid transparent";
    }

    function makeWordSection(label) {
        const section = document.createElement("div");
        section.style.marginTop = "14px";
        section.style.display = "none";
        const labelEl = document.createElement("div");
        labelEl.textContent = label;
        labelEl.style.color = "#999";
        labelEl.style.fontSize = "12px";
        labelEl.style.marginBottom = "4px";
        const list = document.createElement("div");
        list.style.display = "flex";
        list.style.flexWrap = "wrap";
        list.style.gap = "4px";
        section.appendChild(labelEl);
        section.appendChild(list);
        modal.appendChild(section);
        return { section, list };
    }

    function renderWordPills(container, words) {
        container.innerHTML = "";
        for (const word of words || []) {
            const pill = document.createElement("span");
            pill.textContent = word;
            pill.title = "Click to select, then use Copy Selected";
            stylePill(pill, selectedWords.has(word));
            pill.addEventListener("click", () => {
                if (selectedWords.has(word)) {
                    selectedWords.delete(word);
                } else {
                    selectedWords.add(word);
                }
                stylePill(pill, selectedWords.has(word));
                updateCopyButton();
            });
            container.appendChild(pill);
        }
    }

    // Trigger Words = what the Civitai uploader says to type in a prompt.
    // Trained Words = tags actually seen during training, read straight out
    // of the lora file itself (when the trainer embedded that data) - these
    // are two genuinely different things, kept in separate sections rather
    // than merged together.
    const triggerWords = makeWordSection("Trigger Words (from Civitai)");
    const trainedWords = makeWordSection("Trained Words (from the lora file)");

    const galleryHint = document.createElement("div");
    galleryHint.textContent = "Click an image/video to view it larger and copy its prompt.";
    galleryHint.style.color = "#888";
    galleryHint.style.fontSize = "11px";
    galleryHint.style.margin = "14px 0 6px";
    galleryHint.style.display = "none";

    const gallery = document.createElement("div");
    gallery.style.display = "grid";
    gallery.style.gridTemplateColumns = "repeat(auto-fill, minmax(110px, 1fr))";
    gallery.style.gap = "6px";
    modal.appendChild(galleryHint);
    modal.appendChild(gallery);

    function saveMetadata() {
        fetch("/gibby_nodes/lora_info", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                lora: loraFilename,
                name: nameInput.value,
                strength_min: minInput.value,
                strength_max: maxInput.value,
                notes: notesInput.value,
            }),
        }).catch(() => {});
    }
    [nameInput, minInput, maxInput, notesInput].forEach((el) =>
        el.addEventListener("blur", saveMetadata)
    );

    function renderGallery(images) {
        gallery.innerHTML = "";
        
        // Get media limit from settings
        const mediaLimit = app.ui.settings.getSettingValue("GibbyNodes.CivitaiMedia.MediaLimit") || 5;
        
        const withMedia = (images || []).filter((img) => img.url).slice(0, mediaLimit);
        galleryHint.style.display = withMedia.length ? "block" : "none";
        for (const img of withMedia) {
            const cell = document.createElement("div");
            cell.style.position = "relative";
            cell.style.cursor = "pointer";

            const isVideo = isVideoMedia(img);
            const thumbEl = document.createElement(isVideo ? "video" : "img");
            thumbEl.src = img.url;
            thumbEl.style.width = "100%";
            thumbEl.style.height = "110px";
            thumbEl.style.objectFit = "cover";
            thumbEl.style.borderRadius = "4px";
            thumbEl.style.display = "block";
            thumbEl.title = img.prompt || `(no prompt data for this ${isVideo ? "video" : "image"})`;
            if (isVideo) {
                thumbEl.muted = true;
                thumbEl.preload = "metadata";
                // Deliberately not autoplay/loop here - this is the grid
                // view, where many of these could be on screen at once.
            }
            cell.appendChild(thumbEl);

            // Play button for videos (now on the left)
            if (isVideo) {
                const badge = document.createElement("div");
                badge.textContent = "\u25b6";
                badge.style.position = "absolute";
                badge.style.top = "4px";
                badge.style.left = "4px";
                badge.style.width = "18px";
                badge.style.height = "18px";
                badge.style.borderRadius = "50%";
                badge.style.background = "rgba(0,0,0,0.6)";
                badge.style.color = "#fff";
                badge.style.fontSize = "9px";
                badge.style.display = "flex";
                badge.style.alignItems = "center";
                badge.style.justifyContent = "center";
                badge.style.pointerEvents = "none";
                cell.appendChild(badge);
            }

            // Delete button (red circle with white X) on the right
            const deleteBtn = document.createElement("div");
            deleteBtn.innerHTML = "&times;";
            deleteBtn.style.position = "absolute";
            deleteBtn.style.top = "4px";
            deleteBtn.style.right = "4px";
            deleteBtn.style.width = "18px";
            deleteBtn.style.height = "18px";
            deleteBtn.style.borderRadius = "50%";
            deleteBtn.style.background = "rgba(255,0,0,0.8)";
            deleteBtn.style.color = "#fff";
            deleteBtn.style.fontSize = "14px";
            deleteBtn.style.display = "flex";
            deleteBtn.style.alignItems = "center";
            deleteBtn.style.justifyContent = "center";
            deleteBtn.style.cursor = "pointer";
            deleteBtn.style.zIndex = "10";
            deleteBtn.addEventListener("click", (e) => {
                e.stopPropagation();
                // Remove this media item from the cached civitai info
                deleteMediaItem(img.url);
            });
            cell.appendChild(deleteBtn);

            cell.addEventListener("click", () => openMediaPreview(img));
            gallery.appendChild(cell);
        }
    }

    // Delete a media item from the cached civitai info
    async function deleteMediaItem(url) {
        try {
            // Fetch current info
            const response = await fetch("/gibby_nodes/lora_info?lora=" + encodeURIComponent(loraFilename));
            if (!response.ok) return;
            
            const data = await response.json();
            const civitai = data.civitai;
            
            if (!civitai || !civitai.images) return;
            
            // Filter out the deleted image
            civitai.images = civitai.images.filter(img => img.url !== url);
            
            // Save the updated info
            const saveResponse = await fetch("/gibby_nodes/lora_info", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ lora: loraFilename, civitai: civitai })
            });
            
            if (saveResponse.ok) {
                // Re-render the gallery
                applyInfo(data);
            }
        } catch (e) {
            console.warn("[GibbyNodes] Error deleting media:", e);
        }
    }

    let hasCivitaiInfo = false;

    function applyInfo(data) {
        if (data.hash) hashValue.textContent = data.hash;
        if (data.name) nameInput.value = data.name;
        if (data.strength_min) minInput.value = data.strength_min;
        if (data.strength_max) maxInput.value = data.strength_max;
        if (data.notes) notesInput.value = data.notes;
        if (data.trained_words && data.trained_words.length) {
            trainedWords.section.style.display = "block";
            renderWordPills(trainedWords.list, data.trained_words);
        }
        if (data.civitai) {
            hasCivitaiInfo = true;
            fetchBtn.textContent = "Re-fetch from Civitai";
            civitaiLinkContainer.innerHTML = "";
            const link = document.createElement("a");
            link.href = data.civitai.url || "#";
            link.target = "_blank";
            link.rel = "noopener noreferrer";
            link.textContent = "View on Civitai \u2197";
            link.style.color = "#7cf";
            civitaiLinkContainer.appendChild(link);
            if (data.civitai.trigger_words && data.civitai.trigger_words.length) {
                triggerWords.section.style.display = "block";
                renderWordPills(triggerWords.list, data.civitai.trigger_words);
            }
            renderGallery(data.civitai.images);
        }
    }

    fetch(`/gibby_nodes/lora_info?lora=${encodeURIComponent(loraFilename)}`)
        .then((r) => r.json())
        .then((data) => applyInfo(data))
        .catch(() => {
            hashValue.textContent = "(unavailable)";
        });

    fetchBtn.addEventListener("click", () => {
        fetchBtn.disabled = true;
        fetchBtn.textContent = "Fetching...";
        civitaiError.textContent = "";
        fetch(`/gibby_nodes/civitai_fetch?lora=${encodeURIComponent(loraFilename)}`)
            .then((r) => r.json())
            .then((data) => {
                fetchBtn.disabled = false;
                fetchBtn.textContent = hasCivitaiInfo ? "Re-fetch from Civitai" : "Fetch info from Civitai";
                if (data.error) {
                    civitaiError.textContent = data.error;
                    return;
                }
                applyInfo(data);
            })
            .catch(() => {
                fetchBtn.disabled = false;
                fetchBtn.textContent = hasCivitaiInfo ? "Re-fetch from Civitai" : "Fetch info from Civitai";
                civitaiError.textContent = "Could not reach Civitai.";
            });
    });

    document.body.appendChild(backdrop);
    document.body.appendChild(modal);
}

// Builds one row's DOM + accessors. Does NOT register it as its own
// ComfyUI widget - it's just a plain element, appended as a child of the
// single container widget that setupDynamicLoraRows owns.
function createRowController(loraOptions, initialValue, callbacks, getGlobalFilter) {
    const row = document.createElement("div");
    row.style.display = "flex";
    row.style.alignItems = "center";
    row.style.gap = "5px";
    row.style.width = "100%";
    row.style.boxSizing = "border-box";
    row.style.height = `${ROW_HEIGHT}px`;

    // Highlight missing loras with a red background
    if (initialValue?.missing && initialValue.lora !== "None") {
        row.style.background = "rgba(255, 0, 0, 0.2)";
        row.style.borderRadius = "4px";
    }

    const toggle = makeTogglePill(initialValue?.on !== false, () => {
        callbacks.onChange?.(controller);
    });

    // Placeholder for combo - will be filled after all elements are created
    let comboWrapper = null;
    let combo = null;

    const infoBtn = makeSmallButton("i", "#9cf");
    infoBtn.title = "View / fetch lora info";
    infoBtn.style.fontStyle = "italic";
    infoBtn.style.fontFamily = "Georgia, serif";
    infoBtn.style.borderRadius = "50%";

    const strength = document.createElement("input");
    strength.type = "number";
    strength.step = "0.05"; // arrow/spinner step; the wheel below still uses 0.01
    strength.value = initialValue?.strength ?? 1.0;
    strength.style.flex = "0 0 62px";
    strength.style.width = "62px";
    styleField(strength);
    addWheelAdjust(strength, { step: 0.01 });

    // Auto-select value on focus so user can type a new one immediately.
    // Also re-select on click: spinner-arrow clicks don't re-fire focus,
    // so without this the value would be left unselected after an arrow bump.
    strength.addEventListener("focus", () => strength.select());
    strength.addEventListener("click", () => strength.select());

    // Normalize values like ".6" → "0.6" or "-.3" → "-0.3" when done typing
    strength.addEventListener("blur", () => {
        const val = parseFloat(strength.value);
        if (!isNaN(val)) {
            strength.value = val;
        }
    });

    // Move arrows - reorder this row within the stack (position = apply order).
    const upBtn = makeSmallButton("\u2191", "#8c8");
    upBtn.title = "Move this lora slot up";

    const downBtn = makeSmallButton("\u2193", "#8c8");
    downBtn.title = "Move this lora slot down";

    function paintMoveBtn(btn, ok) {
        btn.disabled = !ok;
        btn.style.color = ok ? "#8c8" : "#555";
        btn.style.cursor = ok ? "pointer" : "default";
    }

    const removeBtn = makeSmallButton("\u00d7", "#f88");
    removeBtn.title = "Remove this lora slot";

    // When lora is "None", hide everything except the dropdown.
    const extraFields = [infoBtn, strength, upBtn, downBtn, removeBtn];
    function updateRowVisibility() {
        const isNone = combo?.getValue() === "None";
        for (const el of extraFields) {
            el.style.display = isNone ? "none" : "";
        }
        // The toggle pill needs its flex display restored to stay centered -
        // resetting to "" would drop it back to a plain block.
        toggle.element.style.display = isNone ? "none" : "flex";
    }

    // Now create the combo with visibility callback
    combo = createSearchableLoraSelect(loraOptions, initialValue?.lora, (newValue) =>
        callbacks.onCommitted(controller, newValue), getGlobalFilter, updateRowVisibility, initialValue?.missing);
    comboWrapper = combo.wrapper;

    infoBtn.addEventListener("click", () => openLoraInfoModal(combo.getValue()));
    upBtn.addEventListener("click", () => callbacks.onMove?.(controller, -1));
    downBtn.addEventListener("click", () => callbacks.onMove?.(controller, 1));
    removeBtn.addEventListener("click", () => callbacks.onRemove(controller));

    row.appendChild(toggle.element);
    row.appendChild(comboWrapper);
    row.appendChild(strength);
    row.appendChild(infoBtn);
    row.appendChild(upBtn);
    row.appendChild(downBtn);
    row.appendChild(removeBtn);

    // Initial visibility check
    updateRowVisibility();

    const controller = {
        element: row,
        getValue: () => ({
            on: toggle.getOn(),
            lora: combo.getValue(),
            strength: parseFloat(strength.value) || 0,
        }),
        setValue: (value) => {
            toggle.setOn(value?.on !== false);
            if (value?.lora) combo.setValue(value.lora);
            else combo.setValue("None");
            if (typeof value?.strength === "number") strength.value = value.strength;
            updateRowVisibility();
        },
        setMoveEnabled(upOk, downOk) {
            paintMoveBtn(upBtn, upOk);
            paintMoveBtn(downBtn, downOk);
        },
        // Update lora options for this row's dropdown (called on model refresh).
        updateOptions: (newOptions) => {
            combo.updateOptions(newOptions);
        },
        destroy: () => {
            combo.destroy();
            removeWheelAdjust(strength);
        },
    };

    return controller;
}

function setupDynamicLoraRows(node) {
    let loraOptions = ["None"];
    let rowControllers = [];
    let globalFilterText = "";

    const getGlobalFilter = () => globalFilterText;

    // Filter input widget - placed above everything else so it's always visible
    const filterRow = document.createElement("div");
    filterRow.style.display = "flex";
    filterRow.style.alignItems = "center";
    filterRow.style.gap = "6px";
    filterRow.style.width = "100%";
    filterRow.style.boxSizing = "border-box";
    filterRow.style.height = `${ROW_HEIGHT}px`;

    const filterLabel = document.createElement("div");
    filterLabel.textContent = "filter:";
    filterLabel.style.color = "#aaa";
    filterLabel.style.fontSize = "11px";
    filterLabel.style.flex = "0 0 auto";
    filterLabel.style.whiteSpace = "nowrap";

    const filterInput = document.createElement("input");
    filterInput.type = "text";
    filterInput.placeholder = "type to filter…  space = AND, | = OR, ! = exclude";
    filterInput.title =
        "Filter the lora dropdown as you type. Matches any part of the name or " +
        "folder. Space separates AND terms (all must match); " +
        "| separates OR terms (any can match). " +
        'A term starting with ! excludes anything containing it ("!3d" hides names with 3d). ' +
        'Example: "model_name/ slider|style" = has "model_name" in path AND ("slider" or "style").';
    filterInput.autocomplete = "off";
    filterInput.spellcheck = false;
    styleField(filterInput);
    filterInput.style.flex = "1 1 auto";
    filterInput.style.width = "100%";

    // Clear button (X) next to the filter input
    const clearFilterBtn = makeSmallButton("\u00d7", "#9cf");
    clearFilterBtn.title = "Clear filter";
    clearFilterBtn.addEventListener("click", () => {
        filterInput.value = "";
        globalFilterText = "";
    });

    filterRow.appendChild(filterLabel);
    filterRow.appendChild(filterInput);
    filterRow.appendChild(clearFilterBtn);

    node.addDOMWidget("lora_filter", "GIBBY_LORA_FILTER", filterRow, {
        getValue: () => globalFilterText,
        setValue: (value) => {
            if (typeof value === "string") {
                filterInput.value = value;
                globalFilterText = value;
            }
        },
        getHeight: () => ROW_HEIGHT,
    });

    // Update global filter when user types in the filter input
    filterInput.addEventListener("input", () => {
        globalFilterText = filterInput.value;
    });

    // All-rows control row: Toggle All / Enable All / Disable All buttons.
    const allToggleRow = document.createElement("div");
    allToggleRow.style.display = "flex";
    allToggleRow.style.alignItems = "center";
    allToggleRow.style.justifyContent = "flex-start";
    allToggleRow.style.gap = "4px";
    allToggleRow.style.width = "100%";
    allToggleRow.style.boxSizing = "border-box";
    allToggleRow.style.height = `${ROW_HEIGHT}px`;

    function makeAllButton(label, color) {
        const btn = document.createElement("button");
        btn.textContent = label;
        btn.style.height = "18px";
        btn.style.padding = "0 8px";
        btn.style.fontSize = "11px";
        btn.style.cursor = "pointer";
        btn.style.background = "#1b1b1b";
        btn.style.color = color;
        btn.style.border = "1px solid #444";
        btn.style.borderRadius = "3px";
        btn.style.flex = "0 0 auto";
        return btn;
    }

    function setAllOn(state) {
        for (const c of rowControllers) {
            const current = c.getValue();
            c.setValue({ ...current, on: state });
        }
        syncWidgetValues();
    }

    const toggleAllBtn = makeAllButton("Toggle All", "#cc8");
    toggleAllBtn.title = "Flip every row's on/off state";
    toggleAllBtn.addEventListener("click", () => {
        for (const c of rowControllers) {
            const current = c.getValue();
            c.setValue({ ...current, on: !(current.on !== false) });
        }
        syncWidgetValues();
    });

    const enableAllBtn = makeAllButton("Enable All", "#6c6");
    enableAllBtn.title = "Turn every row on";
    enableAllBtn.addEventListener("click", () => setAllOn(true));

    const disableAllBtn = makeAllButton("Disable All", "#c66");
    disableAllBtn.title = "Turn every row off";
    disableAllBtn.addEventListener("click", () => setAllOn(false));

    const clearAllBtn = makeAllButton("Clear", "#999");
    clearAllBtn.title = "Empty every lora row";
    clearAllBtn.addEventListener("click", () => {
        for (let i = rowControllers.length - 1; i > 0; i--) {
            const c = rowControllers[i];
            c.destroy();
            c.element.remove();
            rowControllers.splice(i, 1);
        }
        if (!rowControllers.length) return;
        const first = rowControllers[0];
        first.setValue({ ...first.getValue(), lora: "None" });
        refreshMoveButtons();
        resizeNode();
        syncWidgetValues();
    });

    allToggleRow.appendChild(toggleAllBtn);
    allToggleRow.appendChild(enableAllBtn);
    allToggleRow.appendChild(disableAllBtn);
    allToggleRow.appendChild(clearAllBtn);

    node.addDOMWidget("lora_all_toggle", "GIBBY_LORA_ALL_TOGGLE", allToggleRow, {
        getHeight: () => ROW_HEIGHT,
    });

    const container = document.createElement("div");
    container.style.display = "flex";
    container.style.flexDirection = "column";
    container.style.gap = "2px";
    container.style.width = "100%";
    container.style.boxSizing = "border-box";

    const resizeNode = () => {
        node.setSize([node.size[0], node.computeSize()[1]]); // keep the user's chosen width, only grow/shrink height
        node.graph?.setDirtyCanvas(true, true);
    };

    const addRows = (n, valuesToRestore) => {
        for (let k = 0; k < n; k++) {
            const initial = valuesToRestore?.[rowControllers.length] || null;
            const controller = createRowController(loraOptions, initial, {
                onCommitted: handleLoraCommitted,
                onRemove: handleRemoveRow,
                onMove: handleMoveRow,
                onChange: () => {
                    syncWidgetValues();
                },
            }, getGlobalFilter);
            container.appendChild(controller.element);
            rowControllers.push(controller);
        }
        refreshMoveButtons();
        syncWidgetValues();
    };

    function handleRemoveRow(controller) {
        const idx = rowControllers.indexOf(controller);
        if (idx === -1) return;
        controller.destroy();
        controller.element.remove();
        rowControllers.splice(idx, 1);
        if (rowControllers.length === 0) {
            addRows(1);
        } else {
            refreshMoveButtons();
        }
        resizeNode();
        syncWidgetValues();
    }

    // Gray out move arrows that can't do anything: up on the first row, down
    // on a row with no next row or whose next row is empty.
    function refreshMoveButtons() {
        for (let i = 0; i < rowControllers.length; i++) {
            const next = rowControllers[i + 1];
            rowControllers[i].setMoveEnabled(i > 0, !!next && next.getValue().lora !== "None");
        }
    }

    // Swap this row with its neighbor (dir -1 = up, +1 = down). Row order IS
    // the lora apply order, so this changes which lora blends on top of which.
    function handleMoveRow(controller, dir) {
        const idx = rowControllers.indexOf(controller);
        const target = idx + dir;
        if (idx === -1 || target < 0 || target >= rowControllers.length) return;
        // Empty slots always stay at the bottom - don't let a row sink below one.
        if (dir > 0 && rowControllers[target].getValue().lora === "None") return;
        [rowControllers[idx], rowControllers[target]] = [rowControllers[target], rowControllers[idx]];
        // Re-append in array order so the DOM matches (appendChild moves an
        // existing child rather than duplicating it).
        for (const c of rowControllers) container.appendChild(c.element);
        refreshMoveButtons();
        syncWidgetValues();
    }

    // Auto-grow: filling in the LAST row's lora adds a fresh empty row
    // after it, the same way rgthree's "Any Switch" grows once its last
    // socket is connected.
    function handleLoraCommitted(controller, newValue) {
        // A slot going empty/filled changes which arrows are usable.
        refreshMoveButtons();
        if (newValue === "None") return;
        const isLast = rowControllers[rowControllers.length - 1] === controller;
        if (!isLast) return;
        if (rowControllers.length >= MAX_SLOTS) return;

        addRows(1);
        resizeNode();
        syncWidgetValues();
    }

    const rowsWidget = node.addDOMWidget(ROWS_WIDGET_NAME, ROWS_WIDGET_TYPE, container, {
        getValue: () => rowControllers.map((c) => c.getValue()),
        setValue: (value) => {
            const arr = Array.isArray(value) ? value : [];
            // During node duplication, ComfyUI calls setValue([]) even though the
            // original node had rows. node.properties is cloned, so we restore
            // from there when the incoming value is empty.
            if (arr.length === 0) {
                const savedRows = node.properties?._savedLoraRows;
                if (Array.isArray(savedRows) && savedRows.length > 0) {
                    getLoraOptions().then((options) => {
                        loraOptions = options;
                        for (const c of rowControllers) {
                            c.destroy();
                            c.element.remove();
                        }
                        rowControllers = [];
                        // Keep deleted loras but mark them as missing
                        const validRows = savedRows.map((row) => {
                            const lora = row?.lora || "None";
                            if (lora !== "None" && !options.includes(lora)) {
                                // Lora was deleted from disk, mark as missing
                                return { ...row, lora, missing: true };
                            }
                            return row;
                        });
                        addRows(validRows.length, validRows);
                        resizeNode();
                    });
                    return;
                }
            }

            getLoraOptions().then((options) => {
                loraOptions = options;
                for (const c of rowControllers) {
                    c.destroy();
                    c.element.remove();
                }
                rowControllers = [];
                // Keep deleted loras but mark them as missing
                const validRows = arr.map((row) => {
                    const lora = row?.lora || "None";
                    if (lora !== "None" && !options.includes(lora)) {
                        // Lora was deleted from disk, mark as missing
                        return { ...row, lora, missing: true };
                    }
                    return row;
                });
                addRows(Math.max(1, validRows.length), validRows);
                resizeNode();
            });
        },
        getHeight: () => Math.max(1, rowControllers.length) * ROW_HEIGHT,
    });

    // Store row data on the widget so it survives node duplication
    // (without modifying widget.value which would trigger setValue recursion)
    const rowsWidgetIndex = node.widgets.length - 1;
    const filterWidgetIndex = rowsWidgetIndex - 1;

    function syncWidgetValues() {
        const newRows = rowControllers.map((c) => c.getValue());
        const oldRows = node.properties._savedLoraRows;

        if (!oldRows || JSON.stringify(newRows) !== JSON.stringify(oldRows)) {
            // Store in node.properties so it survives duplication
            node.properties._savedLoraRows = newRows;
        }

        const filterWidget = node.widgets[filterWidgetIndex];
        if (filterWidget) {
            filterWidget.value = globalFilterText;
        }
        node.graph?.setDirtyCanvas(true);
    }

    // Covers a brand-new node with no saved workflow data yet, where
    // lora_rows' setValue() above is never called. (If this node IS being
    // restored from a saved workflow, that setValue() will have already
    // populated rowControllers by the time this resolves, so the check
    // below is a no-op in that case.)
    getLoraOptions().then((options) => {
        loraOptions = options;
        if (rowControllers.length === 0) {
            addRows(1);
            resizeNode();
        }
    });

    // Store a refresh function on the node instance so refreshComboInNode can call it.
    // This is called when ComfyUI's model refresh button is pressed.
    node._refreshLoraOptions = async () => {
        invalidateLoraCache();
        const newOptions = await fetchLoraOptions();
        loraOptions = newOptions;
        for (const c of rowControllers) {
            c.updateOptions(newOptions);
        }
    };
}

app.registerExtension({
    name: "GibbyNodes.dynamicRows",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            setupDynamicLoraRows(this);
            return result;
        };

        // Called by ComfyUI when the model refresh button is pressed.
        nodeType.prototype.refreshComboInNode = function () {
            if (this._refreshLoraOptions) {
                this._refreshLoraOptions();
            }
        };
    },
});

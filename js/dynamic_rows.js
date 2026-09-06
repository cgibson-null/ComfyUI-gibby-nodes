import { app } from "../../scripts/app.js";

// Standalone Power Lora Loader - compact dynamic row widgets
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

const NODE_NAME = "Standalone_PowerLoraLoader";
const ROWS_WIDGET_NAME = "lora_rows";
const ROWS_WIDGET_TYPE = "STANDALONE_LORA_ROWS";
const MAX_SLOTS = 50;
const ROW_HEIGHT = 24;

// Fetch the list of available loras once (from the built-in LoraLoader
// node's own schema, so it's always current) and reuse it everywhere.
let _loraOptionsPromise = null;
function getLoraOptions() {
    if (!_loraOptionsPromise) {
        _loraOptionsPromise = fetch("/object_info/LoraLoader")
            .then((r) => r.json())
            .then((data) => {
                const names = data?.LoraLoader?.input?.required?.lora_name?.[0] || [];
                return ["None", ...names];
            })
            .catch(() => ["None"]);
    }
    return _loraOptionsPromise;
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
function createSearchableLoraSelect(loraOptions, initialLora, onCommit) {
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

    let currentValue = loraOptions.includes(initialLora) ? initialLora : "None";
    input.value = currentValue;

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
        const filtered = loraOptions.filter((o) =>
            o.toLowerCase().includes(filterText.toLowerCase())
        );
        if (filtered.length === 0) {
            const none = document.createElement("div");
            none.textContent = "No matches";
            none.style.padding = "6px 10px";
            none.style.color = "#888";
            none.style.fontSize = "12px";
            dropdown.appendChild(none);
            return;
        }
        for (const opt of filtered) {
            const optEl = document.createElement("div");
            optEl.textContent = opt;
            optEl.style.padding = "5px 10px";
            optEl.style.fontSize = "12px";
            optEl.style.cursor = "pointer";
            optEl.style.whiteSpace = "nowrap";
            optEl.style.overflow = "hidden";
            optEl.style.textOverflow = "ellipsis";
            optEl.addEventListener("mouseenter", () => (optEl.style.background = "#333"));
            optEl.addEventListener("mouseleave", () => (optEl.style.background = ""));
            optEl.addEventListener("mousedown", (e) => {
                e.preventDefault();
                input.value = opt;
                const changed = opt !== currentValue;
                currentValue = opt;
                hideDropdown();
                if (changed) onCommit?.(currentValue);
            });
            dropdown.appendChild(optEl);
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
            if (loraOptions.includes(input.value)) {
                const changed = input.value !== currentValue;
                currentValue = input.value;
                if (changed) onCommit?.(currentValue);
            } else {
                input.value = currentValue;
            }
            hideDropdown();
        }, 150);
    });
    input.addEventListener("keydown", (e) => {
        if (e.key === "Escape") {
            input.value = currentValue;
            hideDropdown();
            input.blur();
        } else if (e.key === "Enter") {
            e.preventDefault();
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
            if (loraOptions.includes(value)) {
                input.value = value;
                currentValue = value;
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

// The "i" button opens this - a small standalone info/Civitai panel. Built
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
        fetch("/standalone_power_lora_loader/lora_info", {
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
        const withMedia = (images || []).filter((img) => img.url);
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

            if (isVideo) {
                const badge = document.createElement("div");
                badge.textContent = "\u25b6";
                badge.style.position = "absolute";
                badge.style.top = "4px";
                badge.style.right = "4px";
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

            cell.addEventListener("click", () => openMediaPreview(img));
            gallery.appendChild(cell);
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

    fetch(`/standalone_power_lora_loader/lora_info?lora=${encodeURIComponent(loraFilename)}`)
        .then((r) => r.json())
        .then((data) => applyInfo(data))
        .catch(() => {
            hashValue.textContent = "(unavailable)";
        });

    fetchBtn.addEventListener("click", () => {
        fetchBtn.disabled = true;
        fetchBtn.textContent = "Fetching...";
        civitaiError.textContent = "";
        fetch(`/standalone_power_lora_loader/civitai_fetch?lora=${encodeURIComponent(loraFilename)}`)
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
function createRowController(loraOptions, initialValue, callbacks) {
    const row = document.createElement("div");
    row.style.display = "flex";
    row.style.alignItems = "center";
    row.style.gap = "5px";
    row.style.width = "100%";
    row.style.boxSizing = "border-box";
    row.style.height = `${ROW_HEIGHT}px`;

    const toggle = document.createElement("input");
    toggle.type = "checkbox";
    toggle.checked = initialValue?.on !== false;
    toggle.style.flex = "0 0 auto";
    toggle.style.cursor = "pointer";
    toggle.title = "Enable/disable this lora";

    const combo = createSearchableLoraSelect(loraOptions, initialValue?.lora, (newValue) =>
        callbacks.onCommitted(controller, newValue)
    );

    const infoBtn = makeSmallButton("i", "#9cf");
    infoBtn.title = "View / fetch lora info";
    infoBtn.style.fontStyle = "italic";
    infoBtn.style.fontFamily = "Georgia, serif";
    infoBtn.style.borderRadius = "50%";
    infoBtn.addEventListener("click", () => openLoraInfoModal(combo.getValue()));

    const strength = document.createElement("input");
    strength.type = "number";
    strength.step = "0.01";
    strength.value = initialValue?.strength ?? 1.0;
    strength.style.flex = "0 0 62px";
    strength.style.width = "62px";
    styleField(strength);
    addWheelAdjust(strength, { step: 0.01, min: -10, max: 10 });

    // Kept a little apart from the info button (separate from it by the
    // strength field) so a slightly-off click doesn't remove a row when
    // you meant to open its info panel.
    const removeBtn = makeSmallButton("\u00d7", "#f88");
    removeBtn.title = "Remove this lora slot";
    removeBtn.addEventListener("click", () => callbacks.onRemove(controller));

    row.appendChild(toggle);
    row.appendChild(combo.wrapper);
    row.appendChild(infoBtn);
    row.appendChild(strength);
    row.appendChild(removeBtn);

    const controller = {
        element: row,
        getValue: () => ({
            on: toggle.checked,
            lora: combo.getValue(),
            strength: parseFloat(strength.value) || 0,
        }),
        setValue: (value) => {
            toggle.checked = value?.on !== false;
            if (value?.lora) combo.setValue(value.lora);
            if (typeof value?.strength === "number") strength.value = value.strength;
        },
        destroy: () => {
            combo.destroy();
            removeWheelAdjust(strength);
        },
    };

    return controller;
}

// A compact single-row control: a number field that adjusts the row count
// live as it's typed/scrubbed, plus a small refresh button that re-applies
// the same thing manually (mostly redundant with the live update, but kept
// as an explicit fallback).
function createCountControl(initialCount, onApply) {
    const row = document.createElement("div");
    row.style.display = "flex";
    row.style.alignItems = "center";
    row.style.gap = "6px";
    row.style.width = "100%";
    row.style.boxSizing = "border-box";
    row.style.height = `${ROW_HEIGHT}px`;

    const label = document.createElement("div");
    label.textContent = "lora slots";
    label.style.color = "#aaa";
    label.style.fontSize = "11px";
    label.style.flex = "0 0 auto";
    label.style.whiteSpace = "nowrap";

    const numberInput = document.createElement("input");
    numberInput.type = "number";
    numberInput.min = "1";
    numberInput.max = String(MAX_SLOTS);
    numberInput.step = "1";
    numberInput.value = initialCount;
    styleField(numberInput);
    numberInput.style.flex = "1 1 auto";
    numberInput.style.width = "100%";

    function applyCurrentValue() {
        const val = Math.max(1, Math.min(MAX_SLOTS, Math.floor(parseFloat(numberInput.value) || 1)));
        onApply(val);
    }

    numberInput.addEventListener("input", applyCurrentValue);
    addWheelAdjust(numberInput, {
        step: 1,
        min: 1,
        max: MAX_SLOTS,
        onChange: applyCurrentValue,
    });

    const refreshBtn = makeSmallButton("\u27f3", "#9cf");
    refreshBtn.title = "Manually re-apply the slot count (rows normally update live as you type)";
    refreshBtn.addEventListener("click", applyCurrentValue);

    row.appendChild(label);
    row.appendChild(numberInput);
    row.appendChild(refreshBtn);

    return {
        element: row,
        setDisplayValue: (v) => {
            numberInput.value = v;
        },
    };
}

function setupDynamicLoraRows(node) {
    let loraOptions = ["None"];
    let rowControllers = [];

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

    const syncCountDisplay = () => {
        countControl.setDisplayValue(rowControllers.length);
    };

    const addRows = (n, valuesToRestore) => {
        for (let k = 0; k < n; k++) {
            const initial = valuesToRestore?.[rowControllers.length] || null;
            const controller = createRowController(loraOptions, initial, {
                onCommitted: handleLoraCommitted,
                onRemove: handleRemoveRow,
            });
            container.appendChild(controller.element);
            rowControllers.push(controller);
        }
    };

    const removeRowsFromEnd = (n) => {
        for (let k = 0; k < n && rowControllers.length > 1; k++) {
            const controller = rowControllers.pop();
            controller.destroy();
            controller.element.remove();
        }
    };

    function handleRemoveRow(controller) {
        const idx = rowControllers.indexOf(controller);
        if (idx === -1) return;
        controller.destroy();
        controller.element.remove();
        rowControllers.splice(idx, 1);
        if (rowControllers.length === 0) {
            addRows(1);
        }
        syncCountDisplay();
        resizeNode();
    }

    // Auto-grow: filling in the LAST row's lora adds a fresh empty row
    // after it, the same way rgthree's "Any Switch" grows once its last
    // socket is connected.
    function handleLoraCommitted(controller, newValue) {
        if (newValue === "None") return;
        const isLast = rowControllers[rowControllers.length - 1] === controller;
        if (!isLast) return;
        if (rowControllers.length >= MAX_SLOTS) return;

        addRows(1);
        syncCountDisplay();
        resizeNode();
    }

    const applyTargetCount = async (target) => {
        loraOptions = await getLoraOptions();
        target = Math.max(1, Math.min(MAX_SLOTS, Math.floor(target)));
        if (target > rowControllers.length) {
            addRows(target - rowControllers.length);
        } else if (target < rowControllers.length) {
            removeRowsFromEnd(rowControllers.length - target);
        }
        syncCountDisplay();
        resizeNode();
    };

    const countControl = createCountControl(1, (target) => applyTargetCount(target));

    node.addDOMWidget("lora_slot_count_display", "STANDALONE_LORA_COUNT", countControl.element, {
        // Purely a UI convenience widget - execute() reads the actual rows
        // from "lora_rows" below, not this. Kept lightweight: just enough
        // getValue/setValue to have its displayed number survive a reload
        // on its own, though lora_rows' own setValue already keeps it in
        // sync regardless. Registered BEFORE lora_rows so it has a fixed
        // position above the rows, rather than shifting down every time a
        // row is added.
        getValue: () => rowControllers.length,
        setValue: (value) => {
            if (typeof value === "number") countControl.setDisplayValue(value);
        },
        getHeight: () => ROW_HEIGHT,
    });

    const rowsWidget = node.addDOMWidget(ROWS_WIDGET_NAME, ROWS_WIDGET_TYPE, container, {
        getValue: () => rowControllers.map((c) => c.getValue()),
        // This is what actually fixes the reload/renderer-switch ghost-row
        // bug: there's now exactly one widget, and restoring a saved
        // workflow means rebuilding straight from this array - no separate
        // "does the row count widget match the row list yet" bookkeeping
        // needed at all.
        setValue: (value) => {
            const arr = Array.isArray(value) ? value : [];
            getLoraOptions().then((options) => {
                loraOptions = options;
                for (const c of rowControllers) {
                    c.destroy();
                    c.element.remove();
                }
                rowControllers = [];
                addRows(Math.max(1, arr.length), arr);
                syncCountDisplay();
                resizeNode();
            });
        },
        getHeight: () => Math.max(1, rowControllers.length) * ROW_HEIGHT,
    });

    // Covers a brand-new node with no saved workflow data yet, where
    // lora_rows' setValue() above is never called. (If this node IS being
    // restored from a saved workflow, that setValue() will have already
    // populated rowControllers by the time this resolves, so the check
    // below is a no-op in that case.)
    getLoraOptions().then((options) => {
        loraOptions = options;
        if (rowControllers.length === 0) {
            addRows(1);
            syncCountDisplay();
            resizeNode();
        }
    });
}

app.registerExtension({
    name: "StandalonePowerLoraLoader.dynamicRows",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            setupDynamicLoraRows(this);
            return result;
        };
    },
});

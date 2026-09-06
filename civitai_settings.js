import { app } from "../../../scripts/app.js";

// Gibby Nodes - Civitai Media Settings
// ------------------------------------
// Configure Civitai media filtering under
// GibbyNodes > Civitai Media (Lora Loader) in the settings dialog.

const SETTING_PREFIX = "GibbyNodes.CivitaiMedia";

app.registerExtension({
    name: "GibbyNodes.CivitaiMediaSettings",
    async init() {
        // Load settings from backend
        await loadCivitaiSettings();
    },
    settings: [
        {
            id: SETTING_PREFIX + ".ExcludeWords",
            name: "Exclude Words",
            type: "string",
            defaultValue: "",
            tooltip: "Comma-separated list of words to exclude from media prompts and tags",
            onChange: (value) => {
                const words = value.split(",").map(w => w.trim()).filter(w => w);
                saveCivitaiSettings({ exclude_words: words });
            },
        },
        {
            id: SETTING_PREFIX + ".MediaLimit",
            name: "Loaded Media Limit",
            type: "slider",
            defaultValue: 5,
            tooltip: "Maximum number of media items to load for each lora",
            sliderConfig: {
                min: 1,
                max: 20,
                step: 1,
            },
            onChange: (value) => {
                saveCivitaiSettings({ media_limit: value });
            },
        },
    ],
});

// Load settings from backend and sync to UI
async function loadCivitaiSettings() {
    try {
        const response = await fetch("/gibby_nodes/civitai_settings");
        if (response.ok) {
            const settings = await response.json();
            
            // Update UI settings from backend
            if (settings.exclude_words) {
                app.ui.settings.setSettingValue(
                    SETTING_PREFIX + ".ExcludeWords",
                    settings.exclude_words.join(", ")
                );
            }
            if (settings.media_limit) {
                app.ui.settings.setSettingValue(
                    SETTING_PREFIX + ".MediaLimit",
                    settings.media_limit
                );
            }
        }
    } catch (e) {
        console.warn("[GibbyNodes] Could not load civitai settings:", e);
    }
}

// Save settings to backend
async function saveCivitaiSettings(data) {
    try {
        await fetch("/gibby_nodes/civitai_settings", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data),
        });
    } catch (e) {
        console.warn("[GibbyNodes] Error saving civitai settings:", e);
    }
}
